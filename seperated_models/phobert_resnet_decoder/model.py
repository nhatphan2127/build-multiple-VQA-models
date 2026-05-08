import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Cấu hình
PHOBERT_NAME = "vinai/phobert-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoAttention(nn.Module):
    def __init__(self, d_model):
        super(CoAttention, self).__init__()
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_attn = nn.Linear(d_model, 1)

    def forward(self, v, q):
        # v: [batch, num_regions, d_model] (ResNet features)
        # q: [batch, seq_len, d_model] (PhoBERT features)
        
        # Simple Parallel Co-Attention
        v_proj = self.W_v(v).unsqueeze(2) # [B, num_regions, 1, D]
        q_proj = self.W_q(q).unsqueeze(1) # [B, 1, seq_len, D]
        
        # Interaction matrix
        C = torch.tanh(v_proj + q_proj) # [B, num_regions, seq_len, D]
        attn_scores = self.W_attn(C).squeeze(-1) # [B, num_regions, seq_len]
        
        # Attention on Image (guided by text)
        a_v = torch.softmax(torch.max(attn_scores, dim=2)[0], dim=1) # [B, num_regions]
        v_hat = (v * a_v.unsqueeze(-1)).sum(1) # [B, D]
        
        # Attention on Question (guided by image)
        a_q = torch.softmax(torch.max(attn_scores, dim=1)[0], dim=1) # [B, seq_len]
        q_hat = (q * a_q.unsqueeze(-1)).sum(1) # [B, D]
        
        return v_hat, q_hat

class VQAModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_type='transformer'):
        super(VQAModel, self).__init__()
        
        # 1. Image Encoder (ResNet152)
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # Bỏ lớp Pooling và FC
        self.resnet = nn.Sequential(*modules)
        self.img_proj = nn.Linear(2048, d_model) # ResNet152 output 2048
        
        # FREEZE ResNet152
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 2. Text Encoder (PhoBERT)
        self.phobert = AutoModel.from_pretrained(PHOBERT_NAME)
        self.text_proj = nn.Linear(768, d_model) # PhoBERT base output 768
        
        # FREEZE PhoBERT
        for param in self.phobert.parameters():
            param.requires_grad = False

        # 3. Co-Attention
        self.co_attention = CoAttention(d_model)
        
        # 4. Decoder
        self.decoder_type = decoder_type
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if decoder_type == 'lstm':
            self.decoder = nn.LSTMCell(d_model, d_model)
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
            
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, images, question_ids, question_mask, target_ids=None):
        # Extract Image Features
        with torch.no_grad():
            img_features = self.resnet(images) # [B, 2048, 7, 7]
        img_features = img_features.flatten(2).permute(0, 2, 1) # [B, 49, 2048]
        img_features = self.img_proj(img_features) # [B, 49, d_model]
        
        # Extract Text Features
        text_outputs = self.phobert(input_ids=question_ids, attention_mask=question_mask)
        text_features = self.text_proj(text_outputs.last_hidden_state) # [B, seq_len, d_model]
        
        # Co-Attention
        v_hat, q_hat = self.co_attention(img_features, text_features)
        context = v_hat + q_hat # Fusion features [B, d_model]
        
        # Decoder logic
        if self.decoder_type == 'lstm':
            h, c = context, torch.zeros_like(context)
            outputs = []
            for i in range(target_ids.size(1)):
                char_emb = self.embedding(target_ids[:, i])
                h, c = self.decoder(char_emb, (h, c))
                outputs.append(self.fc_out(h))
            return torch.stack(outputs, dim=1)
            
        else: # Transformer Decoder
            tgt_len = target_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(images.device)
            tgt_emb = self.embedding(target_ids) # [B, tgt_len, D]
            memory = context.unsqueeze(1) # [B, 1, D]
            output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            return self.fc_out(output)

    def generate_greedy(self, images, question_ids, question_mask, max_len, start_id, end_id):
        batch_size = images.size(0)
        device = images.device

        with torch.no_grad():
            # Encoder
            img_features = self.resnet(images).flatten(2).permute(0, 2, 1)
            img_features = self.img_proj(img_features)
            text_outputs = self.phobert(input_ids=question_ids, attention_mask=question_mask)
            text_features = self.text_proj(text_outputs.last_hidden_state)
            v_hat, q_hat = self.co_attention(img_features, text_features)
            context = v_hat + q_hat
            memory = context.unsqueeze(1)

            generated = torch.full((batch_size, 1), start_id, dtype=torch.long).to(device)
            
            if self.decoder_type == 'lstm':
                h, c = context, torch.zeros_like(context)
                for _ in range(max_len):
                    char_emb = self.embedding(generated[:, -1])
                    h, c = self.decoder(char_emb, (h, c))
                    logits = self.fc_out(h)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    if (next_token == end_id).all(): break
            else:
                for _ in range(max_len):
                    tgt_emb = self.embedding(generated)
                    output = self.decoder(tgt=tgt_emb, memory=memory)
                    logits = self.fc_out(output[:, -1])
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    if (next_token == end_id).all(): break
        return generated

    def generate_beam_search(self, images, question_ids, question_mask, max_len, start_id, end_id, beam_size=3):
        batch_size = images.size(0)
        device = images.device

        all_generated = []
        for b in range(batch_size):
            img_b = images[b:b+1]
            q_ids_b = question_ids[b:b+1]
            q_mask_b = question_mask[b:b+1]
            
            with torch.no_grad():
                img_features = self.resnet(img_b).flatten(2).permute(0, 2, 1)
                img_features = self.img_proj(img_features)
                text_outputs = self.phobert(input_ids=q_ids_b, attention_mask=q_mask_b)
                text_features = self.text_proj(text_outputs.last_hidden_state)
                v_hat, q_hat = self.co_attention(img_features, text_features)
                context = v_hat + q_hat
                memory = context.unsqueeze(1)

                if self.decoder_type == 'lstm':
                    beams = [([start_id], 0.0, (context, torch.zeros_like(context)))]
                else:
                    beams = [([start_id], 0.0)]

                for _ in range(max_len):
                    new_beams = []
                    for beam in beams:
                        seq, score = beam[0], beam[1]
                        if seq[-1] == end_id:
                            new_beams.append(beam)
                            continue
                        
                        if self.decoder_type == 'lstm':
                            h, c = beam[2]
                            char_emb = self.embedding(torch.tensor([seq[-1]]).to(device))
                            h_new, c_new = self.decoder(char_emb, (h, c))
                            logits = self.fc_out(h_new)
                        else:
                            tgt_emb = self.embedding(torch.tensor([seq]).to(device))
                            output = self.decoder(tgt=tgt_emb, memory=memory)
                            logits = self.fc_out(output[:, -1])

                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                        topk_probs, topk_ids = torch.topk(log_probs, beam_size)

                        for i in range(beam_size):
                            new_seq = seq + [topk_ids[i].item()]
                            new_score = score + topk_probs[i].item()
                            if self.decoder_type == 'lstm':
                                new_beams.append((new_seq, new_score, (h_new, c_new)))
                            else:
                                new_beams.append((new_seq, new_score))
                    
                    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                    if all(s[0][-1] == end_id for s in beams): break
                
                all_generated.append(torch.tensor(beams[0][0]))
        
        return torch.nn.utils.rnn.pad_sequence(all_generated, batch_first=True, padding_value=0)
