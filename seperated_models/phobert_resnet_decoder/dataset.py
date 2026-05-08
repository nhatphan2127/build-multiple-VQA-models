import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class VQADataset(Dataset):
    def __init__(self, data, phobert_tokenizer, answer_vocab, max_q_len=32, max_a_len=20, img_root="."):
        self.data = data
        self.phobert_tokenizer = phobert_tokenizer
        self.answer_vocab = answer_vocab
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.img_root = img_root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_root, item["image_path"])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize question using PhoBERT tokenizer
        q_tokens = self.phobert_tokenizer(item['question'], padding='max_length', 
                                          truncation=True, max_length=self.max_q_len, return_tensors="pt")
        
        # Encode answer using our custom Vocab
        a_ids = self.answer_vocab.encode(item['answer'], max_len=self.max_a_len)
        
        return {
            'image': image,
            'q_ids': q_tokens['input_ids'].squeeze(),
            'q_mask': q_tokens['attention_mask'].squeeze(),
            'a_ids': torch.tensor(a_ids, dtype=torch.long),
            'answer_text': item['answer'],
            'type': item['type'],
            'img_path': img_path
        }
