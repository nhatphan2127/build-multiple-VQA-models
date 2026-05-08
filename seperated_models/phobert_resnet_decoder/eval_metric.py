import torch
from tqdm import tqdm
from seperated_models.phobert_resnet_decoder.model import VQAModel
from seperated_models.phobert_resnet_decoder.dataset import VQADataset
from utils.metrics import calculate_metrics
import os
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.vocab import Vocab


DATA_PATH = "./data/data.json"
IMG_ROOT = "./data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-4
MODEL_DIR = "seperated_models/phobert_resnet_decoder/checkpoint"

def evaluate_test(model, tokenizer:AutoTokenizer, test_loader, vocab, decoder_type="transformer"):
    model.eval()
    all_preds = []
    all_gts = []
    all_question = []
    all_types = [] # Đổi tên để tránh trùng lặp
    all_images = []
    results_detail = []
    start_id = vocab.word2idx[vocab.start_token]
    end_id = vocab.word2idx[vocab.end_token]

    DEVICE = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch['image'].to(DEVICE)
            q_ids = batch['q_ids'].to(DEVICE)
            q_mask = batch['q_mask'].to(DEVICE)
            gts = batch['answer_text']
            batch_types = batch['type'] # Lấy type từ batch
            images_path = batch['img_path']
            generated_ids = model.generate_beam_search(images, q_ids, q_mask, max_len=20, start_id=start_id, end_id=end_id)
            preds = vocab.batch_decode(generated_ids)
            
            all_preds.extend(preds)
            all_gts.extend(gts)
            all_question.extend(tokenizer.batch_decode(q_ids, skip_special_tokens=True))
            all_types.extend(batch_types) # Lưu type vào list tổng
            all_images.extend(images_path)
    # Gọi hàm tính toán metrics theo type
    metrics_results = calculate_metrics(all_preds, all_gts, all_question, all_types)
    
    for question, gt, pred, type_, image_path in zip(all_question, all_gts, all_preds, all_types, all_images):
        results_detail.append({
            "question": question,
            "gt": gt,
            "pred": pred,
            "type": type_,
            "image": image_path
        })

    print("\nTest Set Evaluation Metrics (Overall):")
    overall = metrics_results['overall']
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")
    
    # Lưu kết quả chi tiết vào file json
    with open(os.path.join(MODEL_DIR, f"test_metrics_{decoder_type}.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "summary": metrics_results,
            "details": results_detail
        }, f, ensure_ascii=False, indent=4)
    
    return metrics_results

if __name__ == "__main__":


    for decoder_type in ['lstm']:
            # Load best before test
        with open(f'seperated_models/phobert_resnet_decoder/checkpoint/config_{decoder_type}.json', 'r', encoding='utf-8') as file:
            config = json.loads(file.read())
        model = VQAModel(vocab_size=config['vocab_size'], d_model=config['d_model'], decoder_type=config['decoder_type']).to(DEVICE)
        
        with open('seperated_models/phobert_resnet_decoder/checkpoint/test.json', 'r', encoding='utf-8') as file:
            test = json.loads(file.read())
            vocab = Vocab.load('./seperated_models/phobert_resnet_decoder/checkpoint/vocab.json')
            phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            test_dataset = VQADataset(test, phobert_tokenizer, vocab, img_root='./data/')
            test_loader = DataLoader(test_dataset, batch_size=64)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"best_model_{decoder_type}.pth")))
        evaluate_test(model, phobert_tokenizer, test_loader, vocab, decoder_type=decoder_type)