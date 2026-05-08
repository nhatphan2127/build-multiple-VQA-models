import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import json
import os
from tqdm import tqdm
import sys
from multimodal.dataset import VQADataset

# Thêm path để import từ utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import calculate_metrics # Giả sử bạn đặt tên hàm mới là calculate_metrics_vn


def eval_finetuned():
    model_id = "Qwen/Qwen2-VL-2B-Instruct" # Sửa lại đúng tên model Qwen2
    adapter_path = "multimodal/checkpoint/qwen2-vl-lora/checkpoint-825"
    
    # 1. Load model & adapter
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 2. Load dataset
    dataset = VQADataset("multimodal/checkpoint/test.json", "data")
    
    all_preds = []
    all_gts = []
    all_questions = []
    all_types = []
    results_detail = []

    # 3. Inference loop
    print(f"Starting inference on {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        messages = [item['messages']] 
        
        # Tiền xử lý
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
        
        # Lưu kết quả thô
        all_preds.append(response)
        all_gts.append(item['answer'])
        all_questions.append(item['question'])
        all_types.append(item['type'])
        
        results_detail.append({
            "question": item['question'],
            "gt": item['answer'],
            "pred": response,
            "type": item['type'],
            "image": item['image_path']
        })

    # 4. Tính toán metrics (Theo type và Overall)
    print("\nCalculating metrics...")
    # Gọi hàm xử lý logic tính toán (xem nội dung hàm này ở mục 2 bên dưới)
    metrics_results = calculate_metrics(all_preds, all_gts, all_questions, all_types)
    
    # 5. Save results
    output_file = "multimodal/checkpoint/finetuned_results_detailed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": metrics_results,
            "details": results_detail
        }, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file}")

    # 6. In kết quả tổng hợp
    print("\n" + "="*50)
    print(f"{'TYPE':<20} | {'ACC':<10} | {'BLEU':<10}")
    print("-" * 50)
    for t, m in metrics_results['per_type'].items():
        print(f"{t:<20} | {m['avg_vqa_accuracy']:<10.4f} | {m['avg_bleu']:<10.4f}")
    print("-" * 50)
    print(f"{'OVERALL':<20} | {metrics_results['overall']['avg_vqa_accuracy']:<10.4f} | {metrics_results['overall']['avg_bleu']:<10.4f}")
    print("="*50)

    

if __name__ == "__main__":
    eval_finetuned()