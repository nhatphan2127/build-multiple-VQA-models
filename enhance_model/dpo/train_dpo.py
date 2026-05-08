"""
train_dpo.py
Fine-tune Qwen2-VL-2B bằng DPO (Direct Preference Optimization).
Input : preference_data.json (chosen/rejected pairs)
Output: checkpoint/qwen2-vl-dpo/

Yêu cầu: pip install trl>=0.9.0 peft transformers accelerate bitsandbytes
"""

import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from qwen_vl_utils import process_vision_info

# Thêm path để import các module local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from enhance_model.dpo.DPOTrainConfig import DPOTrainConfig
from utils.load_preference_dataset import load_preference_dataset
from utils.metrics import calculate_metrics
from multimodal.dataset import VQADataset
# Cấu hình đường dẫn
OUTPUT_DIR = './enhance_model/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_for_dpo(cfg: DPOTrainConfig):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # 1. Load Base Model
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 2. Load SFT adapter và chuẩn bị cho training
    # Thay vì chỉ is_trainable=True, ta nên load config từ adapter cũ hoặc tạo mới
    model = PeftModel.from_pretrained(
        base_model, 
        cfg.sft_adapter_path, 
        is_trainable=True,
        adapter_name="default" # Đặt tên adapter
    )
    
    # QUAN TRỌNG: Phải gọi cái này trước khi config LoRA cho kbit
    model = prepare_model_for_kbit_training(model)

    # Nếu model vẫn báo 0 trainable params, hãy ép buộc enable gradient cho các adapter:
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    # 3. Reference model (Giữ nguyên - Frozen)
    # Tối ưu: Nếu thiếu VRAM, bạn có thể truyền ref_model=None vào DPOTrainer 
    # và set model_init=... (DPOTrainer sẽ tự handle việc tắt adapter để làm reference)
    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, cfg.sft_adapter_path)
    ref_model.eval()

    print("--- Trainable Parameters Check ---")
    model.print_trainable_parameters() 
    # Bây giờ con số này phải > 0
    
    return model, ref_model, processor


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train_dpo(cfg: DPOTrainConfig = DPOTrainConfig()):
    print("=== DPO Training: Qwen2-VL ===")
    print(f"Config: beta={cfg.beta}, lr={cfg.learning_rate}, loss={cfg.loss_type}")

    model, ref_model, processor = load_model_for_dpo(cfg)
    train_ds, val_ds = load_preference_dataset(
        cfg.preference_data, processor, cfg.val_split
    )


    # 1. DPOConfig: Chỉ để các tham số huấn luyện cơ bản (giống TrainingArguments)
    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,

        warmup_ratio=0.05,
        bf16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        optim="paged_adamw_32bit",

        logging_steps=5,
        save_strategy="steps",
        save_steps=50,

        eval_strategy="steps",
        eval_steps=50,

        load_best_model_at_end=True,
        report_to="tensorboard",

        remove_unused_columns=False,

        push_to_hub=True,
        hub_model_id="nhatphan2127/finetuned-rlhf-qwen",
        hub_strategy="every_save",

        # ===== DPO arguments =====
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
        # max_prompt_length=cfg.max_prompt_length,
        # label_pad_token_id=-100,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )
    print(f"Starting DPO training on {len(train_ds)} preference pairs...")
    trainer.train()

    # Lưu adapter
    # --- Lưu Model & Processor ---
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    trainer.push_to_hub()
    processor.push_to_hub("nhatphan2127/finetuned-rlhf-qwen")

    # --- Trích xuất log và vẽ đồ thị ---
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    
    # Lưu log ra csv
    df.to_csv(os.path.join(OUTPUT_DIR, "dpo_train_history.csv"))

    # Vẽ đồ thị Loss và Rewards
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    train_loss = df[df['loss'].notna()]
    plt.plot(train_loss['step'], train_loss['loss'], label='DPO Loss')
    plt.title('DPO Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Subplot 2: Rewards Accuracy
    plt.subplot(1, 2, 2)
    if 'rewards/accuracies' in df.columns:
        rewards_acc = df[df['rewards/accuracies'].notna()]
        plt.plot(rewards_acc['step'], rewards_acc['rewards/accuracies'], label='Reward Accuracy', color='green')
        plt.title('DPO Reward Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dpo_metrics_chart.png"))
    print(f"Metrics chart saved to {OUTPUT_DIR}")


# ─── EVAL (so sánh SFT vs DPO) ────────────────────────────────────────────────

def evaluate_dpo_only():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    print(f"Loading DPO Model from: {'./enhance_model/dpo/checkpoints/qwen2-vl-dpo/checkpoint-114'}")
    
    # 1. Load Model & Processor
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, './enhance_model/dpo/checkpoints/qwen2-vl-dpo/checkpoint-114')
    model.eval()

    # 2. Load Dataset
    dataset = VQADataset('dataset_for_models/test.json', 'data')
    
    all_preds = []
    all_gts = []
    all_questions = []
    all_types = []
    all_images = []
    results_detail = []

    # 3. Inference Loop
    print(f"Starting inference on {len(dataset)} samples...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            messages = [item['messages']]
            
            # Tiền xử lý cho Qwen2-VL
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=text, 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)

            # Model sinh câu trả lời
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            
            # Lưu dữ liệu thô
            all_preds.append(response)
            all_gts.append(item['answer'])
            all_questions.append(item['question'])
            all_types.append(item['type'])
            all_images.append(item['image_path'])

    # 4. Tính toán Metrics (Summary)
    print("\nCalculating metrics...")
    metrics_results = calculate_metrics(all_preds, all_gts, all_questions, all_types)
    
    # 5. Tổ chức dữ liệu chi tiết (Details)
    for question, gt, pred, type_, image_path in zip(all_questions, all_gts, all_preds, all_types, all_images):
        results_detail.append({
            "question": question,
            "gt": gt,
            "pred": pred,
            "type": type_,
            "image": image_path
        })

    # 6. Xuất file JSON tổng hợp
    output_data = {
        "summary": metrics_results,
        "details": results_detail
    }
    
    save_path = os.path.join(OUTPUT_DIR, "rlhf_dpo_eval_results.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nEvaluation Complete!")
    print(f"Results saved to: {save_path}")

    # 7. In kết quả nhanh ra màn hình
    overall = metrics_results['overall']
    print("\n" + "="*30)
    print(f"OVERALL ACCURACY: {overall['avg_vqa_accuracy']:.4f}")
    print(f"OVERALL BLEU:     {overall['avg_bleu']:.4f}")
    print("="*30)

if __name__ == "__main__":
    cfg = DPOTrainConfig()
    
    # 1. Chạy training DPO
    train_dpo(cfg)
    
    # 2. Chạy Evaluation so sánh (tùy chọn)
    evaluate_dpo_only()