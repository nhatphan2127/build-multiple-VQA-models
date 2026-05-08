"""
train_dpo.py
Fine-tune Qwen2-VL-2B bằng DPO (Direct Preference Optimization).
Tích hợp: Logging, Vẽ đồ thị Loss/Rewards, Push to Hub và So sánh SFT vs DPO.
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

# Cấu hình đường dẫn
OUTPUT_DIR = './enhance_model/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_for_dpo(cfg: DPOTrainConfig):
    """Load SFT checkpoint làm nền tảng và khởi tạo Reference Model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # 1. Load Policy Model (Model sẽ được train)
    # Load base
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Load SFT adapter đã train trước đó
    model = PeftModel.from_pretrained(
        base_model, 
        cfg.sft_adapter_path, 
        is_trainable=True,
        adapter_name="default"
    )
    model = prepare_model_for_kbit_training(model)

    # 2. Load Reference Model (Model cố định để so sánh KL divergence)
    ref_base = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(
        ref_base, 
        cfg.sft_adapter_path, 
        is_trainable=False,
        adapter_name="reference"
    )
    ref_model.eval()

    print("Trainable parameters for DPO:")
    model.print_trainable_parameters()
    
    return model, ref_model, processor

def train_dpo(cfg: DPOTrainConfig = DPOTrainConfig()):
    print("=== DPO Training: Qwen2-VL ===")
    
    model, ref_model, processor = load_model_for_dpo(cfg)
    train_ds, val_ds = load_preference_dataset(
        cfg.preference_data, processor, cfg.val_split
    )

    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        # DPO specific
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        # Hub & Logging
        push_to_hub=True,
        hub_model_id="nhatphan2127/finetuned-rlhf-qwen",
        hub_strategy="every_save",
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        load_best_model_at_end=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer, # DPOTrainer dùng tokenizer để padding
    )

    print(f"Starting DPO training on {len(train_ds)} preference pairs...")
    trainer.train()

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

def compare_sft_vs_dpo(test_json, img_dir, sft_path, dpo_path):
    """So sánh kết quả giữa model SFT cũ và model DPO mới."""
    from multimodal.dataset import VQADataset
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    dataset = VQADataset(test_json, img_dir)
    processor = AutoProcessor.from_pretrained(model_id)
    
    comparison_results = []

    for name, adapter_path in [("SFT", sft_path), ("DPO", dpo_path)]:
        print(f"Evaluating {name}...")
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        preds, gts, questions, types = [], [], [], []

        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            messages = [item['messages']]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(text=text, images=image_inputs, videos=video_inputs, 
                               padding=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            preds.append(response)
            gts.append(item['answer'])
            questions.append(item['question'])
            types.append(item['type'])

        metrics = calculate_metrics(preds, gts, questions, types)
        comparison_results.append({"model": name, "metrics": metrics})
        
        # Dọn dẹp VRAM
        del model, base_model
        torch.cuda.empty_cache()

    # In bảng so sánh
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'SFT':<15} | {'DPO':<15}")
    print("-" * 60)
    sft_m = comparison_results[0]['metrics']['overall']
    dpo_m = comparison_results[1]['metrics']['overall']
    
    for key in ['avg_vqa_accuracy', 'avg_bleu']:
        print(f"{key:<25} | {sft_m[key]:<15.4f} | {dpo_m[key]:<15.4f}")
    print("="*60)

if __name__ == "__main__":
    cfg = DPOTrainConfig()
    
    # 1. Chạy training DPO
    train_dpo(cfg)
    
    # 2. Chạy Evaluation so sánh (tùy chọn)
    # compare_sft_vs_dpo(
    #     test_json="./dataset_for_models/test.json",
    #     img_dir="data",
    #     sft_path=cfg.sft_adapter_path,
    #     dpo_path=cfg.output_dir
    # )