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

from utils.load_preference_dataset import load_preference_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from trl import DPOConfig, DPOTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dpo.DPOTrainConfig import DPOTrainConfig


def load_model_for_dpo(cfg: DPOTrainConfig):
    """Load SFT checkpoint + thêm LoRA mới cho DPO."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # Load base model
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load SFT adapter (policy model ban đầu)
    model = PeftModel.from_pretrained(base_model, cfg.sft_adapter_path,
                                       is_trainable=True)
    model = prepare_model_for_kbit_training(model)

    # Reference model: SFT adapter FROZEN (DPOTrainer sẽ tự xử lý nếu truyền vào)
    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, cfg.sft_adapter_path,
                                           is_trainable=False)
    ref_model.eval()

    model.print_trainable_parameters()
    return model, ref_model, processor


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train_dpo(cfg: DPOTrainConfig = DPOTrainConfig()):
    print("=== DPO Training: Qwen2-VL ===")
    print(f"Config: beta={cfg.beta}, lr={cfg.learning_rate}, loss={cfg.loss_type}")

    model, ref_model, processor = load_model_for_dpo(cfg)
    train_ds, val_ds = load_preference_dataset(
        cfg.preference_data, processor, cfg.val_split
    )

    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        # Optimization
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        # DPO specific
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        # Training stability
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        # Logging
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        # DPO label padding
        label_pad_token_id=-100,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor.tokenizer,
    )

    print(f"Starting DPO training on {len(train_ds)} preference pairs...")
    trainer.train()

    # Lưu adapter
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"DPO adapter saved to {cfg.output_dir}")


# ─── EVAL (so sánh SFT vs DPO) ────────────────────────────────────────────────

def compare_sft_vs_dpo(test_json: str, img_dir: str,
                        sft_adapter: str, dpo_adapter: str,
                        n_samples: int = 50):
    """
    In ra bảng so sánh SFT vs DPO trên n_samples mẫu test.
    """
    from multimodal.dataset import VQADataset
    from qwen_vl_utils import process_vision_info
    from utils.metrics import calculate_metrics

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    dataset   = VQADataset(test_json, img_dir)

    results = {}
    for name, adapter in [("SFT", sft_adapter), ("DPO", dpo_adapter)]:
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        m = PeftModel.from_pretrained(base, adapter)
        m.eval()

        preds, gts = [], []
        for i in range(min(n_samples, len(dataset))):
            item = dataset[i]
            msgs = [item["messages"]]
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            img_inp, vid_inp = process_vision_info(msgs)
            inputs = processor(text=text, images=img_inp, videos=vid_inp,
                               padding=True, return_tensors="pt").to(m.device)
            with torch.no_grad():
                out = m.generate(**inputs, max_new_tokens=20)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            preds.append(pred)
            gts.append(item["answer"])

        metrics = calculate_metrics(preds, gts)
        results[name] = metrics
        print(f"\n{name}: {metrics}")
        del m, base

    # Delta
    print("\n=== SFT → DPO Delta ===")
    for k in results["SFT"]:
        delta = results["DPO"].get(k, 0) - results["SFT"].get(k, 0)
        print(f"  {k}: {delta:+.4f}")

    return results


if __name__ == "__main__":
    cfg = DPOTrainConfig()
    train_dpo(cfg)
