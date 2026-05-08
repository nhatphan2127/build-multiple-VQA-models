from dataclasses import dataclass, field

@dataclass
class DPOTrainConfig:
    model_id: str         = "Qwen/Qwen2-VL-2B-Instruct"
    sft_adapter_path: str = "./multimodal/checkpoint/qwen2-vl-lora/checkpoint-828"
    preference_data: str  = "./dataset_for_models/preference.json"
    output_dir: str       = "./enhance_model/dpo/checkpoints/qwen2-vl-dpo"
    val_split: float      = 0.1

    # LoRA
    lora_r: int           = 16
    lora_alpha: int       = 32
    lora_dropout: float   = 0.05

    # DPO hyperparams
    beta: float           = 0.1      # KL penalty; tăng → bám SFT nhiều hơn
    learning_rate: float  = 5e-7     # nhỏ hơn SFT để tránh forgetting
    num_epochs: int       = 2
    batch_size: int       = 4
    grad_accum: int       = 8        # effective batch = 16
    max_length: int       = 512
    max_prompt_length: int = 384

    loss_type: str        = "sigmoid"  # "sigmoid" | "ipo" | "hinge"