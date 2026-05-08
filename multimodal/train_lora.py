import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import pandas as pd

import os

OUTPUT_DIR = './multimodal/results'
MODEL_DIR = './multimodal/checkpoint/qwen2-vl-lora'
DATASET_FOR_MODELS = "./dataset_for_models"
def collate_fn(examples, processor):
    # 1. Lấy tin nhắn và xử lý hình ảnh/video
    messages = [example["messages"] for example in examples]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 2. Processor xử lý tổng hợp (Tự động chèn vision tokens vào đúng vị trí)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = inputs.input_ids.clone()
    
    # 3. Masking Labels chính xác
    # Qwen2-VL sử dụng token <|im_start|>assistant\n để bắt đầu câu trả lời
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    # Lưu ý: Cụm "assistant\n" có thể gồm nhiều tokens
    # Cách an toàn nhất là tìm sequence: <|im_start|> + assistant + \n
    
    for i in range(len(texts)):
        # Tìm vị trí của token 'assistant' đầu tiên sau phần user
        # Chúng ta tìm cụm token bắt đầu phần trả lời của AI
        input_id_list = inputs.input_ids[i].tolist()
        
        # Tìm vị trí bắt đầu của câu trả lời assistant
        # Template: ...<|im_start|>assistant\n[ANSWER]<|im_end|>
        # Ta mask tất cả cho đến hết chữ "assistant\n"
        
        target_seq = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        
        # Tìm sequence target_seq trong input_id_list
        start_idx = -1
        for j in range(len(input_id_list) - len(target_seq)):
            if input_id_list[j : j + len(target_seq)] == target_seq:
                start_idx = j + len(target_seq)
                break
        
        if start_idx != -1:
            labels[i, :start_idx] = -100
        else:
            # Nếu không tìm thấy (lỗi hiếm gặp), mask toàn bộ để tránh lỗi loss
            labels[i, :] = -100

    # Mask các token PAD
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    
    return inputs

def train_lora():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # BitsAndBytes Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Quan trọng cho training ổn định
    model = prepare_model_for_kbit_training(model)

    # LoRA Config - Mở rộng ra cả Vision Tower nếu cần (optional)
    # Ở đây chỉ tập trung vào Language Model để tiết kiệm VRAM
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset (Giả sử bạn đã có VQADataset)
    from multimodal.dataset import VQADataset
    train_dataset = VQADataset(f"{DATASET_FOR_MODELS}/train.json", "data")
    val_dataset = VQADataset(f"{DATASET_FOR_MODELS}/val.json", "data")
    
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=4,   # giảm từ 16 → 1
        gradient_accumulation_steps=8,  # tăng để effective batch = 32
        hub_model_id="nhatphan2127/finetuned-qwen-vl", # Thay bằng "username/ten-repo" của bạn
        hub_strategy="every_save",
        push_to_hub=True,
        learning_rate=1e-4,
        num_train_epochs=3,
        warmup_ratio=0.03,               # thêm warmup
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",           # thêm eval
        eval_steps=100,
        load_best_model_at_end=True,     # lưu model tốt nhất
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        dataloader_num_workers=2,
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,        # thêm
        data_collator=lambda x: collate_fn(x, processor)
    )

    trainer.train()
    trainer.push_to_hub()
    processor.push_to_hub("nhatphan2127/finetuned-qwen2-vl")



    # Trích xuất dữ liệu log
    history = trainer.state.log_history
    
    # Chuyển thành DataFrame để dễ xử lý
    df = pd.DataFrame(history)
    
    # Tách log của training và evaluation
    train_loss = df[df['loss'].notna()]
    eval_loss = df[df['eval_loss'].notna()]

    # Vẽ đồ thị Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss['step'], train_loss['loss'], label='Training Loss')
    if not eval_loss.empty:
        plt.plot(eval_loss['step'], eval_loss['eval_loss'], label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Lưu đồ thị
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_chart.png"))
    plt.show()

    # Lưu Model
    trainer.save_model(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)

if __name__ == "__main__":
    train_lora()