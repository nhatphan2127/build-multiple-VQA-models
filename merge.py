import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os

def merge_and_upload():
    # 1. Cấu hình tên repo
    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    # ĐƯỜNG DẪN ADAPTER: 
    # Nếu adapter trên Hugging Face đang lỗi, hãy trỏ vào thư mục local mà bạn đã train xong
    adapter_model_path = "./multimodal/checkpoint/qwen2-vl-lora" 
    
    # Tên repo mới để lưu bản đầy đủ (Merge xong sẽ nặng khoảng 4.5GB - 5GB)
    new_merged_repo_id = "nhatphan2127/finetuned-qwen2-vl-full"

    print("--- Đang load Base Model ---")
    # Load base model với định dạng bfloat16 để giữ nguyên độ chính xác
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Dùng cpu để merge cho ổn định nếu VRAM yếu, hoặc "cuda" nếu GPU mạnh
        trust_remote_code=True
    )

    print("--- Đang nạp LoRA Adapter ---")
    # Nạp các trọng số đã fine-tune vào base model
    model = PeftModel.from_pretrained(
        base_model,
        adapter_model_path
    )

    print("--- Đang thực hiện Merge (Hợp nhất) ---")
    # Merge và giải phóng các tham số thừa của LoRA
    merged_model = model.merge_and_unload()

    print("--- Đang load Processor ---")
    processor = AutoProcessor.from_pretrained(base_model_id)

    # 2. Lưu tạm ra máy local (Tùy chọn)
    temp_local_dir = "./qwen2-vl-merged-final"
    print(f"--- Đang lưu mô hình đã merge vào {temp_local_dir} ---")
    merged_model.save_pretrained(temp_local_dir)
    processor.save_pretrained(temp_local_dir)

    # 3. Đẩy lên Hugging Face Hub
    print(f"--- Đang push lên Hub: {new_merged_repo_id} ---")
    # Lưu ý: Bạn cần chạy 'huggingface-cli login' trước khi thực hiện
    merged_model.push_to_hub(new_merged_repo_id)
    processor.push_to_hub(new_merged_repo_id)

    print(f"✅ THÀNH CÔNG! Mô hình đã được hợp nhất và đẩy lên tại: https://huggingface.co/{new_merged_repo_id}")

if __name__ == "__main__":
    merge_and_upload()