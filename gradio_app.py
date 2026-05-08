import os
import sys
import torch
import gradio as gr
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import json

# Giữ nguyên các phần import và hàm get_model như cũ của bạn...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils.vocab import Vocab
from seperated_models.phobert_resnet_decoder.model import VQAModel as ResNetVQAModel
from transformers import BitsAndBytesConfig
from huggingface_hub import hf_hub_download

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_cache = {}

# --- [Giữ nguyên các hàm get_qwen2_vl_model, get_resnet_phobert_model, predict_qwen2_vl, predict_separated của bạn] ---
# (Tôi sẽ bỏ qua phần code lặp lại để tập trung vào phần thay đổi chính)


def get_qwen2_vl_model(model_type="finetuned"):
    cache_key = f"qwen2_vl_{model_type}"
    if cache_key in models_cache:
        return models_cache[cache_key]

    if model_type == "finetuned": model_id = "nhatphan2127/finetuned-qwen2-vl" 
    elif model_type == "rlhf": model_id = "nhatphan2127/finetuned-rlhf-qwen"
    else: model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # Cấu hình nén 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config, # Dùng nén 4-bit ở đây
        device_map="auto",
        trust_remote_code=True
    )
    
    models_cache[cache_key] = (model, processor)
    return model, processor

def get_resnet_phobert_model(decoder_type="transformer", repo_id="nhatphan2127/phoBert-resnet-transformer-decoder"):
    cache_key = f"resnet_phobert_{decoder_type}"
    if cache_key in models_cache: 
        return models_cache[cache_key]

    # 1. Định nghĩa đường dẫn thư mục
    model_dir = "seperated_models/phobert_resnet_decoder/checkpoint"
    os.makedirs(model_dir, exist_ok=True) # Tạo thư mục nếu chưa có

    # 2. Định nghĩa tên file
    config_file = f"config_{decoder_type}.json"
    vocab_file = "vocab.json"
    model_file = f"best_model_{decoder_type}.pth"

    # 3. Danh sách các file cần kiểm tra và tải
    files_to_check = [config_file, vocab_file, model_file]
    
    for file_name in files_to_check:
        local_path = os.path.join(model_dir, file_name)
        if not os.path.exists(local_path):
            print(f"--- Đang tải {file_name} từ Hugging Face ({repo_id})... ---")
            try:
                # Tải file từ HF và lưu vào model_dir
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Lỗi khi tải file {file_name}: {e}")
                return None, None, None

    # 4. Sau khi đảm bảo file đã tồn tại, tiến hành load như cũ
    config_path = os.path.join(model_dir, config_file)
    vocab_path = os.path.join(model_dir, vocab_file)
    model_path = os.path.join(model_dir, model_file)

    try:
        with open(config_path, 'r') as f: 
            config = json.load(f)
        
        # Giả sử Vocab là class của bạn đã định nghĩa trước đó
        vocab = Vocab.load(vocab_path) 
        
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # Khởi tạo model (Đảm bảo ResNetVQAModel và DEVICE đã được định nghĩa)
        model = ResNetVQAModel(
            vocab_size=config['vocab_size'], 
            d_model=config.get('d_model', 256), 
            decoder_type=config.get('decoder_type', 'None')
        ).to(DEVICE)
        
        # Load trọng số
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        models_cache[cache_key] = (model, tokenizer, vocab)
        return model, tokenizer, vocab

    except Exception as e:
        print(f"Lỗi khi khởi tạo model: {e}")
        return None, None, None

def predict_qwen2_vl(image, question, model_type="finetuned"):
    try:
        model, processor = get_qwen2_vl_model(model_type)
        temp_img_path = f"temp_{model_type}.png"
        image.save(temp_img_path)
        messages = [{"role": "user", "content": [{"type": "image", "image": temp_img_path}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    except Exception as e: return f"Lỗi: {str(e)}"

def predict_separated(image, question, decoder_type="transformer"):
    try:
        model, tokenizer, vocab = get_resnet_phobert_model(decoder_type=decoder_type)
        if model is None: return "Không tìm thấy checkpoint."
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)
        q_tokens = tokenizer(question, padding='max_length', truncation=True, max_length=32, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate_beam_search(img_tensor, q_tokens['input_ids'], q_tokens['attention_mask'], max_len=20, start_id=vocab.word2idx[vocab.start_token], end_id=vocab.word2idx[vocab.end_token])
            return vocab.decode(generated_ids[0])
    except Exception as e: return f"Lỗi: {str(e)}"

# --- [PHẦN THAY ĐỔI CHÍNH] ---

def vqa_inference_all(image, question):
    """Hàm này sẽ chạy lần lượt tất cả các mô hình và trả về kết quả cho từng ô"""
    if image is None or question == "":
        return ["Vui lòng nhập đủ thông tin"] * 5

    # 1. Chạy Qwen Finetuned
    res_finetuned = predict_qwen2_vl(image, question, "finetuned")
    # 2. Chạy Qwen RLHF
    res_rlhf = predict_qwen2_vl(image, question, "rlhf")
    # 3. Chạy Qwen Zero-shot
    res_zeroshot = predict_qwen2_vl(image, question, "zero_shot")
    # 4. Chạy ResNet+Transformer
    res_transformer = predict_separated(image, question, "transformer")
    # 5. Chạy ResNet+LSTM
    res_lstm = predict_separated(image, question, "lstm")

    return res_finetuned, res_rlhf, res_zeroshot, res_transformer, res_lstm

# --- Gradio UI ---
with gr.Blocks(title="Vietnamese VQA Comparison") as demo:
    gr.Markdown("# 🎨 So sánh kết quả các mô hình Vietnamese VQA")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Ảnh đầu vào")
            input_text = gr.Textbox(lines=2, placeholder="Nhập câu hỏi tại đây...", label="Câu hỏi")
            submit_btn = gr.Button("Chạy tất cả mô hình", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Kết quả dự đoán:")
            out_finetuned = gr.Textbox(label="1. Qwen2-VL (Finetuned)", interactive=False)
            out_rlhf = gr.Textbox(label="2. Qwen2-VL (RLHF)", interactive=False)
            out_zeroshot = gr.Textbox(label="3. Qwen2-VL (Zero-shot)", interactive=False)
            out_transformer = gr.Textbox(label="4. PhoBERT + ResNet + Transformer", interactive=False)
            out_lstm = gr.Textbox(label="5. PhoBERT + ResNet + LSTM", interactive=False)

    # Khi click, gọi hàm vqa_inference_all và map kết quả vào các ô tương ứng
    submit_btn.click(
        fn=vqa_inference_all,
        inputs=[input_img, input_text],
        outputs=[out_finetuned, out_rlhf, out_zeroshot, out_transformer, out_lstm]
    )

    gr.Examples(
        examples=[
            ["./temp_finetuned.png", "Món ăn này là món gì?"],
        ],
        inputs=[input_img, input_text]
    )

if __name__ == "__main__":
    # Lưu ý: Chạy đồng thời 3 bản Qwen2-VL (2B) cần khoảng 12-16GB VRAM. 
    # Nếu bị tràn RAM, bạn nên bỏ bớt hoặc dùng model.to("cpu") sau khi dùng xong.
    demo.launch(share=True)