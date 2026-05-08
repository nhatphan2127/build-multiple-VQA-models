import os
import sys
import torch
import gradio as gr
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import json

# Thêm đường dẫn để import các module local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils.vocab import Vocab

# Import models from separated_models
from seperated_models.phobert_resnet_decoder.model import VQAModel as ResNetVQAModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Globals for lazy loading ---
models_cache = {}

def get_qwen2_vl_model(model_type="finetuned"):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    adapter_path = "multimodal/checkpoint/qwen2-vl-lora-vqa/checkpoint-600"
    
    cache_key = f"qwen2_vl_{model_type}"
    if cache_key in models_cache:
        return models_cache[cache_key]
    
    print(f"Loading Qwen2-VL ({model_type})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    
    if model_type == "finetuned" and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    models_cache[cache_key] = (model, processor)
    return model, processor

def get_resnet_phobert_model(decoder_type="transformer"):
    cache_key = "resnet_phobert"
    if cache_key in models_cache:
        return models_cache[cache_key]
    
    model_dir = "seperated_models/phobert_resnet_decoder/checkpoint"
    config_path = os.path.join(model_dir, f"config_{decoder_type}.json")
    vocab_path = os.path.join(model_dir, "vocab.json")
    model_path = os.path.join(model_dir, f"best_model_{decoder_type}.pth")
    
    if not all(os.path.exists(p) for p in [config_path, vocab_path, model_path]):
        return None, None, None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    vocab = Vocab.load(vocab_path)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    model = ResNetVQAModel(
        vocab_size=config['vocab_size'], 
        d_model=config.get('d_model', 256), 
        decoder_type=config.get('decoder_type', 'None')
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    models_cache[cache_key] = (model, tokenizer, vocab)
    return model, tokenizer, vocab


def predict_qwen2_vl(image, question, model_type="finetuned"):
    model, processor = get_qwen2_vl_model(model_type)
    
    # Save temp image for Qwen2VL processor
    temp_img_path = "temp_gradio_img.png"
    image.save(temp_img_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_img_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    
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
    
    return response

def predict_separated(image, question, model_name="resnet_phobert", decoder_type="transformer"):
    model, tokenizer, vocab = get_resnet_phobert_model(decoder_type=decoder_type)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if model is None:
        return "Model checkpoint not found. Please train the model first."

    # Prepare Image
    img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    
    # Prepare Question
    q_tokens = tokenizer(question, padding='max_length', truncation=True, max_length=32, return_tensors="pt").to(DEVICE)
    
    start_id = vocab.word2idx[vocab.start_token]
    end_id = vocab.word2idx[vocab.end_token]
    
    with torch.no_grad():
        generated_ids = model.generate_beam_search(
            img_tensor, 
            q_tokens['input_ids'], 
            q_tokens['attention_mask'], 
            max_len=20, 
            start_id=start_id, 
            end_id=end_id
        )
        response = vocab.decode(generated_ids[0])
    
    return response

def vqa_inference(image, question, model_choice):
    if image is None or question == "":
        return "Vui lòng cung cấp cả ảnh và câu hỏi."
    
    try:
        if model_choice == "Qwen2-VL (Finetuned)":
            return predict_qwen2_vl(image, question, "finetuned")
        elif model_choice == "Qwen2-VL (Zero-shot)":
            return predict_qwen2_vl(image, question, "zero_shot")
        elif model_choice == "PhoBERT + ResNet + transformer":
            return predict_separated(image, question, "resnet_phobert", decoder_type="transformer")
        elif model_choice == "PhoBERT + ResNet + lstm":
            return predict_separated(image, question, "resnet_phobert", decoder_type="lstm")
        else:
            return "Lựa chọn mô hình không hợp lệ."
    except Exception as e:
        import traceback
        return f"Lỗi trong quá trình inference: {str(e)}\n{traceback.format_exc()}"

# --- Gradio UI ---
with gr.Blocks(title="Vietnamese VQA Demo") as demo:
    gr.Markdown("# 🎨 Vietnamese Visual Question Answering (VQA)")
    gr.Markdown("Tải ảnh lên và đặt câu hỏi về nội dung bức ảnh.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Ảnh đầu vào")
            input_text = gr.Textbox(lines=2, placeholder="Nhập câu hỏi tại đây...", label="Câu hỏi")
            model_dropdown = gr.Dropdown(
                choices=[
                    "Qwen2-VL (Finetuned)", 
                    "Qwen2-VL (Zero-shot)", 
                    "PhoBERT + ResNet + transformer", 
                    "PhoBERT + ResNet + lstm", 
                ],
                value="Qwen2-VL (Finetuned)",
                label="Chọn mô hình"
            )
            submit_btn = gr.Button("Trả lời", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Kết quả dự đoán")

    submit_btn.click(
        fn=vqa_inference,
        inputs=[input_img, input_text, model_dropdown],
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            ["data/images/cơm_âm_phu_Hue_0179.jpg", "Món ăn này là món gì?", "Qwen2-VL (Finetuned)"],
            ["data/images/cơm_âm_phu_Hue_0179.jpg", "Có bao nhiêu người trong ảnh?", "PhoBERT + ResNet + transformer"]
        ],
        inputs=[input_img, input_text, model_dropdown]
    )

if __name__ == "__main__":
    demo.launch(share=True)
