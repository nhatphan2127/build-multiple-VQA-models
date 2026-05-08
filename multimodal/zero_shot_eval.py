import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from multimodal.dataset import VQADataset
import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import calculate_metrics

def zero_shot_eval():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    print(f"Loading model and processor: {model_id}")
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    ).eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    test_json = "./multimodal/checkpoint/test.json"
    img_dir = "data"

    dataset = VQADataset(test_json, img_dir)
    
    results = []
    preds = []
    gts = []
    questions = []
    types = []

    print("Starting Zero-shot evaluation with Qwen2-VL...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        messages = item['messages']
        gt_answer = item['answer']
        question = messages[0]["content"][1]["text"]

        # Prepare for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        preds.append(response)
        gts.append(gt_answer)
        questions.append(question)
        types.append(item['type'])
        
        results.append({
            "question": question,
            "gt": gt_answer,
            "pred": response,
            "image": item['image_path'],
            'type' : item['type']
        })

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(preds, gts, questions, types)
    print("Results:", metrics)

    # Save results
    output_file = "./multimodal/zero_shot_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": metrics,
            "details": results
        }, f, ensure_ascii=False, indent=4)
    
    print(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    zero_shot_eval()
