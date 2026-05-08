"""
generate_preference_data.py
Tạo preference dataset (chosen/rejected pairs) cho DPO training.
Chiến lược:
  1. Load B2_SFT đã fine-tune
  2. Với mỗi sample, sinh 2 câu trả lời (greedy + sampling)
  3. Dùng LLM-as-judge (Qwen2-VL text-only hoặc Anthropic API) để chọn winner
  4. Lưu ra preference_data.json
"""

import json
import os
import sys
import torch
import random
from pathlib import Path
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multimodal.dataset import VQADataset
from utils.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_COMPARE_TEMPLATE,
    build_inference_messages,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH   = "multimodal/checkpoint/qwen2-vl-lora-vqa/checkpoint-600"
DATA_JSON      = "multimodal/checkpoint/train_data.json"
IMG_DIR        = "data"
OUTPUT_FILE    = "multimodal/checkpoint/preference_data.json"
N_PAIRS        = 150      # yêu cầu tối thiểu 100; lấy 150 để có buffer
JUDGE_BACKEND  = "local"  # "local" dùng Qwen2-VL text, "anthropic" dùng Claude API
# ──────────────────────────────────────────────────────────────────────────────


def load_model_and_processor(model_id: str, adapter_path: str | None = None):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded LoRA adapter from {adapter_path}")
    model.eval()
    return model, processor


def generate_answer(model, processor, image_path: str, question: str,
                    do_sample: bool = False, temperature: float = 0.8) -> str:
    """Sinh một câu trả lời từ model."""
    messages = build_inference_messages(image_path, question)
    text = processor.apply_chat_template([messages], tokenize=False,
                                          add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=20, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ─── JUDGE: LOCAL (Qwen2-VL text-only) ───────────────────────────────────────

def judge_local(processor, model, question: str, ground_truth: str,
                answer_a: str, answer_b: str) -> str:
    """
    Dùng chính Qwen2-VL (text-only mode) để so sánh 2 câu trả lời.
    Trả về "A" hoặc "B".
    """
    user_content = JUDGE_COMPARE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer_a=answer_a,
        answer_b=answer_b
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content}
    ]
    text = processor.apply_chat_template([messages], tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    verdict = processor.decode(trimmed, skip_special_tokens=True).strip().upper()
    return "A" if verdict.startswith("A") else "B"


# ─── JUDGE: ANTHROPIC API ─────────────────────────────────────────────────────

def judge_anthropic(question: str, ground_truth: str,
                    answer_a: str, answer_b: str) -> str:
    """
    Dùng Claude API làm judge. Cần ANTHROPIC_API_KEY trong env.
    Trả về "A" hoặc "B".
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        user_content = JUDGE_COMPARE_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            answer_a=answer_a,
            answer_b=answer_b
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )
        verdict = response.content[0].text.strip().upper()
        return "A" if verdict.startswith("A") else "B"
    except Exception as e:
        print(f"[WARN] Anthropic judge failed: {e}. Falling back to random.")
        return random.choice(["A", "B"])


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def generate_preference_data():
    print("=== Generating Preference Data for DPO ===")
    model, processor = load_model_and_processor(MODEL_ID, ADAPTER_PATH)

    dataset = VQADataset(DATA_JSON, IMG_DIR)

    # Lấy ngẫu nhiên N_PAIRS samples từ train set
    indices = random.sample(range(len(dataset)), min(N_PAIRS, len(dataset)))

    pairs = []
    skipped = 0

    for idx in tqdm(indices, desc="Generating pairs"):
        item = dataset[idx]
        image_path = item["messages"][0]["content"][0]["image"]
        question   = item["messages"][0]["content"][1]["text"]
        ground_truth = item["answer"]

        # Sinh 2 câu trả lời
        ans_greedy = generate_answer(model, processor, image_path, question,
                                     do_sample=False)
        ans_sample = generate_answer(model, processor, image_path, question,
                                     do_sample=True, temperature=0.9)

        # Bỏ qua nếu 2 câu trả lời giống nhau
        if ans_greedy.lower().strip() == ans_sample.lower().strip():
            skipped += 1
            continue

        # Judge
        if JUDGE_BACKEND == "anthropic":
            winner = judge_anthropic(question, ground_truth, ans_greedy, ans_sample)
        else:
            winner = judge_local(processor, model, question, ground_truth,
                                  ans_greedy, ans_sample)

        chosen   = ans_greedy if winner == "A" else ans_sample
        rejected = ans_sample if winner == "A" else ans_greedy

        pairs.append({
            "image_path":   image_path,
            "question":     question,
            "ground_truth": ground_truth,
            "chosen":       chosen,
            "rejected":     rejected,
            "judge":        JUDGE_BACKEND,
        })

    print(f"\nGenerated {len(pairs)} pairs (skipped {skipped} identical pairs)")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}")
    return pairs


if __name__ == "__main__":
    generate_preference_data()
