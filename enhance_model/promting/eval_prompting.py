"""
eval_prompting.py
Đánh giá 3 chiến lược prompting thay thế cho SFT/fine-tuning.
Chiến lược: zero_shot | few_shot | cot (chain-of-thought)

Chạy:
    python eval_prompting.py --strategy few_shot
    python eval_prompting.py --strategy all   # chạy cả 3, so sánh
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from multimodal.dataset import VQADataset
from utils.metrics import calculate_metrics, llm_as_a_judge
from prompts_vqa import build_messages, normalize_qtype

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2-VL-2B-Instruct"
TEST_JSON   = "multimodal/checkpoint/test_data.json"
IMG_DIR     = "data"
OUTPUT_DIR  = "multimodal/checkpoint"
STRATEGIES  = ["zero_shot", "few_shot", "cot"]
# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, processor


def run_inference(model, processor, item: dict,
                  strategy: str, verbose: bool = False) -> str:
    """
    Chạy inference cho 1 sample với strategy chỉ định.
    Trả về predicted answer string.
    """
    image_path   = item["messages"][0]["content"][0]["image"]
    question     = item["messages"][0]["content"][1]["text"]
    question_type = item.get("question_type", None)

    messages = build_messages(
        image_path=image_path,
        question=question,
        question_type=question_type,
        strategy=strategy,
    )

    # Bọc trong list cho Qwen2-VL (batch_size=1)
    messages_batch = [messages]

    text = processor.apply_chat_template(
        messages_batch, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=25, do_sample=False)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    if verbose:
        print(f"  Q [{question_type}]: {question}")
        print(f"  GT: {item['answer']}")
        print(f"  PR: {response}")

    return response


def evaluate_strategy(model, processor, dataset: VQADataset,
                       strategy: str) -> dict:
    """
    Chạy toàn bộ test set với 1 strategy.
    Trả về dict: {metrics, details, per_type}
    """
    print(f"\n{'='*55}")
    print(f"  Strategy: {strategy.upper()}")
    print(f"{'='*55}")

    results  = []
    preds    = []
    gts      = []
    questions = []

    for i in tqdm(range(len(dataset)), desc=strategy):
        item = dataset[i]
        pred = run_inference(model, processor, item, strategy)

        question = item["messages"][0]["content"][1]["text"]
        gt       = item["answer"]

        preds.append(pred)
        gts.append(gt)
        questions.append(question)

        results.append({
            "question":      question,
            "gt":            gt,
            "pred":          pred,
            "image":         item["image_path"],
            "question_type": item.get("question_type", "unknown"),
            "strategy":      strategy,
        })

    # ── Metrics tổng ──
    print("Computing metrics...")
    metrics = calculate_metrics(preds, gts)

    print("Computing LLM-as-a-judge...")
    llm_score = llm_as_a_judge(preds, gts, questions)
    metrics["llm_judge_score"] = llm_score

    # ── Metrics theo loại câu hỏi ──
    per_type: dict[str, list] = {}
    for r in results:
        qt = normalize_qtype(r.get("question_type", "unknown"))
        if qt not in per_type:
            per_type[qt] = {"preds": [], "gts": [], "questions": []}
        per_type[qt]["preds"].append(r["pred"])
        per_type[qt]["gts"].append(r["gt"])
        per_type[qt]["questions"].append(r["question"])

    per_type_metrics = {}
    for qt, data in per_type.items():
        m = calculate_metrics(data["preds"], data["gts"])
        per_type_metrics[qt] = m

    print(f"\nResults [{strategy}]:")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return {
        "strategy": strategy,
        "metrics":  metrics,
        "per_type": per_type_metrics,
        "details":  results,
    }


def run_all_strategies(strategies: list[str]) -> dict:
    """Chạy tất cả strategies, trả về dict {strategy: result}."""
    model, processor = load_model(MODEL_ID)
    dataset = VQADataset(TEST_JSON, IMG_DIR)

    all_results = {}
    for strategy in strategies:
        result = evaluate_strategy(model, processor, dataset, strategy)
        all_results[strategy] = result

        # Lưu kết quả từng strategy
        out_file = os.path.join(OUTPUT_DIR, f"prompting_{strategy}_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_file}")

    return all_results


def print_comparison_table(all_results: dict):
    """In bảng so sánh các strategies."""
    strategies = list(all_results.keys())
    metrics_keys = ["avg_bleu", "avg_rougeL", "avg_bert_score",
                    "avg_vqa_accuracy", "llm_judge_score"]

    col_w = 15
    header = f"{'Metric':<20}" + "".join(f"{s:>{col_w}}" for s in strategies)
    sep    = "─" * len(header)

    print(f"\n{'='*60}")
    print("  COMPARISON: Prompting Strategies")
    print(f"{'='*60}")
    print(header)
    print(sep)

    for key in metrics_keys:
        row = f"{key:<20}"
        for s in strategies:
            v = all_results[s]["metrics"].get(key, 0)
            row += f"{v:>{col_w}.4f}"
        print(row)

    print(sep)

    # Per-type breakdown cho strategy tốt nhất (theo vqa_accuracy)
    best_strategy = max(
        strategies,
        key=lambda s: all_results[s]["metrics"].get("avg_vqa_accuracy", 0)
    )
    print(f"\nBest strategy: {best_strategy}")
    print(f"\nPer-type breakdown [{best_strategy}]:")
    print(f"{'Type':<15}{'VQA Acc':>12}{'BLEU':>10}{'BERTScore':>12}")
    print("─" * 50)
    for qt, m in all_results[best_strategy]["per_type"].items():
        print(
            f"{qt:<15}"
            f"{m.get('avg_vqa_accuracy', 0):>12.4f}"
            f"{m.get('avg_bleu', 0):>10.4f}"
            f"{m.get('avg_bert_score', 0):>12.4f}"
        )


def compare_with_finetuned(prompting_results: dict,
                            finetuned_json: str = None,
                            zero_shot_json: str = None):
    """
    Load kết quả SFT/fine-tuned và so sánh với prompting.
    """
    comparison = {}

    # Load fine-tuned results (nếu có)
    if finetuned_json and os.path.exists(finetuned_json):
        with open(finetuned_json, "r", encoding="utf-8") as f:
            ft = json.load(f)
        comparison["B2_finetuned"] = ft.get("metrics", {})

    # Load zero-shot results (nếu có)
    if zero_shot_json and os.path.exists(zero_shot_json):
        with open(zero_shot_json, "r", encoding="utf-8") as f:
            zs = json.load(f)
        comparison["B1_zero_shot"] = zs.get("metrics", {})

    # Thêm prompting results
    for strategy, result in prompting_results.items():
        comparison[f"Prompting_{strategy}"] = result["metrics"]

    # In bảng so sánh toàn bộ
    metrics_keys = ["avg_bleu", "avg_rougeL", "avg_bert_score",
                    "avg_vqa_accuracy", "llm_judge_score"]
    methods = list(comparison.keys())

    print(f"\n{'='*75}")
    print("  FULL COMPARISON: All Methods")
    print(f"{'='*75}")
    print(f"{'Metric':<22}" + "".join(f"{m[:14]:>14}" for m in methods))
    print("─" * (22 + 14 * len(methods)))

    for key in metrics_keys:
        row = f"{key:<22}"
        for m in methods:
            v = comparison[m].get(key, 0)
            row += f"{v:>14.4f}"
        print(row)

    # Lưu comparison
    out = os.path.join(OUTPUT_DIR, "comparison_all_methods.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved: {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA prompting strategies")
    parser.add_argument(
        "--strategy",
        default="few_shot",
        choices=STRATEGIES + ["all"],
        help="Prompting strategy to evaluate"
    )
    parser.add_argument(
        "--compare_finetuned",
        default="multimodal/checkpoint/finetuned_results.json",
        help="Path to fine-tuned model results JSON"
    )
    parser.add_argument(
        "--compare_zero_shot",
        default="multimodal/checkpoint/zero_shot_results.json",
        help="Path to zero-shot results JSON"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.strategy == "all":
        all_results = run_all_strategies(STRATEGIES)
        print_comparison_table(all_results)
        compare_with_finetuned(
            all_results,
            finetuned_json=args.compare_finetuned,
            zero_shot_json=args.compare_zero_shot,
        )
    else:
        model, processor = load_model(MODEL_ID)
        dataset = VQADataset(TEST_JSON, IMG_DIR)
        result = evaluate_strategy(model, processor, dataset, args.strategy)

        out_file = os.path.join(OUTPUT_DIR, f"prompting_{args.strategy}_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out_file}")

        # So sánh với fine-tuned nếu có
        compare_with_finetuned(
            {args.strategy: result},
            finetuned_json=args.compare_finetuned,
            zero_shot_json=args.compare_zero_shot,
        )


if __name__ == "__main__":
    main()
