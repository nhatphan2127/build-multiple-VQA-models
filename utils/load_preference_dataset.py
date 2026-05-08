import json
from datasets import Dataset
from utils.prompts import (
    SYSTEM_PROMPT_VI
)

def load_preference_dataset(json_path: str, processor, val_split: float = 0.1):
    """
    Chuyển preference_data.json → HuggingFace Dataset với cột:
    prompt, chosen, rejected  (dạng string sau apply_chat_template)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        # Build prompt messages (user turn only, không có assistant)
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_VI},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item["image_path"],
                        "min_pixels": 224 * 224,
                        "max_pixels": 224 * 224,
                    },
                    {"type": "text", "text": item["question"]},
                ],
            },
        ]

        # DPOTrainer expects string format khi dùng custom collator
        # Nếu dùng multimodal, cần build full text cho chosen/rejected
        chosen_messages  = prompt_messages + [
            {"role": "assistant", "content": item["chosen"]}
        ]
        rejected_messages = prompt_messages + [
            {"role": "assistant", "content": item["rejected"]}
        ]

        # apply_chat_template → string
        prompt_text   = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        chosen_text   = processor.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False
        )
        rejected_text = processor.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False
        )

        records.append({
            "prompt":         prompt_text,
            "chosen":         chosen_text,
            "rejected":       rejected_text,
            # Lưu thêm để debug
            "question":       item["question"],
            "ground_truth":   item["ground_truth"],
        })

    dataset = Dataset.from_list(records)

    # Train/val split
    split    = dataset.train_test_split(test_size=val_split, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]

    print(f"DPO dataset: {len(train_ds)} train | {len(val_ds)} val")
    return train_ds, val_ds