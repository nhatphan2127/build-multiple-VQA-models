"""
prompts.py
Tập trung toàn bộ prompt templates cho hệ thống VQA tiếng Việt.
Dùng chung cho: inference, DPO judge, PPO reward, human eval.
"""

# ─────────────────────────────────────────────
# 1. INFERENCE PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT_VI = (
    "Bạn là trợ lý thông minh chuyên trả lời câu hỏi về hình ảnh bằng tiếng Việt. "
    "Hãy trả lời ngắn gọn, chính xác, tối đa 10 từ."
)

def build_inference_messages(image_path: str, question: str,
                              system: str = SYSTEM_PROMPT_VI,
                              min_pixels: int = 224 * 224,
                              max_pixels: int = 224 * 224) -> list:
    """
    Tạo messages list cho Qwen2-VL inference.
    Trả về list messages (không có assistant turn) để generate.
    """
    return [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]


def build_train_messages(image_path: str, question: str, answer: str,
                          system: str = SYSTEM_PROMPT_VI,
                          min_pixels: int = 224 * 224,
                          max_pixels: int = 224 * 224) -> list:
    """
    Tạo messages list cho training (có assistant turn).
    """
    msgs = build_inference_messages(image_path, question, system,
                                    min_pixels, max_pixels)
    msgs.append({
        "role": "assistant",
        "content": [{"type": "text", "text": answer}]
    })
    return msgs
