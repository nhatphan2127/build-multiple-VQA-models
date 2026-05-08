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

SYSTEM_PROMPT_EN = (
    "You are a smart assistant answering questions about images. "
    "Answer concisely in Vietnamese, maximum 10 words."
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


# ─────────────────────────────────────────────
# 2. LLM-AS-JUDGE PROMPTS (dùng cho DPO + PPO)
# ─────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia đánh giá hệ thống VQA (Visual Question Answering) tiếng Việt. "
    "Nhiệm vụ của bạn là so sánh hai câu trả lời và chọn câu nào tốt hơn. "
    "Chỉ trả lời bằng một chữ: A hoặc B. Không giải thích thêm."
)

JUDGE_COMPARE_TEMPLATE = """Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}

Đáp án A: {answer_a}
Đáp án B: {answer_b}

Tiêu chí đánh giá (theo thứ tự ưu tiên):
1. Chính xác ngữ nghĩa so với đáp án chuẩn
2. Ngắn gọn, súc tích (tối đa 10 từ)
3. Tự nhiên, đúng ngữ pháp tiếng Việt

Câu trả lời nào tốt hơn? Chỉ trả lời A hoặc B:"""


JUDGE_SCORE_TEMPLATE = """Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}
Đáp án mô hình: {prediction}

Chấm điểm từ 0.0 đến 1.0 dựa trên:
- 1.0: Hoàn toàn đúng, tự nhiên
- 0.7: Đúng nghĩa nhưng diễn đạt khác
- 0.4: Một phần đúng
- 0.1: Sai nhưng liên quan
- 0.0: Hoàn toàn sai

Chỉ trả lời một số thập phân (ví dụ: 0.8). Không giải thích:"""


# ─────────────────────────────────────────────
# 3. PPO REWARD PROMPT
# ─────────────────────────────────────────────

REWARD_SYSTEM_PROMPT = (
    "Bạn là bộ đánh giá câu trả lời VQA tiếng Việt. "
    "Chỉ trả lời bằng một số thập phân từ 0.0 đến 1.0."
)

REWARD_SCORE_TEMPLATE = """Ảnh: [đã được phân tích]
Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}
Câu trả lời cần đánh giá: {prediction}

Tiêu chí chấm:
- Độ chính xác ngữ nghĩa (60%)
- Độ ngắn gọn, ≤10 từ (20%)  
- Tính tự nhiên tiếng Việt (20%)

Điểm (0.0-1.0):"""


# ─────────────────────────────────────────────
# 4. PREFERENCE DATA GENERATION PROMPT
# ─────────────────────────────────────────────

PREFERENCE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia tạo dữ liệu huấn luyện cho AI. "
    "Nhiệm vụ: tạo một cặp (chosen, rejected) cho DPO training."
)

PREFERENCE_GENERATE_TEMPLATE = """Ảnh chứa: {image_description}
Câu hỏi: {question}
Đáp án đúng: {ground_truth}

Hãy tạo:
1. chosen: câu trả lời TỐT (chính xác, ngắn gọn, tự nhiên, ≤10 từ)
2. rejected: câu trả lời KÉM (sai, dài dòng, hoặc không tự nhiên)

Trả lời theo JSON:
{{"chosen": "...", "rejected": "..."}}"""


# ─────────────────────────────────────────────
# 5. HUMAN EVALUATION TEMPLATE
# ─────────────────────────────────────────────

HUMAN_EVAL_CRITERIA = {
    "accuracy": {
        "vi": "Độ chính xác: Câu trả lời có đúng với nội dung ảnh không?",
        "scale": "1 (Sai hoàn toàn) → 5 (Hoàn toàn đúng)"
    },
    "fluency": {
        "vi": "Tính tự nhiên: Câu trả lời có tự nhiên, đúng ngữ pháp tiếng Việt không?",
        "scale": "1 (Rất cứng nhắc) → 5 (Rất tự nhiên)"
    },
    "conciseness": {
        "vi": "Súc tích: Câu trả lời có ngắn gọn, không dài dòng không?",
        "scale": "1 (Rất dài dòng) → 5 (Rất súc tích)"
    }
}


# ─────────────────────────────────────────────
# HELPER: build Qwen2-VL text-only messages
# ─────────────────────────────────────────────

def build_text_only_messages(system: str, user_content: str) -> list:
    """Dùng cho judge/reward calls (không có ảnh)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]
