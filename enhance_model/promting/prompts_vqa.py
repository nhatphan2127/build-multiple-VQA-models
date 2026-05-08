"""
prompts_vqa.py
Prompt templates cho hệ thống VQA tiếng Việt - Món ăn Việt Nam.
Hỗ trợ 3 chiến lược:
  1. Zero-shot   : system prompt rõ ràng, không ví dụ
  2. Few-shot    : 2-3 ví dụ per loại câu hỏi
  3. Chain-of-Thought (CoT): hướng dẫn suy luận từng bước
"""

# ─── Mapping nhãn loại câu hỏi về chuẩn ─────────────────────────────────────
QTYPE_NORMALIZE = {
    "nhan_dang": "nhan_dang",
    "mau_sac":   "mau_sac",
    "so_luong":  "so_luong",
    "mo_ta":     "mo_ta",
    "mô tả":     "mo_ta",
    "vi_tri":    "vi_tri",
    "yes/no":    "yes_no",
    "yes_no":    "yes_no",
}

def normalize_qtype(raw: str) -> str:
    return QTYPE_NORMALIZE.get(raw.strip().lower(), "mo_ta")


# ─── SYSTEM PROMPTS ──────────────────────────────────────────────────────────

SYSTEM_BASE = (
    "Bạn là trợ lý AI chuyên trả lời câu hỏi về ảnh món ăn Việt Nam bằng tiếng Việt. "
    "Luôn trả lời ngắn gọn, chính xác, tối đa 10 từ. "
    "Không giải thích, không thêm câu dẫn. Chỉ trả lời thẳng."
)

SYSTEM_BY_TYPE = {
    "yes_no": (
        "Bạn là trợ lý AI trả lời câu hỏi có/không về ảnh món ăn Việt Nam. "
        "Chỉ trả lời bằng: 'Đúng, ...' hoặc 'Không, ...' kèm 1-2 từ giải thích ngắn. "
        "Tối đa 8 từ."
    ),
    "nhan_dang": (
        "Bạn là trợ lý AI nhận dạng món ăn Việt Nam từ ảnh. "
        "Trả lời tên món hoặc vật thể được hỏi. Tối đa 8 từ, không cần câu hoàn chỉnh."
    ),
    "mau_sac": (
        "Bạn là trợ lý AI nhận diện màu sắc trong ảnh món ăn Việt Nam. "
        "Trả lời màu sắc cụ thể. Tối đa 8 từ."
    ),
    "so_luong": (
        "Bạn là trợ lý AI đếm số lượng trong ảnh món ăn Việt Nam. "
        "Trả lời bằng số hoặc cụm số từ. Tối đa 8 từ."
    ),
    "mo_ta": (
        "Bạn là trợ lý AI mô tả đặc điểm trong ảnh món ăn Việt Nam. "
        "Trả lời mô tả ngắn gọn, tối đa 10 từ."
    ),
    "vi_tri": (
        "Bạn là trợ lý AI xác định vị trí vật thể trong ảnh món ăn Việt Nam. "
        "Trả lời vị trí cụ thể (ví dụ: góc trái, giữa, phía trên...). Tối đa 8 từ."
    ),
}


# ─── FEW-SHOT EXAMPLES (lấy từ dữ liệu thực tế của bạn) ─────────────────────

FEW_SHOT_EXAMPLES = {
    "yes_no": [
        {
            "question": "Có phải bánh này được xào chín?",
            "answer":   "Không, bánh được hấp cách thủy."
        },
        {
            "question": "Đây là món nước đúng không?",
            "answer":   "Đúng, nước dùng đầy bát."
        },
        {
            "question": "Món này có phải món khô?",
            "answer":   "Không, món này có nước."
        },
    ],
    "nhan_dang": [
        {
            "question": "Trong hình là món ăn gì?",
            "answer":   "Đây là món bánh bao hấp."
        },
        {
            "question": "Bức ảnh này chụp món ăn gì vậy?",
            "answer":   "Bánh bèo tôm cháy xuất hiện trong khung hình."
        },
        {
            "question": "Món này tên là gì?",
            "answer":   "Bún bò Huế kèm móng giò."
        },
    ],
    "mau_sac": [
        {
            "question": "Làn khói bốc lên có màu gì?",
            "answer":   "Hơi nước mang màu trắng mờ."
        },
        {
            "question": "Các lớp bánh có màu sắc gì?",
            "answer":   "Bánh có màu xanh và vàng nhạt."
        },
        {
            "question": "Tô bún có màu chủ đạo gì?",
            "answer":   "Tô bún có màu đen tuyền."
        },
    ],
    "so_luong": [
        {
            "question": "Có bao nhiêu cái bánh đang bốc khói?",
            "answer":   "Có bốn chiếc bánh bao nóng."
        },
        {
            "question": "Có bao nhiêu chiếc bánh bèo trên đĩa này?",
            "answer":   "Đĩa bánh có khoảng mười hai chiếc bánh."
        },
        {
            "question": "Có mấy miếng chả trắng?",
            "answer":   "Có hai miếng chả lụa."
        },
    ],
    "mo_ta": [
        {
            "question": "Trạng thái của bánh lúc này thế nào?",
            "answer":   "Bánh đang bốc hơi nghi ngút."
        },
        {
            "question": "Miếng thịt bò trông ra sao?",
            "answer":   "Thịt bò thái lát to bản."
        },
        {
            "question": "Lá chanh được thái như thế nào?",
            "answer":   "Lá chanh thái sợi cực mỏng."
        },
    ],
    "vi_tri": [
        {
            "question": "Bánh được đặt ở đâu để hấp?",
            "answer":   "Bánh nằm trong nồi hấp gỗ."
        },
        {
            "question": "Phần tôm cháy được đặt ở đâu trên đĩa?",
            "answer":   "Tôm cháy nằm ngay chính giữa mỗi chiếc bánh."
        },
        {
            "question": "Lát chanh nằm ở đâu?",
            "answer":   "Chanh ở góc dưới phải."
        },
    ],
}


# ─── CHAIN-OF-THOUGHT INSTRUCTIONS ───────────────────────────────────────────

COT_INSTRUCTIONS = {
    "yes_no": (
        "Hãy quan sát kỹ ảnh, xác định đặc điểm liên quan đến câu hỏi, "
        "rồi trả lời 'Đúng' hoặc 'Không' kèm lý do ngắn trong 8 từ."
    ),
    "nhan_dang": (
        "Hãy nhìn tổng thể ảnh, xác định món ăn/vật thể chính, "
        "rồi nêu tên ngắn gọn trong 8 từ."
    ),
    "mau_sac": (
        "Hãy tập trung vào đối tượng được hỏi trong ảnh, "
        "xác định màu sắc chủ đạo, trả lời trong 8 từ."
    ),
    "so_luong": (
        "Hãy đếm kỹ đối tượng trong ảnh, "
        "trả lời số lượng cụ thể trong 8 từ."
    ),
    "mo_ta": (
        "Hãy quan sát đặc điểm, hình dạng, trạng thái của đối tượng, "
        "mô tả ngắn gọn trong 10 từ."
    ),
    "vi_tri": (
        "Hãy xác định vị trí của đối tượng trong ảnh "
        "(góc nào, trên/dưới, trái/phải, giữa...), trả lời trong 8 từ."
    ),
}


# ─── BUILDER FUNCTIONS ───────────────────────────────────────────────────────

def build_zero_shot_messages(image_path: str, question: str,
                              question_type: str = None,
                              min_pixels: int = 224 * 224,
                              max_pixels: int = 224 * 224) -> list:
    """
    Zero-shot: system prompt theo loại câu hỏi + user turn.
    Nếu không biết loại, dùng SYSTEM_BASE.
    """
    qtype  = normalize_qtype(question_type or "")
    system = SYSTEM_BY_TYPE.get(qtype, SYSTEM_BASE)

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": question},
            ],
        },
    ]


def build_few_shot_messages(image_path: str, question: str,
                             question_type: str = None,
                             n_examples: int = 2,
                             min_pixels: int = 224 * 224,
                             max_pixels: int = 224 * 224) -> list:
    """
    Few-shot: thêm n_examples ví dụ text-only trước câu hỏi thực.
    Ảnh ví dụ không có (text-only) để tiết kiệm context.
    """
    qtype    = normalize_qtype(question_type or "")
    system   = SYSTEM_BY_TYPE.get(qtype, SYSTEM_BASE)
    examples = FEW_SHOT_EXAMPLES.get(qtype, [])[:n_examples]

    messages = [{"role": "system", "content": system}]

    # Thêm ví dụ text-only
    for ex in examples:
        messages.append({
            "role": "user",
            "content": f"[Ảnh món ăn Việt Nam]\nCâu hỏi: {ex['question']}"
        })
        messages.append({
            "role": "assistant",
            "content": ex["answer"]
        })

    # Câu hỏi thực với ảnh
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            {"type": "text", "text": f"Câu hỏi: {question}"},
        ],
    })

    return messages


def build_cot_messages(image_path: str, question: str,
                        question_type: str = None,
                        min_pixels: int = 224 * 224,
                        max_pixels: int = 224 * 224) -> list:
    """
    Chain-of-Thought: hướng dẫn model suy nghĩ theo bước trước khi trả lời.
    Có thêm instruction 'Sau đó trả lời ngắn gọn:'
    """
    qtype       = normalize_qtype(question_type or "")
    system      = SYSTEM_BY_TYPE.get(qtype, SYSTEM_BASE)
    instruction = COT_INSTRUCTIONS.get(qtype, "Trả lời ngắn gọn:")

    user_text = (
        f"{instruction}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Trả lời (tối đa 10 từ):"
    )

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_messages(image_path: str, question: str,
                   question_type: str = None,
                   strategy: str = "few_shot",
                   **kwargs) -> list:
    """
    Entry point chính. strategy: 'zero_shot' | 'few_shot' | 'cot'
    """
    if strategy == "zero_shot":
        return build_zero_shot_messages(image_path, question, question_type, **kwargs)
    elif strategy == "cot":
        return build_cot_messages(image_path, question, question_type, **kwargs)
    else:  # few_shot (default)
        return build_few_shot_messages(image_path, question, question_type, **kwargs)
