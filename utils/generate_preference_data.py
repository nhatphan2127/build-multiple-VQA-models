import json
import random

OUTPUT_PATH = 'dataset_for_models/preference.json'
INPUT_PATH = 'dataset_for_models/val.json'

def create_preference_data(input_file, output_file, target_count=1000):
    # 1. Đọc file JSON đầu vào
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # Mapping loại câu hỏi
    type_mapping = {
        "nhan_dang": "identify",
        "mau_sac": "color",
        "so_luong": "quantity",
        "mo_ta": "description",
        "yes/no": "yes_no",
        "yes_no": "yes_no",
        "vi_tri": "location"
    }

    # 2. Gom nhóm câu trả lời theo loại để lấy mẫu "rejected"
    answers_by_type = {}
    for item in source_data:
        q_type = item['type']
        if q_type not in answers_by_type:
            answers_by_type[q_type] = []
        answers_by_type[q_type].append(item['answer'])

    preference_data = []

    # 3. Vòng lặp để tạo đúng 1000 mẫu
    for i in range(target_count):
        # Lấy xoay vòng hoặc ngẫu nhiên từ dữ liệu gốc
        item = source_data[i % len(source_data)]
        
        current_type = item['type']
        current_answer = item['answer']
        
        # Chọn loại câu hỏi khác để lấy câu trả lời sai (rejected)
        other_types = [t for t in answers_by_type.keys() if t != current_type]
        
        if other_types:
            wrong_type = random.choice(other_types)
            rejected_answer = random.choice(answers_by_type[wrong_type])
            reject_reason = f"wrong_type ({type_mapping.get(wrong_type, wrong_type)} answer for {type_mapping.get(current_type, current_type)} question)"
        else:
            # Trường hợp file chỉ có 1 loại câu hỏi
            rejected_answer = "Dữ liệu không liên quan."
            reject_reason = "irrelevant_content"

        # Tạo object theo format yêu cầu
        entry = {
            "image_path": item["image_path"],
            "question": item["question"],
            "question_type": type_mapping.get(current_type, current_type),
            "ground_truth": current_answer,
            "chosen": current_answer,
            "rejected": rejected_answer,
            "reject_reason": reject_reason
        }
        preference_data.append(entry)

    # 4. Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)
    
    print(f"Hoàn thành! Đã tạo {len(preference_data)} mẫu preference tại '{output_file}'")

# Thực thi
if __name__ == "__main__":
    create_preference_data(INPUT_PATH, OUTPUT_PATH)