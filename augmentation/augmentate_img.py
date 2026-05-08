import cv2
import albumentations as A
import json
import os
import numpy as np
from tqdm import tqdm

def augment_vqa_dataset(input_json, img_root, out_img_dir, output_dir, num_aug_per_img=3):
    abs_input_json = os.path.abspath(input_json)
    abs_img_root = os.path.abspath(img_root)
    abs_out_img_dir = os.path.abspath(out_img_dir)
    abs_output_dir = os.path.abspath(output_dir)

    if not os.path.exists(abs_out_img_dir):
        os.makedirs(abs_out_img_dir)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    with open(abs_input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, p=0.3),
    ])

    new_data = []

    for item in tqdm(data, desc="Đang xử lý"):
        orig_rel_path = item['image_path']
        # Chuẩn hóa đường dẫn để tránh lỗi mix giữa / và \
        img_full_path = os.path.normpath(os.path.join(abs_img_root, orig_rel_path))
        
        # --- CÁCH ĐỌC ẢNH HỖ TRỢ TIẾNG VIỆT (UNICODE) ---
        if not os.path.exists(img_full_path):
            print(f"\n[Bỏ qua] Không tìm thấy file: {img_full_path}")
            continue
            
        try:
            # Đọc file bằng numpy thay vì cv2.imread
            img_array = np.fromfile(img_full_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"\n[Lỗi] Không thể decode ảnh: {img_full_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"\n[Lỗi] Gặp vấn đề khi đọc {img_full_path}: {e}")
            continue

        for ann_idx, ann in enumerate(item['annotations']):
            answers = [ann['answer_original']]
            if 'paraphrase_1' in ann and ann['paraphrase_1']: answers.append(ann['paraphrase_1'])
            
            # 1. Thêm dữ liệu gốc
            new_data.append({
                "type": f'{ann['type']}',
                "image_path": orig_rel_path,
                "question": ann['question'],
                "answer": answers[0]
            })

            # 2. Tạo ảnh tăng cường
            # Lấy tên file không bao gồm phần mở rộng, thay dấu cách bằng gạch dưới cho an toàn
            base_name = os.path.basename(orig_rel_path).rsplit('.', 1)[0].replace(" ", "_")
            
            for i in range(num_aug_per_img):
                augmented = transform(image=image)['image']
                
                # Tên file mới (ví dụ: banh_cuon_0029_q0_aug0.jpg)
                new_img_filename = f"{base_name}_q{ann_idx}_aug{i}.jpg"
                save_path = os.path.normpath(os.path.join(abs_out_img_dir, new_img_filename))
                
                # --- CÁCH LƯU ẢNH HỖ TRỢ TIẾNG VIỆT ---
                img_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                is_success, buffer = cv2.imencode(".jpg", img_bgr)
                if is_success:
                    buffer.tofile(save_path)
                
                ans_idx = (i + 1) % len(answers)
                new_data.append({
                    "type": f'{ann['type']}',
                    "image_path": os.path.join("imgs", new_img_filename).replace("\\", "/"),
                    "question": ann['question'],
                    "answer": answers[ans_idx]
                })

    out_json_path = os.path.join(abs_output_dir, "data_final.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"\nThành công! Đã tạo {len(new_data)} mẫu dữ liệu.")

# --- Cấu hình chạy ---
# Giả sử file của bạn tên là 'train_vqa.json', ảnh ở 'images/', lưu ảnh mới vào 'augmented_images/'
augment_vqa_dataset(
    input_json=r'/home/haloha/Documents/brother/DL/test/raw/test.json', 
    img_root=r'/home/haloha/Documents/brother/DL/test/data/', 
    out_img_dir=r'/home/haloha/Documents/brother/DL/test/data/imgs',
    output_dir=r'/home/haloha/Documents/brother/DL/test/data', 
    num_aug_per_img=2
)