import json
import os
from torch.utils.data import Dataset
from urllib.parse import unquote
from pathlib import Path

class VQADataset(Dataset):
    def __init__(self, json_file, img_dir):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_img_path = item['image_path']
        # Thử các cách kết hợp đường dẫn
        potential_path = os.path.join(self.img_dir, raw_img_path)
        if not os.path.exists(potential_path):
            potential_path = os.path.join(self.img_dir, os.path.basename(raw_img_path))

        if not os.path.exists(potential_path):
            raise FileNotFoundError(f"Image not found: {potential_path}")

        abs_image_path = os.path.abspath(potential_path)
        
        # Cách an toàn nhất để tạo file:// URL trên cả Windows và Linux
        image_url = Path(abs_image_path).as_uri()
        image_path = unquote(abs_image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        # Giảm pixel giúp tránh OOM trên card 8GB-12GB
                        "min_pixels": 224 * 224, 
                        "max_pixels": 224 * 224, 
                    },
                    {"type": "text", "text": item['question']},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item.get('answer', "")}]
            }
        ]
        return {
            "messages": messages,
            "answer": item.get('answer', ""),
            "image_path": raw_img_path,
            "type": item.get('type', 'general'),
            "question": item['question']
        }