import json 
import random
import os

OUTPUT_DIR = 'raw'
# Đảm bảo thư mục tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Đọc dữ liệu từ data1.json
with open(f'{OUTPUT_DIR}/data1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

random.shuffle(data)

# 2. Lấy ra 50 phần tử đầu tiên
test_part = []
for i in range(min(50, len(data))): # Thêm min để tránh lỗi nếu data có ít hơn 50 phần tử
    test_part.append(data.pop(0))

# 3. Đọc dữ liệu cũ từ test.json (nếu có) để gộp vào
try:
    with open(f'{OUTPUT_DIR}/test.json', 'r', encoding='utf-8') as file:
        test = json.load(file)
except FileNotFoundError:
    test = [] # Nếu chưa có file test.json thì tạo list rỗng

# 4. Gộp và lưu vào test1.json
test.extend(test_part) # Gộp phần mới vào phần cũ
with open(f'{OUTPUT_DIR}/test1.json', 'w', encoding='utf-8') as file:
    json.dump(test, file, ensure_ascii=False, indent=4)

# 5. Lưu lại data1.json sau khi đã lấy mất 50 phần tử
with open(f'{OUTPUT_DIR}/data1.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Đã tách 50 mẫu và cập nhật các file thành công!")