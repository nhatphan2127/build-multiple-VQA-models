import json
import os
import random

def split_data(json_path, output_dir, train_ratio=.889, force_split=False):
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    if not force_split and os.path.exists(train_path) and os.path.exists(val_path):
        print("Data already split. Loading existing splits.")
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)

        return train_data, val_data

    print("Splitting data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    total = len(data)
    train_end = int(train_ratio * total)

    train_data = data[:train_end]
    val_data = data[train_end:]

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    return train_data, val_data
