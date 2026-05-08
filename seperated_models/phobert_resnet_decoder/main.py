import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from model import VQAModel
from dataset import VQADataset

sys.path.append('../../')
from utils.vocab import Vocab
from utils.data_splitter import split_data
import matplotlib.pyplot as plt
from tqdm import tqdm

# Settings
DATA_PATH = "./data/data.json"
TEST_PATH = "./data/test.json"

IMG_ROOT = "./data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-4
MODEL_DIR = "seperated_models/phobert_resnet_decoder/checkpoint"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(force_split=False, decoder_type="transformer"):
    # Split/Load Data
    train_data, val_data = split_data(DATA_PATH, MODEL_DIR, force_split=force_split)
    with open(TEST_PATH, 'r', encoding='utf-8') as file:
        test_data = json.loads(file.read())
    
    # Build/Load Vocab for answers
    all_answers = [item['answer'] for item in train_data] # Build vocab only from train_data ideally
    answer_vocab = Vocab()
    answer_vocab.build_vocab(all_answers)
    answer_vocab.save(os.path.join(MODEL_DIR, "vocab.json"))
    
    # Save config for loading
    config = {}
    if decoder_type == 'transformer':

        config = {
            'vocab_size': answer_vocab.vocab_size,
            'decoder_type': decoder_type,
            'd_model': 256,
            'max_q_len': 20,
            'max_a_len': 10
        }
    else:
        config = {
            'vocab_size': answer_vocab.vocab_size,
            'decoder_type': decoder_type,
            'd_model': 256,
            'max_q_len': 20,
            'max_a_len': 10
        }
    with open(os.path.join(MODEL_DIR, f"config_{decoder_type}.json"), 'w') as f:
        json.dump(config, f)

    # Tokenizer for PhoBERT
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # Datasets
    train_ds = VQADataset(train_data, phobert_tokenizer, answer_vocab, img_root=IMG_ROOT)
    val_ds = VQADataset(val_data, phobert_tokenizer, answer_vocab, img_root=IMG_ROOT)
    test_ds = VQADataset(test_data, phobert_tokenizer, answer_vocab, img_root=IMG_ROOT)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = VQAModel(vocab_size=config['vocab_size'], d_model=config['d_model'], decoder_type=config['decoder_type']).to(DEVICE)
    
    # Param analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=answer_vocab.word2idx[answer_vocab.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = batch['image'].to(DEVICE)
            q_ids = batch['q_ids'].to(DEVICE)
            q_mask = batch['q_mask'].to(DEVICE)
            a_ids = batch['a_ids'].to(DEVICE)

            optimizer.zero_grad()
            target_input = a_ids[:, :-1]
            target_expected = a_ids[:, 1:]
            
            outputs = model(images, q_ids, q_mask, target_input)
            
            loss = criterion(outputs.reshape(-1, answer_vocab.vocab_size), target_expected.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                q_ids = batch['q_ids'].to(DEVICE)
                q_mask = batch['q_mask'].to(DEVICE)
                a_ids = batch['a_ids'].to(DEVICE)

                target_input = a_ids[:, :-1]
                target_expected = a_ids[:, 1:]
                outputs = model(images, q_ids, q_mask, target_input)
                loss = criterion(outputs.reshape(-1, answer_vocab.vocab_size), target_expected.reshape(-1))
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_model_{decoder_type}.pth"))
            print("Saved best model.")

    # Save Losses Graph
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Val Loss')
    plt.title(f'Losses - {decoder_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f"loss_graph_{decoder_type}.png"))
    
    return model, test_loader, answer_vocab



if __name__ == "__main__":
    for decoder_type in ['transformer', 'lstm']:
        model, test_loader, vocab = train_model(force_split=False, decoder_type=decoder_type)

        
