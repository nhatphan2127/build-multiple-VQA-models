import json
from collections import Counter
import torch
from underthesea import word_tokenize

class Vocab:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", start_token="<sos>", end_token="<eos>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        
        self.word2idx = {pad_token: 0, unk_token: 1, start_token: 2, end_token: 3}
        self.idx2word = {0: pad_token, 1: unk_token, 2: start_token, 3: end_token}
        self.vocab_size = 4

    def build_vocab(self, texts, min_freq=0):
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                tokens = word_tokenize(text.lower())
                counter.update(tokens)
        
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text, max_len=None, add_special_tokens=True):
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())
        else:
            tokens = text

        ids = []
        if add_special_tokens:
            ids.append(self.word2idx[self.start_token])
        
        for token in tokens:
            ids.append(self.word2idx.get(token, self.word2idx[self.unk_token]))
            
        if add_special_tokens:
            ids.append(self.word2idx[self.end_token])
            
        if max_len is not None:
            if len(ids) < max_len:
                ids += [self.word2idx[self.pad_token]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.idx2word.get(idx, self.unk_token)
            if skip_special_tokens and word in [self.pad_token, self.start_token, self.end_token]:
                if word == self.end_token:
                    break
                continue
            words.append(word)
        return " ".join(words)

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': {str(k): v for k, v in self.idx2word.items()}}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.vocab_size = len(vocab.word2idx)
        return vocab
