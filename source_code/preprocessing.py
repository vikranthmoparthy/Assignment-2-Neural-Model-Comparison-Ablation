import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
PAD = "<pad>"
UNK = "<unk>"

def tokenize(text: str) -> list:
    return TOKEN_RE.findall(str(text).lower())

def build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
        
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab

def numericalize(tokens: list, vocab: dict) -> list:
    return [vocab.get(tok, vocab[UNK]) for tok in tokens]

@dataclass
class Batch:
    x: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor

class TextDataset(Dataset):
    def __init__(self, df, vocab: dict, max_len: int = 200) -> None:
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
            item = self.df.iloc[idx]
            tokens = tokenize(item["text"])

            if len(tokens) == 0:
                ids = [self.vocab[UNK]]
            else:
                ids = numericalize(tokens, self.vocab)[: self.max_len]
                if len(ids) == 0:
                    ids = [self.vocab[UNK]]
                    
            label = int(item["label"]) - 1 
            
            return ids, label

class TextCollate:
    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch: list) -> Batch:
        lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(batch) > 0 else 0
        
        x = torch.full((len(batch), max_len), self.pad_idx, dtype=torch.long)
        y = torch.tensor([y for _, y in batch], dtype=torch.long)
        
        for i, (ids, _) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            
        return Batch(x=x, lengths=lengths, y=y)

def create_dataloaders(train_df, dev_df, test_df, max_len=128, batch_size=64):
    vocab = build_vocab(train_df["text"], min_freq=2, max_size=30000)
    pad_idx = vocab[PAD]
    
    train_ds = TextDataset(train_df, vocab, max_len=max_len)
    dev_ds = TextDataset(dev_df, vocab, max_len=max_len)
    test_ds = TextDataset(test_df, vocab, max_len=max_len)
    
    collate_fn = TextCollate(pad_idx=pad_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, vocab