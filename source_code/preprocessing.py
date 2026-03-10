"""
In this file, tokenzation, vocabulary building, numercalization, batching and padding are implemented.
Again, almost all of the code comes from the jupyter notebook lstm_cnn_gclip_yelp.ipynb provided in the practical
We added some extra minor functionality and made small changes
Sources:
    lstm_cnn_gclip_yelp.ipynb
    PyTorch DataLoader: https://shorturl.at/OPP6i
    Pandas reset_index and .iloc: https://shorturl.at/RsjbL

"""
import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

#Regex tokenizer to extract alphanumeric sequences and apostrophes
TOKEN_RE = re.compile(r"[A-Za-z0-9']+") 
PAD = "<pad>"
UNK = "<unk>"

def tokenize(text: str) -> list: #Lowercase text and split into list of words according to regex.
    return TOKEN_RE.findall(str(text).lower())

def build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict: #Create a dictionary mapping words to unique integers
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

@dataclass #Custom data structure to hold batch of processed text
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
            label = int(item["label"]) - 1 #AG news labels are 1-4. Subtract 1 to make them 0-3 for PyTorch.
            
            return ids, label

class TextCollate:
    #Instead of a collate function (like in the notebook), we make it a class
    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch: list) -> Batch:
        #Extract true lengths of all sequences in this specific batch
        lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(batch) > 0 else 0
        
        #Create empty tensor with padding tokens
        x = torch.full((len(batch), max_len), self.pad_idx, dtype=torch.long)
        y = torch.tensor([y for _, y in batch], dtype=torch.long) #Extract labels
        
        for i, (ids, _) in enumerate(batch): #We overwrite padding tokens with actual word IDs for each sequence
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            
        return Batch(x=x, lengths=lengths, y=y)

def create_dataloaders(train_df, dev_df, test_df, max_len=128, batch_size=64):
    #A helper function to abstract away dataset creation.
    #First, we build vocab
    vocab = build_vocab(train_df["text"], min_freq=2, max_size=30000)
    pad_idx = vocab[PAD]
    
    #Initialize dataset
    train_ds = TextDataset(train_df, vocab, max_len=max_len)
    dev_ds = TextDataset(dev_df, vocab, max_len=max_len)
    test_ds = TextDataset(test_df, vocab, max_len=max_len)
    
    collate_fn = TextCollate(pad_idx=pad_idx) #Initialize collate function
    
    #Finally, we wrap in dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, vocab