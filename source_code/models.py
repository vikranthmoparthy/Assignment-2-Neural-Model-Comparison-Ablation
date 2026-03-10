"""
In this file, we define the NN models required for the assignment: LSTM and CNN.
They take as input tokenized integer sequences and output class scores for the 4 AG news categories.
The functionality in this file was taken entirely from the jupyter notebook: lstm_cnn_gclip_yelp.ipynb
Source:
    lstm_cnn_gclip_yelp.ipynb
"""

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    #An LSTM network that processes text sequentially to capture long-term dependencies.
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        #This parameter is different compared to the Jupyter notebook
        #The AG news dataset has 4 classes, rather than 2.
        num_classes: int = 4, 
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        #Convert integer token IDs into floating point vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        #Core LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, #Dropout between layers if more than 1
            bidirectional=bidirectional,
        )
        rep_dim = hidden_dim * (2 if bidirectional else 1) #Hidden state size is doubled if bidirectional
        self.rep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rep_dim, num_classes) #Mapping LSTM output to 4 class logits

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        rep = self.rep_dropout(h_last)
        return self.fc(rep)

class CNNTextClassifier(nn.Module):
    #1D CNN for classification.
    def __init__( #Except for num_classes, all other params are same as notebook.
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
             for k in kernel_sizes]
        )
        self.rep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))
        emb_t = emb.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(emb_t)) #Applying convolution and ReLU
            p = torch.max(z, dim=2).values
            pooled.append(p)
        rep = torch.cat(pooled, dim=1)
        rep = self.rep_dropout(rep)
        return self.fc(rep)