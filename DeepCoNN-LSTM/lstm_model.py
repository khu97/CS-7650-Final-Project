import torch
from torch import nn
from utils import *
from config import *

class LSTMRecommender(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embedding_matrix=None, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False  

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def encode(self, x):
        """
        x shape: (B, R, L) → flatten identical to DeepCoNN
        """

        B, R, L = x.shape
        x = x.reshape(B * R, L)  

        emb = self.embedding(x)  

        out, (h, c) = self.lstm(emb)
        final = h[-1]


        final = final.reshape(B, R, self.hidden_dim)

        final = final.mean(dim=1)

        return final

    def forward(self, user_reviews, item_reviews):
        u = self.encode(user_reviews)
        i = self.encode(item_reviews)

        concat = torch.cat([u, i], dim=1)
        return self.fc(concat)
