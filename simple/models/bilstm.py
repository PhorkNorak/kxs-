"""Character-level BiLSTM + attention with a scalar regression head.

Two-sided: encodes side_a and side_b independently, then builds the 4-way
interaction [e_a; e_b; |e_a-e_b|; e_a*e_b] → MLP → sigmoid scalar in [0,1].
"""

import torch
import torch.nn as nn

from simple import config as C


class BiLSTMScorer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = C.BILSTM_EMBED,
        hidden_dim: int = C.BILSTM_HIDDEN,
        num_layers: int = C.BILSTM_LAYERS,
        dropout: float = C.BILSTM_DROPOUT,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def encode(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        x, _ = self.lstm(x)
        w = self.attn(x).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        a = torch.softmax(w, dim=1).unsqueeze(1)
        return (a @ x).squeeze(1)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        e_a = self.encode(input_ids_a, attention_mask_a)
        e_b = self.encode(input_ids_b, attention_mask_b)
        inter = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b], dim=1)
        return self.head(inter).squeeze(-1)
