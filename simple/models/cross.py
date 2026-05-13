"""Transformer CrossEncoder for the simple pipeline.

Single forward pass over the joint sequence produced by the tokenizer
(`tokenizer(text_a, text_b, …)` inserts [CLS]/[SEP] correctly). We take the
[CLS] hidden state and pass it through an MLP → sigmoid scalar in [0,1].
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from simple import config as C
from simple.models.dual import _patch_rope


class CrossEncoderScorer(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        dropout: float = C.TXFMR_DROPOUT,
        freeze_layers: int = C.TXFMR_FREEZE_N,
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            backbone_name, config=cfg, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.hidden_dim = cfg.hidden_size
        _patch_rope(self.encoder)
        if freeze_layers > 0:
            self._freeze(freeze_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def _freeze(self, n: int):
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        layers = getattr(self.encoder, "encoder", self.encoder)
        if hasattr(layers, "layer"):
            total = len(layers.layer)
            n_freeze = min(n, max(0, total - 2))
            for i, layer in enumerate(layers.layer):
                if i < n_freeze:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bs, sl = input_ids.shape
        position_ids = (
            torch.arange(sl, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bs, -1)
        )
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)
