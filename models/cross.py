"""Transformer CrossEncoder for the simple pipeline.

Single forward pass over the joint sequence produced by the tokenizer
(`tokenizer(text_a, text_b, …)` inserts [CLS]/[SEP] correctly). We take the
[CLS] hidden state and pass it through an MLP → sigmoid scalar in [0,1].
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import config as C
from models.dual import _patch_rope


class CrossEncoderScorer(nn.Module):
    """Transformer cross encoder with optional max-score scalar input (v03b)."""

    def __init__(
        self,
        backbone_name: str,
        dropout: float = C.TXFMR_DROPOUT,
        freeze_layers: int = C.TXFMR_FREEZE_N,
        max_feat_dim: int = 0,
    ):
        super().__init__()
        self.max_feat_dim = max_feat_dim
        cfg = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            backbone_name, config=cfg, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.hidden_dim = cfg.hidden_size
        _patch_rope(self.encoder)
        if freeze_layers > 0:
            self._freeze(freeze_layers)
        head_in = self.hidden_dim + max_feat_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 256),
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

    def forward(self, input_ids, attention_mask, max_score_feat=None):
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
        if self.max_feat_dim > 0:
            assert max_score_feat is not None, "max_score_feat required when max_feat_dim>0"
            cls = torch.cat([cls, max_score_feat], dim=1)
        return self.head(cls).squeeze(-1)
