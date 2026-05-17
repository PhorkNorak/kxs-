"""Transformer DualEncoder for the simple pipeline.

Shared HF encoder over side_a and side_b separately, mean-pool with the
attention mask, build the 4-way interaction, then a small MLP → sigmoid scalar.

GTE's rotary cache ships with NaN under some load paths; we rebuild it in
fp32 (port of `_patch_rope` from kxs/models/dual_encoder.py).
"""

import types
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import config as C


def _patch_rope(encoder):
    emb = getattr(encoder, "embeddings", None)
    if emb is None:
        return
    re = getattr(emb, "rotary_emb", None)
    if re is None:
        return

    seq_len = int(re.max_seq_len_cached)
    device = re.cos_cached.device
    dim = re.dim
    base = re.base * (re.scaling_factor if getattr(re, "mixed_b", None) is None else 1)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if getattr(re, "mixed_b", None) is None:
        inv_freq = inv_freq / (re.scaling_factor ** (2 / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb_full = torch.cat((freqs, freqs), dim=-1)
    re.register_buffer("inv_freq", inv_freq, persistent=False)
    re.register_buffer("cos_cached", emb_full.cos(), persistent=False)
    re.register_buffer("sin_cached", emb_full.sin(), persistent=False)

    def _safe_forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, torch.float32)
        cos = self.cos_cached.to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached.to(device=x.device, dtype=x.dtype)
        return cos, sin

    re.forward = types.MethodType(_safe_forward, re)


class DualEncoderScorer(nn.Module):
    """Transformer dual encoder with optional max-score scalar input (v03b)."""

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
        head_in = 4 * self.hidden_dim + max_feat_dim
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

    @staticmethod
    def _pool(hidden, mask):
        m = mask.unsqueeze(-1).expand(hidden.size()).to(dtype=hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def encode(self, input_ids, attention_mask):
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
        return self._pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b,
                max_score_feat=None):
        e_a = self.encode(input_ids_a, attention_mask_a)
        e_b = self.encode(input_ids_b, attention_mask_b)
        inter = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b], dim=1)
        if self.max_feat_dim > 0:
            assert max_score_feat is not None, "max_score_feat required when max_feat_dim>0"
            inter = torch.cat([inter, max_score_feat], dim=1)
        return self.head(inter).squeeze(-1)
