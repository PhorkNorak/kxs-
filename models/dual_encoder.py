"""
Transformer Scoring Models
- DualEncoder: Siamese with 4-way interaction [e_A; e_R; e_A-e_R; e_A⊙e_R] (KX-CL)
- CrossEncoder: Joint encoding ablation

References:
    Conneau et al. 2017 (InferSent) — 4-way interaction
    Reimers & Gurevych 2019 (SBERT) — dual-encoder
    Gal & Ghahramani 2016 — MC Dropout
"""

import types
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from models.losses import corn_logits_to_label, corn_logits_to_score


def _load_encoder(model_name, config):
    return AutoModel.from_pretrained(
        model_name, config=config, trust_remote_code=True, torch_dtype=torch.float32
    )


def _patch_rope(encoder):
    """Patch GTE-style NTK RoPE: rebuild cos/sin cache in fp32 to avoid fp16
    overflow in `t * inv_freq` (the largest products exceed fp16's 65504 max
    and produce NaN through cos()), then return the full cache (no slicing) on
    the correct device.
    """
    emb = getattr(encoder, 'embeddings', None)
    if emb is None:
        return
    re = getattr(emb, 'rotary_emb', None)
    if re is None:
        return
    print(f"    [RoPE] cos_cached.shape={re.cos_cached.shape}  "
          f"max_seq_len_cached={re.max_seq_len_cached}  "
          f"cos.dtype={re.cos_cached.dtype}  "
          f"cos.nan_count={int(torch.isnan(re.cos_cached).sum())}")

    # Rebuild cache manually in fp32. The model's _set_cos_sin_cache may have
    # produced NaN if it ran under an fp16 default-dtype context during loading.
    seq_len = int(re.max_seq_len_cached)
    device = re.cos_cached.device
    dim = re.dim
    base = re.base * (re.scaling_factor if getattr(re, "mixed_b", None) is None else 1)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if getattr(re, "mixed_b", None) is None:
        inv_freq = inv_freq / (re.scaling_factor ** (2 / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    re.register_buffer("inv_freq", inv_freq, persistent=False)
    re.register_buffer("cos_cached", emb.cos(), persistent=False)
    re.register_buffer("sin_cached", emb.sin(), persistent=False)
    print(f"    [RoPE] after manual rebuild: cos.dtype={re.cos_cached.dtype}  "
          f"cos.nan_count={int(torch.isnan(re.cos_cached).sum())}  "
          f"freqs.max={float(freqs.abs().max()):.3g}")

    _diag = {"printed": False}

    def _safe_forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, torch.float32)
        cos = self.cos_cached.to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached.to(device=x.device, dtype=x.dtype)
        if not _diag["printed"]:
            print(f"    [RoPE call] x.shape={tuple(x.shape)}  x.device={x.device}  "
                  f"seq_len={seq_len}  cos.shape={tuple(cos.shape)}  cos.device={cos.device}  "
                  f"cos.nan={int(torch.isnan(cos).sum())}")
            _diag["printed"] = True
        return cos, sin

    re.forward = types.MethodType(_safe_forward, re)


class DualEncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_classes=5,
                 dropout=0.2, freeze_layers=6, loss_type="corn"):
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = _load_encoder(model_name, config)
        self.hidden_dim = config.hidden_size
        _patch_rope(self.encoder)
        if freeze_layers > 0:
            self._freeze(freeze_layers)
        out = (num_classes - 1) if loss_type == "corn" else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * self.hidden_dim, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out),
        )
        self.scl_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def _freeze(self, n):
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

    def _pool(self, hidden, mask):
        m = mask.unsqueeze(-1).expand(hidden.size()).to(dtype=hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def _install_nan_hooks(self):
        if getattr(self, "_nan_hooks_done", False):
            return
        self._nan_hooks_done = True
        state = {"first": None}

        def make_hook(name):
            def hook(_mod, _inp, out):
                if state["first"] is not None:
                    return
                t = out[0] if isinstance(out, tuple) else out
                if not torch.is_tensor(t):
                    return
                if torch.isnan(t).any() or torch.isinf(t).any():
                    state["first"] = name
                    nan_n = torch.isnan(t).sum().item()
                    inf_n = torch.isinf(t).sum().item()
                    print(f"\n  [NaN-loc] FIRST nan/inf at module: {name}")
                    print(f"  [NaN-loc]   out.shape={tuple(t.shape)}  nan={nan_n}  inf={inf_n}")
                    # also print input stats
                    if isinstance(_inp, tuple) and len(_inp) > 0 and torch.is_tensor(_inp[0]):
                        x = _inp[0]
                        print(f"  [NaN-loc]   in.shape={tuple(x.shape)}  "
                              f"in.nan={torch.isnan(x).any().item()}  "
                              f"in.min={x.float().min().item():.3g}  in.max={x.float().max().item():.3g}")
            return hook

        for name, mod in self.encoder.named_modules():
            if name == "":
                continue
            mod.register_forward_hook(make_hook(name))

    def encode(self, input_ids, attention_mask):
        self._install_nan_hooks()
        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, return_dict=True
        )
        return self._pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids_a, attention_mask_a, input_ids_r, attention_mask_r,
                return_embeddings=False):
        e_a = self.encode(input_ids_a, attention_mask_a)
        e_r = self.encode(input_ids_r, attention_mask_r)
        inter = torch.cat([e_a, e_r, e_a - e_r, e_a * e_r], dim=1)
        logits = self.head(inter)
        if return_embeddings:
            return logits, self.scl_projector(e_a)
        return logits

    def predict_with_uncertainty(self, input_ids_a, attention_mask_a,
                                 input_ids_r, attention_mask_r, T=10):
        self.train()  # Enable dropout
        scores_list, labels_list = [], []
        with torch.no_grad():
            for _ in range(T):
                logits = self.forward(input_ids_a, attention_mask_a,
                                      input_ids_r, attention_mask_r)
                if self.loss_type == "corn":
                    scores_list.append(corn_logits_to_score(logits, self.num_classes))
                    labels_list.append(corn_logits_to_label(logits))
                else:
                    s = logits.squeeze(-1).clamp(0, 1)
                    scores_list.append(s)
                    labels_list.append((s * 4).round().long().clamp(0, 4))
        self.eval()
        st = torch.stack(scores_list)
        lt = torch.stack(labels_list)
        return {
            "mean_score": st.mean(0), "std_score": st.std(0),
            "mode_label": lt.mode(0).values, "raw_scores": st,
        }


class CrossEncoder(nn.Module):
    """Joint-encoding ablation (no separable embeddings → no SCL)."""
    def __init__(self, model_name="xlm-roberta-base", num_classes=5,
                 dropout=0.2, freeze_layers=6, loss_type="corn"):
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = _load_encoder(model_name, config)
        self.hidden_dim = config.hidden_size
        out = (num_classes - 1) if loss_type == "corn" else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out),
        )

    def forward(self, input_ids_a, attention_mask_a, input_ids_r, attention_mask_r,
                return_embeddings=False):
        comb_ids = torch.cat([input_ids_a, input_ids_r[:, 1:]], dim=1)[:, :512]
        comb_mask = torch.cat([attention_mask_a, attention_mask_r[:, 1:]], dim=1)[:, :512]
        out = self.encoder(input_ids=comb_ids, attention_mask=comb_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls)
        return (logits, cls) if return_embeddings else logits


def create_model(backbone="xlmr", topology="dual", loss_type="corn",
                 num_classes=5, dropout=0.2, freeze_layers=6):
    from config import TRANSFORMER_BACKBONES
    name = TRANSFORMER_BACKBONES[backbone]
    if topology == "dual":
        return DualEncoder(name, num_classes, dropout, freeze_layers, loss_type)
    return CrossEncoder(name, num_classes, dropout, freeze_layers, loss_type)
