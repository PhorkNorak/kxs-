"""Gradient x input saliency on the best transformer cell.

Approach: take the trained scalar score `s = sigmoid(MLP(...))`, treat it as a
function of the answer-side input embeddings, take `g = ds/dE`, then per-token
saliency = ||g_t * E_t||_2 with L2 norm over embedding dim. Render a PNG of
the token strip colored by saliency magnitude.

Only runs on transformer models (dual or cross). For dual models we saliency
the answer side (side_a). For cross models we saliency the full joint sequence.
"""

import os
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import config as C
from data import load_dataframe, split_dataframe, apply_preprocess, build_pair, build_single_pair


def _best_transformer_run():
    if not os.path.exists(C.LEADERBOARD):
        return None
    rows = []
    with open(C.LEADERBOARD, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["family"] in ("dual", "cross"):
                try:
                    r["test_qwk_f"] = float(r["test_qwk"])
                except Exception:
                    continue
                rows.append(r)
    if not rows:
        return None
    rows.sort(key=lambda r: r["test_qwk_f"], reverse=True)
    return rows[0]


def _load_run(run_id: str):
    run_dir = os.path.join(C.RUNS_DIR, run_id)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    ckpt = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        return None
    return cfg, torch.load(ckpt, map_location=C.DEVICE)


def _instantiate_model(cfg):
    if cfg["family"] == "dual":
        from models.dual import DualEncoderScorer
        return DualEncoderScorer(cfg["backbone"])
    if cfg["family"] == "cross":
        from models.cross import CrossEncoderScorer
        return CrossEncoderScorer(cfg["backbone"])
    raise ValueError(cfg["family"])


def _pick_samples(test_p, k_per_score: int = 2):
    chosen_idx = []
    for label in sorted(test_p["score_label"].unique()):
        sub = test_p[test_p["score_label"] == label]
        if len(sub) == 0:
            continue
        idxs = sub.sample(n=min(k_per_score, len(sub)), random_state=C.SEED).index.tolist()
        chosen_idx.extend(idxs)
    return chosen_idx


def _saliency_dual(model, tokenizer, side_a_text, side_b_text, device):
    enc_a = tokenizer(side_a_text, max_length=C.TXFMR_MAX_LEN, padding="max_length",
                      truncation=True, return_tensors="pt").to(device)
    enc_b = tokenizer(side_b_text, max_length=C.TXFMR_MAX_LEN, padding="max_length",
                      truncation=True, return_tensors="pt").to(device)
    model.train(False)
    embed_layer = model.encoder.get_input_embeddings()
    E_a = embed_layer(enc_a["input_ids"]).detach()
    E_a.requires_grad_(True)
    E_b = embed_layer(enc_b["input_ids"]).detach()

    bs, sl = enc_a["input_ids"].shape
    position_ids = torch.arange(sl, device=device).unsqueeze(0).expand(bs, -1)
    out_a = model.encoder(inputs_embeds=E_a, attention_mask=enc_a["attention_mask"],
                          position_ids=position_ids, return_dict=True)
    e_a = model._pool(out_a.last_hidden_state, enc_a["attention_mask"])
    out_b = model.encoder(inputs_embeds=E_b, attention_mask=enc_b["attention_mask"],
                          position_ids=position_ids, return_dict=True)
    e_b = model._pool(out_b.last_hidden_state, enc_b["attention_mask"])
    inter = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b], dim=1)
    score = model.head(inter).squeeze(-1)

    score.sum().backward()
    grad = E_a.grad.detach()
    sal = (grad * E_a.detach()).norm(dim=-1).squeeze(0).cpu().numpy()
    mask = enc_a["attention_mask"].squeeze(0).cpu().numpy()
    sal = sal * mask
    tokens = tokenizer.convert_ids_to_tokens(enc_a["input_ids"].squeeze(0).cpu().tolist())
    return tokens, sal, float(score.detach().cpu().item()), mask


def _saliency_cross(model, tokenizer, text_a, text_b, device):
    enc = tokenizer(text_a, text_b, max_length=C.TXFMR_MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    model.train(False)
    embed_layer = model.encoder.get_input_embeddings()
    E = embed_layer(enc["input_ids"]).detach()
    E.requires_grad_(True)
    bs, sl = enc["input_ids"].shape
    position_ids = torch.arange(sl, device=device).unsqueeze(0).expand(bs, -1)
    out = model.encoder(inputs_embeds=E, attention_mask=enc["attention_mask"],
                        position_ids=position_ids, return_dict=True)
    cls = out.last_hidden_state[:, 0, :]
    score = model.head(cls).squeeze(-1)
    score.sum().backward()
    grad = E.grad.detach()
    sal = (grad * E.detach()).norm(dim=-1).squeeze(0).cpu().numpy()
    mask = enc["attention_mask"].squeeze(0).cpu().numpy()
    sal = sal * mask
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0).cpu().tolist())
    return tokens, sal, float(score.detach().cpu().item()), mask


def _render_png(tokens, sal, mask, out_path, title):
    valid = mask.astype(bool)
    tokens = [t for t, m in zip(tokens, valid) if m]
    sal = sal[valid]
    if len(tokens) == 0:
        return
    sal = sal / (sal.max() + 1e-9)

    cmap = LinearSegmentedColormap.from_list("sal", ["#ffffff", "#ff5555"])
    n = len(tokens)
    cols = min(20, n)
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 0.6, rows * 0.55 + 0.6))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows + 0.5)
    ax.axis("off")
    for i, (tok, s) in enumerate(zip(tokens, sal)):
        r = rows - 1 - (i // cols)
        c = i % cols
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=cmap(float(s))))
        ax.text(c + 0.5, r + 0.5, tok.replace("▁", "_"),
                ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def run_xai_on_best(k_per_score: int = 2):
    best = _best_transformer_run()
    if best is None:
        print("[xai] no transformer rows on leaderboard yet, skipping")
        return
    run_id = best["run_id"]
    print(f"[xai] best transformer = {run_id}  (test_qwk={best['test_qwk']})")

    loaded = _load_run(run_id)
    if loaded is None:
        print(f"[xai] no best.pt for {run_id}, skipping")
        return
    cfg, state = loaded
    model = _instantiate_model(cfg).to(C.DEVICE)
    model.load_state_dict(state)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], trust_remote_code=True)

    df = load_dataframe()
    _, _, test_df = split_dataframe(df)
    test_p = apply_preprocess(test_df, cfg["preprocess"])

    idxs = _pick_samples(test_p, k_per_score=k_per_score)
    out_dir = os.path.join(C.XAI_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[xai] rendering {len(idxs)} samples -> {out_dir}")

    for i, idx in enumerate(idxs):
        row = test_p.iloc[idx]
        true_label = int(row["score_label"])
        if cfg["family"] == "dual":
            a, b = build_pair(row, cfg["input"])
            tokens, sal, pred, mask = _saliency_dual(model, tokenizer, a, b, C.DEVICE)
        else:
            ta, tb = build_single_pair(row, cfg["input"])
            tokens, sal, pred, mask = _saliency_cross(model, tokenizer, ta, tb, C.DEVICE)
        pred_label = int(round(max(0.0, min(1.0, pred)) * 4))
        title = f"{run_id}  sample {i}  true={true_label}  pred={pred_label}  raw={pred:.3f}"
        out_path = os.path.join(out_dir, f"sample_{i:02d}_true{true_label}_pred{pred_label}.png")
        _render_png(tokens, sal, mask, out_path, title)
    print(f"[xai] done: {len(idxs)} PNGs in {out_dir}")


if __name__ == "__main__":
    run_xai_on_best()
