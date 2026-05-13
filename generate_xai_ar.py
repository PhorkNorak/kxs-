"""
Generate results/xai_ar_data.json for the BiLSTM AR model.
This file is consumed by xai_visualizer.py.

Prerequisites:
    python run_all.py --stage prepare
    python run_all.py --stage bilstm

Run:
    python generate_xai_ar.py
    python generate_xai_ar.py --n_samples 20 --seed 0
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (RESULTS_DIR, PROCESSED_DIR, CHECKPOINT_DIR,
                    BILSTM_CFG, DEVICE, SHAP_N_SAMPLES)
from models.baselines import BiLSTMAttention
from models.char_tokenizer import CharTokenizer


def load_splits_kcc():
    splits = {}
    for name in ["train", "val", "test"]:
        p = os.path.join(PROCESSED_DIR, f"{name}_kcc.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} — run: python run_all.py --stage prepare")
        splits[name] = pd.read_csv(p)
    return splits["train"], splits["val"], splits["test"]


def build_tokenizer(train_df, suffix):
    cfg = BILSTM_CFG
    tok = CharTokenizer(cfg.max_vocab).fit(
        train_df[f"Answer{suffix}"].tolist() + train_df[f"Reference{suffix}"].tolist()
    )
    return tok


def compute_saliency(model, tok, answer_text, ref_text, max_seq_len, device):
    """Return (char_saliency, pred_label, pred_score, uncertainty)."""
    ids_a, mask_a = tok.encode(answer_text, max_len=max_seq_len)
    ids_r, mask_r = tok.encode(ref_text,    max_len=max_seq_len)

    ids_a_t  = torch.tensor([ids_a],  dtype=torch.long,  device=device)
    mask_a_t = torch.tensor([mask_a], dtype=torch.long,  device=device)
    ids_r_t  = torch.tensor([ids_r],  dtype=torch.long,  device=device)
    mask_r_t = torch.tensor([mask_r], dtype=torch.long,  device=device)

    # Gradient saliency: norm of embedding gradient per character position
    model.train(False)
    emb = model.emb(ids_a_t).detach().requires_grad_(True)
    logits = model.forward_from_emb(emb, mask_a_t, ids_r_t, mask_r_t)
    logits[0, 0].backward()  # gradient of first CORN ordinal logit

    sal = emb.grad[0].norm(dim=-1).detach().cpu().numpy()  # [max_seq_len]

    # Position 0 is CLS; answer chars are at positions 1..n_chars
    chars = list(str(answer_text))
    n_chars = min(len(chars), max_seq_len - 2)
    raw_sal = sal[1 : n_chars + 1]

    s_min, s_max = raw_sal.min(), raw_sal.max()
    norm_sal = (raw_sal - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(raw_sal)

    char_saliency = [[ch, float(s)] for ch, s in zip(chars[:n_chars], norm_sal)]

    # MC Dropout uncertainty (T=10 forward passes with dropout enabled)
    with torch.no_grad():
        res = model.predict_with_uncertainty(ids_a_t, mask_a_t, ids_r_t, mask_r_t, T=10)
    pred_label  = int(res["mode_label"].item())
    pred_score  = float(res["mean_score"].item())
    uncertainty = float(res["std_score"].item())

    return char_saliency, pred_label, pred_score, uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=SHAP_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("KhmerXScore — Generate XAI data (BiLSTM AR)")
    print("=" * 50)

    train_df, _, test_df = load_splits_kcc()
    suffix = "_proc" if "Answer_proc" in train_df.columns else ""
    tok = build_tokenizer(train_df, suffix)
    print(f"  vocab={tok.vocab_size}  test_n={len(test_df)}")

    cfg = BILSTM_CFG
    ckpt_path = os.path.join(CHECKPOINT_DIR, "bilstm_ar_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found.")
        print("  Run first: python run_all.py --stage bilstm")
        return

    model = BiLSTMAttention(
        vocab_size=tok.vocab_size, embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        dropout=cfg.dropout, num_classes=5, loss_type="corn",
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).train(False)
    print(f"  Loaded: {ckpt_path}")

    # Stratified sample: ~n//5 examples per label for visual diversity
    n = min(args.n_samples, len(test_df))
    rng = np.random.default_rng(args.seed)
    indices = []
    per_label = max(1, n // 5)
    for label in range(5):
        pool = np.where(test_df["score_label"].values == label)[0]
        chosen = rng.choice(pool, size=min(per_label, len(pool)), replace=False)
        indices.extend(chosen.tolist())
    indices = sorted(set(indices))[:n]
    print(f"  Selected {len(indices)} samples (stratified across labels 0-4)")

    samples = []
    for step, idx in enumerate(indices):
        row = test_df.iloc[idx]
        answer_text = str(row[f"Answer{suffix}"])
        ref_text    = str(row[f"Reference{suffix}"])
        question    = str(row.get("Question", row.get(f"Question{suffix}", "")))
        subject     = str(row.get("Subject", ""))
        true_label  = int(row["score_label"])
        true_score  = float(row["normalized_score"])

        print(f"  [{step+1:2d}/{len(indices)}] idx={idx:<4} label={true_label}  "
              f"len={len(answer_text):<4}", end="  ")

        char_sal, pred_label, pred_score, unc = compute_saliency(
            model, tok, answer_text, ref_text, cfg.max_seq_len, DEVICE)

        correct = (true_label == pred_label)
        print(f"pred={pred_label}  unc={unc:.4f}  {'OK' if correct else '--'}")

        samples.append({
            "idx":            idx,
            "student_answer": answer_text,
            "reference":      ref_text,
            "question":       question,
            "subject":        subject,
            "true_label":     true_label,
            "pred_label":     pred_label,
            "true_score":     true_score,
            "pred_score":     pred_score,
            "correct":        correct,
            "uncertainty":    unc,
            "char_saliency":  char_sal,
            "model_window":   cfg.max_seq_len - 2,
        })

    out_path = os.path.join(RESULTS_DIR, "xai_ar_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    acc = sum(1 for s in samples if s["correct"]) / len(samples)
    print(f"\nSaved {len(samples)} samples → {out_path}")
    print(f"Accuracy: {acc:.1%}  |  Mean uncertainty: "
          f"{np.mean([s['uncertainty'] for s in samples]):.4f}")


if __name__ == "__main__":
    main()
