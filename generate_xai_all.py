"""
Generate results/xai_{model_key}_data.json for all model families.

Models covered
  Baselines:    mean_ar, tfidf_cosine_ar, tfidf_svr_ar, fasttext_cosine_ar
  BiLSTM:       bilstm_ar, bilstm_qar
  Transformers: xlmr_ar, xlmr_qar, mbert_ar, mbert_qar, gte_ar, gte_qar

Prerequisites (run before this script):
  python run_all.py --stage prepare
  python run_all.py --stage baselines
  python run_all.py --stage bilstm
  python run_all.py --stage transformers

Run:
  python generate_xai_all.py
  python generate_xai_all.py --models bilstm_ar xlmr_ar
  python generate_xai_all.py --n_samples 15 --seed 0
"""

import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (RESULTS_DIR, PROCESSED_DIR, CHECKPOINT_DIR,
                    BILSTM_CFG, TRAIN_CFG, TRANSFORMER_BACKBONES,
                    DEVICE, SHAP_N_SAMPLES)
from models.baselines import BiLSTMAttention, create_baseline
from models.char_tokenizer import CharTokenizer
from models.dual_encoder import create_model
from models.losses import corn_logits_to_label, corn_logits_to_score
from data import KhmerSAGTextDataset


MODEL_SPECS = {
    # Classical baselines -- deterministic, no gradient saliency
    "mean_ar":             {"type": "baseline",    "fmt": "ar",  "baseline": "mean_predictor",   "name": "Mean Predictor (AR)"},
    "tfidf_cosine_ar":     {"type": "baseline",    "fmt": "ar",  "baseline": "tfidf_cosine",     "name": "TF-IDF Cosine (AR)"},
    "tfidf_svr_ar":        {"type": "baseline",    "fmt": "ar",  "baseline": "tfidf_svr",        "name": "TF-IDF + SVR (AR)"},
    "fasttext_cosine_ar":  {"type": "baseline",    "fmt": "ar",  "baseline": "fasttext_cosine",  "name": "FastText Cosine (AR)"},
    # BiLSTM -- character-level gradient saliency
    "bilstm_ar":           {"type": "bilstm",      "fmt": "ar",  "name": "BiLSTM+Attn (AR)"},
    "bilstm_qar":          {"type": "bilstm",      "fmt": "qar", "name": "BiLSTM+Attn (QAR)"},
    # Transformers -- subword token saliency
    "xlmr_ar":             {"type": "transformer", "fmt": "ar",  "backbone": "xlmr",  "name": "XLM-R Dual (AR)"},
    "xlmr_qar":            {"type": "transformer", "fmt": "qar", "backbone": "xlmr",  "name": "XLM-R Dual (QAR)"},
    "mbert_ar":            {"type": "transformer", "fmt": "ar",  "backbone": "mbert", "name": "mBERT Dual (AR)"},
    "mbert_qar":           {"type": "transformer", "fmt": "qar", "backbone": "mbert", "name": "mBERT Dual (QAR)"},
    "gte_ar":              {"type": "transformer", "fmt": "ar",  "backbone": "gte",   "name": "GTE Dual (AR)"},
    "gte_qar":             {"type": "transformer", "fmt": "qar", "backbone": "gte",   "name": "GTE Dual (QAR)"},
}

# SentencePiece word-boundary marker (U+2581) as escape to avoid encoding issues
_SPM_MARK = "▁"


# ── Data ──────────────────────────────────────────────────────────────────

def load_splits():
    splits = {}
    for name in ["train", "val", "test"]:
        p = os.path.join(PROCESSED_DIR, f"{name}_kcc.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} -- run: python run_all.py --stage prepare")
        splits[name] = pd.read_csv(p)
    return splits["train"], splits["val"], splits["test"]


def select_indices(test_df, n, seed=42):
    """Stratified sample of n indices across all 5 score labels."""
    rng = np.random.default_rng(seed)
    indices = []
    per_label = max(1, n // 5)
    for label in range(5):
        pool = np.where(test_df["score_label"].values == label)[0]
        chosen = rng.choice(pool, size=min(per_label, len(pool)), replace=False)
        indices.extend(chosen.tolist())
    return sorted(set(indices))[:n]


def input_text_for(row, fmt, suffix):
    """Build the text the model receives as input_a (answer or Q+A)."""
    ans = str(row[f"Answer{suffix}"])
    if fmt == "qar":
        qst = str(row.get(f"Question{suffix}", row.get("Question", "")))
        return qst + " " + ans
    return ans


def make_sample(row, idx, input_text, ref, question, char_saliency,
                pred_label, pred_score, uncertainty, model_window,
                model_key, has_saliency, fmt):
    spec = MODEL_SPECS[model_key]
    return {
        "idx":            int(idx),
        "student_answer": input_text,
        "reference":      ref,
        "question":       question,
        "subject":        str(row.get("Subject", "")),
        "true_label":     int(row["score_label"]),
        "pred_label":     int(pred_label),
        "true_score":     float(row["normalized_score"]),
        "pred_score":     float(pred_score),
        "correct":        int(row["score_label"]) == int(pred_label),
        "uncertainty":    float(uncertainty),
        "char_saliency":  char_saliency,
        "model_window":   int(model_window),
        "model_name":     spec["name"],
        "model_key":      model_key,
        "input_format":   fmt,
        "has_saliency":   has_saliency,
    }


def _normalize_sal(raw_sal):
    s_min, s_max = raw_sal.min(), raw_sal.max()
    if s_max > s_min:
        return (raw_sal - s_min) / (s_max - s_min)
    return np.zeros_like(raw_sal)


def _log(idx, s):
    tag = "OK" if s["correct"] else "--"
    print(f"    idx={idx:<4} label={s['true_label']} pred={s['pred_label']} "
          f"unc={s['uncertainty']:.4f}  {tag}")


# ── Classical baselines ────────────────────────────────────────────────

def generate_baseline(model_key, test_df, train_df, suffix, indices):
    spec = MODEL_SPECS[model_key]
    fmt  = spec["fmt"]
    print(f"  Fitting {spec['baseline']} ...", end=" ", flush=True)
    try:
        tr_d = KhmerSAGTextDataset(train_df, input_format=fmt)
        te_d = KhmerSAGTextDataset(test_df,  input_format=fmt)
        mdl  = create_baseline(spec["baseline"])
        mdl.fit(tr_d)
        preds = mdl.predict(te_d)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return None
    print("done")

    samples = []
    for idx in indices:
        row  = test_df.iloc[idx]
        ans  = str(row[f"Answer{suffix}"])
        ref  = str(row[f"Reference{suffix}"])
        qst  = str(row.get("Question", row.get(f"Question{suffix}", "")))
        inp  = input_text_for(row, fmt, suffix)
        chars = list(ans)[: BILSTM_CFG.max_seq_len - 2]
        char_saliency = [[ch, 0.0] for ch in chars]
        samples.append(make_sample(
            row, idx, inp, ref, qst, char_saliency,
            int(preds["labels"][idx]), float(preds["scores"][idx]), 0.0, len(chars),
            model_key, has_saliency=False, fmt=fmt,
        ))
        _log(idx, samples[-1])
    return samples


# ── BiLSTM ─────────────────────────────────────────────────────────────

def _tok_encode(tok, text, max_len, device):
    ids, mask = tok.encode(text, max_len=max_len)
    return (torch.tensor([ids],  dtype=torch.long, device=device),
            torch.tensor([mask], dtype=torch.long, device=device))


def generate_bilstm(model_key, test_df, train_df, suffix, indices):
    spec = MODEL_SPECS[model_key]
    fmt  = spec["fmt"]
    cfg  = BILSTM_CFG
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"bilstm_{fmt}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  SKIP -- checkpoint not found: {ckpt_path}")
        return None

    tok = CharTokenizer(cfg.max_vocab).fit(
        train_df[f"Answer{suffix}"].tolist() + train_df[f"Reference{suffix}"].tolist()
    )
    model = BiLSTMAttention(
        vocab_size=tok.vocab_size, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, dropout=cfg.dropout, num_classes=5, loss_type="corn",
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).train(False)
    print(f"  Loaded: {ckpt_path}")

    samples = []
    for idx in indices:
        row = test_df.iloc[idx]
        ans = str(row[f"Answer{suffix}"])
        ref = str(row[f"Reference{suffix}"])
        qst = str(row.get("Question", row.get(f"Question{suffix}", "")))
        inp = input_text_for(row, fmt, suffix)

        ia, ma = _tok_encode(tok, inp, cfg.max_seq_len, DEVICE)
        ir, mr = _tok_encode(tok, ref, cfg.max_seq_len, DEVICE)

        # Gradient saliency: norm of embedding gradient per character position
        model.train(False)
        emb = model.emb(ia).detach().requires_grad_(True)
        logits = model.forward_from_emb(emb, ma, ir, mr)
        logits[0, 0].backward()
        sal = emb.grad[0].norm(dim=-1).detach().cpu().numpy()

        chars = list(str(inp))
        n_ch  = min(len(chars), cfg.max_seq_len - 2)
        char_saliency = [[ch, float(s)]
                         for ch, s in zip(chars[:n_ch], _normalize_sal(sal[1:n_ch + 1]))]

        # MC Dropout via BiLSTM's built-in (uses train(False) internally)
        with torch.no_grad():
            res = model.predict_with_uncertainty(ia, ma, ir, mr, T=10)

        samples.append(make_sample(
            row, idx, inp, ref, qst, char_saliency,
            int(res["mode_label"].item()), float(res["mean_score"].item()),
            float(res["std_score"].item()), n_ch,
            model_key, has_saliency=True, fmt=fmt,
        ))
        _log(idx, samples[-1])
    return samples


# ── Transformers ──────────────────────────────────────────────────────

def _clean_tok(tok_str):
    """Strip SentencePiece and WordPiece tokenizer markers for display."""
    t = tok_str.replace(_SPM_MARK, " ").replace("##", "").strip()
    return t if t else " "


def _mc_dropout(model, ids_a, mask_a, ids_r, mask_r, T=10):
    """Inline MC Dropout for DualEncoder -- avoids calling .train(True).eval()."""
    model.train(True)
    labels_mc, scores_mc = [], []
    with torch.no_grad():
        for _ in range(T):
            lg = model(ids_a, mask_a, ids_r, mask_r)
            labels_mc.append(int(corn_logits_to_label(lg).item()))
            scores_mc.append(float(corn_logits_to_score(lg).item()))
    model.train(False)
    pred_label  = int(max(set(labels_mc), key=labels_mc.count))
    pred_score  = float(np.mean(scores_mc))
    uncertainty = float(np.std(scores_mc))
    return pred_label, pred_score, uncertainty


def generate_transformer(model_key, test_df, suffix, indices):
    from transformers import AutoTokenizer
    spec      = MODEL_SPECS[model_key]
    fmt       = spec["fmt"]
    backbone  = spec["backbone"]
    bpath     = TRANSFORMER_BACKBONES[backbone]
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{backbone}_dual_{fmt}_s42_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  SKIP -- checkpoint not found: {ckpt_path}")
        return None

    tokenizer = AutoTokenizer.from_pretrained(bpath, trust_remote_code=True)
    model = create_model(backbone, "dual", TRAIN_CFG.loss_type,
                         dropout=TRAIN_CFG.dropout, freeze_layers=TRAIN_CFG.freeze_layers)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).train(False)
    print(f"  Loaded: {ckpt_path}")

    max_len = TRAIN_CFG.max_seq_len
    samples = []
    for idx in indices:
        row = test_df.iloc[idx]
        ans = str(row[f"Answer{suffix}"])
        ref = str(row[f"Reference{suffix}"])
        qst = str(row.get("Question", row.get(f"Question{suffix}", "")))
        inp = input_text_for(row, fmt, suffix)

        enc_a = tokenizer(inp, max_length=max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
        enc_r = tokenizer(ref, max_length=max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
        ids_a  = enc_a["input_ids"].to(DEVICE)
        mask_a = enc_a["attention_mask"].to(DEVICE)
        ids_r  = enc_r["input_ids"].to(DEVICE)
        mask_r = enc_r["attention_mask"].to(DEVICE)

        # Gradient saliency over answer token embeddings
        model.train(False)
        emb     = model.encoder.embeddings(ids_a).detach().requires_grad_(True)
        enc_out = model.encoder(inputs_embeds=emb, attention_mask=mask_a)
        e_a     = model._pool(enc_out.last_hidden_state, mask_a)
        e_r     = model.encode(ids_r, mask_r)
        logits  = model.head(torch.cat([e_a, e_r, e_a - e_r, e_a * e_r], dim=1))
        logits[0, 0].backward()
        sal = emb.grad[0].norm(dim=-1).detach().cpu().numpy()

        # Positions 1..n_real are real tokens (excl. CLS at 0 and SEP at end)
        n_real  = max(0, min(int(mask_a.sum()) - 2, max_len - 2))
        tok_ids = ids_a.squeeze(0)[1 : n_real + 1].cpu().tolist()
        toks    = tokenizer.convert_ids_to_tokens(tok_ids)
        char_saliency = [[_clean_tok(t), float(s)]
                         for t, s in zip(toks, _normalize_sal(sal[1 : n_real + 1]))]

        pred_label, pred_score, uncertainty = _mc_dropout(
            model, ids_a, mask_a, ids_r, mask_r, T=10)

        samples.append(make_sample(
            row, idx, inp, ref, qst, char_saliency,
            pred_label, pred_score, uncertainty, n_real,
            model_key, has_saliency=True, fmt=fmt,
        ))
        _log(idx, samples[-1])
    return samples


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=SHAP_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_SPECS.keys()),
        metavar="KEY",
        help=f"Model keys to generate. All by default.",
    )
    args = parser.parse_args()

    print("KhmerXScore -- Generate XAI data (all models)")
    print("=" * 60)

    train_df, _, test_df = load_splits()
    suffix = "_proc" if "Answer_proc" in train_df.columns else ""
    indices = select_indices(test_df, args.n_samples, args.seed)
    print(f"  test_n={len(test_df)}  selected={len(indices)}  "
          f"(stratified, seed={args.seed})")

    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            print(f"\n[WARN] Unknown model key: {model_key!r}  -- skipping")
            continue

        spec = MODEL_SPECS[model_key]
        print(f"\n-- {model_key}  [{spec['name']}] --")

        m_type = spec["type"]
        if m_type == "bilstm":
            samples = generate_bilstm(model_key, test_df, train_df, suffix, indices)
        elif m_type == "transformer":
            samples = generate_transformer(model_key, test_df, suffix, indices)
        else:
            samples = generate_baseline(model_key, test_df, train_df, suffix, indices)

        if samples is None:
            continue

        out_path = os.path.join(RESULTS_DIR, f"xai_{model_key}_data.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        acc = sum(1 for s in samples if s["correct"]) / len(samples)
        print(f"  Saved {len(samples)} samples  -->  {out_path}")
        print(f"  Accuracy: {acc:.1%}  |  Mean uncertainty: "
              f"{np.mean([s['uncertainty'] for s in samples]):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
