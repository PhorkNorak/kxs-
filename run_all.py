"""
KhmerXScore Experiment Runner
================================
Usage:
    python run_all.py --stage prepare
    python run_all.py --stage baselines
    python run_all.py --stage bilstm
    python run_all.py --stage transformers
    python run_all.py --stage ablations
    python run_all.py --stage xai
    python run_all.py --stage summary
    python run_all.py --stage all
"""

import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from copy import deepcopy
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from preprocessing import preprocess
from data import (load_raw_data, split_data, preprocess_dataframe,
                  KhmerSAGTextDataset, get_dataloaders)
from models.baselines import create_baseline, BiLSTMAttention
from models.dual_encoder import create_model
from models.char_tokenizer import CharTokenizer
from models.losses import corn_logits_to_label, corn_logits_to_score
from train import train_transformer, evaluate_model, evaluate_with_uncertainty
from evaluation import (compute_all_metrics, bootstrap_qwk_ci, paired_bootstrap_test,
                         selective_prediction_analysis, compute_per_subject,
                         format_results, compute_comprehensiveness, compute_sufficiency)


# ── Helpers ──────────────────────────────────────────────────
def save_results(results, filename):
    path = os.path.join(RESULTS_DIR, filename)
    def _c(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_c, ensure_ascii=False)
    print(f"  → Saved: {path}")


def load_splits(mode="kcc"):
    r = {}
    for n in ["train", "val", "test"]:
        p = os.path.join(PROCESSED_DIR, f"{n}_{mode}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} — run --stage prepare first.")
        r[n] = pd.read_csv(p)
    return r["train"], r["val"], r["test"]


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def collect_preds(model, loader, loss_type):
    model.eval()
    tl, pl, ts, ps = [], [], [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids_a"].to(DEVICE), b["attention_mask_a"].to(DEVICE),
                          b["input_ids_r"].to(DEVICE), b["attention_mask_r"].to(DEVICE))
            if loss_type == "corn":
                pl.extend(corn_logits_to_label(logits).cpu().numpy())
                ps.extend(corn_logits_to_score(logits).cpu().numpy())
            else:
                _s = logits.squeeze(-1).clamp(0, 1).cpu().numpy()
                ps.extend(_s); pl.extend(np.round(_s * 4).astype(np.int64).clip(0, 4))
            tl.extend(b["label"].numpy()); ts.extend(b["score"].numpy())
    return np.array(tl), np.array(pl), np.array(ts), np.array(ps)


def free(model):
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ── STAGE 1: Prepare ────────────────────────────────────────
def stage_prepare():
    print("\n" + "=" * 60 + "\nSTAGE 1: Data Preparation\n" + "=" * 60)
    df = load_raw_data(RAW_CSV)
    print(f"  Samples: {len(df)}")
    print(f"  Score dist: {dict(df['score_label'].value_counts().sort_index())}")
    train_df, val_df, test_df = split_data(df, seed=SPLIT_SEED)
    for mode in PREPROC_MODES:
        for nm, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            p = preprocess_dataframe(sdf, mode)
            p.to_csv(os.path.join(PROCESSED_DIR, f"{nm}_{mode}.csv"), index=False)
        print(f"  [{mode}] saved")
    print("Stage 1 done.\n")


# ── STAGE 2: Baselines ──────────────────────────────────────
def stage_baselines():
    print("\n" + "=" * 60 + "\nSTAGE 2: Classical Baselines\n" + "=" * 60)
    tr, va, te = load_splits("kcc")
    results = {}
    for fmt in INPUT_FORMATS:
        tr_d = KhmerSAGTextDataset(tr, input_format=fmt)
        te_d = KhmerSAGTextDataset(te, input_format=fmt)
        for name in ["mean_predictor", "tfidf_cosine", "tfidf_svr", "fasttext_cosine"]:
            key = f"{name}_{fmt}"
            print(f"  {key} ...", end=" ", flush=True)
            t0 = time.time()
            mdl = create_baseline(name); mdl.fit(tr_d); preds = mdl.predict(te_d)
            m = compute_all_metrics(te_d.labels, preds["labels"], te_d.scores, preds["scores"])
            qp, ql, qh = bootstrap_qwk_ci(te_d.labels, preds["labels"], BOOTSTRAP_N)
            m["qwk_ci_95"] = [ql, qh]; m["elapsed_sec"] = round(time.time() - t0, 2)
            m["per_subject"] = compute_per_subject(te, preds["labels"], preds["scores"])
            results[key] = m
            print(format_results(m, key))
    save_results(results, "baselines.json")


# ── STAGE 3: BiLSTM ─────────────────────────────────────────
def stage_bilstm():
    print("\n" + "=" * 60 + "\nSTAGE 3: BiLSTM + Attention\n" + "=" * 60)
    tr, va, te = load_splits("kcc")
    cfg = BILSTM_CFG
    results = {}
    for fmt in INPUT_FORMATS:
        key = f"bilstm_{fmt}"
        print(f"\n  {key}")
        # Build texts
        suffix = "_proc" if "Answer_proc" in tr.columns else ""
        if fmt == "qar":
            tra = (tr[f"Question{suffix}"] + " " + tr[f"Answer{suffix}"]).tolist()
            vaa = (va[f"Question{suffix}"] + " " + va[f"Answer{suffix}"]).tolist()
            tea = (te[f"Question{suffix}"] + " " + te[f"Answer{suffix}"]).tolist()
        else:
            tra, vaa, tea = tr[f"Answer{suffix}"].tolist(), va[f"Answer{suffix}"].tolist(), te[f"Answer{suffix}"].tolist()
        trr = tr[f"Reference{suffix}"].tolist()
        var2 = va[f"Reference{suffix}"].tolist()
        ter = te[f"Reference{suffix}"].tolist()

        tok = CharTokenizer(cfg.max_vocab).fit(tra + trr)
        print(f"    vocab={tok.vocab_size}")

        from data import get_class_balanced_sampler, KhmerSAGDataset
        from torch.utils.data import DataLoader

        # Build datasets using the CharTokenizer (same interface as HF tokenizer)
        tr_ds = KhmerSAGDataset(tr, tok, cfg.max_seq_len, fmt)
        va_ds = KhmerSAGDataset(va, tok, cfg.max_seq_len, fmt)
        te_ds = KhmerSAGDataset(te, tok, cfg.max_seq_len, fmt)
        sampler = get_class_balanced_sampler(tr_ds.labels)

        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        te_ld = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        set_seed(42)
        model = BiLSTMAttention(
            vocab_size=tok.vocab_size, embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            dropout=cfg.dropout, num_classes=5, loss_type="corn",
        )
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    params={npar:,}")

        # Use a simple config object for train_transformer
        from dataclasses import dataclass, field
        @dataclass
        class _Cfg:
            lr: float = cfg.lr
            batch_size: int = cfg.batch_size
            max_epochs: int = cfg.max_epochs
            early_stop_patience: int = cfg.early_stop_patience
            weight_decay: float = 0.01
            warmup_ratio: float = 0.10
            dropout: float = cfg.dropout
            max_seq_len: int = cfg.max_seq_len
            freeze_layers: int = 0
            scl_alpha: float = 1.0
            scl_beta: float = cfg.scl_beta
            scl_temperature: float = 0.07
            scl_margin_scale: float = 1.0
            use_class_balanced: bool = True
            loss_type: str = "corn"
            mc_dropout_T: int = 10

        t0 = time.time()
        tr_res = train_transformer(model, tr_ld, va_ld, _Cfg(), DEVICE, CHECKPOINT_DIR, key)
        tl, pl, ts, ps = collect_preds(model, te_ld, "corn")
        m = compute_all_metrics(tl, pl, ts, ps)
        qp, ql, qh = bootstrap_qwk_ci(tl, pl, BOOTSTRAP_N)
        m.update({"qwk_ci_95": [ql, qh], "best_val_qwk": tr_res["best_val_qwk"],
                  "n_params": npar, "elapsed_sec": round(time.time() - t0, 2),
                  "per_subject": compute_per_subject(te, pl, ps)})
        # MC Dropout + selective prediction (mirrors transformer stage)
        mc, mc_tl, mc_pl, unc = evaluate_with_uncertainty(model, te_ld, "corn", DEVICE, _Cfg().mc_dropout_T)
        m["mc_uncertainty"] = mc["mean_uncertainty"]
        m["selective_prediction"] = selective_prediction_analysis(mc_tl, mc_pl, unc)
        results[key] = m
        print(format_results(m, key))
        free(model)

    save_results(results, "bilstm.json")


# ── STAGE 4: Transformers ───────────────────────────────────
def stage_transformers():
    print("\n" + "=" * 60 + "\nSTAGE 4: Transformer Backbone Study\n" + "=" * 60)
    tr, va, te = load_splits("kcc")
    results = {}
    for bname, bpath in TRANSFORMER_BACKBONES.items():
        tokenizer = AutoTokenizer.from_pretrained(bpath)
        for fmt in INPUT_FORMATS:
            for seed in TRAIN_CFG.seeds:
                run = f"{bname}_dual_{fmt}_s{seed}"
                print(f"\n  ── {run} ──")
                set_seed(seed)
                trl, vl, tel = get_dataloaders(tr, va, te, tokenizer,
                    TRAIN_CFG.batch_size, TRAIN_CFG.max_seq_len, fmt, "kcc",
                    TRAIN_CFG.use_class_balanced, NUM_WORKERS)
                model = create_model(bname, "dual", TRAIN_CFG.loss_type,
                                     dropout=TRAIN_CFG.dropout, freeze_layers=TRAIN_CFG.freeze_layers)
                tr_res = train_transformer(model, trl, vl, TRAIN_CFG, DEVICE, CHECKPOINT_DIR, run)
                tl, pl, ts, ps = collect_preds(model, tel, TRAIN_CFG.loss_type)
                m = compute_all_metrics(tl, pl, ts, ps)
                qp, ql, qh = bootstrap_qwk_ci(tl, pl, BOOTSTRAP_N)
                m.update({"qwk_ci_95": [ql, qh], "best_val_qwk": tr_res["best_val_qwk"],
                          "seed": seed, "backbone": bname, "input_format": fmt,
                          "per_subject": compute_per_subject(te, pl, ps)})
                # MC Dropout + selective prediction
                mc, mc_tl, mc_pl, unc = evaluate_with_uncertainty(model, tel, TRAIN_CFG.loss_type, DEVICE, TRAIN_CFG.mc_dropout_T)
                m["mc_uncertainty"] = mc["mean_uncertainty"]
                m["selective_prediction"] = selective_prediction_analysis(mc_tl, mc_pl, unc)
                results[run] = m
                print(format_results(m, run))
                free(model)
    save_results(results, "transformers.json")


# ── STAGE 5: Ablations ──────────────────────────────────────
def stage_ablations():
    print("\n" + "=" * 60 + "\nSTAGE 5: Ablation Studies\n" + "=" * 60)
    bpath = TRANSFORMER_BACKBONES[PROPOSED_BACKBONE]
    tok = AutoTokenizer.from_pretrained(bpath)
    results = {}

    # A1: Preprocessing
    print("\n  [A1] Preprocessing")
    for mode in PREPROC_MODES:
        tr, va, te = load_splits(mode)
        trl, vl, tel = get_dataloaders(tr, va, te, tok, TRAIN_CFG.batch_size,
            TRAIN_CFG.max_seq_len, "qar", mode, True, NUM_WORKERS)
        model = create_model(PROPOSED_BACKBONE, "dual", "corn",
                             dropout=TRAIN_CFG.dropout, freeze_layers=TRAIN_CFG.freeze_layers)
        set_seed(42)
        train_transformer(model, trl, vl, TRAIN_CFG, DEVICE, CHECKPOINT_DIR, f"abl_pp_{mode}")
        tl, pl, ts, ps = collect_preds(model, tel, "corn")
        m = compute_all_metrics(tl, pl, ts, ps)
        results[f"preproc_{mode}"] = m
        print(f"    {mode:<20} QWK={m['qwk']:.4f}  RMSE={m['rmse']:.4f}")
        free(model)

    # A2: Topology
    print("\n  [A2] Topology")
    tr, va, te = load_splits("kcc")
    for topo in ["dual", "cross"]:
        trl, vl, tel = get_dataloaders(tr, va, te, tok, TRAIN_CFG.batch_size,
            TRAIN_CFG.max_seq_len, "qar", "kcc", True, NUM_WORKERS)
        cfg = deepcopy(TRAIN_CFG)
        if topo == "cross": cfg.scl_beta = 0
        model = create_model(PROPOSED_BACKBONE, topo, "corn",
                             dropout=cfg.dropout, freeze_layers=cfg.freeze_layers)
        set_seed(42)
        train_transformer(model, trl, vl, cfg, DEVICE, CHECKPOINT_DIR, f"abl_topo_{topo}")
        tl, pl, ts, ps = collect_preds(model, tel, "corn")
        results[f"topology_{topo}"] = compute_all_metrics(tl, pl, ts, ps)
        print(f"    {topo:<20} QWK={results[f'topology_{topo}']['qwk']:.4f}")
        free(model)

    # A3: SCL beta
    print("\n  [A3] SCL beta")
    for beta in [0.0, 0.25, 0.5, 1.0]:
        trl, vl, tel = get_dataloaders(tr, va, te, tok, TRAIN_CFG.batch_size,
            TRAIN_CFG.max_seq_len, "qar", "kcc", True, NUM_WORKERS)
        cfg = deepcopy(TRAIN_CFG); cfg.scl_beta = beta
        model = create_model(PROPOSED_BACKBONE, "dual", "corn",
                             dropout=cfg.dropout, freeze_layers=cfg.freeze_layers)
        set_seed(42)
        train_transformer(model, trl, vl, cfg, DEVICE, CHECKPOINT_DIR, f"abl_scl_{beta}")
        tl, pl, ts, ps = collect_preds(model, tel, "corn")
        results[f"scl_beta_{beta}"] = compute_all_metrics(tl, pl, ts, ps)
        print(f"    beta={beta:<17} QWK={results[f'scl_beta_{beta}']['qwk']:.4f}")
        free(model)

    # A4: Loss type
    print("\n  [A4] Loss type")
    for loss in ["mse", "weighted_mse", "corn"]:
        trl, vl, tel = get_dataloaders(tr, va, te, tok, TRAIN_CFG.batch_size,
            TRAIN_CFG.max_seq_len, "qar", "kcc", True, NUM_WORKERS)
        cfg = deepcopy(TRAIN_CFG); cfg.loss_type = loss
        if loss != "corn": cfg.scl_beta = 0
        model = create_model(PROPOSED_BACKBONE, "dual", loss,
                             dropout=cfg.dropout, freeze_layers=cfg.freeze_layers)
        set_seed(42)
        train_transformer(model, trl, vl, cfg, DEVICE, CHECKPOINT_DIR, f"abl_loss_{loss}")
        tl, pl, ts, ps = collect_preds(model, tel, loss)
        results[f"loss_{loss}"] = compute_all_metrics(tl, pl, ts, ps)
        print(f"    {loss:<20} QWK={results[f'loss_{loss}']['qwk']:.4f}")
        free(model)

    # A5: Imbalance handling
    print("\n  [A5] Imbalance")
    for bal in [False, True]:
        trl, vl, tel = get_dataloaders(tr, va, te, tok, TRAIN_CFG.batch_size,
            TRAIN_CFG.max_seq_len, "qar", "kcc", bal, NUM_WORKERS)
        model = create_model(PROPOSED_BACKBONE, "dual", "corn",
                             dropout=TRAIN_CFG.dropout, freeze_layers=TRAIN_CFG.freeze_layers)
        set_seed(42)
        train_transformer(model, trl, vl, TRAIN_CFG, DEVICE, CHECKPOINT_DIR, f"abl_bal_{bal}")
        tl, pl, ts, ps = collect_preds(model, tel, "corn")
        label = "balanced" if bal else "standard"
        results[f"imbalance_{label}"] = compute_all_metrics(tl, pl, ts, ps)
        print(f"    {label:<20} QWK={results[f'imbalance_{label}']['qwk']:.4f}")
        free(model)

    save_results(results, "ablations.json")


# ── STAGE 6: XAI ────────────────────────────────────────────
def stage_xai():
    print("\n" + "=" * 60 + "\nSTAGE 6: XAI — Gradient Saliency + Faithfulness\n" + "=" * 60)
    bpath = TRANSFORMER_BACKBONES[PROPOSED_BACKBONE]
    tr, va, te = load_splits("kcc")
    tok = AutoTokenizer.from_pretrained(bpath)
    _, _, tel = get_dataloaders(tr, va, te, tok, batch_size=1,
        max_len=TRAIN_CFG.max_seq_len, input_format="qar", preproc_mode="kcc",
        use_class_balanced=False, num_workers=0)

    ckpt = os.path.join(CHECKPOINT_DIR, f"{PROPOSED_BACKBONE}_dual_qar_s42_best.pt")
    if not os.path.exists(ckpt):
        print(f"  No checkpoint at {ckpt} — run --stage transformers first.")
        return
    model = create_model(PROPOSED_BACKBONE, "dual", "corn",
                         dropout=TRAIN_CFG.dropout, freeze_layers=TRAIN_CFG.freeze_layers)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True)["model_state_dict"])
    model = model.to(DEVICE).eval()
    print(f"  Loaded: {ckpt}")

    from tqdm import tqdm
    ig_attr, token_lists = [], []

    print(f"  Computing gradient saliency on {SHAP_N_SAMPLES} samples...")
    for i, b in enumerate(tqdm(tel, desc="  Saliency")):
        if i >= SHAP_N_SAMPLES: break
        ids_a, mask_a = b["input_ids_a"].to(DEVICE), b["attention_mask_a"].to(DEVICE)
        ids_r, mask_r = b["input_ids_r"].to(DEVICE), b["attention_mask_r"].to(DEVICE)
        emb = model.encoder.embeddings(ids_a).detach().requires_grad_(True)
        enc = model.encoder(inputs_embeds=emb, attention_mask=mask_a)
        e_a = model._pool(enc.last_hidden_state, mask_a)
        e_r = model.encode(ids_r, mask_r)
        logits = model.head(torch.cat([e_a, e_r, e_a - e_r, e_a * e_r], dim=1))
        logits[0, 0].backward()
        sal = emb.grad[0].norm(dim=-1).detach().cpu().numpy()
        ig_attr.append(sal.tolist())
        token_lists.append(tok.convert_ids_to_tokens(ids_a.squeeze(0).cpu().tolist()))

    print("  Computing faithfulness...")
    comp_k, suff_k = {k: [] for k in FAITHFULNESS_TOP_K}, {k: [] for k in FAITHFULNESS_TOP_K}
    for i, b in enumerate(tel):
        if i >= min(SHAP_N_SAMPLES, len(ig_attr)): break
        attr = np.array(ig_attr[i])
        for k in FAITHFULNESS_TOP_K:
            comp_k[k].append(compute_comprehensiveness(model, b, [attr], k, DEVICE))
            suff_k[k].append(compute_sufficiency(model, b, [attr], k, DEVICE))

    xai = {"ig_attributions": ig_attr, "tokens": token_lists, "n_samples": len(ig_attr)}
    print(f"\n  {'k':>4}  {'Comprehensiveness':>18}  {'Sufficiency':>12}")
    for k in FAITHFULNESS_TOP_K:
        c = float(np.mean(comp_k[k])) if comp_k[k] else 0
        s = float(np.mean(suff_k[k])) if suff_k[k] else 0
        xai[f"comprehensiveness_k{k}"], xai[f"sufficiency_k{k}"] = c, s
        print(f"  {k:>4}  {c:>18.4f}  {s:>12.4f}")

    save_results(xai, "xai.json")


# ── STAGE 7: Summary ────────────────────────────────────────
def stage_summary():
    import re
    print("\n" + "=" * 60 + "\nFINAL RESULTS SUMMARY\n" + "=" * 60)
    rows = []
    seed_groups = {}  # base_key -> list of per-seed metric dicts
    for fn in ["baselines.json", "bilstm.json", "transformers.json", "ablations.json"]:
        fp = os.path.join(RESULTS_DIR, fn)
        if not os.path.exists(fp): continue
        data = json.load(open(fp))
        for key, val in data.items():
            if not (isinstance(val, dict) and "qwk" in val):
                continue
            m = re.match(r"^(.+)_s\d+$", key)
            if m and "seed" in val:
                seed_groups.setdefault(m.group(1), []).append(val)
            else:
                rows.append({"model": key, "qwk": round(float(val["qwk"]), 4),
                              "qwk_std": 0.0,
                              "rmse": round(float(val.get("rmse", 0)), 4),
                              "pearson": round(float(val.get("pearson", 0)), 4),
                              "f1_w": round(float(val.get("f1_weighted", 0)), 4),
                              "pass": float(val["qwk"]) >= DEPLOYMENT_QWK_THRESHOLD,
                              "n_seeds": 1})
    for base, vals in seed_groups.items():
        qwks  = [float(v["qwk"])              for v in vals]
        rmses = [float(v.get("rmse", 0))      for v in vals]
        pears = [float(v.get("pearson", 0))   for v in vals]
        f1s   = [float(v.get("f1_weighted",0)) for v in vals]
        mean_qwk = float(np.mean(qwks))
        rows.append({"model": base, "qwk": round(mean_qwk, 4),
                      "qwk_std": round(float(np.std(qwks)), 4),
                      "rmse": round(float(np.mean(rmses)), 4),
                      "rmse_std": round(float(np.std(rmses)), 4),
                      "pearson": round(float(np.mean(pears)), 4),
                      "f1_w": round(float(np.mean(f1s)), 4),
                      "pass": mean_qwk >= DEPLOYMENT_QWK_THRESHOLD,
                      "n_seeds": len(vals)})
    if not rows:
        print("  No results. Run experiments first."); return
    rows.sort(key=lambda x: x["qwk"], reverse=True)
    print(f"\n  {'Model':<48} {'QWK':>7} {'±std':>6} {'RMSE':>7} {'Pearson':>8} {'F1':>7} {'n':>3} {'≥0.70':>6}")
    print(f"  {'-'*48} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*3} {'-'*6}")
    for r in rows:
        f = "✓" if r["pass"] else "✗"
        std = r.get("qwk_std", 0.0)
        print(f"  {r['model']:<48} {r['qwk']:>7.4f} {std:>6.4f} {r['rmse']:>7.4f} "
              f"{r['pearson']:>8.4f} {r['f1_w']:>7.4f} {r['n_seeds']:>3d} {f:>6}")
    save_results({"ranked": rows}, "summary.json")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KhmerXScore Experiment Runner")
    parser.add_argument("--stage", default="all",
        choices=["prepare", "baselines", "bilstm", "transformers",
                 "ablations", "xai", "summary", "all"])
    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}  |  Backbone: {PROPOSED_BACKBONE}  |  Seeds: {TRAIN_CFG.seeds}")

    order = ["prepare", "baselines", "bilstm", "transformers", "ablations", "xai", "summary"]
    fns = {"prepare": stage_prepare, "baselines": stage_baselines, "bilstm": stage_bilstm,
           "transformers": stage_transformers, "ablations": stage_ablations,
           "xai": stage_xai, "summary": stage_summary}

    if args.stage == "all":
        for s in order:
            fns[s]()
    else:
        fns[args.stage]()
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
