"""
Evaluation Module — metrics, bootstrap CIs, significance tests, XAI faithfulness.

References:
    Williamson et al. (2012) — QWK threshold
    Berg-Kirkpatrick et al. (2012) — paired bootstrap
    DeYoung et al. (2020) — ERASER comprehensiveness/sufficiency
    Gal & Ghahramani (2016) — MC Dropout
    Li et al. (2023) — Selective prediction for ASAG
"""

import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (cohen_kappa_score, mean_squared_error,
                              mean_absolute_error, f1_score, confusion_matrix)


def quadratic_weighted_kappa(y_true, y_pred):
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_all_metrics(y_true_l, y_pred_l, y_true_s, y_pred_s, num_classes=5):
    m = {}
    m["qwk"] = quadratic_weighted_kappa(y_true_l, y_pred_l)
    m["rmse"] = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
    m["mae"] = float(mean_absolute_error(y_true_s, y_pred_s))
    m["accuracy"] = float(np.mean(y_true_l == y_pred_l))
    m["exact_agreement"] = m["accuracy"]
    m["adjacent_agreement"] = float(np.mean(np.abs(y_true_l - y_pred_l) <= 1))
    if len(np.unique(y_pred_s)) > 1:
        m["pearson"], _ = pearsonr(y_true_s, y_pred_s)
        m["spearman"], _ = spearmanr(y_true_s, y_pred_s)
        m["pearson"], m["spearman"] = float(m["pearson"]), float(m["spearman"])
    else:
        m["pearson"] = m["spearman"] = 0.0
    m["f1_weighted"] = float(f1_score(y_true_l, y_pred_l, average="weighted", zero_division=0))
    per = f1_score(y_true_l, y_pred_l, average=None, labels=list(range(num_classes)), zero_division=0)
    for i, v in enumerate(per):
        m[f"f1_class_{i}"] = float(v)
    m["confusion_matrix"] = confusion_matrix(y_true_l, y_pred_l, labels=list(range(num_classes))).tolist()
    return m


def bootstrap_qwk_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    pt = quadratic_weighted_kappa(y_true, y_pred)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            vals.append(cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic"))
        except Exception:
            pass
    v = np.array(vals)
    return float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_bootstrap_test(y_true, pa, pb, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    obs = quadratic_weighted_kappa(y_true, pa) - quadratic_weighted_kappa(y_true, pb)
    cnt = 0
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            if (cohen_kappa_score(y_true[idx], pa[idx], weights="quadratic") >
                cohen_kappa_score(y_true[idx], pb[idx], weights="quadratic")):
                cnt += 1
        except Exception:
            pass
    return float(obs), float(2 * min(cnt / n, 1 - cnt / n))


def selective_prediction_analysis(y_true, y_pred, uncertainties, percentiles=None):
    if percentiles is None:
        percentiles = [90, 80, 70, 60, 50]
    results = []
    for pct in percentiles:
        thresh = np.percentile(uncertainties, pct)
        keep = uncertainties <= thresh
        n_kept, n_def = int(keep.sum()), int((~keep).sum())
        cov = n_kept / len(y_true) * 100
        try:
            qk = float(cohen_kappa_score(y_true[keep], y_pred[keep], weights="quadratic"))
        except Exception:
            qk = 0.0
        results.append({"pct": pct, "threshold": float(thresh), "coverage": cov,
                        "n_kept": n_kept, "n_deferred": n_def, "qwk_kept": qk})
    return results


def compute_per_subject(test_df, pred_labels, pred_scores, min_n=30):
    results = {}
    for subj in test_df["Subject"].unique():
        mask = test_df["Subject"].values == subj
        n = int(mask.sum())
        if n < min_n:
            results[subj] = {"n": n, "qwk": None, "note": f"n<{min_n}"}
            continue
        try:
            q = float(cohen_kappa_score(test_df["score_label"].values[mask],
                                        pred_labels[mask], weights="quadratic"))
        except Exception:
            q = None
        r = float(np.sqrt(mean_squared_error(test_df["normalized_score"].values[mask],
                                              pred_scores[mask])))
        results[subj] = {"n": n, "qwk": q, "rmse": r}
    return results


def compute_comprehensiveness(model, batch, attributions, top_k=10, device="cpu"):
    import torch
    model.eval()
    ia = batch["input_ids_a"].to(device)
    ma = batch["attention_mask_a"].to(device)
    ir = batch["input_ids_r"].to(device)
    mr = batch["attention_mask_r"].to(device)
    with torch.no_grad():
        from models.losses import corn_logits_to_score
        orig = corn_logits_to_score(model(ia, ma, ir, mr)).cpu().numpy()
        masked = ia.clone()
        for i, attr in enumerate(attributions):
            if len(attr) == 0:
                continue
            top_idx = np.argsort(attr)[-top_k:]
            for idx in top_idx:
                if idx < masked.shape[1]:
                    masked[i, idx] = 0
        erased = corn_logits_to_score(model(masked, ma, ir, mr)).cpu().numpy()
    return float(np.mean(np.abs(orig - erased)))


def compute_sufficiency(model, batch, attributions, top_k=10, device="cpu"):
    import torch
    model.eval()
    ia = batch["input_ids_a"].to(device)
    ma = batch["attention_mask_a"].to(device)
    ir = batch["input_ids_r"].to(device)
    mr = batch["attention_mask_r"].to(device)
    with torch.no_grad():
        from models.losses import corn_logits_to_score
        orig = corn_logits_to_score(model(ia, ma, ir, mr)).cpu().numpy()
        kept = torch.zeros_like(ia)
        for i, attr in enumerate(attributions):
            if len(attr) == 0:
                continue
            top_idx = np.argsort(attr)[-top_k:]
            for idx in top_idx:
                if idx < kept.shape[1]:
                    kept[i, idx] = ia[i, idx]
        suff = corn_logits_to_score(model(kept, ma, ir, mr)).cpu().numpy()
    return float(np.mean(np.abs(orig - suff)))


def benchmark_latency(model, dataloader, device="cuda", n_warmup=5):
    import torch
    model.eval().to(device)
    times = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i < n_warmup:
                continue
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_r = batch["input_ids_r"].to(device)
            mask_r = batch["attention_mask_r"].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids_a, mask_a, ids_r, mask_r)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000 / len(ids_a))
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)), "p99_ms": float(np.percentile(t, 99))}


def format_results(metrics, name):
    q = metrics.get("qwk", 0)
    flag = "PASS ✓" if q >= 0.70 else "FAIL ✗"
    return (f"  {name:<45s} QWK={q:.4f} [{flag}]  RMSE={metrics.get('rmse',0):.4f}  "
            f"Pearson={metrics.get('pearson',0):.4f}  F1w={metrics.get('f1_weighted',0):.4f}")
