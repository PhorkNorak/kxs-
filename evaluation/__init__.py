"""
KhmerXScore Evaluation Module
================================
- Primary: QWK (deployment threshold >= 0.70)
- Secondary: RMSE, Pearson correlation
- Diagnostic: per-score F1, confusion matrix, per-subject breakdown
- Engineering: latency (mean, P50, P95, P99)
- Statistical: bootstrap 95% CIs, paired bootstrap significance test
- XAI: comprehensiveness, sufficiency (ERASER: DeYoung et al. 2020)

References:
    Williamson et al. (2012) - QWK threshold
    Koehn (2004), Berg-Kirkpatrick et al. (2012) - paired bootstrap
    DeYoung et al. (2020) - ERASER faithfulness metrics
    Gal & Ghahramani (2016) - MC Dropout uncertainty
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    cohen_kappa_score, mean_squared_error, mean_absolute_error,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)


# ============================================================
# Core Metrics
# ============================================================
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa (primary metric)."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_all_metrics(y_true_labels: np.ndarray, y_pred_labels: np.ndarray,
                        y_true_scores: np.ndarray, y_pred_scores: np.ndarray,
                        num_classes: int = 5) -> Dict:
    """
    Compute the full metric suite.
    
    Args:
        y_true_labels: Ground truth discrete labels (0-4)
        y_pred_labels: Predicted discrete labels (0-4)
        y_true_scores: Ground truth continuous scores (0-1)
        y_pred_scores: Predicted continuous scores (0-1)
    """
    metrics = {}
    
    # Primary
    metrics["qwk"] = quadratic_weighted_kappa(y_true_labels, y_pred_labels)
    
    # Secondary - correlation
    if len(np.unique(y_pred_scores)) > 1:
        metrics["pearson"], metrics["pearson_p"] = pearsonr(y_true_scores, y_pred_scores)
        metrics["spearman"], metrics["spearman_p"] = spearmanr(y_true_scores, y_pred_scores)
    else:
        metrics["pearson"] = metrics["spearman"] = 0.0
        metrics["pearson_p"] = metrics["spearman_p"] = 1.0
    
    # Secondary - error
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_scores, y_pred_scores)))
    metrics["mae"] = float(mean_absolute_error(y_true_scores, y_pred_scores))
    
    # Diagnostic - classification
    metrics["accuracy"] = float(np.mean(y_true_labels == y_pred_labels))
    metrics["exact_agreement"] = metrics["accuracy"]
    
    # Adjacent agreement: within ±1
    metrics["adjacent_agreement"] = float(
        np.mean(np.abs(y_true_labels - y_pred_labels) <= 1)
    )
    
    # Weighted F1/Precision/Recall
    metrics["f1_weighted"] = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    metrics["precision_weighted"] = precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(y_true_labels, y_pred_labels, average=None,
                            labels=list(range(num_classes)), zero_division=0)
    for i, f in enumerate(per_class_f1):
        metrics[f"f1_class_{i}"] = float(f)
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(
        y_true_labels, y_pred_labels, labels=list(range(num_classes))
    ).tolist()
    
    return metrics


# ============================================================
# Per-Subject Breakdown
# ============================================================
def compute_per_subject_metrics(df, pred_labels: np.ndarray, pred_scores: np.ndarray,
                                min_n: int = 30) -> Dict:
    """Compute QWK per subject. Only report for subjects with n >= min_n."""
    results = {}
    
    for subject in df["Subject"].unique():
        mask = df["Subject"].values == subject
        n = mask.sum()
        
        if n < min_n:
            results[subject] = {"n": n, "qwk": None, "note": f"n={n} < {min_n}, unreliable"}
            continue
        
        true_labels = df["score_label"].values[mask]
        p_labels = pred_labels[mask]
        true_scores = df["normalized_score"].values[mask]
        p_scores = pred_scores[mask]
        
        results[subject] = {
            "n": int(n),
            "qwk": quadratic_weighted_kappa(true_labels, p_labels),
            "rmse": float(np.sqrt(mean_squared_error(true_scores, p_scores))),
        }
    
    return results


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Returns:
        (point_estimate, lower_bound, upper_bound)
    
    Reference: Efron & Tibshirani (1993)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
            boot_values.append(val)
        except Exception:
            continue
    
    boot_values = np.array(boot_values)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_values, alpha * 100)
    upper = np.percentile(boot_values, (1 - alpha) * 100)
    
    return float(point), float(lower), float(upper)


def bootstrap_qwk_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """Bootstrap 95% CI specifically for QWK."""
    return bootstrap_ci(y_true, y_pred, quadratic_weighted_kappa, n_bootstrap)


# ============================================================
# Paired Bootstrap Significance Test
# ============================================================
def paired_bootstrap_test(y_true: np.ndarray,
                          y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                          metric_fn, n_bootstrap: int = 1000,
                          seed: int = 42) -> Tuple[float, float]:
    """
    Paired bootstrap significance test.
    Tests H0: system A and system B perform equally.
    
    Returns:
        (observed_diff, p_value)
    
    Reference:
        Koehn (2004). "Statistical Significance Tests for MT Evaluation."
        Berg-Kirkpatrick et al. (2012). "An Empirical Investigation of 
        Statistical Significance in NLP."
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    # Observed difference
    score_a = metric_fn(y_true, y_pred_a)
    score_b = metric_fn(y_true, y_pred_b)
    observed_diff = score_a - score_b
    
    # Bootstrap
    count_better = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_a = metric_fn(y_true[idx], y_pred_a[idx])
            boot_b = metric_fn(y_true[idx], y_pred_b[idx])
            if boot_a - boot_b > 0:
                count_better += 1
        except Exception:
            continue
    
    # Two-tailed p-value
    p_value = 2 * min(count_better / n_bootstrap, 1 - count_better / n_bootstrap)
    
    return float(observed_diff), float(p_value)


# ============================================================
# Selective Prediction (Abstention)
# ============================================================
def selective_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                                  uncertainties: np.ndarray,
                                  thresholds: Optional[List[float]] = None) -> List[Dict]:
    """
    Analyze selective prediction: at various uncertainty thresholds,
    what fraction of predictions are deferred and what's the QWK on kept ones?
    
    Reference:
        Li, Zhang, Jin et al. (2023). "Learning When to Defer to Humans 
        for Short Answer Grading." AIED.
    """
    if thresholds is None:
        thresholds = np.percentile(uncertainties, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    results = []
    for thresh in thresholds:
        keep_mask = uncertainties <= thresh
        n_kept = keep_mask.sum()
        n_deferred = (~keep_mask).sum()
        coverage = n_kept / len(y_true)
        
        if n_kept < 2:
            qwk_kept = None
        else:
            qwk_kept = quadratic_weighted_kappa(y_true[keep_mask], y_pred[keep_mask])
        
        results.append({
            "threshold": float(thresh),
            "coverage": float(coverage),
            "n_kept": int(n_kept),
            "n_deferred": int(n_deferred),
            "qwk_kept": qwk_kept,
        })
    
    return results


# ============================================================
# Latency Benchmarking
# ============================================================
def benchmark_latency(model, dataloader, device: str = "cuda",
                      n_warmup: int = 5) -> Dict:
    """Measure inference latency statistics."""
    import torch
    
    model.eval()
    model.to(device)
    
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i < n_warmup:
                continue
            
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_r = batch["input_ids_r"].to(device)
            attention_mask_r = batch["attention_mask_r"].to(device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(input_ids_a, attention_mask_a, input_ids_r, attention_mask_r)
            
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            batch_time = (end - start) * 1000  # ms
            per_sample = batch_time / len(input_ids_a)
            times.append(per_sample)
    
    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_per_sec": float(1000 / np.mean(times)) if np.mean(times) > 0 else 0,
    }


# ============================================================
# XAI Faithfulness: Comprehensiveness & Sufficiency
# ============================================================
def compute_comprehensiveness(model, batch, attributions: np.ndarray,
                              top_k: int = 10, device: str = "cuda") -> float:
    """
    Comprehensiveness (DeYoung et al. 2020):
    Erase top-k attributed tokens → score should drop significantly.
    Higher = more faithful.
    """
    import torch
    
    model.eval()
    input_ids_a = batch["input_ids_a"].to(device)
    attention_mask_a = batch["attention_mask_a"].to(device)
    input_ids_r = batch["input_ids_r"].to(device)
    attention_mask_r = batch["attention_mask_r"].to(device)
    
    with torch.no_grad():
        # Original prediction
        orig_logits = model(input_ids_a, attention_mask_a, input_ids_r, attention_mask_r)
        
        # Mask top-k tokens
        masked_ids = input_ids_a.clone()
        for i in range(len(attributions)):
            if len(attributions[i]) == 0:
                continue
            top_indices = np.argsort(attributions[i])[-top_k:]
            for idx in top_indices:
                if idx < masked_ids.shape[1]:
                    masked_ids[i, idx] = 0  # Mask token
        
        # Prediction without top-k tokens
        masked_logits = model(masked_ids, attention_mask_a, input_ids_r, attention_mask_r)
    
    # Comprehensiveness = change in prediction confidence
    from models.losses import corn_logits_to_score
    orig_scores = corn_logits_to_score(orig_logits).cpu().numpy()
    masked_scores = corn_logits_to_score(masked_logits).cpu().numpy()
    
    return float(np.mean(np.abs(orig_scores - masked_scores)))


def compute_sufficiency(model, batch, attributions: np.ndarray,
                        top_k: int = 10, device: str = "cuda") -> float:
    """
    Sufficiency (DeYoung et al. 2020):
    Keep ONLY top-k attributed tokens → score should remain similar.
    Lower = more faithful.
    """
    import torch
    
    model.eval()
    input_ids_a = batch["input_ids_a"].to(device)
    attention_mask_a = batch["attention_mask_a"].to(device)
    input_ids_r = batch["input_ids_r"].to(device)
    attention_mask_r = batch["attention_mask_r"].to(device)
    
    with torch.no_grad():
        orig_logits = model(input_ids_a, attention_mask_a, input_ids_r, attention_mask_r)
        
        # Keep only top-k tokens, mask everything else
        kept_ids = torch.zeros_like(input_ids_a)
        for i in range(len(attributions)):
            if len(attributions[i]) == 0:
                continue
            top_indices = np.argsort(attributions[i])[-top_k:]
            for idx in top_indices:
                if idx < kept_ids.shape[1]:
                    kept_ids[i, idx] = input_ids_a[i, idx]
        
        suff_logits = model(kept_ids, attention_mask_a, input_ids_r, attention_mask_r)
    
    from models.losses import corn_logits_to_score
    orig_scores = corn_logits_to_score(orig_logits).cpu().numpy()
    suff_scores = corn_logits_to_score(suff_logits).cpu().numpy()
    
    return float(np.mean(np.abs(orig_scores - suff_scores)))


# ============================================================
# Results formatting
# ============================================================
def format_results(metrics: Dict, model_name: str) -> str:
    """Format metrics as a readable string."""
    lines = [f"\n{'='*60}", f"Results: {model_name}", f"{'='*60}"]
    
    # Primary
    qwk = metrics.get("qwk", 0)
    passed = "PASS" if qwk >= 0.70 else "FAIL"
    lines.append(f"  QWK:              {qwk:.4f}  [{passed} >= 0.70 threshold]")
    
    # Secondary
    lines.append(f"  RMSE:             {metrics.get('rmse', 0):.4f}")
    lines.append(f"  Pearson:          {metrics.get('pearson', 0):.4f}")
    lines.append(f"  Spearman:         {metrics.get('spearman', 0):.4f}")
    
    # Diagnostic
    lines.append(f"  Exact agreement:  {metrics.get('exact_agreement', 0):.4f}")
    lines.append(f"  Adjacent agr.:    {metrics.get('adjacent_agreement', 0):.4f}")
    lines.append(f"  F1 (weighted):    {metrics.get('f1_weighted', 0):.4f}")
    
    # Per-class
    for i in range(5):
        key = f"f1_class_{i}"
        if key in metrics:
            lines.append(f"  F1 class {i}:       {metrics[key]:.4f}")
    
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)
