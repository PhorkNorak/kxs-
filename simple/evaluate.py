"""Evaluation metrics for the simple pipeline.

Inputs are always continuous predicted scores in [0,1] and integer true labels
in {0,1,2,3,4}. We rescale predictions to [0,4] and round to integers for
classification-style metrics; we keep them continuous for correlation/MAE/RMSE.
"""

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)
from scipy.stats import pearsonr, spearmanr


def metrics(pred_scores: np.ndarray, true_labels: np.ndarray) -> dict:
    pred_scores = np.asarray(pred_scores, dtype=np.float64).clip(0.0, 1.0)
    true_labels = np.asarray(true_labels, dtype=np.int64).clip(0, 4)
    pred_continuous = pred_scores * 4.0
    pred_labels = np.round(pred_continuous).astype(np.int64).clip(0, 4)
    true_continuous = true_labels.astype(np.float64)

    qwk = cohen_kappa_score(true_labels, pred_labels, weights="quadratic")
    acc = accuracy_score(true_labels, pred_labels)
    adj_acc = float(np.mean(np.abs(pred_labels - true_labels) <= 1))
    rmse = float(np.sqrt(mean_squared_error(true_continuous, pred_continuous)))
    mae = float(mean_absolute_error(true_continuous, pred_continuous))
    try:
        pearson = float(pearsonr(pred_continuous, true_continuous)[0])
    except Exception:
        pearson = float("nan")
    try:
        spearman = float(spearmanr(pred_continuous, true_continuous)[0])
    except Exception:
        spearman = float("nan")
    try:
        f1w = float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0))
    except Exception:
        f1w = float("nan")

    return {
        "qwk": float(qwk),
        "accuracy": float(acc),
        "adjacent_accuracy": adj_acc,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "f1_weighted": f1w,
    }
