"""Evaluation metrics for the simple pipeline.

We report four metrics, chosen to cover ordinal agreement, exact and lenient
classification accuracy, and average regression error in score units:

    qwk                — Quadratic Weighted Kappa (primary)
    accuracy           — exact-match rate
    adjacent_accuracy  — within +/-1 score point
    mae                — mean absolute error on the 0..4 scale

Inputs are continuous predicted scores in [0,1] and integer true labels in
{0,1,2,3,4}. Predictions are rescaled to [0,4] and rounded for classification
metrics; MAE is computed on the rounded predictions vs. the integer labels.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error


def metrics(pred_scores: np.ndarray, true_labels: np.ndarray) -> dict:
    pred_scores = np.asarray(pred_scores, dtype=np.float64).clip(0.0, 1.0)
    true_labels = np.asarray(true_labels, dtype=np.int64).clip(0, 4)
    pred_labels = np.round(pred_scores * 4.0).astype(np.int64).clip(0, 4)

    qwk = cohen_kappa_score(true_labels, pred_labels, weights="quadratic")
    acc = accuracy_score(true_labels, pred_labels)
    adj_acc = float(np.mean(np.abs(pred_labels - true_labels) <= 1))
    mae = float(mean_absolute_error(true_labels.astype(np.float64),
                                    pred_labels.astype(np.float64)))

    return {
        "qwk": float(qwk),
        "accuracy": float(acc),
        "adjacent_accuracy": adj_acc,
        "mae": mae,
    }
