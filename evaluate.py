"""Evaluation metrics for the simple pipeline.

Two metric families are reported per run:

Ordinal (5-class normalized rubric, 0..4):
    qwk                — Quadratic Weighted Kappa (primary research metric)
    accuracy           — exact-match on the 5-class label
    adjacent_accuracy  — within +/-1 on the 5-class label
    mae                — mean abs error on the 0..4 label scale

Raw (per-question integer score 0..Max Score):
    raw_exact     — exact-match on the integer raw teacher score
    raw_within1   — within +/-1 raw point
    raw_mae       — mean abs raw-point error (in points)
    pct_mae       — mean abs error as percentage of the per-question max score

Inputs:
    pred_scores  — continuous normalized prediction in [0,1], shape (N,)
    true_labels  — integer 5-class label, shape (N,)
    max_scores   — integer max score per question, shape (N,) [optional]
    true_raw     — integer raw teacher score, shape (N,)       [optional]

If max_scores+true_raw are omitted, only the 5-class metrics are returned.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error


def metrics(pred_scores, true_labels, max_scores=None, true_raw=None) -> dict:
    pred_scores = np.asarray(pred_scores, dtype=np.float64).clip(0.0, 1.0)
    true_labels = np.asarray(true_labels, dtype=np.int64).clip(0, 4)
    pred_labels = np.round(pred_scores * 4.0).astype(np.int64).clip(0, 4)

    qwk = cohen_kappa_score(true_labels, pred_labels, weights="quadratic",
                            labels=[0, 1, 2, 3, 4])
    acc = accuracy_score(true_labels, pred_labels)
    adj_acc = float(np.mean(np.abs(pred_labels - true_labels) <= 1))
    mae = float(mean_absolute_error(true_labels.astype(np.float64),
                                    pred_labels.astype(np.float64)))

    out = {
        "qwk": float(qwk),
        "accuracy": float(acc),
        "adjacent_accuracy": adj_acc,
        "mae": mae,
    }

    if max_scores is not None and true_raw is not None:
        max_scores = np.asarray(max_scores, dtype=np.float64)
        true_raw   = np.asarray(true_raw,   dtype=np.int64)
        pred_raw = np.round(pred_scores * max_scores).astype(np.int64)
        pred_raw = np.minimum(pred_raw, max_scores.astype(np.int64))
        pred_raw = np.maximum(pred_raw, 0)
        raw_err = np.abs(pred_raw - true_raw)

        out["raw_exact"]   = float((raw_err == 0).mean())
        out["raw_within1"] = float((raw_err <= 1).mean())
        out["raw_mae"]     = float(raw_err.mean())
        out["pct_mae"]     = float((raw_err / np.maximum(max_scores, 1)).mean() * 100.0)

    return out
