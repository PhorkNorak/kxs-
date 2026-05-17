"""Compare baseline (v01) + all enhancement experiments (v02..v07).

Loads every results_<ds>_v0X/leaderboard.csv, stacks them, sorts by test_qwk,
and prints a top-K table per dataset. Also computes per-experiment averages
to show which enhancement moved the needle.

Usage:
    python experiments/compare_all.py
    python experiments/compare_all.py --topk 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import DATASETS  # noqa: E402

EXPERIMENT_DIRS = {
    "v01_baseline":          "results_{ds}",
    "v02_calibrated":        "results_{ds}_v02_calibrated",
    "v03_maxfeat":           "results_{ds}_v03_maxfeat",
    "v03b_maxfeat_neural":   "results_{ds}_v03b_maxfeat_neural",
    "v04_bucket":            "results_{ds}_v04_bucket",
    "v05_bilstm":            "results_{ds}_v05_bilstm",
    "v06_transformer":       "results_{ds}_v06_transformer",
    "v07_ensemble":          "results_{ds}_v07_ensemble",
}

NUMERIC_COLS = [
    "train_qwk", "train_accuracy", "train_raw_exact", "train_raw_within1",
    "test_qwk",  "test_accuracy",  "test_raw_exact",  "test_raw_within1",
    "test_raw_mae", "val_qwk",
]


def load_all(ds: str) -> pd.DataFrame:
    rows = []
    for exp, pat in EXPERIMENT_DIRS.items():
        path = os.path.join(_ROOT, pat.format(ds=ds), "leaderboard.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["experiment"] = exp
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def summarize(df: pd.DataFrame, dataset_label: str, top_k: int = 10):
    if df.empty:
        print(f"\n=== {dataset_label}: NO DATA ===")
        return

    print(f"\n=== {dataset_label} ===")
    print(f"Total cells across all experiments: {len(df)}")

    # 1. Top-K by test QWK
    cols = ["experiment", "run_id", "train_qwk", "test_qwk",
            "test_accuracy", "test_raw_exact", "test_raw_within1",
            "test_raw_mae"]
    top = df.sort_values("test_qwk", ascending=False).head(top_k)
    print(f"\nTop {top_k} cells by test_qwk:")
    print(top[cols].to_string(index=False))

    # 2. Per-experiment summary: best cell per experiment
    print("\nBest cell per experiment (by test_qwk):")
    best_per = df.sort_values("test_qwk", ascending=False).groupby(
        "experiment", as_index=False).first()
    best_per = best_per.sort_values("test_qwk", ascending=False)
    print(best_per[cols].to_string(index=False))

    # 3. Per-experiment mean (just supervised cells, excludes cosine-only)
    print("\nPer-experiment mean test_qwk (all cells):")
    means = df.groupby("experiment")["test_qwk"].agg(["mean", "max", "count"]).round(4)
    means = means.sort_values("max", ascending=False)
    print(means.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    for ds in DATASETS:
        df = load_all(ds["run_name"])
        summarize(df, ds["label"], top_k=args.topk)


if __name__ == "__main__":
    main()
