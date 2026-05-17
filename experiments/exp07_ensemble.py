"""exp07 — Ensemble of the top-K cells across v01..v06 by val QWK.

Strategy:
  1. Collect every cell from every results_<ds>_v0X directory (and the v01
     baseline at results_<ds>/).
  2. Rank by val_qwk (using val to avoid test leakage in selection).
  3. Take the top-K (default 3).
  4. Read their `predictions_{train,val,test}.csv`.
  5. Compute a softmax-of-val-qwk-weighted mean of `pred_score` over rows.
     (For test we need identical row indices across cells — all cells share
      the same dataset split, so row order matches.)
  6. Recompute metrics with the ensemble pred_score and write to a new run dir.

Output:
  results_<ds>_v07_ensemble/runs/ensemble_topK/{config, metrics, predictions_*}.csv
  results_<ds>_v07_ensemble/leaderboard.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from glob import glob

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import (  # noqa: E402
    DATASETS, select_datasets, patch_config, reset_leaderboard, append_row,
    row_from_metrics, banner, add_datasets_flag,
)
from evaluate import metrics as evaluate_metrics  # noqa: E402
from exp02_threshold_calibration import calibrate, _label_from_cuts  # noqa: E402


TOP_K = 3

# v02_calibrated reuses the same pred_score as v01 (only relabels), so we
# *exclude it from candidate pools* to keep score-level diversity. We apply
# threshold calibration AFTER ensembling instead. The benefit is two-fold:
#  - selection diversity (different model families / preprocessing)
#  - bin boundaries optimized for the ensemble's score distribution
EXPERIMENT_DIRS = [
    "",                          # v01 baseline (results_<ds>/)
    "v03_maxfeat",
    "v03b_maxfeat_neural",
    "v04_bucket",
    "v05_bilstm",
    "v06_transformer",
]


def collect_cells(ds_run_name: str) -> pd.DataFrame:
    """Return a DataFrame: one row per (experiment, cell) with val_qwk and paths."""
    rows = []
    for exp in EXPERIMENT_DIRS:
        results_dir_name = (
            f"results_{ds_run_name}" if not exp
            else f"results_{ds_run_name}_{exp}"
        )
        leaderboard = os.path.join(
            os.path.dirname(_HERE), results_dir_name, "leaderboard.csv"
        )
        if not os.path.exists(leaderboard):
            continue
        df = pd.read_csv(leaderboard)
        df["experiment"] = exp or "v01_baseline"
        df["leaderboard_path"] = leaderboard
        df["run_dir"] = df["run_id"].apply(
            lambda r: os.path.join(os.path.dirname(_HERE), results_dir_name, "runs", r)
        )
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["val_qwk"]  = pd.to_numeric(out["val_qwk"], errors="coerce")
    out["test_qwk"] = pd.to_numeric(out["test_qwk"], errors="coerce")
    return out.dropna(subset=["val_qwk"])


def load_predictions(run_dir: str):
    pieces = {}
    for split in ("train", "val", "test"):
        path = os.path.join(run_dir, f"predictions_{split}.csv")
        if os.path.exists(path):
            pieces[split] = pd.read_csv(path)
    return pieces


def softmax(vals, temperature: float = 0.05):
    a = np.asarray(vals, dtype=np.float64) / temperature
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def ensemble_cells(top_rows: pd.DataFrame, top_k: int) -> dict:
    """Produce ensembled predictions and metrics for train/val/test splits."""
    chosen = top_rows.sort_values("val_qwk", ascending=False).head(top_k)
    print(f"  ensembling {len(chosen)} cells:")
    for _, r in chosen.iterrows():
        print(f"    - {r['experiment']:25s} {r['run_id']:40s} "
              f"val_qwk={r['val_qwk']:.4f}  test_qwk={r['test_qwk']:.4f}")

    weights = softmax(chosen["val_qwk"].tolist())
    components = []
    for _, r in chosen.iterrows():
        p = load_predictions(r["run_dir"])
        if {"train", "val", "test"} - set(p):
            print(f"  [!] {r['run_id']} missing splits — skipping")
            continue
        components.append(p)

    # Combine on test (and val/train too for consistency)
    def combine(split):
        # Use the first component as the row-template (idx, true, max_score…)
        base = components[0][split].copy()
        score_stack = np.zeros(len(base))
        for w, c in zip(weights, components):
            sc = c[split]["pred_score"].to_numpy()
            if len(sc) != len(score_stack):
                raise ValueError(f"row count mismatch on {split}: "
                                 f"{len(sc)} vs {len(score_stack)}")
            score_stack = score_stack + w * sc
        base["pred_score"] = np.clip(score_stack, 0.0, 1.0)
        base["pred_label"] = np.round(base["pred_score"] * 4).clip(0, 4).astype(int)
        base["pred_raw"]   = np.minimum(
            np.round(base["pred_score"] * base["Max Score"]).astype(int),
            base["Max Score"].astype(int),
        ).clip(0)
        base["abs_error"]     = np.abs(base["pred_label"] - base["true_label"])
        base["raw_abs_error"] = np.abs(base["pred_raw"]   - base["true_raw"])
        return base

    splits = {s: combine(s) for s in ("train", "val", "test")}

    # Calibrate the ensembled pred_score on val (same algorithm as exp02).
    cuts = calibrate(splits["val"]["pred_score"].to_numpy(),
                     splits["val"]["true_label"].to_numpy())
    # Re-apply labels using calibrated cuts
    for split in splits:
        sc = splits[split]["pred_score"].to_numpy()
        splits[split]["pred_label"] = _label_from_cuts(sc, cuts)
        splits[split]["abs_error"]  = np.abs(
            splits[split]["pred_label"] - splits[split]["true_label"]
        )

    def metrics_of(df):
        # Recompute 5-class metrics from the re-labeled predictions
        pred_label = df["pred_label"].to_numpy()
        true_label = df["true_label"].to_numpy()
        from sklearn.metrics import cohen_kappa_score
        qwk = cohen_kappa_score(true_label, pred_label,
                                weights="quadratic", labels=[0, 1, 2, 3, 4])
        acc = float((pred_label == true_label).mean())
        adj = float((np.abs(pred_label - true_label) <= 1).mean())
        mae = float(np.abs(pred_label - true_label).mean())

        raw_err = np.abs(df["pred_raw"].to_numpy() - df["true_raw"].to_numpy())
        max_s = df["Max Score"].to_numpy().astype(np.float64)
        return {
            "qwk": float(qwk), "accuracy": acc,
            "adjacent_accuracy": adj, "mae": mae,
            "raw_exact":   float((raw_err == 0).mean()),
            "raw_within1": float((raw_err <= 1).mean()),
            "raw_mae":     float(raw_err.mean()),
            "pct_mae":     float((raw_err / np.maximum(max_s, 1)).mean() * 100.0),
        }

    return {
        "chosen": chosen.to_dict(orient="records"),
        "weights": weights.tolist(),
        "calibrated_cuts": cuts,
        "splits": splits,
        "train": metrics_of(splits["train"]),
        "val":   metrics_of(splits["val"]),
        "test":  metrics_of(splits["test"]),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_datasets_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        dst = patch_config(ds["run_name"], ds["drop_zero"],
                           exp_suffix="v07_ensemble")
        banner(f"exp07 ensemble top-{TOP_K}  {ds['label']}", dst)

        reset_leaderboard()
        cells_df = collect_cells(ds["run_name"])
        if cells_df.empty:
            print(f"  [!] no upstream cells found for {ds['run_name']} — skipping")
            continue
        print(f"  found {len(cells_df)} candidate cells across "
              f"{cells_df['experiment'].nunique()} experiments")

        t0 = time.time()
        try:
            out = ensemble_cells(cells_df, TOP_K)
        except Exception as e:
            print(f"  [!] ensemble failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # Write the ensemble cell
        run_id = f"ensemble_top{TOP_K}"
        run_dir = os.path.join(dst, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        for split, df in out["splits"].items():
            df.to_csv(os.path.join(run_dir, f"predictions_{split}.csv"),
                      index=False, encoding="utf-8-sig")
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id, "model": "ensemble",
                "family": "ensemble_v07",
                "preprocess": "mixed", "input": "mixed",
                "experiment": "v07_ensemble", "top_k": TOP_K,
                "weights": out["weights"],
                "calibrated_cuts": out["calibrated_cuts"],
                "chosen": [{"experiment": c["experiment"], "run_id": c["run_id"],
                            "val_qwk": c["val_qwk"], "test_qwk": c["test_qwk"]}
                           for c in out["chosen"]],
            }, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"train": out["train"], "val": out["val"], "test": out["test"],
                       "best_epoch": None}, f, indent=2, ensure_ascii=False)

        row = row_from_metrics(
            run_id=run_id, prep="mixed", inp="mixed",
            model_id="ensemble_top3", family="ensemble_v07",
            train_m=out["train"], val_m=out["val"], test_m=out["test"],
            best_epoch=None, seconds=time.time() - t0,
        )
        append_row(row)
        print(f"  ensemble test_qwk={out['test']['qwk']:.4f}  "
              f"acc5={out['test']['accuracy']:.4f}  "
              f"raw_w1={out['test']['raw_within1']:.4f}")

    print("\n[*] exp07 done")


if __name__ == "__main__":
    main()
