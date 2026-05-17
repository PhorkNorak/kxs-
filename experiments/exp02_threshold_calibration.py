"""exp02 — Post-hoc threshold calibration on val.

The baseline rounds `pred_score * 4` at fixed boundaries {0.5, 1.5, 2.5, 3.5}
in label space (== {0.125, 0.375, 0.625, 0.875} in [0,1]). Those are
class-frequency-agnostic. We can lift QWK by sliding the four cut points to
the values that maximize val QWK.

Method: coordinate descent on 4 cut points in (0, 1), starting from the
default {0.125, 0.375, 0.625, 0.875}, sweeping each cut over 0.01-step
candidates while keeping the others fixed and the order constraint
t1 < t2 < t3 < t4. Runs 5 passes (usually converges in 2).

Raw-score metrics are unchanged (they depend on continuous pred_score *
max_score, not on label binning), but we still recompute and store them so
the leaderboard schema stays uniform.

Reads:   results_<ds>/runs/<cell>/predictions_{train,val,test}.csv
Writes:  results_<ds>_v02_calibrated/runs/<cell>/{config,metrics,predictions_*}.csv
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Bootstrap project root + experiments dir onto sys.path so we work whether
# called via `python experiments/expNN.py` or `python -m experiments.expNN`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import (  # noqa: E402
    DATASETS, select_datasets, patch_config, reset_leaderboard, append_row,
    row_from_metrics, banner, add_datasets_flag,
)
from evaluate import metrics as evaluate_metrics
from sklearn.metrics import cohen_kappa_score


DEFAULT_CUTS = [0.125, 0.375, 0.625, 0.875]


def _label_from_cuts(scores: np.ndarray, cuts) -> np.ndarray:
    """Bin continuous [0,1] scores into {0..4} using 4 ascending cuts."""
    bins = np.array(cuts, dtype=np.float64)
    return np.digitize(scores, bins).astype(np.int64).clip(0, 4)


def _eval_cuts(scores, true_labels, cuts) -> float:
    pred = _label_from_cuts(scores, cuts)
    return cohen_kappa_score(true_labels, pred, weights="quadratic",
                             labels=[0, 1, 2, 3, 4])


def calibrate(val_scores, val_labels, n_passes: int = 5,
              step: float = 0.01) -> list:
    """Coordinate-descent search for 4 ordered cut points in (0,1)."""
    cuts = list(DEFAULT_CUTS)
    candidates = np.arange(step, 1.0, step)
    for _ in range(n_passes):
        improved = False
        for i in range(4):
            best_qwk = _eval_cuts(val_scores, val_labels, cuts)
            best_c   = cuts[i]
            lo = cuts[i - 1] + step if i > 0 else step
            hi = cuts[i + 1] - step if i < 3 else 1.0 - step
            for c in candidates:
                if c < lo or c > hi:
                    continue
                trial = cuts.copy()
                trial[i] = float(c)
                q = _eval_cuts(val_scores, val_labels, trial)
                if q > best_qwk + 1e-9:
                    best_qwk = q
                    best_c = float(c)
                    improved = True
            cuts[i] = best_c
        if not improved:
            break
    return cuts


def _apply_to_predictions_csv(src_path: str, dst_path: str, cuts) -> None:
    """Recompute pred_label and abs_error using new cuts; copy raw-score cols."""
    df = pd.read_csv(src_path)
    new_pred_label = _label_from_cuts(df["pred_score"].to_numpy(), cuts)
    df["pred_label"] = new_pred_label
    df["abs_error"] = np.abs(df["pred_label"] - df["true_label"])
    # raw_abs_error / pred_raw are unaffected — they come from continuous score
    df.to_csv(dst_path, index=False, encoding="utf-8-sig")


def _metrics_from_predictions(df: pd.DataFrame) -> dict:
    return evaluate_metrics(
        pred_scores=df["pred_score"].to_numpy(),
        true_labels=df["true_label"].to_numpy(),
        max_scores=df["Max Score"].to_numpy(),
        true_raw=df["true_raw"].to_numpy(),
    )


def _metrics_from_predictions_with_cuts(df: pd.DataFrame, cuts) -> dict:
    """Recompute the 5-class metrics using calibrated cuts; raw metrics unchanged."""
    scores = df["pred_score"].to_numpy()
    pred_labels = _label_from_cuts(scores, cuts)
    true_labels = df["true_label"].to_numpy()

    qwk = cohen_kappa_score(true_labels, pred_labels,
                            weights="quadratic", labels=[0, 1, 2, 3, 4])
    acc = float((pred_labels == true_labels).mean())
    adj = float((np.abs(pred_labels - true_labels) <= 1).mean())
    mae = float(np.abs(pred_labels - true_labels).mean())

    # raw-score metrics are independent of cuts, take from the predictions CSV
    raw_err = np.abs(df["pred_raw"].to_numpy() - df["true_raw"].to_numpy())
    max_s   = df["Max Score"].to_numpy().astype(np.float64)
    raw_exact   = float((raw_err == 0).mean())
    raw_within1 = float((raw_err <= 1).mean())
    raw_mae     = float(raw_err.mean())
    pct_mae     = float((raw_err / np.maximum(max_s, 1)).mean() * 100.0)

    return {
        "qwk": float(qwk), "accuracy": acc,
        "adjacent_accuracy": adj, "mae": mae,
        "raw_exact": raw_exact, "raw_within1": raw_within1,
        "raw_mae": raw_mae, "pct_mae": pct_mae,
    }


def calibrate_cell(src_run_dir: str, dst_run_dir: str) -> dict:
    """Process one cell from baseline -> calibrated."""
    os.makedirs(dst_run_dir, exist_ok=True)

    val_df  = pd.read_csv(os.path.join(src_run_dir, "predictions_val.csv"))
    train_df = pd.read_csv(os.path.join(src_run_dir, "predictions_train.csv"))
    test_df = pd.read_csv(os.path.join(src_run_dir, "predictions_test.csv"))

    cuts = calibrate(val_df["pred_score"].to_numpy(),
                     val_df["true_label"].to_numpy())

    # Write calibrated predictions
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        new = df.copy()
        new["pred_label"] = _label_from_cuts(new["pred_score"].to_numpy(), cuts)
        new["abs_error"]  = np.abs(new["pred_label"] - new["true_label"])
        new.to_csv(os.path.join(dst_run_dir, f"predictions_{split}.csv"),
                   index=False, encoding="utf-8-sig")

    train_m = _metrics_from_predictions_with_cuts(train_df, cuts)
    val_m   = _metrics_from_predictions_with_cuts(val_df,   cuts)
    test_m  = _metrics_from_predictions_with_cuts(test_df,  cuts)

    src_cfg = os.path.join(src_run_dir, "config.json")
    cfg = json.load(open(src_cfg, encoding="utf-8")) if os.path.exists(src_cfg) else {}
    cfg["experiment"] = "v02_threshold_calibration"
    cfg["calibrated_cuts"] = cuts
    json.dump(cfg, open(os.path.join(dst_run_dir, "config.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

    json.dump({"train": train_m, "val": val_m, "test": test_m,
               "best_epoch": None, "calibrated_cuts": cuts},
              open(os.path.join(dst_run_dir, "metrics.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)

    return {"train": train_m, "val": val_m, "test": test_m,
            "cuts": cuts, "config": cfg}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="",
                    help="Experiment suffix to calibrate (e.g. 'v05_bilstm'). "
                         "Empty -> v01 baseline.")
    ap.add_argument("--dest", default="v02_calibrated",
                    help="Output suffix. Default 'v02_calibrated'.")
    add_datasets_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        # Source: any v0x directory (or the v01 baseline if --source is empty)
        src_name = f"results_{ds['run_name']}" if not args.source \
                   else f"results_{ds['run_name']}_{args.source}"
        src_results = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            src_name,
        )
        if not os.path.exists(src_results):
            print(f"[!] missing source {src_results} — skipping {ds['run_name']}")
            continue

        # Destination: a fresh calibrated directory
        dst_suffix = args.dest if args.source == "" else f"{args.source}_calibrated"
        dst_dir = patch_config(ds["run_name"], ds["drop_zero"],
                               exp_suffix=dst_suffix)
        banner(f"exp02 calibrate  source={args.source or 'v01'}  {ds['label']}", dst_dir)

        reset_leaderboard()

        # Iterate over cells present in the source
        src_runs_dir = os.path.join(src_results, "runs")
        cells = sorted(os.listdir(src_runs_dir))
        for cell in cells:
            src_cell = os.path.join(src_runs_dir, cell)
            dst_cell = os.path.join(dst_dir, "runs", cell)
            t0 = time.time()
            try:
                out = calibrate_cell(src_cell, dst_cell)
                cfg = out["config"]
                row = row_from_metrics(
                    run_id=cell,
                    prep=cfg.get("preprocess", ""),
                    inp=cfg.get("input", ""),
                    model_id=cfg.get("model", ""),
                    family=cfg.get("family", ""),
                    train_m=out["train"], val_m=out["val"], test_m=out["test"],
                    best_epoch=None,
                    seconds=time.time() - t0,
                )
                append_row(row)
                print(f"  [{cell:30s}] cuts={[round(c,3) for c in out['cuts']]}  "
                      f"test_qwk={out['test']['qwk']:.4f}  "
                      f"acc5={out['test']['accuracy']:.4f}")
            except Exception as e:
                print(f"  [{cell:30s}] FAILED: {e}")

    print("\n[*] exp02 done")


if __name__ == "__main__":
    main()
