"""Per-run analysis: per-Max-Score breakdown of raw-score performance.

Reads `results_<run_name>/runs/<run_id>/predictions_test.csv` and produces:
  - Overall metrics (5-class + raw)
  - Per-Max-Score breakdown table
  - Top-K worst-error rows (for manual inspection)
  - Writes the breakdown to `results_<run_name>/runs/<run_id>/per_max_score.csv`

Usage:
    python analyze_run.py <results_dir> <run_id>
    python analyze_run.py results_no10c_no0 segment_qar_tfidf_svr
    python analyze_run.py results_no10c     clean_ra_tfidf_svr      --worst 15
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def overall_metrics(df: pd.DataFrame) -> dict:
    out = {}
    out["n"] = len(df)
    # 5-class
    out["acc_5class"]   = float((df["pred_label"] == df["true_label"]).mean())
    out["adj_acc_5class"] = float((np.abs(df["pred_label"] - df["true_label"]) <= 1).mean())
    out["mae_5class"]   = float(np.abs(df["pred_label"] - df["true_label"]).mean())
    # Raw-score
    raw_err = df["raw_abs_error"]
    out["raw_exact"]    = float((raw_err == 0).mean())
    out["raw_within1"]  = float((raw_err <= 1).mean())
    out["raw_within2"]  = float((raw_err <= 2).mean())
    out["raw_mae_pt"]   = float(raw_err.mean())
    out["pct_mae"]      = float((raw_err / df["Max Score"]).mean() * 100.0)
    return out


def per_max_score_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for max_s, g in df.groupby("Max Score"):
        err = g["raw_abs_error"]
        rows.append({
            "Max Score": int(max_s),
            "n": len(g),
            "exact":    (err == 0).mean(),
            "within_1": (err <= 1).mean(),
            "within_2": (err <= 2).mean(),
            "mae_pt":   err.mean(),
            "pct_mae":  (err / g["Max Score"]).mean() * 100.0,
        })
    return pd.DataFrame(rows).sort_values("Max Score").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="e.g. results_no10c or results_no10c_no0")
    ap.add_argument("run_id", help="e.g. clean_ra_tfidf_svr")
    ap.add_argument("--worst", type=int, default=10,
                    help="Number of worst-error rows to show")
    ap.add_argument("--save", action="store_true",
                    help="Also write the breakdown to per_max_score.csv in the run dir")
    args = ap.parse_args()

    pred_path = os.path.join(args.results_dir, "runs", args.run_id, "predictions_test.csv")
    if not os.path.exists(pred_path):
        sys.exit(f"[!] not found: {pred_path}")
    df = pd.read_csv(pred_path)

    needed = {"Max Score", "true_raw", "pred_raw", "raw_abs_error", "true_label", "pred_label"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"[!] predictions CSV missing columns {missing}.  "
                 "Re-run training with the updated train.py to regenerate.")

    print(f"\n=== {args.results_dir} / {args.run_id} ===\n")

    om = overall_metrics(df)
    print(f"Test samples: {om['n']}")
    print(f"\n5-class ordinal:")
    print(f"  accuracy           = {om['acc_5class']:.4f}  ({int(om['acc_5class']*om['n'])}/{om['n']})")
    print(f"  adjacent accuracy  = {om['adj_acc_5class']:.4f}")
    print(f"  MAE (label scale)  = {om['mae_5class']:.4f}")
    print(f"\nRaw-score (per-question integer 0..MaxScore):")
    print(f"  exact match        = {om['raw_exact']:.4f}  ({int(om['raw_exact']*om['n'])}/{om['n']})")
    print(f"  within +/-1 point  = {om['raw_within1']:.4f}")
    print(f"  within +/-2 points = {om['raw_within2']:.4f}")
    print(f"  MAE (points)       = {om['raw_mae_pt']:.3f}")
    print(f"  MAE (% of max)     = {om['pct_mae']:.2f}%")

    print(f"\n=== Per Max-Score breakdown ===")
    pms = per_max_score_table(df)
    print(pms.to_string(index=False, formatters={
        "exact":    lambda v: f"{v:.4f}",
        "within_1": lambda v: f"{v:.4f}",
        "within_2": lambda v: f"{v:.4f}",
        "mae_pt":   lambda v: f"{v:.3f}",
        "pct_mae":  lambda v: f"{v:.2f}%",
    }))

    if args.save:
        out_path = os.path.join(args.results_dir, "runs", args.run_id, "per_max_score.csv")
        pms.to_csv(out_path, index=False)
        print(f"\n[*] breakdown saved -> {out_path}")

    print(f"\n=== Worst {args.worst} cases (highest raw_abs_error) ===")
    worst_cols = ["Max Score", "true_raw", "pred_raw", "raw_abs_error",
                  "true_label", "pred_label", "pred_score", "true_score"]
    worst = df.nlargest(args.worst, "raw_abs_error")[worst_cols]
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
