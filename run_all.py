"""Grid orchestrator.

Runs the 3 × 2 × 10 = 60-cell grid (or a single cell with `--only <run_id>`),
appends each result to `results/leaderboard.csv`, and finally invokes
XAI on the best transformer cell.

Examples:
    python run_all.py
    python run_all.py --only segment_ra_tfidf_svr
    python run_all.py --only raw_ra_dual_gte --epochs 2
    python run_all.py --skip-xai
"""

import argparse
import csv
import os
import sys
import time
import traceback

import config as C
from data import load_dataframe, split_dataframe
from train import (
    train_classical,
    train_bilstm,
    train_transformer,
)


LEADERBOARD_HEADER = [
    "run_id", "preprocess", "input", "model", "family",
    # ===== TRAIN-set metrics (for overfit-gap reporting, like Alaoui et al. 2024) =====
    "train_qwk", "train_accuracy", "train_adjacent_accuracy", "train_mae",
    "train_raw_exact", "train_raw_within1", "train_raw_mae", "train_pct_mae",
    # ===== TEST-set metrics (primary deployment numbers) =====
    "test_qwk", "test_accuracy", "test_adjacent_accuracy", "test_mae",
    "test_raw_exact", "test_raw_within1", "test_raw_mae", "test_pct_mae",
    "val_qwk", "best_epoch", "seconds",
]


def grid_iter():
    """Yield (run_id, preprocess, input, model_id, family, backbone) tuples."""
    for prep in C.PREPROC_MODES:
        for inp in C.INPUT_FORMATS:
            for model_id, family, backbone in C.MODELS:
                run_id = f"{prep}_{inp}_{model_id}"
                yield run_id, prep, inp, model_id, family, backbone


def run_one(prep, inp, model_id, family, backbone, run_id,
            train_df, val_df, test_df, max_epochs):
    if family == "classical":
        return train_classical(model_id, prep, inp, train_df, val_df, test_df, run_id)
    if family == "bilstm":
        return train_bilstm(prep, inp, train_df, val_df, test_df, run_id, max_epochs)
    if family in ("dual", "cross"):
        return train_transformer(
            family, backbone, prep, inp, train_df, val_df, test_df, run_id, max_epochs
        )
    raise ValueError(family)


def append_row(row: dict):
    new_file = not os.path.exists(C.LEADERBOARD)
    with open(C.LEADERBOARD, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="Run a single run_id")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override max epochs (for smoke tests)")
    ap.add_argument("--skip-xai", action="store_true",
                    help="Don't run XAI on the best transformer after the grid")
    args = ap.parse_args()

    print("[*] loading data")
    df = load_dataframe()
    train_df, val_df, test_df = split_dataframe(df)
    print(f"    train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    grid = list(grid_iter())
    if args.only:
        grid = [g for g in grid if g[0] == args.only]
        if not grid:
            print(f"[!] no grid cell matches run_id={args.only!r}")
            print(f"    available examples: {[g[0] for g in grid_iter()][:5]} ...")
            sys.exit(1)

    print(f"[*] {len(grid)} cell(s) to run")
    for i, (run_id, prep, inp, model_id, family, backbone) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] ===== {run_id} =====")
        t0 = time.time()
        try:
            result = run_one(prep, inp, model_id, family, backbone, run_id,
                             train_df, val_df, test_df, args.epochs)
            dt = time.time() - t0
            test_m = result["test"]
            val_m = result["val"]
            train_m = result.get("train", {})
            row = {
                "run_id": run_id,
                "preprocess": prep,
                "input": inp,
                "model": model_id,
                "family": family,
                "train_qwk": f"{train_m.get('qwk', float('nan')):.4f}",
                "train_accuracy": f"{train_m.get('accuracy', float('nan')):.4f}",
                "train_adjacent_accuracy": f"{train_m.get('adjacent_accuracy', float('nan')):.4f}",
                "train_mae": f"{train_m.get('mae', float('nan')):.4f}",
                "train_raw_exact":   f"{train_m.get('raw_exact', float('nan')):.4f}",
                "train_raw_within1": f"{train_m.get('raw_within1', float('nan')):.4f}",
                "train_raw_mae":     f"{train_m.get('raw_mae', float('nan')):.4f}",
                "train_pct_mae":     f"{train_m.get('pct_mae', float('nan')):.4f}",
                "test_qwk": f"{test_m['qwk']:.4f}",
                "test_accuracy": f"{test_m['accuracy']:.4f}",
                "test_adjacent_accuracy": f"{test_m['adjacent_accuracy']:.4f}",
                "test_mae": f"{test_m['mae']:.4f}",
                "test_raw_exact":   f"{test_m.get('raw_exact', float('nan')):.4f}",
                "test_raw_within1": f"{test_m.get('raw_within1', float('nan')):.4f}",
                "test_raw_mae":     f"{test_m.get('raw_mae', float('nan')):.4f}",
                "test_pct_mae":     f"{test_m.get('pct_mae', float('nan')):.4f}",
                "val_qwk": f"{val_m['qwk']:.4f}",
                "best_epoch": result.get("best_epoch", ""),
                "seconds": f"{dt:.1f}",
            }
            append_row(row)
            print(f"    DONE  train_qwk={train_m.get('qwk', 0):.4f} -> test_qwk={test_m['qwk']:.4f}  "
                  f"gap={train_m.get('qwk', 0)-test_m['qwk']:+.4f}  "
                  f"test_raw_w1={test_m.get('raw_within1', 0):.4f}  "
                  f"({dt:.1f}s)")
        except Exception as e:
            traceback.print_exc()
            print(f"[!] {run_id} failed: {e}")
            append_row({
                "run_id": run_id, "preprocess": prep, "input": inp,
                "model": model_id, "family": family,
                "train_qwk": "ERR", "train_accuracy": "", "train_adjacent_accuracy": "",
                "train_mae": "",
                "train_raw_exact": "", "train_raw_within1": "",
                "train_raw_mae": "", "train_pct_mae": "",
                "test_qwk": "ERR", "test_accuracy": "", "test_adjacent_accuracy": "",
                "test_mae": "",
                "test_raw_exact": "", "test_raw_within1": "",
                "test_raw_mae": "", "test_pct_mae": "",
                "val_qwk": "", "best_epoch": "",
                "seconds": f"{time.time() - t0:.1f}",
            })

    # XAI on best transformer
    if not args.skip_xai and not args.only:
        try:
            from xai import run_xai_on_best
            run_xai_on_best()
        except Exception as e:
            print(f"[!] XAI step failed: {e}")
            traceback.print_exc()

    print(f"\n[*] leaderboard written to {C.LEADERBOARD}")


if __name__ == "__main__":
    main()
