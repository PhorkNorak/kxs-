"""exp05 — Full BiLSTM+Attention grid on both datasets.

3 preprocess x 2 input formats x 2 datasets = 12 cells.
Uses the existing `train.train_bilstm` (char-level BiLSTM with attention,
MSE on normalized score, early-stop on val QWK).

No network required.
"""

from __future__ import annotations

import os
import sys
import time
import importlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import (  # noqa: E402
    DATASETS, select_datasets, patch_config, reset_leaderboard, append_row,
    row_from_metrics, banner, add_datasets_flag, add_resume_flag,
    cell_already_done, backfill_leaderboard_row_from_metrics,
)

import config  # noqa: E402
import data    # noqa: E402
import train   # noqa: E402


PREPROCESS = ["raw", "clean", "segment"]
INPUTS     = ["ra", "qar"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        dst = patch_config(ds["run_name"], ds["drop_zero"],
                           exp_suffix="v05_bilstm")
        importlib.reload(data)
        importlib.reload(train)
        banner(f"exp05 BiLSTM  {ds['label']}  resume={args.resume}", dst)

        if not args.resume:
            reset_leaderboard()
        df = data.load_dataframe()
        train_df, val_df, test_df = data.split_dataframe(df)
        print(f"  rows={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        for prep in PREPROCESS:
            for inp in INPUTS:
                run_id = f"{prep}_{inp}_bilstm"
                if args.resume and cell_already_done(run_id):
                    backfill_leaderboard_row_from_metrics(
                        run_id, prep, inp, model_id="bilstm", family="bilstm",
                    )
                    print(f"  [{run_id:30s}] SKIP (already done)")
                    continue
                t0 = time.time()
                try:
                    result = train.train_bilstm(
                        prep, inp, train_df, val_df, test_df, run_id,
                    )
                    dt = time.time() - t0
                    trm = result.get("train", {})
                    vm  = result["val"]
                    tm  = result["test"]
                    be  = result.get("best_epoch", "")
                    print(f"  [{run_id:30s}] train_qwk={trm.get('qwk', 0):.4f} -> "
                          f"test_qwk={tm['qwk']:.4f}  "
                          f"acc5={tm['accuracy']:.4f}  "
                          f"raw_w1={tm.get('raw_within1', 0):.4f}  "
                          f"epoch={be}  ({dt:.1f}s)")
                    row = row_from_metrics(
                        run_id=run_id, prep=prep, inp=inp,
                        model_id="bilstm", family="bilstm",
                        train_m=trm, val_m=vm, test_m=tm,
                        best_epoch=be, seconds=dt,
                    )
                    append_row(row)
                except Exception as e:
                    print(f"  [{run_id:30s}] FAILED: {e}")
                    import traceback; traceback.print_exc()

    print("\n[*] exp05 done")


if __name__ == "__main__":
    main()
