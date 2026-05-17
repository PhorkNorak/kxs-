"""exp03b — Neural models with max_score as an extra scalar feature.

Parallels v03 (TF-IDF SVR + max_score feature) but for neural models.
Concatenates `max_score / MAX_SCORE_NORMALIZER` to the head input of:
  - BiLSTM + Attention (after 4-way interaction)
  - Transformer DualEncoder (after 4-way interaction)
  - Transformer CrossEncoder (after [CLS] hidden)

Grid: 3 preprocess x 2 input x 7 models = 42 cells per dataset.
  (1 BiLSTM + 3 dual + 3 cross transformers)

HPC ONLY for the transformer cells. BiLSTM cells are CPU-feasible.

Run order: AFTER exp01/v01 baseline (so the original v01 TF-IDF cells aren't
overwritten). Output lives at `results_<ds>_v03b_maxfeat_neural/`.
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
BACKBONES  = ["mbert", "xlmr", "gte"]
ARCHS      = ["dual", "cross"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bilstm", action="store_true",
                    help="Skip BiLSTM cells (e.g. only want transformer maxfeat)")
    ap.add_argument("--skip-transformer", action="store_true",
                    help="Skip transformer cells (e.g. CPU-only run, BiLSTM only)")
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        dst = patch_config(ds["run_name"], ds["drop_zero"],
                           exp_suffix="v03b_maxfeat_neural")
        importlib.reload(data)
        importlib.reload(train)
        banner(f"exp03b max-feat neural  {ds['label']}  resume={args.resume}", dst)

        if not args.resume:
            reset_leaderboard()
        df = data.load_dataframe()
        train_df, val_df, test_df = data.split_dataframe(df)
        print(f"  rows={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        # BiLSTM with max-score feature
        if not args.skip_bilstm:
            for prep in PREPROCESS:
                for inp in INPUTS:
                    run_id = f"{prep}_{inp}_bilstm_maxfeat"
                    if args.resume and cell_already_done(run_id):
                        backfill_leaderboard_row_from_metrics(
                            run_id, prep, inp,
                            model_id="bilstm_maxfeat", family="bilstm_maxfeat",
                        )
                        print(f"  [{run_id:35s}] SKIP (already done)")
                        continue
                    t0 = time.time()
                    try:
                        result = train.train_bilstm(
                            prep, inp, train_df, val_df, test_df, run_id,
                            max_feat=True,
                        )
                        dt = time.time() - t0
                        trm = result.get("train", {})
                        vm  = result["val"]
                        tm  = result["test"]
                        be  = result.get("best_epoch", "")
                        print(f"  [{run_id:35s}] train_qwk={trm.get('qwk', 0):.4f} -> "
                              f"test_qwk={tm['qwk']:.4f}  acc5={tm['accuracy']:.4f}  "
                              f"raw_w1={tm.get('raw_within1', 0):.4f}  "
                              f"epoch={be}  ({dt:.1f}s)")
                        row = row_from_metrics(
                            run_id=run_id, prep=prep, inp=inp,
                            model_id="bilstm_maxfeat", family="bilstm_maxfeat",
                            train_m=trm, val_m=vm, test_m=tm,
                            best_epoch=be, seconds=dt,
                        )
                        append_row(row)
                    except Exception as e:
                        print(f"  [{run_id:35s}] FAILED: {e}")
                        import traceback; traceback.print_exc()

        # Transformer dual + cross with max-score feature
        if not args.skip_transformer:
            for prep in PREPROCESS:
                for inp in INPUTS:
                    for arch in ARCHS:
                        for bb in BACKBONES:
                            run_id = f"{prep}_{inp}_{arch}_{bb}_maxfeat"
                            if args.resume and cell_already_done(run_id):
                                backfill_leaderboard_row_from_metrics(
                                    run_id, prep, inp,
                                    model_id=f"{arch}_{bb}_maxfeat",
                                    family=f"{arch}_maxfeat",
                                )
                                print(f"  [{run_id:40s}] SKIP (already done)")
                                continue
                            t0 = time.time()
                            try:
                                result = train.train_transformer(
                                    arch, bb, prep, inp,
                                    train_df, val_df, test_df, run_id,
                                    max_feat=True,
                                )
                                dt = time.time() - t0
                                trm = result.get("train", {})
                                vm  = result["val"]
                                tm  = result["test"]
                                be  = result.get("best_epoch", "")
                                print(f"  [{run_id:40s}] train_qwk={trm.get('qwk', 0):.4f} -> "
                                      f"test_qwk={tm['qwk']:.4f}  "
                                      f"raw_w1={tm.get('raw_within1', 0):.4f}  "
                                      f"epoch={be}  ({dt:.1f}s)")
                                row = row_from_metrics(
                                    run_id=run_id, prep=prep, inp=inp,
                                    model_id=f"{arch}_{bb}_maxfeat",
                                    family=f"{arch}_maxfeat",
                                    train_m=trm, val_m=vm, test_m=tm,
                                    best_epoch=be, seconds=dt,
                                )
                                append_row(row)
                            except Exception as e:
                                dt = time.time() - t0
                                print(f"  [{run_id:40s}] FAILED in {dt:.1f}s: "
                                      f"{type(e).__name__}: {e}")
                                row = row_from_metrics(
                                    run_id=run_id, prep=prep, inp=inp,
                                    model_id=f"{arch}_{bb}_maxfeat",
                                    family=f"{arch}_maxfeat",
                                    train_m={}, val_m={},
                                    test_m={"qwk": float("nan")},
                                    best_epoch="", seconds=dt,
                                )
                                row["test_qwk"] = "ERR"
                                append_row(row)

    print("\n[*] exp03b done")


if __name__ == "__main__":
    main()
