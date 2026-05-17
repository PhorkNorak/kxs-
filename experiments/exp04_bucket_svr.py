"""exp04 — Per-max-score-bucket TF-IDF + SVR.

For each unique `Max Score` value, train a SEPARATE SVR using only the train
samples whose Max Score matches the bucket. At inference, route each test
sample to its matching bucket model.

Buckets with too few training samples (< MIN_BUCKET_SIZE) fall back to a
global model trained on all training data — this avoids overfitting the
solo-sample buckets (e.g. Max=12 has only ~4 rows).

Grid: 3 preprocess × 2 input formats × 2 datasets = 12 cells.
Each cell trains up to 8 bucket SVRs + 1 global fallback.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

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
import importlib
from models.classical import TFIDFSVR  # noqa: E402
from evaluate import metrics as evaluate_metrics  # noqa: E402
from train import save_predictions_csv, save_json, run_dir  # noqa: E402


PREPROCESS = ["raw", "clean", "segment"]
INPUTS     = ["ra", "qar"]
MIN_BUCKET_SIZE = 40  # fall back to global model if bucket has fewer


def predict_with_bucketed_svrs(train_p, val_p, test_p, prep, inp):
    """Return (train_pred, val_pred, test_pred, used_bucket_info)."""
    train_a, train_b = data.build_text_lists(train_p, inp)
    val_a,   val_b   = data.build_text_lists(val_p,   inp)
    test_a,  test_b  = data.build_text_lists(test_p,  inp)

    train_max = train_p["Max Score"].values
    val_max   = val_p["Max Score"].values
    test_max  = test_p["Max Score"].values

    # Global fallback model
    global_model = TFIDFSVR()
    global_model.fit(train_a, train_b, train_p["normalized_score"].values)

    bucket_info = {}
    bucket_models = {}

    for m in sorted(np.unique(train_max)):
        idx = np.where(train_max == m)[0]
        if len(idx) < MIN_BUCKET_SIZE:
            bucket_info[int(m)] = {"n_train": len(idx), "uses": "global_fallback"}
            continue
        a = [train_a[i] for i in idx]
        b = [train_b[i] for i in idx]
        s = train_p["normalized_score"].values[idx]
        try:
            model = TFIDFSVR()
            model.fit(a, b, s)
            bucket_models[int(m)] = model
            bucket_info[int(m)] = {"n_train": len(idx), "uses": "bucket_specific"}
        except Exception as e:
            bucket_info[int(m)] = {"n_train": len(idx), "uses": "global_fallback",
                                   "error": str(e)}

    def _predict(a_list, b_list, maxes):
        out = np.zeros(len(a_list), dtype=np.float64)
        for i, m in enumerate(maxes):
            m_int = int(m)
            if m_int in bucket_models:
                out[i] = bucket_models[m_int].predict([a_list[i]], [b_list[i]])[0]
            else:
                out[i] = global_model.predict([a_list[i]], [b_list[i]])[0]
        return np.clip(out, 0.0, 1.0)

    train_pred = _predict(train_a, train_b, train_max)
    val_pred   = _predict(val_a,   val_b,   val_max)
    test_pred  = _predict(test_a,  test_b,  test_max)

    return train_pred, val_pred, test_pred, bucket_info


def train_one_cell(prep, inp, train_df, val_df, test_df, run_id):
    train_p = data.apply_preprocess(train_df, prep)
    val_p   = data.apply_preprocess(val_df,   prep)
    test_p  = data.apply_preprocess(test_df,  prep)

    train_pred, val_pred, test_pred, bucket_info = \
        predict_with_bucketed_svrs(train_p, val_p, test_p, prep, inp)

    train_m = evaluate_metrics(train_pred, train_p["score_label"].values,
                               max_scores=train_p["Max Score"].values,
                               true_raw=train_p["Student Score"].values)
    val_m   = evaluate_metrics(val_pred,   val_p["score_label"].values,
                               max_scores=val_p["Max Score"].values,
                               true_raw=val_p["Student Score"].values)
    test_m  = evaluate_metrics(test_pred,  test_p["score_label"].values,
                               max_scores=test_p["Max Score"].values,
                               true_raw=test_p["Student Score"].values)

    out = run_dir(run_id)
    save_json(os.path.join(out, "config.json"),
              {"run_id": run_id, "model": "tfidf_svr_bucket",
               "family": "classical_v04",
               "preprocess": prep, "input": inp,
               "experiment": "v04_bucket_svr",
               "min_bucket_size": MIN_BUCKET_SIZE,
               "bucket_info": bucket_info})
    save_json(os.path.join(out, "metrics.json"),
              {"train": train_m, "val": val_m, "test": test_m, "best_epoch": None})
    save_predictions_csv(out, "train", train_p, train_pred)
    save_predictions_csv(out, "val",   val_p,   val_pred)
    save_predictions_csv(out, "test",  test_p,  test_pred)
    return {"train": train_m, "val": val_m, "test": test_m,
            "bucket_info": bucket_info}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        dst = patch_config(ds["run_name"], ds["drop_zero"],
                           exp_suffix="v04_bucket")
        importlib.reload(data)
        banner(f"exp04 bucket-SVR  {ds['label']}  resume={args.resume}", dst)

        if not args.resume:
            reset_leaderboard()
        df = data.load_dataframe()
        train_df, val_df, test_df = data.split_dataframe(df)
        print(f"  rows={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        for prep in PREPROCESS:
            for inp in INPUTS:
                run_id = f"{prep}_{inp}_tfidf_svr_bucket"
                if args.resume and cell_already_done(run_id):
                    backfill_leaderboard_row_from_metrics(
                        run_id, prep, inp,
                        model_id="tfidf_svr_bucket", family="classical_v04",
                    )
                    print(f"  [{run_id:35s}] SKIP (already done)")
                    continue
                t0 = time.time()
                try:
                    r = train_one_cell(prep, inp, train_df, val_df, test_df, run_id)
                    dt = time.time() - t0
                    used = [m for m, info in r["bucket_info"].items()
                            if info["uses"] == "bucket_specific"]
                    print(f"  [{run_id:35s}] used_buckets={used}  "
                          f"train_qwk={r['train']['qwk']:.4f} -> "
                          f"test_qwk={r['test']['qwk']:.4f}  "
                          f"acc5={r['test']['accuracy']:.4f}  "
                          f"raw_w1={r['test']['raw_within1']:.4f}  ({dt:.1f}s)")
                    row = row_from_metrics(
                        run_id=run_id, prep=prep, inp=inp,
                        model_id="tfidf_svr_bucket",
                        family="classical_v04",
                        train_m=r["train"], val_m=r["val"], test_m=r["test"],
                        best_epoch=None, seconds=dt,
                    )
                    append_row(row)
                except Exception as e:
                    print(f"  [{run_id:35s}] FAILED: {e}")
                    import traceback; traceback.print_exc()

    print("\n[*] exp04 done")


if __name__ == "__main__":
    main()
