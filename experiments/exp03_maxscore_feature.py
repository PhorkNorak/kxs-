"""exp03 — TF-IDF + SVR with `max_score` concatenated to the feature vector.

The vanilla SVR has no knowledge of per-question scoring scale. If a max=20
question gets the same TF-IDF features as a max=5 question with similar
content, the SVR can't tell that a high similarity should map to a 20 vs a 5.

Adding the max_score as one extra feature lets the SVR learn a per-scale
calibration on top of the standard interaction features
`[a; b; |a-b|; a*b; cos]`.

Grid: 3 preprocess × 2 input formats × 2 datasets = 12 cells.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity

import config  # noqa: E402
import data   # noqa: E402
import importlib
from evaluate import metrics as evaluate_metrics  # noqa: E402
from train import save_predictions_csv, save_json, run_dir  # noqa: E402


PREPROCESS = ["raw", "clean", "segment"]
INPUTS     = ["ra", "qar"]


class TFIDFSVRWithMaxScore:
    """TF-IDF char-n-gram SVR with max_score as an extra feature."""

    def __init__(self, C_svr: float = 1.0,
                 max_features: int = 15000, ngram=(2, 4)):
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram, max_features=max_features,
        )
        self.svr = SVR(kernel="rbf", C=C_svr)
        self._max_score_scale = 1.0  # normalize the feature once fit

    def _features(self, side_a, side_b, max_scores):
        a = self.vec.transform(side_a).toarray()
        b = self.vec.transform(side_b).toarray()
        cos = np.array([
            cosine_similarity(a[i:i+1], b[i:i+1])[0, 0] for i in range(a.shape[0])
        ]).reshape(-1, 1)
        ms = np.asarray(max_scores, dtype=np.float64).reshape(-1, 1) / self._max_score_scale
        return np.hstack([a, b, np.abs(a - b), a * b, cos, ms])

    def fit(self, train_a, train_b, train_scores, train_max_scores):
        self.vec.fit(list(train_a) + list(train_b))
        self._max_score_scale = float(np.max(train_max_scores)) or 1.0
        X = self._features(train_a, train_b, train_max_scores)
        self.svr.fit(X, np.asarray(train_scores, dtype=np.float32))
        return self

    def predict(self, side_a, side_b, max_scores):
        X = self._features(side_a, side_b, max_scores)
        return np.clip(self.svr.predict(X), 0.0, 1.0)


def train_one_cell(prep, inp, train_df, val_df, test_df, run_id):
    train_p = data.apply_preprocess(train_df, prep)
    val_p   = data.apply_preprocess(val_df,   prep)
    test_p  = data.apply_preprocess(test_df,  prep)
    train_a, train_b = data.build_text_lists(train_p, inp)
    val_a,   val_b   = data.build_text_lists(val_p,   inp)
    test_a,  test_b  = data.build_text_lists(test_p,  inp)

    model = TFIDFSVRWithMaxScore()
    model.fit(train_a, train_b,
              train_p["normalized_score"].values,
              train_p["Max Score"].values)

    train_pred = model.predict(train_a, train_b, train_p["Max Score"].values)
    val_pred   = model.predict(val_a,   val_b,   val_p["Max Score"].values)
    test_pred  = model.predict(test_a,  test_b,  test_p["Max Score"].values)

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
              {"run_id": run_id, "model": "tfidf_svr_maxfeat",
               "family": "classical_v03",
               "preprocess": prep, "input": inp,
               "experiment": "v03_maxscore_feature"})
    save_json(os.path.join(out, "metrics.json"),
              {"train": train_m, "val": val_m, "test": test_m, "best_epoch": None})
    save_predictions_csv(out, "train", train_p, train_pred)
    save_predictions_csv(out, "val",   val_p,   val_pred)
    save_predictions_csv(out, "test",  test_p,  test_pred)
    return {"train": train_m, "val": val_m, "test": test_m}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        dst = patch_config(ds["run_name"], ds["drop_zero"],
                           exp_suffix="v03_maxfeat")
        importlib.reload(data)  # data.py reads C.DROP_SCORE_ZERO
        banner(f"exp03 maxfeat-SVR  {ds['label']}  resume={args.resume}", dst)

        if not args.resume:
            reset_leaderboard()
        df = data.load_dataframe()
        train_df, val_df, test_df = data.split_dataframe(df)
        print(f"  rows={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        for prep in PREPROCESS:
            for inp in INPUTS:
                run_id = f"{prep}_{inp}_tfidf_svr_maxfeat"
                if args.resume and cell_already_done(run_id):
                    backfill_leaderboard_row_from_metrics(
                        run_id, prep, inp,
                        model_id="tfidf_svr_maxfeat", family="classical_v03",
                    )
                    print(f"  [{run_id:35s}] SKIP (already done)")
                    continue
                t0 = time.time()
                try:
                    r = train_one_cell(prep, inp, train_df, val_df, test_df, run_id)
                    dt = time.time() - t0
                    row = row_from_metrics(
                        run_id=run_id, prep=prep, inp=inp,
                        model_id="tfidf_svr_maxfeat",
                        family="classical_v03",
                        train_m=r["train"], val_m=r["val"], test_m=r["test"],
                        best_epoch=None, seconds=dt,
                    )
                    append_row(row)
                    print(f"  [{run_id:35s}] train_qwk={r['train']['qwk']:.4f} -> "
                          f"test_qwk={r['test']['qwk']:.4f}  "
                          f"acc5={r['test']['accuracy']:.4f}  "
                          f"raw_w1={r['test']['raw_within1']:.4f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_id:35s}] FAILED: {e}")

    print("\n[*] exp03 done")


if __name__ == "__main__":
    main()
