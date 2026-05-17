"""exp06 — Transformer (DualEncoder + CrossEncoder x mBERT/XLM-R/GTE) full grid.

Grid: 3 preprocess x 2 input x (3 backbones x 2 architectures) = 36 cells.
x 2 datasets = 72 cells.

REQUIRES NETWORK ACCESS to huggingface.co (or pre-cached models in
~/.cache/huggingface/hub/). On Windows, the SSL cert store sometimes blocks
HF downloads. The script attempts a soft fix by loading certifi's CA bundle
into the environment before importing transformers.

If you still hit MaxRetryError: SSLCertVerificationError, run
`pip install pip-system-certs` once in your venv and re-run this script.
"""

from __future__ import annotations

import os
import sys
import time
import importlib

# Attempt to fix Windows SSL cert verify failures BEFORE any HF import.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE",     certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("CURL_CA_BUNDLE",    certifi.where())
except Exception:
    pass

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
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    for ds in select_datasets(args.datasets):
        dst = patch_config(ds["run_name"], ds["drop_zero"],
                           exp_suffix="v06_transformer")
        importlib.reload(data)
        importlib.reload(train)
        banner(f"exp06 transformer  {ds['label']}  resume={args.resume}", dst)

        if not args.resume:
            reset_leaderboard()
        df = data.load_dataframe()
        train_df, val_df, test_df = data.split_dataframe(df)
        print(f"  rows={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        for prep in PREPROCESS:
            for inp in INPUTS:
                for arch in ARCHS:
                    for bb in BACKBONES:
                        run_id = f"{prep}_{inp}_{arch}_{bb}"
                        if args.resume and cell_already_done(run_id):
                            backfill_leaderboard_row_from_metrics(
                                run_id, prep, inp,
                                model_id=f"{arch}_{bb}", family=arch,
                            )
                            print(f"  [{run_id:35s}] SKIP (already done)")
                            continue
                        t0 = time.time()
                        try:
                            result = train.train_transformer(
                                arch, bb, prep, inp,
                                train_df, val_df, test_df, run_id,
                            )
                            dt = time.time() - t0
                            trm = result.get("train", {})
                            vm  = result["val"]
                            tm  = result["test"]
                            be  = result.get("best_epoch", "")
                            print(f"  [{run_id:35s}] train_qwk={trm.get('qwk', 0):.4f} -> "
                                  f"test_qwk={tm['qwk']:.4f}  "
                                  f"raw_w1={tm.get('raw_within1', 0):.4f}  "
                                  f"epoch={be}  ({dt:.1f}s)")
                            row = row_from_metrics(
                                run_id=run_id, prep=prep, inp=inp,
                                model_id=f"{arch}_{bb}", family=arch,
                                train_m=trm, val_m=vm, test_m=tm,
                                best_epoch=be, seconds=dt,
                            )
                            append_row(row)
                        except Exception as e:
                            dt = time.time() - t0
                            print(f"  [{run_id:35s}] FAILED in {dt:.1f}s: "
                                  f"{type(e).__name__}: {e}")
                            row = row_from_metrics(
                                run_id=run_id, prep=prep, inp=inp,
                                model_id=f"{arch}_{bb}", family=arch,
                                train_m={}, val_m={}, test_m={"qwk": float("nan")},
                                best_epoch="", seconds=dt,
                            )
                            row["test_qwk"] = "ERR"
                            append_row(row)

    print("\n[*] exp06 done")


if __name__ == "__main__":
    main()
