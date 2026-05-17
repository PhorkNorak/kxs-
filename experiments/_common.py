"""Shared utilities for all experiments.

Provides:
  * DATASETS — iteration over the two dataset variants (909 / 895)
  * patch_config — swap RAW_CSV / RUN_NAME / DROP_SCORE_ZERO / paths for a variant
  * LEADERBOARD_HEADER — same 24-column schema as run_all.py
  * append_row, reset_leaderboard, row_from_metrics — leaderboard helpers
  * banner — uniform per-dataset section header
"""

from __future__ import annotations

import csv
import os
import sys
import time

# Make the project root importable when this file is imported via
# `python experiments/expNN_*.py` or `python -m experiments.expNN_*`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402


# Three dataset variants. Every experiment runs on all of them by default.
# `raw_csv` is relative to PROJECT_ROOT/data/.
DATASETS = [
    {"run_name": "full",      "drop_zero": False, "raw_csv": "dataset.csv",
     "label": "1184 (full)"},
    {"run_name": "no10c",     "drop_zero": False, "raw_csv": "dataset_no_10c_biology.csv",
     "label": "909 (no 10C biology)"},
    {"run_name": "no10c_no0", "drop_zero": True,  "raw_csv": "dataset_no_10c_biology.csv",
     "label": "895 (no 10C, drop 0)"},
]


def select_datasets(only=None):
    """Return DATASETS filtered by a list of run_names. None -> all."""
    if not only:
        return list(DATASETS)
    only_set = set(only)
    keep = [d for d in DATASETS if d["run_name"] in only_set]
    missing = only_set - {d["run_name"] for d in keep}
    if missing:
        raise ValueError(f"Unknown dataset run_name(s): {sorted(missing)}. "
                         f"Choose from {[d['run_name'] for d in DATASETS]}.")
    return keep


# Same 24 columns as run_all.py / run_tfidf_3datasets.py.
LEADERBOARD_HEADER = [
    "run_id", "preprocess", "input", "model", "family",
    # train-set metrics
    "train_qwk", "train_accuracy", "train_adjacent_accuracy", "train_mae",
    "train_raw_exact", "train_raw_within1", "train_raw_mae", "train_pct_mae",
    # test-set metrics
    "test_qwk", "test_accuracy", "test_adjacent_accuracy", "test_mae",
    "test_raw_exact", "test_raw_within1", "test_raw_mae", "test_pct_mae",
    "val_qwk", "best_epoch", "seconds",
]


def patch_config(run_name: str, drop_zero: bool, exp_suffix: str = "",
                 raw_csv: str = None):
    """Mutate the shared `config` module to target one dataset variant + experiment.

    Parameters
    ----------
    run_name : str
        Dataset identifier — "full", "no10c", or "no10c_no0".
    drop_zero : bool
        Whether to filter out score_label==0 rows.
    exp_suffix : str
        Optional suffix appended to the output directory, e.g. "v02_calibrated".
    raw_csv : str
        Dataset CSV file name (under data/). If None, looks it up from DATASETS
        by run_name. None is allowed for backward compatibility.

    Returns the new RESULTS_DIR.
    """
    if raw_csv is None:
        # Look up from DATASETS
        for ds in DATASETS:
            if ds["run_name"] == run_name:
                raw_csv = ds["raw_csv"]
                break
    config.RUN_NAME        = run_name
    config.DROP_SCORE_ZERO = drop_zero
    if raw_csv:
        config.RAW_CSV = os.path.join(config.PROJECT_ROOT, "data", raw_csv)
    out_name = f"results_{run_name}" if not exp_suffix else f"results_{run_name}_{exp_suffix}"
    config.RESULTS_DIR = os.path.join(config.PROJECT_ROOT, out_name)
    config.RUNS_DIR    = os.path.join(config.RESULTS_DIR, "runs")
    config.LEADERBOARD = os.path.join(config.RESULTS_DIR, "leaderboard.csv")
    config.XAI_DIR     = os.path.join(config.PROJECT_ROOT, f"xai_visuals_{run_name}_{exp_suffix}" if exp_suffix else f"xai_visuals_{run_name}")
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    return config.RESULTS_DIR


def add_datasets_flag(parser):
    """Add a --datasets argument to an argparse parser, listing valid names."""
    valid = [d["run_name"] for d in DATASETS]
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=valid,
        help=f"Subset of datasets to run on. Default: all ({valid}).",
    )


def reset_leaderboard():
    if os.path.exists(config.LEADERBOARD):
        os.remove(config.LEADERBOARD)


def append_row(row: dict):
    new_file = not os.path.exists(config.LEADERBOARD)
    with open(config.LEADERBOARD, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _fmt(metric_dict, key, fmt=".4f"):
    v = metric_dict.get(key)
    if v is None:
        return ""
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return ""


def row_from_metrics(run_id, prep, inp, model_id, family,
                     train_m, val_m, test_m,
                     best_epoch=None, seconds=0.0):
    """Build a leaderboard row dict from the metric dicts."""
    train_m = train_m or {}
    val_m   = val_m   or {}
    test_m  = test_m  or {}
    return {
        "run_id": run_id, "preprocess": prep, "input": inp,
        "model": model_id, "family": family,
        "train_qwk": _fmt(train_m, "qwk"),
        "train_accuracy": _fmt(train_m, "accuracy"),
        "train_adjacent_accuracy": _fmt(train_m, "adjacent_accuracy"),
        "train_mae": _fmt(train_m, "mae"),
        "train_raw_exact":   _fmt(train_m, "raw_exact"),
        "train_raw_within1": _fmt(train_m, "raw_within1"),
        "train_raw_mae":     _fmt(train_m, "raw_mae"),
        "train_pct_mae":     _fmt(train_m, "pct_mae"),
        "test_qwk": _fmt(test_m, "qwk"),
        "test_accuracy": _fmt(test_m, "accuracy"),
        "test_adjacent_accuracy": _fmt(test_m, "adjacent_accuracy"),
        "test_mae": _fmt(test_m, "mae"),
        "test_raw_exact":   _fmt(test_m, "raw_exact"),
        "test_raw_within1": _fmt(test_m, "raw_within1"),
        "test_raw_mae":     _fmt(test_m, "raw_mae"),
        "test_pct_mae":     _fmt(test_m, "pct_mae"),
        "val_qwk": _fmt(val_m, "qwk"),
        "best_epoch": "" if best_epoch is None else best_epoch,
        "seconds": f"{seconds:.1f}",
    }


def banner(title: str, results_dir: str):
    print(f"\n{'='*10} {title}  ->  {results_dir} {'='*10}")
