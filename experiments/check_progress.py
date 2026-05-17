"""Inspect experiment completion status and print a resume command.

For every (experiment, dataset) cell, checks:
  - whether results_<ds>_<exp>/leaderboard.csv exists
  - whether it has the expected number of rows (cells)
  - reports DONE / PARTIAL / MISSING

Then synthesizes a `mkdir + python -u && ...` chain that only runs the
experiments and dataset variants that haven't completed yet, using the
`--datasets` flag to skip already-done dataset variants per experiment.

Usage:
    python experiments/check_progress.py            # human-readable + resume cmd
    python experiments/check_progress.py --quiet    # only the resume cmd
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import DATASETS  # noqa: E402


# (key, dir_suffix, expected_cells_per_dataset, resume_command_template, supports_resume)
# resume_command_template uses {ds_args} for --datasets and {resume_arg} for --resume
EXPECTED = [
    ("v01_baseline",                    "",                                  12,
     "python -u experiments/exp01_tfidf_baseline.py {ds_args} {resume_arg}", True),
    ("v02_calibrated",                  "v02_calibrated",                    12,
     "python -u experiments/exp02_threshold_calibration.py {ds_args}", False),
    ("v03_maxfeat",                     "v03_maxfeat",                        6,
     "python -u experiments/exp03_maxscore_feature.py {ds_args} {resume_arg}", True),
    ("v04_bucket",                      "v04_bucket",                         6,
     "python -u experiments/exp04_bucket_svr.py {ds_args} {resume_arg}", True),
    ("v05_bilstm",                      "v05_bilstm",                         6,
     "python -u experiments/exp05_bilstm.py {ds_args} {resume_arg}", True),
    ("v06_transformer",                 "v06_transformer",                   36,
     "python -u experiments/exp06_transformer.py {ds_args} {resume_arg}", True),
    ("v03b_maxfeat_neural",             "v03b_maxfeat_neural",               42,
     "python -u experiments/exp03b_maxfeat_neural.py {ds_args} {resume_arg}", True),
    ("v05_bilstm_calibrated",           "v05_bilstm_calibrated",              6,
     "python -u experiments/exp02_threshold_calibration.py --source v05_bilstm {ds_args}", False),
    ("v06_transformer_calibrated",      "v06_transformer_calibrated",        36,
     "python -u experiments/exp02_threshold_calibration.py --source v06_transformer {ds_args}", False),
    ("v03b_maxfeat_neural_calibrated",  "v03b_maxfeat_neural_calibrated",    42,
     "python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural {ds_args}", False),
    ("v07_ensemble",                    "v07_ensemble",                       1,
     "python -u experiments/exp07_ensemble.py {ds_args}", False),
]


def check_cell(ds_run_name: str, suffix: str, expected: int):
    name = f"results_{ds_run_name}" if not suffix else f"results_{ds_run_name}_{suffix}"
    lb = os.path.join(_ROOT, name, "leaderboard.csv")
    if not os.path.exists(lb):
        return "MISSING", 0
    try:
        df = pd.read_csv(lb)
    except Exception as e:
        return f"CORRUPT ({e})", 0
    # Count rows excluding ERR rows (failed cells should be re-run)
    if "test_qwk" in df.columns:
        valid = df[df["test_qwk"].astype(str).str.upper() != "ERR"]
        n = len(valid)
    else:
        n = len(df)
    if n >= expected:
        return "DONE", n
    if n == 0:
        return "EMPTY", n
    return "PARTIAL", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="Only print the resume command, no status table.")
    args = ap.parse_args()

    missing_per_exp = {}  # exp_key -> [ds_run_name, ...]
    partial_per_exp = set()  # exp_keys that have at least one PARTIAL ds
    rows = []
    total_done = total_expected = 0
    for exp_key, suffix, expected, _cmd, _resume_ok in EXPECTED:
        for ds in DATASETS:
            status, n = check_cell(ds["run_name"], suffix, expected)
            rows.append((exp_key, ds["run_name"], status, n, expected))
            total_expected += expected
            if status == "DONE":
                total_done += expected
            else:
                missing_per_exp.setdefault(exp_key, []).append(ds["run_name"])
                if status == "PARTIAL":
                    partial_per_exp.add(exp_key)

    if not args.quiet:
        print(f"\n{'experiment':<35s} {'dataset':<12s} {'status':<10s} {'rows':>10s}")
        print("-" * 75)
        last_exp = None
        for exp_key, ds_name, status, n, expected in rows:
            if last_exp and exp_key != last_exp:
                print()
            print(f"{exp_key:<35s} {ds_name:<12s} {status:<10s} {n:>4d}/{expected:<4d}")
            last_exp = exp_key
        print(f"\nTotals: {total_done} / {total_expected} cells complete "
              f"({100*total_done/max(total_expected,1):.1f}%)")

    if not missing_per_exp:
        print("\n[OK] Everything is done. No resume needed.")
        return 0

    # Build resume command. Always force re-run of v07 ensemble + compare_all
    # at the end if anything else is being re-run, because v07 depends on
    # upstream predictions that may have just changed.
    lines = []
    for exp_key, suffix, expected, cmd_template, supports_resume in EXPECTED:
        if exp_key == "v07_ensemble":
            continue  # handled separately below
        if exp_key not in missing_per_exp:
            continue
        miss = missing_per_exp[exp_key]
        if len(miss) == len(DATASETS):
            ds_args = ""
        else:
            ds_args = "--datasets " + " ".join(miss)
        # Add --resume on training scripts so partial runs aren't redone.
        resume_arg = "--resume" if (supports_resume and exp_key in partial_per_exp) else ""
        line = cmd_template.format(ds_args=ds_args, resume_arg=resume_arg).strip()
        while "  " in line:
            line = line.replace("  ", " ")
        lines.append(line)

    # Decide if v07 needs re-run: yes if any of its upstreams will run, or if
    # v07 itself is incomplete.
    upstream_keys = {"v01_baseline", "v03_maxfeat", "v03b_maxfeat_neural",
                     "v04_bucket", "v05_bilstm", "v06_transformer"}
    v07_needs_rerun = ("v07_ensemble" in missing_per_exp) or \
                       any(k in missing_per_exp for k in upstream_keys)
    if v07_needs_rerun:
        lines.append("python -u experiments/exp07_ensemble.py")

    if not lines:
        print("\n[OK] Everything is done. No resume needed.")
        return 0

    if not args.quiet:
        print("\n=== Resume command (run in tmux) ===\n")
    print('mkdir -p logs && ( \\')
    for ln in lines:
        print(f"  {ln} && \\")
    print('  python -u experiments/compare_all.py --topk 15 \\')
    print(') 2>&1 | tee "logs/resume_$(date +%Y%m%d_%H%M%S).log"')
    return 1


if __name__ == "__main__":
    sys.exit(main())
