"""Plot training-curve PNGs from existing metrics.json files.

Two layouts depending on what's in `metrics.json["history"]`:

**Full layout** (when each epoch records `train_*`, `val_*`, `test_*`)
  Mirrors Alaoui et al. 2024 Figure 8 — a 2x2 grid:
    - model accuracy:  train (blue) vs test (orange) per epoch
    - model loss:      train_loss (blue) vs val_loss (orange) per epoch
    - cohen kappa/QWK: train (blue) vs test (orange) per epoch
    - raw within +/-1: train (blue) vs test (orange) per epoch
  Best epoch marked with a vertical dotted line.

**Compact layout** (legacy runs with only val metrics per epoch)
  Falls back to the old behavior: val curves + final train/test as horizontal
  dashed lines.

Usage:
  python experiments/plot_history.py                            # all runs
  python experiments/plot_history.py --filter dual_gte_maxfeat  # only matching
  python experiments/plot_history.py --dir results_no10c_v06_transformer
  python experiments/plot_history.py --run results_no10c_v06_transformer/runs/clean_ra_dual_gte
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def has_full_history(history):
    """True if at least one history row has both train_* and test_* metrics."""
    return any(("train_qwk" in h or "train_accuracy" in h) and
               ("test_qwk" in h  or "test_accuracy" in h)
               for h in history)


def _plot_pair(ax, xs, train_ys, test_ys, *, title, ylabel,
               train_label="train", test_label="test",
               best_epoch=None, ylim=None):
    if train_ys is not None:
        ax.plot(xs, train_ys, linewidth=1.6, label=train_label, color="C0")
    if test_ys is not None:
        ax.plot(xs, test_ys, linewidth=1.6, label=test_label, color="C1")
    if best_epoch is not None and isinstance(best_epoch, (int, float)):
        ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_full(metrics_path: str, history, train_m_final, val_m_final, test_m_final,
              best_epoch):
    """Arabic-paper-style 2x2 figure: train vs test per epoch for 4 metrics."""
    epochs = [h["epoch"] for h in history]
    def col(key):
        return [h.get(key) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    run_id = os.path.basename(os.path.dirname(metrics_path))
    parent = os.path.basename(os.path.dirname(os.path.dirname(metrics_path)))
    fig.suptitle(f"{parent} / {run_id}", fontsize=11)

    # (0,0) model accuracy
    _plot_pair(
        axes[0, 0], epochs,
        col("train_accuracy"), col("test_accuracy"),
        title="model accuracy", ylabel="accuracy",
        best_epoch=best_epoch, ylim=(0, 1.0),
    )
    # (0,1) model loss  (train_loss + val_loss [held-out] per epoch)
    # We also have test_loss in new history; show train vs test for consistency
    test_loss = col("test_loss") if any(h.get("test_loss") is not None for h in history) else col("val_loss")
    _plot_pair(
        axes[0, 1], epochs,
        col("train_loss"), test_loss,
        title="model loss", ylabel="loss",
        best_epoch=best_epoch,
    )
    # (1,0) cohen kappa / QWK
    _plot_pair(
        axes[1, 0], epochs,
        col("train_qwk"), col("test_qwk"),
        title="cohen kappa (QWK)", ylabel="cohen_kappa",
        best_epoch=best_epoch, ylim=(0, 1.0),
    )
    # (1,1) raw within +/-1 (deployment metric, swapping in for F1)
    _plot_pair(
        axes[1, 1], epochs,
        col("train_raw_within1"), col("test_raw_within1"),
        title="raw within +/-1 point", ylabel="raw_within1",
        best_epoch=best_epoch, ylim=(0, 1.0),
    )

    plt.tight_layout()
    out = os.path.join(os.path.dirname(metrics_path), "train_history.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compact(metrics_path: str, history, train_m, val_m, test_m, best_epoch):
    """Legacy fallback when only val metrics are per-epoch."""
    epochs = [h["epoch"] for h in history]
    val_qwk    = [h.get("qwk") for h in history]
    val_acc    = [h.get("accuracy") for h in history]
    val_w1     = [h.get("raw_within1") for h in history]
    train_loss = [h.get("train_loss") for h in history]
    val_loss   = [h.get("val_loss") for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    run_id = os.path.basename(os.path.dirname(metrics_path))
    parent = os.path.basename(os.path.dirname(os.path.dirname(metrics_path)))
    fig.suptitle(f"{parent} / {run_id}  (val-only per-epoch history)", fontsize=11)

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train_loss", linewidth=1.6, color="C0")
    ax.plot(epochs, val_loss,   label="val_loss",   linewidth=1.6, color="C1")
    if best_epoch: ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.set_title("Loss curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for ax, (curve, key, lab, ylim) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [(val_qwk, "qwk", "model accuracy QWK", (0,1)),
         (val_acc, "accuracy", "model accuracy", (0,1)),
         (val_w1, "raw_within1", "raw within +/-1", (0,1))],
    ):
        ax.plot(epochs, curve, label=f"val_{key}", linewidth=1.6, color="C1")
        if key in train_m:
            ax.axhline(train_m[key], color="C0", linestyle="--", alpha=0.7,
                       label=f"final train = {train_m[key]:.3f}")
        if key in test_m:
            ax.axhline(test_m[key], color="C2", linestyle="--", alpha=0.7,
                       label=f"final test = {test_m[key]:.3f}")
        ax.set_xlabel("epoch"); ax.set_ylabel(key); ax.set_title(lab)
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(*ylim)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(metrics_path), "train_history.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_one(metrics_path: str) -> str:
    """Returns 'full', 'compact', or 'skipped'."""
    d = json.load(open(metrics_path, encoding="utf-8"))
    history = d.get("history")
    if not history or len(history) < 2:
        return "skipped"
    train_m = d.get("train", {}) or {}
    val_m   = d.get("val",   {}) or {}
    test_m  = d.get("test",  {}) or {}
    best_ep = d.get("best_epoch")

    if has_full_history(history):
        plot_full(metrics_path, history, train_m, val_m, test_m, best_ep)
        return "full"
    plot_compact(metrics_path, history, train_m, val_m, test_m, best_ep)
    return "compact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="Single results dir, e.g. results_no10c_v06_transformer")
    ap.add_argument("--run", default=None,
                    help="Single run dir, e.g. results_X/runs/Y")
    ap.add_argument("--filter", default=None,
                    help="Only plot runs whose id contains this substring")
    args = ap.parse_args()

    if args.run:
        mpath = os.path.join(args.run, "metrics.json")
        if not os.path.exists(mpath):
            print(f"[!] {mpath} missing"); return
        layout = plot_one(mpath)
        print(f"{mpath}  layout={layout}")
        return

    if args.dir:
        roots = [os.path.join(_ROOT, args.dir)]
    else:
        roots = sorted(glob(os.path.join(_ROOT, "results_*")))

    counts = {"full": 0, "compact": 0, "skipped": 0}
    for root in roots:
        runs_dir = os.path.join(root, "runs")
        if not os.path.isdir(runs_dir):
            continue
        for cell in sorted(os.listdir(runs_dir)):
            if args.filter and args.filter not in cell:
                continue
            mpath = os.path.join(runs_dir, cell, "metrics.json")
            if not os.path.exists(mpath):
                continue
            counts[plot_one(mpath)] += 1

    print(f"full(train+test): {counts['full']}   "
          f"compact(val-only): {counts['compact']}   "
          f"skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
