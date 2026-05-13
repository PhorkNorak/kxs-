# Class Imbalance Fixes — Design Spec
**Date:** 2026-05-13  
**Scope:** `models/losses.py`, `evaluation/__init__.py`, `run_all.py`

---

## Problem

KhmerSAG has a severe 5-bin class imbalance (Bin 0: 1.3% → Bin 4: 42.1%).  
Three gaps exist in the current code:

1. `CORNLoss` has no class weighting — minority bins are under-penalised.  
2. `KXCLLoss` never passes class weights into `CORNLoss`.  
3. No threshold calibration at inference — the model over-predicts bin 4.

---

## Component 1 — Class-Weighted CORN (`models/losses.py`)

### Change

Add an optional `class_weights` tensor and `label_smoothing` float to `CORNLoss`.

**Class weighting:** For each binary sub-task k, compute a per-sample weight as
`w_i = class_weights[label_i]` for samples where `label_i >= k`. Pass these as the
`weight` argument of `F.binary_cross_entropy_with_logits`.

**Label smoothing:** Smooth the hard binary target `b ∈ {0, 1}` toward ε, giving
`b_smooth = b * (1 - ε) + (1 - b) * ε`. The same ε applies across all k.
Default `epsilon = 0.0` (off, backward-compatible).

### Interface

```python
CORNLoss(num_classes=5, epsilon=0.0)
forward(logits, labels, class_weights=None)
```

`class_weights` is a 1-D tensor of shape `[num_classes]` (e.g., from
`compute_class_weights`), broadcast per-sample inside `forward`.

---

## Component 2 — Weighted CORN in `KXCLLoss` (`models/losses.py`)

### Change

Add `epsilon` and `use_class_weights` flags to `KXCLLoss.__init__`.  
`forward` accepts `class_weights=None` and passes it through to `CORNLoss.forward`.

### Interface

```python
KXCLLoss(num_classes=5, alpha=1.0, beta=0.5,
         temperature=0.07, margin_scale=1.0,
         epsilon=0.0, use_class_weights=False)
forward(logits, labels, embeddings, scores, class_weights=None)
```

---

## Component 3 — QWK Threshold Calibration (`evaluation/__init__.py`)

### Purpose

After model training, find optimal bin boundaries on the validation set that
maximise QWK, then apply them at test time instead of the fixed `round(score * 4)`.

### Implementation

```
calibrate_thresholds(val_scores, val_labels, n_classes=5) → np.ndarray[n_classes-1]
apply_thresholds(scores, thresholds) → np.ndarray[int]
```

`calibrate_thresholds` uses `scipy.optimize.minimize` (Nelder-Mead) to search
over 4 boundary values in [0, 1]. The objective is negative QWK on the val set.
Initialisation: uniform boundaries `[0.1, 0.3, 0.5, 0.7, 0.9]` (5 params → keeps
`n_classes - 1 = 4` after sorting interior boundaries; initialise at 4 values).

`apply_thresholds(scores, thresholds)` wraps `np.digitize` and clips to
`[0, n_classes - 1]`.

### Scope

Calibration runs per model **after training**, using validation predictions.
Results are stored alongside the existing metrics dict under `"calibrated_qwk"`,
`"calibrated_thresholds"`, and `"calibrated_predictions"`.
Training code is unchanged.

---

## Config additions (`config.py`)

```python
# Loss tuning
CORN_EPSILON: float = 0.05         # label smoothing for CORN (0.0 = off)
CORN_CLASS_WEIGHTED: bool = True   # pass class_weights into CORN

# Calibration (enabled by default for new runs; existing ablation stages opt-out)
CALIBRATE_THRESHOLDS: bool = True
```

Existing ablation runner calls in `run_all.py` pass `calibrate=False` explicitly so
their output JSON remains unchanged. New runs (e.g., transformer stage) use the
default `True`.

---

## Integration points

| File | What changes |
|---|---|
| `models/losses.py` | `CORNLoss` + `KXCLLoss` gain `epsilon` and `class_weights` |
| `evaluation/__init__.py` | Add `calibrate_thresholds` + `apply_thresholds` |
| `train.py` | Compute `class_weights` from train labels; pass to `KXCLLoss.forward`; call calibration after final eval |
| `config.py` | Three new constants |
| `run_all.py` | Pass new config flags through to training calls; surface calibrated QWK in summary |

---

## Backward compatibility

All new parameters default to their current behaviour (`epsilon=0.0`,
`class_weights=None`, `CALIBRATE_THRESHOLDS=False` in the existing ablation
stages). Existing ablation JSON output is unaffected.

---

## What is NOT in scope

- SMOTE or text augmentation  
- Merging bin 0 into bin 1  
- Additional oversampling (WeightedRandomSampler already handles this)  
- Changes to the split protocol (separate issue)
