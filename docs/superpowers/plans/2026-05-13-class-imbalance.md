# Class Imbalance Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add class-weighted CORN loss, ordinal label smoothing, and QWK threshold calibration to close the three gaps that let the bin-4-dominant label distribution degrade model performance.

**Architecture:** Three independent improvements wired together: (1) `CORNLoss` gains `epsilon` smoothing and per-sample class weights; (2) `KXCLLoss` passes both through; (3) a post-training calibration step in `evaluation/__init__.py` finds optimal bin thresholds on the val set and applies them at test time. Config constants gate the new behaviour; all defaults preserve backward compatibility.

**Tech Stack:** PyTorch, SciPy (Nelder-Mead), NumPy, scikit-learn, pytest

---

## File Map

| File | Action | What changes |
|---|---|---|
| `models/losses.py` | Modify | `CORNLoss`: add `epsilon`, `class_weights` arg; `KXCLLoss`: pass both through |
| `evaluation/__init__.py` | Modify | Add `calibrate_thresholds` + `apply_thresholds` |
| `config.py` | Modify | Add `CORN_EPSILON`, `CORN_CLASS_WEIGHTED` to `TrainConfig`; add `CALIBRATE_THRESHOLDS` top-level |
| `train.py` | Modify | Pre-compute class weights from train_loader; pass to `KXCLLoss.forward` each batch |
| `run_all.py` | Modify | `stage_transformers`: collect val preds, calibrate thresholds, store `calibrated_qwk` in result dict |
| `tests/test_losses.py` | Create | Unit tests for CORNLoss and KXCLLoss changes |
| `tests/test_calibration.py` | Create | Unit tests for calibration functions |

---

### Task 1: Class-weighted CORN with label smoothing

**Files:**
- Modify: `models/losses.py:16-29`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Create failing tests**

Create `tests/__init__.py` (empty) and `tests/test_losses.py`:

```python
import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.losses import CORNLoss, KXCLLoss, compute_class_weights


# ── CORNLoss ──────────────────────────────────────────────────

def make_corn_inputs():
    torch.manual_seed(0)
    logits = torch.randn(8, 4)           # batch=8, K-1=4 thresholds
    labels = torch.tensor([0, 1, 1, 2, 2, 3, 4, 4])
    return logits, labels


def test_corn_epsilon_zero_unchanged():
    """epsilon=0 must reproduce original behaviour."""
    logits, labels = make_corn_inputs()
    loss_orig = CORNLoss(5, epsilon=0.0)(logits, labels)
    loss_new  = CORNLoss(5)(logits, labels)          # default must still be 0
    assert torch.isclose(loss_orig, loss_new, atol=1e-6)


def test_corn_epsilon_changes_loss():
    """Smoothing must produce a different value than no smoothing."""
    logits, labels = make_corn_inputs()
    l0 = CORNLoss(5, epsilon=0.0)(logits, labels)
    l1 = CORNLoss(5, epsilon=0.1)(logits, labels)
    assert not torch.isclose(l0, l1, atol=1e-6)


def test_corn_epsilon_positive():
    """Loss with smoothing must still be positive."""
    logits, labels = make_corn_inputs()
    loss = CORNLoss(5, epsilon=0.1)(logits, labels)
    assert loss.item() > 0


def test_corn_class_weights_none_unchanged():
    """Passing class_weights=None must give the same result as before."""
    logits, labels = make_corn_inputs()
    loss_no_w = CORNLoss(5)(logits, labels)
    loss_none = CORNLoss(5)(logits, labels, class_weights=None)
    assert torch.isclose(loss_no_w, loss_none, atol=1e-6)


def test_corn_class_weights_changes_loss():
    """Non-uniform class weights must change the loss value."""
    logits, labels = make_corn_inputs()
    weights = compute_class_weights(labels.numpy())   # shape [5]
    l_unweighted = CORNLoss(5)(logits, labels)
    l_weighted   = CORNLoss(5)(logits, labels, class_weights=weights)
    assert not torch.isclose(l_unweighted, l_weighted, atol=1e-6)


def test_corn_class_weights_positive():
    """Weighted loss must still be a positive scalar."""
    logits, labels = make_corn_inputs()
    weights = compute_class_weights(labels.numpy())
    loss = CORNLoss(5)(logits, labels, class_weights=weights)
    assert loss.item() > 0


# ── KXCLLoss ─────────────────────────────────────────────────

def make_kxcl_inputs():
    torch.manual_seed(1)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 1, 2, 2, 3, 4, 4])
    emb    = torch.randn(8, 64)
    scores = labels.float() / 4.0
    return logits, labels, emb, scores


def test_kxcl_epsilon_changes_loss():
    logits, labels, emb, scores = make_kxcl_inputs()
    l0, _ = KXCLLoss(5, epsilon=0.0)(logits, labels, emb, scores)
    l1, _ = KXCLLoss(5, epsilon=0.1)(logits, labels, emb, scores)
    assert not torch.isclose(l0, l1, atol=1e-6)


def test_kxcl_class_weights_forwarded():
    """class_weights passed to KXCLLoss.forward must change the loss."""
    logits, labels, emb, scores = make_kxcl_inputs()
    weights = compute_class_weights(labels.numpy())
    l_no, _  = KXCLLoss(5)(logits, labels, emb, scores)
    l_w, _   = KXCLLoss(5)(logits, labels, emb, scores, class_weights=weights)
    assert not torch.isclose(l_no, l_w, atol=1e-6)


def test_kxcl_returns_breakdown_dict():
    logits, labels, emb, scores = make_kxcl_inputs()
    total, bd = KXCLLoss(5)(logits, labels, emb, scores)
    assert set(bd.keys()) == {"corn", "scl", "total"}
    assert total.item() > 0
```

- [ ] **Step 2: Run tests — verify they fail**

```
python -m pytest tests/test_losses.py -v
```

Expected: `FAILED` on `test_corn_epsilon_changes_loss`, `test_corn_class_weights_changes_loss`, `test_corn_class_weights_none_unchanged` (because the new signature doesn't exist yet), and the KXCLLoss tests.

- [ ] **Step 3: Implement changes in `models/losses.py`**

Replace the entire `CORNLoss` class (lines 16–29):

```python
class CORNLoss(nn.Module):
    """Conditional Ordinal Regression for Neural Networks.

    Args:
        num_classes: number of ordinal score bins (default 5).
        epsilon: label-smoothing factor in [0, 1). 0 = no smoothing.
    """
    def __init__(self, num_classes=5, epsilon=0.0):
        super().__init__()
        self.K = num_classes
        self.epsilon = epsilon

    def forward(self, logits, labels, class_weights=None):
        """
        Args:
            logits:        [B, K-1] raw threshold logits
            labels:        [B] integer class labels
            class_weights: optional [num_classes] inverse-frequency tensor
        """
        losses = []
        for k in range(self.K - 1):
            mask = labels >= k
            if mask.sum() == 0:
                continue
            bl = (labels[mask] > k).float()
            if self.epsilon > 0:
                bl = bl * (1.0 - self.epsilon) + (1.0 - bl) * self.epsilon
            if class_weights is not None:
                w = class_weights.to(logits.device)[labels[mask]]
                losses.append(
                    F.binary_cross_entropy_with_logits(logits[mask, k], bl, weight=w)
                )
            else:
                losses.append(
                    F.binary_cross_entropy_with_logits(logits[mask, k], bl)
                )
        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()
```

Replace the entire `KXCLLoss` class (lines 83–97):

```python
class KXCLLoss(nn.Module):
    """Joint: L = α·L_CORN + β·L_SCL

    Args:
        epsilon: label-smoothing forwarded to CORNLoss (default 0.0 = off).
    """
    def __init__(self, num_classes=5, alpha=1.0, beta=0.5,
                 temperature=0.07, margin_scale=1.0, epsilon=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.corn = CORNLoss(num_classes, epsilon=epsilon)
        self.scl = ScoreAwareContrastiveLoss(temperature, margin_scale=margin_scale)

    def forward(self, logits, labels, embeddings, scores, class_weights=None):
        l_corn = self.corn(logits, labels, class_weights=class_weights)
        l_scl  = self.scl(embeddings, scores)
        total  = self.alpha * l_corn + self.beta * l_scl
        return total, {"corn": l_corn.item(), "scl": l_scl.item(), "total": total.item()}
```

- [ ] **Step 4: Run tests — verify they pass**

```
python -m pytest tests/test_losses.py -v
```

Expected: all 11 tests PASS.

---

### Task 2: Threshold calibration functions

**Files:**
- Modify: `evaluation/__init__.py` (append at end of file)
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration.py`:

```python
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import calibrate_thresholds, apply_thresholds


def synthetic_data(n=200, seed=7):
    """Scores in [0,1] with matching 5-bin labels (uniform bins)."""
    rng = np.random.RandomState(seed)
    scores = rng.uniform(0, 1, n)
    labels = np.floor(scores * 5).astype(int).clip(0, 4)
    return scores, labels


def test_calibrate_returns_four_thresholds():
    scores, labels = synthetic_data()
    t = calibrate_thresholds(scores, labels)
    assert t.shape == (4,)


def test_calibrate_thresholds_sorted():
    scores, labels = synthetic_data()
    t = calibrate_thresholds(scores, labels)
    assert np.all(np.diff(t) >= 0), f"Thresholds not sorted: {t}"


def test_calibrate_thresholds_in_unit_interval():
    scores, labels = synthetic_data()
    t = calibrate_thresholds(scores, labels)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_apply_thresholds_clips_below_zero():
    thresholds = np.array([0.2, 0.4, 0.6, 0.8])
    preds = apply_thresholds(np.array([-0.5, 0.0, 0.1]), thresholds)
    assert all(p >= 0 for p in preds)


def test_apply_thresholds_clips_above_max_class():
    thresholds = np.array([0.2, 0.4, 0.6, 0.8])
    preds = apply_thresholds(np.array([0.9, 1.0, 1.5]), thresholds)
    assert all(p <= 4 for p in preds)


def test_apply_thresholds_correct_binning():
    thresholds = np.array([0.25, 0.50, 0.75, 0.90])
    scores = np.array([0.0, 0.30, 0.60, 0.80, 0.95])
    preds = apply_thresholds(scores, thresholds)
    assert list(preds) == [0, 1, 2, 3, 4]


def test_calibrate_then_apply_noop_on_uniform_data():
    """On perfectly uniform data, calibrated QWK should be >= uncalibrated."""
    from sklearn.metrics import cohen_kappa_score
    scores, labels = synthetic_data()
    # split into val / test halves
    val_s, val_l = scores[:100], labels[:100]
    tst_s, tst_l = scores[100:], labels[100:]
    t = calibrate_thresholds(val_s, val_l)
    cal_preds   = apply_thresholds(tst_s, t)
    fixed_preds = np.floor(tst_s * 5).astype(int).clip(0, 4)
    qwk_cal   = cohen_kappa_score(tst_l, cal_preds,   weights="quadratic")
    qwk_fixed = cohen_kappa_score(tst_l, fixed_preds, weights="quadratic")
    assert qwk_cal >= qwk_fixed - 0.05   # calibrated should not be much worse
```

- [ ] **Step 2: Run tests — verify they fail**

```
python -m pytest tests/test_calibration.py -v
```

Expected: `ImportError` on `calibrate_thresholds` (not yet defined).

- [ ] **Step 3: Append calibration functions to `evaluation/__init__.py`**

Add after the last function (`format_results`) at the end of the file:

```python
def calibrate_thresholds(val_scores, val_labels, n_classes=5):
    """Find bin boundaries that maximise QWK on val_scores / val_labels.

    Uses Nelder-Mead over the interior of [0, 1]^(n_classes-1).

    Returns:
        thresholds: sorted np.ndarray of shape [n_classes - 1]
    """
    from scipy.optimize import minimize
    val_scores = np.asarray(val_scores, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    def neg_qwk(t):
        t_sorted = np.sort(t)
        preds = np.digitize(val_scores, t_sorted).clip(0, n_classes - 1)
        try:
            return -cohen_kappa_score(val_labels, preds, weights="quadratic")
        except Exception:
            return 0.0

    init = np.linspace(0.15, 0.85, n_classes - 1)
    result = minimize(neg_qwk, init, method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 10000})
    return np.sort(result.x).clip(0.0, 1.0)


def apply_thresholds(scores, thresholds, n_classes=5):
    """Apply calibrated bin boundaries to continuous scores.

    Args:
        scores:     array-like of continuous values in [0, 1]
        thresholds: sorted array of shape [n_classes - 1]
        n_classes:  total number of bins (default 5)

    Returns:
        integer label array clipped to [0, n_classes - 1]
    """
    return np.digitize(
        np.asarray(scores, dtype=float),
        np.sort(thresholds)
    ).clip(0, n_classes - 1)
```

- [ ] **Step 4: Run tests — verify they pass**

```
python -m pytest tests/test_calibration.py -v
```

Expected: all 8 tests PASS.

---

### Task 3: Config additions

**Files:**
- Modify: `config.py:48-72` (`TrainConfig` dataclass) and end of file

- [ ] **Step 1: Add fields to `TrainConfig` and a top-level constant**

In `config.py`, inside the `TrainConfig` dataclass (after `loss_type: str = "corn"`, line 71), add:

```python
    # Class-imbalance: weighted CORN + label smoothing
    corn_class_weighted: bool = True
    corn_epsilon: float = 0.05
```

After `TRAIN_CFG = TrainConfig()` at line 73, add:

```python
# Post-training threshold calibration
CALIBRATE_THRESHOLDS: bool = True
```

- [ ] **Step 2: Verify the config loads without error**

```
python -c "from config import TRAIN_CFG, CALIBRATE_THRESHOLDS; print(TRAIN_CFG.corn_epsilon, TRAIN_CFG.corn_class_weighted, CALIBRATE_THRESHOLDS)"
```

Expected output: `0.05 True True`

---

### Task 4: Wire class weights into `train.py`

**Files:**
- Modify: `train.py:14-16` (imports) and `train.py:31-45` (loss construction block) and `train.py:63-65` (forward call in training loop)

- [ ] **Step 1: Update imports at the top of `train.py`**

The line currently reads:
```python
from models.losses import (CORNLoss, KXCLLoss, WeightedMSELoss,
                            corn_logits_to_label, corn_logits_to_score, compute_class_weights)
```

No import change needed — `compute_class_weights` is already imported.

- [ ] **Step 2: Precompute class weights and pass epsilon into the loss**

In `train_transformer()`, replace the loss construction block (lines 31–45):

```python
    # Pre-compute class weights from the full training set (one pass, done once)
    _all_labels = np.concatenate([b["label"].numpy() for b in train_loader])
    class_weights = compute_class_weights(_all_labels).to(device)

    use_scl = config.loss_type == "corn" and config.scl_beta > 0
    if config.loss_type == "corn":
        epsilon = getattr(config, "corn_epsilon", 0.0)
        if use_scl:
            criterion = KXCLLoss(5, config.scl_alpha, config.scl_beta,
                                 config.scl_temperature, config.scl_margin_scale,
                                 epsilon=epsilon)
        else:
            criterion = CORNLoss(5, epsilon=epsilon)
    elif config.loss_type == "weighted_mse":
        criterion = WeightedMSELoss(class_weights)
        class_weights = None   # already baked in; don't double-weight
    else:
        criterion = nn.MSELoss()
        class_weights = None
```

- [ ] **Step 3: Pass class weights to CORNLoss / KXCLLoss in the training loop**

Replace the forward block inside the epoch loop (lines 63–73):

```python
            optimizer.zero_grad()
            use_cw = class_weights is not None and getattr(config, "corn_class_weighted", True)
            if use_scl:
                logits, emb = model(ids_a, mask_a, ids_r, mask_r, return_embeddings=True)
                loss, _ = criterion(logits, labels, emb, scores,
                                    class_weights=class_weights if use_cw else None)
            else:
                logits = model(ids_a, mask_a, ids_r, mask_r)
                if config.loss_type == "corn":
                    loss = criterion(logits, labels,
                                     class_weights=class_weights if use_cw else None)
                elif config.loss_type == "weighted_mse":
                    loss = criterion(logits.squeeze(-1), scores, labels)
                else:
                    loss = criterion(logits.squeeze(-1), scores)
```

- [ ] **Step 4: Verify the training module imports cleanly**

```
python -c "import train; print('train.py OK')"
```

Expected: `train.py OK` (no errors).

---

### Task 5: Wire calibration into `run_all.py`

**Files:**
- Modify: `run_all.py:33-35` (imports) and `run_all.py:220-256` (`stage_transformers`)

- [ ] **Step 1: Add calibration imports in `run_all.py`**

Find the evaluation import line (currently line 33–35):
```python
from evaluation import (compute_all_metrics, bootstrap_qwk_ci, paired_bootstrap_test,
                         selective_prediction_analysis, compute_per_subject,
                         format_results, compute_comprehensiveness, compute_sufficiency)
```

Replace with:
```python
from evaluation import (compute_all_metrics, bootstrap_qwk_ci, paired_bootstrap_test,
                         selective_prediction_analysis, compute_per_subject,
                         format_results, compute_comprehensiveness, compute_sufficiency,
                         calibrate_thresholds, apply_thresholds)
```

- [ ] **Step 2: Add calibration step inside `stage_transformers`**

Find the block inside the `stage_transformers` loop that runs test predictions and stores results. Currently it looks like:

```python
                tl, pl, ts, ps = collect_preds(model, tel, TRAIN_CFG.loss_type)
                m = compute_all_metrics(tl, pl, ts, ps)
                qp, ql, qh = bootstrap_qwk_ci(tl, pl, BOOTSTRAP_N)
                m.update({"qwk_ci_95": [ql, qh], "best_val_qwk": tr_res["best_val_qwk"],
                          "seed": seed, "backbone": bname, "input_format": fmt,
                          "per_subject": compute_per_subject(te, pl, ps)})
```

Replace with:

```python
                # Test predictions
                tl, pl, ts, ps = collect_preds(model, tel, TRAIN_CFG.loss_type)
                m = compute_all_metrics(tl, pl, ts, ps)
                qp, ql, qh = bootstrap_qwk_ci(tl, pl, BOOTSTRAP_N)
                m.update({"qwk_ci_95": [ql, qh], "best_val_qwk": tr_res["best_val_qwk"],
                          "seed": seed, "backbone": bname, "input_format": fmt,
                          "per_subject": compute_per_subject(te, pl, ps)})

                # QWK threshold calibration (val → test)
                if CALIBRATE_THRESHOLDS:
                    vl_tl, vl_pl, vl_ts, vl_ps = collect_preds(model, vl, TRAIN_CFG.loss_type)
                    thresholds = calibrate_thresholds(vl_ps, vl_tl)
                    pl_cal = apply_thresholds(ps, thresholds)
                    m_cal  = compute_all_metrics(tl, pl_cal, ts, ps)
                    m["calibrated_qwk"]        = m_cal["qwk"]
                    m["calibrated_thresholds"] = thresholds.tolist()
                    print(f"    Calibrated QWK: {m_cal['qwk']:.4f}  (raw: {m['qwk']:.4f})")
```

- [ ] **Step 3: Verify run_all.py imports cleanly**

```
python -c "import run_all; print('run_all.py OK')"
```

Expected: `run_all.py OK`

---

### Task 6: Full test suite

- [ ] **Step 1: Run all tests**

```
python -m pytest tests/ -v
```

Expected: all 19 tests PASS, no warnings about missing functions or imports.

- [ ] **Step 2: Smoke-test the full config + training path (CPU, 2 epochs)**

```python
# paste into a Python REPL from the project root
import numpy as np, torch
from config import TRAIN_CFG, CALIBRATE_THRESHOLDS
from models.losses import KXCLLoss, CORNLoss, compute_class_weights

# Simulate a mini-batch
labels  = torch.tensor([0, 1, 2, 3, 4, 1, 2, 4])
logits  = torch.randn(8, 4)
emb     = torch.randn(8, 64)
scores  = labels.float() / 4.0
weights = compute_class_weights(labels.numpy())

# CORN + SCL with epsilon + class weights
criterion = KXCLLoss(5, epsilon=TRAIN_CFG.corn_epsilon)
total, bd = criterion(logits, labels, emb, scores, class_weights=weights)
print(f"total={total.item():.4f}  corn={bd['corn']:.4f}  scl={bd['scl']:.4f}")
assert total.item() > 0

# Calibration smoke test
from evaluation import calibrate_thresholds, apply_thresholds
val_s = np.random.rand(50)
val_l = np.floor(val_s * 5).astype(int).clip(0, 4)
t = calibrate_thresholds(val_s, val_l)
preds = apply_thresholds(val_s, t)
print(f"thresholds={t.round(3)}  pred range=[{preds.min()},{preds.max()}]")
assert CALIBRATE_THRESHOLDS is True
print("Smoke test passed.")
```

Expected:
```
total=<positive float>  corn=<positive float>  scl=<positive float>
thresholds=[...]  pred range=[0,4]
Smoke test passed.
```
