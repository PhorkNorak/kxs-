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
    val_s, val_l = scores[:100], labels[:100]
    tst_s, tst_l = scores[100:], labels[100:]
    t = calibrate_thresholds(val_s, val_l)
    cal_preds   = apply_thresholds(tst_s, t)
    fixed_preds = np.floor(tst_s * 5).astype(int).clip(0, 4)
    qwk_cal   = cohen_kappa_score(tst_l, cal_preds,   weights="quadratic")
    qwk_fixed = cohen_kappa_score(tst_l, fixed_preds, weights="quadratic")
    assert qwk_cal >= qwk_fixed - 0.05
