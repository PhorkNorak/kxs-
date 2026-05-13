import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.losses import CORNLoss, KXCLLoss, compute_class_weights


# ── helpers ──────────────────────────────────────────────────

def make_corn_inputs():
    torch.manual_seed(0)
    logits = torch.randn(8, 4)           # batch=8, K-1=4 thresholds
    labels = torch.tensor([0, 1, 1, 2, 2, 3, 4, 4])
    return logits, labels


def make_kxcl_inputs():
    torch.manual_seed(1)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 1, 2, 2, 3, 4, 4])
    emb    = torch.randn(8, 64)
    scores = labels.float() / 4.0
    return logits, labels, emb, scores


# ── CORNLoss ─────────────────────────────────────────────────

def test_corn_epsilon_zero_unchanged():
    """epsilon=0 must reproduce original behaviour (no smoothing)."""
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
    weights = compute_class_weights(labels.numpy())
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

def test_kxcl_epsilon_changes_loss():
    logits, labels, emb, scores = make_kxcl_inputs()
    l0, _ = KXCLLoss(5, epsilon=0.0)(logits, labels, emb, scores)
    l1, _ = KXCLLoss(5, epsilon=0.1)(logits, labels, emb, scores)
    assert not torch.isclose(l0, l1, atol=1e-6)


def test_kxcl_class_weights_forwarded():
    """class_weights passed to KXCLLoss.forward must change the loss."""
    logits, labels, emb, scores = make_kxcl_inputs()
    weights = compute_class_weights(labels.numpy())
    l_no, _ = KXCLLoss(5)(logits, labels, emb, scores)
    l_w,  _ = KXCLLoss(5)(logits, labels, emb, scores, class_weights=weights)
    assert not torch.isclose(l_no, l_w, atol=1e-6)


def test_kxcl_returns_breakdown_dict():
    logits, labels, emb, scores = make_kxcl_inputs()
    total, bd = KXCLLoss(5)(logits, labels, emb, scores)
    assert set(bd.keys()) == {"corn", "scl", "total"}
    assert total.item() > 0
