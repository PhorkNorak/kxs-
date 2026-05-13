"""
Loss Functions: CORN ordinal regression, Score-Aware Contrastive Learning, MSE variants.

References:
    Shi, Cao & Raschka (2023) — CORN
    Zha et al. (2022) — Supervised Contrastive Regression
    Khosla et al. (2020) — Supervised Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CORNLoss(nn.Module):
    """Conditional Ordinal Regression for Neural Networks.

    Args:
        num_classes: number of ordinal score bins (default 5).
        epsilon: label-smoothing factor in [0, 1). 0.0 = no smoothing.
    """
    def __init__(self, num_classes=5, epsilon=0.0):
        super().__init__()
        self.K = num_classes
        self.epsilon = epsilon

    def forward(self, logits, labels, class_weights=None):
        """
        Args:
            logits:        [B, K-1] raw threshold logits
            labels:        [B] integer class labels in [0, K-1]
            class_weights: optional [K] inverse-frequency tensor; upweights
                           minority classes by weighting each sample by its
                           class's inverse frequency.
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


def corn_logits_to_label(logits):
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(dim=1)


def corn_logits_to_score(logits, num_classes=5):
    return corn_logits_to_label(logits).float() / (num_classes - 1)


class ScoreAwareContrastiveLoss(nn.Module):
    """Score-distance-weighted contrastive loss on embeddings."""
    def __init__(self, temperature=0.07, score_threshold=0.1, margin_scale=1.0):
        super().__init__()
        self.T = temperature
        self.thr = score_threshold
        self.ms = margin_scale

    def forward(self, embeddings, scores):
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        emb = F.normalize(embeddings, dim=1)
        sim = torch.matmul(emb, emb.T) / self.T
        diff = torch.abs(scores.unsqueeze(1) - scores.unsqueeze(0))
        pos = (diff < self.thr).float()
        eye = torch.eye(B, device=emb.device)
        pos = pos * (1 - eye)
        neg_w = (1.0 + self.ms * diff) * (1 - pos) * (1 - eye)
        mx, _ = sim.max(dim=1, keepdim=True)
        logits = sim - mx.detach()
        exp = torch.exp(logits) * neg_w
        log_prob = logits - torch.log(exp.sum(1, keepdim=True) + 1e-8)
        npos = pos.sum(1).clamp(min=1)
        return (-(pos * log_prob).sum(1) / npos).mean()


class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.cw = class_weights

    def forward(self, pred, target, labels=None):
        mse = (pred - target) ** 2
        if self.cw is not None and labels is not None:
            mse = mse * self.cw.to(pred.device)[labels]
        return mse.mean()


class KXCLLoss(nn.Module):
    """Joint: L = α·L_CORN + β·L_SCL

    Args:
        epsilon: label-smoothing forwarded to CORNLoss (0.0 = off).
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


def compute_class_weights(labels, num_classes=5):
    counts = np.maximum(np.bincount(labels, minlength=num_classes).astype(np.float32), 1)
    return torch.from_numpy(len(labels) / (num_classes * counts))
