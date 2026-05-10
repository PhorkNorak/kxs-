"""
KhmerXScore Loss Functions
===========================
- CORN ordinal regression (Shi, Cao & Raschka 2023)
- Score-Aware Contrastive Learning (adapted from Zha et al. 2022 SupCR)
- Standard MSE and Weighted MSE for ablation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# CORN Loss (Conditional Ordinal Regression for Neural Networks)
# ============================================================
class CORNLoss(nn.Module):
    """
    CORN loss for ordinal regression.
    Trains K-1 binary classifiers on conditional subsets.
    
    Reference:
        Shi, Cao & Raschka (2023). "Deep Neural Networks for Rank-Consistent 
        Ordinal Regression Based On Conditional Probabilities."
        Pattern Analysis and Applications.
    """
    
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes - 1) raw logits from CORN output head
            labels: (batch,) integer labels in [0, num_classes-1]
        Returns:
            Scalar loss
        """
        num_classes = self.num_classes
        sets = []
        
        for i in range(num_classes - 1):
            # Conditional subset: samples with label >= i
            mask = labels >= i
            if mask.sum() == 0:
                continue
            
            # Binary labels: 1 if label > i, 0 if label == i
            binary_labels = (labels[mask] > i).float()
            task_logits = logits[mask, i]
            
            loss_i = F.binary_cross_entropy_with_logits(task_logits, binary_labels)
            sets.append(loss_i)
        
        if not sets:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return torch.stack(sets).mean()


def corn_logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits to predicted ordinal labels.
    Uses chain rule: P(Y > k) = P(Y > k | Y >= k) * P(Y >= k)
    """
    probas = torch.sigmoid(logits)
    # Cumulative product for unconditional probabilities
    # P(Y > 0), P(Y > 1), ..., P(Y > K-2)
    cum_probs = torch.cumprod(probas, dim=1)
    
    # P(Y = k) = P(Y > k-1) - P(Y > k)
    # P(Y = 0) = 1 - P(Y > 0)
    ones = torch.ones(cum_probs.shape[0], 1, device=logits.device)
    extended = torch.cat([ones, cum_probs], dim=1)
    
    # Predicted label = sum of P(Y > k) > 0.5 for each k
    predicted = (cum_probs > 0.5).sum(dim=1)
    return predicted


def corn_logits_to_score(logits: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert CORN logits to continuous score (0.0-1.0)."""
    labels = corn_logits_to_label(logits)
    return labels.float() / (num_classes - 1)


# ============================================================
# Score-Aware Contrastive Learning (SCL)
# ============================================================
class ScoreAwareContrastiveLoss(nn.Module):
    """
    Contrastive loss weighted by score distance.
    
    Pairs of samples with similar scores are pulled together;
    pairs with different scores are pushed apart, with the
    repulsion strength proportional to their score difference.
    
    Adapted from:
        Zha et al. (2022). "Supervised Contrastive Regression." NeurIPS.
        Khosla et al. (2020). "Supervised Contrastive Learning." NeurIPS.
    """
    
    def __init__(self, temperature: float = 0.07, margin_scale: float = 1.0,
                 score_threshold: float = 0.1):
        """
        Args:
            temperature: Scaling factor for similarity
            margin_scale: Scale for score-distance-based weighting
            score_threshold: Scores within this distance are "same class"
        """
        super().__init__()
        self.temperature = temperature
        self.margin_scale = margin_scale
        self.score_threshold = score_threshold
    
    def forward(self, embeddings: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, dim) L2-normalized embeddings
            scores: (batch,) continuous scores in [0, 1]
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Pairwise cosine similarity
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Score distance matrix
        score_diff = torch.abs(scores.unsqueeze(1) - scores.unsqueeze(0))
        
        # Positive mask: pairs with similar scores (within threshold)
        pos_mask = (score_diff < self.score_threshold).float()
        
        # Remove self-pairs
        identity = torch.eye(batch_size, device=embeddings.device)
        pos_mask = pos_mask * (1 - identity)
        
        # Weight negatives by score distance (larger distance = stronger push)
        neg_weight = 1.0 + self.margin_scale * score_diff
        neg_weight = neg_weight * (1 - pos_mask) * (1 - identity)
        
        # Log-sum-exp trick for numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Weighted negative log-sum-exp
        exp_logits = torch.exp(logits) * neg_weight
        neg_log_sum = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Positive pairs log-prob
        log_prob = logits - neg_log_sum
        
        # Mean log-prob over positive pairs
        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_count
        loss = -mean_log_prob.mean()
        
        return loss


# ============================================================
# Standard losses for ablation
# ============================================================
class WeightedMSELoss(nn.Module):
    """MSE weighted by inverse class frequency."""
    
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        mse = (pred - target) ** 2
        if self.class_weights is not None and labels is not None:
            weights = self.class_weights[labels]
            mse = mse * weights
        return mse.mean()


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """Compute inverse-frequency weights for each class."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1)  # Avoid div by zero
    weights = len(labels) / (num_classes * counts)
    return torch.from_numpy(weights)


# ============================================================
# Joint loss: CORN + SCL
# ============================================================
class KXCLLoss(nn.Module):
    """
    Joint training objective for KX-CL:
    L = α · L_CORN + β · L_SCL
    """
    
    def __init__(self, num_classes: int = 5, alpha: float = 1.0, beta: float = 0.5,
                 temperature: float = 0.07, margin_scale: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.corn_loss = CORNLoss(num_classes)
        self.scl_loss = ScoreAwareContrastiveLoss(temperature, margin_scale)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                embeddings: torch.Tensor, scores: torch.Tensor):
        """
        Args:
            logits: (batch, num_classes-1) CORN logits
            labels: (batch,) integer labels 0-4
            embeddings: (batch, dim) from the encoder
            scores: (batch,) continuous scores 0-1
        """
        l_corn = self.corn_loss(logits, labels)
        l_scl = self.scl_loss(embeddings, scores)
        
        total = self.alpha * l_corn + self.beta * l_scl
        
        return total, {"corn": l_corn.item(), "scl": l_scl.item(), "total": total.item()}
