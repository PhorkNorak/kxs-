"""
KhmerXScore Transformer Models
================================
- DualEncoder: Siamese dual-encoder with 4-way interaction (KX-CL proposed model)
- CrossEncoder: Cross-encoder ablation for comparison
Both support CORN ordinal regression and MC Dropout.

Architecture references:
- Dual-encoder 4-way: Conneau et al. 2017 (InferSent); Reimers & Gurevych 2019 (SBERT)
- CORN output: Shi, Cao & Raschka 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# ============================================================
# Dual-Encoder with 4-Way Interaction (Proposed: KX-CL)
# ============================================================
class DualEncoder(nn.Module):
    """
    Siamese dual-encoder with shared weights.
    
    Tower A: encodes [Q + A] (or just A for ar input format)
    Tower R: encodes R
    
    Interaction: [e_A; e_R; e_A - e_R; e_A ⊙ e_R] → 4 * hidden_dim
    Prediction: MLP → CORN ordinal logits
    
    This is the KX-CL architecture when trained with CORN + SCL joint loss.
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base",
                 num_classes: int = 5, dropout: float = 0.2,
                 freeze_layers: int = 6, loss_type: str = "corn"):
        super().__init__()
        
        self.loss_type = loss_type
        self.num_classes = num_classes
        
        # Shared encoder
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_dim = config.hidden_size  # 768 for base models
        
        # Freeze first N layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Interaction dimension: 4 * hidden_dim
        interaction_dim = 4 * self.hidden_dim  # 3072 for base
        
        # MLP prediction head
        if loss_type == "corn":
            output_dim = num_classes - 1  # K-1 binary classifiers
        else:
            output_dim = 1  # Regression
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )
        
        # Projection for SCL (optional, used when SCL is enabled)
        self.scl_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def _freeze_layers(self, n: int):
        """Freeze the first n encoder layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first n layers
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "layers"):
            layers = self.encoder.layers
        else:
            return
        
        for i, layer in enumerate(layers):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _pool(self, hidden_states: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over non-padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a single text through the shared encoder."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(outputs.last_hidden_state, attention_mask)
    
    def forward(self, input_ids_a, attention_mask_a,
                input_ids_r, attention_mask_r,
                return_embeddings: bool = False):
        """
        Forward pass.
        
        Returns:
            logits: (batch, num_classes-1) for CORN, or (batch, 1) for regression
            embeddings_a: (batch, hidden_dim) if return_embeddings=True
        """
        # Encode both sides with shared weights
        e_a = self.encode(input_ids_a, attention_mask_a)   # (batch, 768)
        e_r = self.encode(input_ids_r, attention_mask_r)   # (batch, 768)
        
        # 4-way interaction (Conneau et al. 2017; Mou et al. 2016)
        interaction = torch.cat([
            e_a,
            e_r,
            e_a - e_r,             # Element-wise difference
            e_a * e_r,             # Element-wise product
        ], dim=1)  # (batch, 3072)
        
        # Prediction
        logits = self.head(interaction)
        
        if return_embeddings:
            # Project embeddings for SCL
            scl_emb = self.scl_projector(e_a)
            return logits, scl_emb
        
        return logits
    
    def predict_score(self, input_ids_a, attention_mask_a,
                      input_ids_r, attention_mask_r) -> torch.Tensor:
        """Predict continuous score (0.0-1.0)."""
        logits = self.forward(input_ids_a, attention_mask_a,
                              input_ids_r, attention_mask_r)
        
        if self.loss_type == "corn":
            from models.losses import corn_logits_to_score
            return corn_logits_to_score(logits, self.num_classes)
        else:
            return logits.squeeze(-1).clamp(0, 1)
    
    def predict_label(self, input_ids_a, attention_mask_a,
                      input_ids_r, attention_mask_r) -> torch.Tensor:
        """Predict discrete label (0-4)."""
        logits = self.forward(input_ids_a, attention_mask_a,
                              input_ids_r, attention_mask_r)
        
        if self.loss_type == "corn":
            from models.losses import corn_logits_to_label
            return corn_logits_to_label(logits)
        else:
            score = logits.squeeze(-1).clamp(0, 1)
            return (score * 4).round().long().clamp(0, 4)
    
    def predict_with_uncertainty(self, input_ids_a, attention_mask_a,
                                 input_ids_r, attention_mask_r,
                                 T: int = 10) -> dict:
        """
        MC Dropout prediction with uncertainty estimation.
        (Gal & Ghahramani 2016)
        
        Keep dropout ON during inference, run T forward passes,
        report mean prediction and variance.
        """
        self.train()  # Enable dropout
        
        scores_list = []
        labels_list = []
        
        with torch.no_grad():
            for _ in range(T):
                logits = self.forward(input_ids_a, attention_mask_a,
                                      input_ids_r, attention_mask_r)
                
                if self.loss_type == "corn":
                    from models.losses import corn_logits_to_score, corn_logits_to_label
                    s = corn_logits_to_score(logits, self.num_classes)
                    l = corn_logits_to_label(logits)
                else:
                    s = logits.squeeze(-1).clamp(0, 1)
                    l = (s * 4).round().long().clamp(0, 4)
                
                scores_list.append(s)
                labels_list.append(l)
        
        self.eval()
        
        scores_tensor = torch.stack(scores_list, dim=0)  # (T, batch)
        labels_tensor = torch.stack(labels_list, dim=0)   # (T, batch)
        
        return {
            "mean_score": scores_tensor.mean(dim=0),
            "std_score": scores_tensor.std(dim=0),
            "mean_label": labels_tensor.float().mean(dim=0),
            "mode_label": labels_tensor.mode(dim=0).values,
            "raw_scores": scores_tensor,
        }


# ============================================================
# Cross-Encoder (Ablation comparison)
# ============================================================
class CrossEncoder(nn.Module):
    """
    Cross-encoder: joint encoding of [Q + A] [SEP] [R].
    Single forward pass, no separable embeddings.
    Used as an ablation to compare against dual-encoder.
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base",
                 num_classes: int = 5, dropout: float = 0.2,
                 freeze_layers: int = 6, loss_type: str = "corn"):
        super().__init__()
        
        self.loss_type = loss_type
        self.num_classes = num_classes
        
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_dim = config.hidden_size
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        if loss_type == "corn":
            output_dim = num_classes - 1
        else:
            output_dim = 1
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )
    
    def _freeze_layers(self, n):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        if hasattr(self.encoder, "encoder"):
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < n:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(self, input_ids_a, attention_mask_a,
                input_ids_r, attention_mask_r,
                return_embeddings=False):
        """
        Cross-encoder: concatenate A and R tokens, encode jointly.
        Reuses the same DataLoader interface by concatenating at model level.
        """
        # Concatenate tokens (remove [CLS] from R, keep [SEP])
        # Simple approach: use tokenizer's pair encoding
        # Since we receive pre-tokenized, we concatenate manually
        batch_size = input_ids_a.shape[0]
        seq_len = input_ids_a.shape[1]
        
        # For cross-encoder, we use only tower A's input and
        # concatenate R as a second segment
        # Use [CLS] A_tokens [SEP] R_tokens [SEP]
        combined_ids = torch.cat([input_ids_a, input_ids_r[:, 1:]], dim=1)
        combined_mask = torch.cat([attention_mask_a, attention_mask_r[:, 1:]], dim=1)
        
        # Truncate to max_len
        max_len = min(combined_ids.shape[1], 512)
        combined_ids = combined_ids[:, :max_len]
        combined_mask = combined_mask[:, :max_len]
        
        outputs = self.encoder(input_ids=combined_ids, attention_mask=combined_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_output)
        
        if return_embeddings:
            return logits, cls_output
        return logits
    
    def predict_label(self, input_ids_a, attention_mask_a,
                      input_ids_r, attention_mask_r):
        logits = self.forward(input_ids_a, attention_mask_a,
                              input_ids_r, attention_mask_r)
        if self.loss_type == "corn":
            from models.losses import corn_logits_to_label
            return corn_logits_to_label(logits)
        else:
            score = logits.squeeze(-1).clamp(0, 1)
            return (score * 4).round().long().clamp(0, 4)


# ============================================================
# Factory function
# ============================================================
def create_model(backbone: str = "xlmr", topology: str = "dual",
                 loss_type: str = "corn", num_classes: int = 5,
                 dropout: float = 0.2, freeze_layers: int = 6) -> nn.Module:
    """
    Create a scoring model.
    
    Args:
        backbone: 'mbert', 'xlmr', 'prahokbart', 'gte'
        topology: 'dual' or 'cross'
        loss_type: 'corn', 'mse', 'weighted_mse'
        num_classes: Number of ordinal classes
        dropout: Dropout rate
        freeze_layers: Number of encoder layers to freeze
    """
    from config import TRANSFORMER_BACKBONES
    
    model_name = TRANSFORMER_BACKBONES[backbone]
    
    if topology == "dual":
        return DualEncoder(model_name, num_classes, dropout, freeze_layers, loss_type)
    elif topology == "cross":
        return CrossEncoder(model_name, num_classes, dropout, freeze_layers, loss_type)
    else:
        raise ValueError(f"Unknown topology: {topology}")
