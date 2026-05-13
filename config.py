"""
KhmerXScore Configuration
=========================
All hyperparameters, paths, and experimental settings.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RAW_CSV      = os.path.join(DATA_DIR, "dataset.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

for d in [PROCESSED_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
NUM_SCORE_CLASSES = 5   # 0, 1, 2, 3, 4

# ── Preprocessing ablation modes ────────────────────────────
PREPROC_RAW      = "raw"
PREPROC_KCC      = "kcc"
PREPROC_KCC_SEG  = "kcc_seg"
PREPROC_FULL     = "kcc_seg_punct"
PREPROC_MODES    = [PREPROC_RAW, PREPROC_KCC, PREPROC_KCC_SEG, PREPROC_FULL]

# ── Input format ablation ────────────────────────────────────
INPUT_AR  = "ar"     # (Answer, Reference) only
INPUT_QAR = "qar"    # (Question, Answer, Reference)
INPUT_FORMATS = [INPUT_AR, INPUT_QAR]

# ── Transformer backbones ────────────────────────────────────
TRANSFORMER_BACKBONES = {
    "mbert":      "bert-base-multilingual-cased",
    "xlmr":       "xlm-roberta-base",
    "gte":        "Alibaba-NLP/gte-multilingual-base",
}
PROPOSED_BACKBONE = "xlmr"   # KX-CL headline model

# ── Training hyperparameters ─────────────────────────────────
@dataclass
class TrainConfig:
    lr: float = 2e-5
    batch_size: int = 16
    max_epochs: int = 30
    early_stop_patience: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    dropout: float = 0.2
    max_seq_len: int = 256
    freeze_layers: int = 6
    num_seeds: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])
    # MC Dropout
    mc_dropout_T: int = 10
    # SCL
    scl_alpha: float = 1.0
    scl_beta: float = 0.5
    scl_temperature: float = 0.07
    scl_margin_scale: float = 1.0
    # Class-balanced sampling
    use_class_balanced: bool = True
    # Loss: "mse", "weighted_mse", "corn"
    loss_type: str = "corn"
    # Class-imbalance: inverse-frequency weighting inside CORN sub-tasks
    corn_class_weighted: bool = True
    # Label smoothing for CORN binary targets (0.0 = off)
    corn_epsilon: float = 0.05

TRAIN_CFG = TrainConfig()

# Post-training QWK threshold calibration (val-set optimised bin boundaries)
CALIBRATE_THRESHOLDS: bool = True

# ── BiLSTM-specific config ───────────────────────────────────
@dataclass
class BiLSTMConfig:
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    max_seq_len: int = 256
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 30
    early_stop_patience: int = 5
    max_vocab: int = 5000
    scl_beta: float = 0.5

BILSTM_CFG = BiLSTMConfig()

# ── Evaluation ───────────────────────────────────────────────
DEPLOYMENT_QWK_THRESHOLD = 0.70
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95
SIGNIFICANCE_ALPHA = 0.05

# ── XAI ──────────────────────────────────────────────────────
SHAP_N_SAMPLES = 50
IG_N_STEPS = 50
FAITHFULNESS_TOP_K = [5, 10, 15, 20]

# ── Hardware ─────────────────────────────────────────────────
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0    # Set to 4+ on HPC
