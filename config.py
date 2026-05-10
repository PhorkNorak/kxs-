"""
KhmerXScore Configuration
=========================
All hyperparameters, paths, and experimental settings in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "dataset.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

for d in [PROCESSED_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Data
# ============================================================
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_SCORE_CLASSES = 5  # 0, 1, 2, 3, 4

# ============================================================
# Preprocessing ablation modes
# ============================================================
PREPROC_RAW = "raw"                        # No preprocessing
PREPROC_KCC = "kcc"                        # KCC normalization only
PREPROC_KCC_SEG = "kcc_seg"                # KCC + segmentation
PREPROC_FULL = "kcc_seg_punct"             # KCC + segmentation + punctuation strip
PREPROC_MODES = [PREPROC_RAW, PREPROC_KCC, PREPROC_KCC_SEG, PREPROC_FULL]

# ============================================================
# Input format ablation
# ============================================================
INPUT_AR = "ar"          # (Answer, Reference) only
INPUT_QAR = "qar"        # (Question, Answer, Reference)
INPUT_FORMATS = [INPUT_AR, INPUT_QAR]

# ============================================================
# Model registry
# ============================================================
# Baselines
BASELINE_MODELS = [
    "mean_predictor",
    "tfidf_cosine",
    "tfidf_svr",
    "fasttext_cosine",
    "bilstm_attention",
]

# Transformer backbones for dual-encoder
TRANSFORMER_BACKBONES = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "prahokbart": "KIST-AIP/PrahokBART",
    "gte": "Alibaba-NLP/gte-multilingual-base",
}
PROPOSED_BACKBONE = "xlmr"  # KX-CL headline model

# LLM scorers
LLM_MODELS = ["gpt4", "claude", "gemini"]

# ============================================================
# Training hyperparameters
# ============================================================
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
    freeze_layers: int = 6       # Freeze first N encoder layers
    num_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 101])
    # MC Dropout
    mc_dropout_T: int = 10       # Forward passes for uncertainty
    # SCL
    scl_alpha: float = 1.0       # Weight for scoring loss
    scl_beta: float = 0.5        # Weight for contrastive loss
    scl_temperature: float = 0.07
    scl_margin_scale: float = 1.0
    # Class-balanced sampling
    use_class_balanced: bool = True
    # Loss type: "mse", "weighted_mse", "corn"
    loss_type: str = "corn"

TRAIN_CFG = TrainConfig()

# ============================================================
# Evaluation
# ============================================================
DEPLOYMENT_QWK_THRESHOLD = 0.70
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95
SIGNIFICANCE_ALPHA = 0.05

# ============================================================
# XAI
# ============================================================
SHAP_N_SAMPLES = 50        # Background samples for GradientSHAP
IG_N_STEPS = 50             # Steps for Integrated Gradients
FAITHFULNESS_TOP_K = [5, 10, 15, 20]  # Top-k tokens for comprehensiveness/sufficiency

# ============================================================
# LLM API keys (load from environment)
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ============================================================
# Hardware
# ============================================================
DEVICE = "cuda"  # Override with "cpu" if no GPU
NUM_WORKERS = 4
