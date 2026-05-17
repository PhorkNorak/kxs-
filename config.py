"""Simple-pipeline config — single source of truth for paths, models, hparams."""

import os
import torch

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
# RAW_CSV       = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
RAW_CSV       = os.path.join(PROJECT_ROOT, "data", "dataset_no_10c_biology.csv")

# Drop rows whose ordinal score_label is 0 (only 14 samples in the corpus,
# practically untrainable). Set False to keep the original 5-class task.
DROP_SCORE_ZERO = False

# Output directory is versioned by dataset variant so results from different
# experiments don't overwrite each other. Change RUN_NAME when switching
# datasets / score filters.
RUN_NAME      = "no10c"            # v2: 909 rows (no 10C biology, keeps score=0)
RESULTS_DIR   = os.path.join(PROJECT_ROOT, f"results_{RUN_NAME}")
RUNS_DIR      = os.path.join(RESULTS_DIR, "runs")
LEADERBOARD   = os.path.join(RESULTS_DIR, "leaderboard.csv")
XAI_DIR       = os.path.join(PROJECT_ROOT, f"xai_visuals_{RUN_NAME}")

for d in [RESULTS_DIR, RUNS_DIR, XAI_DIR]:
    os.makedirs(d, exist_ok=True)

# Grid axes
PREPROC_MODES = ["raw", "clean", "segment"]
INPUT_FORMATS = ["ra", "qar"]

# Denominator used when feeding max_score as a scalar input feature to
# neural heads (v03b). 20 is the largest observed Max Score in the corpus,
# so max_score_feat = Max Score / 20.0 lands in (0, 1].
MAX_SCORE_NORMALIZER = 20.0

TRANSFORMER_BACKBONES = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr":  "xlm-roberta-base",
    "gte":   "Alibaba-NLP/gte-multilingual-base",
}

# Model registry: (model_id, family, backbone_or_None)
MODELS = [
    ("tfidf_cos",    "classical", None),
    ("tfidf_svr",    "classical", None),
    ("fasttext_cos", "classical", None),
    ("bilstm",       "bilstm",    None),
    ("dual_mbert",   "dual",      "mbert"),
    ("dual_xlmr",    "dual",      "xlmr"),
    ("dual_gte",     "dual",      "gte"),
    ("cross_mbert",  "cross",     "mbert"),
    ("cross_xlmr",   "cross",     "xlmr"),
    ("cross_gte",    "cross",     "gte"),
]

# Hyperparameters
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Classical
TFIDF_NGRAMS   = (2, 4)
TFIDF_ANALYZER = "char_wb"
TFIDF_MAX_FEAT = 15000
SVR_C          = 1.0
FASTTEXT_DIM   = 100
FASTTEXT_EPOCHS = 10

# BiLSTM
BILSTM_HIDDEN  = 128
BILSTM_LAYERS  = 2
BILSTM_EMBED   = 128
BILSTM_VOCAB   = 5000
BILSTM_LR      = 1e-3
BILSTM_BATCH   = 64
BILSTM_MAX_EP  = 20
BILSTM_PATIENCE = 4
BILSTM_DROPOUT = 0.3

# Transformer
TXFMR_LR        = 2e-5
TXFMR_BATCH     = 16
TXFMR_MAX_EP    = 20
TXFMR_PATIENCE  = 4
TXFMR_DROPOUT   = 0.2
TXFMR_MAX_LEN   = 256
TXFMR_FREEZE_N  = 6
TXFMR_WEIGHT_DECAY = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SCORE_CLASSES = 5
