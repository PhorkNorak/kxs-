# KhmerXScore (KX-CL)
### End-to-End Pipeline & Results

---

## The Problem

> A teacher gives students a question. Students write short answers in **Khmer**. The teacher must grade each answer 0–4. We want a system to do this automatically.

**Why it is hard:** Khmer is a low-resource language. No prior ASAG system exists for Khmer. The dataset is small (~1,187 samples across 4 subjects).

---

## End-to-End Pipeline

```
RAW DATA
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1 — DATASET                                   │
│                                                     │
│  1,187 student answers                              │
│  4 subjects: History, Geography, Biology,           │
│              Earth Science                          │
│  Each answer has: Question + Student Answer +       │
│                   Reference Answer + Score (0–4)    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2 — PREPROCESSING                             │
│                                                     │
│  KCC segmentation (Khmer word boundaries)           │
│  Punctuation normalization                          │
│  Two input formats:                                 │
│    • AR  → [Answer] vs [Reference]                  │
│    • QAR → [Question + Answer] vs [Reference]       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3 — SPLIT                                     │
│                                                     │
│  Train   831 samples  (70%)                         │
│  Val     178 samples  (15%)   ← early stopping      │
│  Test    178 samples  (15%)   ← final evaluation    │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────┐
│  STEP 4A         │    │  STEP 4B                     │
│  CLASSICAL       │    │  DEEP LEARNING               │
│  BASELINES       │    │  MODELS                      │
└────────┬─────────┘    └──────────────┬───────────────┘
         │                             │
         ▼                             ▼
┌──────────────────┐    ┌──────────────────────────────┐
│ TF-IDF cosine    │    │  BiLSTM                      │
│ TF-IDF + SVR  ◄──┼────┼── best non-transformer       │
│ FastText cosine  │    │                              │
└──────────────────┘    │  mBERT     (12-layer)        │
                        │  GTE    ◄──── best so far    │
                        │  mDeBERTa  (collapsed)       │
                        │  XLM-R     (pending)         │
                        └──────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 5 — KX-CL FRAMEWORK  (proposed system)        │
│                                                     │
│  Encoder A ──┐                                      │
│  (shared)    ├──► [eA ; eR ; eA-eR ; eA×eR]        │
│  Encoder B ──┘         │                            │
│                        ▼                            │
│               CORN Loss + SCL Loss                  │
│               (ordinal regression +                 │
│                score-aware contrastive)             │
│                        │                            │
│                        ▼                            │
│               Predicted Score (0–4)                 │
│             + Uncertainty (MC Dropout)              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 6 — EVALUATION                                │
│                                                     │
│  Primary metric: QWK (Quadratic Weighted Kappa)     │
│    → standard for ordinal grading tasks             │
│    → 0.0 = random,  1.0 = perfect agreement         │
│                                                     │
│  Also: RMSE, Accuracy, Adjacent Agreement,          │
│        Per-subject QWK, Per-class F1                │
└─────────────────────────────────────────────────────┘
```

---

## Results at Each Step

### Step 4A — Classical Baselines

| Model | What it does | QWK |
|-------|-------------|-----|
| Mean predictor | Always predicts average score | 0.000 |
| FastText cosine | Word vectors, cosine similarity | 0.007 |
| TF-IDF cosine | Bag-of-words overlap | 0.468 |
| **TF-IDF + SVR** | **Bag-of-words + support vector regression** | **0.767** |

> TF-IDF+SVR is surprisingly strong. It becomes the bar to beat.

---

### Step 4B — Deep Learning Models

| Model | QWK | Note |
|-------|-----|------|
| BiLSTM | 0.628 | Better than cosine, worse than SVR |
| mBERT (KX-CL) | 0.682 | Solid multilingual transformer |
| **GTE (KX-CL)** | **0.755** | **Best transformer so far** |
| mDeBERTa (KX-CL) | 0.000 | Collapsed — needs fixing |
| XLM-R (KX-CL) | *pending* | Proposed headline model |

---

### Step 5 — What KX-CL Adds Beyond Raw QWK

**Uncertainty-aware grading (MC Dropout):**

The model runs 10 stochastic forward passes and measures how much predictions vary → a confidence score per answer.

```
Low uncertainty  →  Model is confident  →  Grade automatically
High uncertainty →  Model is unsure     →  Send to teacher
```

**Selective prediction results (GTE, test set n=178):**

| Auto-grade | Defer to teacher | QWK on auto-graded |
|-----------|-----------------|-------------------|
| 100% (178) | 0% (0) | 0.758 |
| 90% (160) | 10% (18) | 0.802 |
| 77% (137) | 23% (41) | **0.824** |
| 61% (109) | 39% (69) | 0.873 |

> Deferring just 23% of answers (the uncertain ones) to a teacher raises QWK from 0.758 to 0.824. This is the practical argument for KX-CL over TF-IDF+SVR — the SVR cannot identify which answers to defer.

---

### Step 6 — Where the Model Struggles

**By score class** (GTE, 178 test samples):

```
Score 4 (full credit)    ████████████████████  F1 = 0.879  n=75  ✓ Easy
Score 0 (no credit)      ████████████████████  F1 = 0.800  n=2   (too few)
Score 2 (partial)        ████████████████      F1 = 0.608  n=49
Score 1 (minimal)        ██████████████        F1 = 0.558  n=23
Score 3 (mostly correct) █████████             F1 = 0.369  n=29  ✗ Hard
```

The Score 3 / Score 2 boundary is the hardest: 15 of 29 score-3 answers are misclassified as score 2. This boundary ("mostly correct" vs "partially correct") is semantically difficult even for human raters.

**By subject** (GTE, best seed):

```
History      ████████████████████████  QWK = 0.878  n=72  ✓ Strong
Geography    ███████████████████       QWK = 0.708  n=49
Biology      ██████████                QWK = 0.371  n=56  ✗ Weak
Earth Sci.   —                         QWK = 0.000  n=1   (not evaluable)
```

Biology is a significant weak spot. Technical Khmer scientific vocabulary is likely underrepresented in all pre-trained models.

---

## Summary: Where We Are

```
Mean predictor   0.000  ─────────────────────────────────────────
FastText         0.007
                          │
TF-IDF cosine    0.468  ──┤  Classical gap
                          │
BiLSTM           0.628  ──┤  Deep learning gap
mBERT            0.682    │
                          │
TF-IDF + SVR     0.767  ──┼── Target to beat
GTE (KX-CL)      0.755  ──┘  ← best transformer, nearly tied
                          
XLM-R (KX-CL)   ????   ← must run — this is the paper's model
```

**The central open question:**
GTE matches but does not clearly beat TF-IDF+SVR in raw QWK. The case for KX-CL rests on uncertainty estimation and selective prediction — capabilities SVR cannot provide. Whether XLM-R (the proposed model) opens a clear gap remains to be seen.

---

## What is Still Running / Pending

| Task | Status |
|------|--------|
| Classical baselines | Done |
| BiLSTM | Done |
| mBERT (6 runs) | Done |
| GTE AR (3 seeds) | Done |
| GTE QAR (3 seeds) | 1 of 3 done |
| mDeBERTa | Needs re-run (collapsed) |
| **XLM-R** | **Not started — highest priority** |
| XAI analysis (SHAP, IG) | Not started |

---

*2026-05-13*
