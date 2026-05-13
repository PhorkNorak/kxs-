# KX-CL System Flow
### Khmer Short Answer Grading — How It Works

---

## Overview

```
Raw CSV
  │
  ├─► Preprocessing  →  Tokenization  →  Dataset
  │
  ├─► Model: Dual Encoder + CORN Head
  │         ↑ trained with CORN Loss + SCL Loss
  │
  └─► Inference: MC Dropout → Score + Uncertainty → Selective Prediction
```

---

## 1. Data Flow

### Input (one sample)
```
Question  : "អ្វីទៅជាការផ្លាស់ប្តូររបស់ ..."
Answer    : "ការផ្លាស់ប្តូរគឺ ..."           ← student wrote this
Reference : "ការផ្លាស់ប្តូរគឺមានន័យថា ..."   ← correct answer
Score     : 3                                ← human label (0–4)
```

### Two input formats tested

```
AR  format:   [Answer]            vs  [Reference]
QAR format:   [Question + Answer] vs  [Reference]
```

### Preprocessing (Khmer-specific)
```
Raw text
  │
  ├─ KCC segmentation     →  insert word boundaries
  ├─ Punctuation cleanup  →  normalize Khmer punctuation
  └─ Whitespace strip     →  remove extra spaces

Output: clean text string, ready for tokenizer
```

### Tokenization & Batching
```
clean_answer    → Tokenizer → input_ids_A  [batch, 256]
                               attention_mask_A [batch, 256]

clean_reference → Tokenizer → input_ids_R  [batch, 256]
                               attention_mask_R [batch, 256]

score_label     →              labels  [batch]   integer 0–4
norm_score      →              scores  [batch]   float   0.0–1.0
```

### Dataset split
```
1,187 samples
  ├─ Train   831  (70%)  ← model learns from this
  ├─ Val     178  (15%)  ← early stopping (patience = 5 epochs)
  └─ Test    178  (15%)  ← final evaluation only, never seen during training
```

Class-balanced sampler on train: over-samples rare scores (0,1) so each batch has a more even distribution.

---

## 2. Model Architecture

### Dual Encoder (Siamese)
```
input_ids_A  ──►  Encoder  ──►  last_hidden_state [batch, 256, 768]
attention_mask_A ──┘              │
                            Mean Pooling
                                  │
                                e_A  [batch, 768]


input_ids_R  ──►  Encoder  ──►  last_hidden_state [batch, 256, 768]
attention_mask_R ──┘    ↑         │
                   same weights  Mean Pooling
                   (shared)       │
                                e_R  [batch, 768]
```

Same encoder processes both answer and reference — this is the "Siamese" structure. Weights are shared so the two embeddings live in the same semantic space.

### 4-way Interaction (InferSent-style)
```
e_A  [768]
e_R  [768]
        │
        ▼
[ e_A | e_R | e_A - e_R | e_A ⊙ e_R ]    concat
        │
        ▼
      [3072]     (768 × 4)
```

Four combinations capture: absolute values of both, their difference, and their element-wise product. Together they let the head learn whether the answer matches, misses, or contradicts the reference.

### Scoring Head (CORN)
```
[3072]
  │
  Dropout(0.2)
  Linear(3072 → 256)
  ReLU
  Dropout(0.2)
  Linear(256 → 4)          ← 4 outputs, not 5
  │
  ▼
logits  [batch, 4]          one logit per threshold boundary
```

CORN (Conditional Ordinal Regression) uses K-1 = 4 logits for K=5 classes. Each logit models one threshold: *P(score > k | score > k-1)*. This respects the ordinal nature of scores — score 3 is between 2 and 4, not arbitrary.

### Projector (for SCL only, used during training)
```
e_A  [768]
  │
  Linear(768 → 256)
  ReLU
  Linear(256 → 128)
  │
  ▼
z_A  [128]    ← used only in SCL loss, not in scoring
```

---

## 3. Training Flow

### One training step
```
Batch (16 samples)
  │
  ▼
Forward pass
  ├─ encode(A) → e_A
  ├─ encode(R) → e_R
  ├─ interact  → [e_A ; e_R ; e_A-e_R ; e_A⊙e_R]
  ├─ head      → logits  [16, 4]
  └─ projector → z_A     [16, 128]    (for SCL)
  │
  ▼
Loss computation
  ├─ L_CORN  =  CORNLoss(logits, labels, class_weights)
  └─ L_SCL   =  SCLoss(z_A, scores)
  │
  L_total = 1.0 × L_CORN + 0.5 × L_SCL
  │
  ▼
Backward + AdamW optimizer step
  └─ gradient clipping at 1.0
  │
  ▼
Repeat for all 52 batches → end of epoch
  └─ evaluate on val set → track QWK
     └─ if best QWK so far → save checkpoint
```

### CORN Loss — how it works
```
For K=5 classes, 4 binary sub-tasks:

  sub-task 0:  Is score > 0 ?   (is it not zero?)
  sub-task 1:  Is score > 1 ?   (given score > 0)
  sub-task 2:  Is score > 2 ?   (given score > 1)
  sub-task 3:  Is score > 3 ?   (given score > 2)

Each is binary cross-entropy.
Class weights applied per sample to handle imbalance.
Label smoothing = 0.05 (softens hard 0/1 targets).

Final loss = mean of 4 sub-task losses.
```

### SCL Loss — how it works
```
Within a batch of 16 answers:

  For each pair (i, j):
    diff = | score_i - score_j |

    if diff < 0.1  →  "positive pair"   (similar scores)
    if diff ≥ 0.1  →  "negative pair"   (different scores)

  Negative pairs are weighted by (1 + diff):
    answers scored very differently push further apart.

  Objective: pull positive pairs together, push negatives apart.
  Uses temperature = 0.07 (sharp distribution).
```

In short: SCL shapes the embedding space so that answers with similar scores cluster together. This makes the head's job easier and improves generalisation.

### Training schedule
```
Epoch 1 ──── warmup (lr: 0 → 2e-5, first 10% of steps)
         ──── linear decay (lr: 2e-5 → 0)
...
Epoch N ──── val QWK improves? save checkpoint
         ──── no improvement for 5 epochs? stop early
```

---

## 4. Inference Flow

### Standard inference (single prediction)
```
New answer + reference
  │
  ▼
encode(A) → e_A
encode(R) → e_R
interact  → combined vector
head      → logits  [4]
  │
  ▼
CORN decoding:
  cum_prob = cumprod( sigmoid(logits) )
  predicted_label = count of cum_prob values > 0.5

  Example:
    logits     = [2.1,  1.3, -0.4, -1.8]
    sigmoid    = [0.89, 0.79, 0.40, 0.14]
    cumprod    = [0.89, 0.70, 0.28, 0.04]
    > 0.5 ?    = [ ✓    ✓    ✗    ✗  ]
    count      =  2  →  predicted score = 2
```

### Uncertainty inference (MC Dropout, T=10)
```
Same answer + reference, 10 times:
  (dropout is ON during all 10 passes)

  pass 1 → score = 2
  pass 2 → score = 2
  pass 3 → score = 3
  pass 4 → score = 2
  pass 5 → score = 2
  pass 6 → score = 3
  pass 7 → score = 2
  pass 8 → score = 2
  pass 9 → score = 3
  pass 10→ score = 2

  mean_score  = 2.2   ← final prediction
  std_score   = 0.40  ← uncertainty
```

High std → model is unsure → flag for human review.
Low std  → model is confident → auto-grade.

### Selective prediction
```
Grade all 178 test answers, compute uncertainty for each.

Sort by uncertainty (low → high):

  Rank 1   (most confident)  uncertainty = 0.02  → auto-grade ✓
  Rank 2                     uncertainty = 0.03  → auto-grade ✓
  ...
  Rank 137 (77% threshold)   uncertainty = 0.11  → auto-grade ✓
  Rank 138                   uncertainty = 0.12  → defer to teacher ✗
  ...
  Rank 178 (least confident) uncertainty = 0.48  → defer to teacher ✗

  Result: 137 auto-graded (QWK = 0.824)
          41  sent to teacher
```

---

## 5. Evaluation

### QWK — the main metric
```
Predicted:  [2, 4, 3, 1, 4, 2, ...]
True:       [2, 4, 2, 1, 4, 3, ...]

QWK weights disagreements by how far apart the scores are:
  off by 0 → weight 0  (no penalty)
  off by 1 → weight 1  (small penalty)
  off by 2 → weight 4  (large penalty)
  off by 3 → weight 9
  off by 4 → weight 16

Range: -1 (worse than random) to 1 (perfect).
Above 0.60 is generally considered acceptable for automated grading.
```

### All metrics computed
```
QWK              ← main, weighted agreement
RMSE             ← distance from true score
MAE              ← mean absolute error
Accuracy         ← exact match rate
Adjacent Agr.    ← within 1 score of truth
Pearson / Spearman ← correlation with true score
F1 per class     ← per-score precision/recall
Per-subject QWK  ← History, Geography, Biology
Confusion matrix ← full N×N breakdown
```

---

## 6. One-Page System Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                          │
│  Question (optional) + Student Answer + Reference Answer        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    Khmer preprocessing
                    (KCC segmentation)
                            │
            ┌───────────────┴───────────────┐
            │                               │
      Tokenize Answer               Tokenize Reference
      max 256 tokens                max 256 tokens
            │                               │
     ┌──────▼──────┐               ┌────────▼──────┐
     │  Encoder A  │               │  Encoder B    │
     │  (GTE /     │               │  (same weights│
     │   XLM-R /   │               │   shared)     │
     │   mBERT)    │               │               │
     └──────┬──────┘               └───────┬───────┘
            │  mean pool                   │  mean pool
            e_A [768]                      e_R [768]
            │                               │
            └───────────┬───────────────────┘
                        │
            [ e_A | e_R | e_A-e_R | e_A⊙e_R ]  [3072]
                        │
                  MLP scoring head
                        │
                  CORN logits [4]
                        │
            ┌───────────┴──────────────┐
            │                          │
     single pass                  10× MC Dropout
            │                          │
     Predicted Score             Score + Uncertainty
         0 – 4                    0–4   +  std
                                        │
                              low std → auto-grade
                             high std → defer to teacher
```

---

*2026-05-13*
