# KX-Simple — Khmer ASAG benchmark

A stripped-down parallel implementation of the Khmer Automatic Short Answer
Grading (ASAG) pipeline. No CORN loss, no SCL, no MC Dropout, no bootstrap
CIs, no faithfulness — just a clean **3 × 2 × 10 = 60-cell factorial grid**
trained as bounded MSE regression with early stopping on validation QWK,
plus a small XAI pass on the top transformer cell.

The headline current result (on the existing `kxs/` pipeline) is a non-neural
TF-IDF + SVR baseline at **QWK ≈ 0.77**. The aim of `simple/` is to find out
which simple recipe — across three preprocessing modes, two input formats,
and ten scorer families — gets us furthest with minimal machinery.

---

## Folder layout

```
simple/
├── README.md                   ← you are here
├── __init__.py
├── config.py                   ← paths, model list, hparams (single source of truth)
├── preprocess.py               ← 3 modes: raw / clean / segment
├── data.py                     ← CSV loader, 70/15/15 split, datasets
├── evaluate.py                 ← 4 metrics: QWK · Acc · Adj-Acc · MAE
├── train.py                    ← train_classical · train_bilstm · train_transformer
├── run_all.py                  ← grid orchestrator + CLI
├── xai.py                      ← gradient × input saliency
├── models/
│   ├── classical.py            ← TF-IDF Cosine · TF-IDF + SVR · FastText Cosine
│   ├── bilstm.py               ← BiLSTM + Attention (char-level)
│   ├── dual.py                 ← Transformer dual-encoder (Siamese, 4-way interaction)
│   └── cross.py                ← Transformer cross-encoder ([CLS] head)
├── data/
│   └── dataset.csv             ← 1,184 graded short answers
├── results/
│   ├── leaderboard.csv         ← appended once per cell, ranked by test QWK
│   └── runs/<run_id>/
│       ├── config.json         ← what was trained
│       ├── metrics.json        ← val + test metrics + epoch history (neural)
│       ├── predictions_val.csv ← per-sample val predictions + abs error
│       ├── predictions_test.csv← per-sample test predictions + abs error
│       └── best.pt             ← best checkpoint (neural runs only)
└── xai_visuals/<best_run_id>/  ← saliency PNG heatmaps
```

---

## The 60-cell grid

```
preprocess (3) × input (2) × model (10) = 60 runs
```

**Preprocess (Φ).** Khmer-aware text normalization, applied to question / reference / answer:

| mode    | what it does                                          |
| ------- | ----------------------------------------------------- |
| `raw`   | strip whitespace only                                 |
| `clean` | KCC syllable normalization + strip punctuation        |
| `segment` | `clean` + `khmernltk.word_tokenize` word segmentation |

**Input (Ψ).** Two-sided pair fed into the scorer:

| format | side A (`x_a`)           | side B (`x_b`) |
| ------ | ------------------------ | -------------- |
| `ra`   | `Answer`                 | `Reference`    |
| `qar`  | `Question + Answer`      | `Reference`    |

**Models (M).** Ten scorer pipelines spanning four architectural classes:

| family            | identifier(s)                                   | encoder                   | scorer head                              |
| ----------------- | ----------------------------------------------- | ------------------------- | ---------------------------------------- |
| Lexical           | `tfidf_cos`, `fasttext_cos`                     | TF-IDF char-2-4gm, FastText | cosine similarity                       |
| Supervised class. | `tfidf_svr`                                     | TF-IDF char-2-4gm         | SVR(RBF) on `[a; b; |a−b|; a⊙b; cos]`    |
| Recurrent neural  | `bilstm`                                        | char BiLSTM + Attention   | 4-way MLP → σ                            |
| Transformer dual  | `dual_mbert`, `dual_xlmr`, `dual_gte`           | mBERT / XLM-R / GTE       | 4-way MLP → σ                            |
| Transformer cross | `cross_mbert`, `cross_xlmr`, `cross_gte`        | mBERT / XLM-R / GTE       | `[CLS]` MLP → σ                          |

GTE ships with NaN entries in its NTK-RoPE cache under some load paths;
`models/dual.py::_patch_rope` rebuilds `cos`/`sin` caches in fp32 to fix this.

---

## Training recipe (single shared)

Same recipe across every trainable model:

- **Target:** `ỹ = score / 4 ∈ [0, 1]` (regression, post-hoc round to `{0..4}` at eval).
- **Loss:** MSE.
- **Optimizer:** AdamW, weight decay 0.01.
- **LR:** `2e-5` (transformer), `1e-3` (BiLSTM).
- **Batch:** 16 (transformer), 64 (BiLSTM).
- **Max epochs:** 20. **Early-stop on val QWK, patience 4.**
- **Mixed precision** when CUDA available.
- **Transformer freeze:** bottom 6 encoder layers.

Classical and cosine-only baselines are not trained iteratively (fit
vectorizer / SVR once, no epochs).

---

## Evaluation

Four metrics, computed on the rounded discrete prediction `ŷ* = round(4·ŷ) ∈ {0..4}`:

| metric              | what it captures                                              |
| ------------------- | ------------------------------------------------------------- |
| `qwk` (primary)     | Quadratic Weighted Kappa — ordinal agreement, magnitude-weighted |
| `accuracy`          | exact-match rate                                              |
| `adjacent_accuracy` | within ±1 score point ("close enough")                        |
| `mae`               | mean absolute error in score units                            |

Per-sample predictions are saved to `predictions_{val,test}.csv` for any
downstream analysis (worst-error inspection, additional metrics, etc.).

---

## XAI

After the 60-cell grid, `xai.py` picks the top transformer cell by test QWK
and produces **Gradient × Input** saliency on the answer-side tokens:

```
α_t = ‖ ∂ŷ / ∂E_t  ⊙  E_t ‖₂
```

It samples ~2 test items per score level (0..4), renders a token-strip PNG
colored white → red by saliency magnitude, and saves to
`simple/xai_visuals/<best_run_id>/sample_NN_trueX_predY.png`.

---

## Quickstart

### Run the full grid (all 60 cells)

```powershell
python -m simple.run_all
```

This trains every cell, appends to `simple/results/leaderboard.csv` sorted
by test QWK, and finally runs the XAI pass.

Rough wall-clock budget: classical cells finish in seconds, BiLSTM cells in
~30 s, transformer cells in 5–30 min each on a single GPU. The full grid is
~10–12 GPU-hours.

### Run a single cell (smoke test)

```powershell
python -m simple.run_all --only segment_ra_tfidf_svr
python -m simple.run_all --only raw_ra_bilstm        --epochs 2 --skip-xai
python -m simple.run_all --only raw_ra_cross_xlmr    --epochs 2 --skip-xai
python -m simple.run_all --only raw_ra_dual_gte      --epochs 2 --skip-xai   # GTE NaN check
```

### Skip XAI

```powershell
python -m simple.run_all --skip-xai
```

### Run XAI on the current top transformer cell

```powershell
python -m simple.xai
```

### Run IDs

`run_id` has the form `"<preprocess>_<input>_<model>"`. Examples:

```
raw_ra_tfidf_cos
clean_ra_tfidf_svr
segment_qar_dual_xlmr
segment_qar_cross_gte
```

---

## Outputs

After running, you'll have:

| Path                                                  | Content                                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------------------- |
| `simple/results/leaderboard.csv`                      | 60 rows × 12 cols, ranked by test QWK. Your headline result.            |
| `simple/results/runs/<run_id>/config.json`            | what was trained (preprocess / input / model / backbone)                |
| `simple/results/runs/<run_id>/metrics.json`           | val + test metric vectors + best epoch + per-epoch history (neural)     |
| `simple/results/runs/<run_id>/predictions_val.csv`    | val: `idx, Q, R, A, true_label, true_score, pred_score, pred_label, abs_error` |
| `simple/results/runs/<run_id>/predictions_test.csv`   | same, on test split                                                     |
| `simple/results/runs/<run_id>/best.pt`                | state-dict at best val QWK (neural runs only)                           |
| `simple/xai_visuals/<best_run_id>/sample_NN_*.png`    | gradient × input saliency heatmaps                                      |

---

## Smoke checks (already verified)

| run                                  | result                                          |
| ------------------------------------ | ----------------------------------------------- |
| `segment_ra_tfidf_svr` (1 epoch)     | test QWK = 0.7699, acc = 0.6798 — wiring OK     |
| `raw_ra_bilstm` (2 epochs)           | non-NaN, loss decreasing, leaderboard append OK |

Transformer cells require network access to download HF backbones
(`bert-base-multilingual-cased`, `xlm-roberta-base`,
`Alibaba-NLP/gte-multilingual-base`). On the development machine an SSL cert
verify failure blocks this (likely a TLS-intercepting proxy). Workarounds:

```powershell
# Option A — install the Windows root-cert bridge:
pip install pip-system-certs

# Option B — explicit certifi bundle:
$env:SSL_CERT_FILE = (python -c "import certifi; print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE

# Option C — corporate proxy:
$env:HTTPS_PROXY = "http://<proxy>:<port>"

# Option D — pre-download on another machine and copy to
#           %USERPROFILE%\.cache\huggingface\hub\
```

---

## Configuration

All knobs live in [`config.py`](config.py). The most relevant ones:

```python
PREPROC_MODES = ["raw", "clean", "segment"]
INPUT_FORMATS = ["ra", "qar"]
MODELS = [ ... 10 entries ... ]

TXFMR_LR        = 2e-5
TXFMR_BATCH     = 16
TXFMR_MAX_EP    = 20
TXFMR_PATIENCE  = 4
TXFMR_FREEZE_N  = 6

BILSTM_LR       = 1e-3
BILSTM_MAX_EP   = 20
BILSTM_PATIENCE = 4
```

To prune the grid (e.g. for a quick sanity sweep), edit `MODELS`,
`PREPROC_MODES`, or `INPUT_FORMATS` in `config.py`.

---

## Honest expectations on QWK 0.95

The current `kxs/` best across all 60+ baselines is **QWK ≈ 0.77** (TF-IDF +
SVR). With only this grid — preprocessing × input × model swap, no
augmentation, no ensembling, no extra loss machinery — getting to **0.95** on
**1,184** samples is unlikely. What you should expect from `simple/`:

- A clean, defensible leaderboard that tells you **which simple recipe wins**.
- A clear answer on whether `qar` conditioning helps over `ra`.
- A clear answer on whether `khmernltk` segmentation moves the needle.
- A reasonable XAI artifact for the top cell.

The 0.95 stretch will likely need a follow-up phase: data augmentation
(back-translation, paraphrase), ensembling the top 2–3 cells, or pretraining
adaptation on a related Khmer corpus.

---

## Design philosophy

Two principles deliberately violated by the larger `kxs/` pipeline that
`simple/` re-asserts:

1. **One recipe per family.** Every trainable model uses the same MSE +
   AdamW + early-stop-on-val-QWK loop. No CORN, no SCL, no MC Dropout,
   no weighted MSE.
2. **Reproducibility before completeness.** Single seed (42), no
   bootstrap CIs, no significance tests. The leaderboard is a ranking
   tool, not a publication-ready confidence-interval table — that comes
   later if a cell is worth promoting.

If a cell on the `simple/` leaderboard looks compelling, the next step is to
promote it back to the full `kxs/` pipeline with multi-seed runs, bootstrap
CIs, MC Dropout uncertainty, and CORN+SCL to see if those bells and whistles
push the headline number further.
