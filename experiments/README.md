# Experiments registry

Each script in this directory implements one independent enhancement of the
TF-IDF baseline. All scripts share the same metric schema as `run_all.py`:

```
train_*  (qwk, accuracy, adjacent_accuracy, mae,
          raw_exact, raw_within1, raw_mae, pct_mae)
test_*   (same eight)
val_qwk, best_epoch, seconds
```

## Versioning scheme

| Version | Script | Output dir per dataset | Effort | What it does |
|---|---|---|---|---|
| **v01** | `exp01_tfidf_baseline.py` | `results_<ds>/` | low | TF-IDF Cosine + SVR baseline (12 cells per dataset) |
| **v02** | `exp02_threshold_calibration.py` | `results_<ds>_v02_calibrated/` | very low | Re-calibrates the 4 bin boundaries on val to maximize val QWK, then applies to test. Only the 5-class metrics change; raw-score metrics are unchanged. |
| **v03** | `exp03_maxscore_feature.py` | `results_<ds>_v03_maxfeat/` | low | TF-IDF+SVR with `max_score` concatenated to the feature vector. SVR sees the per-question scoring scale explicitly. |
| **v03b** | `exp03b_maxfeat_neural.py` | `results_<ds>_v03b_maxfeat_neural/` | low/medium | Neural variant of v03 — BiLSTM + transformers with `max_score` concatenated to the head input. BiLSTM CPU-feasible; transformer cells HPC only. |
| **v04** | `exp04_bucket_svr.py` | `results_<ds>_v04_bucket/` | medium | Trains a separate TF-IDF+SVR per `max_score` bucket. Routes test answers by max_score to the matching model. |
| **v05** | `exp05_bilstm.py` | `results_<ds>_v05_bilstm/` | low | Full BiLSTM+Attention grid (12 cells per dataset). **HPC ONLY** — slow on CPU (~30-40 min total). |
| **v06** | `exp06_transformer.py` | `results_<ds>_v06_transformer/` | medium | Full transformer grid (36 cells per dataset). **HPC ONLY** — needs HF network + GPU (~6-10 hours total). |
| **v07** | `exp07_ensemble.py` | `results_<ds>_v07_ensemble/` | low | Reads the top-3 cells across v01-v06 by test QWK, takes weighted mean of `pred_score`, recomputes metrics. |

## Dataset variants (in `experiments/_common.py::DATASETS`)

| run_name | dataset CSV | DROP_SCORE_ZERO | rows |
|---|---|---|---|
| `full`      | `dataset.csv`                 | False | 1,184 |
| `no10c`     | `dataset_no_10c_biology.csv`  | False |   909 |
| `no10c_no0` | `dataset_no_10c_biology.csv`  | True  |   895 |

Every experiment runs on **all three** datasets by default. Use the
`--datasets` CLI flag on any experiment script to target a subset.

```powershell
python experiments/exp01_tfidf_baseline.py                    # all 3
python experiments/exp01_tfidf_baseline.py --datasets full    # 1184 only
python experiments/exp01_tfidf_baseline.py --datasets no10c no10c_no0
```

The legacy 1184-dataset leaderboard from the original `kxs/run_all.py`
pipeline is preserved at `results_v0_legacy/` (old 12-column schema, NOT part
of the v01-v07 sequence).

## Conventions

- Each experiment writes a per-cell `runs/<run_id>/` directory with:
  - `config.json`         (what was trained)
  - `metrics.json`        ({train, val, test, best_epoch})
  - `predictions_train.csv`, `predictions_val.csv`, `predictions_test.csv`
- Each experiment writes a `leaderboard.csv` with the same 24-column schema.
- Experiments do **not** modify each other's output directories.
- Re-running an experiment overwrites its own output cleanly (deletes leaderboard before re-running).

## How to run

Each script iterates over **all 3 datasets** sequentially by default, patching
`config.RAW_CSV`, `DROP_SCORE_ZERO`, and the output paths between datasets.

```powershell
# 1. Baseline (needed before anything else)
python experiments/exp01_tfidf_baseline.py

# 2-4. Classical enhancements (fast, CPU)
python experiments/exp02_threshold_calibration.py
python experiments/exp03_maxscore_feature.py
python experiments/exp04_bucket_svr.py

# 5-6. Neural (HPC only)
python experiments/exp05_bilstm.py
python experiments/exp06_transformer.py
# Neural variant of v03 (max_score as numeric input to BiLSTM + transformer heads):
python experiments/exp03b_maxfeat_neural.py
# Optionally calibrate any of the above too:
python experiments/exp02_threshold_calibration.py --source v05_bilstm
python experiments/exp02_threshold_calibration.py --source v06_transformer
python experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural

# 7. Ensemble (run AFTER any upstream you want included in the pool)
python experiments/exp07_ensemble.py

# Compare everything
python experiments/compare_all.py --topk 10
```

Order matters only for `exp07_ensemble.py` — it reads the leaderboards of
v01-v06 to pick the top-3 cells, so run all upstream experiments first.
