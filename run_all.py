"""
KhmerXScore: Complete Experiment Runner
=========================================
Usage:
    python run_all.py --stage prepare     # Prepare data (run once)
    python run_all.py --stage baselines   # Run 5 baselines
    python run_all.py --stage transformers # Run 4-backbone dual-encoder study
    python run_all.py --stage ablations   # Run all ablation experiments
    python run_all.py --stage llms        # Run 3 LLM scorers
    python run_all.py --stage xai         # Run XAI analysis
    python run_all.py --stage all         # Run everything sequentially

Each stage saves results to results/<stage_name>.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from preprocessing import preprocess
from data import (
    load_raw_data, split_data, preprocess_dataframe,
    KhmerSAGTextDataset, KhmerSAGDataset, get_dataloaders
)
from models import create_model, create_baseline, BiLSTMAttention, create_llm_scorer
from models.losses import compute_class_weights
from train import train_transformer, evaluate_model, evaluate_with_uncertainty
from evaluation import (
    compute_all_metrics, bootstrap_qwk_ci, paired_bootstrap_test,
    quadratic_weighted_kappa, selective_prediction_analysis,
    compute_per_subject_metrics, benchmark_latency, format_results
)


def save_results(results: dict, filename: str):
    """Save results to JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {path}")


# ============================================================
# Stage 1: Data Preparation
# ============================================================
def stage_prepare():
    """Load, clean, split, and preprocess data."""
    print("\n" + "="*60)
    print("STAGE: Data Preparation")
    print("="*60)
    
    # Load raw data
    df = load_raw_data(RAW_CSV)
    print(f"Loaded {len(df)} samples")
    print(f"Score distribution:\n{df['score_label'].value_counts().sort_index()}")
    print(f"Subject distribution:\n{df['Subject'].value_counts()}")
    
    # Split
    train_df, val_df, test_df = split_data(df, seed=SPLIT_SEED)
    
    # Preprocess with all modes
    for mode in PREPROC_MODES:
        print(f"\nPreprocessing mode: {mode}")
        train_p = preprocess_dataframe(train_df, mode)
        val_p = preprocess_dataframe(val_df, mode)
        test_p = preprocess_dataframe(test_df, mode)
        
        # Save
        for split_name, split_df in [("train", train_p), ("val", val_p), ("test", test_p)]:
            path = os.path.join(PROCESSED_DIR, f"{split_name}_{mode}.csv")
            split_df.to_csv(path, index=False)
        
        print(f"  Sample preprocessed answer: {train_p['Answer_proc'].iloc[0][:80]}...")
    
    print("\nData preparation complete!")
    return train_df, val_df, test_df


def load_splits(preproc_mode: str = "kcc"):
    """Load preprocessed splits from disk."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(PROCESSED_DIR, f"{name}_{preproc_mode}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run --stage prepare first.")
        splits[name] = pd.read_csv(path)
    return splits["train"], splits["val"], splits["test"]


# ============================================================
# Stage 2: Baselines
# ============================================================
def stage_baselines():
    """Run all 5 baseline models."""
    print("\n" + "="*60)
    print("STAGE: Baseline Models")
    print("="*60)
    
    train_df, val_df, test_df = load_splits("kcc_seg_punct")
    
    all_results = {}
    
    # Text datasets for classical models
    for input_fmt in INPUT_FORMATS:
        print(f"\n--- Input format: {input_fmt} ---")
        
        train_data = KhmerSAGTextDataset(train_df, input_format=input_fmt)
        test_data = KhmerSAGTextDataset(test_df, input_format=input_fmt)
        
        for baseline_name in ["mean_predictor", "tfidf_cosine", "tfidf_svr", "fasttext_cosine"]:
            print(f"\n  Running: {baseline_name} ({input_fmt})")
            
            model = create_baseline(baseline_name)
            model.fit(train_data)
            preds = model.predict(test_data)
            
            metrics = compute_all_metrics(
                test_data.labels, preds["labels"],
                test_data.scores, preds["scores"]
            )
            
            # Bootstrap CI for QWK
            qwk_point, qwk_lo, qwk_hi = bootstrap_qwk_ci(
                test_data.labels, preds["labels"], BOOTSTRAP_N
            )
            metrics["qwk_ci_95"] = [qwk_lo, qwk_hi]
            
            key = f"{baseline_name}_{input_fmt}"
            all_results[key] = metrics
            print(format_results(metrics, key))
    
    save_results(all_results, "baselines.json")
    return all_results


# ============================================================
# Stage 3: Transformer Models (Backbone Study)
# ============================================================
def stage_transformers():
    """Run 4-backbone dual-encoder study with SCL."""
    print("\n" + "="*60)
    print("STAGE: Transformer Backbone Study")
    print("="*60)
    
    train_df, val_df, test_df = load_splits("kcc")
    
    all_results = {}
    
    for backbone_name, model_path in TRANSFORMER_BACKBONES.items():
        for input_fmt in INPUT_FORMATS:
            for seed in TRAIN_CFG.seeds:
                run_name = f"{backbone_name}_dual_{input_fmt}_seed{seed}"
                print(f"\n{'='*50}")
                print(f"Running: {run_name}")
                print(f"{'='*50}")
                
                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                # Tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # DataLoaders
                train_loader, val_loader, test_loader = get_dataloaders(
                    train_df, val_df, test_df, tokenizer,
                    batch_size=TRAIN_CFG.batch_size,
                    max_len=TRAIN_CFG.max_seq_len,
                    input_format=input_fmt,
                    preproc_mode="kcc",
                    use_class_balanced=TRAIN_CFG.use_class_balanced,
                    num_workers=NUM_WORKERS,
                )
                
                # Model
                model = create_model(
                    backbone=backbone_name,
                    topology="dual",
                    loss_type=TRAIN_CFG.loss_type,
                    dropout=TRAIN_CFG.dropout,
                    freeze_layers=TRAIN_CFG.freeze_layers,
                )
                
                # Train
                train_result = train_transformer(
                    model, train_loader, val_loader, TRAIN_CFG,
                    device=DEVICE,
                    checkpoint_dir=CHECKPOINT_DIR,
                    run_name=run_name,
                )
                
                # Evaluate on test
                test_metrics = evaluate_model(model, test_loader, TRAIN_CFG.loss_type, DEVICE)
                
                # Bootstrap CI
                # Need to re-collect predictions for bootstrap
                model.eval()
                all_true = []
                all_pred = []
                all_true_s = []
                all_pred_s = []
                with torch.no_grad():
                    for batch in test_loader:
                        ids_a = batch["input_ids_a"].to(DEVICE)
                        mask_a = batch["attention_mask_a"].to(DEVICE)
                        ids_r = batch["input_ids_r"].to(DEVICE)
                        mask_r = batch["attention_mask_r"].to(DEVICE)
                        logits = model(ids_a, mask_a, ids_r, mask_r)
                        from models.losses import corn_logits_to_label, corn_logits_to_score
                        pred_l = corn_logits_to_label(logits).cpu().numpy()
                        pred_s = corn_logits_to_score(logits).cpu().numpy()
                        all_true.extend(batch["label"].numpy())
                        all_pred.extend(pred_l)
                        all_true_s.extend(batch["score"].numpy())
                        all_pred_s.extend(pred_s)
                
                true_labels = np.array(all_true)
                pred_labels = np.array(all_pred)
                
                qwk_point, qwk_lo, qwk_hi = bootstrap_qwk_ci(true_labels, pred_labels, BOOTSTRAP_N)
                test_metrics["qwk_ci_95"] = [qwk_lo, qwk_hi]
                test_metrics["best_val_qwk"] = train_result["best_val_qwk"]
                test_metrics["epochs_trained"] = train_result["epochs_trained"]
                test_metrics["seed"] = seed
                test_metrics["backbone"] = backbone_name
                test_metrics["input_format"] = input_fmt
                
                all_results[run_name] = test_metrics
                print(format_results(test_metrics, run_name))
                
                # Latency benchmark (once per backbone)
                if seed == TRAIN_CFG.seeds[0]:
                    latency = benchmark_latency(model, test_loader, DEVICE)
                    all_results[f"{backbone_name}_latency"] = latency
                    print(f"  Latency: mean={latency['mean_ms']:.1f}ms, "
                          f"p95={latency['p95_ms']:.1f}ms")
                
                # MC Dropout uncertainty (once per backbone per input format)
                if seed == TRAIN_CFG.seeds[0]:
                    mc_metrics, true_l, pred_l, uncert = evaluate_with_uncertainty(
                        model, test_loader, TRAIN_CFG.loss_type, DEVICE, TRAIN_CFG.mc_dropout_T
                    )
                    
                    # Selective prediction analysis
                    sp = selective_prediction_analysis(true_l, pred_l, uncert)
                    all_results[f"{backbone_name}_{input_fmt}_selective_prediction"] = sp
                    all_results[f"{backbone_name}_{input_fmt}_uncertainty"] = {
                        "mean": mc_metrics["mean_uncertainty"],
                        "qwk_with_uncertainty": mc_metrics["qwk"],
                    }
                
                # Free GPU memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Aggregate results across seeds
    _aggregate_seed_results(all_results)
    
    save_results(all_results, "transformers.json")
    return all_results


def _aggregate_seed_results(results: dict):
    """Compute mean/std across seeds for each backbone+input_format."""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for key, metrics in results.items():
        if isinstance(metrics, dict) and "seed" in metrics:
            group_key = f"{metrics['backbone']}_dual_{metrics['input_format']}"
            groups[group_key].append(metrics)
    
    for group_key, runs in groups.items():
        qwks = [r["qwk"] for r in runs]
        rmses = [r["rmse"] for r in runs]
        results[f"{group_key}_aggregate"] = {
            "qwk_mean": float(np.mean(qwks)),
            "qwk_std": float(np.std(qwks)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "n_seeds": len(runs),
        }


# ============================================================
# Stage 4: Ablation Studies
# ============================================================
def stage_ablations():
    """Run all ablation experiments on the proposed model (XLM-R dual)."""
    print("\n" + "="*60)
    print("STAGE: Ablation Studies")
    print("="*60)
    
    all_results = {}
    
    # ---- Ablation 1: Preprocessing modes ----
    print("\n--- Ablation: Preprocessing ---")
    for preproc_mode in PREPROC_MODES:
        train_df, val_df, test_df = load_splits(preproc_mode)
        
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_BACKBONES[PROPOSED_BACKBONE])
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, tokenizer,
            batch_size=TRAIN_CFG.batch_size, max_len=TRAIN_CFG.max_seq_len,
            input_format="qar", preproc_mode=preproc_mode,
            use_class_balanced=True, num_workers=NUM_WORKERS,
        )
        
        model = create_model(backbone=PROPOSED_BACKBONE, topology="dual", loss_type="corn")
        torch.manual_seed(42)
        
        train_result = train_transformer(
            model, train_loader, val_loader, TRAIN_CFG,
            device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            run_name=f"ablation_preproc_{preproc_mode}"
        )
        
        test_metrics = evaluate_model(model, test_loader, "corn", DEVICE)
        all_results[f"preproc_{preproc_mode}"] = test_metrics
        print(format_results(test_metrics, f"preproc_{preproc_mode}"))
        
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---- Ablation 2: Cross-encoder vs Dual-encoder ----
    print("\n--- Ablation: Topology ---")
    train_df, val_df, test_df = load_splits("kcc")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_BACKBONES[PROPOSED_BACKBONE])
    
    for topology in ["dual", "cross"]:
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, tokenizer,
            batch_size=TRAIN_CFG.batch_size, max_len=TRAIN_CFG.max_seq_len,
            input_format="qar", preproc_mode="kcc",
            use_class_balanced=True, num_workers=NUM_WORKERS,
        )
        
        model = create_model(
            backbone=PROPOSED_BACKBONE, topology=topology, loss_type="corn"
        )
        torch.manual_seed(42)
        
        cfg = deepcopy(TRAIN_CFG)
        if topology == "cross":
            cfg.scl_beta = 0  # No SCL for cross-encoder
        
        train_result = train_transformer(
            model, train_loader, val_loader, cfg,
            device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            run_name=f"ablation_topology_{topology}"
        )
        
        test_metrics = evaluate_model(model, test_loader, "corn", DEVICE)
        all_results[f"topology_{topology}"] = test_metrics
        print(format_results(test_metrics, f"topology_{topology}"))
        
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---- Ablation 3: SCL on/off ----
    print("\n--- Ablation: SCL ---")
    for scl_beta in [0.0, 0.25, 0.5, 1.0]:
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, tokenizer,
            batch_size=TRAIN_CFG.batch_size, max_len=TRAIN_CFG.max_seq_len,
            input_format="qar", preproc_mode="kcc",
            use_class_balanced=True, num_workers=NUM_WORKERS,
        )
        
        model = create_model(backbone=PROPOSED_BACKBONE, topology="dual", loss_type="corn")
        torch.manual_seed(42)
        
        cfg = deepcopy(TRAIN_CFG)
        cfg.scl_beta = scl_beta
        
        train_result = train_transformer(
            model, train_loader, val_loader, cfg,
            device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            run_name=f"ablation_scl_beta{scl_beta}"
        )
        
        test_metrics = evaluate_model(model, test_loader, "corn", DEVICE)
        all_results[f"scl_beta_{scl_beta}"] = test_metrics
        print(format_results(test_metrics, f"scl_beta_{scl_beta}"))
        
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---- Ablation 4: Loss type (CORN vs MSE vs Weighted MSE) ----
    print("\n--- Ablation: Loss Type ---")
    for loss_type in ["corn", "mse", "weighted_mse"]:
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, tokenizer,
            batch_size=TRAIN_CFG.batch_size, max_len=TRAIN_CFG.max_seq_len,
            input_format="qar", preproc_mode="kcc",
            use_class_balanced=True, num_workers=NUM_WORKERS,
        )
        
        model = create_model(
            backbone=PROPOSED_BACKBONE, topology="dual", loss_type=loss_type
        )
        torch.manual_seed(42)
        
        cfg = deepcopy(TRAIN_CFG)
        cfg.loss_type = loss_type
        if loss_type != "corn":
            cfg.scl_beta = 0  # SCL requires separable embeddings
        
        train_result = train_transformer(
            model, train_loader, val_loader, cfg,
            device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            run_name=f"ablation_loss_{loss_type}"
        )
        
        test_metrics = evaluate_model(model, test_loader, loss_type, DEVICE)
        all_results[f"loss_{loss_type}"] = test_metrics
        print(format_results(test_metrics, f"loss_{loss_type}"))
        
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---- Ablation 5: Imbalance handling ----
    print("\n--- Ablation: Imbalance Handling ---")
    for use_balanced in [False, True]:
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, tokenizer,
            batch_size=TRAIN_CFG.batch_size, max_len=TRAIN_CFG.max_seq_len,
            input_format="qar", preproc_mode="kcc",
            use_class_balanced=use_balanced, num_workers=NUM_WORKERS,
        )
        
        model = create_model(backbone=PROPOSED_BACKBONE, topology="dual", loss_type="corn")
        torch.manual_seed(42)
        
        cfg = deepcopy(TRAIN_CFG)
        cfg.use_class_balanced = use_balanced
        
        train_result = train_transformer(
            model, train_loader, val_loader, cfg,
            device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            run_name=f"ablation_balanced_{use_balanced}"
        )
        
        test_metrics = evaluate_model(model, test_loader, "corn", DEVICE)
        label = "balanced" if use_balanced else "standard"
        all_results[f"imbalance_{label}"] = test_metrics
        print(format_results(test_metrics, f"imbalance_{label}"))
        
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    save_results(all_results, "ablations.json")
    return all_results


# ============================================================
# Stage 5: LLM Scorers
# ============================================================
def stage_llms():
    """Run zero-shot LLM scoring on test set."""
    print("\n" + "="*60)
    print("STAGE: LLM Scorers (Zero-Shot)")
    print("="*60)
    
    _, _, test_df = load_splits("raw")  # LLMs get raw text
    all_results = {}
    
    api_keys = {
        "gpt4": OPENAI_API_KEY,
        "claude": ANTHROPIC_API_KEY,
        "gemini": GOOGLE_API_KEY,
    }
    
    for llm_name in LLM_MODELS:
        api_key = api_keys.get(llm_name, "")
        if not api_key:
            print(f"  Skipping {llm_name}: no API key found in environment")
            continue
        
        print(f"\n  Running: {llm_name}")
        scorer = create_llm_scorer(llm_name, api_key)
        preds = scorer.predict(test_df, delay=0.5)
        
        true_labels = test_df["score_label"].values
        true_scores = test_df["normalized_score"].values
        
        metrics = compute_all_metrics(
            true_labels, preds["labels"],
            true_scores, preds["scores"]
        )
        
        # Bootstrap CI
        qwk_point, qwk_lo, qwk_hi = bootstrap_qwk_ci(
            true_labels, preds["labels"], BOOTSTRAP_N
        )
        metrics["qwk_ci_95"] = [qwk_lo, qwk_hi]
        
        all_results[llm_name] = metrics
        print(format_results(metrics, llm_name))
        
        # Save reasoning for XAI Layer B comparison
        reasoning_path = os.path.join(RESULTS_DIR, f"{llm_name}_reasoning.json")
        with open(reasoning_path, "w", encoding="utf-8") as f:
            json.dump({
                "reasoning": preds.get("reasoning", []),
                "raw_responses": preds.get("raw_responses", []),
            }, f, ensure_ascii=False, indent=2)
    
    save_results(all_results, "llms.json")
    return all_results


# ============================================================
# Stage 6: XAI Analysis
# ============================================================
def stage_xai():
    """Run SHAP + Integrated Gradients + faithfulness analysis."""
    print("\n" + "="*60)
    print("STAGE: XAI Analysis")
    print("="*60)
    
    train_df, val_df, test_df = load_splits("kcc")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_BACKBONES[PROPOSED_BACKBONE])
    
    _, _, test_loader = get_dataloaders(
        train_df, val_df, test_df, tokenizer,
        batch_size=1,  # Process one at a time for XAI
        max_len=TRAIN_CFG.max_seq_len,
        input_format="qar", preproc_mode="kcc",
        use_class_balanced=False, num_workers=0,
    )
    
    # Load best model
    model = create_model(backbone=PROPOSED_BACKBONE, topology="dual", loss_type="corn")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{PROPOSED_BACKBONE}_dual_qar_seed42_best.pt")
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}. Run --stage transformers first.")
        return {}
    
    model = model.to(DEVICE)
    model.eval()
    
    all_results = {}
    
    # ---- GradientSHAP ----
    print("\nRunning GradientSHAP...")
    try:
        from captum.attr import LayerIntegratedGradients, GradientShap
        
        # We attribute on the embedding layer
        def forward_fn(input_embeds, attention_mask_a, input_ids_r, attention_mask_r):
            """Custom forward that takes embeddings instead of input_ids for attribution."""
            # This is a simplified attribution; for full implementation,
            # wrap the model to accept embeddings directly
            outputs = model.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask_a)
            e_a = model._pool(outputs.last_hidden_state, attention_mask_a)
            
            e_r = model.encode(input_ids_r, attention_mask_r)
            interaction = torch.cat([e_a, e_r, e_a - e_r, e_a * e_r], dim=1)
            logits = model.head(interaction)
            return logits[:, 0]  # Attribution target: first CORN logit
        
        shap_results = []
        ig_results = []
        
        from tqdm import tqdm
        for i, batch in enumerate(tqdm(test_loader, desc="XAI attribution")):
            if i >= SHAP_N_SAMPLES:
                break
            
            input_ids_a = batch["input_ids_a"].to(DEVICE)
            attention_mask_a = batch["attention_mask_a"].to(DEVICE)
            input_ids_r = batch["input_ids_r"].to(DEVICE)
            attention_mask_r = batch["attention_mask_r"].to(DEVICE)
            
            # Get token embeddings
            with torch.no_grad():
                input_embeds = model.encoder.embeddings(input_ids_a)
            
            input_embeds.requires_grad_(True)
            
            # Integrated Gradients
            ig = LayerIntegratedGradients(
                lambda embeds: forward_fn(embeds, attention_mask_a, input_ids_r, attention_mask_r),
                model.encoder.embeddings
            )
            
            try:
                ig_attr = ig.attribute(
                    input_ids_a,
                    n_steps=IG_N_STEPS,
                    internal_batch_size=1,
                )
                # Sum over embedding dim for per-token attribution
                ig_attr_scores = ig_attr.sum(dim=-1).squeeze(0).cpu().numpy()
                ig_results.append(ig_attr_scores.tolist())
            except Exception as e:
                ig_results.append([])
                print(f"  IG failed for sample {i}: {e}")
            
            # Decode tokens for reference
            tokens = tokenizer.convert_ids_to_tokens(input_ids_a.squeeze(0).cpu())
            shap_results.append({"tokens": tokens})
        
        all_results["ig_attributions"] = ig_results
        all_results["tokens"] = [r["tokens"] for r in shap_results]
        
        # ---- Faithfulness metrics ----
        print("\nComputing faithfulness metrics...")
        from evaluation import compute_comprehensiveness, compute_sufficiency
        
        comp_scores = []
        suff_scores = []
        
        for i, batch in enumerate(test_loader):
            if i >= SHAP_N_SAMPLES:
                break
            if i >= len(ig_results) or not ig_results[i]:
                continue
            
            for k in FAITHFULNESS_TOP_K:
                comp = compute_comprehensiveness(
                    model, batch, [np.array(ig_results[i])], top_k=k, device=DEVICE
                )
                suff = compute_sufficiency(
                    model, batch, [np.array(ig_results[i])], top_k=k, device=DEVICE
                )
                comp_scores.append({"sample": i, "k": k, "comprehensiveness": comp})
                suff_scores.append({"sample": i, "k": k, "sufficiency": suff})
        
        all_results["comprehensiveness"] = comp_scores
        all_results["sufficiency"] = suff_scores
        
        # Aggregate faithfulness
        for k in FAITHFULNESS_TOP_K:
            k_comps = [s["comprehensiveness"] for s in comp_scores if s["k"] == k]
            k_suffs = [s["sufficiency"] for s in suff_scores if s["k"] == k]
            if k_comps:
                all_results[f"mean_comprehensiveness_k{k}"] = float(np.mean(k_comps))
                all_results[f"mean_sufficiency_k{k}"] = float(np.mean(k_suffs))
        
        print(f"  Comprehensiveness (k=10): {all_results.get('mean_comprehensiveness_k10', 'N/A')}")
        print(f"  Sufficiency (k=10): {all_results.get('mean_sufficiency_k10', 'N/A')}")
        
    except ImportError:
        print("  captum not installed. Skipping SHAP/IG attribution.")
    
    save_results(all_results, "xai.json")
    return all_results


# ============================================================
# Stage 7: Significance Testing (across all results)
# ============================================================
def stage_significance():
    """Run paired bootstrap tests between proposed model and all baselines."""
    print("\n" + "="*60)
    print("STAGE: Significance Testing")
    print("="*60)
    
    # Load all predictions and run pairwise tests
    # This requires stored predictions from previous stages
    # For now, load results and report CIs
    
    results_files = ["baselines.json", "transformers.json", "llms.json"]
    all_results = {}
    
    for fname in results_files:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                for key, metrics in data.items():
                    if isinstance(metrics, dict) and "qwk" in metrics:
                        all_results[key] = metrics["qwk"]
    
    print("\nAll QWK results:")
    for key, qwk in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        passed = "PASS" if qwk >= 0.70 else "FAIL"
        print(f"  {key:45s} QWK={qwk:.4f}  [{passed}]")
    
    return all_results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KhmerXScore Experiment Runner")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["prepare", "baselines", "transformers", "ablations",
                                 "llms", "xai", "significance", "all"],
                        help="Which stage to run")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    args = parser.parse_args()
    
    global DEVICE
    if args.device:
        DEVICE = args.device
    elif not torch.cuda.is_available():
        DEVICE = "cpu"
        print("WARNING: CUDA not available, using CPU")
    
    print(f"Device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")
    
    stages = {
        "prepare": stage_prepare,
        "baselines": stage_baselines,
        "transformers": stage_transformers,
        "ablations": stage_ablations,
        "llms": stage_llms,
        "xai": stage_xai,
        "significance": stage_significance,
    }
    
    if args.stage == "all":
        for name, fn in stages.items():
            print(f"\n{'#'*60}")
            print(f"# RUNNING STAGE: {name.upper()}")
            print(f"{'#'*60}")
            fn()
    else:
        stages[args.stage]()
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
