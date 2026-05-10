"""
KhmerXScore Training Loop
===========================
Handles training for all transformer models with:
- CORN / MSE / Weighted MSE loss selection
- Optional SCL joint training
- Early stopping on validation QWK
- MC Dropout inference for uncertainty
- Checkpointing and result logging
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.losses import (
    CORNLoss, KXCLLoss, WeightedMSELoss,
    corn_logits_to_label, corn_logits_to_score, compute_class_weights
)
from evaluation import compute_all_metrics, quadratic_weighted_kappa


def train_transformer(model, train_loader, val_loader, config,
                      device: str = "cuda", checkpoint_dir: str = "./checkpoints",
                      run_name: str = "kxcl"):
    """
    Full training pipeline for a transformer scoring model.
    
    Args:
        model: DualEncoder or CrossEncoder
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: TrainConfig dataclass
        device: 'cuda' or 'cpu'
        checkpoint_dir: Where to save checkpoints
        run_name: Identifier for this run
    
    Returns:
        Dict with training history and best metrics
    """
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    
    # ---- Loss function ----
    if config.loss_type == "corn":
        if config.scl_beta > 0:
            criterion = KXCLLoss(
                num_classes=5, alpha=config.scl_alpha, beta=config.scl_beta,
                temperature=config.scl_temperature, margin_scale=config.scl_margin_scale
            )
        else:
            criterion = CORNLoss(num_classes=5)
    elif config.loss_type == "weighted_mse":
        # Compute class weights from training labels
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch["label"].numpy())
        weights = compute_class_weights(np.array(train_labels)).to(device)
        criterion = WeightedMSELoss(weights)
    else:  # mse
        criterion = nn.MSELoss()
    
    # ---- Training loop ----
    best_val_qwk = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_qwk": [], "val_rmse": []}
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}",
                          leave=False):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_r = batch["input_ids_r"].to(device)
            attention_mask_r = batch["attention_mask_r"].to(device)
            scores = batch["score"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            use_scl = config.loss_type == "corn" and config.scl_beta > 0
            
            if use_scl:
                logits, embeddings = model(
                    input_ids_a, attention_mask_a,
                    input_ids_r, attention_mask_r,
                    return_embeddings=True
                )
                loss, loss_dict = criterion(logits, labels, embeddings, scores)
            else:
                logits = model(
                    input_ids_a, attention_mask_a,
                    input_ids_r, attention_mask_r
                )
                
                if config.loss_type == "corn":
                    loss = criterion(logits, labels)
                elif config.loss_type == "weighted_mse":
                    pred = logits.squeeze(-1)
                    loss = criterion(pred, scores, labels)
                else:
                    pred = logits.squeeze(-1)
                    loss = criterion(pred, scores)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        # ---- Validation ----
        val_metrics = evaluate_model(model, val_loader, config.loss_type, device)
        val_qwk = val_metrics["qwk"]
        val_rmse = val_metrics["rmse"]
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        
        history["train_loss"].append(avg_loss)
        history["val_qwk"].append(val_qwk)
        history["val_rmse"].append(val_rmse)
        
        print(f"Epoch {epoch+1}/{config.max_epochs} | "
              f"Loss: {avg_loss:.4f} | Val QWK: {val_qwk:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | Time: {epoch_time:.1f}s")
        
        # ---- Early stopping ----
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            patience_counter = 0
            # Save best model
            ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
            }, ckpt_path)
            print(f"  -> Saved best model (QWK={val_qwk:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (patience={config.early_stop_patience})")
                break
    
    # ---- Load best model ----
    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return {
        "best_val_qwk": best_val_qwk,
        "history": history,
        "epochs_trained": epoch + 1,
    }


def evaluate_model(model, dataloader, loss_type: str = "corn",
                   device: str = "cuda") -> dict:
    """
    Evaluate model on a dataset.
    Returns full metric suite.
    """
    model.eval()
    
    all_true_labels = []
    all_pred_labels = []
    all_true_scores = []
    all_pred_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_r = batch["input_ids_r"].to(device)
            attention_mask_r = batch["attention_mask_r"].to(device)
            
            logits = model(input_ids_a, attention_mask_a,
                          input_ids_r, attention_mask_r)
            
            if loss_type == "corn":
                pred_labels = corn_logits_to_label(logits).cpu().numpy()
                pred_scores = corn_logits_to_score(logits).cpu().numpy()
            else:
                pred_scores = logits.squeeze(-1).clamp(0, 1).cpu().numpy()
                pred_labels = np.round(pred_scores * 4).astype(np.int64).clip(0, 4)
            
            all_true_labels.extend(batch["label"].numpy())
            all_pred_labels.extend(pred_labels)
            all_true_scores.extend(batch["score"].numpy())
            all_pred_scores.extend(pred_scores)
    
    y_true_labels = np.array(all_true_labels)
    y_pred_labels = np.array(all_pred_labels)
    y_true_scores = np.array(all_true_scores)
    y_pred_scores = np.array(all_pred_scores)
    
    return compute_all_metrics(y_true_labels, y_pred_labels,
                               y_true_scores, y_pred_scores)


def evaluate_with_uncertainty(model, dataloader, loss_type: str = "corn",
                              device: str = "cuda", T: int = 10) -> dict:
    """
    Evaluate with MC Dropout uncertainty estimation.
    (Gal & Ghahramani 2016)
    """
    all_true_labels = []
    all_mean_scores = []
    all_std_scores = []
    all_pred_labels = []
    all_true_scores = []
    
    for batch in tqdm(dataloader, desc="MC Dropout evaluation"):
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_r = batch["input_ids_r"].to(device)
        attention_mask_r = batch["attention_mask_r"].to(device)
        
        result = model.predict_with_uncertainty(
            input_ids_a, attention_mask_a,
            input_ids_r, attention_mask_r, T=T
        )
        
        all_true_labels.extend(batch["label"].numpy())
        all_true_scores.extend(batch["score"].numpy())
        all_mean_scores.extend(result["mean_score"].cpu().numpy())
        all_std_scores.extend(result["std_score"].cpu().numpy())
        all_pred_labels.extend(result["mode_label"].cpu().numpy())
    
    y_true_labels = np.array(all_true_labels)
    y_pred_labels = np.array(all_pred_labels).astype(np.int64)
    y_true_scores = np.array(all_true_scores)
    y_pred_scores = np.array(all_mean_scores)
    uncertainties = np.array(all_std_scores)
    
    metrics = compute_all_metrics(y_true_labels, y_pred_labels,
                                  y_true_scores, y_pred_scores)
    metrics["uncertainties"] = uncertainties.tolist()
    metrics["mean_uncertainty"] = float(np.mean(uncertainties))
    
    return metrics, y_true_labels, y_pred_labels, uncertainties
