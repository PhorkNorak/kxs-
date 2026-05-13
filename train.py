"""
Training Loop — early stopping on val QWK, joint CORN+SCL, MC Dropout inference.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.losses import (CORNLoss, KXCLLoss, WeightedMSELoss,
                            corn_logits_to_label, corn_logits_to_score, compute_class_weights)
from evaluation import compute_all_metrics, quadratic_weighted_kappa


def train_transformer(model, train_loader, val_loader, config,
                      device="cpu", checkpoint_dir="./checkpoints", run_name="model"):
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.max_epochs
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    # Pre-compute class weights from the full training set (one pass, done once)
    _all_labels = np.concatenate([b["label"].numpy() for b in train_loader])
    class_weights = compute_class_weights(_all_labels).to(device)

    # Loss
    use_scl = config.loss_type == "corn" and config.scl_beta > 0
    epsilon = getattr(config, "corn_epsilon", 0.0)
    if config.loss_type == "corn":
        if use_scl:
            criterion = KXCLLoss(5, config.scl_alpha, config.scl_beta,
                                 config.scl_temperature, config.scl_margin_scale,
                                 epsilon=epsilon)
        else:
            criterion = CORNLoss(5, epsilon=epsilon)
    elif config.loss_type == "weighted_mse":
        criterion = WeightedMSELoss(class_weights)
        class_weights = None   # already baked into criterion; avoid double-weighting
    else:
        criterion = nn.MSELoss()
        class_weights = None

    best_qwk, patience = -1.0, 0
    history = {"train_loss": [], "val_qwk": []}
    ckpt = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    for ep in range(config.max_epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}", leave=False):
            ids_a  = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_r  = batch["input_ids_r"].to(device)
            mask_r = batch["attention_mask_r"].to(device)
            scores = batch["score"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            use_cw = class_weights is not None and getattr(config, "corn_class_weighted", True)
            if use_scl:
                logits, emb = model(ids_a, mask_a, ids_r, mask_r, return_embeddings=True)
                loss, _ = criterion(logits, labels, emb, scores,
                                    class_weights=class_weights if use_cw else None)
            else:
                logits = model(ids_a, mask_a, ids_r, mask_r)
                if config.loss_type == "corn":
                    loss = criterion(logits, labels,
                                     class_weights=class_weights if use_cw else None)
                elif config.loss_type == "weighted_mse":
                    loss = criterion(logits.squeeze(-1), scores, labels)
                else:
                    loss = criterion(logits.squeeze(-1), scores)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        # Validate
        val_m = evaluate_model(model, val_loader, config.loss_type, device)
        vqwk = val_m["qwk"]
        history["train_loss"].append(float(np.mean(losses)))
        history["val_qwk"].append(vqwk)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1:2d}  loss={np.mean(losses):.4f}  val_qwk={vqwk:.4f}  best={best_qwk:.4f}")

        if vqwk > best_qwk:
            best_qwk, patience = vqwk, 0
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_qwk": vqwk}, ckpt)
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"  Early stop ep={ep+1}  best_val_qwk={best_qwk:.4f}")
                break

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                          weights_only=True)["model_state_dict"])
    return {"best_val_qwk": best_qwk, "history": history, "epochs_trained": ep + 1}


def evaluate_model(model, dataloader, loss_type="corn", device="cpu"):
    model.eval()
    tl, pl, ts, ps = [], [], [], []
    with torch.no_grad():
        for b in dataloader:
            ids_a  = b["input_ids_a"].to(device)
            mask_a = b["attention_mask_a"].to(device)
            ids_r  = b["input_ids_r"].to(device)
            mask_r = b["attention_mask_r"].to(device)
            logits = model(ids_a, mask_a, ids_r, mask_r)
            if loss_type == "corn":
                _pl = corn_logits_to_label(logits).cpu().numpy()
                _ps = corn_logits_to_score(logits).cpu().numpy()
            else:
                _ps = logits.squeeze(-1).clamp(0, 1).cpu().numpy()
                _pl = np.round(_ps * 4).astype(np.int64).clip(0, 4)
            tl.extend(b["label"].numpy()); pl.extend(_pl)
            ts.extend(b["score"].numpy()); ps.extend(_ps)
    return compute_all_metrics(np.array(tl), np.array(pl), np.array(ts), np.array(ps))


def evaluate_with_uncertainty(model, dataloader, loss_type="corn", device="cpu", T=10):
    all_tl, all_ms, all_ss, all_pl, all_ts = [], [], [], [], []
    for b in dataloader:
        ids_a  = b["input_ids_a"].to(device)
        mask_a = b["attention_mask_a"].to(device)
        ids_r  = b["input_ids_r"].to(device)
        mask_r = b["attention_mask_r"].to(device)
        result = model.predict_with_uncertainty(ids_a, mask_a, ids_r, mask_r, T=T)
        all_tl.extend(b["label"].numpy())
        all_ts.extend(b["score"].numpy())
        all_ms.extend(result["mean_score"].cpu().numpy())
        all_ss.extend(result["std_score"].cpu().numpy())
        all_pl.extend(result["mode_label"].cpu().numpy())
    tl = np.array(all_tl); pl = np.array(all_pl).astype(np.int64)
    ts = np.array(all_ts); ps = np.array(all_ms); unc = np.array(all_ss)
    m = compute_all_metrics(tl, pl, ts, ps)
    m["mean_uncertainty"] = float(unc.mean())
    return m, tl, pl, unc
