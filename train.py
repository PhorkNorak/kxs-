"""Training entry points for the three model families.

All trainable models share one recipe: MSE on the normalized score in [0,1],
early-stop on validation QWK with patience, save best checkpoint.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import config as C
from evaluate import metrics


# ────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(run_id: str) -> str:
    d = os.path.join(C.RUNS_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions_csv(out_dir: str, split_name: str, df_proc, pred_scores):
    """Write per-sample predictions to <out_dir>/predictions_<split_name>.csv.

    Columns:
        idx, Question, Reference, Answer,
        Max Score, true_raw, true_label, true_score,
        pred_score, pred_label, pred_raw,
        abs_error (5-class), raw_abs_error (raw points)
    """
    import pandas as pd

    pred_scores = np.asarray(pred_scores, dtype=np.float64).clip(0.0, 1.0)
    pred_labels = np.round(pred_scores * 4).astype(int).clip(0, 4)
    true_labels = df_proc["score_label"].values.astype(int)
    true_scores = df_proc["normalized_score"].values.astype(float)
    max_scores  = df_proc["Max Score"].values.astype(int)
    true_raw    = df_proc["Student Score"].values.astype(int)
    pred_raw = np.minimum(np.round(pred_scores * max_scores).astype(int),
                          max_scores).clip(0)

    out = pd.DataFrame({
        "idx": np.arange(len(df_proc)),
        "Question":   df_proc["Question"].values,
        "Reference":  df_proc["Reference"].values,
        "Answer":     df_proc["Answer"].values,
        "Max Score":  max_scores,
        "true_raw":   true_raw,
        "true_label": true_labels,
        "true_score": true_scores,
        "pred_score": pred_scores,
        "pred_label": pred_labels,
        "pred_raw":   pred_raw,
        "abs_error":     np.abs(pred_labels - true_labels),
        "raw_abs_error": np.abs(pred_raw - true_raw),
    })
    out_path = os.path.join(out_dir, f"predictions_{split_name}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Classical
# ────────────────────────────────────────────────────────────────────────────


def train_classical(model_id, preprocess_mode, input_fmt, train_df, val_df, test_df, run_id):
    from data import apply_preprocess, build_text_lists
    from models.classical import make_classical

    set_seed()
    train_p = apply_preprocess(train_df, preprocess_mode)
    val_p   = apply_preprocess(val_df,   preprocess_mode)
    test_p  = apply_preprocess(test_df,  preprocess_mode)

    train_a, train_b = build_text_lists(train_p, input_fmt)
    val_a,   val_b   = build_text_lists(val_p,   input_fmt)
    test_a,  test_b  = build_text_lists(test_p,  input_fmt)

    model = make_classical(model_id)
    model.fit(train_a, train_b, train_p["normalized_score"].values)

    train_pred = model.predict(train_a, train_b)
    val_pred   = model.predict(val_a,   val_b)
    test_pred  = model.predict(test_a,  test_b)

    train_m = metrics(train_pred, train_p["score_label"].values,
                      max_scores=train_p["Max Score"].values,
                      true_raw=train_p["Student Score"].values)
    val_m   = metrics(val_pred,   val_p["score_label"].values,
                      max_scores=val_p["Max Score"].values,
                      true_raw=val_p["Student Score"].values)
    test_m  = metrics(test_pred,  test_p["score_label"].values,
                      max_scores=test_p["Max Score"].values,
                      true_raw=test_p["Student Score"].values)

    out = run_dir(run_id)
    save_json(os.path.join(out, "config.json"),
              {"run_id": run_id, "model": model_id, "family": "classical",
               "preprocess": preprocess_mode, "input": input_fmt})
    save_json(os.path.join(out, "metrics.json"),
              {"train": train_m, "val": val_m, "test": test_m, "best_epoch": None})
    save_predictions_csv(out, "train", train_p, train_pred)
    save_predictions_csv(out, "val",   val_p,   val_pred)
    save_predictions_csv(out, "test",  test_p,  test_pred)
    return {"train": train_m, "val": val_m, "test": test_m}


# ────────────────────────────────────────────────────────────────────────────
# Neural training loop (shared by BiLSTM, Dual, Cross)
# ────────────────────────────────────────────────────────────────────────────


def _epoch(model, loader, optimizer, loss_fn, device, is_train, scaler=None, forward_keys=None):
    model.train(is_train)
    preds, labels = [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        targets = batch["score"]
        kwargs = {k: batch[k] for k in forward_keys}
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(**kwargs)
                    loss = loss_fn(out, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(**kwargs)
                loss = loss_fn(out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
        else:
            with torch.no_grad():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(**kwargs)
                        loss = loss_fn(out, targets)
                else:
                    out = model(**kwargs)
                    loss = loss_fn(out, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds.append(out.detach().float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return total_loss / max(n, 1), preds, labels


def _train_neural_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    forward_keys,
    lr,
    max_epochs,
    patience,
    device,
    run_id,
    use_amp,
    weight_decay,
):
    model = model.to(device)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    train_df_ref = train_loader.dataset.df
    val_df_ref   = val_loader.dataset.df
    test_df_ref  = test_loader.dataset.df
    train_max = train_df_ref["Max Score"].values
    train_raw = train_df_ref["Student Score"].values
    val_max   = val_df_ref["Max Score"].values
    val_raw   = val_df_ref["Student Score"].values
    test_max  = test_df_ref["Max Score"].values
    test_raw  = test_df_ref["Student Score"].values

    # Non-shuffled inference loader over the train set (so predictions align
    # row-for-row with train_df_ref for metric computation and CSV export).
    train_infer_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size or 16,
        shuffle=False,
    )

    best_qwk = -1e9
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for ep in range(1, max_epochs + 1):
        tr_loss, _, _ = _epoch(
            model, train_loader, optimizer, loss_fn, device,
            is_train=True, scaler=scaler, forward_keys=forward_keys,
        )
        # End-of-epoch evaluation passes on train (non-shuffled), val, and test —
        # gives full per-epoch train/val/test curves for plotting.
        ep_train_loss, ep_train_pred, ep_train_lab = _epoch(
            model, train_infer_loader, optimizer, loss_fn, device,
            is_train=False, scaler=scaler, forward_keys=forward_keys,
        )
        val_loss, val_pred, val_lab = _epoch(
            model, val_loader, optimizer, loss_fn, device,
            is_train=False, scaler=scaler, forward_keys=forward_keys,
        )
        ep_test_loss, ep_test_pred, ep_test_lab = _epoch(
            model, test_loader, optimizer, loss_fn, device,
            is_train=False, scaler=scaler, forward_keys=forward_keys,
        )
        train_m_ep = metrics(ep_train_pred, ep_train_lab,
                             max_scores=train_max, true_raw=train_raw)
        val_m      = metrics(val_pred, val_lab,
                             max_scores=val_max, true_raw=val_raw)
        test_m_ep  = metrics(ep_test_pred, ep_test_lab,
                             max_scores=test_max, true_raw=test_raw)
        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "test_loss": ep_test_loss,
            # train-set metrics this epoch
            **{f"train_{k}": v for k, v in train_m_ep.items()},
            # val-set metrics (kept under bare keys for backward compat with
            # the older plot script that reads "qwk", "accuracy", etc.)
            **val_m,
            **{f"val_{k}": v for k, v in val_m.items()},
            # test-set metrics this epoch
            **{f"test_{k}": v for k, v in test_m_ep.items()},
        })
        print(f"    epoch {ep:2d}  loss(tr/va/te)={tr_loss:.4f}/{val_loss:.4f}/{ep_test_loss:.4f}  "
              f"qwk(tr/va/te)={train_m_ep['qwk']:.4f}/{val_m['qwk']:.4f}/{test_m_ep['qwk']:.4f}  "
              f"acc(tr/va/te)={train_m_ep['accuracy']:.4f}/{val_m['accuracy']:.4f}/{test_m_ep['accuracy']:.4f}")

        if val_m["qwk"] > best_qwk:
            best_qwk = val_m["qwk"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early-stop at epoch {ep} (best epoch={best_epoch}, best_qwk={best_qwk:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test eval with best checkpoint
    _, test_pred, test_lab = _epoch(
        model, test_loader, optimizer, loss_fn, device,
        is_train=False, scaler=scaler, forward_keys=forward_keys,
    )
    test_m = metrics(test_pred, test_lab, max_scores=test_max, true_raw=test_raw)

    # Re-evaluate val with best state for the metrics record
    _, val_pred, val_lab = _epoch(
        model, val_loader, optimizer, loss_fn, device,
        is_train=False, scaler=scaler, forward_keys=forward_keys,
    )
    val_m = metrics(val_pred, val_lab, max_scores=val_max, true_raw=val_raw)

    # Train metrics with best checkpoint (overfit-gap reporting)
    _, train_pred, train_lab = _epoch(
        model, train_infer_loader, optimizer, loss_fn, device,
        is_train=False, scaler=scaler, forward_keys=forward_keys,
    )
    train_m = metrics(train_pred, train_lab, max_scores=train_max, true_raw=train_raw)

    out = run_dir(run_id)
    save_json(os.path.join(out, "metrics.json"),
              {"train": train_m, "val": val_m, "test": test_m,
               "best_epoch": best_epoch, "history": history})

    if best_state is not None:
        torch.save(best_state, os.path.join(out, "best.pt"))

    save_predictions_csv(out, "train", train_df_ref,           train_pred)
    save_predictions_csv(out, "val",   val_loader.dataset.df,  val_pred)
    save_predictions_csv(out, "test",  test_loader.dataset.df, test_pred)

    return {"train": train_m, "val": val_m, "test": test_m, "best_epoch": best_epoch}


# ────────────────────────────────────────────────────────────────────────────
# BiLSTM
# ────────────────────────────────────────────────────────────────────────────


def train_bilstm(preprocess_mode, input_fmt, train_df, val_df, test_df, run_id,
                 max_epochs=None, max_feat=False):
    from data import apply_preprocess, build_pair, CharPairDataset, build_char_vocab
    from models.bilstm import BiLSTMScorer

    set_seed()
    train_p = apply_preprocess(train_df, preprocess_mode)
    val_p   = apply_preprocess(val_df,   preprocess_mode)
    test_p  = apply_preprocess(test_df,  preprocess_mode)

    # Build char vocab from train side_a + side_b
    texts = []
    for _, row in train_p.iterrows():
        a, b = build_pair(row, input_fmt)
        texts.append(a)
        texts.append(b)
    char2id = build_char_vocab(texts)

    train_ds = CharPairDataset(train_p, char2id, input_fmt, provide_max_score=max_feat)
    val_ds   = CharPairDataset(val_p,   char2id, input_fmt, provide_max_score=max_feat)
    test_ds  = CharPairDataset(test_p,  char2id, input_fmt, provide_max_score=max_feat)

    train_loader = DataLoader(train_ds, batch_size=C.BILSTM_BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=C.BILSTM_BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=C.BILSTM_BATCH, shuffle=False)

    max_feat_dim = 1 if max_feat else 0
    model = BiLSTMScorer(vocab_size=len(char2id), max_feat_dim=max_feat_dim)
    forward_keys = ["input_ids_a", "attention_mask_a",
                    "input_ids_b", "attention_mask_b"]
    if max_feat:
        forward_keys.append("max_score_feat")

    family_name = "bilstm_maxfeat" if max_feat else "bilstm"
    model_name  = "bilstm_maxfeat" if max_feat else "bilstm"
    out = run_dir(run_id)
    save_json(os.path.join(out, "config.json"),
              {"run_id": run_id, "model": model_name, "family": family_name,
               "preprocess": preprocess_mode, "input": input_fmt,
               "vocab_size": len(char2id), "max_feat": max_feat})

    return _train_neural_loop(
        model, train_loader, val_loader, test_loader, forward_keys,
        lr=C.BILSTM_LR,
        max_epochs=max_epochs or C.BILSTM_MAX_EP,
        patience=C.BILSTM_PATIENCE,
        device=C.DEVICE,
        run_id=run_id,
        use_amp=False,
        weight_decay=0.0,
    )


# ────────────────────────────────────────────────────────────────────────────
# Dual / Cross transformer
# ────────────────────────────────────────────────────────────────────────────


def _make_tokenizer(backbone_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)


def train_transformer(
    arch, backbone_key, preprocess_mode, input_fmt,
    train_df, val_df, test_df, run_id, max_epochs=None, max_feat=False,
):
    from data import apply_preprocess, PairDataset, CrossDataset
    from models.dual import DualEncoderScorer
    from models.cross import CrossEncoderScorer

    set_seed()
    backbone_name = C.TRANSFORMER_BACKBONES[backbone_key]
    tokenizer = _make_tokenizer(backbone_name)

    train_p = apply_preprocess(train_df, preprocess_mode)
    val_p   = apply_preprocess(val_df,   preprocess_mode)
    test_p  = apply_preprocess(test_df,  preprocess_mode)

    max_feat_dim = 1 if max_feat else 0
    if arch == "dual":
        train_ds = PairDataset(train_p, tokenizer, input_fmt, provide_max_score=max_feat)
        val_ds   = PairDataset(val_p,   tokenizer, input_fmt, provide_max_score=max_feat)
        test_ds  = PairDataset(test_p,  tokenizer, input_fmt, provide_max_score=max_feat)
        forward_keys = ["input_ids_a", "attention_mask_a",
                        "input_ids_b", "attention_mask_b"]
        model = DualEncoderScorer(backbone_name, max_feat_dim=max_feat_dim)
    elif arch == "cross":
        train_ds = CrossDataset(train_p, tokenizer, input_fmt, provide_max_score=max_feat)
        val_ds   = CrossDataset(val_p,   tokenizer, input_fmt, provide_max_score=max_feat)
        test_ds  = CrossDataset(test_p,  tokenizer, input_fmt, provide_max_score=max_feat)
        forward_keys = ["input_ids", "attention_mask"]
        model = CrossEncoderScorer(backbone_name, max_feat_dim=max_feat_dim)
    else:
        raise ValueError(arch)
    if max_feat:
        forward_keys.append("max_score_feat")

    train_loader = DataLoader(train_ds, batch_size=C.TXFMR_BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=C.TXFMR_BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=C.TXFMR_BATCH, shuffle=False)

    family_name = f"{arch}_maxfeat" if max_feat else arch
    model_id    = f"{arch}_{backbone_key}_maxfeat" if max_feat else f"{arch}_{backbone_key}"
    out = run_dir(run_id)
    save_json(os.path.join(out, "config.json"),
              {"run_id": run_id, "model": model_id, "family": family_name,
               "backbone": backbone_name, "preprocess": preprocess_mode,
               "input": input_fmt, "max_feat": max_feat})

    return _train_neural_loop(
        model, train_loader, val_loader, test_loader, forward_keys,
        lr=C.TXFMR_LR,
        max_epochs=max_epochs or C.TXFMR_MAX_EP,
        patience=C.TXFMR_PATIENCE,
        device=C.DEVICE,
        run_id=run_id,
        use_amp=True,
        weight_decay=C.TXFMR_WEIGHT_DECAY,
    )
