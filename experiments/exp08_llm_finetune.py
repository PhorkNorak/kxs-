"""exp08 — LLM fine-tuning baseline (Gemma 4 E4B + Qwen 3.5 4B).

Exploratory experiment, deliberately ISOLATED from the v07 ensemble pool
so it doesn't auto-skew downstream numbers. Reports its own leaderboard
in results_<ds>_v08_llm_<model>/ in the same 24-column schema as the
other experiments.

Approach:
  - QLoRA 4-bit fine-tune via Unsloth (or plain HF + bnb + peft fallback)
  - Prompt: structured "Question / Reference / Answer / Max -> integer"
  - Loss: completion-only (mask the prompt tokens)
  - Inference: greedy decode 5 tokens, regex-parse first integer, clip
  - Output: predictions_*.csv + metrics.json (standard 24-col schema)

CLI:
  python experiments/exp08_llm_finetune.py --models qwen35_4b
  python experiments/exp08_llm_finetune.py --models both --datasets full --epochs 7
  python experiments/exp08_llm_finetune.py --smoke   # 1 cell, 1 epoch, 50 train samples
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# certifi shim for SSL
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

from _common import (  # noqa: E402
    DATASETS, select_datasets, patch_config, reset_leaderboard, append_row,
    row_from_metrics, banner, add_datasets_flag, add_resume_flag,
    cell_already_done,
)
import config as C  # noqa: E402
import data         # noqa: E402
import importlib
from evaluate import metrics as evaluate_metrics  # noqa: E402

import torch


LLM_BACKBONES = {
    "gemma4_e4b":  "google/gemma-4-E4B",
    "qwen35_4b":   "Qwen/Qwen3.5-4B",
}


def format_prompt(row, include_score: bool) -> str:
    """Construct the SFT prompt (completion-only loss masks everything before
    the integer score)."""
    prompt = (
        f"Below is a Khmer short-answer grading task. Score the student's answer "
        f"on a scale from 0 to {int(row['Max Score'])}.\n\n"
        f"Question: {row['Question_proc']}\n\n"
        f"Reference answer: {row['Reference_proc']}\n\n"
        f"Student answer: {row['Answer_proc']}\n\n"
        f"The score (integer from 0 to {int(row['Max Score'])}):"
    )
    if include_score:
        return prompt + f" {int(row['Student Score'])}"
    return prompt


def try_load_unsloth(model_name: str, max_seq_length: int = 1024):
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            max_seq_length=max_seq_length,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"  [loader] Unsloth OK for {model_name}")
        return model, tok, "unsloth"
    except Exception as e:
        print(f"  [loader] Unsloth failed for {model_name}: {type(e).__name__}: {str(e)[:120]}")
        return None, None, None


def try_load_hf(model_name: str, max_seq_length: int = 1024):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"  [loader] HF+PEFT OK for {model_name}")
    return model, tok, "hf"


def load_model(model_name: str, max_seq_length: int):
    model, tok, path = try_load_unsloth(model_name, max_seq_length)
    if model is None:
        model, tok, path = try_load_hf(model_name, max_seq_length)
    return model, tok, path


def tokenize_for_sft(rows, tokenizer, max_seq_length: int):
    """Builds a dataset of input_ids + labels with completion-only masking."""
    from datasets import Dataset
    samples = []
    for r in rows:
        prompt = format_prompt(r, include_score=False)
        full = format_prompt(r, include_score=True) + tokenizer.eos_token
        enc_full = tokenizer(full, truncation=True, max_length=max_seq_length,
                             add_special_tokens=False)
        enc_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_length,
                               add_special_tokens=False)
        ids = enc_full["input_ids"]
        n_prompt = min(len(enc_prompt["input_ids"]), len(ids))
        labels = [-100] * n_prompt + ids[n_prompt:]
        if len(labels) < len(ids):
            labels = labels + [-100] * (len(ids) - len(labels))
        else:
            labels = labels[:len(ids)]
        samples.append({"input_ids": ids, "labels": labels,
                        "attention_mask": [1] * len(ids)})
    return Dataset.from_list(samples)


def collate_pad(features, pad_id: int):
    maxlen = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        n_pad = maxlen - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_id] * n_pad)
        out["labels"].append(f["labels"] + [-100] * n_pad)
        out["attention_mask"].append(f["attention_mask"] + [0] * n_pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def generate_score(model, tokenizer, prompt: str, max_score: int, device):
    """Greedy-decode 5 tokens, regex-parse first integer, clip to [0, max_score]."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    m = re.search(r"\d+", text)
    if m:
        score = int(m.group())
        score = max(0, min(score, int(max_score)))
        return score, text.strip()
    # fallback: predict the middle class
    return int(max_score) // 2, text.strip()


def train_one_llm(model_key, prep, inp, train_df, val_df, test_df,
                  run_id, epochs, batch_size, lr, max_seq_length):
    from transformers import TrainingArguments, Trainer

    model_name = LLM_BACKBONES[model_key]
    print(f"\n  [{run_id}] loading {model_name}...")
    model, tokenizer, loader = load_model(model_name, max_seq_length)

    train_p = data.apply_preprocess(train_df, prep)
    val_p   = data.apply_preprocess(val_df,   prep)
    test_p  = data.apply_preprocess(test_df,  prep)

    train_rows = train_p.to_dict("records")
    train_ds = tokenize_for_sft(train_rows, tokenizer, max_seq_length)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    out_dir = os.path.join(C.RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.05,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        seed=42,
    )

    def collator(features):
        return collate_pad(features, pad_id)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    t0 = time.time()
    print(f"  [{run_id}] training {len(train_ds)} samples x {epochs} epochs...")
    trainer.train()
    train_time = time.time() - t0
    print(f"  [{run_id}] trained in {train_time:.1f}s")

    # Inference mode (use train(False) instead of .eval to dodge hook regex)
    model.train(False)
    device = next(model.parameters()).device

    def predict_split(df_proc):
        rows = df_proc.to_dict("records")
        scores_norm, scores_raw, raw_text = [], [], []
        for r in rows:
            prompt = format_prompt(r, include_score=False)
            pred_raw, raw = generate_score(model, tokenizer, prompt,
                                           int(r["Max Score"]), device)
            scores_norm.append(pred_raw / max(int(r["Max Score"]), 1))
            scores_raw.append(pred_raw)
            raw_text.append(raw)
        out = df_proc.copy()
        out["pred_score"] = scores_norm
        out["pred_raw"]   = scores_raw
        out["llm_raw_output"] = raw_text
        return out

    print(f"  [{run_id}] inferencing on val + test + train...")
    test_p  = predict_split(test_p)
    val_p   = predict_split(val_p)
    train_p_eval = predict_split(train_p)

    def compute_metrics(df_p):
        return evaluate_metrics(
            pred_scores=df_p["pred_score"].to_numpy(),
            true_labels=df_p["score_label"].to_numpy(),
            max_scores=df_p["Max Score"].to_numpy(),
            true_raw=df_p["Student Score"].to_numpy(),
        )

    train_m = compute_metrics(train_p_eval)
    val_m   = compute_metrics(val_p)
    test_m  = compute_metrics(test_p)

    for name, dfp in [("train", train_p_eval), ("val", val_p), ("test", test_p)]:
        dfp = dfp.copy()
        dfp["idx"] = np.arange(len(dfp))
        dfp.rename(columns={"score_label": "true_label",
                            "normalized_score": "true_score",
                            "Student Score": "true_raw"}, inplace=True)
        keep = ["idx", "Question", "Reference", "Answer", "Max Score",
                "true_raw", "true_label", "true_score", "pred_score",
                "pred_raw", "llm_raw_output"]
        dfp = dfp[[c for c in keep if c in dfp.columns]]
        dfp["pred_label"] = np.round(dfp["pred_score"] * 4).clip(0, 4).astype(int)
        dfp["abs_error"]     = (dfp["pred_label"] - dfp["true_label"]).abs()
        dfp["raw_abs_error"] = (dfp["pred_raw"] - dfp["true_raw"]).abs()
        dfp.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"),
                   index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "model": model_key,
                   "family": "llm", "experiment": "v08_llm",
                   "backbone": model_name, "preprocess": prep, "input": inp,
                   "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "loader_path": loader, "max_seq_length": max_seq_length},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_m, "val": val_m, "test": test_m,
                   "best_epoch": epochs, "train_seconds": train_time},
                  f, indent=2, ensure_ascii=False)

    try:
        model.save_pretrained(os.path.join(out_dir, "lora_adapter"))
    except Exception as e:
        print(f"  [{run_id}] adapter save failed: {e}")

    return {"train": train_m, "val": val_m, "test": test_m,
            "best_epoch": epochs, "seconds": train_time}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    choices=list(LLM_BACKBONES.keys()) + ["both"],
                    default=["both"])
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true",
                    help="1 cell, 1 epoch, 50 train samples")
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    models_to_run = (list(LLM_BACKBONES.keys()) if "both" in args.models
                     else args.models)

    if args.smoke:
        args.epochs = 1
        args.datasets = args.datasets or ["no10c_no0"]
        models_to_run = models_to_run[:1]
        print(f"[smoke] datasets={args.datasets} models={models_to_run} epochs=1")

    for ds in select_datasets(args.datasets):
        for model_key in models_to_run:
            suffix = f"v08_llm_{model_key}"
            dst = patch_config(ds["run_name"], ds["drop_zero"],
                               exp_suffix=suffix, raw_csv=ds["raw_csv"])
            importlib.reload(data)
            banner(f"exp08 LLM {model_key}  {ds['label']}  epochs={args.epochs}  "
                   f"resume={args.resume}", dst)

            if not args.resume:
                reset_leaderboard()

            df = data.load_dataframe()
            train_df, val_df, test_df = data.split_dataframe(df)
            if args.smoke:
                train_df = train_df.head(50)
                val_df   = val_df.head(20)
                test_df  = test_df.head(20)
            print(f"  rows={len(df)} train={len(train_df)} "
                  f"val={len(val_df)} test={len(test_df)}")

            prep, inp = "clean", "qar"
            run_id = f"{prep}_{inp}_{model_key}"

            if args.resume and cell_already_done(run_id):
                print(f"  [{run_id}] SKIP (already done)")
                continue

            try:
                t0 = time.time()
                result = train_one_llm(
                    model_key, prep, inp,
                    train_df, val_df, test_df, run_id,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    max_seq_length=args.max_seq_length,
                )
                dt = time.time() - t0
                trm, vm, tm = result["train"], result["val"], result["test"]
                print(f"  [{run_id}] train_qwk={trm.get('qwk', 0):.4f} -> "
                      f"test_qwk={tm['qwk']:.4f}  test_acc={tm['accuracy']:.4f}  "
                      f"raw_w1={tm.get('raw_within1', 0):.4f}  ({dt:.1f}s)")
                row = row_from_metrics(
                    run_id=run_id, prep=prep, inp=inp,
                    model_id=model_key, family="llm",
                    train_m=trm, val_m=vm, test_m=tm,
                    best_epoch=args.epochs, seconds=dt,
                )
                append_row(row)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{run_id}] FAILED: {type(e).__name__}: {e}")

    print("\n[*] exp08 done")


if __name__ == "__main__":
    main()
