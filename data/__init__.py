"""
Data Module — load, clean, split, DataLoader with class-balanced sampling.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from preprocessing import preprocess


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Subject"] = df["Subject"].str.strip().replace({"History ": "History"})
    df["normalized_score"] = df["Student Score"] / df["Max Score"]
    df["score_label"] = (df["normalized_score"] * 4).round().astype(int).clip(0, 4)
    df = df.dropna(subset=["Question", "Reference", "Answer"])
    for col in ["Question", "Reference", "Answer"]:
        df[col] = df[col].astype(str).str.replace(r"\n", " ", regex=True).str.strip()
    return df.reset_index(drop=True)


def split_data(df, seed=42, train_r=0.70, val_r=0.15, test_r=0.15):
    min_count = df["score_label"].value_counts().min()
    strat = df["score_label"] if min_count >= 4 else None
    if strat is None:
        print("  NOTE: Stratification skipped — some classes too small.")
    train_df, temp = train_test_split(df, test_size=val_r+test_r, random_state=seed, stratify=strat)
    strat2 = temp["score_label"] if (strat is not None and temp["score_label"].value_counts().min() >= 2) else None
    val_df, test_df = train_test_split(temp, test_size=test_r/(val_r+test_r), random_state=seed, stratify=strat2)
    for s, n in [(train_df,"train"), (val_df,"val"), (test_df,"test")]:
        s.__dict__["split_name"] = n
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    print(f"  Split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def preprocess_dataframe(df, mode="kcc_seg_punct"):
    df = df.copy()
    for col in ["Question", "Reference", "Answer"]:
        df[f"{col}_proc"] = df[col].apply(lambda x: preprocess(x, mode))
    return df


class KhmerSAGDataset(Dataset):
    """Transformer dataset — supports (A,R) and (Q,A,R) input formats."""
    def __init__(self, df, tokenizer, max_len=256, input_format="qar",
                 use_processed=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_format = input_format
        suffix = "_proc" if use_processed and "Answer_proc" in df.columns else ""
        self.questions  = df[f"Question{suffix}"].values
        self.answers    = df[f"Answer{suffix}"].values
        self.references = df[f"Reference{suffix}"].values
        self.scores = df["normalized_score"].values.astype(np.float32)
        self.labels = df["score_label"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        answer = str(self.answers[idx])
        reference = str(self.references[idx])
        if self.input_format == "qar":
            answer = str(self.questions[idx]) + " " + answer
        enc_a = self.tokenizer(answer, max_length=self.max_len, padding="max_length",
                               truncation=True, return_tensors="pt")
        enc_r = self.tokenizer(reference, max_length=self.max_len, padding="max_length",
                               truncation=True, return_tensors="pt")
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_r": enc_r["input_ids"].squeeze(0),
            "attention_mask_r": enc_r["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class KhmerSAGTextDataset:
    """Simple text dataset for classical ML models (no tokenization)."""
    def __init__(self, df, input_format="qar", use_processed=True):
        suffix = "_proc" if use_processed and "Answer_proc" in df.columns else ""
        q, a, r = f"Question{suffix}", f"Answer{suffix}", f"Reference{suffix}"
        self.answers = (df[q] + " " + df[a]).tolist() if input_format == "qar" else df[a].tolist()
        self.references = df[r].tolist()
        self.scores = df["normalized_score"].values.astype(np.float32)
        self.labels = df["score_label"].values.astype(np.int64)


def get_class_balanced_sampler(labels):
    counts = np.maximum(np.bincount(labels, minlength=5).astype(np.float32), 1)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(torch.from_numpy(weights).float(), len(labels), True)


def get_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=16,
                    max_len=256, input_format="qar", preproc_mode="kcc",
                    use_class_balanced=True, num_workers=0):
    train_ds = KhmerSAGDataset(train_df, tokenizer, max_len, input_format)
    val_ds   = KhmerSAGDataset(val_df,   tokenizer, max_len, input_format)
    test_ds  = KhmerSAGDataset(test_df,  tokenizer, max_len, input_format)
    sampler = get_class_balanced_sampler(train_ds.labels) if use_class_balanced else None
    shuffle = not use_class_balanced
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld, test_ld
