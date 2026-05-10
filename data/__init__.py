"""
KhmerXScore Data Module
========================
- Load and clean the KhmerSAG dataset
- Stratified train/val/test split
- Torch Dataset and DataLoader with class-balanced batch sampling
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from preprocessing import preprocess


# ============================================================
# Load and clean raw CSV
# ============================================================
def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the KhmerSAG dataset."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Fix column names (leading spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Fix Subject trailing spaces
    df["Subject"] = df["Subject"].str.strip()
    
    # Merge 'History ' and 'History'
    df["Subject"] = df["Subject"].replace({"History ": "History"})
    
    # Compute normalized score (0.0–1.0)
    df["normalized_score"] = df["Student Score"] / df["Max Score"]
    
    # Compute discrete score label (0–4) for QWK
    df["score_label"] = (df["normalized_score"] * 4).round().astype(int).clip(0, 4)
    
    # Drop any rows with missing text
    df = df.dropna(subset=["Question", "Reference", "Answer"])
    
    # Clean newlines in text fields
    for col in ["Question", "Reference", "Answer"]:
        df[col] = df[col].str.replace(r"\n", " ", regex=True).str.strip()
    
    return df.reset_index(drop=True)


# ============================================================
# Stratified split
# ============================================================
def split_data(df: pd.DataFrame, seed: int = 42,
               train_r: float = 0.70, val_r: float = 0.15, test_r: float = 0.15):
    """
    Stratified train/val/test split by score_label.
    Returns DataFrames with a 'split' column.
    """
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_r + test_r), random_state=seed,
        stratify=df["score_label"]
    )
    
    # Second split: val vs test
    relative_test = test_r / (val_r + test_r)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, random_state=seed,
        stratify=temp_df["score_label"]
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Train label dist: {dict(train_df['score_label'].value_counts().sort_index())}")
    
    return train_df, val_df, test_df


# ============================================================
# Preprocessing wrapper
# ============================================================
def preprocess_dataframe(df: pd.DataFrame, mode: str = "kcc_seg_punct") -> pd.DataFrame:
    """Apply preprocessing to all text columns."""
    df = df.copy()
    for col in ["Question", "Reference", "Answer"]:
        df[f"{col}_proc"] = df[col].apply(lambda x: preprocess(x, mode))
    return df


# ============================================================
# Torch Dataset for Transformers
# ============================================================
class KhmerSAGDataset(Dataset):
    """
    Dataset for transformer-based models.
    Supports both (A, R) and (Q, A, R) input formats.
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256,
                 input_format: str = "qar", preproc_mode: str = "kcc",
                 use_processed: bool = True):
        """
        Args:
            df: DataFrame with text columns
            tokenizer: HuggingFace tokenizer
            max_len: Maximum sequence length
            input_format: 'ar' for (A,R) or 'qar' for (Q,A,R)
            preproc_mode: Preprocessing mode for text columns
            use_processed: If True, use preprocessed columns (with _proc suffix)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_format = input_format
        self.preproc_mode = preproc_mode
        
        # Determine column names
        suffix = "_proc" if use_processed and f"Answer_proc" in df.columns else ""
        self.q_col = f"Question{suffix}"
        self.a_col = f"Answer{suffix}"
        self.r_col = f"Reference{suffix}"
        
        # Pre-extract arrays for speed
        self.questions = self.df[self.q_col].values
        self.answers = self.df[self.a_col].values
        self.references = self.df[self.r_col].values
        self.scores = self.df["normalized_score"].values.astype(np.float32)
        self.labels = self.df["score_label"].values.astype(np.int64)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        answer = str(self.answers[idx])
        reference = str(self.references[idx])
        question = str(self.questions[idx])
        
        # Construct answer-side input
        if self.input_format == "qar":
            answer_input = question + " " + answer
        else:
            answer_input = answer
        
        # Tokenize answer side (for dual-encoder, this becomes tower A)
        enc_a = self.tokenizer(
            answer_input,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize reference side (tower R)
        enc_r = self.tokenizer(
            reference,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_r": enc_r["input_ids"].squeeze(0),
            "attention_mask_r": enc_r["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Dataset for classical models (returns raw text + scores)
# ============================================================
class KhmerSAGTextDataset:
    """Simple dataset for classical ML models (no tokenization)."""
    
    def __init__(self, df: pd.DataFrame, input_format: str = "qar",
                 use_processed: bool = True):
        suffix = "_proc" if use_processed and "Answer_proc" in df.columns else ""
        q_col = f"Question{suffix}"
        a_col = f"Answer{suffix}"
        r_col = f"Reference{suffix}"
        
        if input_format == "qar":
            self.answers = (df[q_col] + " " + df[a_col]).tolist()
        else:
            self.answers = df[a_col].tolist()
        
        self.references = df[r_col].tolist()
        self.scores = df["normalized_score"].values.astype(np.float32)
        self.labels = df["score_label"].values.astype(np.int64)


# ============================================================
# Class-balanced sampler (Cui et al. 2019)
# ============================================================
def get_class_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a weighted sampler that balances class frequencies per batch.
    Each sample's weight = 1 / count(its class).
    """
    class_counts = np.bincount(labels, minlength=5)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True
    )
    return sampler


# ============================================================
# DataLoader factory
# ============================================================
def get_dataloaders(train_df, val_df, test_df, tokenizer,
                    batch_size=16, max_len=256, input_format="qar",
                    preproc_mode="kcc", use_class_balanced=True,
                    num_workers=4):
    """Create train/val/test DataLoaders."""
    
    train_ds = KhmerSAGDataset(train_df, tokenizer, max_len, input_format, preproc_mode)
    val_ds = KhmerSAGDataset(val_df, tokenizer, max_len, input_format, preproc_mode)
    test_ds = KhmerSAGDataset(test_df, tokenizer, max_len, input_format, preproc_mode)
    
    # Class-balanced sampling for training
    train_sampler = None
    train_shuffle = True
    if use_class_balanced:
        train_sampler = get_class_balanced_sampler(train_ds.labels)
        train_shuffle = False  # Sampler handles ordering
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=train_shuffle,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
