"""Data loading, splitting, and dataset classes for the simple pipeline.

Uses the same CSV and the same 70/15/15 stratified split (seed=42) as the
main `kxs/data` module so results are comparable across the leaderboard.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config as C
from preprocess import preprocess


def load_dataframe(csv_path: str = C.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Subject"] = df["Subject"].str.strip().replace({"History ": "History"})
    df["normalized_score"] = df["Student Score"] / df["Max Score"]
    df["score_label"] = (df["normalized_score"] * 4).round().astype(int).clip(0, 4)
    df = df.dropna(subset=["Question", "Reference", "Answer"])
    if C.DROP_SCORE_ZERO:
        df = df[df["score_label"] > 0].reset_index(drop=True)
    for col in ["Question", "Reference", "Answer"]:
        df[col] = df[col].astype(str).str.replace(r"\n", " ", regex=True).str.strip()
    return df.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, seed: int = C.SEED):
    min_count = df["score_label"].value_counts().min()
    strat = df["score_label"] if min_count >= 4 else None
    train_df, temp = train_test_split(
        df, test_size=C.VAL_RATIO + C.TEST_RATIO, random_state=seed, stratify=strat
    )
    strat2 = (
        temp["score_label"]
        if strat is not None and temp["score_label"].value_counts().min() >= 2
        else None
    )
    val_df, test_df = train_test_split(
        temp,
        test_size=C.TEST_RATIO / (C.VAL_RATIO + C.TEST_RATIO),
        random_state=seed,
        stratify=strat2,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def apply_preprocess(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    for col in ["Question", "Reference", "Answer"]:
        df[f"{col}_proc"] = df[col].apply(lambda x: preprocess(x, mode))
    return df


def build_pair(row, input_fmt: str):
    """Two-sided input for dual encoder, BiLSTM, classical-pair models.

    ra  → (answer,                reference)
    qar → (question + answer,     reference)
    """
    answer = str(row["Answer_proc"])
    reference = str(row["Reference_proc"])
    if input_fmt == "qar":
        question = str(row["Question_proc"])
        return (question + " " + answer, reference)
    return (answer, reference)


def build_text_lists(df: pd.DataFrame, input_fmt: str):
    """Bulk equivalent of build_pair for classical models."""
    answers = df["Answer_proc"].astype(str).tolist()
    references = df["Reference_proc"].astype(str).tolist()
    if input_fmt == "qar":
        questions = df["Question_proc"].astype(str).tolist()
        side_a = [q + " " + a for q, a in zip(questions, answers)]
    else:
        side_a = answers
    return side_a, references


def build_single_pair(row, input_fmt: str):
    """For CrossEncoder: returns (text_a, text_b) to feed tokenizer(text_a, text_b).

    The tokenizer inserts [CLS]/[SEP] automatically and handles the segment ids.

    ra  → (reference,             answer)
    qar → (question, answer + " [SEP_TOK] " + reference)
        — but to keep this clean we instead pack as text_a=question,
          text_b=answer+reference; the tokenizer's own [SEP] separates them
          and the question/answer-reference asymmetry is what matters.
    """
    answer = str(row["Answer_proc"])
    reference = str(row["Reference_proc"])
    if input_fmt == "qar":
        question = str(row["Question_proc"])
        return (question, answer + " " + reference)
    return (reference, answer)


class PairDataset(Dataset):
    """Two-side tokenized dataset for DualEncoder.

    When `provide_max_score=True`, each batch also yields `max_score_feat`,
    a scalar in (0, 1] equal to `Max Score / MAX_SCORE_NORMALIZER`. Used by
    v03b neural-head max-score feature.
    """

    def __init__(self, df, tokenizer, input_fmt: str, max_len: int = C.TXFMR_MAX_LEN,
                 provide_max_score: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.input_fmt = input_fmt
        self.max_len = max_len
        self.scores = df["normalized_score"].values.astype(np.float32)
        self.labels = df["score_label"].values.astype(np.int64)
        self.provide_max_score = provide_max_score
        if provide_max_score:
            self.max_score_feat = (
                df["Max Score"].values.astype(np.float32) / float(C.MAX_SCORE_NORMALIZER)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        side_a, side_b = build_pair(self.df.iloc[idx], self.input_fmt)
        enc_a = self.tokenizer(
            side_a,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            side_b,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out = {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.provide_max_score:
            out["max_score_feat"] = torch.tensor([self.max_score_feat[idx]],
                                                  dtype=torch.float32)
        return out


class CrossDataset(Dataset):
    """Joint-input dataset for CrossEncoder — tokenizer handles [CLS]/[SEP]."""

    def __init__(self, df, tokenizer, input_fmt: str, max_len: int = C.TXFMR_MAX_LEN,
                 provide_max_score: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.input_fmt = input_fmt
        self.max_len = max_len
        self.scores = df["normalized_score"].values.astype(np.float32)
        self.labels = df["score_label"].values.astype(np.int64)
        self.provide_max_score = provide_max_score
        if provide_max_score:
            self.max_score_feat = (
                df["Max Score"].values.astype(np.float32) / float(C.MAX_SCORE_NORMALIZER)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_a, text_b = build_single_pair(self.df.iloc[idx], self.input_fmt)
        enc = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.provide_max_score:
            out["max_score_feat"] = torch.tensor([self.max_score_feat[idx]],
                                                  dtype=torch.float32)
        return out


class CharPairDataset(Dataset):
    """Character-level tokenized two-side dataset for BiLSTM."""

    def __init__(self, df, char2id, input_fmt: str, max_len: int = 256,
                 provide_max_score: bool = False):
        self.df = df.reset_index(drop=True)
        self.char2id = char2id
        self.input_fmt = input_fmt
        self.max_len = max_len
        self.scores = df["normalized_score"].values.astype(np.float32)
        self.labels = df["score_label"].values.astype(np.int64)
        self.provide_max_score = provide_max_score
        if provide_max_score:
            self.max_score_feat = (
                df["Max Score"].values.astype(np.float32) / float(C.MAX_SCORE_NORMALIZER)
            )

    def __len__(self):
        return len(self.df)

    def _encode(self, text: str):
        ids = [self.char2id.get(ch, 1) for ch in text[: self.max_len]]
        pad = self.max_len - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        side_a, side_b = build_pair(self.df.iloc[idx], self.input_fmt)
        ids_a, mask_a = self._encode(side_a)
        ids_b, mask_b = self._encode(side_b)
        out = {
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.provide_max_score:
            out["max_score_feat"] = torch.tensor([self.max_score_feat[idx]],
                                                  dtype=torch.float32)
        return out


def build_char_vocab(texts, max_vocab: int = C.BILSTM_VOCAB):
    """Build a top-K character vocab. 0 = PAD, 1 = UNK."""
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(str(t))
    most_common = [ch for ch, _ in counter.most_common(max_vocab - 2)]
    char2id = {"<pad>": 0, "<unk>": 1}
    for ch in most_common:
        char2id[ch] = len(char2id)
    return char2id
