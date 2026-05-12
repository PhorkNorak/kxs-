"""
Character-level tokenizer for BiLSTM. Mimics HuggingFace tokenizer interface
so it works with the same DataLoader code.
"""

import torch
from collections import Counter


class CharTokenizer:
    PAD, UNK, CLS, SEP = 0, 1, 2, 3

    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.c2i = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}

    def fit(self, texts):
        counter = Counter(c for t in texts for c in str(t))
        for c, _ in counter.most_common(self.max_vocab - 4):
            if c not in self.c2i:
                self.c2i[c] = len(self.c2i)
        return self

    @property
    def vocab_size(self):
        return len(self.c2i)

    @property
    def inv(self):
        return {v: k for k, v in self.c2i.items()}

    def encode(self, text, max_len=128):
        ids = [self.CLS] + [self.c2i.get(c, self.UNK) for c in str(text)] + [self.SEP]
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.SEP]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.PAD)
            mask.append(0)
        return ids, mask

    def convert_ids_to_tokens(self, ids):
        inv = self.inv
        return [inv.get(i, "<UNK>") for i in ids]

    def __call__(self, text, max_length=128, padding="max_length",
                 truncation=True, return_tensors="pt"):
        ids, mask = self.encode(text, max_len=max_length)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": mask}
