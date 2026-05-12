"""
Baseline Models (5 total)
1. Mean Predictor — statistical floor
2. TF-IDF + Cosine — lexical unsupervised
3. TF-IDF + SVR — classical supervised ML
4. FastText + Cosine — static embedding baseline
5. BiLSTM + Attention — pre-transformer neural (dual-encoder with 4-way interaction)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity


# ── 1. Mean Predictor ──────────────────────────────────────────────────
class MeanPredictor:
    def __init__(self):
        self.mean_score = 0.0

    def fit(self, dataset):
        self.mean_score = float(np.mean(dataset.scores))

    def predict(self, dataset):
        n = len(dataset.scores)
        ps = np.full(n, self.mean_score)
        pl = np.round(ps * 4).astype(np.int64).clip(0, 4)
        return {"scores": ps, "labels": pl}


# ── 2. TF-IDF + Cosine ────────────────────────────────────────────────
class TFIDFCosineScorer:
    def __init__(self, max_features=15000):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   max_features=max_features)

    def fit(self, dataset):
        self.vec.fit(dataset.answers + dataset.references)

    def predict(self, dataset):
        a = self.vec.transform(dataset.answers)
        r = self.vec.transform(dataset.references)
        sims = np.array([cosine_similarity(a[i:i+1], r[i:i+1])[0, 0]
                         for i in range(len(dataset.answers))])
        ps = np.clip(sims, 0, 1)
        pl = np.round(ps * 4).astype(np.int64).clip(0, 4)
        return {"scores": ps, "labels": pl}


# ── 3. TF-IDF + SVR ───────────────────────────────────────────────────
class TFIDFSVRScorer:
    def __init__(self, max_features=15000, C=1.0):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   max_features=max_features)
        self.svr = SVR(kernel="rbf", C=C)

    def _features(self, answers, references):
        a = self.vec.transform(answers).toarray()
        r = self.vec.transform(references).toarray()
        cos = np.array([cosine_similarity(a[i:i+1], r[i:i+1])[0, 0]
                        for i in range(len(answers))]).reshape(-1, 1)
        return np.hstack([a, r, np.abs(a - r), a * r, cos])

    def fit(self, dataset):
        from models.losses import compute_class_weights
        self.vec.fit(dataset.answers + dataset.references)
        X = self._features(dataset.answers, dataset.references)
        cw = compute_class_weights(dataset.labels).numpy()
        sw = cw[dataset.labels]
        self.svr.fit(X, dataset.scores, sample_weight=sw)

    def predict(self, dataset):
        X = self._features(dataset.answers, dataset.references)
        ps = np.clip(self.svr.predict(X), 0, 1)
        pl = np.round(ps * 4).astype(np.int64).clip(0, 4)
        return {"scores": ps, "labels": pl}


# ── 4. FastText + Cosine ──────────────────────────────────────────────
class FastTextCosineScorer:
    def __init__(self, dim=100, epochs=10):
        self.dim = dim
        self.epochs = epochs
        self.model = None

    def fit(self, dataset):
        from gensim.models import FastText as FT
        corpus = [list(t) for t in dataset.answers + dataset.references if t.strip()]
        self.model = FT(sentences=corpus, vector_size=self.dim, min_count=1,
                        epochs=self.epochs, window=3, sg=1, workers=1, seed=42)

    def _embed(self, text):
        chars = list(str(text))
        if not chars:
            return np.zeros(self.dim)
        vecs = [self.model.wv.get_vector(c) for c in chars if c in self.model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.dim)

    def predict(self, dataset):
        ae = np.array([self._embed(t) for t in dataset.answers])
        re = np.array([self._embed(t) for t in dataset.references])
        sims = np.array([cosine_similarity(ae[i:i+1], re[i:i+1])[0, 0]
                         for i in range(len(ae))])
        ps = np.clip(sims, 0, 1)
        pl = np.round(ps * 4).astype(np.int64).clip(0, 4)
        return {"scores": ps, "labels": pl}


# ── 5. BiLSTM + Attention ─────────────────────────────────────────────
class BiLSTMAttention(nn.Module):
    """Dual-encoder BiLSTM with self-attention and 4-way interaction.
    Architecture mirrors KX-CL but uses BiLSTM instead of transformer."""

    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=5, loss_type="corn"):
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.scl_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        out = (num_classes - 1) if loss_type == "corn" else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 8, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out),
        )

    def encode(self, ids, mask):
        x, _ = self.lstm(self.emb(ids))
        w = self.attn(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return (torch.softmax(w, 1).unsqueeze(1) @ x).squeeze(1)

    def forward(self, input_ids_a, attention_mask_a, input_ids_r, attention_mask_r,
                return_embeddings=False):
        ea = self.encode(input_ids_a, attention_mask_a)
        er = self.encode(input_ids_r, attention_mask_r)
        inter = torch.cat([ea, er, ea - er, ea * er], dim=1)
        logits = self.head(inter)
        if return_embeddings:
            return logits, self.scl_proj(ea)
        return logits

    def forward_from_emb(self, emb_a, mask_a, input_ids_r, attention_mask_r):
        """Forward using pre-computed embeddings (for gradient saliency)."""
        x, _ = self.lstm(emb_a)
        w = self.attn(x).squeeze(-1).masked_fill(mask_a == 0, -1e9)
        ea = (torch.softmax(w, 1).unsqueeze(1) @ x).squeeze(1)
        er = self.encode(input_ids_r, attention_mask_r)
        return self.head(torch.cat([ea, er, ea - er, ea * er], dim=1))

    def predict_with_uncertainty(self, input_ids_a, attention_mask_a,
                                 input_ids_r, attention_mask_r, T=10):
        from models.losses import corn_logits_to_label, corn_logits_to_score
        self.train(True)  # keep dropout active during inference
        scores_list, labels_list = [], []
        with torch.no_grad():
            for _ in range(T):
                logits = self.forward(input_ids_a, attention_mask_a,
                                      input_ids_r, attention_mask_r)
                if self.loss_type == "corn":
                    scores_list.append(corn_logits_to_score(logits, self.num_classes))
                    labels_list.append(corn_logits_to_label(logits))
                else:
                    s = logits.squeeze(-1).clamp(0, 1)
                    scores_list.append(s)
                    labels_list.append((s * 4).round().long().clamp(0, 4))
        self.train(False)  # restore inference mode
        st = torch.stack(scores_list)
        lt = torch.stack(labels_list)
        return {
            "mean_score": st.mean(0), "std_score": st.std(0),
            "mode_label": lt.mode(0).values, "raw_scores": st,
        }


# ── Factory ────────────────────────────────────────────────────────────
def create_baseline(name, **kwargs):
    return {
        "mean_predictor": MeanPredictor,
        "tfidf_cosine": TFIDFCosineScorer,
        "tfidf_svr": TFIDFSVRScorer,
        "fasttext_cosine": FastTextCosineScorer,
    }[name](**kwargs)
