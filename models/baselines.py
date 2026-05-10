"""
KhmerXScore Baseline Models
==============================
5 baselines establishing the performance floor:
1. Mean Predictor — statistical floor
2. TF-IDF + Cosine Similarity — lexical unsupervised
3. TF-IDF + SVR — classical supervised ML
4. FastText + Cosine Similarity — static embedding baseline
5. BiLSTM + Attention — pre-transformer neural
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict


# ============================================================
# 1. Mean Predictor (Statistical floor)
# ============================================================
class MeanPredictor:
    """Predicts the training set mean score for all samples."""
    
    def __init__(self):
        self.mean_score = 0.0
        self.mean_label = 0
    
    def fit(self, dataset):
        self.mean_score = float(np.mean(dataset.scores))
        self.mean_label = int(round(self.mean_score * 4))
    
    def predict(self, dataset) -> Dict:
        n = len(dataset.scores)
        return {
            "scores": np.full(n, self.mean_score),
            "labels": np.full(n, self.mean_label, dtype=np.int64),
        }


# ============================================================
# 2. TF-IDF + Cosine Similarity (Lexical unsupervised)
# ============================================================
class TFIDFCosineScorer:
    """
    Compute TF-IDF vectors for answers and references,
    then use cosine similarity as the predicted score.
    No training required (unsupervised).
    """
    
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer="char_wb",    # Character n-grams work better for Khmer
            ngram_range=(2, 4),
        )
        self.fitted = False
    
    def fit(self, dataset):
        # Fit on all answers + references
        all_texts = dataset.answers + dataset.references
        self.vectorizer.fit(all_texts)
        self.fitted = True
    
    def predict(self, dataset) -> Dict:
        a_vecs = self.vectorizer.transform(dataset.answers)
        r_vecs = self.vectorizer.transform(dataset.references)
        
        # Pairwise cosine similarity (row-wise)
        scores = np.array([
            cosine_similarity(a_vecs[i:i+1], r_vecs[i:i+1])[0, 0]
            for i in range(len(dataset.answers))
        ])
        scores = np.clip(scores, 0, 1)
        labels = np.round(scores * 4).astype(np.int64).clip(0, 4)
        
        return {"scores": scores, "labels": labels}


# ============================================================
# 3. TF-IDF + SVR (Classical supervised)
# ============================================================
class TFIDFSVRScorer:
    """
    TF-IDF features for (answer, reference), concatenated with cosine similarity.
    SVR regressor trained on normalized scores.
    """
    
    def __init__(self, max_features: int = 10000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer="char_wb",
            ngram_range=(2, 4),
        )
        self.svr = SVR(kernel="rbf", C=C)
    
    def _build_features(self, dataset) -> np.ndarray:
        a_vecs = self.vectorizer.transform(dataset.answers).toarray()
        r_vecs = self.vectorizer.transform(dataset.references).toarray()
        
        # Cosine similarity as additional feature
        cos_sim = np.array([
            cosine_similarity(a_vecs[i:i+1], r_vecs[i:i+1])[0, 0]
            for i in range(len(dataset.answers))
        ]).reshape(-1, 1)
        
        # Concatenate: [a_tfidf, r_tfidf, |a-r|, a*r, cos_sim]
        diff = np.abs(a_vecs - r_vecs)
        prod = a_vecs * r_vecs
        
        return np.hstack([a_vecs, r_vecs, diff, prod, cos_sim])
    
    def fit(self, dataset):
        all_texts = dataset.answers + dataset.references
        self.vectorizer.fit(all_texts)
        
        X = self._build_features(dataset)
        y = dataset.scores
        self.svr.fit(X, y)
    
    def predict(self, dataset) -> Dict:
        X = self._build_features(dataset)
        scores = self.svr.predict(X)
        scores = np.clip(scores, 0, 1)
        labels = np.round(scores * 4).astype(np.int64).clip(0, 4)
        return {"scores": scores, "labels": labels}


# ============================================================
# 4. FastText + Cosine Similarity (Static embedding baseline)
# ============================================================
class FastTextCosineScorer:
    """
    Mean-pooled FastText word vectors, cosine similarity as score.
    Uses gensim's FastText model (handles OOV via subword).
    """
    
    def __init__(self, dim: int = 100, min_count: int = 1, epochs: int = 10):
        self.dim = dim
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, dataset):
        from gensim.models import FastText as FT
        
        # Tokenize by spaces (assumes segmented text)
        corpus = [text.split() for text in dataset.answers + dataset.references]
        
        self.model = FT(
            sentences=corpus,
            vector_size=self.dim,
            min_count=self.min_count,
            epochs=self.epochs,
            window=5,
            sg=1,
        )
    
    def _embed(self, text: str) -> np.ndarray:
        tokens = text.split()
        if not tokens:
            return np.zeros(self.dim)
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vectors:
            # Use subword fallback
            vectors = [self.model.wv.get_vector(t) for t in tokens]
        return np.mean(vectors, axis=0)
    
    def predict(self, dataset) -> Dict:
        a_embs = np.array([self._embed(t) for t in dataset.answers])
        r_embs = np.array([self._embed(t) for t in dataset.references])
        
        # Pairwise cosine similarity
        scores = np.array([
            cosine_similarity(a_embs[i:i+1], r_embs[i:i+1])[0, 0]
            for i in range(len(dataset.answers))
        ])
        scores = np.clip(scores, 0, 1)
        labels = np.round(scores * 4).astype(np.int64).clip(0, 4)
        
        return {"scores": scores, "labels": labels}


# ============================================================
# 5. BiLSTM + Attention (Pre-transformer neural)
# ============================================================
class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with self-attention for answer/reference encoding.
    Uses dual-encoder topology with 4-way interaction, matching the 
    proposed KX-CL architecture but with BiLSTM instead of transformer.
    """
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_classes: int = 5,
                 loss_type: str = "corn"):
        super().__init__()
        
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 4-way interaction: 4 * (hidden_dim * 2)
        interaction_dim = 4 * hidden_dim * 2
        
        if loss_type == "corn":
            output_dim = num_classes - 1
        else:
            output_dim = 1
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )
    
    def _attend(self, lstm_out: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Self-attention pooling."""
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
    
    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        return self._attend(lstm_out, attention_mask)
    
    def forward(self, input_ids_a, attention_mask_a,
                input_ids_r, attention_mask_r,
                return_embeddings=False):
        e_a = self.encode(input_ids_a, attention_mask_a)
        e_r = self.encode(input_ids_r, attention_mask_r)
        
        interaction = torch.cat([e_a, e_r, e_a - e_r, e_a * e_r], dim=1)
        logits = self.head(interaction)
        
        if return_embeddings:
            return logits, e_a
        return logits
    
    def predict_label(self, input_ids_a, attention_mask_a,
                      input_ids_r, attention_mask_r):
        logits = self.forward(input_ids_a, attention_mask_a,
                              input_ids_r, attention_mask_r)
        if self.loss_type == "corn":
            from models.losses import corn_logits_to_label
            return corn_logits_to_label(logits)
        else:
            score = logits.squeeze(-1).clamp(0, 1)
            return (score * 4).round().long().clamp(0, 4)


# ============================================================
# Factory
# ============================================================
def create_baseline(name: str, **kwargs):
    """Create a baseline model by name."""
    baselines = {
        "mean_predictor": MeanPredictor,
        "tfidf_cosine": TFIDFCosineScorer,
        "tfidf_svr": TFIDFSVRScorer,
        "fasttext_cosine": FastTextCosineScorer,
    }
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines.keys())}")
    return baselines[name](**kwargs)
