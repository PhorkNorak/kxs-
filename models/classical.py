"""Classical baselines for the simple pipeline.

All three operate on the (side_a, side_b) text pair produced by `build_text_lists`:
  - ra:  side_a = answer,            side_b = reference
  - qar: side_a = question + answer, side_b = reference

Predictions are returned as continuous scores in [0,1] (the regression target).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity

import config as C


def _cosine_rowwise(a, b):
    """Row-wise cosine similarity for two (n, d) arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    sims = []
    for i in range(a.shape[0]):
        sims.append(cosine_similarity(a[i : i + 1], b[i : i + 1])[0, 0])
    return np.array(sims)


class TFIDFCosine:
    """Char-wb TF-IDF + cosine similarity. No training (fit on train corpus only)."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            analyzer=C.TFIDF_ANALYZER,
            ngram_range=C.TFIDF_NGRAMS,
            max_features=C.TFIDF_MAX_FEAT,
        )

    def fit(self, train_a, train_b, train_scores):
        self.vec.fit(list(train_a) + list(train_b))
        return self

    def predict(self, side_a, side_b):
        a = self.vec.transform(side_a)
        b = self.vec.transform(side_b)
        sims = _cosine_rowwise(a.toarray(), b.toarray())
        return np.clip(sims, 0.0, 1.0)


class TFIDFSVR:
    """Char-wb TF-IDF features → SVR(RBF) on normalized score in [0,1]."""

    def __init__(self, C_svr: float = C.SVR_C):
        self.vec = TfidfVectorizer(
            analyzer=C.TFIDF_ANALYZER,
            ngram_range=C.TFIDF_NGRAMS,
            max_features=C.TFIDF_MAX_FEAT,
        )
        self.svr = SVR(kernel="rbf", C=C_svr)

    def _features(self, side_a, side_b):
        a = self.vec.transform(side_a).toarray()
        b = self.vec.transform(side_b).toarray()
        cos = _cosine_rowwise(a, b).reshape(-1, 1)
        return np.hstack([a, b, np.abs(a - b), a * b, cos])

    def fit(self, train_a, train_b, train_scores):
        self.vec.fit(list(train_a) + list(train_b))
        X = self._features(train_a, train_b)
        self.svr.fit(X, np.asarray(train_scores, dtype=np.float32))
        return self

    def predict(self, side_a, side_b):
        X = self._features(side_a, side_b)
        return np.clip(self.svr.predict(X), 0.0, 1.0)


class FastTextCosine:
    """Char-as-token FastText embeddings → mean-pool → cosine similarity."""

    def __init__(self, dim: int = C.FASTTEXT_DIM, epochs: int = C.FASTTEXT_EPOCHS):
        self.dim = dim
        self.epochs = epochs
        self.model = None

    def fit(self, train_a, train_b, train_scores):
        from gensim.models import FastText as FT

        corpus = [list(t) for t in list(train_a) + list(train_b) if str(t).strip()]
        self.model = FT(
            sentences=corpus,
            vector_size=self.dim,
            min_count=1,
            epochs=self.epochs,
            window=3,
            sg=1,
            workers=1,
            seed=C.SEED,
        )
        return self

    def _embed(self, text: str):
        chars = list(str(text))
        if not chars or self.model is None:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = [self.model.wv.get_vector(c) for c in chars if c in self.model.wv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def predict(self, side_a, side_b):
        ea = np.array([self._embed(t) for t in side_a])
        eb = np.array([self._embed(t) for t in side_b])
        sims = _cosine_rowwise(ea, eb)
        return np.clip(sims, 0.0, 1.0)


def make_classical(model_id: str):
    if model_id == "tfidf_cos":
        return TFIDFCosine()
    if model_id == "tfidf_svr":
        return TFIDFSVR()
    if model_id == "fasttext_cos":
        return FastTextCosine()
    raise ValueError(f"Unknown classical model: {model_id}")
