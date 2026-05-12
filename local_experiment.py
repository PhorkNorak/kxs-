"""
KhmerXScore Local Experiment
==============================
Runs ALL non-HuggingFace models on the full KhmerSAG dataset (1,184 samples).
Produces real thesis-quality baseline results.

Models:
  Classical:  Mean predictor, TF-IDF+Cosine, TF-IDF+SVR, FastText+Cosine
  Neural:     BiLSTM+Attention (ar) and BiLSTM+Attention (qar)

Input formats tested: ar (Answer+Reference) and qar (Question+Answer+Reference)

Output: results/local_experiment_results.json + printed summary table
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (cohen_kappa_score, mean_squared_error,
                              mean_absolute_error, f1_score, confusion_matrix)
from collections import Counter
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
DEPLOY_THRESHOLD = 0.70
BOOTSTRAP_N = 1000

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ============================================================
# 1. DATA LOADING & SPLITTING
# ============================================================
def load_and_split(csv_path: str, seed: int = 42):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Subject"]   = df["Subject"].str.strip().replace({"History ": "History"})
    df["Reference"] = df[" Reference"] if " Reference" in df.columns else df["Reference"]

    # Drop if Reference column was the old name
    if " Reference" in df.columns:
        df = df.drop(columns=[" Reference"])

    df["normalized_score"] = df["Student Score"] / df["Max Score"]
    df["score_label"]      = (df["normalized_score"] * 4).round().astype(int).clip(0, 4)

    # Drop missing text
    before = len(df)
    df = df.dropna(subset=["Question", "Reference", "Answer"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing text")

    # Clean newlines
    for col in ["Question", "Reference", "Answer"]:
        df[col] = df[col].astype(str).str.replace(r"\n", " ", regex=True).str.strip()

    # Stratified split
    min_count = df["score_label"].value_counts().min()
    strat = df["score_label"] if min_count >= 4 else None
    train_df, temp = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)
    s2 = temp["score_label"] if (strat is not None and temp["score_label"].value_counts().min() >= 2) else None
    val_df, test_df = train_test_split(temp, test_size=0.50, random_state=seed, stratify=s2)

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"\nDataset: {len(df)} samples")
    print(f"Split:   train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Score distribution (0-4):")
    for lbl, cnt in sorted(df["score_label"].value_counts().items()):
        pct = cnt/len(df)*100
        print(f"  {lbl}: {cnt:4d}  ({pct:5.1f}%)  {'█'*int(pct/2)}")

    return train_df, val_df, test_df


# ============================================================
# 2. TEXT PREPARATION
# ============================================================
def build_texts(df, fmt):
    """Build answer-side text and reference text based on input format."""
    if fmt == "qar":
        answers = (df["Question"].fillna("") + " " + df["Answer"].fillna("")).tolist()
    else:
        answers = df["Answer"].fillna("").tolist()
    refs    = df["Reference"].fillna("").tolist()
    scores  = df["normalized_score"].values.astype(np.float32)
    labels  = df["score_label"].values.astype(np.int64)
    return answers, refs, scores, labels


# ============================================================
# 3. METRICS
# ============================================================
def metrics(y_true_l, y_pred_l, y_true_s, y_pred_s):
    """Compute full metric suite."""
    m = {}
    m["qwk"]      = float(cohen_kappa_score(y_true_l, y_pred_l, weights="quadratic"))
    m["rmse"]     = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
    m["mae"]      = float(mean_absolute_error(y_true_s, y_pred_s))
    m["accuracy"] = float(np.mean(y_true_l == y_pred_l))
    m["adj_agr"]  = float(np.mean(np.abs(y_true_l - y_pred_l) <= 1))
    if len(np.unique(y_pred_s)) > 1:
        m["pearson"], _  = pearsonr(y_true_s, y_pred_s)
        m["spearman"], _ = spearmanr(y_true_s, y_pred_s)
        m["pearson"]  = float(m["pearson"])
        m["spearman"] = float(m["spearman"])
    else:
        m["pearson"] = m["spearman"] = 0.0
    m["f1_w"] = float(f1_score(y_true_l, y_pred_l, average="weighted", zero_division=0))
    per = f1_score(y_true_l, y_pred_l, average=None, labels=[0,1,2,3,4], zero_division=0)
    for i, v in enumerate(per):
        m[f"f1_{i}"] = float(v)
    m["confusion_matrix"] = confusion_matrix(y_true_l, y_pred_l,
                                              labels=[0,1,2,3,4]).tolist()
    return m


def bootstrap_qwk(y_true, y_pred, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            vals.append(cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic"))
        except Exception:
            pass
    vals = np.array(vals)
    pt   = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return float(pt), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(y_true, pa, pb, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    obs = cohen_kappa_score(y_true, pa, weights="quadratic") - \
          cohen_kappa_score(y_true, pb, weights="quadratic")
    count = 0
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            da = cohen_kappa_score(y_true[idx], pa[idx], weights="quadratic")
            db = cohen_kappa_score(y_true[idx], pb[idx], weights="quadratic")
            if da - db > 0:
                count += 1
        except Exception:
            pass
    p = 2 * min(count/n, 1 - count/n)
    return float(obs), float(p)


# ============================================================
# 4. CLASSICAL BASELINES
# ============================================================
def run_mean(train_a, train_r, train_s, train_l, test_a, test_r, test_s, test_l):
    mean_s = float(np.mean(train_s))
    pred_s = np.full(len(test_s), mean_s)
    pred_l = np.round(pred_s * 4).astype(np.int64).clip(0, 4)
    return pred_s, pred_l


def run_tfidf_cosine(train_a, train_r, train_s, train_l, test_a, test_r, test_s, test_l):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=15000)
    vec.fit(train_a + train_r)
    ta = vec.transform(test_a)
    tr = vec.transform(test_r)
    sims = np.array([cosine_similarity(ta[i:i+1], tr[i:i+1])[0,0] for i in range(len(test_a))])
    pred_s = np.clip(sims, 0, 1)
    pred_l = np.round(pred_s * 4).astype(np.int64).clip(0, 4)
    return pred_s, pred_l


def run_tfidf_svr(train_a, train_r, train_s, train_l, test_a, test_r, test_s, test_l):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=15000)
    vec.fit(train_a + train_r)

    def featurize(answers, refs):
        av = vec.transform(answers).toarray()
        rv = vec.transform(refs).toarray()
        cos = np.array([cosine_similarity(av[i:i+1], rv[i:i+1])[0,0]
                        for i in range(len(answers))]).reshape(-1,1)
        diff = np.abs(av - rv)
        prod = av * rv
        return np.hstack([av, rv, diff, prod, cos])

    X_tr = featurize(train_a, train_r)
    X_te = featurize(test_a, test_r)

    svr = SVR(kernel="rbf", C=1.0)
    svr.fit(X_tr, train_s)
    pred_s = np.clip(svr.predict(X_te), 0, 1)
    pred_l = np.round(pred_s * 4).astype(np.int64).clip(0, 4)
    return pred_s, pred_l


def run_fasttext(train_a, train_r, train_s, train_l, test_a, test_r, test_s, test_l):
    from gensim.models import FastText
    corpus = [t.split() for t in train_a + train_r if t.strip()]
    if not corpus:
        return np.zeros(len(test_s)), np.zeros(len(test_s), dtype=np.int64)

    model = FastText(sentences=corpus, vector_size=100, min_count=1,
                     epochs=15, window=5, sg=1, workers=1, seed=SEED)

    def embed(text):
        toks = text.split()
        if not toks:
            return np.zeros(100)
        return np.mean([model.wv.get_vector(t) for t in toks], axis=0)

    ae = np.array([embed(t) for t in test_a])
    re = np.array([embed(t) for t in test_r])
    sims = np.array([cosine_similarity(ae[i:i+1], re[i:i+1])[0,0] for i in range(len(ae))])
    pred_s = np.clip(sims, 0, 1)
    pred_l = np.round(pred_s * 4).astype(np.int64).clip(0, 4)
    return pred_s, pred_l


# ============================================================
# 5. CHAR TOKENIZER
# ============================================================
class CharTokenizer:
    PAD, UNK, CLS, SEP = 0, 1, 2, 3

    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.char2idx = {"<PAD>":0,"<UNK>":1,"<CLS>":2,"<SEP>":3}
        self.fitted = False

    def fit(self, texts):
        counter = Counter(ch for t in texts for ch in str(t))
        for ch, _ in counter.most_common(self.max_vocab - 4):
            if ch not in self.char2idx:
                self.char2idx[ch] = len(self.char2idx)
        self.fitted = True
        return self

    @property
    def vocab_size(self): return len(self.char2idx)

    def encode(self, text, max_len=128):
        ids = [self.CLS] + [self.char2idx.get(c, self.UNK) for c in str(text)] + [self.SEP]
        if len(ids) > max_len:
            ids = ids[:max_len-1] + [self.SEP]
        mask = [1]*len(ids)
        while len(ids) < max_len:
            ids.append(self.PAD); mask.append(0)
        return ids, mask


# ============================================================
# 6. BILSTM DATASET & MODEL
# ============================================================
class PairDataset(Dataset):
    def __init__(self, answers, refs, scores, labels, tok, max_len=128):
        self.a, self.r = answers, refs
        self.scores, self.labels = scores, labels
        self.tok, self.max_len = tok, max_len

    def __len__(self): return len(self.a)

    def __getitem__(self, i):
        ids_a, mask_a = self.tok.encode(self.a[i], self.max_len)
        ids_r, mask_r = self.tok.encode(self.r[i], self.max_len)
        return {
            "input_ids_a":      torch.tensor(ids_a,  dtype=torch.long),
            "attention_mask_a": torch.tensor(mask_a, dtype=torch.long),
            "input_ids_r":      torch.tensor(ids_r,  dtype=torch.long),
            "attention_mask_r": torch.tensor(mask_r, dtype=torch.long),
            "score":            torch.tensor(self.scores[i], dtype=torch.float32),
            "label":            torch.tensor(self.labels[i], dtype=torch.long),
        }


class BiLSTMDual(nn.Module):
    """
    Dual-encoder BiLSTM+Attention with 4-way interaction and CORN head.
    Architecture mirrors KX-CL but uses BiLSTM instead of transformer.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden=128,
                 n_layers=2, dropout=0.2, num_classes=5, loss_type="corn"):
        super().__init__()
        self.loss_type   = loss_type
        self.num_classes = num_classes

        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        out = (num_classes-1) if loss_type=="corn" else 1
        interaction_dim = 4 * hidden * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out),
        )
        self.scl_proj = nn.Sequential(
            nn.Linear(hidden*2, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def encode(self, ids, mask):
        x, _ = self.lstm(self.emb(ids))
        w = self.attn(x).squeeze(-1)
        w = w.masked_fill(mask == 0, -1e9)
        w = torch.softmax(w, dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)

    def forward(self, input_ids_a, attention_mask_a,
                input_ids_r, attention_mask_r, return_embeddings=False):
        ea = self.encode(input_ids_a, attention_mask_a)
        er = self.encode(input_ids_r, attention_mask_r)
        inter = torch.cat([ea, er, ea-er, ea*er], dim=1)
        logits = self.head(inter)
        if return_embeddings:
            return logits, self.scl_proj(ea)
        return logits


# ============================================================
# 7. CORN + SCL LOSSES
# ============================================================
class CORNLoss(nn.Module):
    def __init__(self, K=5):
        super().__init__()
        self.K = K

    def forward(self, logits, labels):
        losses = []
        for k in range(self.K - 1):
            mask = labels >= k
            if mask.sum() == 0: continue
            bl = (labels[mask] > k).float()
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                logits[mask, k], bl))
        return torch.stack(losses).mean() if losses else logits.sum() * 0


class SCLoss(nn.Module):
    def __init__(self, temperature=0.07, threshold=0.1):
        super().__init__()
        self.T = temperature
        self.thr = threshold

    def forward(self, emb, scores):
        if emb.shape[0] < 2: return torch.tensor(0., device=emb.device)
        emb = torch.nn.functional.normalize(emb, dim=1)
        sim = torch.matmul(emb, emb.T) / self.T
        diff = torch.abs(scores.unsqueeze(1) - scores.unsqueeze(0))
        pos  = (diff < self.thr).float()
        eye  = torch.eye(emb.shape[0], device=emb.device)
        pos  = pos * (1 - eye)
        mx, _ = sim.max(dim=1, keepdim=True)
        logits = sim - mx.detach()
        exp = torch.exp(logits) * (1-eye)
        log_prob = logits - torch.log(exp.sum(1, keepdim=True) + 1e-8)
        npos = pos.sum(1).clamp(min=1)
        loss = -(pos * log_prob).sum(1) / npos
        return loss.mean()


def corn_to_label(logits, K=5):
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(dim=1)

def corn_to_score(logits, K=5):
    return corn_to_label(logits, K).float() / (K-1)


# ============================================================
# 8. BALANCED SAMPLER
# ============================================================
def balanced_sampler(labels):
    counts = np.bincount(labels, minlength=5).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = (1.0 / counts)[labels]
    return WeightedRandomSampler(torch.from_numpy(w).float(), len(labels), True)


# ============================================================
# 9. BILSTM TRAINING LOOP
# ============================================================
def train_bilstm(model, tr_loader, va_loader, test_loader,
                 run_name="bilstm", epochs=30, lr=1e-3, patience=5,
                 alpha=1.0, beta=0.5, device=DEVICE):
    model = model.to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    corn  = CORNLoss(5)
    scl   = SCLoss()

    best_qwk = -1.0
    no_imp   = 0
    history  = []

    ckpt = os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pt")

    for ep in range(epochs):
        model.train()
        ep_loss = []

        for batch in tr_loader:
            ids_a  = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_r  = batch["input_ids_r"].to(device)
            mask_r = batch["attention_mask_r"].to(device)
            scores = batch["score"].to(device)
            labels = batch["label"].to(device)

            opt.zero_grad()
            logits, emb = model(ids_a, mask_a, ids_r, mask_r, return_embeddings=True)

            l_corn = corn(logits, labels)
            l_scl  = scl(emb, scores)
            loss   = alpha * l_corn + beta * l_scl

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss.append(loss.item())

        # Validate
        model.eval()
        vl_true, vl_pred = [], []
        with torch.no_grad():
            for batch in va_loader:
                ids_a  = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                ids_r  = batch["input_ids_r"].to(device)
                mask_r = batch["attention_mask_r"].to(device)
                logits = model(ids_a, mask_a, ids_r, mask_r)
                vl_true.extend(batch["label"].numpy())
                vl_pred.extend(corn_to_label(logits).cpu().numpy())

        vl_true = np.array(vl_true)
        vl_pred = np.array(vl_pred)
        vqwk = float(cohen_kappa_score(vl_true, vl_pred, weights="quadratic"))

        history.append({"epoch": ep+1, "loss": np.mean(ep_loss), "val_qwk": vqwk})

        if vqwk > best_qwk:
            best_qwk = vqwk
            no_imp = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop at epoch {ep+1}  best_val_qwk={best_qwk:.4f}")
                break

        if (ep+1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1:2d}  loss={np.mean(ep_loss):.4f}  val_qwk={vqwk:.4f}  best={best_qwk:.4f}")

    # Load best
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))

    # Test
    model.eval()
    all_tl, all_pl, all_ts, all_ps = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids_a  = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_r  = batch["input_ids_r"].to(device)
            mask_r = batch["attention_mask_r"].to(device)
            logits = model(ids_a, mask_a, ids_r, mask_r)
            all_tl.extend(batch["label"].numpy())
            all_pl.extend(corn_to_label(logits).cpu().numpy())
            all_ts.extend(batch["score"].numpy())
            all_ps.extend(corn_to_score(logits).cpu().numpy())

    return (np.array(all_tl), np.array(all_pl),
            np.array(all_ts), np.array(all_ps),
            best_qwk, history)


# ============================================================
# 10. PER-SUBJECT BREAKDOWN
# ============================================================
def per_subject(test_df, pred_labels, pred_scores, min_n=30):
    results = {}
    for subj in test_df["Subject"].unique():
        mask = test_df["Subject"].values == subj
        n = mask.sum()
        if n < min_n:
            results[subj] = {"n": int(n), "qwk": None, "note": f"n<{min_n}"}
            continue
        try:
            q = float(cohen_kappa_score(
                test_df["score_label"].values[mask],
                pred_labels[mask], weights="quadratic"))
        except Exception:
            q = None
        r = float(np.sqrt(mean_squared_error(
            test_df["normalized_score"].values[mask], pred_scores[mask])))
        results[subj] = {"n": int(n), "qwk": q, "rmse": r}
    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "dataset.csv")
    train_df, val_df, test_df = load_and_split(csv, SEED)

    all_results = {}
    all_preds   = {}   # Store predictions for significance testing

    input_formats = ["ar", "qar"]

    # ── CLASSICAL BASELINES ──────────────────────────────────
    print("\n" + "="*60)
    print("CLASSICAL BASELINES")
    print("="*60)

    classical = {
        "mean_predictor": run_mean,
        "tfidf_cosine":   run_tfidf_cosine,
        "tfidf_svr":      run_tfidf_svr,
        "fasttext_cosine": run_fasttext,
    }

    for fmt in input_formats:
        tr_a, tr_r, tr_s, tr_l = build_texts(train_df, fmt)
        te_a, te_r, te_s, te_l = build_texts(test_df,  fmt)

        for name, fn in classical.items():
            key = f"{name}_{fmt}"
            print(f"\n  {key}")
            t0 = time.time()

            ps, pl = fn(tr_a, tr_r, tr_s, tr_l, te_a, te_r, te_s, te_l)
            elapsed = time.time() - t0

            m = metrics(te_l, pl, te_s, ps)
            qp, ql, qh = bootstrap_qwk(te_l, pl, BOOTSTRAP_N)
            m.update({"qwk_ci_95": [ql, qh], "elapsed_sec": round(elapsed, 2),
                      "input_format": fmt,
                      "deploy": m["qwk"] >= DEPLOY_THRESHOLD})

            all_results[key] = m
            all_preds[key]   = pl

            flag = "✓" if m["qwk"] >= DEPLOY_THRESHOLD else "✗"
            print(f"    QWK={m['qwk']:.4f} [{ql:.4f},{qh:.4f}]  "
                  f"RMSE={m['rmse']:.4f}  Pearson={m['pearson']:.4f}  "
                  f"F1={m['f1_w']:.4f}  [{flag}]  ({elapsed:.1f}s)")

    # ── BILSTM + ATTENTION ───────────────────────────────────
    print("\n" + "="*60)
    print("BILSTM + ATTENTION (Neural Baseline)")
    print("="*60)

    for fmt in input_formats:
        key = f"bilstm_{fmt}"
        print(f"\n  {key}")

        tr_a, tr_r, tr_s, tr_l = build_texts(train_df, fmt)
        va_a, va_r, va_s, va_l = build_texts(val_df,   fmt)
        te_a, te_r, te_s, te_l = build_texts(test_df,  fmt)

        # Build character vocabulary
        tok = CharTokenizer(max_vocab=5000)
        tok.fit(tr_a + tr_r)
        print(f"    Vocab size: {tok.vocab_size}")

        MAX_LEN = 128
        BATCH   = 32

        tr_ds = PairDataset(tr_a, tr_r, tr_s, tr_l, tok, MAX_LEN)
        va_ds = PairDataset(va_a, va_r, va_s, va_l, tok, MAX_LEN)
        te_ds = PairDataset(te_a, te_r, te_s, te_l, tok, MAX_LEN)

        sampler = balanced_sampler(tr_l)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
        te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0)

        torch.manual_seed(SEED)
        model = BiLSTMDual(
            vocab_size=tok.vocab_size,
            embed_dim=128, hidden=128,
            n_layers=2, dropout=0.3,
            num_classes=5, loss_type="corn",
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {n_params:,}")

        t0 = time.time()
        tl, pl, ts, ps, best_val, history = train_bilstm(
            model, tr_ld, va_ld, te_ld,
            run_name=key, epochs=30, lr=1e-3, patience=5,
            alpha=1.0, beta=0.5, device=DEVICE,
        )
        elapsed = time.time() - t0

        m = metrics(tl, pl, ts, ps)
        qp, ql, qh = bootstrap_qwk(tl, pl, BOOTSTRAP_N)

        subj_breakdown = per_subject(test_df, pl, ps, min_n=30)

        m.update({
            "qwk_ci_95": [ql, qh],
            "best_val_qwk": best_val,
            "elapsed_sec": round(elapsed, 2),
            "input_format": fmt,
            "per_subject": subj_breakdown,
            "deploy": m["qwk"] >= DEPLOY_THRESHOLD,
            "n_params": n_params,
        })

        all_results[key] = m
        all_preds[key]   = pl

        flag = "✓" if m["qwk"] >= DEPLOY_THRESHOLD else "✗"
        print(f"    QWK={m['qwk']:.4f} [{ql:.4f},{qh:.4f}]  "
              f"RMSE={m['rmse']:.4f}  Pearson={m['pearson']:.4f}  "
              f"F1={m['f1_w']:.4f}  [{flag}]  ({elapsed:.0f}s)")
        print(f"    Per-subject: { {k: round(v['qwk'],3) if v['qwk'] else f'n={v[chr(110)]}' for k,v in subj_breakdown.items()} }")

    # ── SIGNIFICANCE TESTS ───────────────────────────────────
    print("\n" + "="*60)
    print("SIGNIFICANCE TESTS (vs best baseline per format)")
    print("="*60)

    test_labels = test_df["score_label"].values.astype(np.int64)

    for fmt in input_formats:
        # Find best baseline for this format
        fmt_models = {k: v for k,v in all_results.items() if k.endswith(f"_{fmt}")}
        best_key   = max(fmt_models, key=lambda k: fmt_models[k]["qwk"])
        best_pl    = all_preds[best_key]
        best_qwk   = all_results[best_key]["qwk"]

        print(f"\n  Format [{fmt}]  —  Best: {best_key} (QWK={best_qwk:.4f})")
        print(f"  {'Model':<30} {'ΔQWK':>8} {'p-value':>10} {'Sig?':>6}")
        print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*6}")

        for k in sorted(fmt_models.keys()):
            if k == best_key: continue
            diff, p = paired_bootstrap(test_labels, best_pl, all_preds[k],
                                        n=BOOTSTRAP_N)
            sig = "yes" if p < 0.05 else "no"
            print(f"  {k:<30} {diff:>+8.4f} {p:>10.4f} {sig:>6}")
        all_results[best_key]["is_best_for_format"] = fmt

    # ── FINAL TABLE ──────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL RESULTS TABLE")
    print("="*60)
    print(f"\n  {'Model':<30} {'Fmt':>4} {'QWK':>7} {'95% CI':>14} {'RMSE':>6} "
          f"{'Pearson':>8} {'F1':>6} {'≥0.70':>6}")
    print(f"  {'-'*30} {'-'*4} {'-'*7} {'-'*14} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")

    rows = sorted(all_results.items(), key=lambda x: x[1]["qwk"], reverse=True)
    for k, m in rows:
        fmt  = m.get("input_format", "?")
        ci   = m.get("qwk_ci_95", [0,0])
        flag = "✓" if m["qwk"] >= DEPLOY_THRESHOLD else "✗"
        print(f"  {k:<30} {fmt:>4} {m['qwk']:>7.4f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] {m['rmse']:>6.4f} "
              f"{m.get('pearson',0):>8.4f} {m.get('f1_w',0):>6.4f} {flag:>6}")

    # ── SAVE ─────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "local_experiment_results.json")

    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=conv, ensure_ascii=False)

    print(f"\n  Results saved → {out_path}")
    print("\n✓ Local experiment complete.")


if __name__ == "__main__":
    main()
