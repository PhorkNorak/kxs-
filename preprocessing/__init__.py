"""
KhmerXScore Preprocessing Pipeline
====================================
- KCC Character Cluster normalization (syllable-boundary aware)
- Optional khmernltk word segmentation
- Optional punctuation stripping
- No spellcheck (pedagogically motivated: teachers deduct for spelling)
"""

import re
import unicodedata
from typing import Optional


# ============================================================
# Khmer Unicode Ranges
# ============================================================
KHMER_CONSONANTS = set(range(0x1780, 0x17A3))
KHMER_INDEP_VOWELS = set(range(0x17A5, 0x17B4))
KHMER_DEPENDENT_VOWELS = set(range(0x17B6, 0x17C6))
KHMER_SIGNS = set(range(0x17C6, 0x17D4))
KHMER_COENG = 0x17D2  # Coeng (subscript consonant marker)

# Punctuation: ASCII + Khmer punctuation (។ through ៚)
KHMER_PUNCT = set(range(0x17D4, 0x17DB))
ASCII_PUNCT = set(
    ord(c) for c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
)


def is_khmer_base_consonant(c: str) -> bool:
    """Check if character is a Khmer base consonant."""
    return ord(c) in KHMER_CONSONANTS


def is_coeng(c: str) -> bool:
    """Check if character is the Coeng (subscript marker)."""
    return ord(c) == KHMER_COENG


# ============================================================
# KCC Normalization
# ============================================================
def kcc_normalize(text: str) -> str:
    """
    Normalize Khmer text using Khmer Character Cluster (KCC) decomposition.
    
    Splits at syllable boundaries (new base consonant not preceded by coeng).
    Reorders combining marks within each cluster to canonical form:
      Base consonant → Coeng sequences → Dependent vowels → Signs
    
    This fixes garbled text from inconsistent keyboard input ordering.
    """
    if not text:
        return text
    
    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Step 2: Split into KCC clusters at syllable boundaries
    clusters = []
    current_cluster = []
    prev_was_coeng = False
    
    for i, ch in enumerate(text):
        cp = ord(ch)
        
        # If this is a base consonant and NOT preceded by coeng → new cluster
        if cp in KHMER_CONSONANTS and not prev_was_coeng and current_cluster:
            # Check if current cluster has any Khmer content
            has_khmer = any(0x1780 <= ord(c) <= 0x17FF for c in current_cluster)
            if has_khmer:
                clusters.append(current_cluster)
                current_cluster = []
        
        current_cluster.append(ch)
        prev_was_coeng = (cp == KHMER_COENG)
    
    if current_cluster:
        clusters.append(current_cluster)
    
    # Step 3: Reorder each cluster to canonical form
    normalized = []
    for cluster in clusters:
        # Separate into categories
        bases = []
        coengs = []       # Coeng + following consonant pairs
        vowels = []
        signs = []
        others = []
        
        i = 0
        chars = cluster
        while i < len(chars):
            cp = ord(chars[i])
            if cp in KHMER_CONSONANTS and not coengs and not vowels and not signs:
                bases.append(chars[i])
            elif cp == KHMER_COENG and i + 1 < len(chars):
                coengs.append(chars[i])
                coengs.append(chars[i + 1])
                i += 1
            elif cp in KHMER_DEPENDENT_VOWELS:
                vowels.append(chars[i])
            elif cp in KHMER_SIGNS:
                signs.append(chars[i])
            else:
                others.append(chars[i])
            i += 1
        
        # Canonical order: bases → coengs → vowels → signs → others
        reordered = bases + coengs + vowels + signs + others
        normalized.extend(reordered)
    
    return "".join(normalized)


# ============================================================
# Punctuation Stripping
# ============================================================
def strip_punctuation(text: str) -> str:
    """Remove ASCII and Khmer punctuation."""
    result = []
    for ch in text:
        cp = ord(ch)
        if cp in ASCII_PUNCT or cp in KHMER_PUNCT:
            result.append(" ")  # Replace with space to preserve word boundaries
        else:
            result.append(ch)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", "".join(result)).strip()


# ============================================================
# Word Segmentation
# ============================================================
def segment_khmer(text: str) -> str:
    """
    Segment Khmer text using khmernltk.
    Falls back to original text if khmernltk is unavailable.
    """
    try:
        import khmernltk
        # khmernltk.word_tokenize returns a list of tokens
        tokens = khmernltk.word_tokenize(text)
        return " ".join(tokens)
    except ImportError:
        print("WARNING: khmernltk not installed. Returning unsegmented text.")
        return text
    except Exception as e:
        print(f"WARNING: Segmentation failed: {e}. Returning original text.")
        return text


# ============================================================
# Full Pipeline
# ============================================================
def preprocess(text: str, mode: str = "kcc_seg_punct") -> str:
    """
    Apply preprocessing pipeline based on mode.
    
    Modes:
        raw           - No preprocessing (return as-is)
        kcc           - KCC normalization only
        kcc_seg       - KCC + word segmentation
        kcc_seg_punct - KCC + segmentation + punctuation stripping (full)
    
    No spellcheck in any mode (pedagogically motivated: teachers deduct
    marks for spelling errors, so the model must see them).
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    if mode == "raw":
        return text
    
    # KCC normalization (all non-raw modes)
    text = kcc_normalize(text)
    
    if mode == "kcc":
        return text
    
    # Punctuation stripping (before segmentation for kcc_seg_punct,
    # after for kcc_seg — but we apply segmentation last for consistency)
    if mode == "kcc_seg_punct":
        text = strip_punctuation(text)
    
    # Word segmentation
    if mode in ("kcc_seg", "kcc_seg_punct"):
        text = segment_khmer(text)
    
    return text


# ============================================================
# Batch preprocessing
# ============================================================
def preprocess_batch(texts: list, mode: str = "kcc_seg_punct") -> list:
    """Preprocess a list of texts."""
    return [preprocess(t, mode) for t in texts]
