"""
Khmer Text Preprocessing Pipeline
- KCC Character Cluster normalization (syllable-boundary aware)
- Optional khmernltk word segmentation
- Optional punctuation stripping
- No spellcheck (pedagogically motivated: teachers deduct for spelling)
"""

import re
import unicodedata

KHMER_CONSONANTS = set(range(0x1780, 0x17A3))
KHMER_DEPENDENT_VOWELS = set(range(0x17B6, 0x17C6))
KHMER_SIGNS = set(range(0x17C6, 0x17D4))
KHMER_COENG = 0x17D2
KHMER_PUNCT = set(range(0x17D4, 0x17DB))
ASCII_PUNCT = set(ord(c) for c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

_khmernltk_warned = False


def kcc_normalize(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    clusters, current, prev_coeng = [], [], False
    for ch in text:
        cp = ord(ch)
        if cp in KHMER_CONSONANTS and not prev_coeng and current:
            if any(0x1780 <= ord(c) <= 0x17FF for c in current):
                clusters.append(current)
                current = []
        current.append(ch)
        prev_coeng = (cp == KHMER_COENG)
    if current:
        clusters.append(current)
    normalized = []
    for cluster in clusters:
        bases, coengs, vowels, signs, others = [], [], [], [], []
        i = 0
        while i < len(cluster):
            cp = ord(cluster[i])
            if cp in KHMER_CONSONANTS and not coengs and not vowels and not signs:
                bases.append(cluster[i])
            elif cp == KHMER_COENG and i + 1 < len(cluster):
                coengs += [cluster[i], cluster[i + 1]]; i += 1
            elif cp in KHMER_DEPENDENT_VOWELS:
                vowels.append(cluster[i])
            elif cp in KHMER_SIGNS:
                signs.append(cluster[i])
            else:
                others.append(cluster[i])
            i += 1
        normalized.extend(bases + coengs + vowels + signs + others)
    return "".join(normalized)


def strip_punctuation(text: str) -> str:
    result = []
    for ch in text:
        cp = ord(ch)
        result.append(" " if cp in ASCII_PUNCT or cp in KHMER_PUNCT else ch)
    return re.sub(r"\s+", " ", "".join(result)).strip()


def segment_khmer(text: str) -> str:
    global _khmernltk_warned
    try:
        import khmernltk
        return " ".join(khmernltk.word_tokenize(text))
    except ImportError:
        if not _khmernltk_warned:
            print("NOTE: khmernltk not installed — segmentation skipped.")
            _khmernltk_warned = True
        return text
    except Exception:
        return text


def preprocess(text: str, mode: str = "kcc_seg_punct") -> str:
    """
    Modes: raw | kcc | kcc_seg | kcc_seg_punct
    No spellcheck (teachers deduct marks for spelling errors).
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if mode == "raw":
        return text
    text = kcc_normalize(text)
    if mode == "kcc":
        return text
    if mode == "kcc_seg_punct":
        text = strip_punctuation(text)
    if mode in ("kcc_seg", "kcc_seg_punct"):
        text = segment_khmer(text)
    return text
