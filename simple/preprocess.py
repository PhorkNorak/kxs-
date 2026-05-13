"""Three preprocessing modes built on the existing kxs preprocessing primitives.

raw      → strip whitespace only
clean    → KCC normalize + strip punctuation (- , + ( ) : ; " ' ! / ? ។ etc.)
segment  → clean + khmernltk.word_tokenize
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import kcc_normalize, strip_punctuation, segment_khmer


def preprocess(text: str, mode: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if mode == "raw":
        return text
    text = strip_punctuation(kcc_normalize(text))
    if mode == "clean":
        return text
    if mode == "segment":
        return segment_khmer(text)
    raise ValueError(f"Unknown preprocess mode: {mode}")
