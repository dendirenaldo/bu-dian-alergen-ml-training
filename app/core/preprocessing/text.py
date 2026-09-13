"""Text preprocessing utilities for the ML training pipeline."""

import re
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize whitespace and line breaks in text.

    Args:
        text: Raw input text.

    Returns:
        Normalized text with single spaces.
    """
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_phrase(text: str) -> str:
    """Clean a text phrase: lowercase, remove non-alphanumeric, normalize spaces.

    Args:
        text: Input text.

    Returns:
        Cleaned phrase.
    """
    text = normalize_text(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_composition_items(composition_text: str) -> list[str]:
    """Split composition text into individual items.

    Args:
        composition_text: Comma/semicolon/pipe separated composition string.

    Returns:
        List of normalized composition items (length >= 2).
    """
    raw_items = re.split(r"[,;•|]", composition_text)
    items = []
    for item in raw_items:
        s = normalize_text(item)
        if len(s) >= 2:
            items.append(s)
    return items


def token_overlap_score(a: str, b: str) -> float:
    """Compute token overlap score between two text strings.

    Args:
        a: First text string.
        b: Second text string.

    Returns:
        Overlap score between 0.0 and 1.0.
    """
    sa = set(clean_phrase(a).split())
    sb = set(clean_phrase(b).split())
    if not sa or not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    return inter / max(1, min(len(sa), len(sb)))


def cleanse_text(text: str) -> str:
    """Alias dari simple-tokenize-then-join (kompatibilitas baca OCR).

    Dipertahankan karena dipakai modul OCR; untuk model gunakan
    simple_tokenize langsung.
    """
    return " ".join(simple_tokenize(text))


def simple_tokenize(text: str) -> list[str]:
    """Tokenisasi model: lowercase, remove special chars, split.

    Tanpa stopword removal (konsisten dengan Keras Tokenizer).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if tok]


def fold_digits_v5(text: str) -> str:
    """Lipastoken angka menjadi 'num' (deviasi preprocessing Fase 2).

    Rasional (audit Fase 1): 132 token murni-digit di real-train
    memecah vocab ('8','70','150',...) padahal maknanya setara
    (persentase/takaran). Diterapkan konsisten ke train/val/holdout
    SEBELUM tokenisasi agar tidak bocor.
    """
    return re.sub(r"\d+", "num", str(text or ""))

