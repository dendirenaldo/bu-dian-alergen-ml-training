"""Text preprocessing utilities for the ML training pipeline."""

import re
import logging

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = logging.getLogger(__name__)

_stopword_factory = StopWordRemoverFactory()
STOPWORDS: set[str] = set(_stopword_factory.get_stop_words())


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
    """Cleanse text for model input: lowercase, remove special chars, normalize.

    Args:
        text: Raw text.

    Returns:
        Cleansed text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> list[str]:
    """Simple tokenize: lowercase, remove special chars, split on whitespace.

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if tok]


def simple_tokenize_v5(text: str) -> list[str]:
    """Tokenisasi parity Notebook V5 Cell 43 (tanpa stopword removal).

    Dipakai untuk Word2Vec V5 agar konsisten dengan Keras Tokenizer
    (yang juga tidak membuang stopwords). Pipeline legacy
    (simple_tokenize + filter_tokens) tetap dipertahankan untuk
    kompatibilitas.
    """
    return simple_tokenize(text)


def filter_tokens(tokens: list[str]) -> list[str]:
    """Filter tokens: remove stopwords, short tokens, and pure digits.

    Args:
        tokens: List of tokens.

    Returns:
        Filtered list of tokens.
    """
    return [
        t
        for t in tokens
        if t not in STOPWORDS
        and len(t) >= 2
        and not t.isdigit()
    ]
