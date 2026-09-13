"""Dataset BERT: teks MENTAH + normalisasi ringan (tanpa stopword removal).

PENTING: Jangan pakai cleanse_text/filter_tokens BiLSTM untuk BERT — WordPiece
membutuhkan konteks asli (stopword, casing secukupnya, tanda baca).
"""

from __future__ import annotations

import re


def normalize_for_bert(text: str) -> str:
    """Normalisasi ringan: rapikan whitespace, jaga isi kalimat."""
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_bert_texts(texts: list[str]) -> list[str]:
    """Terapkan normalisasi ringan ke batch teks."""
    return [normalize_for_bert(t) for t in texts]
