"""Protokol abstraksi dual-model (BiLSTM + BERT).

Trainer / evaluator / service coding ke protokol ini, bukan ke kelas konkret,
agar menambah varian BERT baru hanya ganti ``model_name`` tanpa ubah pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TokenizerBase(Protocol):
    """Kontrak tokenizer apa pun (Keras word-level maupun HF WordPiece)."""

    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str): ...


@runtime_checkable
class ClassifierBase(Protocol):
    """Kontrak classifier: teks mentah -> (label, skor)."""

    model_name: str

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...
    def predict(self, texts: list[str], threshold: float = 0.5) -> list[str]: ...
