"""Keras tokenizer wrapper."""

import json
import logging
import os

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)

# ID 0 = PAD, ID 1 = OOV ("<OOV>"). Jangan dipakai untuk kata lain.
PAD_ID = 0
OOV_ID = 1


class Tokenizer:
    """Wrapper around Keras Tokenizer with padding support."""

    def __init__(self, vocab_size: int = 20000, max_len: int = 120):
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size.
            max_len: Maximum sequence length for padding/truncation.
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self._tokenizer = KerasTokenizer(num_words=vocab_size, oov_token="<OOV>")

    def fit(self, texts: list[str]) -> None:
        """Fit the tokenizer on texts.

        Args:
            texts: List of text strings.
        """
        self._tokenizer.fit_on_texts(texts)
        logger.info("Tokenizer fitted: vocab=%d", len(self._tokenizer.word_index))

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        """Convert texts to sequences of integers.

        FIX OOB (CRITICAL): Keras Tokenizer dengan ``num_words=N`` tetap
        mengembalikan indeks >= N dari ``texts_to_sequences``. Indeks itu
        akan IndexError di ``Embedding(input_dim=N)``. Maka di sini setiap
        indeks >= vocab_size dipetakan ke OOV_ID (1).

        Args:
            texts: List of text strings.

        Returns:
            List of integer sequences (semua idx < vocab_size).
        """
        seqs = self._tokenizer.texts_to_sequences(texts)
        capped: list[list[int]] = []
        for s in seqs:
            capped.append([idx if idx < self.vocab_size else OOV_ID for idx in s])
        return capped

    def pad_sequences(self, sequences: list[list[int]]) -> np.ndarray:
        """Pad sequences to uniform length.

        Args:
            sequences: List of integer sequences.

        Returns:
            Padded numpy array of shape (n, max_len).
        """
        return pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Fit (if not already) and encode texts to padded sequences.

        Args:
            texts: List of text strings.

        Returns:
            Padded numpy array.
        """
        sequences = self.texts_to_sequences(texts)
        return self.pad_sequences(sequences)

    @property
    def word_index(self) -> dict[str, int]:
        """Return the word index mapping."""
        return self._tokenizer.word_index

    @property
    def num_words(self) -> int:
        """Return the effective vocabulary size (capped by vocab_size)."""
        return min(self.vocab_size, len(self._tokenizer.word_index) + 1)

    # -- Persistensi dual-model: JSON vocab (portabel) + pickle legacy --
    def get_config(self) -> dict:
        """Return serializable config (vocab + params) untuk save JSON."""
        return {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "word_index": self._tokenizer.word_index,
            "oov_token": "<OOV>",
        }

    def save_json(self, path: str) -> None:
        """Simpan vocab ke JSON (format dual-model: tokenizer_<model>.json)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, ensure_ascii=False)
        logger.info("Tokenizer JSON saved to: %s", path)

    @classmethod
    def load_json(cls, path: str) -> "Tokenizer":
        """Load tokenizer dari JSON (tanpa perlu fit ulang)."""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        tok = cls(vocab_size=cfg["vocab_size"], max_len=cfg["max_len"])
        # Rekonstruksi Keras Tokenizer dari word_index tersimpan.
        tok._tokenizer.fit_on_texts([])  # inisialisasi struktur internal
        tok._tokenizer.word_index = dict(cfg["word_index"])
        # Rebuild index_word yang dipakai texts_to_sequences.
        tok._tokenizer.index_word = {
            idx: w for w, idx in tok._tokenizer.word_index.items()
        }
        return tok
