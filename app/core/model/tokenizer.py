"""Keras tokenizer wrapper."""

import logging

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)


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

        Args:
            texts: List of text strings.

        Returns:
            List of integer sequences.
        """
        return self._tokenizer.texts_to_sequences(texts)

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
