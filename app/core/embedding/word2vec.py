"""Word2Vec embedding training and embedding matrix construction."""

import logging
from typing import Any

import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 3,
    workers: int = 1,
    epochs: int = 30,
    seed: int = 42,
) -> Word2Vec:
    """Train a Word2Vec model on tokenized sentences.

    Args:
        sentences: List of tokenized sentences.
        vector_size: Dimensionality of word vectors.
        window: Context window size.
        min_count: Minimum word count threshold.
        workers: Number of worker threads.
        epochs: Number of training epochs.
        seed: Random seed.

    Returns:
        Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=epochs,
        seed=seed,
    )
    logger.info(
        "Word2Vec trained: vocab=%d, vector_size=%d",
        len(model.wv),
        vector_size,
    )
    return model


def build_embedding_matrix(
    w2v_model: Word2Vec,
    word_index: dict[str, int],
    vocab_size: int,
    embed_dim: int,
) -> tuple[np.ndarray, int, int]:
    """Build an embedding matrix from a trained Word2Vec model.

    Maps tokenizer word indices to Word2Vec vectors, initializing
    unseen words with random vectors.

    Args:
        w2v_model: Trained Word2Vec model.
        word_index: Keras tokenizer word_index mapping.
        vocab_size: Maximum vocabulary size.
        embed_dim: Embedding dimension.

    Returns:
        Tuple of (embedding_matrix, num_words, hit_count).
    """
    num_words = min(vocab_size, len(word_index) + 1)
    # FIX: init OOV kecil (scale 0.1, bukan 0.6) agar vektor acak tidak
    # mendominasi vektor Word2Vec terlatih (norm ~0.1-1.0). Seed agar stabil.
    rng = np.random.RandomState(42)
    embedding_matrix = rng.normal(scale=0.1, size=(num_words, embed_dim)).astype(
        np.float32
    )
    embedding_matrix[0] = np.zeros((embed_dim,), dtype=np.float32)

    hit_count = 0
    for word, idx in word_index.items():
        if idx >= num_words:
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            hit_count += 1

    logger.info(
        "Embedding matrix: shape=%s, hits=%d/%d (%.1f%%)",
        embedding_matrix.shape,
        hit_count,
        min(num_words, len(word_index)),
        hit_count * 100 / max(1, min(num_words, len(word_index))),
    )
    coverage = hit_count / max(1, min(num_words, len(word_index)))
    if coverage < 0.5:
        logger.warning(
            "Cakupan Word2Vec rendah (%.1f%%). Kemungkinan mismatch preprocessing "
            "W2V (stopword dibuang) vs Tokenizer (stopword ada). "
            "Pertimbangkan samakan pipeline tokenisasi.",
            coverage * 100,
        )
    return embedding_matrix, num_words, hit_count
