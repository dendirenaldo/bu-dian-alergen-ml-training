"""Model architecture definition."""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)


def build_model(
    use_bidirectional: bool,
    lstm_units_1: int,
    lstm_units_2: int,
    dropout_1: float,
    dropout_2: float,
    lr: float,
    num_words: int,
    embed_dim: int,
    embedding_matrix: np.ndarray,
    embed_trainable: bool = False,
    recurrent_dropout: float = 0.2,
    dense_units: int = 64,
    dropout_rate_dense: float = 0.2,
    gradient_clip_norm: float = 1.0,
) -> tf.keras.Model:
    """Build a Sequential LSTM or BiLSTM model.

    Args:
        use_bidirectional: If True, use Bidirectional LSTM layers.
        lstm_units_1: Units for the first LSTM layer.
        lstm_units_2: Units for the second LSTM layer.
        dropout_1: Dropout rate after the first LSTM layer.
        dropout_2: Dropout rate after the second LSTM layer.
        lr: Learning rate.
        num_words: Vocabulary size for the embedding layer.
        embed_dim: Embedding dimension.
        embedding_matrix: Pre-trained embedding weights.
        embed_trainable: Whether the embedding layer is trainable.
        recurrent_dropout: Recurrent dropout (set to 0 for cuDNN compatibility).
        dense_units: Units for the dense layer.
        dropout_rate_dense: Dropout rate after the dense layer.
        gradient_clip_norm: Gradient clipping norm value.

    Returns:
        Compiled Keras model.
    """
    model = Sequential()
    model.add(
        Embedding(
            input_dim=num_words,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=embed_trainable,
        )
    )

    lstm_kwargs = dict(return_sequences=True, recurrent_dropout=recurrent_dropout)
    lstm_kwargs2 = dict(recurrent_dropout=recurrent_dropout)

    if use_bidirectional:
        model.add(Bidirectional(LSTM(lstm_units_1, **lstm_kwargs)))
    else:
        model.add(LSTM(lstm_units_1, **lstm_kwargs))
    model.add(Dropout(dropout_1))

    if use_bidirectional:
        model.add(Bidirectional(LSTM(lstm_units_2, **lstm_kwargs2)))
    else:
        model.add(LSTM(lstm_units_2, **lstm_kwargs2))
    model.add(Dropout(dropout_2))

    model.add(Dense(dense_units, activation="relu", kernel_constraint=MaxNorm(3)))
    model.add(Dropout(dropout_rate_dense))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr, clipnorm=gradient_clip_norm
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info("Model built: bidirectional=%s, units=(%d,%d)", use_bidirectional, lstm_units_1, lstm_units_2)
    return model
