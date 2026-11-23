"""Definisi arsitektur model LSTM/BiLSTM/GRU.

Fungsi ``build_model`` tetap backward-compatible dengan pemanggilan lama,
tetapi sekarang menerima opsi tambahan yang dibutuhkan HPO:

- ``rnn_type``: ``lstm`` atau ``gru``.
- ``num_rnn_layers``: 1 atau 2 lapisan rekuren.
- ``optimizer``: adam, adamw, rmsprop, atau sgd.
- ``weight_decay``, ``maxnorm_value``, dan ``label_smoothing``.

Semua opsi punya default yang sama dengan konfigurasi final lama agar
training lama tidak berubah perilakunya.
"""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
)
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)


class _BinaryConfusionMetric(tf.keras.metrics.Metric):
    """Base metric biner yang deterministik di GPU.

    Metric bawaan ``tf.keras.metrics.Precision/Recall`` memakai
    ``unsorted_segment_sum`` yang tidak punya implementasi deterministik di
    GPU. Akibatnya training gagal saat ``TF_DETERMINISTIC_OPS=1``. Metric ini
    hanya memakai ``reduce_sum`` sehingga aman untuk mode deterministik.
    """

    def __init__(self, threshold: float = 0.5, name: str = "binary_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(
            name="tp", initializer="zeros", aggregation=tf.VariableAggregation.SUM
        )
        self.fp = self.add_weight(
            name="fp", initializer="zeros", aggregation=tf.VariableAggregation.SUM
        )
        self.fn = self.add_weight(
            name="fn", initializer="zeros", aggregation=tf.VariableAggregation.SUM
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (-1,)) >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class BinaryPrecision(_BinaryConfusionMetric):
    """Precision biner tanpa operasi segment-sum."""

    def __init__(self, threshold: float = 0.5, name: str = "precision", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return self.tp / (self.tp + self.fp + 1e-8)


class BinaryRecall(_BinaryConfusionMetric):
    """Recall biner tanpa operasi segment-sum."""

    def __init__(self, threshold: float = 0.5, name: str = "recall", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return self.tp / (self.tp + self.fn + 1e-8)


class BinaryF1(_BinaryConfusionMetric):
    """F1 biner pada threshold 0,5 untuk EarlyStopping/HPO."""

    def __init__(self, threshold: float = 0.5, name: str = "f1", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + 1e-8)


def _build_optimizer(
    name: str,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float | None,
):
    """Buat optimizer sesuai nama, dengan clipping opsional."""
    name = name.lower()
    clipnorm = gradient_clip_norm if gradient_clip_norm is not None else None
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )
    if name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )
    raise ValueError(f"Optimizer tidak dikenal: {name}")


def _bce_with_label_smoothing(smoothing: float):
    """Binary cross-entropy dengan label smoothing (0 = BCE biasa)."""

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        smoothed = y_true * (1.0 - smoothing) + 0.5 * smoothing
        return tf.keras.backend.binary_crossentropy(smoothed, y_pred)

    return loss


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
    recurrent_dropout: float = 0.0,
    dense_units: int = 64,
    dropout_rate_dense: float = 0.2,
    gradient_clip_norm: float | None = 1.0,
    mask_zero: bool = True,
    use_maxnorm: bool = True,
    *,
    rnn_type: str = "lstm",
    num_rnn_layers: int = 2,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    maxnorm_value: float = 3.0,
    label_smoothing: float = 0.0,
) -> tf.keras.Model:
    """Bangun model sekuens (Bi)LSTM/GRU untuk klasifikasi biner.

    Args:
        use_bidirectional: bungkus lapisan rekuren dengan ``Bidirectional``.
        lstm_units_1: unit lapisan rekuren pertama.
        lstm_units_2: unit lapisan rekuren kedua (dipakai bila 2 lapisan).
        dropout_1: dropout setelah lapisan pertama.
        dropout_2: dropout setelah lapisan kedua.
        lr: learning rate.
        num_words: ukuran vocabulary untuk Embedding.
        embed_dim: dimensi embedding.
        embedding_matrix: bobot Word2Vec yang sudah disiapkan.
        embed_trainable: apakah embedding ikut dilatih.
        recurrent_dropout: dropout pada koneksi rekuren (0 untuk cuDNN).
        dense_units: unit Dense sebelum output.
        dropout_rate_dense: dropout setelah Dense.
        gradient_clip_norm: batas norma gradien; ``None`` = nonaktif.
        mask_zero: abaikan token padding (id 0).
        use_maxnorm: batasi bobot Dense dengan MaxNorm.
        rnn_type: ``lstm`` atau ``gru``.
        num_rnn_layers: 1 atau 2.
        optimizer: adam, adamw, rmsprop, atau sgd.
        weight_decay: bobot regularisasi L2 untuk optimizer yang mendukung.
        maxnorm_value: nilai maksimum MaxNorm.
        label_smoothing: smoothing label untuk BCE (0 = nonaktif).

    Returns:
        Model Keras yang sudah dikompilasi.
    """
    rnn_type = rnn_type.lower()
    if rnn_type not in ("lstm", "gru"):
        raise ValueError("rnn_type harus 'lstm' atau 'gru'.")
    if num_rnn_layers not in (1, 2):
        raise ValueError("num_rnn_layers harus 1 atau 2.")
    rnn_cls = LSTM if rnn_type == "lstm" else GRU

    model = Sequential()
    model.add(
        Embedding(
            input_dim=num_words,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=embed_trainable,
            mask_zero=mask_zero,
        )
    )

    first = rnn_cls(
        lstm_units_1,
        return_sequences=(num_rnn_layers == 2),
        recurrent_dropout=recurrent_dropout,
    )
    model.add(Bidirectional(first) if use_bidirectional else first)
    model.add(Dropout(dropout_1))

    if num_rnn_layers == 2:
        second = rnn_cls(
            lstm_units_2,
            return_sequences=False,
            recurrent_dropout=recurrent_dropout,
        )
        model.add(Bidirectional(second) if use_bidirectional else second)
        model.add(Dropout(dropout_2))

    if use_maxnorm:
        model.add(Dense(dense_units, activation="relu", kernel_constraint=MaxNorm(maxnorm_value)))
    else:
        model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate_dense))
    # Output tetap float32 agar loss/BCE stabil di bawah mixed precision.
    model.add(Dense(1, activation="sigmoid", dtype="float32"))

    loss = _bce_with_label_smoothing(label_smoothing) if label_smoothing > 0 else "binary_crossentropy"
    model.compile(
        optimizer=_build_optimizer(optimizer, lr, weight_decay, gradient_clip_norm),
        loss=loss,
        # CATATAN GPU: jangan memakai tf.keras.metrics.AUC/Precision/Recall
        # di sini. Metric bawaan memakai unsorted_segment_sum yang tidak
        # deterministik di GPU dan membuat training gagal saat
        # TF_DETERMINISTIC_OPS=1. ROC-AUC/AP tetap dihitung setelah training
        # dengan scikit-learn pada probabilitas validasi/holdout.
        metrics=[
            "accuracy",
            BinaryPrecision(name="precision"),
            BinaryRecall(name="recall"),
            BinaryF1(name="f1"),
        ],
    )

    logger.info(
        "Model built: rnn=%s bidirectional=%s layers=%d units=(%d,%d)",
        rnn_type, use_bidirectional, num_rnn_layers, lstm_units_1, lstm_units_2,
    )
    return model
