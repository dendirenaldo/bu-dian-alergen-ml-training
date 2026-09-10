"""Hyperparameter tuning and model training."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from app.config import Config
from app.core.model.architecture import build_model

logger = logging.getLogger(__name__)


def tune_model(
    config: Config,
    use_bidirectional: bool,
    model_name: str,
    num_words: int,
    embed_dim: int,
    embedding_matrix: np.ndarray,
    X_train_pad: np.ndarray,
    y_train: np.ndarray,
    strategy: tf.distribute.Strategy,
    class_weight_dict: dict | None = None,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, dict]:
    """Run random search hyperparameter tuning and train the final model.

    Args:
        config: Training configuration.
        use_bidirectional: Whether to use Bidirectional LSTM.
        model_name: Name for logging/checkpointing.
        num_words: Vocabulary size.
        embed_dim: Embedding dimension.
        embedding_matrix: Pre-trained embedding weights.
        X_train_pad: Padded training sequences.
        y_train: Training labels.
        strategy: TensorFlow distribution strategy.
        class_weight_dict: Optional class weight dictionary.

    Returns:
        Tuple of (trained_model, training_history, best_params).
    """
    best_val_loss = float("inf")
    best_params = None

    for trial in range(config.num_trials):
        batch_size = int(np.random.choice(config.batch_size_options))
        lstm_pair = config.lstm_units_options[
            np.random.randint(len(config.lstm_units_options))
        ]
        dropout_pair = config.dropout_options[
            np.random.randint(len(config.dropout_options))
        ]
        tuning_epochs = int(np.random.choice(config.tuning_epochs_options))
        lr = float(np.random.choice(config.lr_options))

        params = {
            "lstm_units_1": lstm_pair[0],
            "lstm_units_2": lstm_pair[1],
            "dropout_1": dropout_pair[0],
            "dropout_2": dropout_pair[1],
            "lr": lr,
            "batch_size": batch_size,
            "tuning_epochs": tuning_epochs,
        }

        bldr_kw = {
            k: v
            for k, v in params.items()
            if k in ["lstm_units_1", "lstm_units_2", "dropout_1", "dropout_2", "lr"]
        }
        logger.info(
            "Trial %d/%d %s: bs=%d, units=(%d,%d), drop=(%.1f,%.1f), lr=%.0e, ep=%d",
            trial + 1,
            config.num_trials,
            model_name,
            batch_size,
            lstm_pair[0],
            lstm_pair[1],
            dropout_pair[0],
            dropout_pair[1],
            lr,
            tuning_epochs,
        )
        tf.random.set_seed(config.seed + trial)

        with strategy.scope():
            m = build_model(
                use_bidirectional,
                **bldr_kw,
                num_words=num_words,
                embed_dim=embed_dim,
                embedding_matrix=embedding_matrix,
                embed_trainable=config.embed_trainable,
                recurrent_dropout=config.recurrent_dropout,
                dense_units=config.dense_units,
                dropout_rate_dense=config.dropout_rate_dense,
                gradient_clip_norm=config.gradient_clip_norm,
            )

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            ckpt_path = f.name

        try:
            cb = [
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
                ModelCheckpoint(
                    ckpt_path, monitor="val_loss", save_best_only=True, verbose=0
                ),
            ]
            h = m.fit(
                X_train_pad,
                y_train,
                validation_split=config.tuning_val_split,
                epochs=tuning_epochs,
                batch_size=batch_size,
                callbacks=cb,
                class_weight=class_weight_dict,
                verbose=0,
            )
            val_loss = min(h.history["val_loss"])
            val_acc = max(h.history["val_accuracy"])
            logger.info("  val_loss=%.4f  val_acc=%.4f", val_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)

    logger.info(
        "Best params for %s: bs=%d, units=(%d,%d), drop=(%.1f,%.1f), lr=%.0e, ep=%d (val_loss=%.4f)",
        model_name,
        best_params["batch_size"],
        best_params["lstm_units_1"],
        best_params["lstm_units_2"],
        best_params["dropout_1"],
        best_params["dropout_2"],
        best_params["lr"],
        best_params["tuning_epochs"],
        best_val_loss,
    )

    # Train final model with best params
    logger.info("Training final %s", model_name)
    tf.random.set_seed(config.seed)
    bldr_kw_best = {
        k: v
        for k, v in best_params.items()
        if k in ["lstm_units_1", "lstm_units_2", "dropout_1", "dropout_2", "lr"]
    }

    with strategy.scope():
        final_model = build_model(
            use_bidirectional,
            **bldr_kw_best,
            num_words=num_words,
            embed_dim=embed_dim,
            embedding_matrix=embedding_matrix,
            embed_trainable=config.embed_trainable,
            recurrent_dropout=config.recurrent_dropout,
            dense_units=config.dense_units,
            dropout_rate_dense=config.dropout_rate_dense,
            gradient_clip_norm=config.gradient_clip_norm,
        )

    callbacks = []
    if config.early_stopping_on:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            )
        )
    if config.lr_reduce_on:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=config.lr_reduce_patience,
                min_lr=config.min_lr,
            )
        )

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        final_ckpt = f.name
    callbacks.append(
        ModelCheckpoint(final_ckpt, monitor="val_loss", save_best_only=True, verbose=0)
    )

    try:
        final_history = final_model.fit(
            X_train_pad,
            y_train,
            validation_split=config.tuning_val_split,
            epochs=config.epochs,
            batch_size=best_params["batch_size"],
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )
    finally:
        if os.path.exists(final_ckpt):
            os.unlink(final_ckpt)

    return final_model, final_history, best_params
