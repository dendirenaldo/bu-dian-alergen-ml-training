"""Main training orchestrator."""

from __future__ import annotations

import logging
import os
import random
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from app.config import Config, get_config
from app.core.embedding.word2vec import build_embedding_matrix, train_word2vec
from app.core.model.evaluator import (
    evaluate_model,
    plot_roc_comparison,
    plot_training_history,
)
from app.core.model.tokenizer import Tokenizer
from app.core.model.tuner import tune_model
from app.core.preprocessing.text import (
    cleanse_text,
    filter_tokens,
    simple_tokenize,
)

logger = logging.getLogger(__name__)


def setup_gpu(config: Config) -> tf.distribute.Strategy:
    """Configure GPU settings and return a distribution strategy.

    Args:
        config: Training configuration.

    Returns:
        TensorFlow distribution strategy.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    gpus = tf.config.list_physical_devices("GPU")
    logger.info("Available GPUs: %d device(s)", len(gpus))

    gpu_available = len(gpus) > 0

    if gpu_available:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("  %s - memory growth enabled", gpu.name)

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("  Mixed precision: mixed_float16")

        from tensorflow.python.distribute import cross_device_ops as cdo

        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=cdo.HierarchicalCopyAllReduce()
        )
        logger.info(
            "  MirroredStrategy: %d replica(s)", strategy.num_replicas_in_sync
        )

        config.recurrent_dropout = 0.0
        logger.info("  Recurrent dropout: 0.0 (disabled for cuDNN)")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        config.recurrent_dropout = 0.2
        logger.info("  Fallback: CPU only")
        logger.info("  Recurrent dropout: 0.2 (enabled)")

    return strategy


def load_data(config: Config, df_ocr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load data from CSV or OCR dataframe.

    Args:
        config: Training configuration.
        df_ocr: Optional OCR dataframe (used when data_source_mode='OCR').

    Returns:
        Cleaned DataFrame with text and label columns.

    Raises:
        ValueError: If data source mode is invalid.
        FileNotFoundError: If CSV file not found.
    """
    if config.data_source_mode.upper() == "OCR":
        if df_ocr is None:
            raise ValueError(
                "df_ocr not found. Run OCR pipeline first or switch to CSV mode."
            )
        df_model_source = df_ocr.copy()
    elif config.data_source_mode.upper() == "CSV":
        logger.info("Loading data from CSV: %s", config.csv_input)
        if not os.path.exists(config.csv_input):
            raise FileNotFoundError(f"CSV not found: {config.csv_input}")
        df_model_source = pd.read_csv(config.csv_input, delimiter=";")
    else:
        raise ValueError("data_source_mode must be 'OCR' or 'CSV'")

    for col in [config.text_col, config.label_col]:
        if col not in df_model_source.columns:
            raise ValueError(f"Column '{col}' not found in data source.")

    df_model = df_model_source[[config.text_col, config.label_col]].copy()
    df_model[config.text_col] = df_model[config.text_col].fillna("").astype(str)
    df_model[config.label_col] = (
        df_model[config.label_col].fillna("").astype(str).str.strip().str.lower()
    )
    df_model = df_model[df_model[config.text_col].str.strip() != ""].copy()
    df_model = df_model[df_model[config.label_col].isin(["safe", "unsafe"])].copy()

    before_dedup = len(df_model)
    df_model = df_model.drop_duplicates(subset=[config.text_col]).reset_index(drop=True)
    logger.info("Duplicates removed: %d", before_dedup - len(df_model))
    logger.info("Data ready: %d rows", len(df_model))
    logger.info("Label distribution:\n%s", df_model[config.label_col].value_counts())

    return df_model


def run_training(config: Config | None = None) -> dict:
    """Run the full training pipeline.

    Args:
        config: Optional configuration override. Uses default if None.

    Returns:
        Dictionary with trained models, histories, and best params.
    """
    if config is None:
        config = get_config()

    config.ensure_dirs()

    # Set seeds
    np.random.seed(config.seed)
    random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # Setup GPU
    strategy = setup_gpu(config)

    # Load data
    df_model = load_data(config)

    # Text cleansing
    df_model[config.text_col] = df_model[config.text_col].apply(cleanse_text)

    # Label encoding
    label_encoder = LabelEncoder()
    df_model["label_id"] = label_encoder.fit_transform(df_model[config.label_col])

    # Train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_model[config.text_col].tolist(),
        df_model["label_id"].values,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=df_model["label_id"].values,
    )

    # Class weights
    if config.use_class_weight:
        cw = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight_dict = dict(enumerate(cw))
    else:
        class_weight_dict = None

    logger.info("Train size: %d", len(X_train_text))
    logger.info("Test size: %d", len(X_test_text))
    logger.info(
        "Label map: %s",
        dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
    )

    # Tokenization
    train_tokens = [simple_tokenize(t) for t in X_train_text]
    train_tokens = [filter_tokens(t) for t in train_tokens]

    # Keras tokenizer
    tokenizer = Tokenizer(vocab_size=config.vocab_size, max_len=config.max_len)
    all_texts = X_train_text + X_test_text
    tokenizer.fit(all_texts)

    X_train_pad = tokenizer.encode(X_train_text)
    X_test_pad = tokenizer.encode(X_test_text)

    # Word2Vec
    w2v_model = train_word2vec(
        sentences=train_tokens,
        vector_size=config.embed_dim,
        window=config.w2v_window,
        min_count=config.min_word_count,
        epochs=config.w2v_epochs,
        seed=config.seed,
    )

    embedding_matrix, num_words, hit_count = build_embedding_matrix(
        w2v_model=w2v_model,
        word_index=tokenizer.word_index,
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
    )

    # BiLSTM training
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING & TRAINING: BiLSTM")
    logger.info("=" * 60)
    model_bilstm, history_bilstm, best_params_bilstm = tune_model(
        config=config,
        use_bidirectional=True,
        model_name="BiLSTM",
        num_words=num_words,
        embed_dim=config.embed_dim,
        embedding_matrix=embedding_matrix,
        X_train_pad=X_train_pad,
        y_train=y_train,
        strategy=strategy,
        class_weight_dict=class_weight_dict,
    )

    # LSTM training
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING & TRAINING: LSTM")
    logger.info("=" * 60)
    model_lstm, history_lstm, best_params_lstm = tune_model(
        config=config,
        use_bidirectional=False,
        model_name="LSTM",
        num_words=num_words,
        embed_dim=config.embed_dim,
        embedding_matrix=embedding_matrix,
        X_train_pad=X_train_pad,
        y_train=y_train,
        strategy=strategy,
        class_weight_dict=class_weight_dict,
    )

    # Plot training histories
    plot_training_history(history_bilstm, "BiLSTM", config.output_dir)
    plot_training_history(history_lstm, "LSTM", config.output_dir)

    # Evaluation
    eval_bilstm, report_bilstm, pred_bilstm, fpr_bilstm, tpr_bilstm, auc_bilstm = (
        evaluate_model(
            model_bilstm,
            "BiLSTM",
            X_test_pad,
            y_test,
            X_test_text,
            label_encoder,
            config.output_dir,
        )
    )

    eval_lstm, report_lstm, pred_lstm, fpr_lstm, tpr_lstm, auc_lstm = evaluate_model(
        model_lstm,
        "LSTM",
        X_test_pad,
        y_test,
        X_test_text,
        label_encoder,
        config.output_dir,
    )

    # ROC comparison
    plot_roc_comparison(
        fpr_bilstm, tpr_bilstm, auc_bilstm,
        fpr_lstm, tpr_lstm, auc_lstm,
        config.output_dir,
    )

    # Comparison table
    comparison = pd.DataFrame(
        [
            {
                "model": "BiLSTM",
                "best_params": str(best_params_bilstm),
                **{row["metric"]: row["value"] for _, row in eval_bilstm.iterrows()},
            },
            {
                "model": "LSTM",
                "best_params": str(best_params_lstm),
                **{row["metric"]: row["value"] for _, row in eval_lstm.iterrows()},
            },
        ]
    )
    comparison.to_csv(os.path.join(config.output_dir, "comparison_bilstm_vs_lstm.csv"), index=False)

    # Save models
    model_bilstm.save(os.path.join(config.model_dir, "bilstm_word2vec.keras"))
    model_lstm.save(os.path.join(config.model_dir, "lstm_word2vec.keras"))

    logger.info("All training and evaluation files saved.")
    logger.info("Model comparison:\n%s", comparison)

    return {
        "model_bilstm": model_bilstm,
        "model_lstm": model_lstm,
        "history_bilstm": history_bilstm,
        "history_lstm": history_lstm,
        "best_params_bilstm": best_params_bilstm,
        "best_params_lstm": best_params_lstm,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "comparison": comparison,
    }
