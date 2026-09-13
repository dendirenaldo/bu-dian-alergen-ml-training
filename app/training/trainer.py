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
    compute_binary_metrics,
    evaluate_model,
    plot_roc_comparison,
    plot_training_history,
    tune_threshold,
)
from app.core.model.tokenizer import Tokenizer
from app.core.model.tuner import _stratified_split, tune_model
from app.core.preprocessing.text import (
    cleanse_text,
    filter_tokens,
    simple_tokenize,
)

logger = logging.getLogger(__name__)


def setup_gpu(config: Config) -> tuple[tf.distribute.Strategy, float]:
    """Configure GPU settings and return (strategy, recurrent_dropout).

    FIX: tidak lagi memutasi ``config`` (side-effect lama). Nilai
    recurrent_dropout dikembalikan dan diteruskan eksplisit ke builder.

    Args:
        config: Training configuration (read-only).

    Returns:
        (strategy, recurrent_dropout).
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    gpus = tf.config.list_physical_devices("GPU")
    logger.info("Available GPUs: %d device(s)", len(gpus))

    gpu_available = len(gpus) > 0

    if gpu_available:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
            logger.info("  %s - memory growth enabled", gpu.name)

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("  Mixed precision: mixed_float16")

        strategy = tf.distribute.MirroredStrategy()
        logger.info(
            "  MirroredStrategy: %d replica(s)", strategy.num_replicas_in_sync
        )
        recurrent_dropout = 0.0
        logger.info("  Recurrent dropout: 0.0 (disabled for cuDNN)")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        recurrent_dropout = 0.2
        logger.info("  Fallback: CPU only")
        logger.info("  Recurrent dropout: 0.2 (enabled)")

    return strategy, recurrent_dropout


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
        df_model_source = pd.read_csv(
            config.csv_input,
            delimiter=config.csv_delimiter,
            encoding=config.csv_encoding,
        )
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
    n_empty = int((df_model[config.text_col].str.strip() == "").sum())
    df_model = df_model[df_model[config.text_col].str.strip() != ""].copy()
    n_bad_label = int((~df_model[config.label_col].isin(["safe", "unsafe"])).sum())
    df_model = df_model[df_model[config.label_col].isin(["safe", "unsafe"])].copy()
    if n_empty:
        logger.warning("Baris teks kosong dibuang: %d", n_empty)
    if n_bad_label:
        logger.warning("Baris label selain safe/unsafe dibuang: %d", n_bad_label)
    if df_model.empty:
        raise ValueError("Dataset kosong setelah filtering (cek CSV/delimiter).")
    if df_model[config.label_col].nunique() < 2:
        raise ValueError(
            "Dataset hanya 1 kelas setelah filtering — "
            "stratified split & ROC-AUC tidak valid. Tambah data kelas lain."
        )

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

    # FIX seed reproduksibel: PYTHONHASHSEED + deterministic ops + semua RNG.
    os.environ.setdefault("PYTHONHASHSEED", str(config.seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    np.random.seed(config.seed)
    random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # Setup GPU (tidak mutasi config lagi)
    strategy, recurrent_dropout = setup_gpu(config)

    # Load data
    df_model = load_data(config)

    # Text cleansing
    df_model[config.text_col] = df_model[config.text_col].apply(cleanse_text)

    # Label encoding
    label_encoder = LabelEncoder()
    df_model["label_id"] = label_encoder.fit_transform(df_model[config.label_col])

    # Train/test split — indeks disimpan ke splits.json agar BERT memakai
    # test set IDENTIK (perbandingan apel-vs-apel) dan evaluate.py me-load ulang.
    import json

    indices = np.arange(len(df_model))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=df_model["label_id"].values,
    )
    X_train_text = df_model[config.text_col].iloc[train_idx].tolist()
    X_test_text = df_model[config.text_col].iloc[test_idx].tolist()
    y_train = df_model["label_id"].values[train_idx]
    y_test = df_model["label_id"].values[test_idx]
    with open(os.path.join(config.output_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": config.seed,
                "test_size": config.test_size,
                "train_idx": [int(i) for i in train_idx],
                "test_idx": [int(i) for i in test_idx],
                "label_classes": list(label_encoder.classes_),
            },
            f,
        )
    logger.info("splits.json tersimpan (train=%d, test=%d)", len(train_idx), len(test_idx))

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

    # Keras tokenizer — fit HANYA di train (hindari leakage ke test).
    tokenizer = Tokenizer(vocab_size=config.vocab_size, max_len=config.max_len)
    tokenizer.fit(X_train_text)

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

    # recurrent_dropout eksplisit per device (dulu side-effect di setup_gpu).
    config.recurrent_dropout = recurrent_dropout

    model_type = (config.model_type or "all").lower()
    want_bilstm = model_type in ("bilstm", "all")
    want_lstm = model_type in ("lstm", "all")
    want_bert = model_type in ("bert", "all")

    model_bilstm = history_bilstm = best_params_bilstm = None
    model_lstm = history_lstm = best_params_lstm = None
    eval_bilstm = fpr_bilstm = tpr_bilstm = auc_bilstm = None
    eval_lstm = fpr_lstm = tpr_lstm = auc_lstm = None
    thr_bilstm = thr_lstm = config.default_threshold

    def _threshold_from_val(model, X_tr_pad, y_tr) -> float:
        """Tune threshold di validation stratified (bukan test)."""
        _, X_va, _, y_va = _stratified_split(
            np.asarray(X_tr_pad), np.asarray(y_tr),
            config.tuning_val_split, config.seed,
        )
        try:
            va_prob = model.predict(X_va, verbose=0).ravel()
            thr, _ = tune_threshold(np.asarray(y_va), va_prob, metric="f1")
            return float(np.clip(thr, 0.05, 0.95))
        except Exception as e:
            logger.warning("Threshold-tuning gagal, pakai 0.5: %s", e)
            return 0.5

    if want_bilstm:
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
        plot_training_history(history_bilstm, "BiLSTM", config.output_dir)
        thr_bilstm = _threshold_from_val(model_bilstm, X_train_pad, y_train)

    if want_lstm:
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
        plot_training_history(history_lstm, "LSTM", config.output_dir)
        thr_lstm = _threshold_from_val(model_lstm, X_train_pad, y_train)

    bert_results = None
    if want_bert:
        # Delegasi ke modul BERT (torch/transformers opsional — tidak wajib untuk BiLSTM).
        try:
            from app.core.model.bert.trainer_bert import run_bert_training

            logger.info("=" * 60)
            logger.info("TRAINING: BERT (%s)", config.bert.model_name)
            logger.info("=" * 60)
            bert_results = run_bert_training(
                config=config,
                X_train_text=X_train_text,
                X_test_text=X_test_text,
                y_train=np.asarray(y_train),
                y_test=np.asarray(y_test),
                label_encoder=label_encoder,
            )
        except ImportError as e:
            logger.warning(
                "Modul BERT dilewati (dependensi belum terinstal: %s). "
                "pip install -r requirements.txt untuk aktifkan.",
                e,
            )
            want_bert = False

    # Evaluation (threshold hasil tuning-val, bukan 0.5 hardcoded)
    rows = []
    extra_curves: dict = {}
    if want_bilstm and model_bilstm is not None:
        eval_bilstm, report_bilstm, pred_bilstm, fpr_bilstm, tpr_bilstm, auc_bilstm = (
            evaluate_model(
                model_bilstm,
                "BiLSTM",
                X_test_pad,
                y_test,
                X_test_text,
                label_encoder,
                config.output_dir,
                threshold=thr_bilstm,
            )
        )
        rows.append(
            {
                "model": "BiLSTM",
                "best_params": str(best_params_bilstm),
                **{row["metric"]: row["value"] for _, row in eval_bilstm.iterrows()},
            }
        )

    if want_lstm and model_lstm is not None:
        eval_lstm, report_lstm, pred_lstm, fpr_lstm, tpr_lstm, auc_lstm = evaluate_model(
            model_lstm,
            "LSTM",
            X_test_pad,
            y_test,
            X_test_text,
            label_encoder,
            config.output_dir,
            threshold=thr_lstm,
        )
        rows.append(
            {
                "model": "LSTM",
                "best_params": str(best_params_lstm),
                **{row["metric"]: row["value"] for _, row in eval_lstm.iterrows()},
            }
        )

    if bert_results is not None:
        rows.append(
            {
                "model": f"BERT ({bert_results.get('model_name', config.bert.model_name)})",
                "best_params": str(bert_results.get("best_params", {})),
                **bert_results.get("metrics", {}),
            }
        )
        b_curve = bert_results.get("roc_curve")
        if b_curve is not None:
            extra_curves[f"BERT"] = b_curve

    # ROC comparison (variadik; tetap kompatibel 2-model lama)
    if fpr_bilstm is not None and fpr_lstm is not None:
        plot_roc_comparison(
            fpr_bilstm, tpr_bilstm, auc_bilstm,
            fpr_lstm, tpr_lstm, auc_lstm,
            config.output_dir,
            extra_curves=extra_curves or None,
            filename="roc_comparison_word2vec.png",
            title="ROC Curve Comparison",
        )
    elif fpr_bilstm is not None:
        plot_roc_comparison(
            fpr_bilstm, tpr_bilstm, auc_bilstm,
            None, None, None,
            config.output_dir,
            extra_curves=extra_curves or None,
            filename="roc_comparison_word2vec.png",
            title="ROC Curve Comparison",
        )

    # Comparison table
    comparison = pd.DataFrame(rows) if rows else pd.DataFrame([{"model": "none"}])
    comparison.to_csv(os.path.join(config.output_dir, "comparison_models.csv"), index=False)
    # Nama lama dipertahankan untuk kompatibilitas.
    comparison.to_csv(os.path.join(config.output_dir, "comparison_bilstm_vs_lstm.csv"), index=False)

    # Threshold per model (dipakai service & evaluate agar konsisten)
    with open(os.path.join(config.output_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"bilstm": thr_bilstm, "lstm": thr_lstm,
             **({"bert": float(bert_results.get("threshold", 0.5))} if bert_results else {})},
            f,
            indent=2,
        )

    # Save models (nama eksperimen + nama kompatibel ML service).
    import pickle

    os.makedirs(config.model_dir, exist_ok=True)
    if model_bilstm is not None:
        model_bilstm.save(os.path.join(config.model_dir, "bilstm_word2vec.keras"))
        # Alias untuk ML service (app/services/model_registry.py).
        model_bilstm.save(os.path.join(config.model_dir, "bilstm_model.keras"))
    if model_lstm is not None:
        model_lstm.save(os.path.join(config.model_dir, "lstm_word2vec.keras"))
    try:
        w2v_model.save(os.path.join(config.model_dir, "word2vec.model"))
    except Exception as e:
        logger.warning("Gagal menyimpan word2vec.model: %s", e)
    try:
        with open(os.path.join(config.model_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)
        tokenizer.save_json(os.path.join(config.model_dir, "tokenizer_bilstm.json"))
        with open(os.path.join(config.model_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)
        with open(os.path.join(config.model_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {str(k): int(v) for k, v in zip(
                    label_encoder.classes_,
                    label_encoder.transform(label_encoder.classes_))},
                f, indent=2,
            )
        with open(os.path.join(config.model_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump({"bilstm": thr_bilstm, "lstm": thr_lstm}, f, indent=2)
    except Exception as e:
        logger.warning("Gagal menyimpan tokenizer/label_encoder: %s", e)

    # metadata.json — identitas artefak untuk registry dual-model.
    try:
        import subprocess

        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_sha = "unknown"
        metadata = {
            "model_types": [r["model"] for r in rows],
            "vocab_size": config.vocab_size,
            "max_len": config.max_len,
            "embed_dim": config.embed_dim,
            "seed": config.seed,
            "label_map": {str(k): int(v) for k, v in zip(
                label_encoder.classes_,
                label_encoder.transform(label_encoder.classes_))},
            "thresholds": {"bilstm": thr_bilstm, "lstm": thr_lstm},
            "git_sha": git_sha,
        }
        with open(os.path.join(config.model_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning("Gagal menulis metadata.json: %s", e)

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
        "bert": bert_results,
        "thresholds": {"bilstm": thr_bilstm, "lstm": thr_lstm},
    }
