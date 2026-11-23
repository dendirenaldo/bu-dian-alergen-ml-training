"""Pipeline training BiLSTM + Word2Vec yang bisa dipakai baseline, HPO, dan final.

Alur satu trial:

1. Vektorisasi teks (Tokenizer + Word2Vec) — hanya dari final-train.
2. Bangun model (LSTM/GRU, 1–2 lapisan, bidirectional/tidak).
3. Latih dengan callbacks (early stopping + reduce LR).
4. Prediksi validasi dan frozen holdout.
5. Hitung metrik pada threshold tetap 0,5.
6. Simpan history/probabilitas (model opsional, agar HPO hemat disk).

Holdout **tidak pernah** dipakai untuk training, early stopping, atau
pemilihan hyperparameter. Holdout hanya diprediksi sekali di akhir.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.model.metrics import (
    evaluate_fixed_threshold,
    oov_stats_for_sequences,
    plot_confusion_v5,
    plot_roc_pr_v5,
    save_training_curves_v5,
    write_experiment_manifest,
    write_thresholds,
)
from app.core.preprocessing.text import fold_digits_v5, simple_tokenize
from app.experiment.artifacts import environment_snapshot, write_json
from app.experiment.configs import BiLSTMConfig
from app.experiment.data_pipeline import DataBundle
from app.experiment.reporting import bootstrap_ci, classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class VectorizedData:
    """Hasil vektorisasi satu konfigurasi."""

    config: BiLSTMConfig
    tokenizer: Any
    w2v_model: Any
    embedding_matrix: np.ndarray
    num_words: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_holdout: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_holdout: np.ndarray
    train_texts: list[str]
    val_texts: list[str]
    holdout_texts: list[str]
    oov_table: pd.DataFrame
    w2v_hits: int
    w2v_coverage: float

    @property
    def vectorizer_key(self) -> str:
        """Kunci cache: parameter yang memengaruhi tokenizer/Word2Vec."""
        keys = (
            "vocab_size", "max_len", "embed_dim", "w2v_window", "w2v_min_count",
            "w2v_epochs", "w2v_sg", "w2v_workers", "embedding_init_scale",
            "digit_fold", "seed",
        )
        payload = {key: getattr(self.config, key) for key in keys}
        return json.dumps(payload, sort_keys=True)


@dataclass
class BilstmTrialResult:
    """Hasil satu trial BiLSTM."""

    metrics: dict[str, float]
    history: pd.DataFrame
    val_prob: np.ndarray
    holdout_prob: np.ndarray | None
    artifacts: dict[str, str] = field(default_factory=dict)
    model: Any = None
    vectorized: VectorizedData | None = None


def _apply_digit_fold(texts: list[str], enabled: bool) -> list[str]:
    return [fold_digits_v5(text) for text in texts] if enabled else list(texts)


def build_vectorizers(
    bundle: DataBundle,
    config: BiLSTMConfig,
    *,
    reuse: VectorizedData | None = None,
) -> VectorizedData:
    """Fit Tokenizer + Word2Vec hanya pada final-train, lalu encode semua split.

    Args:
        bundle: data yang sudah diaudit.
        config: konfigurasi vektorisasi.
        reuse: hasil sebelumnya; dipakai bila parameter vektorisasi sama.
    """
    if reuse is not None and reuse.vectorizer_key == vectorizer_key(config):
        return reuse

    try:
        from app.core.embedding.word2vec import build_embedding_matrix, train_word2vec
        from app.core.model.tokenizer import Tokenizer
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "Training BiLSTM membutuhkan TensorFlow + Gensim. "
            "Aktifkan environment Python 3.10–3.12 lalu "
            "`pip install -r requirements-train.txt`."
        ) from exc

    train_texts = _apply_digit_fold(bundle.X_train_text, config.digit_fold)
    val_texts = _apply_digit_fold(bundle.X_val_text, config.digit_fold)
    holdout_texts = _apply_digit_fold(bundle.X_holdout_text, config.digit_fold)

    tokenizer = Tokenizer(vocab_size=config.vocab_size, max_len=config.max_len)
    tokenizer.fit(train_texts)
    X_train = tokenizer.encode(train_texts)
    X_val = tokenizer.encode(val_texts)
    X_holdout = tokenizer.encode(holdout_texts)

    train_tokens = [simple_tokenize(text) for text in train_texts]
    w2v_model = train_word2vec(
        sentences=train_tokens,
        vector_size=config.embed_dim,
        window=config.w2v_window,
        min_count=config.w2v_min_count,
        workers=config.w2v_workers,
        epochs=config.w2v_epochs,
        seed=config.seed,
        sg=config.w2v_sg,
    )
    embedding_matrix, num_words, hits = build_embedding_matrix(
        w2v_model=w2v_model,
        word_index=tokenizer.word_index,
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        init_scale=config.embedding_init_scale,
        seed=config.seed,
    )
    embedding_matrix = embedding_matrix.astype("float32")
    embedding_matrix[0] = np.zeros((config.embed_dim,), dtype="float32")

    oov_table = pd.DataFrame(
        [
            {"split": "final_train", **oov_stats_for_sequences(tokenizer.texts_to_sequences(train_texts))},
            {"split": "validation", **oov_stats_for_sequences(tokenizer.texts_to_sequences(val_texts))},
            {"split": "frozen_holdout", **oov_stats_for_sequences(tokenizer.texts_to_sequences(holdout_texts))},
        ]
    )
    coverage = hits / max(1, min(num_words, len(tokenizer.word_index)))
    return VectorizedData(
        config=config,
        tokenizer=tokenizer,
        w2v_model=w2v_model,
        embedding_matrix=embedding_matrix,
        num_words=num_words,
        X_train=X_train,
        X_val=X_val,
        X_holdout=X_holdout,
        y_train=bundle.y_train,
        y_val=bundle.y_val,
        y_holdout=bundle.y_holdout,
        train_texts=train_texts,
        val_texts=val_texts,
        holdout_texts=holdout_texts,
        oov_table=oov_table,
        w2v_hits=hits,
        w2v_coverage=float(coverage),
    )


def vectorizer_key(config: BiLSTMConfig) -> str:
    keys = (
        "vocab_size", "max_len", "embed_dim", "w2v_window", "w2v_min_count",
        "w2v_epochs", "w2v_sg", "w2v_workers", "embedding_init_scale",
        "digit_fold", "seed",
    )
    return json.dumps({key: getattr(config, key) for key in keys}, sort_keys=True)


def _class_weight(y_train: np.ndarray) -> dict[int, float] | None:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _history_frame(history: Any) -> pd.DataFrame:
    raw = history.history if hasattr(history, "history") else dict(history)
    frame = pd.DataFrame(raw)
    if {"precision", "recall"}.issubset(frame.columns):
        frame["f1"] = 2 * frame["precision"] * frame["recall"] / (
            frame["precision"] + frame["recall"] + 1e-8
        )
    if {"val_precision", "val_recall"}.issubset(frame.columns):
        frame["val_f1"] = 2 * frame["val_precision"] * frame["val_recall"] / (
            frame["val_precision"] + frame["val_recall"] + 1e-8
        )
    return frame


def _best_epoch(history: pd.DataFrame, config: BiLSTMConfig) -> int:
    if config.monitor in history.columns:
        series = history[config.monitor]
        index = series.idxmax() if config.monitor_mode == "max" else series.idxmin()
        return int(index) + 1
    if "val_loss" in history.columns:
        return int(history["val_loss"].idxmin()) + 1
    return 1


def _save_probabilities(
    output_dir: Path,
    val_texts: list[str],
    y_val: np.ndarray,
    val_prob: np.ndarray,
    holdout_texts: list[str] | None = None,
    y_holdout: np.ndarray | None = None,
    holdout_prob: np.ndarray | None = None,
) -> dict[str, str]:
    """Simpan probabilitas validasi; holdout hanya bila diminta."""
    output_dir.mkdir(parents=True, exist_ok=True)
    val_df = pd.DataFrame(
        {
            "text": val_texts,
            "label": np.where(y_val == 1, "unsafe", "safe"),
            "label_id": y_val.astype(int),
            "prob_unsafe": val_prob.astype(float),
        }
    )
    val_path = output_dir / "probs_val.csv"
    val_df.to_csv(val_path, index=False)
    # Alias lama yang dibaca paper/make_figs.py.
    val_df[["label_id", "prob_unsafe"]].to_csv(output_dir / "val_probs.csv", index=False)
    artifacts = {"probs_val": str(val_path)}
    if holdout_prob is not None and holdout_texts is not None and y_holdout is not None:
        hold_df = pd.DataFrame(
            {
                "text": holdout_texts,
                "label": np.where(y_holdout == 1, "unsafe", "safe"),
                "label_id": y_holdout.astype(int),
                "prob_unsafe": holdout_prob.astype(float),
            }
        )
        hold_path = output_dir / "probs_holdout.csv"
        hold_df.to_csv(hold_path, index=False)
        artifacts["probs_holdout"] = str(hold_path)
    return artifacts


def _save_roc_data(
    output_dir: Path,
    y_val,
    val_prob,
    y_hold=None,
    hold_prob=None,
) -> str:
    from sklearn.metrics import roc_curve

    payload = {}
    for name, y_true, y_prob in (
        ("val", y_val, val_prob), ("holdout", y_hold, hold_prob),
    ):
        if y_true is None or y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        payload[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    path = output_dir / "roc_data.json"
    write_json(path, payload)
    return str(path)


def run_bilstm_trial(
    bundle: DataBundle,
    config: BiLSTMConfig,
    output_dir: str | os.PathLike,
    *,
    vectorized: VectorizedData | None = None,
    save_model: bool = False,
    model_save_path: str | os.PathLike | None = None,
    save_predictions: bool = True,
    evaluate_holdout: bool = True,
    verbose: int = 0,
) -> BilstmTrialResult:
    """Latih satu konfigurasi BiLSTM dan evaluasi di val + holdout.

    Args:
        bundle: data siap-training.
        config: hiperparameter satu trial.
        output_dir: direktori artefak trial.
        vectorized: hasil vektorisasi yang bisa dipakai ulang.
        save_model: simpan model Keras.
        model_save_path: path model (default ``output_dir/model.keras``).
        save_predictions: simpan CSV probabilitas.
        evaluate_holdout: prediksi holdout. Untuk HPO **harus False** agar
            holdout tidak disentuh sampai model terbaik dipilih.
        verbose: nilai ``verbose`` untuk ``model.fit``.
    """
    from app.repro import apply_reproducibility_lock, seed_all

    apply_reproducibility_lock(config.seed, deterministic=config.deterministic)
    seed_all(config.seed, deterministic=config.deterministic)

    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "Training BiLSTM membutuhkan TensorFlow + Gensim. "
            "Aktifkan environment Python 3.10–3.12 lalu "
            "`pip install -r requirements-train.txt`."
        ) from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vectorized = build_vectorizers(bundle, config, reuse=vectorized)
    vectorized.oov_table.to_csv(out / "oov_audit.csv", index=False)

    from app.core.model.architecture import build_model

    tf.keras.backend.clear_session()
    model = build_model(
        config.use_bidirectional,
        config.lstm_units_1,
        config.lstm_units_2,
        config.dropout_1,
        config.dropout_2,
        config.learning_rate,
        vectorized.num_words,
        config.embed_dim,
        vectorized.embedding_matrix,
        embed_trainable=config.embed_trainable,
        recurrent_dropout=config.recurrent_dropout,
        dense_units=config.dense_units,
        dropout_rate_dense=config.dropout_dense,
        gradient_clip_norm=config.gradient_clip_norm,
        mask_zero=config.mask_zero,
        use_maxnorm=config.use_maxnorm,
        rnn_type=config.rnn_type,
        num_rnn_layers=config.num_rnn_layers,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        maxnorm_value=config.maxnorm_value,
        label_smoothing=config.label_smoothing,
    )

    X_train = vectorized.X_train
    y_train = vectorized.y_train
    if config.preshuffle:
        rng = np.random.RandomState(config.seed)
        perm = rng.permutation(len(y_train))
        X_train, y_train = X_train[perm], y_train[perm]

    class_weight = _class_weight(y_train) if config.use_class_weight else None
    checkpoint_path = Path(model_save_path) if model_save_path else out / "model.keras"
    callbacks = [
        EarlyStopping(
            monitor=config.monitor,
            mode=config.monitor_mode,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.lr_reduce_factor,
            patience=config.lr_reduce_patience,
            min_lr=config.min_lr,
        ),
    ]
    if save_model:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=config.monitor,
                mode=config.monitor_mode,
                save_best_only=True,
                verbose=0,
            )
        )

    started = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(vectorized.X_val, vectorized.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        shuffle=config.shuffle,
        class_weight=class_weight,
        verbose=verbose,
    )
    train_seconds = time.time() - started

    val_prob = np.asarray(model.predict(vectorized.X_val, verbose=0)).ravel()
    holdout_prob: np.ndarray | None = None
    if evaluate_holdout:
        holdout_prob = np.asarray(model.predict(vectorized.X_holdout, verbose=0)).ravel()

    val_metrics = classification_metrics(vectorized.y_val, val_prob)
    holdout_metrics = (
        classification_metrics(vectorized.y_holdout, holdout_prob)
        if holdout_prob is not None else {}
    )
    hist = _history_frame(history)
    hist.to_csv(out / "training_history.csv", index=False)
    best_epoch = _best_epoch(hist, config)
    metrics = {
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
        "val_loss": float(hist["val_loss"].iloc[best_epoch - 1]) if "val_loss" in hist else float("nan"),
        "best_epoch": float(best_epoch),
        "epochs_ran": float(len(hist)),
        "train_seconds": float(train_seconds),
        "w2v_coverage": float(vectorized.w2v_coverage),
        "num_words": float(vectorized.num_words),
    }
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "epochs_ran": len(hist),
            "train_seconds": train_seconds,
            "monitor": config.monitor,
            "monitor_mode": config.monitor_mode,
            "class_weight": class_weight,
            "config": config.to_dict(),
        },
    )

    artifacts: dict[str, str] = {
        "metrics": str(out / "metrics.json"),
        "history": str(out / "training_history.csv"),
        "oov_audit": str(out / "oov_audit.csv"),
    }
    if save_predictions:
        artifacts.update(
            _save_probabilities(
                out,
                vectorized.val_texts, vectorized.y_val, val_prob,
                vectorized.holdout_texts if holdout_prob is not None else None,
                vectorized.y_holdout if holdout_prob is not None else None,
                holdout_prob,
            )
        )
        artifacts["roc_data"] = _save_roc_data(
            out, vectorized.y_val, val_prob,
            vectorized.y_holdout if holdout_prob is not None else None,
            holdout_prob,
        )
    if save_model:
        artifacts["model"] = str(checkpoint_path)
        # Simpan juga tokenizer + Word2Vec agar model bisa dipakai ulang.
        with (out / "tokenizer.pkl").open("wb") as handle:
            pickle.dump(vectorized.tokenizer, handle)
        vectorized.tokenizer.save_json(str(out / "tokenizer.json"))
        try:
            vectorized.w2v_model.save(str(out / "word2vec.model"))
        except Exception as exc:  # pragma: no cover - bergantung gensim
            logger.warning("Gagal menyimpan Word2Vec: %s", exc)

    return BilstmTrialResult(
        metrics=metrics,
        history=hist,
        val_prob=val_prob,
        holdout_prob=holdout_prob,
        artifacts=artifacts,
        model=model,
        vectorized=vectorized,
    )


def train_final_bilstm(
    bundle: DataBundle,
    config: BiLSTMConfig,
    output_dir: str | os.PathLike,
    model_dir: str | os.PathLike,
    *,
    hpo_info: dict[str, Any] | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Latih model final, simpan artefak lengkap, dan evaluasi holdout sekali.

    Nama file mengikuti kontrak lama agar `scripts/export_for_serving.py`
    dan paper builder tetap bisa membacanya.
    """
    output = Path(output_dir)
    models = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    canonical_model = models / "bilstm_word2vec_v5.keras"
    result = run_bilstm_trial(
        bundle,
        config,
        output,
        save_model=True,
        model_save_path=canonical_model,
        save_predictions=True,
        verbose=verbose,
    )
    assert result.vectorized is not None
    assert result.holdout_prob is not None, "Training final wajib mengevaluasi holdout."

    # Simpan tokenizer + Word2Vec dengan nama kontrak lama.
    with (models / "tokenizer_v5.pkl").open("wb") as handle:
        pickle.dump(result.vectorized.tokenizer, handle)
    result.vectorized.tokenizer.save_json(str(models / "tokenizer_bilstm_v5.json"))
    try:
        result.vectorized.w2v_model.save(str(models / "word2vec_v5.model"))
    except Exception as exc:  # pragma: no cover
        logger.warning("Gagal menyimpan word2vec_v5.model: %s", exc)

    write_thresholds(str(models / "thresholds.json"), 0.5)
    write_thresholds(str(output / "thresholds.json"), 0.5)

    # Tabel evaluasi kontrak lama.
    eval_table = evaluate_fixed_threshold(
        bundle.y_val, result.val_prob,
        bundle.y_holdout, result.holdout_prob,
        threshold=0.5,
    )
    eval_table.to_csv(output / "evaluation_table_v5.csv", index=False)
    result.vectorized.oov_table.to_csv(output / "oov_audit_v5.csv", index=False)
    bundle.split_manifest.to_csv(output / "split_manifest.csv", index=False)

    # Kurva & confusion matrix.
    try:
        save_training_curves_v5(result.history.to_dict(orient="list"), str(output))
        plot_roc_pr_v5(bundle.y_val, result.val_prob, bundle.y_holdout, result.holdout_prob, str(output))
        plot_confusion_v5(
            bundle.y_holdout, result.holdout_prob, str(output),
            name="frozen_holdout", threshold=0.5,
        )
    except Exception as exc:  # pragma: no cover - plotting tidak boleh menggagalkan training
        logger.warning("Sebagian plot gagal dibuat: %s", exc)

    # Bootstrap CI per split (skema lama: kolom split + metrik).
    ci_rows = []
    for split_name, y_true, y_prob in (
        ("validation", bundle.y_val, result.val_prob),
        ("frozen_holdout", bundle.y_holdout, result.holdout_prob),
    ):
        table = bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=2000, seed=config.seed)
        table.insert(0, "split", split_name)
        ci_rows.append(table)
    pd.concat(ci_rows, ignore_index=True).to_csv(output / "bootstrap_ci_v5.csv", index=False)

    # Triase Gold vs KB vs model (diagnostik, bukan tuning).
    try:
        from app.core.model.audit import triage_gold_kb_bilstm

        triage = triage_gold_kb_bilstm(
            bundle.df_holdout, result.holdout_prob,
            product_col=bundle.config.product_col, threshold=0.5,
        )
        triage["triage"].to_csv(output / "gold_kb_bilstm_triage.csv", index=False)
        triage["false_negatives"].to_csv(output / "frozen_holdout_false_negatives.csv", index=False)
        triage["false_positives"].to_csv(output / "frozen_holdout_false_positives.csv", index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Triase gagal dibuat: %s", exc)

    metadata = {
        "model_type": "bilstm_word2vec",
        "pipeline": "experiment_v6",
        "threshold": 0.5,
        "threshold_fixed": True,
        "max_len": config.max_len,
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "label_mapping": {"safe": 0, "unsafe": 1},
        "config": config.to_dict(),
        "data": bundle.metadata,
        "metrics": result.metrics,
        "hpo": hpo_info or {},
        "environment": environment_snapshot(seed=config.seed),
    }
    write_json(models / "metadata.json", metadata)
    write_json(output / "metadata.json", metadata)
    write_json(
        models / "model_card.json",
        {
            "model": "BiLSTM + Word2Vec",
            "config": config.to_dict(),
            "metrics": result.metrics,
            "data": {
                "csv_sha256": bundle.metadata.get("csv_sha256"),
                "frozen_holdout_sha256": bundle.frozen_sha256,
                "pool_sizes": bundle.pool_sizes,
            },
            "environment": environment_snapshot(seed=config.seed),
        },
    )

    # Alias untuk serving langsung (opsional, kecil).
    try:
        import shutil

        shutil.copy2(canonical_model, models / "bilstm_model.keras")
    except Exception:
        pass

    write_experiment_manifest(
        str(output),
        seed=config.seed,
        threshold=0.5,
        holdout_size=len(bundle.y_holdout),
        extra={
            "train_size": int(len(bundle.y_train)),
            "val_size": int(len(bundle.y_val)),
            "synthetic_total": int(len(bundle.df_synthetic)),
            "frozen_sha256": bundle.frozen_sha256,
            "final_config": config.to_dict(),
            "hpo": hpo_info or {},
            "environment": environment_snapshot(seed=config.seed),
        },
    )

    return {
        "metrics": result.metrics,
        "eval_table": eval_table,
        "config": config.to_dict(),
        "artifacts": result.artifacts,
        "output_dir": str(output),
        "model_dir": str(models),
        "result": result,
    }
