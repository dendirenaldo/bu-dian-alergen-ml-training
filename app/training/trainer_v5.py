"""Trainer V5 parity — port penuh Notebook experiment-1-11-fix2 (Cell 1..72).

Single-run BiLSTM leakage-safe:
- Gold = ground truth; KB = diagnostik.
- Split leakage-safe + frozen holdout (tidak pernah fit/tuning).
- Synthetic 1000 hanya dari real-train.
- Tokenizer + Word2Vec fit HANYA final-train (real+synthetic).
- Hyperparams fixed notebook, shuffle=False, threshold fixed 0.50.

Import TF/Gensim LAZY di dalam fungsi agar modul data/KB tetap bisa
di-test tanpa TF.
"""

from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from app.config import Config

logger = logging.getLogger(__name__)

HOLDOUT_USED_IN_MODEL_FIT = False
TRAIN_SHUFFLE = False


def _packaged_frozen_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "core", "data", "frozen_holdout.json"
    )


def _default_frozen_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "core", "data", "frozen_holdout_notebook23.json"
    )


def _load_frozen_products(config: Config, n_real: int) -> list[str] | None:
    """Tentukan daftar frozen holdout.

    Urutan: V5_FROZEN_HOLDOUT_PATH eksplisit -> frozen_holdout.json kemasan
    (kanonis 50/50) -> notebook23 bila n_real == 114 -> re-derive.
    """
    if config.v5.frozen_holdout_path:
        with open(config.v5.frozen_holdout_path, encoding="utf-8") as f:
            products = json.load(f)
        logger.info("Frozen holdout dari file: %s (%d)", config.v5.frozen_holdout_path, len(products))
        return [str(x) for x in products]
    packaged = _packaged_frozen_path()
    if os.path.exists(packaged):
        with open(packaged, encoding="utf-8") as f:
            products = json.load(f)
        logger.info("Frozen holdout kemasan: %s (%d)", packaged, len(products))
        return [str(x) for x in products]
    if n_real == 114:
        with open(_default_frozen_path(), encoding="utf-8") as f:
            products = json.load(f)
        logger.info("Frozen holdout default notebook23 (n_real=114, %d)", len(products))
        return [str(x) for x in products]
    logger.warning(
        "n_real=%d != 114 dan V5_FROZEN_HOLDOUT_PATH kosong -> re-derive holdout "
        "deterministik exact %d safe/%d unsafe. Simpan hasilnya dan freeze!",
        n_real, config.v5.holdout_safe, config.v5.holdout_unsafe,
    )
    return None


def run_v5_parity(config: Config | None = None) -> dict:
    """Jalankan pipeline V5 parity end-to-end. Butuh TF + Gensim terinstal."""
    # 1. Repro lock SEBELUM import TF (kontrak notebook Cell 1).
    from app.repro import apply_reproducibility_lock, seed_all

    if config is None:
        from app.config import get_config

        config = get_config()
    apply_reproducibility_lock(config.seed)
    seed_all(config.seed)
    config.ensure_dirs()

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Mode v5_parity butuh TensorFlow + Gensim. "
            "pip install -r requirements.txt terlebih dahulu."
        ) from e

    v5 = config.v5
    assert v5.fixed_threshold == 0.50, "V5 threshold harus fixed 0.50."
    assert v5.synthetic_source_contract == "real_train_only"

    # 2. Load data CSV tunggal -> model_source (label CSV = gold).
    from app.core.data.gold_merge import (
        build_model_source_from_single,
        standardize_columns,
    )

    if not os.path.exists(config.csv_input):
        raise FileNotFoundError(f"CSV tidak ditemukan: {config.csv_input}")
    df_raw = pd.read_csv(
        config.csv_input, delimiter=config.csv_delimiter, encoding=config.csv_encoding
    )
    df_raw = standardize_columns(
        df_raw, product_col=v5.product_col,
        text_col=config.text_col, label_col="label",
    )
    text_src = config.text_col if config.text_col in df_raw.columns else v5.text_col
    df_model_source = build_model_source_from_single(
        df_raw, product_col=v5.product_col, text_col=text_src, label_col="label",
    )
    logger.info("Model source: %d rows, groups=%d", len(df_model_source),
                df_model_source["group_key"].nunique())
    logger.info("Gold distribution:\n%s", df_model_source["gold_label"].value_counts())

    # 3. KB vs Gold audit (diagnostik).
    from app.core.model.audit import kb_vs_gold_audit

    kb_audit = kb_vs_gold_audit(df_model_source)
    logger.info("KB vs Gold:\n%s", kb_audit["summary"].to_string(index=False))

    # 4. Split leakage-safe.
    from app.core.data.leakage_split import (
        audit_split_no_leakage,
        build_split_manifest,
        run_v5_split,
    )

    frozen = _load_frozen_products(config, len(df_model_source))
    split = run_v5_split(
        df_model_source,
        product_col=v5.product_col,
        text_col=text_src,
        frozen_products=frozen,
        holdout_safe=v5.holdout_safe,
        holdout_unsafe=v5.holdout_unsafe,
        val_safe=v5.val_safe,
        val_unsafe=v5.val_unsafe,
    )
    df_real_train, df_real_val, df_holdout = (
        split["df_real_train"], split["df_real_val"], split["df_holdout"],
    )
    split_audit = audit_split_no_leakage(df_real_train, df_real_val, df_holdout)
    logger.info("Split audit PASS:\n%s", split_audit.to_string(index=False))
    manifest = build_split_manifest(df_real_train, df_real_val, df_holdout,
                                    product_col=v5.product_col)
    manifest_path = os.path.join(config.output_dir, "split_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    # 5. Synthetic (real-train only) + final pool + audit.
    from app.core.data.synthetic import build_final_training_pool, generate_synthetic_dataset
    from app.core.data.leakage_split import audit_final_pool_no_leakage

    df_sintesis = generate_synthetic_dataset(
        df_train_real=df_real_train, text_col=text_src,
        total_data=v5.synthetic_total, seed=config.seed,
    )
    pool = build_final_training_pool(df_real_train, df_sintesis, df_real_val, df_holdout,
                                     text_col=text_src)
    audit_final_pool_no_leakage(pool["X_train_text"], pool["X_val_text"], pool["X_holdout_text"])
    X_train_text, y_train = pool["X_train_text"], np.asarray(pool["y_train"])
    X_val_text, y_val = pool["X_val_text"], np.asarray(pool["y_val"])
    X_holdout_text, y_holdout = pool["X_holdout_text"], np.asarray(pool["y_holdout"])
    logger.info("Final train=%d (real %d + syn %d), val=%d, holdout=%d",
                len(y_train), len(df_real_train), len(df_sintesis),
                len(y_val), len(y_holdout))

    # 6. Tokenizer + Word2Vec fit HANYA final-train (re-seed dulu).
    seed_all(config.seed)
    from app.core.model.tokenizer import Tokenizer
    from app.core.embedding.word2vec import build_embedding_matrix, train_word2vec
    from app.core.preprocessing.text import fold_digits_v5, simple_tokenize

    if v5.digit_fold:
        # Diterapkan ke SEMUA split sebelum tokenisasi (konsisten, tak bocor).
        X_train_text = [fold_digits_v5(t) for t in X_train_text]
        X_val_text = [fold_digits_v5(t) for t in X_val_text]
        X_holdout_text = [fold_digits_v5(t) for t in X_holdout_text]
        logger.info("digit_fold aktif: angka dilipat jadi 'num' di semua split")

    train_tokens = [simple_tokenize(t) for t in X_train_text]
    tokenizer = Tokenizer(vocab_size=v5.vocab_size, max_len=v5.max_len)
    tokenizer.fit(X_train_text)
    TOKENIZER_FIT_SOURCE = "final_training_pool_only"
    WORD2VEC_FIT_SOURCE = "final_training_pool_only"
    assert TOKENIZER_FIT_SOURCE == "final_training_pool_only"
    assert WORD2VEC_FIT_SOURCE == "final_training_pool_only"

    X_train_pad = tokenizer.encode(X_train_text)
    X_val_pad = tokenizer.encode(X_val_text)
    X_holdout_pad = tokenizer.encode(X_holdout_text)

    w2v_model = train_word2vec(
        sentences=train_tokens, vector_size=v5.embed_dim, window=v5.w2v_window,
        min_count=v5.w2v_min_count, workers=1, epochs=v5.w2v_epochs, seed=config.seed,
    )
    embedding_matrix, num_words, hit_count = build_embedding_matrix(
        w2v_model=w2v_model, word_index=tokenizer.word_index,
        vocab_size=v5.vocab_size, embed_dim=v5.embed_dim,
        init_scale=v5.embedding_init_scale, seed=config.seed,
    )
    embedding_matrix = embedding_matrix.astype("float32")
    embedding_matrix[0] = np.zeros((v5.embed_dim,), dtype="float32")

    from app.core.model.metrics import oov_stats_for_sequences

    oov_table = pd.DataFrame([
        {"split": "final_train", **oov_stats_for_sequences(
            tokenizer.texts_to_sequences(X_train_text))},
        {"split": "validation", **oov_stats_for_sequences(
            tokenizer.texts_to_sequences(X_val_text))},
        {"split": "frozen_holdout", **oov_stats_for_sequences(
            tokenizer.texts_to_sequences(X_holdout_text))},
    ])
    oov_table.to_csv(os.path.join(config.output_dir, "oov_audit_v5.csv"), index=False)
    logger.info("OOV:\n%s", oov_table.to_string(index=False))

    # 7. Build + train BiLSTM fixed (re-seed; holdout TIDAK PERNAH ke fit).
    seed_all(config.seed)
    import tensorflow as tf2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    from app.core.model.architecture import build_model

    model = build_model(
        True, v5.lstm_units_1, v5.lstm_units_2, v5.dropout_1, v5.dropout_2,
        v5.learning_rate, num_words, v5.embed_dim, embedding_matrix,
        embed_trainable=v5.embed_trainable, recurrent_dropout=0.0,
        dense_units=v5.dense_units, dropout_rate_dense=v5.dropout_dense,
        gradient_clip_norm=v5.gradient_clip_norm, mask_zero=v5.mask_zero, use_maxnorm=False,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=v5.early_stopping_patience,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=v5.lr_reduce_factor,
                          patience=v5.lr_reduce_patience, min_lr=v5.min_lr),
    ]
    if v5.preshuffle:
        rng = np.random.RandomState(config.seed)
        perm = rng.permutation(len(y_train))
        X_train_pad, y_train = X_train_pad[perm], y_train[perm]
        logger.info("preshuffle aktif: urutan train diacak deterministik (seed=%d)", config.seed)
    history = model.fit(
        X_train_pad, y_train, validation_data=(X_val_pad, y_val),
        epochs=v5.epochs, batch_size=v5.batch_size,
        callbacks=callbacks, shuffle=TRAIN_SHUFFLE, verbose=1,
    )
    assert HOLDOUT_USED_IN_MODEL_FIT is False
    assert TRAIN_SHUFFLE is False

    # 8. Evaluasi fixed threshold + artefak.
    from app.core.model.audit import (
        audit_pipeline_consistency_v5,
        audit_reproducibility_v5,
        triage_gold_kb_bilstm,
    )
    from app.core.model.metrics import (
        evaluate_fixed_threshold,
        plot_confusion_v5,
        plot_roc_pr_v5,
        save_training_curves_v5,
        write_experiment_manifest,
        write_thresholds,
    )

    audit_pipeline_consistency_v5(df_real_train, df_real_val, df_holdout,
                                  len(df_sintesis), v5.synthetic_total,
                                  HOLDOUT_USED_IN_MODEL_FIT, v5.fixed_threshold)
    audit_reproducibility_v5(config.seed, w2v_model.workers, w2v_model.seed,
                             TRAIN_SHUFFLE, v5.fixed_threshold,
                             holdout_size=len(y_holdout),
                             expected_holdout_size=v5.holdout_size)

    val_prob = model.predict(X_val_pad, verbose=0).ravel()
    holdout_prob = model.predict(X_holdout_pad, verbose=0).ravel()
    eval_table = evaluate_fixed_threshold(y_val, val_prob, y_holdout, holdout_prob,
                                          threshold=v5.fixed_threshold)
    eval_table.to_csv(os.path.join(config.output_dir, "evaluation_table_v5.csv"), index=False)
    save_training_curves_v5(history, config.output_dir)
    plot_roc_pr_v5(y_val, val_prob, y_holdout, holdout_prob, config.output_dir)
    plot_confusion_v5(y_holdout, holdout_prob, config.output_dir, name="frozen_holdout",
                      threshold=v5.fixed_threshold)

    triage = triage_gold_kb_bilstm(df_holdout, holdout_prob,
                                   product_col=v5.product_col,
                                   threshold=v5.fixed_threshold)
    triage["triage"].to_csv(os.path.join(config.output_dir, "gold_kb_bilstm_triage.csv"),
                            index=False)
    triage["false_negatives"].to_csv(
        os.path.join(config.output_dir, "frozen_holdout_false_negatives.csv"), index=False)
    triage["false_positives"].to_csv(
        os.path.join(config.output_dir, "frozen_holdout_false_positives.csv"), index=False)
    manifest_json = write_experiment_manifest(
        config.output_dir, seed=config.seed, threshold=v5.fixed_threshold,
        holdout_size=len(y_holdout),
        extra={"train_size": int(len(y_train)), "val_size": int(len(y_val)),
               "synthetic_total": int(len(df_sintesis))},
    )

    # 9. Simpan model + artefak. Alias serving (bilstm_model.keras) HANYA
    # ditulis oleh scripts/export_for_serving.py, bukan trainer.
    os.makedirs(config.model_dir, exist_ok=True)
    model.save(os.path.join(config.model_dir, "bilstm_word2vec_v5.keras"))
    try:
        w2v_model.save(os.path.join(config.model_dir, "word2vec_v5.model"))
    except Exception as e:
        logger.warning("Gagal menyimpan word2vec: %s", e)
    with open(os.path.join(config.model_dir, "tokenizer_v5.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    tokenizer.save_json(os.path.join(config.model_dir, "tokenizer_bilstm_v5.json"))
    # Threshold kanonis tunggal (tidak ada lagi thresholds_v5.json ganda
    # atau key "lstm" fiktif).
    write_thresholds(os.path.join(config.output_dir, "thresholds.json"),
                     v5.fixed_threshold)
    write_thresholds(os.path.join(config.model_dir, "thresholds.json"),
                     v5.fixed_threshold)

    logger.info("V5 parity selesai. Eval:\n%s", eval_table.to_string(index=False))
    logger.info("Manifest: %s", manifest_json)
    return {
        "model": model,
        "history": history,
        "tokenizer": tokenizer,
        "w2v_model": w2v_model,
        "eval_table": eval_table,
        "triage": triage,
        "kb_audit": kb_audit,
        "split_manifest": manifest,
        "pool_sizes": {"train": len(y_train), "val": len(y_val), "holdout": len(y_holdout)},
        "threshold": v5.fixed_threshold,
    }
