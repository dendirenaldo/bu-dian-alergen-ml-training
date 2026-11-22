"""Kompatibilitas CLI lama: jalankan pipeline final V5 lewat modul eksperimen baru.

Modul ini sengaja dibuat tipis. Seluruh logika data, training, evaluasi, dan
penyimpanan artefak sekarang ada di ``app.experiment`` agar tidak ada dua
implementasi yang bisa berbeda.

Kontrak lama tetap dijaga:

- Split leakage-safe + frozen holdout.
- Threshold final 0,5.
- Nama artefak ``bilstm_word2vec_v5.keras``, ``word2vec_v5.model``,
  ``tokenizer_v5.pkl``, ``evaluation_table_v5.csv``, dst.
- Audit konfigurasi final (D1/D2/D3) tetap dijalankan untuk run legacy.
"""

from __future__ import annotations

import logging

from app.config import Config

logger = logging.getLogger(__name__)


def run_v5_parity(config: Config | None = None) -> dict:
    """Jalankan pipeline final V5 (legacy CLI) memakai modul eksperimen baru.

    Args:
        config: konfigurasi legacy. Default ``Config()``.

    Returns:
        Ringkasan hasil untuk CLI.
    """
    if config is None:
        config = Config()
    v5 = config.v5
    if float(v5.fixed_threshold) != 0.5:
        raise ValueError("V5 threshold harus fixed 0.50.")

    from app.repro import apply_reproducibility_lock, seed_all

    apply_reproducibility_lock(config.seed)
    seed_all(config.seed)

    from app.core.model.audit import (
        audit_final_config_v5,
        audit_pipeline_consistency_v5,
        audit_reproducibility_v5,
    )
    from app.experiment.bilstm_pipeline import train_final_bilstm
    from app.experiment.configs import BiLSTMConfig, DataConfig
    from app.experiment.data_pipeline import prepare_data

    data_config = DataConfig(
        csv_input=config.csv_input,
        csv_delimiter=config.csv_delimiter,
        csv_encoding=config.csv_encoding,
        product_col=v5.product_col,
        text_col=config.text_col if config.text_col else v5.text_col,
        label_col=config.label_col,
        seed=config.seed,
        holdout_safe=v5.holdout_safe,
        holdout_unsafe=v5.holdout_unsafe,
        val_safe=v5.val_safe,
        val_unsafe=v5.val_unsafe,
        synthetic_total=v5.synthetic_total,
        frozen_holdout_path=v5.frozen_holdout_path,
        output_dir=config.output_dir,
        bert_input_dir="./artifacts/bert/input",
    )
    bundle = prepare_data(data_config, save_bert_inputs=True)

    # Audit kontrak final legacy (D1/D2/D3) — sengaja tetap ketat untuk CLI ini.
    final_config = audit_final_config_v5(v5)
    audit_pipeline_consistency_v5(
        bundle.df_train, bundle.df_val, bundle.df_holdout,
        len(bundle.df_synthetic), v5.synthetic_total,
        False, v5.fixed_threshold,
    )
    audit_reproducibility_v5(
        config.seed,
        w2v_workers=1,
        w2v_seed=config.seed,
        train_shuffle=v5.train_shuffle,
        fixed_threshold=v5.fixed_threshold,
        holdout_size=len(bundle.y_holdout),
        expected_holdout_size=v5.holdout_size,
    )

    bilstm_config = BiLSTMConfig(
        vocab_size=v5.vocab_size,
        max_len=v5.max_len,
        embed_dim=v5.embed_dim,
        w2v_window=v5.w2v_window,
        w2v_min_count=v5.w2v_min_count,
        w2v_epochs=v5.w2v_epochs,
        w2v_sg=1,
        w2v_workers=1,
        embedding_init_scale=v5.embedding_init_scale,
        embed_trainable=v5.embed_trainable,
        mask_zero=v5.mask_zero,
        digit_fold=v5.digit_fold,
        rnn_type="lstm",
        use_bidirectional=True,
        num_rnn_layers=2,
        lstm_units_1=v5.lstm_units_1,
        lstm_units_2=v5.lstm_units_2,
        dropout_1=v5.dropout_1,
        dropout_2=v5.dropout_2,
        recurrent_dropout=0.0,
        dense_units=v5.dense_units,
        dropout_dense=v5.dropout_dense,
        use_maxnorm=False,
        optimizer="adam",
        learning_rate=v5.learning_rate,
        weight_decay=0.0,
        batch_size=v5.batch_size,
        epochs=v5.epochs,
        gradient_clip_norm=v5.gradient_clip_norm,
        early_stopping_patience=v5.early_stopping_patience,
        lr_reduce_patience=v5.lr_reduce_patience,
        lr_reduce_factor=v5.lr_reduce_factor,
        min_lr=v5.min_lr,
        use_class_weight=False,
        label_smoothing=0.0,
        shuffle=v5.train_shuffle,
        preshuffle=v5.preshuffle,
        monitor="val_loss",
        monitor_mode="min",
        seed=config.seed,
    )

    result = train_final_bilstm(
        bundle,
        bilstm_config,
        config.output_dir,
        config.model_dir,
        hpo_info={"mode": "legacy_v5_cli", "final_config": final_config},
        verbose=1,
    )
    return {
        "metrics": result["metrics"],
        "eval_table": result["eval_table"],
        "pool_sizes": bundle.pool_sizes,
        "threshold": 0.5,
        "output_dir": result["output_dir"],
        "model_dir": result["model_dir"],
    }
