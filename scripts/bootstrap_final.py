"""Bootstrap CI + ringkasan akhir untuk run final (tanpa retraining).

Merekonstruksi split val/holdout secara deterministik (kode yang sama
dengan training), memuat artefak model, menghitung ulang probabilitas,
lalu menulis bootstrap_ci_v5.csv. Semua path dari Config/env.
"""

from __future__ import annotations

import os
import pickle
import sys

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.config import Config
from app.core.model.metrics import bootstrap_ci_metrics, evaluate_fixed_threshold
from app.experiment.configs import DataConfig
from app.experiment.data_pipeline import prepare_data


@click.command()
@click.option("--csv-input", default=None, help="Default dari CSV_INPUT.")
@click.option("--frozen-holdout", default=None, help="Default dari V5_FROZEN_HOLDOUT_PATH.")
@click.option("--output-dir", default=None, help="Default dari OUTPUT_DIR.")
@click.option("--model-dir", default=None, help="Default dari MODEL_DIR.")
def main(csv_input: str | None, frozen_holdout: str | None,
         output_dir: str | None, model_dir: str | None) -> None:
    config = Config()
    v5 = config.v5
    from app.repro import apply_reproducibility_lock, seed_all

    apply_reproducibility_lock(config.seed)
    seed_all(config.seed)
    import tensorflow as tf  # noqa: PLC0415 — sengaja setelah lock

    csv_input = csv_input or config.csv_input
    frozen_path = frozen_holdout or v5.frozen_holdout_path
    if not frozen_path:
        raise click.UsageError("Isi --frozen-holdout atau V5_FROZEN_HOLDOUT_PATH.")
    output_dir = output_dir or config.output_dir
    model_dir = model_dir or config.model_dir

    data_config = DataConfig(
        csv_input=csv_input,
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
        frozen_holdout_path=frozen_path,
        sort_source=False,  # legacy: pertahankan split artefak lama
    )
    bundle = prepare_data(data_config, save_artifacts=False, save_bert_inputs=False)
    df_val, df_hold = bundle.df_val, bundle.df_holdout
    text_src = data_config.text_col
    assert len(df_val) == v5.val_safe + v5.val_unsafe
    assert len(df_hold) == v5.holdout_size

    with open(os.path.join(model_dir, "tokenizer_v5.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(os.path.join(model_dir, "bilstm_word2vec_v5.keras"))

    X_val = tokenizer.encode(df_val[text_src].fillna("").astype(str).tolist())
    X_hold = tokenizer.encode(df_hold[text_src].fillna("").astype(str).tolist())
    y_val = df_val["label_id"].astype(int).values
    y_hold = df_hold["label_id"].astype(int).values
    p_val = model.predict(X_val, verbose=0).ravel()
    p_hold = model.predict(X_hold, verbose=0).ravel()

    eval_table = evaluate_fixed_threshold(y_val, p_val, y_hold, p_hold,
                                          threshold=v5.fixed_threshold)
    eval_table.to_csv(os.path.join(output_dir, "evaluation_table_v5.csv"), index=False)
    print(eval_table.to_string(index=False))

    rows = []
    for name, yt, yp in (("validation", y_val, p_val), ("frozen_holdout", y_hold, p_hold)):
        ci = bootstrap_ci_metrics(yt, yp, threshold=v5.fixed_threshold,
                                  n_bootstrap=2000, seed=config.seed)
        ci.insert(0, "split", name)
        rows.append(ci)
    ci_table = pd.concat(rows, ignore_index=True)
    ci_table.to_csv(os.path.join(output_dir, "bootstrap_ci_v5.csv"), index=False)
    print()
    print(ci_table.to_string(index=False))


if __name__ == "__main__":
    main()
