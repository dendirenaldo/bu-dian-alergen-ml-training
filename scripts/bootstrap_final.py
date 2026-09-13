"""Bootstrap CI + ringkasan akhir untuk run final (tanpa retraining).

Merekonstruksi split val/holdout secara deterministik (kode yang sama
dengan training), memuat artefak model, menghitung ulang probabilitas,
lalu menulis bootstrap_ci_v5.csv. Semua path dari Config/env.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf

from app.config import Config
from app.core.data.gold_merge import build_model_source_from_single, standardize_columns
from app.core.data.leakage_split import run_v5_split
from app.core.model.metrics import bootstrap_ci_metrics, evaluate_fixed_threshold


@click.command()
@click.option("--csv-input", default=None, help="Default dari CSV_INPUT.")
@click.option("--frozen-holdout", default=None, help="Default dari V5_FROZEN_HOLDOUT_PATH.")
@click.option("--output-dir", default=None, help="Default dari OUTPUT_DIR.")
@click.option("--model-dir", default=None, help="Default dari MODEL_DIR.")
def main(csv_input: str | None, frozen_holdout: str | None,
         output_dir: str | None, model_dir: str | None) -> None:
    config = Config()
    v5 = config.v5
    csv_input = csv_input or config.csv_input
    frozen_path = frozen_holdout or v5.frozen_holdout_path
    if not frozen_path:
        raise click.UsageError("Isi --frozen-holdout atau V5_FROZEN_HOLDOUT_PATH.")
    output_dir = output_dir or config.output_dir
    model_dir = model_dir or config.model_dir

    df_raw = pd.read_csv(csv_input, delimiter=config.csv_delimiter,
                         encoding=config.csv_encoding)
    df_raw = standardize_columns(df_raw, product_col=v5.product_col)
    text_src = config.text_col if config.text_col in df_raw.columns else v5.text_col
    df_model_source = build_model_source_from_single(
        df_raw, product_col=v5.product_col, text_col=text_src)
    frozen = json.load(open(frozen_path, encoding="utf-8"))
    split = run_v5_split(
        df_model_source, product_col=v5.product_col, text_col=text_src,
        frozen_products=frozen, holdout_safe=v5.holdout_safe,
        holdout_unsafe=v5.holdout_unsafe, val_safe=v5.val_safe,
        val_unsafe=v5.val_unsafe,
    )
    df_val, df_hold = split["df_real_val"], split["df_holdout"]
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
                                  n_bootstrap=2000, seed=42)
        ci.insert(0, "split", name)
        rows.append(ci)
    ci_table = pd.concat(rows, ignore_index=True)
    ci_table.to_csv(os.path.join(output_dir, "bootstrap_ci_v5.csv"), index=False)
    print()
    print(ci_table.to_string(index=False))


if __name__ == "__main__":
    main()
