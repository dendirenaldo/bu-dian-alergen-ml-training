"""Bootstrap CI + ringkasan akhir untuk run final V5 (tanpa retraining).

Merekonstruksi split val/holdout secara deterministik (kode yang sama
dengan training), memuat artefak models_final, menghitung ulang
probabilitas, lalu menulis bootstrap_ci_v5.csv.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import Config
from app.core.data.gold_merge import build_model_source_from_single
from app.core.data.leakage_split import run_v5_split
from app.core.model.v5_evaluate import bootstrap_ci_metrics, evaluate_fixed_threshold

OUT_DIR = "./output_final"
MODEL_DIR = "./models_final"
CSV_INPUT = "/Users/dendirenaldo/Downloads/Data Alergen/data.csv"
FROZEN = "./frozen_holdout_50.json"


def main() -> None:
    config = Config()
    df_raw = pd.read_csv(CSV_INPUT, delimiter=config.csv_delimiter, encoding=config.csv_encoding)
    df_model_source = build_model_source_from_single(df_raw, product_col="nama produk")
    text_src = "text"
    frozen = json.load(open(FROZEN, encoding="utf-8"))
    split = run_v5_split(
        df_model_source, product_col="nama produk", text_col=text_src,
        frozen_products=frozen, holdout_safe=config.v5.holdout_safe,
        holdout_unsafe=config.v5.holdout_unsafe, val_safe=config.v5.val_safe,
        val_unsafe=config.v5.val_unsafe,
    )
    df_val, df_hold = split["df_real_val"], split["df_holdout"]
    assert len(df_val) == 50 and len(df_hold) == 50, (len(df_val), len(df_hold))

    with open(os.path.join(MODEL_DIR, "tokenizer_v5.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_word2vec_v5.keras"))

    X_val = tokenizer.encode(df_val[text_src].fillna("").astype(str).tolist())
    X_hold = tokenizer.encode(df_hold[text_src].fillna("").astype(str).tolist())
    y_val = df_val["label_id"].astype(int).values
    y_hold = df_hold["label_id"].astype(int).values
    p_val = model.predict(X_val, verbose=0).ravel()
    p_hold = model.predict(X_hold, verbose=0).ravel()

    eval_table = evaluate_fixed_threshold(y_val, p_val, y_hold, p_hold, threshold=0.50)
    eval_table.to_csv(os.path.join(OUT_DIR, "evaluation_table_v5.csv"), index=False)
    print(eval_table.to_string(index=False))

    rows = []
    for name, yt, yp in (("validation", y_val, p_val), ("frozen_holdout", y_hold, p_hold)):
        ci = bootstrap_ci_metrics(yt, yp, threshold=0.50, n_bootstrap=2000, seed=42)
        ci.insert(0, "split", name)
        rows.append(ci)
    ci_table = pd.concat(rows, ignore_index=True)
    ci_table.to_csv(os.path.join(OUT_DIR, "bootstrap_ci_v5.csv"), index=False)
    print()
    print(ci_table.to_string(index=False))


if __name__ == "__main__":
    main()
