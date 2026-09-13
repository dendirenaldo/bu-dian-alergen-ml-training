"""CLI evaluation script."""

import logging
import os
import sys

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.core.model.evaluator import evaluate_model, plot_roc_comparison
from app.core.model.tokenizer import Tokenizer
from app.core.preprocessing.text import cleanse_text


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """Bu Dian Allergen Detection - ML Evaluation CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@cli.command()
@click.option("--model-bilstm", type=click.Path(exists=True), required=True, help="Path to BiLSTM model file.")
@click.option("--model-lstm", type=click.Path(exists=True), required=True, help="Path to LSTM model file.")
@click.option("--csv-input", type=click.Path(exists=True), required=True, help="Path to test CSV file.")
@click.option("--output-dir", type=click.Path(), default="./output", help="Output directory.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--model-dir", type=click.Path(exists=True), default=None,
              help="Direktori artefak training (tokenizer.pkl, label_encoder.pkl, "
                   "splits.json, thresholds.json). WAJIB agar evaluasi memakai "
                   "artefak training, bukan fit ulang (fix leakage CRITICAL).")
@click.option("--model-bert", type=click.Path(exists=True), default=None,
              help="Path ke model BERT (direktori HF) untuk dibandingkan juga.")
def compare(
    model_bilstm: str,
    model_lstm: str,
    csv_input: str,
    output_dir: str,
    seed: int,
    model_dir: str | None,
    model_bert: str | None,
) -> None:
    """Compare BiLSTM and LSTM models on test data (artefak training)."""
    import json
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    config.seed = seed

    try:
        # Load data (validasi sama seperti trainer)
        df = pd.read_csv(csv_input, delimiter=config.csv_delimiter,
                         encoding=config.csv_encoding)
        df = df[[config.text_col, config.label_col]].copy()
        df[config.text_col] = df[config.text_col].fillna("").astype(str)
        df[config.label_col] = (
            df[config.label_col].fillna("").astype(str).str.strip().str.lower()
        )
        df = df[df[config.text_col].str.strip() != ""].copy()
        df = df[df[config.label_col].isin(["safe", "unsafe"])].copy()
        df = df.drop_duplicates(subset=[config.text_col]).reset_index(drop=True)

        # Cleanse (sama seperti trainer)
        df[config.text_col] = df[config.text_col].apply(cleanse_text)

        # FIX CRITICAL: load artefak training bila tersedia, jangan fit ulang.
        tokenizer = None
        label_encoder = None
        X_test_text: list[str] | None = None
        y_test = None
        thresholds: dict = {}
        if model_dir is not None:
            tok_path = os.path.join(model_dir, "tokenizer.pkl")
            le_path = os.path.join(model_dir, "label_encoder.pkl")
            splits_path = os.path.join(output_dir, "splits.json")
            if not os.path.exists(splits_path):
                splits_path = os.path.join(model_dir, "splits.json")
            thr_path = os.path.join(model_dir, "thresholds.json")
            if os.path.exists(tok_path):
                with open(tok_path, "rb") as f:
                    tokenizer = pickle.load(f)
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    label_encoder = pickle.load(f)
            if os.path.exists(thr_path):
                with open(thr_path, encoding="utf-8") as f:
                    thresholds = json.load(f)
            if tokenizer is not None and label_encoder is not None and os.path.exists(splits_path):
                with open(splits_path, encoding="utf-8") as f:
                    splits = json.load(f)
                # Terapkan label map tersimpan ke df saat ini, lalu ambil test_idx
                # yang tersimpan (fallback ke split ulang bila ukuran beda).
                try:
                    df["label_id"] = label_encoder.transform(df[config.label_col])
                    test_idx = [i for i in splits["test_idx"] if 0 <= i < len(df)]
                    if len(test_idx) >= 2:
                        X_test_text = df[config.text_col].iloc[test_idx].tolist()
                        y_test = df["label_id"].values[test_idx]
                        click.echo(f"Memakai splits.json tersimpan (test={len(test_idx)}).")
                except Exception as e:
                    click.echo(f"splits.json tidak cocok, fallback split ulang: {e}")

        if X_test_text is None:
            # Fallback legacy (dengan peringatan): split ulang + fit HANYA di train.
            # (Dulu: fit(train+test) = leakage + mismatch vocab.)
            click.echo("PERINGATAN: --model-dir tidak diberikan/lengkap; "
                       "fallback split ulang. Berikan --model-dir agar valid.")
            label_encoder = LabelEncoder()
            df["label_id"] = label_encoder.fit_transform(df[config.label_col])
            X_train_text, X_test_text, _, y_test = train_test_split(
                df[config.text_col].tolist(),
                df["label_id"].values,
                test_size=config.test_size,
                random_state=seed,
                stratify=df["label_id"].values,
            )
            tokenizer = Tokenizer(vocab_size=config.vocab_size, max_len=config.max_len)
            tokenizer.fit(X_train_text)  # HANYA train, bukan train+test.

        assert tokenizer is not None and y_test is not None and X_test_text is not None
        X_test_pad = tokenizer.encode(X_test_text)
        thr_bilstm = float(thresholds.get("bilstm", 0.5))
        thr_lstm = float(thresholds.get("lstm", 0.5))

        # Load models
        bilstm_model = tf.keras.models.load_model(model_bilstm)
        lstm_model = tf.keras.models.load_model(model_lstm)

        # Evaluate BiLSTM
        click.echo(f"Evaluating BiLSTM (threshold={thr_bilstm})...")
        eval_bilstm, _, _, fpr_bilstm, tpr_bilstm, auc_bilstm = evaluate_model(
            bilstm_model, "BiLSTM", X_test_pad, y_test, X_test_text, label_encoder, output_dir,
            threshold=thr_bilstm,
        )

        # Evaluate LSTM
        click.echo(f"Evaluating LSTM (threshold={thr_lstm})...")
        eval_lstm, _, _, fpr_lstm, tpr_lstm, auc_lstm = evaluate_model(
            lstm_model, "LSTM", X_test_pad, y_test, X_test_text, label_encoder, output_dir,
            threshold=thr_lstm,
        )

        extra: dict = {}
        bert_row = None
        if model_bert is not None:
            try:
                from app.core.model.bert.evaluate_bert import evaluate_bert_dir
                click.echo("Evaluating BERT...")
                bert_metrics, b_fpr, b_tpr, b_auc = evaluate_bert_dir(
                    model_bert, X_test_text, y_test, label_encoder, output_dir,
                )
                extra["BERT"] = (b_fpr, b_tpr, b_auc)
                bert_row = {"model": "BERT", **bert_metrics}
            except ImportError as e:
                click.echo(f"BERT dilewati (dependensi belum ada: {e})")

        # ROC comparison
        plot_roc_comparison(
            fpr_bilstm, tpr_bilstm, auc_bilstm,
            fpr_lstm, tpr_lstm, auc_lstm,
            output_dir,
            extra_curves=extra or None,
            filename="roc_comparison_word2vec.png",
            title="ROC Curve Comparison",
        )

        # Comparison table
        rows_cmp = [
            {
                "model": "BiLSTM",
                **{row["metric"]: row["value"] for _, row in eval_bilstm.iterrows()},
            },
            {
                "model": "LSTM",
                **{row["metric"]: row["value"] for _, row in eval_lstm.iterrows()},
            },
        ]
        if bert_row is not None:
            rows_cmp.append(bert_row)
        comparison = pd.DataFrame(rows_cmp)
        comparison.to_csv(os.path.join(output_dir, "comparison_bilstm_vs_lstm.csv"), index=False)

        click.echo("\n=== Model Comparison ===")
        click.echo(comparison.to_string(index=False))
        click.echo(f"\nResults saved to: {output_dir}")
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
