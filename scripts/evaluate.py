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
def compare(
    model_bilstm: str,
    model_lstm: str,
    csv_input: str,
    output_dir: str,
    seed: int,
) -> None:
    """Compare BiLSTM and LSTM models on test data."""
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    config.seed = seed

    try:
        # Load data
        df = pd.read_csv(csv_input, delimiter=";")
        df = df[[config.text_col, config.label_col]].copy()
        df[config.text_col] = df[config.text_col].fillna("").astype(str)
        df[config.label_col] = (
            df[config.label_col].fillna("").astype(str).str.strip().str.lower()
        )
        df = df[df[config.text_col].str.strip() != ""].copy()
        df = df[df[config.label_col].isin(["safe", "unsafe"])].copy()
        df = df.drop_duplicates(subset=[config.text_col]).reset_index(drop=True)

        # Cleanse
        df[config.text_col] = df[config.text_col].apply(cleanse_text)

        # Encode labels
        label_encoder = LabelEncoder()
        df["label_id"] = label_encoder.fit_transform(df[config.label_col])

        # Split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df[config.text_col].tolist(),
            df["label_id"].values,
            test_size=config.test_size,
            random_state=seed,
            stratify=df["label_id"].values,
        )

        # Tokenize
        tokenizer = Tokenizer(vocab_size=config.vocab_size, max_len=config.max_len)
        tokenizer.fit(X_train_text + X_test_text)
        X_test_pad = tokenizer.encode(X_test_text)

        # Load models
        bilstm_model = tf.keras.models.load_model(model_bilstm)
        lstm_model = tf.keras.models.load_model(model_lstm)

        # Evaluate BiLSTM
        click.echo("Evaluating BiLSTM...")
        eval_bilstm, _, _, fpr_bilstm, tpr_bilstm, auc_bilstm = evaluate_model(
            bilstm_model, "BiLSTM", X_test_pad, y_test, X_test_text, label_encoder, output_dir
        )

        # Evaluate LSTM
        click.echo("Evaluating LSTM...")
        eval_lstm, _, _, fpr_lstm, tpr_lstm, auc_lstm = evaluate_model(
            lstm_model, "LSTM", X_test_pad, y_test, X_test_text, label_encoder, output_dir
        )

        # ROC comparison
        plot_roc_comparison(
            fpr_bilstm, tpr_bilstm, auc_bilstm,
            fpr_lstm, tpr_lstm, auc_lstm,
            output_dir,
        )

        # Comparison table
        comparison = pd.DataFrame(
            [
                {
                    "model": "BiLSTM",
                    **{row["metric"]: row["value"] for _, row in eval_bilstm.iterrows()},
                },
                {
                    "model": "LSTM",
                    **{row["metric"]: row["value"] for _, row in eval_lstm.iterrows()},
                },
            ]
        )
        comparison.to_csv(os.path.join(output_dir, "comparison_bilstm_vs_lstm.csv"), index=False)

        click.echo("\n=== Model Comparison ===")
        click.echo(comparison.to_string(index=False))
        click.echo(f"\nResults saved to: {output_dir}")
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
