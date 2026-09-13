"""CLI training script using Click."""

from __future__ import annotations

import logging
import os
import sys

import click

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.training.trainer import run_training


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """Bu Dian Allergen Detection - ML Training CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@cli.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Path to .env config file.")
@click.option("--data-source", type=click.Choice(["CSV", "OCR"], case_sensitive=False), help="Data source mode.")
@click.option("--csv-input", type=click.Path(), help="Path to CSV input file.")
@click.option("--output-dir", type=click.Path(), help="Output directory.")
@click.option("--model-dir", type=click.Path(), help="Model save directory.")
@click.option("--epochs", type=int, help="Number of training epochs.")
@click.option("--batch-size", type=int, help="Batch size.")
@click.option("--seed", type=int, help="Random seed.")
@click.option("--model", "model_type",
              type=click.Choice(["bilstm", "lstm", "bert", "all"], case_sensitive=False),
              help="Model yang dilatih (default: all).")
@click.option("--bert-model", "bert_model_name", type=str, default=None,
              help="Nama checkpoint HF BERT (mis. indobenchmark/indobert-base-p1). "
                   "Pluggable: ganti tanpa ubah kode untuk coba varian lain.")
@click.option("--bert-epochs", type=int, default=None, help="Epoch training BERT.")
@click.option("--bert-lr", type=float, default=None, help="Learning rate BERT.")
def train(
    config_file: str | None,
    data_source: str | None,
    csv_input: str | None,
    output_dir: str | None,
    model_dir: str | None,
    epochs: int | None,
    batch_size: int | None,
    seed: int | None,
    model_type: str | None,
    bert_model_name: str | None,
    bert_epochs: int | None,
    bert_lr: float | None,
) -> None:
    """Train BiLSTM and LSTM models with hyperparameter tuning."""
    if config_file:
        from dotenv import load_dotenv

        load_dotenv(config_file, override=True)

    config = Config()

    if data_source is not None:
        config.data_source_mode = data_source.upper()
    if csv_input is not None:
        config.csv_input = csv_input
    if output_dir is not None:
        config.output_dir = output_dir
    if model_dir is not None:
        config.model_dir = model_dir
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if seed is not None:
        config.seed = seed
    if model_type is not None:
        config.model_type = model_type.lower()
    if bert_model_name is not None:
        config.bert.model_name = bert_model_name
    if bert_epochs is not None:
        config.bert.epochs = bert_epochs
    if bert_lr is not None:
        config.bert.learning_rate = bert_lr

    click.echo("Starting training pipeline...")
    click.echo(f"  Data source: {config.data_source_mode}")
    click.echo(f"  Output dir: {config.output_dir}")
    click.echo(f"  Model dir: {config.model_dir}")
    click.echo(f"  Epochs: {config.epochs}")
    click.echo(f"  Seed: {config.seed}")
    click.echo(f"  Model: {config.model_type}")
    if config.model_type in ("bert", "all"):
        click.echo(f"  BERT: {config.bert.model_name} "
                   f"(max_len={config.bert.max_len}, epochs={config.bert.epochs}, "
                   f"lr={config.bert.learning_rate})")

    try:
        results = run_training(config)
    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        raise SystemExit(1)

    click.echo("\nTraining complete!")
    if results.get("best_params_bilstm") is not None:
        click.echo(f"BiLSTM best params: {results['best_params_bilstm']}")
    if results.get("best_params_lstm") is not None:
        click.echo(f"LSTM best params: {results['best_params_lstm']}")
    if results.get("bert") is not None:
        click.echo(f"BERT results: {results['bert'].get('metrics', {})}")
    click.echo(f"Thresholds: {results.get('thresholds', {})}")
    click.echo(f"\nResults saved to: {config.output_dir}")
    click.echo(f"Models saved to: {config.model_dir}")


@cli.command()
def info() -> None:
    """Show current configuration."""
    config = Config()
    click.echo("=== Configuration ===")
    click.echo(f"  Dataset dir: {config.dataset_dir}")
    click.echo(f"  Output dir: {config.output_dir}")
    click.echo(f"  Model dir: {config.model_dir}")
    click.echo(f"  CSV input: {config.csv_input}")
    click.echo(f"  Data source: {config.data_source_mode}")
    click.echo(f"  Seed: {config.seed}")
    click.echo(f"  Epochs: {config.epochs}")
    click.echo(f"  Batch size: {config.batch_size}")
    click.echo(f"  Vocab size: {config.vocab_size}")
    click.echo(f"  Embed dim: {config.embed_dim}")
    click.echo(f"  Max len: {config.max_len}")
    click.echo(f"  Model type: {config.model_type}")
    click.echo(f"  BERT model: {config.bert.model_name}")
    click.echo(f"  BERT max len: {config.bert.max_len}")
    click.echo(f"  Num tuning trials: {config.num_trials}")


if __name__ == "__main__":
    cli()
