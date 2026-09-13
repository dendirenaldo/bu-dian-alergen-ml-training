"""CLI training script using Click."""

from __future__ import annotations

import logging
import os
import sys

import click

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config


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
@click.option("--mode", "train_mode",
              type=click.Choice(["legacy", "v5-parity", "v5_parity"], case_sensitive=False),
              default=None,
              help="Pipeline: legacy (tuning) atau v5-parity (ikuti notebook, default dari TRAIN_MODE).")
@click.option("--frozen-holdout", "frozen_holdout_path", type=click.Path(exists=True), default=None,
              help="JSON daftar frozen holdout (re-derive lalu freeze). Default: notebook23 bila n=114.")
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
    train_mode: str | None,
    frozen_holdout_path: str | None,
    bert_model_name: str | None,
    bert_epochs: int | None,
    bert_lr: float | None,
) -> None:
    """Train models (legacy tuning atau V5 parity leakage-safe)."""
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
    if train_mode is not None:
        config.mode = train_mode.lower().replace("-", "_")
    if frozen_holdout_path is not None:
        config.v5.frozen_holdout_path = frozen_holdout_path

    if config.mode == "v5_parity":
        click.echo("Starting V5 parity pipeline (ikuti notebook: fixed, leakage-safe)...")
        click.echo(f"  CSV: {config.csv_input}")
        click.echo(f"  Output dir: {config.output_dir}")
        click.echo(f"  Model dir: {config.model_dir}")
        click.echo(f"  Seed: {config.seed} | Threshold fixed: {config.v5.fixed_threshold}")
        click.echo(f"  Holdout target: {config.v5.holdout_safe}/{config.v5.holdout_unsafe} "
                   f"dari {config.v5.holdout_size} | Val: {config.v5.val_safe}/{config.v5.val_unsafe}")
        click.echo(f"  Synthetic: {config.v5.synthetic_total} ({config.v5.synthetic_source_contract})")
        try:
            from app.training.trainer_v5 import run_v5_parity

            results = run_v5_parity(config)
        except Exception as e:
            click.echo(f"V5 training failed: {e}", err=True)
            raise SystemExit(1)

        click.echo("\nV5 training complete!")
        click.echo(f"Pool: {results.get('pool_sizes', {})}")
        click.echo(f"Threshold: {results.get('threshold')}")
        click.echo(f"\nResults saved to: {config.output_dir}")
        click.echo(f"Models saved to: {config.model_dir}")
        return

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
        from app.training.trainer import run_training

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
    click.echo(f"  Mode: {config.mode}")
    click.echo(f"  V5 holdout: {config.v5.holdout_safe}/{config.v5.holdout_unsafe} "
               f"dari {config.v5.holdout_size} | val: {config.v5.val_safe}/{config.v5.val_unsafe}")
    click.echo(f"  V5 synthetic: {config.v5.synthetic_total} | threshold: {config.v5.fixed_threshold}")


if __name__ == "__main__":
    cli()
