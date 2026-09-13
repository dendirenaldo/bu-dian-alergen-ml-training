"""CLI training: BiLSTM leakage-safe (pipeline tunggal)."""

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
@click.option("--csv-input", type=click.Path(), help="Path to CSV input file.")
@click.option("--output-dir", type=click.Path(), help="Output directory.")
@click.option("--model-dir", type=click.Path(), help="Model save directory.")
@click.option("--seed", type=int, help="Random seed.")
@click.option("--frozen-holdout", "frozen_holdout_path", type=click.Path(exists=True), default=None,
              help="JSON daftar frozen holdout. Default dari V5_FROZEN_HOLDOUT_PATH "
                   "atau notebook23 bila n=114.")
def train(
    config_file: str | None,
    csv_input: str | None,
    output_dir: str | None,
    model_dir: str | None,
    seed: int | None,
    frozen_holdout_path: str | None,
) -> None:
    """Train BiLSTM leakage-safe (fixed hyperparams, threshold 0.5).

    Penyetelan lanjutan via env V5_* (lihat .env.example).
    """
    if config_file:
        from dotenv import load_dotenv

        load_dotenv(config_file, override=True)

    config = Config()

    if csv_input is not None:
        config.csv_input = csv_input
    if output_dir is not None:
        config.output_dir = output_dir
    if model_dir is not None:
        config.model_dir = model_dir
    if seed is not None:
        config.seed = seed
    if frozen_holdout_path is not None:
        config.v5.frozen_holdout_path = frozen_holdout_path

    v5 = config.v5
    click.echo("Starting BiLSTM training (leakage-safe, fixed hyperparams)...")
    click.echo(f"  CSV: {config.csv_input}")
    click.echo(f"  Output dir: {config.output_dir}")
    click.echo(f"  Model dir: {config.model_dir}")
    click.echo(f"  Seed: {config.seed} | Threshold fixed: {v5.fixed_threshold}")
    click.echo(f"  Holdout: {v5.holdout_safe}/{v5.holdout_unsafe} dari {v5.holdout_size} | "
               f"Val: {v5.val_safe}/{v5.val_unsafe}")
    click.echo(f"  Synthetic: {v5.synthetic_total} ({v5.synthetic_source_contract})")
    click.echo(f"  LR: {v5.learning_rate} | Epochs: {v5.epochs} | Batch: {v5.batch_size} | "
               f"Mask: {v5.mask_zero} | MinCount: {v5.w2v_min_count}")

    try:
        from app.training.trainer_v5 import run_v5_parity

        results = run_v5_parity(config)
    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        raise SystemExit(1)

    click.echo("\nTraining complete!")
    click.echo(f"Pool: {results.get('pool_sizes', {})}")
    click.echo(f"Threshold: {results.get('threshold')}")
    click.echo(f"\nResults saved to: {config.output_dir}")
    click.echo(f"Models saved to: {config.model_dir}")


@cli.command()
def info() -> None:
    """Show current configuration."""
    config = Config()
    v5 = config.v5
    click.echo("=== Configuration ===")
    click.echo(f"  Output dir: {config.output_dir}")
    click.echo(f"  Model dir: {config.model_dir}")
    click.echo(f"  CSV input: {config.csv_input} (delimiter={config.csv_delimiter})")
    click.echo(f"  Seed: {config.seed}")
    click.echo(f"  Architecture: BiLSTM (Word2Vec, mask={v5.mask_zero})")
    click.echo(f"  LR: {v5.learning_rate} | Epochs: {v5.epochs} | Batch: {v5.batch_size}")
    click.echo(f"  Holdout: {v5.holdout_safe}/{v5.holdout_unsafe} dari {v5.holdout_size} | "
               f"Val: {v5.val_safe}/{v5.val_unsafe}")
    click.echo(f"  Synthetic: {v5.synthetic_total} | Threshold fixed: {v5.fixed_threshold}")
    click.echo(f"  Frozen holdout: {v5.frozen_holdout_path or '(default)'}")
    click.echo(f"  BERT: via HPC (artifacts/bert/), bukan CLI ini")


if __name__ == "__main__":
    cli()
