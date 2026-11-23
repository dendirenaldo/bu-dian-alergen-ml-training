"""CLI fine-tuning IndoBERT final (baseline atau konfigurasi JSON tertentu)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

from app.experiment.bert_pipeline import train_final_bert
from app.experiment.configs import BertConfig, DataConfig
from app.experiment.data_pipeline import prepare_data
from app.experiment.gpu import is_main_process


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default=None, help="JSON daftar frozen holdout.")
@click.option("--config-json", default=None, help="JSON BertConfig (opsional).")
@click.option("--output-dir", default="./artifacts/bert/results")
@click.option("--seed", default=42, type=int)
@click.option("--precision", type=click.Choice(["fp32", "fp16", "bf16"]), default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--eval-batch-size", type=int, default=None)
@click.option("--max-len", type=int, default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--gradient-checkpointing", is_flag=True, default=False)
@click.option("--tf32/--no-tf32", default=True)
@click.option("--dataloader-num-workers", type=int, default=None)
@click.option("--gradient-accumulation-steps", type=int, default=None)
@click.option("--ddp-backend", type=click.Choice(["nccl", "gloo", "mpi", "ccl", "hccl"]), default=None)
def main(
    csv_input: str,
    frozen_holdout: str | None,
    config_json: str | None,
    output_dir: str,
    seed: int,
    precision: str | None,
    batch_size: int | None,
    eval_batch_size: int | None,
    max_len: int | None,
    epochs: int | None,
    gradient_checkpointing: bool,
    tf32: bool,
    dataloader_num_workers: int | None,
    gradient_accumulation_steps: int | None,
    ddp_backend: str | None,
) -> None:
    from app.repro import apply_reproducibility_lock, seed_all

    apply_reproducibility_lock(seed)
    seed_all(seed)

    data_config = DataConfig(
        csv_input=csv_input,
        frozen_holdout_path=frozen_holdout,
        sort_source=True,
        output_dir="./artifacts/data",
        bert_input_dir="./artifacts/bert/input",
        seed=seed,
    )
    main_process = is_main_process()
    bundle = prepare_data(
        data_config,
        save_artifacts=main_process,
        save_bert_inputs=main_process,
    )

    if config_json:
        config = BertConfig.from_dict(json.loads(open(config_json, encoding="utf-8").read()))
        config.seed = seed
    else:
        config = BertConfig(seed=seed)
    if precision is not None:
        config.precision = precision
    if batch_size is not None:
        config.batch_size = batch_size
    if eval_batch_size is not None:
        config.eval_batch_size = eval_batch_size
    if max_len is not None:
        config.max_len = max_len
    if epochs is not None:
        config.epochs = epochs
    if gradient_checkpointing:
        config.gradient_checkpointing = True
    config.tf32 = tf32
    if dataloader_num_workers is not None:
        config.dataloader_num_workers = dataloader_num_workers
    if gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = gradient_accumulation_steps
    if ddp_backend is not None:
        config.ddp_backend = ddp_backend

    output = train_final_bert(
        bundle, config, output_dir=output_dir,
        hpo_info={"mode": "cli_train_bert"}, verbose=True,
    )
    if main_process:
        click.echo(json.dumps(output["metrics"]["validation"], indent=2))
        click.echo(json.dumps(output["metrics"]["holdout"], indent=2))
        click.echo(f"Output: {output['output_dir']}")


if __name__ == "__main__":
    main()
