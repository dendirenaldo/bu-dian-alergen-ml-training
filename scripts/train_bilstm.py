"""CLI training BiLSTM final (baseline atau konfigurasi JSON tertentu)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

from app.experiment.bilstm_pipeline import train_final_bilstm
from app.experiment.configs import BiLSTMConfig, DataConfig
from app.experiment.data_pipeline import prepare_data
from app.experiment.gpu import distributed_world_size


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default=None, help="JSON daftar frozen holdout.")
@click.option("--config-json", default=None, help="JSON BiLSTMConfig (opsional).")
@click.option("--output-dir", default="./artifacts/bilstm/output")
@click.option("--model-dir", default="./artifacts/bilstm/models")
@click.option("--seed", default=42, type=int)
@click.option("--mixed-precision", type=click.Choice(["float32", "mixed_float16", "mixed_bfloat16"]), default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--max-len", type=int, default=None)
@click.option("--steps-per-execution", type=int, default=None)
@click.option("--multi-gpu/--single-gpu", default=True)
@click.option("--cross-device-ops", type=click.Choice(["auto", "nccl", "hierarchical", "reduction_to_one"]), default=None)
@click.option("--tf-data/--no-tf-data", default=True)
def main(
    csv_input: str,
    frozen_holdout: str | None,
    config_json: str | None,
    output_dir: str,
    model_dir: str,
    seed: int,
    mixed_precision: str | None,
    batch_size: int | None,
    epochs: int | None,
    max_len: int | None,
    steps_per_execution: int | None,
    multi_gpu: bool,
    cross_device_ops: str | None,
    tf_data: bool,
) -> None:
    if distributed_world_size() > 1:
        raise click.UsageError(
            "Training BiLSTM multi-GPU memakai MirroredStrategy dalam SATU proses. "
            "Jangan jalankan lewat `accelerate launch`/`torchrun`."
        )
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
    bundle = prepare_data(data_config)

    if config_json:
        config = BiLSTMConfig.from_dict(json.loads(open(config_json, encoding="utf-8").read()))
        config.seed = seed
    else:
        config = BiLSTMConfig(seed=seed)
    if mixed_precision is not None:
        config.mixed_precision = mixed_precision
    if batch_size is not None:
        config.batch_size = batch_size
    if epochs is not None:
        config.epochs = epochs
    if max_len is not None:
        config.max_len = max_len
    if steps_per_execution is not None:
        config.steps_per_execution = steps_per_execution
    config.multi_gpu = multi_gpu
    if cross_device_ops is not None:
        config.cross_device_ops = cross_device_ops
    config.use_tf_data = tf_data

    output = train_final_bilstm(
        bundle, config, output_dir=output_dir, model_dir=model_dir,
        hpo_info={"mode": "cli_train_bilstm"}, verbose=1,
    )
    click.echo(output["eval_table"].to_string(index=False))
    click.echo(f"Model: {output['model_dir']}")
    click.echo(f"Output: {output['output_dir']}")


if __name__ == "__main__":
    main()
