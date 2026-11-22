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


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default=None, help="JSON daftar frozen holdout.")
@click.option("--config-json", default=None, help="JSON BertConfig (opsional).")
@click.option("--output-dir", default="./artifacts/bert/results")
@click.option("--seed", default=42, type=int)
def main(
    csv_input: str,
    frozen_holdout: str | None,
    config_json: str | None,
    output_dir: str,
    seed: int,
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
    bundle = prepare_data(data_config)

    if config_json:
        config = BertConfig.from_dict(json.loads(open(config_json, encoding="utf-8").read()))
        config.seed = seed
    else:
        config = BertConfig(seed=seed)

    output = train_final_bert(
        bundle, config, output_dir=output_dir,
        hpo_info={"mode": "cli_train_bert"}, verbose=True,
    )
    click.echo(json.dumps(output["metrics"]["validation"], indent=2))
    click.echo(json.dumps(output["metrics"]["holdout"], indent=2))
    click.echo(f"Output: {output['output_dir']}")


if __name__ == "__main__":
    main()
