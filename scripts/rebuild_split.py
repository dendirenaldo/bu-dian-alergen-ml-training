"""Rebuild split final yang identik untuk training IndoBERT di HPC.

Skrip ini sekarang hanya pembungkus tipis dari ``app.experiment.data_pipeline``
agar tidak ada dua implementasi split yang bisa berbeda.

Output (default ``./artifacts/bert/input/``):

- ``train_real.csv``, ``synthetic.csv``, ``train_combined.csv``
- ``val.csv``, ``holdout.csv``
- ``split_contract.json`` (komposisi + hash audit)
- ``split_manifest.csv``

Deterministik: sumber disortir berdasarkan group_key + teks, sehingga urutan
baris CSV tidak mengubah split. Verifikasi di HPC dengan membandingkan
``split_contract.json``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

from app.experiment.configs import DataConfig
from app.experiment.data_pipeline import prepare_data


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default="app/core/data/frozen_holdout.json")
@click.option("--out-dir", default="./artifacts/bert/input")
def main(csv_input: str, frozen_holdout: str, out_dir: str) -> None:
    config = DataConfig(
        csv_input=csv_input,
        frozen_holdout_path=frozen_holdout,
        bert_input_dir=out_dir,
        sort_source=True,
    )
    bundle = prepare_data(
        config,
        save_artifacts=False,
        save_bert_inputs=True,
    )
    contract = json.load(open(os.path.join(out_dir, "split_contract.json"), encoding="utf-8"))
    click.echo(
        "BERT-HPC split siap di "
        + out_dir
        + ": "
        + json.dumps(
            {
                key: contract[key]
                for key in ("train_real", "synthetic", "train_combined", "val", "holdout")
            }
        )
    )
    click.echo(
        "Frozen holdout sha256: "
        + str(contract.get("frozen_holdout_sha256", contract.get("frozen_sha256")))
    )
    click.echo("Verifikasi di HPC: bandingkan split_contract.json (sha256 harus sama).")
    click.echo(f"Pool: {bundle.pool_sizes}")


if __name__ == "__main__":
    main()
