"""Ekspor tabel search space untuk lampiran paper.

Contoh:
    python scripts/export_search_space.py --model bilstm --preset full \
        --out artifacts/hpo/search_space_bilstm.csv
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import pandas as pd

from app.experiment.search_spaces import (
    bert_search_space,
    bilstm_search_space,
    space_summary,
)


@click.command()
@click.option("--model", "model_name", type=click.Choice(["bilstm", "bert"]), default="bilstm")
@click.option("--preset", type=click.Choice(["smoke", "standard", "full"]), default="full")
@click.option("--out", default=None, help="Path CSV/Markdown output.")
def main(model_name: str, preset: str, out: str | None) -> None:
    space = bilstm_search_space(preset) if model_name == "bilstm" else bert_search_space(preset)
    table = pd.DataFrame(space_summary(space))
    print(table.to_string(index=False))
    print(f"\nJumlah parameter: {len(table)}")
    if space.size() is not None:
        print(f"Jumlah kombinasi grid: {space.size()}")
    else:
        print("Jumlah kombinasi grid: tak hingga (ada parameter kontinu/log).")

    if out:
        import pathlib

        path = pathlib.Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".md", ".markdown"):
            headers = list(table.columns)
            lines = [
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|",
            ]
            for _, row in table.iterrows():
                lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            table.to_csv(path, index=False)
        print(f"Tabel disimpan: {path}")


if __name__ == "__main__":
    main()
