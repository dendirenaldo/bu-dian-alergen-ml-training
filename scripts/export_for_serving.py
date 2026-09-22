"""Export artefak model final V5 ke format serving (ml-service registry).

Registry meminta: bilstm_model.keras, word2vec.model, tokenizer.pkl,
label_encoder.pkl, thresholds.json, metadata.json.
V5 memakai mapping gold fixed {safe:0, unsafe:1} (tanpa LabelEncoder
saat training) -> ditulis di sini agar unsafe_idx serving benar.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import pandas as pd


@click.command()
@click.option("--src-dir", default="./artifacts/bilstm/models", help="Direktori artefak training.")
@click.option("--out-dir", default="../bu-dian-alergen-ml-service/models",
              help="Direktori model serving.")
@click.option("--eval-csv", default="./artifacts/bilstm/output/evaluation_table_v5.csv",
              help="Tabel evaluasi untuk metadata.")
@click.option("--frozen", default="app/core/data/frozen_holdout.json", help="Frozen holdout.")
@click.option("--threshold", type=float, default=0.5,
              help="Threshold fixed (harus sama dengan V5_FIXED_THRESHOLD training).")
def main(src_dir: str, out_dir: str, eval_csv: str, frozen: str, threshold: float) -> None:
    from app.config import Config
    from app.core.model.metrics import write_thresholds

    # Kontrak absolut: 0.5. Bukan env-driven, supaya tidak ada jalur
    # V5_FIXED_THRESHOLD=0.7 yang lolos ke file serving.
    if float(threshold) != 0.5:
        raise click.UsageError(
            f"--threshold {threshold} != kontrak fixed 0.5."
        )
    env_contract = float(Config().v5.fixed_threshold)
    if env_contract != 0.5:
        click.echo(
            f"PERINGATAN: V5_FIXED_THRESHOLD={env_contract} != 0.5; "
            "ekspor tetap memakai 0.5 (kontrak).",
            err=True,
        )
    os.makedirs(out_dir, exist_ok=True)

    renames = {
        "bilstm_word2vec_v5.keras": "bilstm_model.keras",
        "word2vec_v5.model": "word2vec.model",
        "tokenizer_v5.pkl": "tokenizer.pkl",
        # JSON adalah jalur load utama registry (pickle lintas repo rapuh).
        "tokenizer_bilstm_v5.json": "tokenizer_bilstm.json",
    }
    for src_name, dst_name in renames.items():
        src = os.path.join(src_dir, src_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Artefak tidak ada: {src}")
        shutil.copy2(src, os.path.join(out_dir, dst_name))
        click.echo(f"  {src_name} -> {dst_name}")

    # LabelEncoder fixed (konsisten dengan label_id training).
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["safe", "unsafe"])
    assert list(le.transform(["safe", "unsafe"])) == [0, 1]
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    click.echo("  label_encoder.pkl (classes=['safe','unsafe'])")

    # Melalui write_thresholds -> guard internal threshold 0.5 + skema kanonis.
    write_thresholds(os.path.join(out_dir, "thresholds.json"), threshold)
    click.echo(f"  thresholds.json (fixed {threshold})")

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    metadata: dict = {"model": "bilstm_word2vec_v5", "git_sha": git_sha,
                      "threshold_fixed": float(threshold), "pipeline": "v5"}
    if os.path.exists(eval_csv):
        ev = pd.read_csv(eval_csv)
        metadata["eval"] = ev.to_dict(orient="records")
    if os.path.exists(frozen):
        metadata["frozen_holdout"] = json.load(open(frozen, encoding="utf-8"))
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    click.echo("  metadata.json")
    click.echo(f"Export selesai -> {out_dir}")


if __name__ == "__main__":
    main()
