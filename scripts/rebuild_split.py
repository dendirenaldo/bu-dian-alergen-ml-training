"""Rebuild split final V5 yang IDENTIK untuk training BERT di HPC.

Output (default ./artifacts/bert/input/):
- train_real.csv, synthetic.csv, train_combined.csv (pool training)
- val.csv, holdout.csv (REAL GOLD — evaluasi saja, threshold fixed 0.5)
- split_contract.json (komposisi + hash audit)

Deterministik (seed=42): file yang dihasilkan di HPC harus byte-identik
dengan yang dihasilkan lokal bila CSV sumber sama. Verifikasi dengan
membandingkan split_contract.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import pandas as pd

from app.config import Config
from app.core.data.gold_merge import build_model_source_from_single, standardize_columns
from app.core.data.leakage_split import (
    audit_final_pool_no_leakage,
    audit_split_no_leakage,
    build_split_manifest,
    run_v5_split,
)
from app.core.data.synthetic import build_final_training_pool, generate_synthetic_dataset


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default="app/core/data/frozen_holdout.json")
@click.option("--out-dir", default="./artifacts/bert/input")
def main(csv_input: str, frozen_holdout: str, out_dir: str) -> None:
    config = Config()
    v5 = config.v5
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.read_csv(csv_input, delimiter=config.csv_delimiter,
                         encoding=config.csv_encoding)
    df_raw = standardize_columns(df_raw, product_col=v5.product_col)
    text_src = config.text_col if config.text_col in df_raw.columns else v5.text_col
    src = build_model_source_from_single(df_raw, product_col=v5.product_col,
                                         text_col=text_src)
    frozen = json.load(open(frozen_holdout, encoding="utf-8"))
    split = run_v5_split(
        src, product_col=v5.product_col, text_col=text_src,
        frozen_products=frozen, holdout_safe=v5.holdout_safe,
        holdout_unsafe=v5.holdout_unsafe, val_safe=v5.val_safe,
        val_unsafe=v5.val_unsafe,
    )
    dtr, dva, dho = split["df_real_train"], split["df_real_val"], split["df_holdout"]
    audit_split_no_leakage(dtr, dva, dho)

    syn = generate_synthetic_dataset(dtr, text_col=text_src,
                                     total_data=v5.synthetic_total, seed=config.seed)
    pool = build_final_training_pool(dtr, syn, dva, dho, text_col=text_src)
    audit_final_pool_no_leakage(pool["X_train_text"], pool["X_val_text"],
                                pool["X_holdout_text"])

    def _save(df: pd.DataFrame, name: str) -> str:
        out = df.copy()
        out["label"] = out["label_id"].map({0: "safe", 1: "unsafe"})
        keep = [v5.product_col, text_src, "label", "label_id", "gold_label",
                "group_key"]
        keep = [c for c in keep if c in out.columns]
        path = os.path.join(out_dir, name)
        out[keep].to_csv(path, index=False)
        return path

    _save(dtr, "train_real.csv")
    _save(dva, "val.csv")
    _save(dho, "holdout.csv")
    syn_text_col = "text" if "text" in syn.columns else text_src
    syn_out = syn[[syn_text_col, "label", "label_id", "synthetic_rule"]].copy()
    syn_out.to_csv(os.path.join(out_dir, "synthetic.csv"), index=False)

    combined = pd.concat([
        dtr[[text_src, "label_id"]].assign(label=dtr["label_id"].map({0: "safe", 1: "unsafe"})),
        syn_out[[syn_text_col, "label", "label_id"]].rename(columns={syn_text_col: text_src}),
    ], ignore_index=True).sample(frac=1, random_state=config.seed).reset_index(drop=True)
    combined.to_csv(os.path.join(out_dir, "train_combined.csv"), index=False)

    contract = {
        "seed": config.seed,
        "train_real": len(dtr),
        "synthetic": len(syn_out),
        "train_combined": len(combined),
        "val": len(dva),
        "holdout": len(dho),
        "frozen_holdout_file": os.path.basename(frozen_holdout),
        "sha256": {},
    }
    for name in ("train_real.csv", "synthetic.csv", "train_combined.csv",
                 "val.csv", "holdout.csv"):
        h = hashlib.sha256()
        with open(os.path.join(out_dir, name), "rb") as f:
            h.update(f.read())
        contract["sha256"][name] = h.hexdigest()
    with open(os.path.join(out_dir, "split_contract.json"), "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    build_split_manifest(dtr, dva, dho, product_col="nama produk").to_csv(
        os.path.join(out_dir, "split_manifest.csv"), index=False)
    click.echo(f"BERT-HPC split siap di {out_dir}: " + json.dumps(
        {k: contract[k] for k in ("train_real", "synthetic", "val", "holdout")}))
    click.echo("Verifikasi di HPC: bandingkan split_contract.json (sha256 harus sama).")


if __name__ == "__main__":
    main()
