"""Regenerasi artefak audit yang dirujuk paper (sekali-per-revisi-data).

Menghasilkan:
- artifacts/bilstm/output/kb_audit_499.csv  (KB vs Gold, n=499)
- artifacts/bilstm/output/token_stats_train_real.json (statistik token
  pada 399 teks latih riil — sumber angka Bab 4.2)
- artifacts/bilstm/output/bootstrap_ci_bert.csv (CI bootstrap BERT,
  seed=42, 2000, stratified — sama seperti bootstrap_ci_v5)

Tanpa TF. Jalankan:
    python paper/regen_paper_artifacts.py --csv-input <path-ke-data.csv>
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from app.config import Config
from app.core.data.gold_merge import build_model_source_from_single, standardize_columns
from app.core.model.audit import kb_vs_gold_audit
from app.core.model.metrics import bootstrap_ci_metrics
from app.core.preprocessing.text import simple_tokenize

OUT = os.path.join("artifacts", "bilstm", "output")
TRAIN_REAL = os.path.join("artifacts", "bert", "input", "train_real.csv")


def token_stats(texts: list[str]) -> dict:
    toks = [simple_tokenize(t) for t in texts]
    lens = np.array([len(t) for t in toks])
    counter: Counter = Counter()
    for t in toks:
        counter.update(t)
    hapax = sum(1 for v in counter.values() if v == 1)
    hapax_tok = sum(v for v in counter.values() if v == 1)
    digit_types = sum(1 for t in counter if t.isdigit())
    return {
        "n_texts": len(texts),
        "mean_tokens": round(float(lens.mean()), 1),
        "median_tokens": int(np.median(lens)),
        "p95_tokens": int(np.percentile(lens, 95)),
        "padding_rate_at_maxlen_120_pct": round(float(1 - lens.mean() / 120) * 100, 1),
        "hapax_vocab_pct": round(hapax / len(counter) * 100, 1),
        "hapax_token_occurrence_pct": round(hapax_tok / lens.sum() * 100, 1),
        "digit_token_types": digit_types,
    }


def main() -> None:
    import click

    @click.command()
    @click.option("--csv-input", required=True, help="CSV sumber 499 (delimiter ';').")
    def _main(csv_input: str) -> None:
        _run(csv_input)

    _main()


def _run(csv_input: str) -> None:
    config = Config()
    os.makedirs(OUT, exist_ok=True)

    # 1. KB vs Gold (n=499) dari CSV kanonis.
    df_raw = pd.read_csv(csv_input, delimiter=config.csv_delimiter,
                         encoding=config.csv_encoding)
    df_raw = standardize_columns(df_raw, product_col=config.v5.product_col)
    src = build_model_source_from_single(
        df_raw, product_col=config.v5.product_col, text_col=config.text_col,
        label_col=config.label_col,
    )
    kb = kb_vs_gold_audit(src)
    kb["summary"].to_csv(os.path.join(OUT, "kb_audit_499.csv"), index=False)
    kb["confusion_matrix"].to_csv(
        os.path.join(OUT, "kb_audit_499.csv"), mode="a", index=True)
    print("KB vs Gold (499):")
    print(kb["summary"].to_string(index=False))

    # 2. Statistik token pada teks latih riil (399).
    train_real = pd.read_csv(TRAIN_REAL)
    stats = token_stats(train_real["text"].fillna("").astype(str).tolist())
    stats["source"] = TRAIN_REAL
    with open(os.path.join(OUT, "token_stats_train_real.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Token stats (399 real-train):", json.dumps(stats, indent=2))

    # 3. CI bootstrap BERT (seed=42 -> identik dengan angka di naskah).
    rows = []
    for name in ("val", "holdout"):
        d = pd.read_csv(f"artifacts/bert/results/probs_{name}.csv")
        yt = (d["label"].astype(str).str.lower() == "unsafe").astype(int).values
        ci = bootstrap_ci_metrics(yt, d["prob_unsafe"].astype(float).values,
                                  threshold=0.50, n_bootstrap=2000, seed=42)
        ci.insert(0, "split", name)
        rows.append(ci)
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(OUT, "bootstrap_ci_bert.csv"), index=False)
    print("CI BERT tersimpan ->", os.path.join(OUT, "bootstrap_ci_bert.csv"))


if __name__ == "__main__":
    main()
