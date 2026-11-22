"""Pipeline data end-to-end: baca CSV → gold → split → synthetic → audit.

Semua aturan leakage-safe dari pipeline lama dipertahankan:

- Gold label adalah ground truth.
- KB hanya diagnostik.
- Frozen holdout dikunci per nama produk.
- Synthetic hanya dari real-train dan divalidasi ulang oleh KB.
- Tokenizer/Word2Vec nanti hanya fit di final-train.

Modul ini tidak mengimpor TensorFlow/Gensim sehingga bisa dipakai di
notebook data dan test ringan.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.data.frozen import resolve_frozen_products
from app.core.data.gold_merge import build_model_source_from_single, standardize_columns
from app.core.data.leakage_split import (
    audit_final_pool_no_leakage,
    audit_split_no_leakage,
    build_split_manifest,
    run_v5_split,
)
from app.core.data.synthetic import build_final_training_pool, generate_synthetic_dataset
from app.core.model.audit import kb_vs_gold_audit
from app.core.preprocessing.text import simple_tokenize
from app.experiment.artifacts import write_json
from app.experiment.configs import DataConfig


@dataclass
class DataBundle:
    """Semua data yang dibutuhkan model, sudah diaudit dan siap dipakai.

    ``X_*_text`` dan ``y_*`` adalah bentuk siap-training. ``df_*`` disimpan
    agar notebook bisa menampilkan contoh dan melakukan error analysis.
    """

    config: DataConfig
    df_source: pd.DataFrame
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_holdout: pd.DataFrame
    df_synthetic: pd.DataFrame
    X_train_text: list[str]
    y_train: np.ndarray
    X_val_text: list[str]
    y_val: np.ndarray
    X_holdout_text: list[str]
    y_holdout: np.ndarray
    kb_audit: dict[str, pd.DataFrame]
    split_audit: pd.DataFrame
    split_manifest: pd.DataFrame
    frozen_products: list[str] | None
    frozen_sha256: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.y_train)),
            "val": int(len(self.y_val)),
            "holdout": int(len(self.y_holdout)),
            "real_train": int(len(self.df_train)),
            "synthetic": int(len(self.df_synthetic)),
        }

    def summary_frame(self) -> pd.DataFrame:
        """Tabel ringkas yang enak ditampilkan di notebook."""
        return pd.DataFrame(
            [
                {"split": "real_train", "n": len(self.df_train),
                 "safe": int((self.df_train["label_id"] == 0).sum()),
                 "unsafe": int((self.df_train["label_id"] == 1).sum())},
                {"split": "synthetic", "n": len(self.df_synthetic),
                 "safe": int((self.df_synthetic["label_id"] == 0).sum()) if len(self.df_synthetic) else 0,
                 "unsafe": int((self.df_synthetic["label_id"] == 1).sum()) if len(self.df_synthetic) else 0},
                {"split": "final_train", "n": len(self.y_train),
                 "safe": int((self.y_train == 0).sum()),
                 "unsafe": int((self.y_train == 1).sum())},
                {"split": "validation", "n": len(self.y_val),
                 "safe": int((self.y_val == 0).sum()),
                 "unsafe": int((self.y_val == 1).sum())},
                {"split": "frozen_holdout", "n": len(self.y_holdout),
                 "safe": int((self.y_holdout == 0).sum()),
                 "unsafe": int((self.y_holdout == 1).sum())},
            ]
        )


def _packaged_frozen_path() -> Path:
    return Path(__file__).resolve().parents[1] / "core" / "data" / "frozen_holdout.json"


def _notebook23_path() -> Path:
    return Path(__file__).resolve().parents[1] / "core" / "data" / "frozen_holdout_notebook23.json"


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_stats(texts: list[str]) -> dict[str, float]:
    lengths = np.asarray([len(simple_tokenize(text)) for text in texts], dtype=float)
    if lengths.size == 0:
        return {"n": 0}
    return {
        "n": int(lengths.size),
        "mean_tokens": float(lengths.mean()),
        "median_tokens": float(np.median(lengths)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "max_tokens": float(lengths.max()),
        "mean_chars": float(np.mean([len(str(t)) for t in texts])),
    }


def prepare_data(
    config: DataConfig | None = None,
    *,
    save_artifacts: bool = True,
    save_bert_inputs: bool = True,
) -> DataBundle:
    """Jalankan seluruh pipeline data dan kembalikan ``DataBundle``.

    Args:
        config: konfigurasi data. Default membaca ``DataConfig()``.
        save_artifacts: simpan manifest/audit/ringkasan ke ``config.output_dir``.
        save_bert_inputs: tulis CSV split identik untuk IndoBERT.
    """
    config = config or DataConfig()
    csv_path = Path(config.csv_input)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV tidak ditemukan: {csv_path}. "
            "Set DataConfig.csv_input ke file dengan kolom produk;text;label."
        )

    df_raw = pd.read_csv(
        csv_path, delimiter=config.csv_delimiter, encoding=config.csv_encoding
    )
    df_raw = standardize_columns(
        df_raw,
        product_col=config.product_col,
        text_col=config.text_col,
        label_col=config.label_col,
    )
    text_col = config.text_col if config.text_col in df_raw.columns else "text"
    df_source = build_model_source_from_single(
        df_raw,
        product_col=config.product_col,
        text_col=text_col,
        label_col=config.label_col,
    )

    if config.sort_source:
        # Urutan baris CSV tidak boleh mengubah split. Sortir berdasarkan
        # group_key + teks agar hasil sama untuk isi data yang sama.
        df_source = (
            df_source.sort_values(["group_key", text_col], kind="mergesort")
            .reset_index(drop=True)
        )

    kb_audit = kb_vs_gold_audit(df_source)

    explicit_frozen = config.frozen_holdout_path
    frozen_products, frozen_sha = resolve_frozen_products(
        explicit_path=explicit_frozen,
        packaged_path=str(_packaged_frozen_path()),
        notebook23_path=str(_notebook23_path()),
        n_real=len(df_source),
    )

    split = run_v5_split(
        df_source,
        product_col=config.product_col,
        text_col=text_col,
        frozen_products=frozen_products,
        holdout_safe=config.holdout_safe,
        holdout_unsafe=config.holdout_unsafe,
        val_safe=config.val_safe,
        val_unsafe=config.val_unsafe,
    )
    df_train = split["df_real_train"]
    df_val = split["df_real_val"]
    df_holdout = split["df_holdout"]
    split_audit = audit_split_no_leakage(df_train, df_val, df_holdout)
    split_manifest = build_split_manifest(
        df_train, df_val, df_holdout, product_col=config.product_col
    )

    if config.synthetic_total > 0:
        df_synthetic = generate_synthetic_dataset(
            df_train_real=df_train,
            text_col=text_col,
            total_data=config.synthetic_total,
            seed=config.seed,
        )
    else:
        df_synthetic = pd.DataFrame(
            {"text": [], "label": [], "label_id": [], "synthetic_rule": []}
        )

    pool = build_final_training_pool(
        df_train, df_synthetic, df_val, df_holdout, text_col=text_col
    )
    audit_final_pool_no_leakage(
        pool["X_train_text"], pool["X_val_text"], pool["X_holdout_text"]
    )

    X_train_text = [str(x) for x in pool["X_train_text"]]
    X_val_text = [str(x) for x in pool["X_val_text"]]
    X_holdout_text = [str(x) for x in pool["X_holdout_text"]]
    y_train = np.asarray(pool["y_train"]).astype(int)
    y_val = np.asarray(pool["y_val"]).astype(int)
    y_holdout = np.asarray(pool["y_holdout"]).astype(int)

    metadata = {
        "csv_input": str(csv_path.resolve()),
        "csv_sha256": _sha256_file(csv_path),
        "n_source_rows": int(len(df_source)),
        "source_sorted": bool(config.sort_source),
        "n_real_train": int(len(df_train)),
        "n_validation": int(len(df_val)),
        "n_frozen_holdout": int(len(df_holdout)),
        "n_synthetic": int(len(df_synthetic)),
        "n_final_train": int(len(y_train)),
        "frozen_holdout_sha256": frozen_sha,
        "frozen_sha256": frozen_sha,
        "frozen_holdout_file": (
            Path(explicit_frozen).name if explicit_frozen else "frozen_holdout.json"
        ),
        "frozen_holdout_names": len(frozen_products) if frozen_products else 0,
        "text_stats_real_train": _text_stats(df_train[text_col].astype(str).tolist()),
        "text_stats_validation": _text_stats(df_val[text_col].astype(str).tolist()),
        "text_stats_holdout": _text_stats(df_holdout[text_col].astype(str).tolist()),
        "kb_audit_summary": kb_audit["summary"].to_dict(orient="records"),
        "split_audit": split_audit.to_dict(orient="records"),
    }

    bundle = DataBundle(
        config=config,
        df_source=df_source,
        df_train=df_train,
        df_val=df_val,
        df_holdout=df_holdout,
        df_synthetic=df_synthetic,
        X_train_text=X_train_text,
        y_train=y_train,
        X_val_text=X_val_text,
        y_val=y_val,
        X_holdout_text=X_holdout_text,
        y_holdout=y_holdout,
        kb_audit=kb_audit,
        split_audit=split_audit,
        split_manifest=split_manifest,
        frozen_products=frozen_products,
        frozen_sha256=frozen_sha,
        metadata=metadata,
    )

    if save_artifacts:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
        kb_audit["summary"].to_csv(output_dir / "kb_audit_summary.csv", index=False)
        kb_audit["confusion_matrix"].to_csv(output_dir / "kb_audit_confusion.csv")
        kb_audit["report"].to_csv(output_dir / "kb_audit_report.csv")
        split_audit.to_csv(output_dir / "split_audit.csv", index=False)
        bundle.summary_frame().to_csv(output_dir / "data_summary.csv", index=False)
        write_json(output_dir / "data_summary.json", metadata)
        if frozen_products:
            write_json(output_dir / "frozen_holdout_used.json", frozen_products)

    if save_bert_inputs:
        write_bert_inputs(bundle, config.bert_input_dir)

    return bundle


def write_bert_inputs(bundle: DataBundle, out_dir: str | os.PathLike) -> dict[str, str]:
    """Tulis CSV split yang byte-identik untuk training IndoBERT.

    File yang dihasilkan sama dengan ``scripts/rebuild_split.py``:

    - ``train_real.csv``, ``synthetic.csv``, ``train_combined.csv``
    - ``val.csv``, ``holdout.csv``
    - ``split_contract.json``, ``split_manifest.csv``
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = bundle.config
    text_col = cfg.text_col

    def _save_split(df: pd.DataFrame, name: str) -> Path:
        keep = [
            cfg.product_col, text_col, "label", "label_id", "gold_label", "group_key",
        ]
        keep = [col for col in keep if col in df.columns]
        target = out / name
        df[keep].to_csv(target, index=False)
        return target

    train_real = _save_split(bundle.df_train, "train_real.csv")
    val = _save_split(bundle.df_val, "val.csv")
    holdout = _save_split(bundle.df_holdout, "holdout.csv")

    synthetic_cols = [text_col, "label", "label_id", "synthetic_rule"]
    synthetic_cols = [col for col in synthetic_cols if col in bundle.df_synthetic.columns]
    synthetic_path = out / "synthetic.csv"
    bundle.df_synthetic[synthetic_cols].to_csv(synthetic_path, index=False)

    combined = pd.concat(
        [
            bundle.df_train[[text_col, "label_id"]].assign(
                label=bundle.df_train["label_id"].map({0: "safe", 1: "unsafe"})
            ),
            bundle.df_synthetic[[text_col, "label", "label_id"]]
            if len(bundle.df_synthetic) else pd.DataFrame(columns=[text_col, "label", "label_id"]),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    combined_path = out / "train_combined.csv"
    combined.to_csv(combined_path, index=False)

    bundle.split_manifest.to_csv(out / "split_manifest.csv", index=False)

    files = {
        "train_real.csv": train_real,
        "synthetic.csv": synthetic_path,
        "train_combined.csv": combined_path,
        "val.csv": val,
        "holdout.csv": holdout,
    }
    contract = {
        "seed": cfg.seed,
        "train_real": len(bundle.df_train),
        "synthetic": len(bundle.df_synthetic),
        "train_combined": len(combined),
        "val": len(bundle.df_val),
        "holdout": len(bundle.df_holdout),
        "frozen_holdout_file": bundle.metadata.get("frozen_holdout_file", "frozen_holdout.json"),
        "frozen_sha256": bundle.frozen_sha256,
        "frozen_holdout_sha256": bundle.frozen_sha256,
        "sha256": {name: _sha256_file(path) for name, path in files.items()},
    }
    write_json(out / "split_contract.json", contract)
    return {name: str(path) for name, path in files.items()}


def load_data_bundle_from_disk(data_dir: str | os.PathLike) -> dict[str, Any]:
    """Baca ringkasan data yang sudah disimpan (tanpa menjalankan split ulang).

    Berguna untuk notebook perbandingan yang hanya butuh tabel/metrik.
    """
    directory = Path(data_dir)
    summary = {}
    for name in ("data_summary.json", "split_contract.json"):
        path = directory / name
        if path.exists():
            summary[name] = json.loads(path.read_text(encoding="utf-8"))
    return summary
