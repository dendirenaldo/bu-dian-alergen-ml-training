"""Gold merge — port Notebook V5 Cell 29.

Dua mode:
A. Notebook: df_ocr (produk,text) + df_gold (Gambar,Label) -> merge via match_key.
B. Lokal CSV parity: satu df (produk,text,label) -> label diperlakukan
   sebagai gold_label, tambah group_key + kolom KB diagnostik.

Tanpa TF. Konflik duplikat Gold -> raise (sesuai notebook: STOP).
"""

from __future__ import annotations

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_key(x: object) -> str:
    """Normalisasi identitas produk/group (port notebook).

    Contoh: 'nestle honey stars.jpg' dan 'nestle_honey_stars.jpg'
    -> 'nestle honey stars'.
    """
    x = os.path.splitext(os.path.basename(str(x).strip()))[0].lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def normalize_product_name(x: object) -> str:
    """Alias eksplisit untuk group_key produk."""
    return normalize_key(x)


def _canon_label(s: object) -> str:
    return (
        str(s or "")
        .strip()
        .lower()
        .replace("0", "safe")
        .replace("1", "unsafe")
    )


def attach_kb_columns(
    df: pd.DataFrame,
    text_col: str = "text",
    product_col: str = "nama_produk",
) -> pd.DataFrame:
    """Tambah kolom KB diagnostik (tidak menggantikan gold)."""
    from app.core.kb.rules import rule_based_kb_label

    out = df.copy()
    kb = out[text_col].fillna("").astype(str).map(rule_based_kb_label)
    out["kb_label"] = kb.map(lambda r: r["label"])
    out["kb_binary_label"] = kb.map(lambda r: r["binary_label"])
    out["kb_confidence"] = kb.map(lambda r: r["confidence"])
    out["kb_allergen_categories"] = kb.map(lambda r: ";".join(r["allergen_categories"]))
    out["kb_matched_terms"] = kb.map(lambda r: ";".join(r["matched_terms"]))
    out["kb_evidence_methods"] = kb.map(lambda r: ";".join(r["evidence_methods"]))
    return out


def standardize_columns(
    df: pd.DataFrame,
    product_col: str = "nama_produk",
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Samakan varian nama kolom CSV ke kanonis (satu-satunya shim kolom).

    Kanonis: ``nama_produk`` / ``text`` / ``label``. Varian spasi
    ("nama produk") dan generik ("product", "produk") dipetakan otomatis.
    Bila kolom produk tak ada sama sekali, sintesis ID per-baris.
    """
    out = df.copy()
    rename: dict[str, str] = {}
    if product_col not in out.columns:
        for cand in ("nama produk", "nama_produk", "product", "produk"):
            if cand in out.columns:
                rename[cand] = product_col
                break
    if text_col not in out.columns:
        for cand in ("text", "komposisi", "composition"):
            if cand in out.columns:
                rename[cand] = text_col
                break
    if label_col not in out.columns and label_col != "label" and "label" in out.columns:
        rename["label"] = label_col
    if rename:
        out = out.rename(columns=rename)
    if product_col not in out.columns:
        out[product_col] = [f"produk_{i}" for i in range(len(out))]
    return out


def build_model_source_from_single(
    df: pd.DataFrame,
    product_col: str = "nama_produk",
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Mode B: satu CSV lokal -> df_model_source.

    - Buang teks kosong & label selain safe/unsafe (log warning).
    - ``label`` dikunci sebagai ``gold_label`` + ``label_id``.
    - Tambah ``match_key``, ``group_key``, kolom KB diagnostik.
    - Duplikat OCR dipertahankan (leakage ditangani saat split).
    """
    for c in (product_col, text_col, label_col):
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ada. Kolom: {list(df.columns)}")
    out = df[[product_col, text_col, label_col]].copy()
    out[product_col] = out[product_col].fillna("").astype(str).str.strip()
    out[text_col] = out[text_col].fillna("").astype(str)
    out["gold_label"] = out[label_col].astype(str).str.strip().str.lower().map(_canon_label)
    n_empty = int((out[text_col].str.strip() == "").sum())
    out = out[out[text_col].str.strip() != ""].copy()
    n_bad = int((~out["gold_label"].isin(["safe", "unsafe"])).sum())
    out = out[out["gold_label"].isin(["safe", "unsafe"])].copy()
    if n_empty:
        logger.warning("Baris teks kosong dibuang: %d", n_empty)
    if n_bad:
        logger.warning("Baris label selain safe/unsafe dibuang: %d", n_bad)
    if out.empty:
        raise ValueError("Dataset kosong setelah filtering.")
    if out["gold_label"].nunique() < 2:
        raise ValueError("Dataset hanya 1 kelas — split & ROC-AUC tidak valid.")
    out["match_key"] = out[product_col].map(normalize_key)
    out["group_key"] = out[product_col].map(normalize_key)
    out["label_id"] = out["gold_label"].map({"safe": 0, "unsafe": 1}).astype(int)
    out = attach_kb_columns(out, text_col=text_col, product_col=product_col)
    return out.reset_index(drop=True)


def merge_ocr_with_gold(
    df_ocr: pd.DataFrame,
    df_gold: pd.DataFrame,
    product_col: str = "nama_produk",
    text_col: str = "text",
    gold_key_col: str = "Gambar",
    gold_label_col: str = "Label",
) -> pd.DataFrame:
    """Mode A (notebook): merge OCR dengan Gold Excel via match_key.

    - Gold didedup ke 1 baris per match_key; konflik label -> ValueError.
    - Merge many-to-one; jumlah baris hasil harus == len(df_ocr).
    - Tambah group_key/label_id + kolom KB.
    """
    for c in (product_col, text_col):
        if c not in df_ocr.columns:
            raise ValueError(f"df_ocr butuh kolom '{c}'. Ada: {list(df_ocr.columns)}")
    for c in (gold_key_col, gold_label_col):
        if c not in df_gold.columns:
            raise ValueError(f"df_gold butuh kolom '{c}'. Ada: {list(df_gold.columns)}")

    ocr = df_ocr[[product_col, text_col]].copy()
    ocr[product_col] = ocr[product_col].fillna("").astype(str).str.strip()
    ocr[text_col] = ocr[text_col].fillna("").astype(str)
    ocr["match_key"] = ocr[product_col].map(normalize_key)

    gold = df_gold.copy()
    gold["match_key"] = gold[gold_key_col].map(normalize_key)
    gold["gold_label"] = gold[gold_label_col].astype(str).str.strip().str.lower().map(_canon_label)
    gold = gold[gold["gold_label"].isin(["safe", "unsafe"])].copy()
    if gold.empty:
        raise ValueError("Gold kosong setelah filtering safe/unsafe.")

    dup = gold[gold["match_key"].duplicated(keep=False)]
    if len(dup):
        conflicts = dup.groupby("match_key")["gold_label"].nunique()
        bad = conflicts[conflicts > 1].index.tolist()
        if bad:
            raise ValueError(f"Duplikat Gold konflik, resolusi manual: {bad[:10]}")
    gold_unique = (
        gold.sort_values(["match_key", gold_key_col])
        .drop_duplicates(subset="match_key", keep="first")
        .copy()
    )

    merged = ocr.merge(
        gold_unique[["match_key", "gold_label"]],
        on="match_key",
        how="inner",
        validate="many_to_one",
    )
    if len(merged) != len(ocr):
        missing = sorted(set(ocr["match_key"]) - set(gold_unique["match_key"]))[:10]
        raise ValueError(
            f"Gold merge gagal: OCR={len(ocr)}, merged={len(merged)}. "
            f"Missing keys contoh: {missing}"
        )
    merged["group_key"] = merged[product_col].map(normalize_key)
    merged["label_id"] = merged["gold_label"].map({"safe": 0, "unsafe": 1}).astype(int)
    merged = attach_kb_columns(merged, text_col=text_col, product_col=product_col)
    logger.info(
        "Gold merge: OCR=%d, gold unik=%d, groups=%d",
        len(ocr), len(gold_unique), merged["group_key"].nunique(),
    )
    return merged.reset_index(drop=True)
