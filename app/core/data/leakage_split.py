"""Leakage-safe split — port Notebook V5 Cell 36 + 38 + 42.

Aturan:
- Unit kebocoran = product group (nama ternormalisasi) ATAU hash teks
  OCR ternormalisasi yang sama -> harus satu split (union-find).
- Frozen holdout dikunci by NAMA PRODUK, lalu diperluas ke komponen penuh.
  Konflik komponen -> raise (tidak diam-diam diubah).
- Validasi dipilih deterministik exact (safe/unsafe) dari development.
- Audit: group disjoint + exact-text disjoint (termasuk final-train pool).

Tanpa TF.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_ocr_text_v5(text: object) -> str:
    """Normalisasi teks audit (port notebook Cell 36)."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s%.,;:()/_-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_product_key(x: object) -> str:
    """Normalisasi nama produk untuk identifier holdout."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower()).strip()


def _ensure_leakage_cols(
    df: pd.DataFrame, product_col: str, text_col: str
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if "label_id" not in out.columns:
        raise ValueError("df_model_source butuh kolom 'label_id'.")
    if "group_key" not in out.columns:
        from app.core.data.gold_merge import normalize_key

        out["group_key"] = out[product_col].map(normalize_key)
    out["_product_key_v5"] = out[product_col].map(normalize_product_key)
    out["audit_text_norm_v5"] = out[text_col].map(normalize_ocr_text_v5)
    out["audit_text_hash_v5"] = out["audit_text_norm_v5"].map(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    return out


def _build_components(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Union-find group_key + text-hash -> leakage_component_id + lookup."""
    n = len(df)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def union_by_key(series: pd.Series) -> None:
        first: dict[str, int] = {}
        for i, key in series.items():
            key = str(key)
            if key in first:
                union(first[key], int(i))
            else:
                first[key] = int(i)

    union_by_key(df["group_key"])
    union_by_key(df["audit_text_hash_v5"])

    roots = [find(i) for i in range(n)]
    root_to_rows: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        root_to_rows.setdefault(r, []).append(i)

    comp_ids = [min(root_to_rows[r]) for r in roots]
    df = df.copy()
    df["leakage_component_id"] = comp_ids

    comp_lookup: dict[int, dict] = {}
    for rows in root_to_rows.values():
        rows = sorted(rows)
        sub = df.iloc[rows]
        cid = int(min(rows))
        comp_lookup[cid] = {
            "component_id": cid,
            "rows": rows,
            "n_rows": len(rows),
            "safe": int((sub["label_id"] == 0).sum()),
            "unsafe": int((sub["label_id"] == 1).sum()),
        }
    return df, comp_lookup


def choose_components_exact_deterministic(
    candidates: list[int],
    comp_lookup: dict,
    target_safe: int,
    target_unsafe: int,
) -> set[int] | None:
    """Pilih subset komponen exact (safe,unsafe) secara deterministik.

    Kandidat diproses terurut; opsi INCLUDE dicoba dulu -> deterministik.
    """
    candidates = sorted(int(x) for x in candidates)
    sys.setrecursionlimit(max(10000, len(candidates) * 2 + 100))

    @lru_cache(maxsize=None)
    def solve(pos: int, safe_left: int, unsafe_left: int):
        if safe_left == 0 and unsafe_left == 0:
            return ()
        if pos >= len(candidates):
            return None
        cid = candidates[pos]
        c = comp_lookup[cid]
        cs, cu = int(c["safe"]), int(c["unsafe"])
        if cs <= safe_left and cu <= unsafe_left:
            res = solve(pos + 1, safe_left - cs, unsafe_left - cu)
            if res is not None:
                return (cid,) + res
        return solve(pos + 1, safe_left, unsafe_left)

    selected = solve(0, int(target_safe), int(target_unsafe))
    return set(selected) if selected is not None else None


def run_v5_split(
    df_model_source: pd.DataFrame,
    *,
    holdout_safe: int,
    holdout_unsafe: int,
    val_safe: int,
    val_unsafe: int,
    product_col: str = "nama_produk",
    text_col: str = "text",
    frozen_products: list[str] | None = None,
) -> dict:
    """Jalankan split V5. Returns dict(df_real_train/df_real_val/df_holdout/...).

    Komposisi (safe/unsafe) WAJIB eksplisit dari V5Config — tidak ada default
    angka agar kontrak split tidak terfragmentasi.

    - Jika ``frozen_products`` diberikan: kunci holdout dari daftar itu
      (cocok by nama ternormalisasi), validasi komponen penuh.
    - Jika None: pilih holdout deterministik exact (holdout_safe/unsafe)
      dari semua komponen terurut (untuk re-derive sebelum freeze).
    - Validasi: pilih deterministik exact (val_safe/unsafe) dari development.
    """
    df = _ensure_leakage_cols(df_model_source, product_col, text_col)
    df, comp_lookup = _build_components(df)

    if frozen_products is not None:
        frozen_keys = {normalize_product_key(x) for x in frozen_products}
        if len(frozen_keys) != len(frozen_products):
            raise ValueError("Daftar frozen holdout mengandung duplikat setelah normalisasi.")
        available = set(df["_product_key_v5"])
        missing = frozen_keys - available
        if missing:
            raise ValueError(f"Frozen holdout tidak ditemukan di data: {sorted(missing)[:10]}")
        holdout_row_idx = sorted(df.index[df["_product_key_v5"].isin(frozen_keys)].tolist())
        # Perluas ke komponen penuh + cek konflik.
        holdout_cids = {int(df.loc[i, "leakage_component_id"]) for i in holdout_row_idx}
        expanded = sorted({i for cid in holdout_cids for i in comp_lookup[cid]["rows"]})
        expanded_keys = {df.loc[i, "_product_key_v5"] for i in expanded}
        unexpected = expanded_keys - frozen_keys
        if unexpected:
            raise RuntimeError(
                "FROZEN HOLDOUT COMPONENT CONFLICT: produk frozen satu komponen "
                f"dengan produk lain: {sorted(unexpected)[:10]}. Holdout TIDAK diubah."
            )
        hold_idx = sorted(expanded)
        sub = df.loc[hold_idx]
        hs, hu = int((sub["label_id"] == 0).sum()), int((sub["label_id"] == 1).sum())
        if (hs, hu) != (holdout_safe, holdout_unsafe):
            raise ValueError(
                f"Komposisi frozen holdout ({hs} safe/{hu} unsafe) != target "
                f"({holdout_safe}/{holdout_unsafe}). Periksa daftar/kontrak."
            )
    else:
        all_cids = sorted(comp_lookup.keys())
        sel = choose_components_exact_deterministic(all_cids, comp_lookup, holdout_safe, holdout_unsafe)
        if sel is None:
            raise RuntimeError(
                f"Tidak ada holdout exact {holdout_safe} safe/{holdout_unsafe} unsafe. "
                "Tentukan daftar frozen eksplisit atau ubah target."
            )
        hold_idx = sorted({i for cid in sel for i in comp_lookup[cid]["rows"]})

    development_idx = sorted(set(range(len(df))) - set(hold_idx))
    dev_cids = sorted({int(df.loc[i, "leakage_component_id"]) for i in development_idx})
    val_cids = choose_components_exact_deterministic(dev_cids, comp_lookup, val_safe, val_unsafe)
    if val_cids is None:
        raise RuntimeError(
            f"Tidak ada validasi exact {val_safe} safe/{val_unsafe} unsafe dari "
            f"{len(development_idx)} development. Ubah target."
        )
    val_idx = sorted({i for cid in val_cids for i in comp_lookup[cid]["rows"]})
    train_idx = sorted(set(development_idx) - set(val_idx))

    df_holdout = df.loc[hold_idx].reset_index(drop=True)
    df_real_val = df.loc[val_idx].reset_index(drop=True)
    df_real_train = df.loc[train_idx].reset_index(drop=True)

    logger.info(
        "V5 split: train=%d val=%d holdout=%d",
        len(df_real_train), len(df_real_val), len(df_holdout),
    )
    return {
        "df_real_train": df_real_train,
        "df_real_val": df_real_val,
        "df_holdout": df_holdout,
        "comp_lookup": comp_lookup,
        "df_indexed": df,
    }


def audit_split_no_leakage(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_holdout: pd.DataFrame
) -> pd.DataFrame:
    """Audit group + exact-text disjoint. Raise bila bocor. Return tabel audit."""
    for name, d in (("train", df_train), ("val", df_val), ("holdout", df_holdout)):
        if "group_key" not in d.columns or "audit_text_hash_v5" not in d.columns:
            raise ValueError(f"Split '{name}' butuh group_key + audit_text_hash_v5.")

    def _set(d: pd.DataFrame, c: str) -> set:
        return set(d[c].fillna("").astype(str))

    rows = []
    for a_name, b_name, col, label in [
        ("train", "val", "group_key", "group train/val"),
        ("train", "holdout", "group_key", "group train/holdout"),
        ("val", "holdout", "group_key", "group val/holdout"),
        ("train", "val", "audit_text_hash_v5", "exact-text train/val"),
        ("train", "holdout", "audit_text_hash_v5", "exact-text train/holdout"),
        ("val", "holdout", "audit_text_hash_v5", "exact-text val/holdout"),
    ]:
        frames = {"train": df_train, "val": df_val, "holdout": df_holdout}
        overlap = _set(frames[a_name], col) & _set(frames[b_name], col)
        # Hash teks kosong (teks kosong) diabaikan bila keduanya kosong? teks kosong
        # sudah dibuang saat merge, tapi tetap: abaikan string kosong.
        overlap = {x for x in overlap if x}
        # Untuk group: string kosong juga abaikan.
        rows.append({
            "comparison": label,
            "overlap": len(overlap),
            "status": "PASS" if not overlap else "FAIL",
        })
    audit = pd.DataFrame(rows)
    if (audit["status"] == "FAIL").any():
        raise RuntimeError(f"Leakage terdeteksi:\n{audit.to_string(index=False)}")
    return audit


def audit_final_pool_no_leakage(
    train_texts: list[str], val_texts: list[str], holdout_texts: list[str]
) -> pd.DataFrame:
    """Audit final-train (real+synthetic) vs val/holdout — port Cell 42."""
    import hashlib

    def _hashes(values: list[str]) -> set:
        out = set()
        for v in values:
            n = normalize_ocr_text_v5(v)
            if n:
                out.add(hashlib.sha256(n.encode("utf-8")).hexdigest())
        return out

    th, vh, hh = _hashes(train_texts), _hashes(val_texts), _hashes(holdout_texts)
    audit = pd.DataFrame([
        {"comparison": "final_train_vs_validation",
         "exact_text_overlap": len(th & vh),
         "status": "PASS" if not (th & vh) else "FAIL"},
        {"comparison": "final_train_vs_holdout",
         "exact_text_overlap": len(th & hh),
         "status": "PASS" if not (th & hh) else "FAIL"},
        {"comparison": "validation_vs_holdout",
         "exact_text_overlap": len(vh & hh),
         "status": "PASS" if not (vh & hh) else "FAIL"},
    ])
    if (audit["status"] == "FAIL").any():
        raise RuntimeError(f"Final-pool overlap:\n{audit.to_string(index=False)}")
    return audit


def build_split_manifest(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_holdout: pd.DataFrame,
    product_col: str = "nama_produk",
) -> pd.DataFrame:
    """Manifest reproduksibilitas split (port Cell 38)."""
    def _rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
        cols = [product_col, "group_key", "gold_label", "label_id",
                "audit_text_hash_v5", "leakage_component_id"]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()
        out["split"] = name
        return out

    return pd.concat([
        _rows(df_train, "real_train"),
        _rows(df_val, "real_validation"),
        _rows(df_holdout, "frozen_holdout"),
    ], ignore_index=True)
