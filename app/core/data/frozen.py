"""Frozen holdout: loader bersama + anti-tamper hash.

Satu-satunya implementasi baca daftar frozen + hash, dipakai oleh
trainer_v5, bootstrap_final, dan rebuild_split agar guard
``V5_FROZEN_SHA256`` tidak bisa dilewati jalur manapun.

Semantik hash: sha256 dari nama produk setelah normalisasi kanonis
(lowercase + spasi tunggal, sebagaimana ``normalize_product_key``),
diurutkan lalu di-join ``\\n``. Dengan begitu hash tahan terhadap
varian kapitalisasi/spasi (mis. "Bakso Sapi Karawaci" == "bakso sapi karawaci")
dan identik dengan yang bisa direproduksi dari ``split_manifest.csv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# Nama env anti-tamper. Bila diset, hash daftar frozen WAJIB cocok.
FROZEN_SHA_ENV = "V5_FROZEN_SHA256"


def normalize_frozen_name(name: str) -> str:
    """Normalisasi kanonis nama produk frozen (lowercase + spasi tunggal)."""
    return re_sub_ws(str(name or "").strip().lower())


def re_sub_ws(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text)


def frozen_list_sha256(products: list[str]) -> str:
    """Hash kanonis daftar frozen: normalized names -> sorted -> '\\n'.join."""
    normalized = sorted(normalize_frozen_name(p) for p in products)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def load_frozen_products(path: str, expected_sha: str | None = None) -> tuple[list[str], str]:
    """Baca JSON daftar frozen + hitung hash kanonis.

    Args:
        path: path file JSON (list string nama produk).
        expected_sha: hash ekspektasi. Default None -> baca env
            ``V5_FROZEN_SHA256`` (kosong = tanpa validasi, hanya catat hash).

    Returns:
        (products, sha256_kanonis).

    Raises:
        FileNotFoundError: file tidak ada.
        ValueError: daftar bukan list / hash tidak cocok.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Frozen holdout {path} harus berupa list string.")
    actual = frozen_list_sha256(data)

    expected = expected_sha if expected_sha is not None else (
        (os.getenv(FROZEN_SHA_ENV) or "").strip().lower()
    )
    if expected and expected != actual:
        raise ValueError(
            f"Hash frozen holdout {actual} != {FROZEN_SHA_ENV} {expected}. "
            "Daftar frozen tidak boleh berubah setelah freeze."
        )
    logger.info("Frozen holdout %s: %d nama, sha256=%s", path, len(data), actual)
    return data, actual


def resolve_frozen_products(
    explicit_path: str | None,
    packaged_path: str,
    notebook23_path: str,
    n_real: int,
) -> tuple[list[str] | None, str | None]:
    """Urutan resolusi frozen holdout (sama dengan trainer historis).

    1. ``explicit_path`` (V5_FROZEN_HOLDOUT_PATH) -> pakai bila diset.
    2. ``packaged_path`` (frozen_holdout.json kemasan) -> bila ada.
    3. ``notebook23_path`` bila n_real == 114 (comparability notebook).
    4. None (re-derive deterministik) — caller diperingatkan untuk freeze.

    Semua jalur melalui ``load_frozen_products`` sehingga hash
    ``V5_FROZEN_SHA256`` selalu divalidasi.

    Returns:
        (products_or_None, sha_or_None)
    """
    if explicit_path:
        return load_frozen_products(explicit_path)
    if packaged_path and os.path.exists(packaged_path):
        return load_frozen_products(packaged_path)
    if n_real == 114 and os.path.exists(notebook23_path):
        return load_frozen_products(notebook23_path)
    logger.warning(
        "n_real=%d, tak ada frozen kemasan -> re-derive deterministik. "
        "SEGERA freeze hasilnya (simpan JSON + set %s).",
        n_real, FROZEN_SHA_ENV,
    )
    return None, None
