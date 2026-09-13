"""Synthetic generator — port Notebook V5 Cell 19/40.

Kontrak:
- HANYA dari real-train (tidak pernah val/holdout).
- Seimbang 50/50; total harus genap.
- Label synthetic di-validasi ulang oleh KB; mismatch -> RuntimeError.
- UNSAFE harus mengandung >=1 kategori alergen kanonis.
- 8 kategori kanonis selalu tersedia (fallback bila train tidak punya).

Tanpa TF.
"""

from __future__ import annotations

import logging
import random
import re

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    text = str(text or "").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def generate_synthetic_dataset(
    df_train_real: pd.DataFrame,
    text_col: str = "text",
    total_data: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic training-only dari real-train."""
    from app.core.kb.rules import ALLERGEN_KB, rule_based_kb_label

    rng = random.Random(seed)
    if total_data < 2 or total_data % 2 != 0:
        raise ValueError("SYNTHETIC_TOTAL harus genap dan >= 2.")
    if text_col not in df_train_real.columns:
        raise ValueError(f"Kolom teks '{text_col}' tidak ada.")
    if df_train_real.empty:
        raise ValueError("df_train_real kosong.")

    train_texts = df_train_real[text_col].fillna("").astype(str).tolist()

    def split_ingredients(text: str) -> list[str]:
        parts = re.split(r"[,;.!\n]+", str(text))
        return [_normalize(p) for p in parts if len(_normalize(p)) >= 3]

    # Common pool: frasa train yang KB-safe.
    common_pool: set[str] = set()
    for text in train_texts:
        for part in split_ingredients(text):
            if not rule_based_kb_label(part)["unsafe"]:
                common_pool.add(part)
    common_pool.update([
        "Gula", "Garam", "Air", "Pati Tapioka", "Pati Jagung",
        "Minyak Nabati", "Pengatur Keasaman Asam Sitrat",
        "Pengental Nabati", "Penstabil Nabati", "Perisa",
    ])
    common_pool = sorted(common_pool)

    # Allergen seeds per kategori dari train.
    seeds: dict[str, set[str]] = {cat: set() for cat in ALLERGEN_KB}
    for text in train_texts:
        result = rule_based_kb_label(text)
        for cat in result["allergen_categories"]:
            for term in result["matched_terms"]:
                if term:
                    seeds[cat].add(term)
    canonical_fallback = {
        "Gluten": ["Tepung Terigu"],
        "Dairy": ["Susu Bubuk"],
        "Egg": ["Telur Bubuk"],
        "Soy": ["Lesitin Kedelai"],
        "Fish": ["Ikan"],
        "Crustacean": ["Udang"],
        "Sesame": ["Biji Wijen"],
        "Tree Nut": ["Kacang Mede"],
    }
    for cat, fb in canonical_fallback.items():
        if not seeds[cat]:
            seeds[cat].update(fb)
    seeds = {k: sorted(v) for k, v in seeds.items()}

    def choose_common(n: int) -> list[str]:
        return rng.sample(common_pool, min(n, len(common_pool)))

    def make_safe() -> str:
        for _ in range(100):
            chosen = choose_common(rng.randint(4, 8))
            text = ", ".join(chosen) + "."
            if not rule_based_kb_label(text)["unsafe"]:
                return text
        return "Gula, Garam, Air, Pati Tapioka, Minyak Nabati."

    def make_unsafe() -> str:
        category = rng.choice(list(seeds.keys()))
        allergen = rng.choice(seeds[category])
        chosen_common = choose_common(rng.randint(3, 6))
        items = chosen_common + [allergen]
        rng.shuffle(items)
        text = ", ".join(items) + "."
        if not rule_based_kb_label(text)["unsafe"]:
            text = f"{allergen}, " + ", ".join(chosen_common) + "."
        return text

    rows = []
    for _ in range(total_data // 2):
        rows.append({"text": make_safe(), "label": "safe", "synthetic_rule": "KB_SAFE"})
    for _ in range(total_data // 2):
        rows.append({"text": make_unsafe(), "label": "unsafe", "synthetic_rule": "KB_UNSAFE"})

    df_syn = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df_syn["kb_check"] = df_syn["text"].map(lambda x: rule_based_kb_label(x)["label"])
    mismatch = (df_syn["label"].str.lower() != df_syn["kb_check"].str.lower()).sum()
    if mismatch:
        raise RuntimeError(f"Synthetic KB validation gagal: {mismatch} baris tidak konsisten.")
    df_syn = df_syn.drop(columns=["kb_check"])
    df_syn["label_id"] = df_syn["label"].map({"safe": 0, "unsafe": 1}).astype(int)
    logger.info("Synthetic: %d rows (50/50) dari %d real-train", len(df_syn), len(df_train_real))
    return df_syn


def build_final_training_pool(
    df_real_train: pd.DataFrame,
    df_sintesis: pd.DataFrame,
    df_real_val: pd.DataFrame,
    df_holdout: pd.DataFrame,
    text_col: str = "text",
) -> dict:
    """Bangun pool final: train = real-train + synthetic; val/holdout real Gold.

    Menegakkan kontrak ukuran val/holdout tidak berubah oleh synthetic.
    """
    for name, d in (("real_train", df_real_train), ("val", df_real_val), ("holdout", df_holdout)):
        if text_col not in d.columns or "label_id" not in d.columns:
            raise ValueError(f"Split '{name}' butuh kolom {text_col} + label_id.")
    if "label_id" not in df_sintesis.columns:
        raise ValueError("df_sintesis butuh kolom label_id.")

    X_real = df_real_train[text_col].fillna("").astype(str).tolist()
    y_real = df_real_train["label_id"].astype(int).values
    X_syn = df_sintesis["text" if "text" in df_sintesis.columns else text_col].fillna("").astype(str).tolist()
    y_syn = df_sintesis["label_id"].astype(int).values

    import numpy as np

    X_train_text = X_real + X_syn
    y_train = np.concatenate([y_real, y_syn])
    X_val_text = df_real_val[text_col].fillna("").astype(str).tolist()
    y_val = df_real_val["label_id"].astype(int).values
    X_holdout_text = df_holdout[text_col].fillna("").astype(str).tolist()
    y_holdout = df_holdout["label_id"].astype(int).values

    assert len(X_val_text) == len(df_real_val)
    assert len(X_holdout_text) == len(df_holdout)
    return {
        "X_train_text": X_train_text,
        "y_train": y_train,
        "X_val_text": X_val_text,
        "y_val": y_val,
        "X_holdout_text": X_holdout_text,
        "y_holdout": y_holdout,
    }
