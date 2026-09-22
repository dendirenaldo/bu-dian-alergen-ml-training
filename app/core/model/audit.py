"""Audit V5 — port Notebook Cell 32/33/58-66/69/71/72.

Semua diagnostik (read-only): tidak pernah tuning dari holdout.
Tanpa TF.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def kb_vs_gold_audit(df_model_source: pd.DataFrame) -> dict:
    """Audit KB vs Gold (port Cell 32/33). Gold tetap ground truth."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = df_model_source["gold_label"].astype(str).str.lower().map({"safe": 0, "unsafe": 1})
    y_pred = pd.to_numeric(df_model_source["kb_binary_label"], errors="coerce")
    valid = pd.DataFrame({"gold": y_true, "kb": y_pred}).dropna()
    valid = valid[valid["gold"].isin([0, 1]) & valid["kb"].isin([0, 1])]
    if valid.empty:
        raise ValueError("Tidak ada baris valid untuk audit KB vs Gold.")
    yt, yp = valid["gold"].astype(int), valid["kb"].astype(int)
    summary = pd.DataFrame([{
        "n_real_samples": len(valid),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_unsafe": float(precision_score(yt, yp, zero_division=0)),
        "recall_unsafe": float(recall_score(yt, yp, zero_division=0)),
        "f1_unsafe": float(f1_score(yt, yp, zero_division=0)),
    }])
    cm = pd.DataFrame(
        confusion_matrix(yt, yp, labels=[0, 1]),
        index=["Gold SAFE", "Gold UNSAFE"], columns=["KB SAFE", "KB UNSAFE"],
    )
    report = pd.DataFrame(
        classification_report(yt, yp, labels=[0, 1],
                              target_names=["SAFE", "UNSAFE"],
                              output_dict=True, zero_division=0)
    ).T
    return {"summary": summary, "confusion_matrix": cm, "report": report}


def triage_gold_kb_bilstm(
    df_holdout: pd.DataFrame,
    holdout_prob: np.ndarray,
    product_col: str = "nama_produk",
    threshold: float = 0.50,
) -> dict:
    """Triase GOLD vs KB vs BiLSTM di frozen holdout (port Cell 58/65).

    Holdout hanya diagnostik — tidak ada tuning dari sini.
    """
    from app.core.model.metrics import _require_fixed_threshold

    threshold = _require_fixed_threshold(threshold)
    tri = df_holdout.copy().reset_index(drop=True)
    tri["prob_unsafe"] = np.asarray(holdout_prob).astype(float).ravel()
    tri["predicted_label"] = np.where(tri["prob_unsafe"] >= threshold, "unsafe", "safe")
    tri["gold_binary"] = tri["gold_label"].astype(str).str.lower().map({"safe": 0, "unsafe": 1})
    tri["kb_binary"] = pd.to_numeric(tri["kb_binary_label"], errors="coerce")
    tri["model_binary"] = tri["predicted_label"].map({"safe": 0, "unsafe": 1})

    def _row(r) -> str:
        g, k, m = r["gold_binary"], r["kb_binary"], r["model_binary"]
        if g == k == m:
            return "GOLD_CORRECT_KB_CORRECT_MODEL_CORRECT"
        if g == 1 and k == 0 and m == 0:
            return "KB_FALSE_NEGATIVE_MODEL_FALSE_NEGATIVE"
        if g == 1 and k == 0 and m == 1:
            return "KB_FALSE_NEGATIVE_MODEL_CORRECT"
        if g == 1 and k == 1 and m == 0:
            return "KB_CORRECT_MODEL_FALSE_NEGATIVE"
        if g == 0 and k == 1 and m == 1:
            return "KB_FALSE_POSITIVE_MODEL_FALSE_POSITIVE"
        if g == 0 and k == 1 and m == 0:
            return "KB_FALSE_POSITIVE_MODEL_CORRECT"
        if g == 0 and k == 0 and m == 1:
            return "KB_CORRECT_MODEL_FALSE_POSITIVE"
        return "UNCLASSIFIED"

    tri["error_triage"] = tri.apply(_row, axis=1)
    counts = tri["error_triage"].value_counts().rename_axis("error_triage").reset_index(name="n")
    fn = tri[(tri["gold_binary"] == 1) & (tri["model_binary"] == 0)].copy()
    fp = tri[(tri["gold_binary"] == 0) & (tri["model_binary"] == 1)].copy()
    return {"triage": tri, "counts": counts, "false_negatives": fn, "false_positives": fp}


def audit_reproducibility_v5(
    seed: int = 42,
    w2v_workers: int = 1,
    w2v_seed: int = 42,
    train_shuffle: bool = False,
    fixed_threshold: float = 0.50,
    holdout_size: int | None = None,
    expected_holdout_size: int = 50,
) -> dict:
    """Audit lock randomness. Raise bila gagal.

    Benih tak harus 42 — yang diaudit adalah konsistensi: PYTHONHASHSEED
    dan seed Word2Vec sama dengan seed run, sehingga --seed CLI legal
    selama deterministik. Ukuran holdout wajib cocok kontrak config
    (bukan angka notebook lama).
    """
    checks = {
        f"SEED recorded ({seed})": isinstance(seed, int) and seed >= 0,
        "PYTHONHASHSEED matches seed": os.environ.get("PYTHONHASHSEED") == str(seed),
        "TF_DETERMINISTIC_OPS=1": os.environ.get("TF_DETERMINISTIC_OPS") == "1",
        "TF_CUDNN_DETERMINISTIC=1": os.environ.get("TF_CUDNN_DETERMINISTIC") == "1",
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "TF_ENABLE_ONEDNN_OPTS=0": os.environ.get("TF_ENABLE_ONEDNN_OPTS") == "0",
        "Word2Vec workers=1": w2v_workers == 1,
        f"Word2Vec seed matches ({seed})": w2v_seed == seed,
        "Train shuffle=False": train_shuffle is False,
        "Fixed threshold=0.50": fixed_threshold == 0.50,
    }
    if holdout_size is not None:
        checks[f"Frozen holdout={expected_holdout_size}"] = (
            holdout_size == expected_holdout_size
        )
    if not all(checks.values()):
        bad = [k for k, v in checks.items() if not v]
        raise AssertionError(f"Reproducibility audit gagal: {bad}")
    return checks


def audit_pipeline_consistency_v5(
    df_real_train: pd.DataFrame,
    df_real_val: pd.DataFrame,
    df_holdout: pd.DataFrame,
    n_synthetic: int,
    synthetic_total: int,
    holdout_used_in_fit: bool = False,
    fixed_threshold: float = 0.50,
) -> None:
    """Guard konsistensi pipeline (port Cell 71). Raise eksplisit bila gagal.

    (Gaya ``raise ValueError`` bukan ``assert`` mentah — assert bisa
    dinonaktifkan dengan ``python -O`` dan tidak membawa pesan audit.)
    """
    from app.core.model.metrics import _require_fixed_threshold

    if holdout_used_in_fit:
        raise ValueError("Holdout tidak boleh masuk model.fit().")
    _require_fixed_threshold(fixed_threshold)
    if n_synthetic != synthetic_total:
        raise ValueError(
            f"Ukuran synthetic {n_synthetic} != kontrak {synthetic_total}."
        )
    for name, d in (("train", df_real_train), ("val", df_real_val), ("holdout", df_holdout)):
        if "label_id" not in d.columns:
            raise ValueError(f"Split {name} butuh kolom label_id.")
        if not d["label_id"].isin([0, 1]).all():
            raise ValueError(f"Split {name} memiliki label_id di luar {0, 1}.")
        if len(np.unique(d["label_id"].to_numpy())) < 2:
            raise ValueError(
                f"Split {name} single-class — ROC/PR/AUC tidak terdefinisi. "
                "Periksa komposisi split."
            )


def audit_final_config_v5(v5) -> dict:
    """Audit konfigurasi final aktual (kontrak D1/D2/D3 + keputusan Fase 2-4).

    Gagal keras bila ada override env yang menyimpang dari kontrak final.
    Return dict nilai aktual untuk direkam ke manifest.
    """
    checks = {
        "learning_rate==1e-4 (D2)": float(v5.learning_rate) == 1e-4,
        "gradient_clip==1.0 (D1)": v5.gradient_clip_norm == 1.0,
        "mask_zero=True (D3)": v5.mask_zero is True,
        "w2v_min_count==1": int(v5.w2v_min_count) == 1,
        "digit_fold==False": v5.digit_fold is False,
        "preshuffle==False": v5.preshuffle is False,
        "train_shuffle==False": v5.train_shuffle is False,
        "holdout 25/25": (v5.holdout_safe, v5.holdout_unsafe) == (25, 25),
        "val 25/25": (v5.val_safe, v5.val_unsafe) == (25, 25),
        "synthetic_total==1000": int(v5.synthetic_total) == 1000,
        "embed_trainable==True": v5.embed_trainable is True,
    }
    if not all(checks.values()):
        bad = [k for k, ok in checks.items() if not ok]
        raise ValueError(f"Audit konfigurasi final GAGAL (override env?): {bad}")
    return {
        "learning_rate": float(v5.learning_rate),
        "gradient_clip_norm": float(v5.gradient_clip_norm),
        "mask_zero": bool(v5.mask_zero),
        "w2v_min_count": int(v5.w2v_min_count),
        "digit_fold": bool(v5.digit_fold),
        "preshuffle": bool(v5.preshuffle),
        "synthetic_total": int(v5.synthetic_total),
    }
