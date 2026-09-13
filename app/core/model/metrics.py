"""Evaluasi V5 — port Notebook Cell 47/49/55/57.

Threshold FIXED (tidak di-tuning). Metrik: accuracy/precision/recall/F1/
ROC-AUC/AP + report + confusion matrix + kurva ROC/PR + kurva training
+ manifest. Tanpa import TF di top-level (model.predict dioper dari luar).
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def oov_stats_for_sequences(sequences: list[list[int]], oov_id: int = 1) -> dict:
    """Statistik OOV untuk satu split (port Cell 43)."""
    total = sum(len(s) for s in sequences)
    unk = sum(1 for s in sequences for t in s if t == oov_id)
    return {
        "total_tokens": total,
        "unk_tokens": unk,
        "unk_rate": (unk / total) if total else 0.0,
    }


def _add_f1_columns(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    if {"precision", "recall"}.issubset(hist.columns):
        hist["f1"] = 2 * hist["precision"] * hist["recall"] / (
            hist["precision"] + hist["recall"] + 1e-8
        )
    if {"val_precision", "val_recall"}.issubset(hist.columns):
        hist["val_f1"] = 2 * hist["val_precision"] * hist["val_recall"] / (
            hist["val_precision"] + hist["val_recall"] + 1e-8
        )
    return hist


def save_training_curves_v5(
    history: object, output_dir: str, prefix: str = "bilstm_word2vec_leakage_safe"
) -> pd.DataFrame:
    """Simpan kurva + CSV training (port Cell 47). Terima History atau dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = history.history if hasattr(history, "history") else dict(history)
    hist = _add_f1_columns(pd.DataFrame(raw))
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(hist["accuracy"], linewidth=2, label="Train Accuracy")
    if "val_accuracy" in hist.columns:
        plt.plot(hist["val_accuracy"], linewidth=2, label="Validation Accuracy")
    # Judul figure dihapus: caption ditulis di dokumen Word.
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_accuracy_{prefix}.png"))
    plt.close()

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(hist["loss"], linewidth=2, label="Train Loss")
    if "val_loss" in hist.columns:
        plt.plot(hist["val_loss"], linewidth=2, label="Validation Loss")
    # Judul figure dihapus: caption ditulis di dokumen Word.
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_loss_{prefix}.png"))
    plt.close()

    plt.figure(figsize=(10, 6), dpi=120)
    for col, label in [
        ("precision", "Train Precision"),
        ("val_precision", "Validation Precision"),
        ("recall", "Train Recall"),
        ("val_recall", "Validation Recall"),
        ("f1", "Train F1"),
        ("val_f1", "Validation F1"),
    ]:
        if col in hist.columns:
            plt.plot(hist[col], linewidth=2, label=label)
    # Judul figure dihapus: caption ditulis di dokumen Word.
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_prf1_{prefix}.png"))
    plt.close()

    hist.to_csv(os.path.join(output_dir, f"training_history_{prefix}.csv"), index=False)

    best_epoch = int(hist["val_loss"].idxmin()) + 1 if "val_loss" in hist.columns else 1
    best_row = hist.loc[best_epoch - 1]
    summary = pd.DataFrame([{
        "best_epoch": best_epoch,
        "best_val_loss": float(best_row.get("val_loss", float("nan"))),
        "train_accuracy_at_best": float(best_row.get("accuracy", float("nan"))),
        "val_accuracy_at_best": float(best_row.get("val_accuracy", float("nan"))),
        "train_f1_at_best": float(best_row.get("f1", float("nan"))),
        "val_f1_at_best": float(best_row.get("val_f1", float("nan"))),
    }])
    summary.to_csv(os.path.join(output_dir, f"training_summary_{prefix}.csv"), index=False)
    logger.info("V5 training curves tersimpan (best_epoch=%d)", best_epoch)
    return summary


def evaluate_fixed_threshold(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_holdout: np.ndarray,
    holdout_prob: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Tabel evaluasi val + holdout pada threshold fixed (port Cell 49/57)."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if float(threshold) != 0.50:
        raise ValueError(f"V5 threshold harus fixed 0.50, dapat {threshold}.")

    def row(name: str, y_true: np.ndarray, prob: np.ndarray) -> dict:
        y_true = np.asarray(y_true).astype(int)
        prob = np.asarray(prob).astype(float)
        pred = (prob >= threshold).astype(int)
        single = len(np.unique(y_true)) < 2
        return {
            "split": name,
            "n": len(y_true),
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "roc_auc": float("nan") if single else float(roc_auc_score(y_true, prob)),
            "average_precision": (
                float("nan") if single else float(average_precision_score(y_true, prob))
            ),
            "threshold": float(threshold),
        }

    return pd.DataFrame([
        row("Validation (real gold)", y_val, val_prob),
        row("Frozen Holdout (real gold)", y_holdout, holdout_prob),
    ])


def plot_roc_pr_v5(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_holdout: np.ndarray,
    holdout_prob: np.ndarray,
    output_dir: str,
    prefix: str = "v5",
) -> None:
    """Kurva ROC + PR val vs holdout (port Cell 55). Skip bila single-class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    os.makedirs(output_dir, exist_ok=True)
    if len(np.unique(np.asarray(y_val))) == 2 and len(np.unique(np.asarray(y_holdout))) == 2:
        fpr_v, tpr_v, _ = roc_curve(y_val, val_prob)
        fpr_h, tpr_h, _ = roc_curve(y_holdout, holdout_prob)
        auc_v, auc_h = roc_auc_score(y_val, val_prob), roc_auc_score(y_holdout, holdout_prob)
        plt.figure(figsize=(8, 6), dpi=120)
        plt.plot(fpr_v, tpr_v, linewidth=2, label=f"Validation ROC-AUC = {auc_v:.4f}")
        plt.plot(fpr_h, tpr_h, linewidth=2, label=f"Frozen Holdout ROC-AUC = {auc_h:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
        # Judul figure dihapus: caption ditulis di dokumen Word.
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"roc_curve_{prefix}.png"))
        plt.close()

        p_v, r_v, _ = precision_recall_curve(y_val, val_prob)
        p_h, r_h, _ = precision_recall_curve(y_holdout, holdout_prob)
        ap_v = average_precision_score(y_val, val_prob)
        ap_h = average_precision_score(y_holdout, holdout_prob)
        plt.figure(figsize=(8, 6), dpi=120)
        plt.plot(r_v, p_v, linewidth=2, label=f"Validation AP = {ap_v:.4f}")
        plt.plot(r_h, p_h, linewidth=2, label=f"Frozen Holdout AP = {ap_h:.4f}")
        # Judul figure dihapus: caption ditulis di dokumen Word.
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pr_curve_{prefix}.png"))
        plt.close()
    else:
        logger.warning("ROC/PR di-skip: salah satu split single-class.")


def plot_confusion_v5(
    y_true: np.ndarray, y_prob: np.ndarray, output_dir: str,
    name: str = "frozen_holdout", threshold: float = 0.50,
) -> pd.DataFrame:
    """Confusion matrix + CSV (port Cell 53)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        _has_sns = True
    except ImportError:
        _has_sns = False
    from sklearn.metrics import confusion_matrix

    os.makedirs(output_dir, exist_ok=True)
    pred = (np.asarray(y_prob).astype(float) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), pred, labels=[0, 1])
    plt.figure(figsize=(6, 5), dpi=120)
    if _has_sns:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["safe", "unsafe"], yticklabels=["safe", "unsafe"])
    else:
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks([0, 1], ["safe", "unsafe"])
        plt.yticks([0, 1], ["safe", "unsafe"])
        plt.colorbar()
    # Judul figure dihapus: caption ditulis di dokumen Word.
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"))
    plt.close()
    table = pd.DataFrame(
        cm,
        index=["Actual safe", "Actual unsafe"],
        columns=["Predicted safe", "Predicted unsafe"],
    )
    table.to_csv(os.path.join(output_dir, f"confusion_matrix_{name}.csv"), index=True)
    return table


def write_experiment_manifest(
    output_dir: str,
    seed: int = 42,
    threshold: float = 0.50,
    holdout_size: int = 23,
    extra: dict | None = None,
) -> str:
    """Manifest eksperimen (port Cell 64)."""
    manifest = {
        "experiment": "Experiment 1 — Reproducible Leakage-Safe BiLSTM (port)",
        "seed": int(seed),
        "threshold": float(threshold),
        "holdout_size": int(holdout_size),
        "holdout_frozen": True,
        "gold_label_is_ground_truth": True,
        "synthetic_after_real_split": True,
        "synthetic_source": "real_train_only",
        "tokenizer_fit_source": "final_training_pool_only",
        "word2vec_workers": 1,
        "architecture_unchanged": True,
        "required_audits": [
            "group leakage",
            "exact OCR text overlap",
            "split manifest",
            "validation curves",
            "frozen holdout curves",
            "ROC-AUC",
            "precision-recall",
            "confusion matrix",
        ],
    }
    if extra:
        manifest.update(extra)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "experiment_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def bootstrap_ci_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Interval kepercayaan bootstrap untuk metrik biner (laporan skripsi).

    Stratified resampling (pertahankan proporsi kelas) agar stabil
    pada split kecil. Tanpa TF.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.RandomState(seed)
    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]
    alpha = 1.0 - ci
    stats: dict[str, list[float]] = {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [],
    }
    for _ in range(n_bootstrap):
        s0 = rng.choice(idx0, size=len(idx0), replace=True) if len(idx0) else np.array([], dtype=int)
        s1 = rng.choice(idx1, size=len(idx1), replace=True) if len(idx1) else np.array([], dtype=int)
        ii = np.concatenate([s0, s1])
        yt, yp = y_true[ii], y_prob[ii]
        pred = (yp >= threshold).astype(int)
        stats["accuracy"].append(float(accuracy_score(yt, pred)))
        stats["precision"].append(float(precision_score(yt, pred, zero_division=0)))
        stats["recall"].append(float(recall_score(yt, pred, zero_division=0)))
        stats["f1"].append(float(f1_score(yt, pred, zero_division=0)))
        stats["roc_auc"].append(float(roc_auc_score(yt, yp)))
    rows = []
    for metric, vals in stats.items():
        lo, hi = float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
        rows.append({"metric": metric, "mean": float(np.mean(vals)), "lo": lo, "hi": hi,
                     "ci": ci, "n_bootstrap": n_bootstrap})
    return pd.DataFrame(rows)


def write_thresholds(path: str, threshold: float = 0.5) -> dict:
    """Tulis thresholds.json kanonis tunggal: {"bilstm": thr, "fixed": True}."""
    import json as _json
    import os as _os

    payload = {"bilstm": float(threshold), "fixed": True}
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2)
    return payload
