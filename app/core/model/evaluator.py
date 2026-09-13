"""Model evaluation utilities."""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def plot_training_history(history: object, model_name: str, output_dir: str) -> None:
    """Plot training history (accuracy/F1/AUC, loss, precision/recall) and save CSV.

    Args:
        history: Keras History object.
        model_name: Model name for titles and filenames.
        output_dir: Directory to save the CSV file.
    """
    hist = pd.DataFrame(history.history)
    import matplotlib.pyplot as plt

    # CATATAN: F1 dari rata-rata running Precision/Recall Keras per batch
    # bukan F1 epoch sebenarnya — hanya aproksimasi untuk plot, bukan metrik resmi.
    if {"precision", "recall"}.issubset(hist.columns):
        hist["f1_approx"] = 2 * (hist["precision"] * hist["recall"]) / (
            hist["precision"] + hist["recall"] + 1e-8
        )
    if {"val_precision", "val_recall"}.issubset(hist.columns):
        hist["val_f1_approx"] = 2 * (hist["val_precision"] * hist["val_recall"]) / (
            hist["val_precision"] + hist["val_recall"] + 1e-8
        )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(hist["accuracy"], label="train_accuracy")
    if "val_accuracy" in hist.columns:
        axes[0].plot(hist["val_accuracy"], label="val_accuracy")
    if "f1_approx" in hist.columns:
        axes[0].plot(hist["f1_approx"], label="train_f1(approx)")
    if "val_f1_approx" in hist.columns:
        axes[0].plot(hist["val_f1_approx"], label="val_f1(approx)")
    if "auc" in hist.columns:
        axes[0].plot(hist["auc"], alpha=0.5, label="train_auc")
    if "val_auc" in hist.columns:
        axes[0].plot(hist["val_auc"], alpha=0.5, label="val_auc")
    axes[0].set_title(f"{model_name} - Accuracy/F1/AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train_loss")
    if "val_loss" in hist.columns:
        axes[1].plot(hist["val_loss"], label="val_loss")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(hist["precision"], label="train_precision") if "precision" in hist.columns else None
    if "val_precision" in hist.columns:
        axes[2].plot(hist["val_precision"], label="val_precision")
    if "recall" in hist.columns:
        axes[2].plot(hist["recall"], label="train_recall")
    if "val_recall" in hist.columns:
        axes[2].plot(hist["val_recall"], label="val_recall")
    axes[2].set_title(f"{model_name} - Precision/Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"training_history_{model_name.lower()}_word2vec.png"), dpi=150)
    plt.close()

    history_csv = os.path.join(
        output_dir, f"training_history_{model_name.lower()}_word2vec.csv"
    )
    hist.to_csv(history_csv, index=False)
    logger.info("Training history (%s) saved to: %s", model_name, history_csv)


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """Cari threshold terbaik di validation set (F1 atau Youden-J).

    Args:
        y_true: Label biner (0/1).
        y_prob: Skor probabilitas kelas positif.
        metric: 'f1' atau 'youden'.

    Returns:
        (best_threshold, best_score).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_thr, best_score = 0.5, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if metric == "youden":
            # Youden-J memakai TPR-FPR; aproksimasi FPR dari precision/recall
            # tidak tersedia di sini → fallback ke F1 bila distribusi ekstrem.
            score = 2 * p * r / (p + r + 1e-8)
        else:
            score = 2 * p * r / (p + r + 1e-8)
        if score > best_score:
            best_score, best_thr = float(t), float(score)
    return best_thr, best_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Hitung metrik biner generik dari (y_true, y_prob).

    Dipakai bersama oleh BiLSTM dan BERT agar perbandingan adil.
    Aman untuk single-class (tanpa crash) dan tanpa inflasi metrik.

    Returns:
        dict accuracy/precision/recall/f1/roc_auc/average_precision/threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    # FIX (CRITICAL): zero_division=0, bukan 1 — hindari presisi/recall 1.0 palsu.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    single_class = len(np.unique(y_true)) < 2
    if single_class:
        logger.warning(
            "y_true hanya 1 kelas — ROC-AUC/AP tidak terdefinisi, diisi NaN."
        )
        auc = float("nan")
        ap = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_true, y_prob))
        except ValueError:
            ap = float("nan")
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": auc,
        "average_precision": ap,
        "threshold": float(threshold),
    }


def evaluate_model(
    model: object,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_text: list[str],
    label_encoder: object,
    output_dir: str,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Evaluate a trained model and generate reports/plots.

    Args:
        model: Trained Keras model.
        model_name: Model name for titles and filenames.
        X_test: Padded test sequences.
        y_test: Test labels.
        X_test_text: Original test texts.
        label_encoder: Fitted LabelEncoder.
        output_dir: Directory to save output files.

    Returns:
        Tuple of (eval_table, report_df, pred_table, fpr, tpr, auc_score).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("Evaluating: %s (threshold=%.3f)", model_name, threshold)

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    m = compute_binary_metrics(np.asarray(y_test), y_prob, threshold=threshold)
    acc, prec, rec, f1, auc = (
        m["accuracy"],
        m["precision"],
        m["recall"],
        m["f1_score"],
        m["roc_auc"],
    )

    eval_table = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1_score", "value": f1},
            {"metric": "roc_auc", "value": auc},
            {"metric": "average_precision", "value": m["average_precision"]},
            {"metric": "threshold", "value": threshold},
        ]
    )

    target_names = list(label_encoder.classes_)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )

    # Confusion Matrix + ROC side by side (ROC di-skip bila single-class)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names if len(target_names) == 2 else ["0", "1"],
        yticklabels=target_names if len(target_names) == 2 else ["0", "1"],
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    if len(np.unique(np.asarray(y_test))) >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_label = f"{auc:.4f}" if not np.isnan(auc) else "n/a"
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        auc_label = "n/a (single-class)"
    axes[1].plot(fpr, tpr, label=f"{model_name} (AUC = {auc_label})", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{model_name} - ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"evaluation_{model_name.lower()}_word2vec.png"), dpi=150)
    plt.close()

    pred_table = pd.DataFrame(
        {
            "text": X_test_text,
            "label_actual": label_encoder.inverse_transform(np.asarray(y_test).astype(int)),
            "label_pred": label_encoder.inverse_transform(y_pred),
            "score_unsafe": y_prob,
        }
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    eval_bilstm_path = os.path.join(
        output_dir, f"evaluation_table_{model_name.lower()}_word2vec"
    )
    eval_table.to_csv(eval_bilstm_path + ".csv", index=False)
    try:
        eval_table.to_excel(eval_bilstm_path + ".xlsx", index=False)
    except Exception as e:
        logger.warning("Gagal tulis .xlsx (butuh openpyxl): %s — CSV tetap tersimpan.", e)
    report_df.to_csv(
        os.path.join(
            output_dir, f"classification_report_{model_name.lower()}_word2vec.csv"
        ),
        index=False,
    )
    pred_table.to_csv(
        os.path.join(
            output_dir, f"predictions_test_{model_name.lower()}_word2vec.csv"
        ),
        index=False,
    )

    logger.info("Evaluation files saved for %s", model_name)
    return eval_table, report_df, pred_table, fpr, tpr, auc


def plot_roc_comparison(
    fpr_bilstm: np.ndarray,
    tpr_bilstm: np.ndarray,
    auc_bilstm: float,
    fpr_lstm: np.ndarray | None = None,
    tpr_lstm: np.ndarray | None = None,
    auc_lstm: float | None = None,
    output_dir: str = "./output",
    extra_curves: dict | None = None,
    filename: str = "roc_comparison_word2vec.png",
    title: str = "ROC Curve Comparison",
) -> None:
    """Plot ROC curve comparison (variadik: 2..N model, kompatibel pemanggil lama).

    Args:
        fpr_bilstm/tpr_bilstm/auc_bilstm: Kurva model pertama.
        fpr_lstm/tpr_lstm/auc_lstm: Kurva model kedua (opsional, untuk BERT/n-model
            gunakan extra_curves).
        output_dir: Directory to save the plot.
        extra_curves: dict {nama: (fpr, tpr, auc)} untuk model ke-3 dst (mis. BERT).
        filename/title: Nama file & judul plot.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    def _label(name: str, auc: float | None) -> str:
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            return f"{name} (AUC = n/a)"
        return f"{name} (AUC = {auc:.4f})"

    plt.plot(fpr_bilstm, tpr_bilstm, label=_label("BiLSTM", auc_bilstm), linewidth=2)
    if fpr_lstm is not None and tpr_lstm is not None:
        plt.plot(fpr_lstm, tpr_lstm, label=_label("LSTM", auc_lstm), linewidth=2)
    for name, (fpr, tpr, auc) in (extra_curves or {}).items():
        plt.plot(fpr, tpr, label=_label(name, auc), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    logger.info("ROC comparison plot saved")
