"""Model evaluation utilities."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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

    if {"precision", "recall"}.issubset(hist.columns):
        hist["f1"] = 2 * (hist["precision"] * hist["recall"]) / (
            hist["precision"] + hist["recall"] + 1e-8
        )
    if {"val_precision", "val_recall"}.issubset(hist.columns):
        hist["val_f1"] = 2 * (hist["val_precision"] * hist["val_recall"]) / (
            hist["val_precision"] + hist["val_recall"] + 1e-8
        )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(hist["accuracy"], label="train_accuracy")
    if "val_accuracy" in hist.columns:
        axes[0].plot(hist["val_accuracy"], label="val_accuracy")
    if "f1" in hist.columns:
        axes[0].plot(hist["f1"], label="train_f1")
    if "val_f1" in hist.columns:
        axes[0].plot(hist["val_f1"], label="val_f1")
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

    axes[2].plot(hist["precision"], label="train_precision")
    if "val_precision" in hist.columns:
        axes[2].plot(hist["val_precision"], label="val_precision")
    axes[2].plot(hist["recall"], label="train_recall")
    if "val_recall" in hist.columns:
        axes[2].plot(hist["val_recall"], label="val_recall")
    axes[2].set_title(f"{model_name} - Precision/Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_history_{model_name.lower()}_word2vec.png"), dpi=150)
    plt.close()

    history_csv = os.path.join(
        output_dir, f"training_history_{model_name.lower()}_word2vec.csv"
    )
    hist.to_csv(history_csv, index=False)
    logger.info("Training history (%s) saved to: %s", model_name, history_csv)


def evaluate_model(
    model: object,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_text: list[str],
    label_encoder: object,
    output_dir: str,
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
    logger.info("Evaluating: %s", model_name)

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=1
    )
    auc = roc_auc_score(y_test, y_prob)

    eval_table = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1_score", "value": f1},
            {"metric": "roc_auc", "value": auc},
        ]
    )

    target_names = list(label_encoder.classes_)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=1
    )
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )

    # Confusion Matrix + ROC side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
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
            "label_actual": label_encoder.inverse_transform(y_test),
            "label_pred": label_encoder.inverse_transform(y_pred),
            "score_unsafe": y_prob,
        }
    )

    # Save outputs
    eval_bilstm_path = os.path.join(
        output_dir, f"evaluation_table_{model_name.lower()}_word2vec"
    )
    eval_table.to_csv(eval_bilstm_path + ".csv", index=False)
    eval_table.to_excel(eval_bilstm_path + ".xlsx", index=False)
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
    fpr_lstm: np.ndarray,
    tpr_lstm: np.ndarray,
    auc_lstm: float,
    output_dir: str,
) -> None:
    """Plot ROC curve comparison between BiLSTM and LSTM.

    Args:
        fpr_bilstm: False positive rates for BiLSTM.
        tpr_bilstm: True positive rates for BiLSTM.
        auc_bilstm: AUC score for BiLSTM.
        fpr_lstm: False positive rates for LSTM.
        tpr_lstm: True positive rates for LSTM.
        auc_lstm: AUC score for LSTM.
        output_dir: Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_bilstm,
        tpr_bilstm,
        label=f"BiLSTM (AUC = {auc_bilstm:.4f})",
        linewidth=2,
    )
    plt.plot(
        fpr_lstm,
        tpr_lstm,
        label=f"LSTM (AUC = {auc_lstm:.4f})",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: BiLSTM vs LSTM")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_comparison_word2vec.png"), dpi=150)
    plt.close()
    logger.info("ROC comparison plot saved")
