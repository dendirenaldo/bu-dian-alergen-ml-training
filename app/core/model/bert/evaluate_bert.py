"""Evaluasi direktori model BERT tersimpan (untuk scripts/evaluate.py --model-bert)."""

from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_bert_dir(
    model_dir: str,
    X_test_text: list[str],
    y_test,
    label_encoder,
    output_dir: str,
) -> tuple[dict, np.ndarray, np.ndarray, float]:
    """Load model BERT dari direktori HF dan evaluasi di test set.

    Returns:
        (metrics_dict, fpr, tpr, auc).
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers/torch belum terinstal untuk evaluasi BERT."
        ) from e

    from sklearn.metrics import roc_curve

    from app.core.model.bert.dataset import prepare_bert_texts
    from app.core.model.evaluator import compute_binary_metrics

    thr_path = os.path.join(model_dir, "threshold.json")
    threshold = 0.5
    if os.path.exists(thr_path):
        with open(thr_path, encoding="utf-8") as f:
            threshold = float(json.load(f).get("threshold", 0.5))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = prepare_bert_texts(list(X_test_text))
    enc = tokenizer(texts, truncation=True, padding=True, max_length=512,
                    return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy().ravel()
    probs = (1 / (1 + np.exp(-logits))).astype(float)

    metrics = compute_binary_metrics(np.asarray(y_test).astype(int), probs, threshold)
    try:
        fpr, tpr, _ = roc_curve(np.asarray(y_test).astype(int), probs)
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    auc = metrics.get("roc_auc", float("nan"))

    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "evaluation_table_bert.csv"), index=False
    )
    logger.info("BERT eval metrics: %s", metrics)
    return metrics, fpr, tpr, auc
