"""Training BERT dengan HuggingFace PyTorch (Trainer API + AdamW + warmup).

Pluggable: ``config.bert.model_name`` bisa diisi checkpoint HF apa pun
(IndoBERT default, mBERT, XLM-R, Distil, ...) tanpa ubah kode.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _require_hf():
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Dependensi BERT belum terinstal. Jalankan: "
            "pip install -r requirements.txt (torch, transformers, datasets, accelerate)."
        ) from e


def run_bert_training(
    config,
    X_train_text: list[str],
    X_test_text: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
) -> dict:
    """Latih classifier BERT di atas split yang SAMA dengan BiLSTM.

    Args:
        config: Config (pakai config.bert.*).
        X_train_text/X_test_text: Teks per split (indeks identik dengan BiLSTM).
        y_train/y_test: Label id (0/1).
        label_encoder: LabelEncoder ter-fit (dipakai untuk mapping & laporan).

    Returns:
        dict {model_name, metrics, threshold, roc_curve, output_dir, best_params}.
    """
    _require_hf()
    import torch
    from datasets import Dataset
    from sklearn.model_selection import StratifiedShuffleSplit
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from app.core.model.bert.dataset import prepare_bert_texts
    from app.core.model.evaluator import (
        compute_binary_metrics,
        tune_threshold,
    )

    bert_cfg = config.bert
    model_name = bert_cfg.model_name
    logger.info("Tokenizing BERT: %s (max_len=%d)", model_name, bert_cfg.max_len)

    # Seed PyTorch untuk reproduksibilitas.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    X_train_n = prepare_bert_texts(X_train_text)
    X_test_n = prepare_bert_texts(X_test_text)

    # Val stratified dari train (untuk early-stop + threshold-tuning, bukan test).
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=config.seed)
    try:
        tr_idx, va_idx = next(sss.split(X_train_n, np.asarray(y_train)))
    except ValueError:
        n = len(X_train_n)
        n_va = max(1, int(n * 0.15))
        tr_idx, va_idx = np.arange(n - n_va), np.arange(n - n_va, n)

    def _tok(texts):
        return tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=bert_cfg.max_len,
        )

    train_enc = _tok([X_train_n[i] for i in tr_idx])
    val_enc = _tok([X_train_n[i] for i in va_idx])
    test_enc = _tok(X_test_n)
    y_tr = np.asarray(y_train)[tr_idx].astype(float)
    y_va = np.asarray(y_train)[va_idx].astype(float)

    train_ds = Dataset.from_dict({**train_enc, "labels": y_tr.tolist()})
    val_ds = Dataset.from_dict({**val_enc, "labels": y_va.tolist()})
    test_ds = Dataset.from_dict({**test_enc, "labels": np.asarray(y_test).astype(float).tolist()})

    # pos_weight untuk imbalance (BCEWithLogitsLoss).
    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = (n_neg / max(1.0, n_pos)) if n_pos > 0 else 1.0
    logger.info("BERT pos_weight=%.3f (neg=%.0f, pos=%.0f)", pos_weight, n_neg, n_pos)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1,
        hidden_dropout_prob=bert_cfg.dropout,
        attention_probs_dropout_prob=bert_cfg.dropout,
    )
    if getattr(bert_cfg, "freeze_layers", 0):
        n_freeze = int(bert_cfg.freeze_layers)
        base = getattr(model, "bert", None) or getattr(model, "roberta", None)
        if base is not None and hasattr(base, "encoder"):
            for layer in base.encoder.layer[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False
            logger.info("Freeze %d encoder layer terbawah", n_freeze)

    out_dir = os.path.join(config.output_dir, "bert")
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=bert_cfg.epochs,
        per_device_train_batch_size=bert_cfg.batch_size,
        per_device_eval_batch_size=max(8, bert_cfg.batch_size * 2),
        learning_rate=bert_cfg.learning_rate,
        weight_decay=bert_cfg.weight_decay,
        warmup_ratio=bert_cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-np.asarray(logits).ravel()))
        m = compute_binary_metrics(np.asarray(labels).astype(int), probs, threshold=0.5)
        return {"accuracy": m["accuracy"], "f1": m["f1_score"], "auc": m["roc_auc"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # Threshold-tuning di VAL (bukan test).
    va_out = trainer.predict(val_ds)
    va_prob = (1 / (1 + np.exp(-np.asarray(va_out.predictions).ravel()))).astype(float)
    threshold, _ = tune_threshold(np.asarray(y_va).astype(int), va_prob, metric="f1")
    threshold = float(np.clip(threshold, 0.05, 0.95))
    logger.info("BERT threshold (val-F1) = %.3f", threshold)

    # Evaluasi final di TEST identik dengan BiLSTM.
    test_out = trainer.predict(test_ds)
    test_prob = (1 / (1 + np.exp(-np.asarray(test_out.predictions).ravel()))).astype(float)
    metrics = compute_binary_metrics(np.asarray(y_test).astype(int), test_prob, threshold=threshold)

    # Simpan model + tokenizer + artefak dual-model.
    save_dir = os.path.join(config.model_dir, "bert")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, **metrics}, f, indent=2)
    with open(os.path.join(config.output_dir, "bert_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, **metrics}, f, indent=2)
    logger.info("BERT metrics (test): %s", metrics)

    # ROC curve untuk plot perbandingan N-model.
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, _ = roc_curve(np.asarray(y_test).astype(int), test_prob)
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    auc = metrics.get("roc_auc", float("nan"))

    return {
        "model_name": model_name,
        "metrics": metrics,
        "threshold": threshold,
        "roc_curve": (fpr, tpr, auc),
        "save_dir": save_dir,
        "best_params": {
            "lr": bert_cfg.learning_rate,
            "epochs": bert_cfg.epochs,
            "batch_size": bert_cfg.batch_size,
        },
    }
