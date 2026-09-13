"""Factory dual-model: sembunyikan perbedaan BiLSTM vs BERT.

- BiLSTM: Embedding(Word2Vec, frozen) + 2x LSTM + Dense sigmoid, input = indeks kata.
- BERT: AutoModelForSequenceClassification + BCE loss, input = input_ids/attention_mask.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_bilstm(*args, **kwargs):
    """Bangun model BiLSTM (delegasi ke architecture.build_model)."""
    from app.core.model.architecture import build_model

    return build_model(use_bidirectional=True, *args, **kwargs)


def build_lstm(*args, **kwargs):
    """Bangun model LSTM (delegasi ke architecture.build_model)."""
    from app.core.model.architecture import build_model

    return build_model(use_bidirectional=False, *args, **kwargs)


def build_bert_classifier(model_name: str, num_labels: int = 1, dropout: float = 0.1):
    """Bangun classifier BERT (lazy import transformers agar BiLSTM tetap ringan).

    Args:
        model_name: Checkpoint HF apa pun (IndoBERT, mBERT, XLM-R, Distil...).
        num_labels: 1 untuk biner sigmoid (BCEWithLogits).
        dropout: Dropout classifier head.

    Raises:
        ImportError: Bila transformers/torch belum terinstal.
    """
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification
    except ImportError as e:
        raise ImportError(
            "transformers belum terinstal. Jalankan: pip install -r requirements.txt"
        ) from e
    cfg = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels,
        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
    )
    # problem_type single_label_classification + num_labels=1 → BCEWithLogitsLoss.
    if num_labels == 1:
        cfg.problem_type = "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
    logger.info("BERT classifier dibangun: %s (num_labels=%d)", model_name, num_labels)
    return model


def list_supported_models() -> list[str]:
    """Daftar checkpoint yang umum dipakai (bebas diisi nama HF lain via --bert-model)."""
    return [
        "indobenchmark/indobert-base-p1",  # default: Bahasa Indonesia
        "indobenchmark/indobert-base-p2",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
    ]
