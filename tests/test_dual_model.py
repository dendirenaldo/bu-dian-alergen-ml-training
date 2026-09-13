"""Regression tests untuk fix BiLSTM + modul dual-model BERT.

- Tanpa TF/transformers: selalu jalan (bert normalize, threshold, metrics).
- Butuh TF/transformers: di-skip otomatis via pytest.importorskip.
"""

import numpy as np
import pytest


def test_bert_normalize_ringan():
    """BERT tidak boleh membuang stopword/tanda baca secara agresif."""
    from app.core.model.bert.dataset import normalize_for_bert, prepare_bert_texts

    assert normalize_for_bert("Komposisi:\n  Susu,  Telur\tKacang") == (
        "Komposisi: Susu, Telur Kacang"
    )
    # Stopword dipertahankan (beda dengan cleanse BiLSTM).
    assert "dan" in prepare_bert_texts(["susu dan telur"])[0]


def test_threshold_tuning_memilih_f1_terbaik():
    from app.core.model.evaluator import tune_threshold

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    thr, score = tune_threshold(y_true, y_prob)
    assert 0.05 <= thr <= 0.95
    assert score >= 0.8


def test_metrics_single_class_tidak_crash():
    """FIX CRITICAL: single-class tidak boleh crash / inflasi 1.0 palsu."""
    from app.core.model.evaluator import compute_binary_metrics

    m = compute_binary_metrics(
        np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])
    )
    assert m["precision"] == 0.0  # dulu zero_division=1 → 1.0 palsu
    assert m["recall"] == 0.0
    assert np.isnan(m["roc_auc"])


def test_metrics_threshold_berpengaruh():
    from app.core.model.evaluator import compute_binary_metrics

    y = np.array([0, 1, 0, 1])
    p = np.array([0.4, 0.6, 0.3, 0.7])
    low = compute_binary_metrics(y, p, threshold=0.9)
    mid = compute_binary_metrics(y, p, threshold=0.5)
    assert low["recall"] <= mid["recall"]


def test_tokenizer_oob_dicap():
    """FIX CRITICAL: indeks >= vocab_size harus dipetakan ke OOV (1)."""
    tf = pytest.importorskip("tensorflow")
    from app.core.model.tokenizer import Tokenizer

    t = Tokenizer(vocab_size=10, max_len=8)
    t.fit(["susu telur kacang gandum susu telur"])
    seqs = t.texts_to_sequences(["kata sangat jarang xyz abcdef qwerty"])
    assert all(i < 10 for s in seqs for i in s)


def test_tokenizer_json_roundtrip(tmp_path):
    pytest.importorskip("tensorflow")
    from app.core.model.tokenizer import Tokenizer

    t = Tokenizer(vocab_size=50, max_len=8)
    t.fit(["susu telur kacang"])
    p = str(tmp_path / "tok.json")
    t.save_json(p)
    t2 = Tokenizer.load_json(p)
    assert t2.word_index == t.word_index
    assert t2.vocab_size == 50 and t2.max_len == 8


def test_config_seed_dari_env(monkeypatch):
    dotenv = pytest.importorskip("dotenv")
    monkeypatch.setenv("SEED", "123")
    monkeypatch.setenv("MODEL_TYPE", "bert")
    monkeypatch.setenv("BERT_MODEL_NAME", "bert-base-multilingual-cased")
    from app.config import Config

    c = Config()
    assert c.seed == 123  # dulu dibaca saat import → env telat
    assert c.model_type == "bert"
    assert c.bert.model_name == "bert-base-multilingual-cased"


def test_model_factory_pluggable():
    pytest.importorskip("transformers")
    from app.core.model import model_factory

    assert "indobenchmark/indobert-base-p1" in model_factory.list_supported_models()
