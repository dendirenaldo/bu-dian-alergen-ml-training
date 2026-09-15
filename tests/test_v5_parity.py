"""Tests parity V5 (tanpa TF/Gensim): KB, merge, split, synthetic, evaluasi, audit."""

import numpy as np
import pandas as pd
import pytest


def _toy_source() -> pd.DataFrame:
    # 12 produk unik, teks unik, 4 safe / 8 unsafe — komponen 1-baris semua.
    rows = [
        ("prod_a", "gula garam air pati tapioka", "safe"),
        ("prod_b", "air garam gula perisa", "safe"),
        ("prod_c", "pati jagung minyak nabati garam", "safe"),
        ("prod_d", "gula air garam penstabil nabati", "safe"),
        ("prod_e", "tepung terigu gula garam", "unsafe"),
        ("prod_f", "susu bubuk gula garam", "unsafe"),
        ("prod_g", "telur bubuk gula garam", "unsafe"),
        ("prod_h", "lesitin kedelai gula garam", "unsafe"),
        ("prod_i", "ikan tuna gula garam", "unsafe"),
        ("prod_j", "udang gula garam", "unsafe"),
        ("prod_k", "biji wijen gula garam", "unsafe"),
        ("prod_l", "kacang mede gula garam", "unsafe"),
    ]
    return pd.DataFrame(rows, columns=["nama_produk", "text", "label"])


def test_kb_rules_dasar():
    from app.core.kb.rules import rule_based_kb_label

    assert rule_based_kb_label("susu bubuk gula")["unsafe"] is True
    assert rule_based_kb_label("susu bubuk gula")["binary_label"] == 1
    assert rule_based_kb_label("gula garam air")["unsafe"] is False
    # Non-kanonis sulfit tetap unsafe (kontrak notebook).
    assert rule_based_kb_label("pengawet sulfit garam")["unsafe"] is True
    # 8 kategori tersedia.
    from app.core.kb.rules import ALLERGEN_KB

    assert set(ALLERGEN_KB) == {
        "Gluten", "Dairy", "Egg", "Soy", "Fish",
        "Crustacean", "Sesame", "Tree Nut",
    }


def test_gold_merge_single_jadi_model_source():
    from app.core.data.gold_merge import build_model_source_from_single

    src = build_model_source_from_single(_toy_source())
    assert set(src["gold_label"]) == {"safe", "unsafe"}
    assert src["label_id"].isin([0, 1]).all()
    assert "group_key" in src.columns and "kb_binary_label" in src.columns
    assert len(src) == 12


def test_merge_ocr_gold_konflik_stop():
    from app.core.data.gold_merge import merge_ocr_with_gold

    ocr = pd.DataFrame([
        {"nama_produk": "prod a", "text": "susu gula"},
        {"nama_produk": "prod_a", "text": "susu gula"},
    ])
    gold = pd.DataFrame([
        {"Gambar": "prod a.jpg", "Label": "unsafe"},
        {"Gambar": "prod_a.jpg", "Label": "safe"},
    ])
    with pytest.raises(ValueError, match="konflik"):
        merge_ocr_with_gold(ocr, gold)


def test_split_deterministik_tanpa_leakage():
    from app.core.data.gold_merge import build_model_source_from_single
    from app.core.data.leakage_split import (
        audit_split_no_leakage,
        build_split_manifest,
        run_v5_split,
    )

    src = build_model_source_from_single(_toy_source())
    frozen = ["prod_a", "prod_e", "prod_f"]  # 1 safe + 2 unsafe
    out1 = run_v5_split(src, frozen_products=frozen,
                        holdout_safe=1, holdout_unsafe=2, val_safe=1, val_unsafe=2)
    out2 = run_v5_split(src, frozen_products=frozen,
                        holdout_safe=1, holdout_unsafe=2, val_safe=1, val_unsafe=2)
    assert len(out1["df_holdout"]) == 3
    assert len(out1["df_real_val"]) == 3
    assert len(out1["df_real_train"]) == 6
    # Deterministik: sama persis.
    assert out1["df_holdout"]["nama_produk"].tolist() == out2["df_holdout"]["nama_produk"].tolist()
    assert out1["df_real_val"]["nama_produk"].tolist() == out2["df_real_val"]["nama_produk"].tolist()
    audit = audit_split_no_leakage(out1["df_real_train"], out1["df_real_val"], out1["df_holdout"])
    assert (audit["status"] == "PASS").all()
    manifest = build_split_manifest(out1["df_real_train"], out1["df_real_val"], out1["df_holdout"])
    assert len(manifest) == 12
    assert set(manifest["split"]) == {"real_train", "real_validation", "frozen_holdout"}


def test_split_group_leakage_dicegah():
    # Dua nama berbeda -> satu group_key -> harus satu split.
    from app.core.data.gold_merge import build_model_source_from_single
    from app.core.data.leakage_split import run_v5_split

    df = pd.DataFrame([
        {"nama_produk": "my prod", "text": "susu gula garam beda1", "label": "unsafe"},
        {"nama_produk": "my_prod", "text": "telur gula garam beda2", "label": "unsafe"},
        {"nama_produk": "other one", "text": "gula garam air", "label": "safe"},
        {"nama_produk": "other two", "text": "gula air perisa", "label": "safe"},
    ])
    src = build_model_source_from_single(df)
    assert src["group_key"].nunique() == 3  # my prod + my_prod menyatu
    # Frozen hanya satu dari grup menyatu -> harus raise conflict, bukan diam-diam.
    with pytest.raises(RuntimeError, match="COMPONENT CONFLICT"):
        run_v5_split(src, frozen_products=["my prod"],
                     holdout_safe=0, holdout_unsafe=1, val_safe=1, val_unsafe=0)


def test_synthetic_hanya_train_dan_kb_konsisten():
    from app.core.data.gold_merge import build_model_source_from_single
    from app.core.data.leakage_split import run_v5_split
    from app.core.data.synthetic import build_final_training_pool, generate_synthetic_dataset

    src = build_model_source_from_single(_toy_source())
    out = run_v5_split(src, frozen_products=["prod_a", "prod_e", "prod_f"],
                       holdout_safe=1, holdout_unsafe=2, val_safe=1, val_unsafe=2)
    syn = generate_synthetic_dataset(out["df_real_train"], total_data=20, seed=42)
    assert len(syn) == 20
    assert (syn["label_id"].value_counts().to_dict() == {0: 10, 1: 10})
    assert syn["label_id"].isin([0, 1]).all()
    pool = build_final_training_pool(out["df_real_train"], syn,
                                     out["df_real_val"], out["df_holdout"])
    assert len(pool["y_train"]) == len(out["df_real_train"]) + 20
    assert len(pool["y_val"]) == len(out["df_real_val"])
    assert len(pool["y_holdout"]) == len(out["df_holdout"])


def test_final_pool_audit_menolak_overlap():
    from app.core.data.leakage_split import audit_final_pool_no_leakage

    ok = audit_final_pool_no_leakage(["susu gula"], ["telur gula"], ["ikan gula"])
    assert (ok["status"] == "PASS").all()
    with pytest.raises(RuntimeError, match="overlap"):
        audit_final_pool_no_leakage(["susu gula"], ["susu gula"], ["ikan gula"])


def test_evaluasi_threshold_fixed():
    from app.core.model.metrics import evaluate_fixed_threshold

    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    table = evaluate_fixed_threshold(y, p, y, p, threshold=0.50)
    assert list(table["split"]) == ["Validation (real gold)", "Frozen Holdout (real gold)"]
    assert float(table.loc[0, "accuracy"]) == 1.0
    with pytest.raises(ValueError, match="fixed 0.50"):
        evaluate_fixed_threshold(y, p, y, p, threshold=0.7)


def test_triage_tidak_tuning():
    from app.core.data.gold_merge import build_model_source_from_single
    from app.core.model.audit import triage_gold_kb_bilstm

    src = build_model_source_from_single(_toy_source())
    hold = src.iloc[[0, 4]].copy()  # 1 safe + 1 unsafe
    out = triage_gold_kb_bilstm(hold, np.array([0.1, 0.9]))
    assert "error_triage" in out["triage"].columns
    assert (out["triage"]["error_triage"] != "UNCLASSIFIED").all()


def test_repro_audit_holdout_sesuai_kontrak():
    import os

    from app.core.model.audit import audit_reproducibility_v5

    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    checks = audit_reproducibility_v5(holdout_size=50, expected_holdout_size=50)
    assert all(checks.values())
    with pytest.raises(AssertionError):
        audit_reproducibility_v5(holdout_size=23, expected_holdout_size=50)
    # Seed custom legal selama konsisten (PYTHONHASHSEED mengikuti seed).
    os.environ["PYTHONHASHSEED"] = "123"
    checks = audit_reproducibility_v5(seed=123, w2v_seed=123,
                                      holdout_size=50, expected_holdout_size=50)
    assert all(checks.values())
    os.environ["PYTHONHASHSEED"] = "42"
    with pytest.raises(AssertionError):
        audit_reproducibility_v5(seed=123)  # hashseed 42 != seed 123


def test_require_fixed_threshold():
    import numpy as np

    from app.core.model.metrics import (
        _require_fixed_threshold,
        bootstrap_ci_metrics,
        evaluate_fixed_threshold,
    )

    assert _require_fixed_threshold(0.5) == 0.5
    with pytest.raises(ValueError):
        _require_fixed_threshold(0.7)
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    with pytest.raises(ValueError):
        evaluate_fixed_threshold(y, p, y, p, threshold=0.7)
    with pytest.raises(ValueError):
        bootstrap_ci_metrics(y, p, threshold=0.7)


def test_write_thresholds_schema(tmp_path):
    import json

    from app.core.model.metrics import write_thresholds

    path = str(tmp_path / "thresholds.json")
    payload = write_thresholds(path, 0.5)
    assert payload == {"bilstm": 0.5, "fixed": True}
    assert "lstm" not in payload
    assert json.load(open(path)) == payload


def test_standardize_columns_variants():
    import pandas as pd

    from app.core.data.gold_merge import standardize_columns

    df = pd.DataFrame({"nama produk": ["a"], "text": ["t"], "label": ["safe"]})
    out = standardize_columns(df, product_col="nama_produk")
    assert "nama_produk" in out.columns
    df2 = pd.DataFrame({"product": ["a"], "text": ["t"], "label": ["safe"]})
    assert "nama_produk" in standardize_columns(df2).columns
    df3 = pd.DataFrame({"text": ["t"], "label": ["safe"]})
    out3 = standardize_columns(df3)
    assert list(out3["nama_produk"]) == ["produk_0"]


def test_parse_clip_norm_edge():
    import importlib
    import os

    import app.config as cfgmod

    cases = [("1.0", 1.0), ("0", None), ("none", None), ("", 1.0), ("  ", 1.0)]
    for val, expected in cases:
        os.environ["V5_GRADIENT_CLIP_NORM"] = val
        importlib.reload(cfgmod)
        assert cfgmod.Config().v5.gradient_clip_norm == expected, val
    del os.environ["V5_GRADIENT_CLIP_NORM"]
    importlib.reload(cfgmod)
    assert cfgmod.Config().v5.gradient_clip_norm == 1.0
    os.environ["V5_GRADIENT_CLIP_NORM"] = "abc"
    try:
        with pytest.raises(ValueError):
            importlib.reload(cfgmod)
            cfgmod.Config()
    finally:
        del os.environ["V5_GRADIENT_CLIP_NORM"]
        importlib.reload(cfgmod)


def test_config_defaults_final():
    from app.config import Config

    c = Config()
    assert not hasattr(c, "mode") and not hasattr(c, "model_type")
    # Konfigurasi FINAL terbaik (Fase 2-4):
    # split 50 (25/25), LR 1e-4 (D2), clip 1.0 (D1), mask (D3).
    assert c.v5.holdout_size == 50
    assert (c.v5.holdout_safe, c.v5.holdout_unsafe) == (25, 25)
    assert (c.v5.val_safe, c.v5.val_unsafe) == (25, 25)
    assert c.v5.synthetic_total == 1000
    assert c.v5.fixed_threshold == 0.50
    assert c.v5.learning_rate == 1e-4
    assert c.v5.gradient_clip_norm == 1.0
    assert c.v5.mask_zero is True
    assert c.v5.embed_trainable is True
    assert c.v5.train_shuffle is False
