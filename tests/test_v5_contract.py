"""Kontrak batch-2: threshold tulis, manifest anti-tamper, frozen hash,
audit pipeline/final-config diperketat. Tanpa TF (kecuali ditandai)."""

import json

import numpy as np
import pandas as pd
import pytest


# ---------- Threshold & manifest ----------

def test_write_thresholds_tolak_non_05(tmp_path):
    from app.core.model.metrics import write_thresholds

    path = str(tmp_path / "thresholds.json")
    with pytest.raises(ValueError, match="0.5"):
        write_thresholds(path, 0.7)
    assert not (tmp_path / "thresholds.json").exists(), "file tidak boleh tertulis"
    payload = write_thresholds(path, 0.5)
    assert payload == {"bilstm": 0.5, "fixed": True}


def test_manifest_validasi_threshold_dan_kunci_inti(tmp_path):
    from app.core.model.metrics import write_experiment_manifest

    out = str(tmp_path)
    path = write_experiment_manifest(out, threshold=0.5, holdout_size=50)
    data = json.load(open(path, encoding="utf-8"))
    assert data["threshold"] == 0.5
    assert data["holdout_size"] == 50  # default terkunci 50, bukan 23 notebook
    with pytest.raises(ValueError, match="0.5"):
        write_experiment_manifest(out, threshold=0.7)
    with pytest.raises(ValueError, match="key inti"):
        write_experiment_manifest(out, extra={"threshold": 0.7})
    with pytest.raises(ValueError, match="key inti"):
        write_experiment_manifest(out, extra={"seed": 0})


# ---------- Frozen holdout bersama ----------

def test_frozen_sha_tahan_kapitalisasi():
    from app.core.data.frozen import frozen_list_sha256

    a = frozen_list_sha256(["Bakso Sapi Karawaci", "alba cheese", "ASTOR CHOCOLATE"])
    b = frozen_list_sha256(["bakso sapi karawaci", "Alba Cheese", "astor chocolate"])
    assert a == b, "hash harus tahan varian kapitalisasi/spasi"


def test_frozen_load_cek_env(tmp_path, monkeypatch):
    from app.core.data.frozen import FROZEN_SHA_ENV, frozen_list_sha256, load_frozen_products

    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(["Alba Cheese"]), encoding="utf-8")
    correct = frozen_list_sha256(["alba cheese"])

    monkeypatch.setenv(FROZEN_SHA_ENV, correct)
    products, sha = load_frozen_products(str(path))
    assert products == ["Alba Cheese"] and sha == correct

    monkeypatch.setenv(FROZEN_SHA_ENV, "0" * 64)
    with pytest.raises(ValueError, match=FROZEN_SHA_ENV):
        load_frozen_products(str(path))

    monkeypatch.delenv(FROZEN_SHA_ENV)
    with pytest.raises(FileNotFoundError):
        load_frozen_products(str(tmp_path / "missing.json"))
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="list string"):
        load_frozen_products(str(path))


def test_resolve_frozen_urutan(tmp_path, monkeypatch):
    from app.core.data.frozen import FROZEN_SHA_ENV, resolve_frozen_products

    monkeypatch.delenv(FROZEN_SHA_ENV, raising=False)
    p = tmp_path / "explicit.json"
    p.write_text(json.dumps(["x"]), encoding="utf-8")
    packaged = tmp_path / "packaged.json"  # tidak ada
    nb23 = tmp_path / "nb23.json"

    # 1. explicit diutamakan
    products, sha = resolve_frozen_products(str(p), str(packaged), str(nb23), 499)
    assert products == ["x"] and len(sha) == 64

    # 2. packaged bila explicit tidak diset
    packaged.write_text(json.dumps(["y"]), encoding="utf-8")
    products, _ = resolve_frozen_products(None, str(packaged), str(nb23), 499)
    assert products == ["y"]

    # 3. notebook23 hanya bila n_real==114
    nb23.write_text(json.dumps(["z"]), encoding="utf-8")
    assert resolve_frozen_products(None, str(packaged), str(nb23), 499)[0] == ["y"]
    assert resolve_frozen_products(None, str(tmp_path / "nope"), str(nb23), 114)[0] == ["z"]

    # 4. tak ada apa-apa -> re-derive (None)
    assert resolve_frozen_products(None, str(tmp_path / "nope"), str(nb23), 499) == (None, None)


# ---------- Audit pipeline diperketat ----------

def _tiny_split(n_safe=2, n_unsafe=2, single=False):
    rows = [{"label_id": 0, "t": f"s{i}"} for i in range(n_safe)]
    if not single:
        rows += [{"label_id": 1, "t": f"u{i}"} for i in range(n_unsafe)]
    return pd.DataFrame(rows)


def test_audit_pipeline_nilai_raise():
    from app.core.model.audit import audit_pipeline_consistency_v5

    ok = _tiny_split()
    # happy path
    audit_pipeline_consistency_v5(ok, ok, ok, 10, 10, False, 0.5)
    with pytest.raises(ValueError, match="model.fit"):
        audit_pipeline_consistency_v5(ok, ok, ok, 10, 10, True, 0.5)
    with pytest.raises(ValueError, match="0.5"):
        audit_pipeline_consistency_v5(ok, ok, ok, 10, 10, False, 0.7)
    with pytest.raises(ValueError, match="synthetic"):
        audit_pipeline_consistency_v5(ok, ok, ok, 9, 10, False, 0.5)
    bad = ok.copy()
    bad.loc[0, "label_id"] = 7
    with pytest.raises(ValueError, match="label_id"):
        audit_pipeline_consistency_v5(bad, ok, ok, 10, 10, False, 0.5)
    single = _tiny_split(single=True)
    with pytest.raises(ValueError, match="single-class"):
        audit_pipeline_consistency_v5(single, ok, ok, 10, 10, False, 0.5)


def test_audit_final_config_kontrak():
    from app.config import Config
    from app.core.model.audit import audit_final_config_v5

    v5 = Config().v5
    result = audit_final_config_v5(v5)
    assert result["learning_rate"] == 1e-4
    assert result["preshuffle"] is False

    # override env menyimpang -> gagal keras
    original = v5.digit_fold
    try:
        v5.digit_fold = True
        with pytest.raises(ValueError, match="digit_fold"):
            audit_final_config_v5(v5)
    finally:
        v5.digit_fold = original


def test_plot_roc_pr_single_class_gagal(tmp_path):
    from app.core.model.metrics import plot_roc_pr_v5

    y_val = np.array([0, 0, 1, 1])
    p_val = np.array([0.1, 0.4, 0.6, 0.9])
    y_hold = np.array([0, 0, 0, 0])  # single-class
    with pytest.raises(ValueError, match="single-class"):
        plot_roc_pr_v5(y_val, p_val, y_hold, p_val[:4], str(tmp_path))


def test_export_threshold_kontrak_literal(tmp_path):
    """export --threshold 0.7 menolak SEBELUM menyentuh file artefak."""
    import importlib.util
    from pathlib import Path

    from click.testing import CliRunner

    script = Path(__file__).resolve().parents[1] / "scripts" / "export_for_serving.py"
    spec = importlib.util.spec_from_file_location("export_for_serving", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    runner = CliRunner()
    result = runner.invoke(mod.main, [
        "--src-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"),
        "--threshold", "0.7",
    ])
    assert result.exit_code != 0
    assert "kontrak fixed 0.5" in result.output


def test_tokenizer_training_roundtrip(tmp_path):
    """Tokenizer training save_json/load_json identik (parity inti)."""
    pytest.importorskip("tensorflow")
    from app.core.model.tokenizer import Tokenizer

    t = Tokenizer(vocab_size=50, max_len=8)
    texts = ["susu telur kacang", "tepung terigu gandum"]
    t.fit(texts)
    p = str(tmp_path / "tok.json")
    t.save_json(p)
    t2 = Tokenizer.load_json(p)
    assert t2.word_index == t.word_index
    seqs_a = t.texts_to_sequences(texts)
    seqs_b = t2.texts_to_sequences(texts)
    assert seqs_a == seqs_b, "roundtrip harus identik"
