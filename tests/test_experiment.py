"""Test modul eksperimen baru (tanpa TensorFlow/Torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.experiment.configs import (
    BertConfig,
    BiLSTMConfig,
    ConfigError,
    DataConfig,
    HPOConfig,
)
from app.experiment.hpo import HPOEngine
from app.experiment.reporting import (
    classification_metrics,
    mcnemar_test,
    paired_bootstrap_auc_diff,
)
from app.experiment.search_spaces import (
    Categorical,
    FloatRange,
    IntRange,
    SearchSpace,
    bert_search_space,
    bilstm_search_space,
    preset_trials,
)


def test_bilstm_config_default_dan_validasi():
    config = BiLSTMConfig()
    assert config.rnn_type == "lstm"
    assert config.use_bidirectional is True
    assert config.shuffle is True
    with pytest.raises(ConfigError):
        BiLSTMConfig(dropout_1=1.5)
    with pytest.raises(ConfigError):
        BiLSTMConfig(optimizer="tidak-ada")
    with pytest.raises(ConfigError):
        BiLSTMConfig(monitor="val_loss", monitor_mode="max")


def test_bert_config_menurunkan_greater_is_better():
    loss_cfg = BertConfig(metric_for_best_model="eval_loss")
    assert loss_cfg.greater_is_better is False
    f1_cfg = BertConfig(metric_for_best_model="eval_f1")
    assert f1_cfg.greater_is_better is True
    with pytest.raises(ConfigError):
        BertConfig(precision="int8")
    with pytest.raises(ConfigError):
        BertConfig(learning_rate=0)


def test_data_config_dan_hpo_config():
    with pytest.raises(ConfigError):
        DataConfig(synthetic_total=11)
    with pytest.raises(ConfigError):
        DataConfig(holdout_safe=0)
    with pytest.raises(ConfigError):
        HPOConfig(strategy="genetic")
    with pytest.raises(ConfigError, match="holdout"):
        HPOConfig(objective_metric="holdout_f1")
    assert DataConfig(synthetic_total=0).synthetic_total == 0


def test_search_space_sampling_valid():
    for preset in ("smoke", "standard", "full"):
        space = bilstm_search_space(preset)
        params = space.sample(__import__("random").Random(0))
        space.validate(params)
        BiLSTMConfig.from_dict(params)
    for preset in ("smoke", "standard", "full"):
        space = bert_search_space(preset)
        params = space.sample(__import__("random").Random(0))
        space.validate(params)
        BertConfig.from_dict(params)


def test_search_space_grid_dan_roundtrip():
    space = SearchSpace(
        {"units": Categorical((16, 32)), "epochs": IntRange(1, 2)}
    )
    grid = space.grid()
    assert len(grid) == 4
    assert space.from_dict(space.to_dict()).grid() == grid
    with pytest.raises(ConfigError):
        SearchSpace(
            {"lr": FloatRange(1e-4, 1e-2, log=True)}
        ).grid()


def test_search_space_menolak_nilai_di_luar_batas():
    space = SearchSpace({"units": Categorical((16, 32)), "epochs": IntRange(1, 2)})
    with pytest.raises(ConfigError):
        space.validate({"units": 64, "epochs": 1})
    with pytest.raises(ConfigError):
        space.validate({"units": 16, "epochs": 5})
    with pytest.raises(ConfigError):
        space.validate({"units": 16})


def test_preset_trials():
    assert preset_trials("smoke") < preset_trials("standard") < preset_trials("full")


def test_hpo_engine_logging_resume_dan_best(tmp_path):
    space = SearchSpace(
        {
            "units": Categorical((16, 32, 64)),
            "epochs": IntRange(1, 3),
            "lr": FloatRange(1e-4, 1e-2, log=True),
        }
    )
    config = HPOConfig(
        n_trials=5, output_dir=str(tmp_path), study_name="demo", seed=7
    )
    run_dir = tmp_path / "run"

    def objective(params, index, trial_dir):
        value = params["units"] / 100 + params["epochs"] - params["lr"] * 100
        return {"metrics": {"val_f1": value, "val_loss": 1.0 - value}}

    first = HPOEngine(config, space, run_dir=run_dir, verbose=False).run(objective)
    assert len(first.trials) == 5
    assert first.best is not None
    assert (run_dir / "trials.jsonl").exists()
    assert (run_dir / "leaderboard.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "best_params.json").exists()

    # Resume dengan target yang sama tidak boleh menambah trial.
    second = HPOEngine(config, space, run_dir=run_dir, verbose=False).run(objective)
    assert len(second.trials) == 5

    # Resume dengan target lebih besar menambah tepat sebanyak kekurangannya.
    config_more = HPOConfig(
        n_trials=7, output_dir=str(tmp_path), study_name="demo", seed=7
    )
    third = HPOEngine(config_more, space, run_dir=run_dir, verbose=False).run(objective)
    assert len(third.trials) == 7
    hashes = [trial.params_hash for trial in third.trials]
    assert len(hashes) == len(set(hashes))


def test_hpo_engine_mencatat_trial_gagal(tmp_path):
    space = SearchSpace({"x": Categorical((1, 2, 3))})
    config = HPOConfig(
        n_trials=3, output_dir=str(tmp_path), study_name="fail", seed=1
    )

    def objective(params, index, trial_dir):
        if params["x"] == 1:
            raise RuntimeError("sengaja gagal")
        return {"metrics": {"val_f1": float(params["x"])}}

    result = HPOEngine(config, space, run_dir=tmp_path / "run", verbose=False).run(objective)
    assert any(trial.status == "failed" for trial in result.trials)
    assert result.best is not None
    assert result.best.params["x"] != 1


def test_metrik_dan_uji_statistik():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6])
    metrics = classification_metrics(y, p)
    assert metrics["accuracy"] == 1.0
    assert 0.9 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0

    mcnemar = mcnemar_test(y, (p >= 0.5).astype(int), (p >= 0.8).astype(int))
    assert "p_value" in mcnemar and 0.0 <= mcnemar["p_value"] <= 1.0

    diff = paired_bootstrap_auc_diff(y, p, p, n_bootstrap=50, seed=0)
    assert diff["auc_a_minus_b"] == 0.0
    assert diff["p_value"] >= 0.9


def test_load_probabilities_dua_skema(tmp_path):
    from app.experiment.reporting import load_probabilities

    old = tmp_path / "old.csv"
    pd.DataFrame({"label_id": [0, 1], "prob_unsafe": [0.1, 0.9]}).to_csv(old, index=False)
    y, p, df = load_probabilities(old)
    assert list(y) == [0, 1] and len(df) == 2

    new = tmp_path / "new.csv"
    pd.DataFrame(
        {"text": ["a", "b"], "label": ["safe", "unsafe"], "label_id": [0, 1],
         "prob_unsafe": [0.2, 0.8]}
    ).to_csv(new, index=False)
    y2, p2, _ = load_probabilities(new)
    assert list(y2) == [0, 1]


def test_hpo_bayesian_fallback_tanpa_optuna(tmp_path, monkeypatch):
    """Strategi bayesian harus fallback ke random bila optuna tidak ada."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("optuna tidak dipasang untuk test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    space = SearchSpace({"x": Categorical((1, 2, 3))})
    config = HPOConfig(
        strategy="bayesian", n_trials=3, output_dir=str(tmp_path),
        study_name="bayes", seed=1,
    )
    result = HPOEngine(config, space, run_dir=tmp_path / "run", verbose=False).run(
        lambda params, index, trial_dir: {"metrics": {"val_f1": float(params["x"])}}
    )
    assert len(result.trials) == 3
    assert all(trial.status == "ok" for trial in result.trials)
