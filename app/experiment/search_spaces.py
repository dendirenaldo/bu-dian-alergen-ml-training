"""Ruang pencarian hiperparameter (search space).

Desain:

- Setiap parameter dijelaskan oleh ``ParamSpec`` (Categorical / IntRange /
  FloatRange). Spesifikasi ini bisa dipakai untuk **grid**, **random**, dan
  **Bayesian** (Optuna) tanpa mengubah kode runner.
- Tiga preset disediakan: ``smoke`` (cepat), ``standard`` (disarankan untuk
  paper), dan ``full`` (eksplorasi maksimal).
- Semua opsi yang tersedia di ``BiLSTMConfig`` / ``BertConfig`` tercakup di
  preset ``full``. Preset lain hanya mempersempit agar waktu training wajar.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any

from app.experiment.configs import ConfigError


class ParamSpec:
    """Kontrak minimum satu parameter hyperparameter."""

    kind = "base"

    def sample(self, rng: random.Random) -> Any:  # pragma: no cover - abstrak
        raise NotImplementedError

    def grid_values(self) -> list[Any]:  # pragma: no cover - abstrak
        raise NotImplementedError

    def suggest(self, trial: Any) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, value: Any) -> None:
        """Validasi nilai; default tidak melakukan apa-apa."""


@dataclass(frozen=True)
class Categorical(ParamSpec):
    """Pilihan diskret, mis. optimizer atau dropout."""

    values: tuple[Any, ...]
    kind = "categorical"

    def __post_init__(self) -> None:
        if not self.values:
            raise ConfigError("Categorical tidak boleh kosong.")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def grid_values(self) -> list[Any]:
        return list(self.values)

    def suggest(self, trial: Any) -> Any:
        # Optuna tidak selalu nyaman dengan None; gunakan sentinel.
        if any(value is None for value in self.values):
            mapping = {None: "__none__"}
            for value in self.values:
                if value is not None:
                    mapping[value] = value
            inverse = {v: k for k, v in mapping.items()}
            choice = trial.suggest_categorical(
                self._name, [mapping[v] for v in self.values]
            )
            return inverse[choice]
        return trial.suggest_categorical(self._name, list(self.values))

    def describe(self) -> dict[str, Any]:
        return {"type": "categorical", "options": list(self.values)}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "values": list(self.values)}

    def validate(self, value: Any) -> None:
        if value not in self.values:
            raise ConfigError(f"Nilai {value!r} tidak ada di pilihan {self.values}.")


@dataclass(frozen=True)
class IntRange(ParamSpec):
    """Bilangan bulat pada rentang tertentu."""

    low: int
    high: int
    step: int = 1
    log: bool = False
    kind = "int"

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ConfigError("IntRange: low tidak boleh > high.")
        if self.step <= 0:
            raise ConfigError("IntRange: step harus > 0.")
        if self.log and self.low <= 0:
            raise ConfigError("IntRange log: low harus > 0.")

    def sample(self, rng: random.Random) -> int:
        if self.log:
            value = int(round(math.exp(rng.uniform(math.log(self.low), math.log(self.high)))))
            return min(self.high, max(self.low, value))
        count = (self.high - self.low) // self.step + 1
        return self.low + self.step * rng.randrange(count)

    def grid_values(self) -> list[int]:
        return list(range(self.low, self.high + 1, self.step))

    def suggest(self, trial: Any) -> int:
        return trial.suggest_int(
            self._name, self.low, self.high, step=self.step, log=self.log
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "int",
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }

    def validate(self, value: Any) -> None:
        if not isinstance(value, int):
            raise ConfigError(f"Nilai {value!r} harus int.")
        if not (self.low <= value <= self.high):
            raise ConfigError(f"Nilai {value} di luar rentang {self.low}..{self.high}.")
        if (value - self.low) % self.step != 0:
            raise ConfigError(f"Nilai {value} tidak mengikuti step {self.step}.")


@dataclass(frozen=True)
class FloatRange(ParamSpec):
    """Bilangan pecahan; bisa linear, log, atau diskret dengan step."""

    low: float
    high: float
    step: float | None = None
    log: bool = False
    kind = "float"

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ConfigError("FloatRange: low tidak boleh > high.")
        if self.log and self.low <= 0:
            raise ConfigError("FloatRange log: low harus > 0.")
        if self.step is not None and self.step <= 0:
            raise ConfigError("FloatRange: step harus > 0 atau None.")
        if self.log and self.step is not None:
            raise ConfigError("FloatRange: log dan step tidak bisa dipakai bersamaan.")

    def sample(self, rng: random.Random) -> float:
        if self.step is not None:
            values = self.grid_values()
            return float(rng.choice(values))
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def grid_values(self) -> list[float]:
        if self.step is None:
            raise ConfigError(
                "Grid untuk FloatRange membutuhkan step. "
                "Gunakan strategi random/bayesian untuk parameter kontinu."
            )
        count = int(math.floor((self.high - self.low) / self.step + 1e-9)) + 1
        return [round(self.low + i * self.step, 10) for i in range(count)]

    def suggest(self, trial: Any) -> float:
        return trial.suggest_float(
            self._name, self.low, self.high, step=self.step, log=self.log
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "float",
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }

    def validate(self, value: Any) -> None:
        if not isinstance(value, (int, float)):
            raise ConfigError(f"Nilai {value!r} harus angka.")
        if not (self.low <= float(value) <= self.high):
            raise ConfigError(f"Nilai {value} di luar rentang {self.low}..{self.high}.")


@dataclass
class SearchSpace:
    """Kumpulan parameter beserta strategi sampling."""

    params: dict[str, ParamSpec]

    def __post_init__(self) -> None:
        if not self.params:
            raise ConfigError("SearchSpace tidak boleh kosong.")
        for name, spec in self.params.items():
            if not name or not name.isidentifier():
                raise ConfigError(f"Nama parameter tidak valid: {name!r}")
            if not isinstance(spec, ParamSpec):
                raise ConfigError(f"Parameter {name} bukan ParamSpec.")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {name: spec.sample(rng) for name, spec in self.params.items()}

    def grid(self, max_combinations: int = 200) -> list[dict[str, Any]]:
        """Buat seluruh kombinasi grid.

        Raises:
            ConfigError: bila jumlah kombinasi melebihi ``max_combinations``
                atau ada parameter kontinu tanpa step.
        """
        keys = list(self.params)
        values = []
        for key in keys:
            try:
                values.append(self.params[key].grid_values())
            except ConfigError as exc:
                raise ConfigError(
                    f"Parameter '{key}' tidak bisa di-grid: {exc}"
                ) from exc
        total = 1
        for item in values:
            total *= len(item)
        if total > max_combinations:
            raise ConfigError(
                f"Grid berisi {total} kombinasi (> batas {max_combinations}). "
                "Gunakan strategi 'random' atau 'bayesian', atau persempit search space."
            )
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def size(self) -> int | None:
        """Perkiraan jumlah kombinasi grid; ``None`` bila kontinu."""
        total = 1
        for spec in self.params.values():
            try:
                total *= len(spec.grid_values())
            except ConfigError:
                return None
        return total

    def validate(self, params: dict[str, Any]) -> None:
        unknown = sorted(set(params) - set(self.params))
        if unknown:
            raise ConfigError(f"Parameter di luar search space: {unknown}")
        missing = sorted(set(self.params) - set(params))
        if missing:
            raise ConfigError(f"Parameter belum diisi: {missing}")
        for name, value in params.items():
            self.params[name].validate(value)

    def suggest(self, trial: Any) -> dict[str, Any]:
        """Sampling Bayesian via Optuna trial."""
        for name, spec in self.params.items():
            if not hasattr(spec, "_name"):
                # Nama parameter dikirim lewat konteks optuna.
                pass
            object.__setattr__(spec, "_name", name)
        return {name: spec.suggest(trial) for name, spec in self.params.items()}

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for name, spec in self.params.items():
            row = {"parameter": name, **spec.describe()}
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {name: spec.to_dict() for name, spec in self.params.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchSpace":
        params: dict[str, ParamSpec] = {}
        for name, spec in payload.items():
            kind = spec.get("kind")
            if kind == "categorical":
                params[name] = Categorical(tuple(spec["values"]))
            elif kind == "int":
                params[name] = IntRange(
                    int(spec["low"]), int(spec["high"]),
                    int(spec.get("step", 1)), bool(spec.get("log", False)),
                )
            elif kind == "float":
                params[name] = FloatRange(
                    float(spec["low"]), float(spec["high"]),
                    spec.get("step"), bool(spec.get("log", False)),
                )
            else:
                raise ConfigError(f"Tipe parameter tidak dikenal: {kind}")
        return cls(params)


def _space(**params: ParamSpec) -> SearchSpace:
    return SearchSpace(dict(params))


def bilstm_search_space(preset: str = "standard") -> SearchSpace:
    """Search space BiLSTM sesuai preset.

    Args:
        preset: ``smoke``, ``standard``, atau ``full``.
    """
    preset = preset.lower()
    if preset == "smoke":
        return _space(
            max_len=Categorical((100, 140)),
            embed_dim=Categorical((100, 200)),
            lstm_units_1=Categorical((64, 128)),
            dropout_1=Categorical((0.2, 0.3)),
            learning_rate=FloatRange(1e-4, 1e-3, log=True),
            batch_size=Categorical((16, 32)),
            optimizer=Categorical(("adam", "adamw")),
            w2v_epochs=Categorical((10,)),
            epochs=Categorical((10,)),
            mask_zero=Categorical((True,)),
            use_bidirectional=Categorical((True,)),
            num_rnn_layers=Categorical((2,)),
        )
    if preset == "standard":
        return _space(
            vocab_size=Categorical((10000, 20000, 30000)),
            max_len=Categorical((100, 120, 160)),
            embed_dim=Categorical((100, 200)),
            w2v_window=Categorical((3, 5, 7)),
            w2v_min_count=Categorical((1, 2, 3)),
            w2v_epochs=Categorical((20, 30)),
            w2v_sg=Categorical((1,)),
            embed_trainable=Categorical((True, False)),
            mask_zero=Categorical((True, False)),
            digit_fold=Categorical((False, True)),
            rnn_type=Categorical(("lstm", "gru")),
            use_bidirectional=Categorical((True, False)),
            num_rnn_layers=Categorical((1, 2)),
            lstm_units_1=Categorical((64, 128, 256)),
            lstm_units_2=Categorical((32, 64, 128)),
            dropout_1=Categorical((0.1, 0.2, 0.3, 0.4)),
            dropout_2=Categorical((0.1, 0.2, 0.3, 0.4)),
            recurrent_dropout=Categorical((0.0, 0.1, 0.2)),
            dense_units=Categorical((32, 64, 128)),
            dropout_dense=Categorical((0.1, 0.2, 0.3)),
            use_maxnorm=Categorical((False, True)),
            optimizer=Categorical(("adam", "adamw")),
            learning_rate=FloatRange(5e-5, 5e-3, log=True),
            weight_decay=Categorical((0.0, 1e-4, 1e-3)),
            batch_size=Categorical((8, 16, 32)),
            epochs=Categorical((20, 40, 60)),
            gradient_clip_norm=Categorical((None, 1.0, 2.0)),
            early_stopping_patience=Categorical((3, 4, 5)),
            lr_reduce_patience=Categorical((2, 3, 4)),
            lr_reduce_factor=Categorical((0.3, 0.5, 0.7)),
            use_class_weight=Categorical((False, True)),
            label_smoothing=Categorical((0.0, 0.05)),
            shuffle=Categorical((True,)),
        )
    if preset == "full":
        return _space(
            vocab_size=Categorical((10000, 20000, 30000, 50000)),
            max_len=IntRange(60, 240, step=20),
            embed_dim=Categorical((50, 100, 200, 300)),
            w2v_window=IntRange(3, 10, step=1),
            w2v_min_count=IntRange(1, 5, step=1),
            w2v_epochs=IntRange(10, 50, step=10),
            w2v_sg=Categorical((0, 1)),
            embed_trainable=Categorical((True, False)),
            mask_zero=Categorical((True, False)),
            digit_fold=Categorical((False, True)),
            rnn_type=Categorical(("lstm", "gru")),
            use_bidirectional=Categorical((True, False)),
            num_rnn_layers=Categorical((1, 2)),
            lstm_units_1=Categorical((32, 64, 128, 256)),
            lstm_units_2=Categorical((16, 32, 64, 128)),
            dropout_1=FloatRange(0.0, 0.6, step=0.1),
            dropout_2=FloatRange(0.0, 0.6, step=0.1),
            recurrent_dropout=FloatRange(0.0, 0.4, step=0.1),
            dense_units=Categorical((32, 64, 128, 256)),
            dropout_dense=FloatRange(0.0, 0.6, step=0.1),
            use_maxnorm=Categorical((False, True)),
            optimizer=Categorical(("adam", "adamw", "rmsprop", "sgd")),
            learning_rate=FloatRange(1e-5, 1e-2, log=True),
            weight_decay=Categorical((0.0, 1e-5, 1e-4, 1e-3, 1e-2)),
            batch_size=Categorical((8, 16, 32, 64)),
            epochs=Categorical((20, 30, 50, 80)),
            gradient_clip_norm=Categorical((None, 0.5, 1.0, 2.0, 5.0)),
            early_stopping_patience=Categorical((2, 3, 4, 5, 8, 10)),
            lr_reduce_patience=Categorical((2, 3, 4, 5)),
            lr_reduce_factor=Categorical((0.1, 0.3, 0.5, 0.7)),
            use_class_weight=Categorical((False, True)),
            label_smoothing=Categorical((0.0, 0.05, 0.1)),
            shuffle=Categorical((True, False)),
        )
    raise ConfigError("preset BiLSTM harus smoke, standard, atau full.")


def bert_search_space(preset: str = "standard") -> SearchSpace:
    """Search space IndoBERT sesuai preset."""
    preset = preset.lower()
    if preset == "smoke":
        return _space(
            max_len=Categorical((128, 192)),
            learning_rate=FloatRange(1e-5, 5e-5, log=True),
            batch_size=Categorical((8, 16)),
            epochs=Categorical((1, 2)),
            weight_decay=Categorical((0.0, 0.01)),
            warmup_ratio=Categorical((0.0, 0.1)),
            lr_scheduler_type=Categorical(("linear", "cosine")),
            freeze_layers=Categorical((0, -1)),
            metric_for_best_model=Categorical(("eval_loss", "eval_f1")),
            precision=Categorical(("fp16", "fp32")),
        )
    if preset == "standard":
        return _space(
            model_name=Categorical((
                "indobenchmark/indobert-base-p1",
                "indobenchmark/indobert-base-p2",
            )),
            max_len=Categorical((128, 192, 256)),
            epochs=Categorical((3, 4, 5, 6)),
            learning_rate=FloatRange(5e-6, 1e-4, log=True),
            batch_size=Categorical((8, 16, 32)),
            weight_decay=Categorical((0.0, 0.01, 0.05)),
            warmup_ratio=Categorical((0.0, 0.06, 0.1, 0.2)),
            lr_scheduler_type=Categorical(("linear", "cosine", "constant_with_warmup")),
            optimizer=Categorical(("adamw_torch", "adamw_torch_fused")),
            max_grad_norm=Categorical((0.5, 1.0, 2.0)),
            gradient_accumulation_steps=Categorical((1, 2)),
            early_stopping_patience=Categorical((1, 2, 3)),
            metric_for_best_model=Categorical((
                "eval_loss", "eval_f1", "eval_roc_auc", "eval_average_precision",
            )),
            freeze_layers=Categorical((0, -1, 2, 4)),
            use_class_weight=Categorical((False, True)),
            label_smoothing_factor=Categorical((0.0, 0.05)),
            hidden_dropout_prob=Categorical((0.1, 0.2)),
            attention_probs_dropout_prob=Categorical((0.1, 0.2)),
            classifier_dropout=Categorical((None, 0.1, 0.2)),
            precision=Categorical(("fp32", "fp16")),
            group_by_length=Categorical((False, True)),
        )
    if preset == "full":
        return _space(
            model_name=Categorical((
                "indobenchmark/indobert-base-p1",
                "indobenchmark/indobert-base-p2",
                "indobenchmark/indobert-large-p2",
            )),
            max_len=Categorical((128, 192, 256, 320, 384, 512)),
            epochs=Categorical((3, 4, 5, 6, 8, 10)),
            learning_rate=FloatRange(5e-6, 1e-4, log=True),
            batch_size=Categorical((8, 16, 32)),
            weight_decay=Categorical((0.0, 0.01, 0.05, 0.1)),
            warmup_ratio=Categorical((0.0, 0.06, 0.1, 0.2)),
            lr_scheduler_type=Categorical((
                "linear", "cosine", "cosine_with_restarts",
                "polynomial", "constant_with_warmup",
            )),
            optimizer=Categorical(("adamw_torch", "adamw_torch_fused", "adafactor")),
            max_grad_norm=Categorical((0.5, 1.0, 2.0, 5.0)),
            gradient_accumulation_steps=Categorical((1, 2, 4)),
            early_stopping_patience=Categorical((1, 2, 3, 4)),
            metric_for_best_model=Categorical((
                "eval_loss", "eval_f1", "eval_roc_auc",
                "eval_average_precision", "eval_accuracy",
            )),
            freeze_layers=Categorical((0, -1, 2, 4, 6, 8)),
            use_class_weight=Categorical((False, True)),
            label_smoothing_factor=Categorical((0.0, 0.05, 0.1)),
            hidden_dropout_prob=Categorical((0.05, 0.1, 0.2, 0.3)),
            attention_probs_dropout_prob=Categorical((0.05, 0.1, 0.2, 0.3)),
            classifier_dropout=Categorical((None, 0.1, 0.2, 0.3)),
            precision=Categorical(("fp32", "fp16", "bf16")),
            group_by_length=Categorical((False, True)),
        )
    raise ConfigError("preset BERT harus smoke, standard, atau full.")


def preset_trials(preset: str) -> int:
    """Jumlah trial default yang masuk akal untuk tiap preset."""
    return {"smoke": 4, "standard": 30, "full": 100}.get(preset.lower(), 30)


def space_summary(space: SearchSpace) -> list[dict[str, Any]]:
    """Ringkasan yang enak ditampilkan di notebook/pandas."""
    return space.describe()
