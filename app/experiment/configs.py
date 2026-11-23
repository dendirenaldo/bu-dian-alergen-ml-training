"""Konfigurasi eksperimen yang tervalidasi dan mudah dibaca.

Ada empat konfigurasi:

- ``DataConfig``   : dari mana data dibaca dan bagaimana split dibentuk.
- ``BiLSTMConfig`` : seluruh hiperparameter Word2Vec + BiLSTM.
- ``BertConfig``   : seluruh hiperparameter fine-tuning IndoBERT.
- ``HPOConfig``    : bagaimana pencarian hiperparameter dijalankan.

Semua konfigurasi memakai ``dataclass`` biasa agar mudah dibaca pemula,
tetapi tetap memvalidasi nilai di ``__post_init__`` supaya typo/parameter
berbahaya gagal cepat dengan pesan yang jelas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal


class ConfigError(ValueError):
    """Kesalahan konfigurasi dengan pesan yang ramah pengguna."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_fraction(name: str, value: float, *, allow_zero: bool = True) -> None:
    low_ok = value >= 0.0 if allow_zero else value > 0.0
    _require(low_ok and value <= 1.0, f"{name} harus di antara 0 dan 1, dapat {value}.")


@dataclass
class DataConfig:
    """Sumber data dan kontrak split leakage-safe.

    Ukuran split default mengikuti kontrak final: 399 train real,
    50 validasi (25/25), 50 frozen holdout (25/25).
    """

    csv_input: str = "./ocr_output/data-mengandung.csv"
    csv_delimiter: str = ";"
    csv_encoding: str = "utf-8"
    product_col: str = "nama_produk"
    text_col: str = "text"
    label_col: str = "label"
    seed: int = 42
    holdout_safe: int = 25
    holdout_unsafe: int = 25
    val_safe: int = 25
    val_unsafe: int = 25
    synthetic_total: int = 1000
    frozen_holdout_path: str | None = None
    sort_source: bool = False
    output_dir: str = "./artifacts/data"
    bert_input_dir: str = "./artifacts/bert/input"

    def __post_init__(self) -> None:
        for name in ("holdout_safe", "holdout_unsafe", "val_safe", "val_unsafe"):
            _require(getattr(self, name) > 0, f"{name} harus > 0.")
        _require(
            self.synthetic_total >= 0,
            "synthetic_total tidak boleh negatif (0 = ablasi tanpa synthetic).",
        )
        _require(
            self.synthetic_total % 2 == 0,
            "synthetic_total harus genap agar seimbang 50/50.",
        )
        _require(bool(self.text_col), "text_col tidak boleh kosong.")
        _require(bool(self.product_col), "product_col tidak boleh kosong.")
        _require(bool(self.label_col), "label_col tidak boleh kosong.")

    @property
    def holdout_size(self) -> int:
        return self.holdout_safe + self.holdout_unsafe

    @property
    def val_size(self) -> int:
        return self.val_safe + self.val_unsafe

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DataConfig":
        return _from_dict(cls, params)


@dataclass
class BiLSTMConfig:
    """Hiperparameter Word2Vec + BiLSTM.

    Default = konfigurasi final yang sudah dipakai di paper, sehingga
    notebook baseline menghasilkan model yang setara dengan artefak lama.
    Setiap field boleh diubah oleh HPO.
    """

    # --- Vektorisasi teks ---
    vocab_size: int = 20000
    max_len: int = 120
    embed_dim: int = 100
    w2v_window: int = 5
    w2v_min_count: int = 1
    w2v_epochs: int = 20
    w2v_sg: int = 1  # 1 = skip-gram, 0 = CBOW
    w2v_workers: int = 1
    embedding_init_scale: float = 0.6
    embed_trainable: bool = True
    mask_zero: bool = True
    digit_fold: bool = False

    # --- Arsitektur ---
    rnn_type: Literal["lstm", "gru"] = "lstm"
    use_bidirectional: bool = True
    num_rnn_layers: int = 2
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_1: float = 0.3
    dropout_2: float = 0.3
    recurrent_dropout: float = 0.0
    dense_units: int = 64
    dropout_dense: float = 0.2
    use_maxnorm: bool = False
    maxnorm_value: float = 3.0

    # --- Optimasi ---
    optimizer: Literal["adam", "adamw", "rmsprop", "sgd"] = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16
    epochs: int = 20
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 4
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    use_class_weight: bool = False
    label_smoothing: float = 0.0

    # --- Urutan data & pemilihan model ---
    shuffle: bool = True
    preshuffle: bool = False
    monitor: Literal["val_loss", "val_f1"] = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"

    # --- GPU / performa ---
    mixed_precision: Literal["float32", "mixed_float16", "mixed_bfloat16"] = "mixed_float16"
    multi_gpu: bool = True
    cross_device_ops: Literal["auto", "nccl", "hierarchical", "reduction_to_one"] = "auto"
    auto_scale_batch_size: bool = True
    use_tf_data: bool = True
    prefetch: bool = True
    memory_growth: bool = True
    steps_per_execution: int = 1

    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        _require(self.vocab_size > 0, "vocab_size harus > 0.")
        _require(self.max_len > 0, "max_len harus > 0.")
        _require(self.embed_dim > 0, "embed_dim harus > 0.")
        _require(self.w2v_window > 0, "w2v_window harus > 0.")
        _require(self.w2v_min_count >= 0, "w2v_min_count tidak boleh negatif.")
        _require(self.w2v_epochs > 0, "w2v_epochs harus > 0.")
        _require(self.w2v_sg in (0, 1), "w2v_sg harus 0 (CBOW) atau 1 (skip-gram).")
        _require(self.w2v_workers > 0, "w2v_workers harus > 0.")
        _require(self.embedding_init_scale >= 0, "embedding_init_scale tidak boleh negatif.")
        _require(self.rnn_type in ("lstm", "gru"), "rnn_type harus 'lstm' atau 'gru'.")
        _require(self.num_rnn_layers in (1, 2), "num_rnn_layers harus 1 atau 2.")
        _require(self.lstm_units_1 > 0, "lstm_units_1 harus > 0.")
        _require(self.lstm_units_2 > 0, "lstm_units_2 harus > 0.")
        for name in ("dropout_1", "dropout_2", "recurrent_dropout", "dropout_dense"):
            _check_fraction(name, float(getattr(self, name)))
        _require(self.dense_units > 0, "dense_units harus > 0.")
        _require(self.use_maxnorm in (True, False), "use_maxnorm harus boolean.")
        _require(self.maxnorm_value > 0, "maxnorm_value harus > 0.")
        _require(
            self.optimizer in ("adam", "adamw", "rmsprop", "sgd"),
            "optimizer harus adam, adamw, rmsprop, atau sgd.",
        )
        _require(self.learning_rate > 0, "learning_rate harus > 0.")
        _require(self.weight_decay >= 0, "weight_decay tidak boleh negatif.")
        _require(self.batch_size > 0, "batch_size harus > 0.")
        _require(self.epochs > 0, "epochs harus > 0.")
        if self.gradient_clip_norm is not None:
            _require(self.gradient_clip_norm > 0, "gradient_clip_norm harus > 0 atau None.")
        _require(self.early_stopping_patience >= 0, "early_stopping_patience tidak boleh negatif.")
        _require(self.lr_reduce_patience >= 0, "lr_reduce_patience tidak boleh negatif.")
        _check_fraction("lr_reduce_factor", self.lr_reduce_factor, allow_zero=False)
        _require(self.min_lr > 0, "min_lr harus > 0.")
        _check_fraction("label_smoothing", self.label_smoothing)
        _require(
            self.monitor in ("val_loss", "val_f1"),
            "monitor harus 'val_loss' atau 'val_f1'. ROC-AUC/AP dihitung "
            "setelah training dengan scikit-learn (bukan metric Keras) agar "
            "kompatibel dengan deterministic GPU.",
        )
        _require(self.monitor_mode in ("min", "max"), "monitor_mode harus 'min' atau 'max'.")
        _require(self.seed >= 0, "seed tidak boleh negatif.")
        _require(isinstance(self.deterministic, bool), "deterministic harus boolean.")
        _require(
            self.mixed_precision in ("float32", "mixed_float16", "mixed_bfloat16"),
            "mixed_precision harus float32, mixed_float16, atau mixed_bfloat16.",
        )
        for name in ("multi_gpu", "auto_scale_batch_size", "use_tf_data", "prefetch", "memory_growth"):
            _require(isinstance(getattr(self, name), bool), f"{name} harus boolean.")
        _require(
            self.cross_device_ops in ("auto", "nccl", "hierarchical", "reduction_to_one"),
            "cross_device_ops harus auto, nccl, hierarchical, atau reduction_to_one.",
        )
        _require(self.steps_per_execution >= 1, "steps_per_execution harus >= 1.")
        if self.monitor == "val_loss":
            _require(self.monitor_mode == "min", "monitor val_loss harus memakai mode 'min'.")
        else:
            _require(self.monitor_mode == "max", f"monitor {self.monitor} harus memakai mode 'max'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "BiLSTMConfig":
        return _from_dict(cls, params)


@dataclass
class BertConfig:
    """Hiperparameter fine-tuning IndoBERT (Hugging Face Trainer)."""

    model_name: str = "indobenchmark/indobert-base-p1"
    max_len: int = 256
    epochs: int = 4
    learning_rate: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw_torch"
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    freeze_layers: int = 0  # 0 = semua dilatih, -1 = embeddings saja, N = N layer awal
    use_class_weight: bool = False
    label_smoothing_factor: float = 0.0
    hidden_dropout_prob: float | None = None
    attention_probs_dropout_prob: float | None = None
    classifier_dropout: float | None = None
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    group_by_length: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = False
    ddp_find_unused_parameters: bool = False
    ddp_backend: str | None = None
    ddp_timeout: int = 1800
    torch_compile: bool = False
    auto_find_batch_size: bool = False
    save_total_limit: int = 2
    logging_steps: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        _require(bool(self.model_name), "model_name tidak boleh kosong.")
        _require(self.max_len > 0, "max_len harus > 0.")
        _require(self.epochs > 0, "epochs harus > 0.")
        _require(self.learning_rate > 0, "learning_rate harus > 0.")
        _require(self.batch_size > 0, "batch_size harus > 0.")
        _require(self.eval_batch_size > 0, "eval_batch_size harus > 0.")
        _require(self.weight_decay >= 0, "weight_decay tidak boleh negatif.")
        _check_fraction("warmup_ratio", self.warmup_ratio)
        _require(
            self.lr_scheduler_type
            in (
                "linear", "cosine", "cosine_with_restarts", "polynomial",
                "constant", "constant_with_warmup", "inverse_sqrt",
            ),
            "lr_scheduler_type tidak dikenal.",
        )
        _require(
            self.optimizer in ("adamw_torch", "adamw_torch_fused", "adafactor", "sgd"),
            "optimizer harus adamw_torch, adamw_torch_fused, adafactor, atau sgd.",
        )
        _require(self.max_grad_norm > 0, "max_grad_norm harus > 0.")
        _require(
            self.gradient_accumulation_steps > 0,
            "gradient_accumulation_steps harus > 0.",
        )
        _require(self.early_stopping_patience >= 0, "early_stopping_patience tidak boleh negatif.")
        _require(
            self.metric_for_best_model
            in ("eval_loss", "eval_f1", "eval_roc_auc", "eval_average_precision", "eval_accuracy"),
            "metric_for_best_model tidak dikenal.",
        )
        _require(self.freeze_layers >= -1, "freeze_layers harus -1, 0, atau bilangan positif.")
        _check_fraction("label_smoothing_factor", self.label_smoothing_factor)
        for name in (
            "hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout",
        ):
            value = getattr(self, name)
            if value is not None:
                _check_fraction(name, float(value))
        _require(self.precision in ("fp32", "fp16", "bf16"), "precision harus fp32, fp16, atau bf16.")
        # greater_is_better diturunkan dari nama metrik agar tidak bisa tidak konsisten.
        self.greater_is_better = not self.metric_for_best_model.endswith("loss")
        _require(self.dataloader_num_workers >= 0, "dataloader_num_workers tidak boleh negatif.")
        for name in (
            "dataloader_pin_memory", "dataloader_persistent_workers", "tf32",
            "gradient_checkpointing", "ddp_find_unused_parameters",
            "torch_compile", "auto_find_batch_size",
        ):
            _require(isinstance(getattr(self, name), bool), f"{name} harus boolean.")
        _require(self.ddp_timeout > 0, "ddp_timeout harus > 0.")
        _require(
            self.ddp_backend in (None, "nccl", "gloo", "mpi", "ccl", "hccl"),
            "ddp_backend harus None, nccl, gloo, mpi, ccl, atau hccl.",
        )
        _require(self.save_total_limit >= 0, "save_total_limit tidak boleh negatif.")
        _require(self.logging_steps > 0, "logging_steps harus > 0.")
        _require(self.seed >= 0, "seed tidak boleh negatif.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "BertConfig":
        return _from_dict(cls, params)


@dataclass
class HPOConfig:
    """Pengaturan orkestrasi pencarian hiperparameter."""

    strategy: Literal["grid", "random", "bayesian"] = "random"
    n_trials: int = 30
    seed: int = 42
    timeout_minutes: float | None = None
    objective_metric: str = "val_f1"
    objective_mode: Literal["max", "min"] = "max"
    output_dir: str = "./artifacts/hpo"
    study_name: str = "bilstm"
    resume: bool = True
    save_trial_models: bool = False
    max_grid_combinations: int = 200
    log_every_trial: bool = True
    forbid_holdout_metrics: bool = True

    def __post_init__(self) -> None:
        _require(
            self.strategy in ("grid", "random", "bayesian"),
            "strategy harus grid, random, atau bayesian.",
        )
        _require(self.n_trials > 0, "n_trials harus > 0.")
        _require(self.seed >= 0, "seed tidak boleh negatif.")
        if self.timeout_minutes is not None:
            _require(self.timeout_minutes > 0, "timeout_minutes harus > 0 atau None.")
        _require(bool(self.objective_metric), "objective_metric tidak boleh kosong.")
        _require(
            not self.objective_metric.lower().startswith(("holdout", "test")),
            "objective_metric tidak boleh memakai holdout/test; "
            "HPO hanya boleh memilih berdasarkan validation.",
        )
        _require(self.objective_mode in ("max", "min"), "objective_mode harus 'max' atau 'min'.")
        _require(bool(self.study_name), "study_name tidak boleh kosong.")
        _require(self.max_grid_combinations > 0, "max_grid_combinations harus > 0.")
        _require(
            isinstance(self.forbid_holdout_metrics, bool),
            "forbid_holdout_metrics harus boolean.",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "HPOConfig":
        return _from_dict(cls, params)


def _from_dict(cls: type, params: dict[str, Any]):
    """Bangun dataclass dari dict, tolak key tak dikenal agar typo ketahuan."""
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(params) - valid)
    if unknown:
        raise ConfigError(
            f"Parameter tidak dikenal untuk {cls.__name__}: {unknown}. "
            f"Parameter valid: {sorted(valid)}"
        )
    return cls(**params)


# Alias yang lebih ramah untuk notebook.
BilstmConfig = BiLSTMConfig
