"""Configuration module for ML Training pipeline (single pipeline: BiLSTM V5).

Satu-satunya model config adalah V5Config (kontrak leakage-safe final).
BERT dilatih di HPC (artifacts/bert/) dengan kontrak split yang sama.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # env ringan tanpa python-dotenv (mis. CI minimal)

    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

load_dotenv()


def _env_flag(name: str, default: str) -> bool:
    """Parse env boolean '1'/'0' (default bila kosong)."""
    return (os.getenv(name, default) or default) == "1"


def _parse_clip_norm() -> float | None:
    """Parse V5_GRADIENT_CLIP_NORM: kosong -> 1.0; '0'/'none' -> None (disable)."""
    raw = (os.getenv("V5_GRADIENT_CLIP_NORM") or "").strip().lower()
    if raw in ("",):
        return 1.0
    if raw in ("0", "none", "off"):
        return None
    return float(raw)


@dataclass
class V5Config:
    """Kontrak eksperimen leakage-safe FINAL (split 50/50/399).

    Deviasi terdokumentasi dari notebook (bukti di komentar tiap field):
    D1 clip=1.0, D2 LR=1e-4, D3 mask_zero=True.
    """

    # --- Split contract (50 real holdout + 50 real val, 25/25) ---
    holdout_size: int = field(
        default_factory=lambda: int(os.getenv("V5_HOLDOUT_SIZE", "50"))
    )
    holdout_safe: int = field(
        default_factory=lambda: int(os.getenv("V5_HOLDOUT_SAFE", "25"))
    )
    holdout_unsafe: int = field(
        default_factory=lambda: int(os.getenv("V5_HOLDOUT_UNSAFE", "25"))
    )
    val_safe: int = field(default_factory=lambda: int(os.getenv("V5_VAL_SAFE", "25")))
    val_unsafe: int = field(
        default_factory=lambda: int(os.getenv("V5_VAL_UNSAFE", "25"))
    )
    # Path JSON frozen holdout. None -> default notebook23 bila n=114,
    # re-derive deterministik bila n!=114 (lalu freeze hasilnya).
    frozen_holdout_path: str | None = field(
        default_factory=lambda: os.getenv("V5_FROZEN_HOLDOUT_PATH") or None
    )

    # --- Synthetic (HANYA dari real-train) ---
    synthetic_total: int = field(
        default_factory=lambda: int(os.getenv("V5_SYNTHETIC_TOTAL", "1000"))
    )
    synthetic_source_contract: str = "real_train_only"

    # --- Threshold FIXED (never tuned; assert 0.5 di trainer) ---
    fixed_threshold: float = field(
        default_factory=lambda: float(os.getenv("V5_FIXED_THRESHOLD", "0.5"))
    )

    # --- NLP ---
    vocab_size: int = 20000
    max_len: int = field(
        default_factory=lambda: int(os.getenv("V5_MAX_LEN", "120"))
    )
    embed_dim: int = 100
    w2v_window: int = 5
    w2v_min_count: int = field(
        default_factory=lambda: int(os.getenv("V5_W2V_MIN_COUNT", "1"))
    )
    w2v_epochs: int = 20
    w2v_seed: int = 42
    embedding_init_scale: float = 0.6
    embed_trainable: bool = True
    mask_zero: bool = field(
        default_factory=lambda: _env_flag("V5_MASK_ZERO", "1")
    )
    digit_fold: bool = field(
        default_factory=lambda: _env_flag("V5_DIGIT_FOLD", "0")
    )

    # --- Model (single-run, no tuning) ---
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_1: float = 0.3
    dropout_2: float = 0.3
    dense_units: int = 64
    dropout_dense: float = 0.2
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("V5_LEARNING_RATE", "1e-4"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("V5_BATCH_SIZE", "16"))
    )
    epochs: int = field(
        default_factory=lambda: int(os.getenv("V5_EPOCHS", "20"))
    )
    early_stopping_patience: int = 4
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    train_shuffle: bool = False
    # Pre-shuffle deterministik SEKALI sebelum fit (campur blok real‖synthetic,
    # tetap reproducible via RandomState(seed)). Off = parity notebook.
    preshuffle: bool = field(
        default_factory=lambda: _env_flag("V5_SHUFFLE", "0")
    )
    gradient_clip_norm: float | None = field(
        default_factory=_parse_clip_norm
    )

    # --- Kolom CSV sumber ---
    product_col: str = field(
        default_factory=lambda: os.getenv("V5_PRODUCT_COL", "nama_produk")
    )
    text_col: str = field(default_factory=lambda: os.getenv("V5_TEXT_COL", "text"))


@dataclass
class Config:
    """Infra + CSV input. Semua hiperparameter model ada di V5Config."""

    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./artifacts/bilstm/output"))
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "./artifacts/bilstm/models"))
    csv_input: str = field(default_factory=lambda: os.getenv("CSV_INPUT", "./ocr_output/data-mengandung.csv"))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    csv_delimiter: str = field(
        default_factory=lambda: os.getenv("CSV_DELIMITER", ";")
    )
    csv_encoding: str = field(
        default_factory=lambda: os.getenv("CSV_ENCODING", "utf-8")
    )
    # Kolom CSV mentah (dipetakan ke kolom V5 di trainer bila beda).
    text_col: str = field(default_factory=lambda: os.getenv("TEXT_COL", "text"))
    label_col: str = field(default_factory=lambda: os.getenv("LABEL_COL", "label"))

    v5: V5Config = field(default_factory=V5Config)

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Return a default Config instance."""
    return Config()
