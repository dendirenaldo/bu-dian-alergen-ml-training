"""Configuration module for ML Training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class BertConfig:
    """Konfigurasi khusus BERT (pluggable via model_name)."""

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "BERT_MODEL_NAME", "indobenchmark/indobert-base-p1"
        )
    )
    max_len: int = field(
        default_factory=lambda: int(os.getenv("BERT_MAX_LEN", "256"))
    )
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("BERT_LR", "2e-5"))
    )
    epochs: int = field(default_factory=lambda: int(os.getenv("BERT_EPOCHS", "4")))
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BERT_BATCH_SIZE", "16"))
    )
    weight_decay: float = field(
        default_factory=lambda: float(os.getenv("BERT_WEIGHT_DECAY", "0.01"))
    )
    warmup_ratio: float = field(
        default_factory=lambda: float(os.getenv("BERT_WARMUP_RATIO", "0.1"))
    )
    freeze_layers: int = field(
        default_factory=lambda: int(os.getenv("BERT_FREEZE_LAYERS", "0"))
    )
    dropout: float = field(
        default_factory=lambda: float(os.getenv("BERT_DROPOUT", "0.1"))
    )
    # Ruang pencarian tuning BERT (kecil, karena mahal).
    lr_options: list = field(default_factory=lambda: [2e-5, 3e-5, 5e-5])
    batch_size_options: list = field(default_factory=lambda: [8, 16])
    num_trials: int = field(
        default_factory=lambda: int(os.getenv("BERT_NUM_TRIALS", "3"))
    )


@dataclass
class Config:
    """Central configuration extracted from the notebook's PANEL KONFIGURASI UTAMA."""

    # --- Paths ---
    dataset_dir: str = field(default_factory=lambda: os.getenv("DATASET_DIR", "./dataset"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./output"))
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "./models"))
    csv_input: str = field(default_factory=lambda: os.getenv("CSV_INPUT", "./ocr_output/data-mengandung.csv"))

    # --- Data source ---
    data_source_mode: str = field(
        default_factory=lambda: os.getenv("DATA_SOURCE_MODE", "CSV")
    )
    text_col: str = field(default_factory=lambda: os.getenv("TEXT_COL", "text"))
    label_col: str = field(default_factory=lambda: os.getenv("LABEL_COL", "label"))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    csv_delimiter: str = field(
        default_factory=lambda: os.getenv("CSV_DELIMITER", ";")
    )
    csv_encoding: str = field(
        default_factory=lambda: os.getenv("CSV_ENCODING", "utf-8")
    )

    # --- Data ---
    test_size: float = field(
        default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2"))
    )
    val_size: float = field(
        default_factory=lambda: float(os.getenv("VAL_SIZE", "0.15"))
    )
    max_len: int = field(default_factory=lambda: int(os.getenv("MAX_LEN", "120")))

    # --- Word2Vec ---
    vocab_size: int = 20000
    embed_dim: int = 100
    min_word_count: int = 3
    w2v_window: int = 5
    w2v_epochs: int = 30
    embed_trainable: bool = False

    # --- Model ---
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_rate_1: float = 0.3
    dropout_rate_2: float = 0.3
    recurrent_dropout: float = 0.2  # set to 0 on GPU (cuDNN compat)
    dense_units: int = 64
    dropout_rate_dense: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 150

    # --- Tuning ---
    tuning_val_split: float = 0.15
    gradient_clip_norm: float = 1.0
    use_class_weight: bool = True

    # --- Tuning search space ---
    batch_size_options: list = field(default_factory=lambda: [16, 32, 64])
    lstm_units_options: list = field(default_factory=lambda: [[32, 64], [16, 32], [64, 64]])
    dropout_options: list = field(default_factory=lambda: [[0.1, 0.1], [0.3, 0.3], [0.0, 0.0]])
    tuning_epochs_options: list = field(default_factory=lambda: [50, 75, 100])
    lr_options: list = field(default_factory=lambda: [1e-4, 1e-3])
    num_trials: int = 10

    # --- Callbacks ---
    early_stopping_on: bool = True
    early_stopping_patience: int = 5
    lr_reduce_on: bool = True
    lr_reduce_patience: int = 3
    min_lr: float = 1e-6

    # --- Image ---
    image_extensions: set = field(default_factory=lambda: {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

    # --- Model selection (dual-model BiLSTM + BERT) ---
    # 'bilstm' | 'lstm' | 'bert' | 'all'
    model_type: str = field(
        default_factory=lambda: os.getenv("MODEL_TYPE", "all").lower()
    )
    bert: BertConfig = field(default_factory=BertConfig)
    # Threshold default; nilai final di-tuning di validation set per model.
    default_threshold: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    )

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Return a default Config instance."""
    return Config()
