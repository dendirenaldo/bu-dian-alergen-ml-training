"""Configuration module for ML Training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration extracted from the notebook's PANEL KONFIGURASI UTAMA."""

    # --- Paths ---
    dataset_dir: str = field(default_factory=lambda: os.getenv("DATASET_DIR", "./dataset"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./output"))
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "./models"))
    csv_input: str = field(default_factory=lambda: os.getenv("CSV_INPUT", "./ocr_output/data-mengandung.csv"))

    # --- Data source ---
    data_source_mode: str = "CSV"  # 'OCR' or 'CSV'
    text_col: str = "text"
    label_col: str = "label"
    seed: int = int(os.getenv("SEED", "42"))

    # --- Data ---
    test_size: float = 0.2
    max_len: int = 120

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

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Return a default Config instance."""
    return Config()
