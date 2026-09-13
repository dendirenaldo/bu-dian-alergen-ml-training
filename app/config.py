"""Configuration module for ML Training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # env ringan tanpa python-dotenv (mis. CI minimal)

    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

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
class V5Config:
    """Kontrak eksperimen V5 leakage-safe (port Notebook Cell 25/26/35).

    Nilai default = angka notebook untuk pool 114 real.
    Untuk pool berukuran lain (mis. 499 CSV lokal), isi holdout/val
    eksplisit via env/CLI atau biarkan solver me-raise dengan pesan jelas.
    """

    # --- Split contract FINAL (revisi Fase 4, split lama 23 kedaluwarsa) ---
    # val/holdout 50 (25/25) agar metrik stabil: 1 sampel = 2% (dulu 5.6%).
    # Derivation deterministik + audit leakage PASS, di-freeze ulang.
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
    # Jika None -> pakai daftar notebook 23 (ada di frozen_holdout_default.json).
    # Jika path JSON ada -> kunci dari file itu (re-derive lalu freeze).
    frozen_holdout_path: str | None = field(
        default_factory=lambda: os.getenv("V5_FROZEN_HOLDOUT_PATH") or None
    )

    # --- Synthetic (notebook Cell 19/40) ---
    synthetic_total: int = field(
        default_factory=lambda: int(os.getenv("V5_SYNTHETIC_TOTAL", "1000"))
    )
    synthetic_source_contract: str = "real_train_only"

    # --- Threshold (notebook: FIXED, never tuned) ---
    fixed_threshold: float = field(
        default_factory=lambda: float(os.getenv("V5_FIXED_THRESHOLD", "0.5"))
    )

    # --- NLP parity (notebook Cell 43) ---
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
    # Deviasi D3: mask_zero default True (terbaik hasil Fase 2-3).
    # Bukti full-run 20ep: mask (S2) val_loss 0.063 + val AUC 1.0 vs
    # tanpa-mask (S1) val_loss 0.166 + AUC 0.969. Padding 71-85%
    # pada MAX_LEN=120 adalah noise bagi LSTM bila tidak di-mask.
    # Notebook parity (False) tetap bisa via V5_MASK_ZERO=0.
    mask_zero: bool = field(
        default_factory=lambda: os.getenv("V5_MASK_ZERO", "1") == "1"
    )
    digit_fold: bool = field(
        default_factory=lambda: os.getenv("V5_DIGIT_FOLD", "0") == "1"
    )

    # --- Model parity (notebook Cell 25/45, single-run, no tuning) ---
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_1: float = 0.3
    dropout_2: float = 0.3
    dense_units: int = 64
    dropout_dense: float = 0.2
    # Deviasi stabilitas D2: LR default 1e-4, bukan 1e-3 notebook.
    # Bukti: run baseline LR 1e-3 collapse di epoch 2 (train acc 0.57->0.48,
    # model flip all-unsafe->all-safe, prob holdout std 0.0008 = degenerat;
    # clipping saja tidak menyembuhkan). Diag LR 1e-4: learning monoton
    # sehat 4 epoch (train acc 0.47->0.68, val AUC ~0.95, tanpa collapse).
    # Semua kontrak V5 lain dipertahankan (split/synthetic/threshold/audit).
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("V5_LEARNING_RATE", "1e-4"))
    )
    batch_size: int = 16
    epochs: int = field(
        default_factory=lambda: int(os.getenv("V5_EPOCHS", "20"))
    )
    early_stopping_patience: int = 4
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    train_shuffle: bool = False
    # Deviasi Fase 3: pre-shuffle deterministik SEKALI sebelum fit.
    # shuffle=False + urutan real‖synthetic membuat gradien berosilasi
    # mengikuti blok distribusi. Pre-shuffle dengan RandomState(seed)
    # mencampur blok namun 100% reproducible (bukan shuffle acak TF).
    # V5_SHUFFLE=0 -> parity notebook (tanpa shuffle).
    preshuffle: bool = field(
        default_factory=lambda: os.getenv("V5_SHUFFLE", "0") == "1"
    )
    # Deviasi stabilitas D1: gradient clipping default 1.0.
    # Bukti: run baseline LR 1e-3 tanpa clip collapse di epoch 2
    # (model flip all-unsafe->all-safe). Legacy pipeline memakai 1.0.
    gradient_clip_norm: float | None = field(
        default_factory=lambda: (
            float(os.getenv("V5_GRADIENT_CLIP_NORM"))
            if os.getenv("V5_GRADIENT_CLIP_NORM")
            else 1.0
        )
    )

    # --- Kolom ---
    product_col: str = field(
        default_factory=lambda: os.getenv("V5_PRODUCT_COL", "nama_produk")
    )
    text_col: str = field(default_factory=lambda: os.getenv("V5_TEXT_COL", "text"))
    gold_label_col: str = field(
        default_factory=lambda: os.getenv("V5_GOLD_LABEL_COL", "gold_label")
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
    # --- Pipeline mode: 'legacy' (tuning+threshold-tuning) | 'v5_parity' (ikuti notebook) ---
    mode: str = field(
        default_factory=lambda: os.getenv("TRAIN_MODE", "legacy").lower()
    )
    v5: V5Config = field(default_factory=V5Config)

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Return a default Config instance."""
    return Config()
