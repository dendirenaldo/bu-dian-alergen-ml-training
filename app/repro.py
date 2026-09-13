"""Reproducibility lock — port dari Notebook V5 Cell 1.

PENTING: panggil ``apply_reproducibility_lock()`` SEBELUM import TensorFlow
agar env determinism (PYTHONHASHSEED, TF_DETERMINISTIC_OPS, dst.) berlaku.
Modul ini sendiri TIDAK import TF di top-level sehingga aman diimpor di mana saja.
"""

from __future__ import annotations

import os
import random


def apply_reproducibility_lock(seed: int = 42) -> dict:
    """Set env determinism (seperti Notebook V5 Cell 1).

    Mengikuti notebook: set sebelum import TF.
    Menggunakan ``setdefault`` agar tidak menimpa setting eksplisit user,
    KECUALI PYTHONHASHSEED yang memang harus == seed untuk reproducibility.

    Returns:
        dict env yang dipasang.
    """
    env = {
        "PYTHONHASHSEED": str(seed),
        "TF_DETERMINISTIC_OPS": "1",
        "TF_CUDNN_DETERMINISTIC": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TF_ENABLE_ONEDNN_OPTS": "0",
    }
    # PYTHONHASHSEED dipaksa (kontrak notebook), sisanya setdefault.
    os.environ["PYTHONHASHSEED"] = str(seed)
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return {k: os.environ.get(k) for k in env}


def seed_all(seed: int = 42) -> None:
    """Seed ulang semua RNG user-space (dipanggil ulang tiap tahap, ala notebook).

    - ``random`` + ``numpy`` selalu di-seed.
    - ``tensorflow`` di-seed bila tersedia (import lazy agar modul ini
      tetap bisa diimpor tanpa TF, mis. saat unit test ringan).
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def reproducibility_status(seed: int = 42) -> dict:
    """Status lock untuk audit (tanpa me-raise)."""
    return {
        "SEED": seed,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
        "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
    }
