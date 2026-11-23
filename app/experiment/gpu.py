"""Utilitas GPU untuk training BiLSTM dan pengecekan proses terdistribusi.

Semua import TensorFlow/Accelerate dilakukan **lazy** di dalam fungsi agar
modul ini tetap bisa diimpor di environment tanpa GPU/TensorFlow.
"""

from __future__ import annotations

import os
from typing import Any

_GPU_CONFIGURED = False
_STRATEGY = None
_STRATEGY_KEY: tuple[bool, int] | None = None


def configure_gpu(memory_growth: bool = True) -> dict[str, Any]:
    """Aktifkan memory growth dan kembalikan ringkasan GPU.

    Memory growth mencegah TensorFlow mengambil seluruh VRAM di awal,
    sehingga GPU lain / proses lain tetap bisa memakai memori.
    """
    global _GPU_CONFIGURED
    try:
        import tensorflow as tf
    except ImportError:
        return {"gpu_count": 0, "gpu_names": [], "memory_growth": False}

    gpus = tf.config.list_physical_devices("GPU")
    names = [getattr(gpu, "name", str(gpu)) for gpu in gpus]
    if gpus and memory_growth and not _GPU_CONFIGURED:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Sudah diinisialisasi proses lain / driver menolak; abaikan.
                pass
        _GPU_CONFIGURED = True
    return {
        "gpu_count": len(gpus),
        "gpu_names": names,
        "memory_growth": bool(gpus and memory_growth and _GPU_CONFIGURED),
    }


def gpu_summary() -> dict[str, Any]:
    """Ringkasan GPU + policy mixed precision saat ini."""
    try:
        import tensorflow as tf
    except ImportError:
        return {"gpu_count": 0, "gpu_names": [], "mixed_precision": "float32"}

    gpus = tf.config.list_physical_devices("GPU")
    return {
        "gpu_count": len(gpus),
        "gpu_names": [getattr(gpu, "name", str(gpu)) for gpu in gpus],
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
    }


def set_mixed_precision(policy: str = "mixed_float16") -> str:
    """Set policy mixed precision global.

    ``mixed_float16`` biasanya memberi throughput lebih tinggi dan memori
    lebih hemat di GPU NVIDIA. Output layer model tetap float32.
    """
    import tensorflow as tf

    allowed = {"float32", "mixed_float16", "mixed_bfloat16"}
    if policy not in allowed:
        raise ValueError(f"mixed_precision harus salah satu: {sorted(allowed)}")
    tf.keras.mixed_precision.set_global_policy(policy)
    return tf.keras.mixed_precision.global_policy().name


def get_strategy(multi_gpu: bool = True):
    """Kembalikan strategy distribusi (di-cache agar HPO tidak re-init NCCL).

    MirroredStrategy melakukan all-reduce gradien antar GPU (data parallel).
    Untuk satu GPU, default strategy menghindari overhead distribusi.
    """
    global _STRATEGY, _STRATEGY_KEY
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    key = (bool(multi_gpu), len(gpus))
    if _STRATEGY is None or _STRATEGY_KEY != key:
        if multi_gpu and len(gpus) > 1:
            _STRATEGY = tf.distribute.MirroredStrategy()
        else:
            _STRATEGY = tf.distribute.get_strategy()
        _STRATEGY_KEY = key
    return _STRATEGY


def replica_count(strategy: Any) -> int:
    """Jumlah replika (GPU) pada strategy."""
    return int(getattr(strategy, "num_replicas_in_sync", 1))


def effective_batch_size(
    config_batch_size: int,
    replicas: int,
    auto_scale: bool = True,
) -> int:
    """Hitung batch global.

    ``BiLSTMConfig.batch_size`` diperlakukan sebagai **batch per GPU**.
    Dengan ``auto_scale=True`` dan 2 GPU, batch global menjadi 2x.
    """
    if config_batch_size <= 0:
        raise ValueError("batch_size harus > 0.")
    replicas = max(1, int(replicas))
    return int(config_batch_size * replicas) if auto_scale else int(config_batch_size)


def is_main_process() -> bool:
    """True hanya di rank 0 (untuk menulis artefak DDP)."""
    try:
        from accelerate.state import PartialState

        return bool(PartialState().is_main_process)
    except Exception:
        return int(os.environ.get("RANK", "0")) == 0


def distributed_world_size() -> int:
    """Jumlah proses DDP/accelerate; 1 bila single-process."""
    try:
        from accelerate.state import PartialState

        return int(PartialState().num_processes)
    except Exception:
        try:
            return max(1, int(os.environ.get("WORLD_SIZE", "1")))
        except ValueError:
            return 1
