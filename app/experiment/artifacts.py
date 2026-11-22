"""Utilitas artefak eksperimen: run directory, provenance, dan writer.

Modul ini sengaja **tidak** mengimpor TensorFlow, Torch, atau Gensim.
Tujuannya agar bisa dipakai di test ringan dan di notebook setup.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def utc_timestamp() -> str:
    """Timestamp UTC ringkas untuk nama direktori run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(base_dir: str | os.PathLike, prefix: str, seed: int | None = None) -> Path:
    """Buat direktori run unik, mis. ``artifacts/runs/bilstm_20260101T120000Z_seed42``.

    Args:
        base_dir: direktori induk (dibuat bila belum ada).
        prefix: nama eksperimen, mis. ``bilstm_hpo``.
        seed: seed yang dipakai (ditulis di nama agar mudah dicari).

    Returns:
        Path direktori run yang sudah dibuat.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{utc_timestamp()}"
    if seed is not None:
        stem += f"_seed{seed}"
    candidate = base / stem
    counter = 2
    while candidate.exists():
        candidate = base / f"{stem}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def safe_filename(value: str, max_length: int = 80) -> str:
    """Ubah teks menjadi nama file yang aman (huruf, angka, dash, underscore)."""
    cleaned = "".join(
        ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_"
        for ch in str(value).strip()
    )
    cleaned = cleaned.strip("._") or "item"
    return cleaned[:max_length]


def to_jsonable(obj: Any) -> Any:
    """Konversi tipe numerik/NumPy/pandas umum menjadi tipe JSON-native.

    Fungsi ini tidak mengimpor NumPy; cukup memeriksa atribut ``item``.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:
            pass
    return str(obj)


def write_json(path: str | os.PathLike, payload: Any, indent: int = 2) -> str:
    """Tulis JSON dengan konversi tipe otomatis dan encoding UTF-8."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=indent, ensure_ascii=False)
    return str(target)


def read_json(path: str | os.PathLike) -> Any:
    """Baca JSON UTF-8."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    """Tambahkan satu baris JSONL (aman untuk resume/trial log)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Baca seluruh JSONL; baris rusak dilewati dengan warning di caller."""
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_csv(path: str | os.PathLike, rows: Iterable[dict], columns: list[str] | None = None) -> str:
    """Tulis daftar dict menjadi CSV tanpa ketergantungan pandas."""
    import csv

    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in columns})
    return str(target)


def params_hash(params: dict) -> str:
    """Hash kanonis parameter trial (untuk resume dan deduplikasi)."""
    payload = json.dumps(to_jsonable(params), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions(names: Iterable[str] | None = None) -> dict[str, str]:
    """Ambil versi paket tanpa mengimpor paketnya (aman & cepat)."""
    if names is None:
        names = (
            "numpy", "pandas", "scikit-learn", "tensorflow", "gensim",
            "torch", "transformers", "accelerate", "matplotlib", "seaborn",
            "optuna", "jupyterlab", "nbformat",
        )
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def git_sha(cwd: str | os.PathLike | None = None) -> str:
    """Ambil commit git singkat; ``unknown`` bila bukan repo git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def gpu_snapshot(timeout: float = 3.0) -> dict:
    """Coba baca GPU via ``nvidia-smi`` tanpa mengimpor Torch/TensorFlow."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi tidak ditemukan"}
    try:
        out = subprocess.check_output(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        ).decode().strip()
        devices = [line.strip() for line in out.splitlines() if line.strip()]
        return {"available": bool(devices), "devices": devices}
    except Exception as exc:  # pragma: no cover - bergantung mesin
        return {"available": False, "reason": str(exc)}


def reproducibility_warnings(seed: int | None = None) -> list[str]:
    """Daftar peringatan reproducibility yang tidak boleh disembunyikan.

    ``PYTHONHASHSEED`` hanya efektif bila diset **sebelum** interpreter
    Python berjalan. Menyetel ``os.environ`` di dalam program tidak
    menonaktifkan hash randomization. Karena itu kita periksa flag proses.
    """
    warnings: list[str] = []
    if sys.flags.hash_randomization:
        warnings.append(
            "Hash randomization masih aktif. Jalankan dari shell dengan "
            "`PYTHONHASHSEED=<seed> python ...` agar urutan hash benar-benar tetap."
        )
    if seed is not None:
        if os.environ.get("PYTHONHASHSEED") != str(seed):
            warnings.append(
                f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r} tidak sama dengan seed {seed}."
            )
    return warnings


def environment_snapshot(cwd: str | os.PathLike | None = None, seed: int | None = None) -> dict:
    """Rekam environment minimum untuk reproduksibilitas dan audit paper."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_sha": git_sha(cwd),
        "cwd": str(Path(cwd or os.getcwd()).resolve()),
        "packages": package_versions(),
        "gpu": gpu_snapshot(),
        "reproducibility_warnings": reproducibility_warnings(seed),
        "env": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED", "TF_DETERMINISTIC_OPS", "TF_CUDNN_DETERMINISTIC",
                "CUBLAS_WORKSPACE_CONFIG", "TF_ENABLE_ONEDNN_OPTS",
                "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE",
            )
        },
    }
