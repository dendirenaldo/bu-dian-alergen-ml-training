"""Validasi notebook tanpa menjalankan training berat."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED = {
    "00_panduan_dan_setup.ipynb",
    "01_data_dan_split.ipynb",
    "02_bilstm_dasar.ipynb",
    "03_bilstm_tuning.ipynb",
    "04_indobert_dasar.ipynb",
    "05_indobert_tuning.ipynb",
    "06_perbandingan_dan_laporan.ipynb",
}


def test_notebook_lengkap():
    names = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert EXPECTED <= names


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_notebook_json_dan_sintaks(name: str):
    path = NOTEBOOK_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["cells"], "notebook tidak boleh kosong"
    code_cells = 0
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        source = "".join(cell["source"])
        compile(source, f"{name}:cell{index}", "exec")
    assert code_cells > 0, f"{name} tidak punya code cell"


def test_notebook_punya_markdown_pembuka():
    for name in EXPECTED:
        payload = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
        first = payload["cells"][0]
        assert first["cell_type"] == "markdown"
        assert "".join(first["source"]).strip().startswith("#")


def test_setup_menyediakan_import_umum():
    """Semua notebook harus punya pandas/matplotlib di sel setup pertama."""
    for name in EXPECTED:
        payload = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
        code_cells = [c for c in payload["cells"] if c["cell_type"] == "code"]
        setup = "".join(code_cells[0]["source"])
        assert "import pandas as pd" in setup, name
        assert "import matplotlib.pyplot as plt" in setup, name
        assert "import numpy as np" in setup, name
