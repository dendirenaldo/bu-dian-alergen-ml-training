"""Composition text extraction from product label layout."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytesseract

from app.core.ocr.engine import _ocr_lines
from app.core.preprocessing.text import normalize_text

logger = logging.getLogger(__name__)


def parse_composition_from_layout(path: str, processed_img: np.ndarray) -> str:
    """Extract composition text from a product label image using layout analysis.

    Finds an anchor keyword (komposisi/composition/ingredients) and collects
    lines below it until a stop pattern is hit or vertical gap exceeds threshold.

    Args:
        path: Image file path (for fallback OCR).
        processed_img: Preprocessed image array.

    Returns:
        Extracted composition text string.
    """
    lines = _ocr_lines(processed_img, lang="eng", min_conf=35)
    if not lines:
        return ""

    anchor_pattern = re.compile(r"\b(komposisi|composition|ingredients?)\b", re.I)
    stop_pattern = re.compile(
        r"\b(informasi nilai gizi|nutrition|takaran saji|energi total|cara penyimpanan|penyimpanan|"
        r"netto|berat bersih|expired|kedaluwarsa|bpom|kode produksi|saran penyajian|perhatian)\b",
        re.I,
    )

    anchors = [ln for ln in lines if anchor_pattern.search(ln["text"])]
    if not anchors:
        # Fallback: use full OCR text
        full_text = pytesseract.image_to_string(processed_img, lang="eng")
        return normalize_text(full_text)

    anchor = anchors[0]
    avg_h = np.mean([ln["height"] for ln in lines]) if lines else 20
    max_gap = max(18, int(avg_h * 1.8))

    selected: list[str] = []
    started = False
    prev_bottom: int | None = None

    for ln in lines:
        if ln["top"] < anchor["top"]:
            continue

        # Start when anchor line is found
        if not started and ln is anchor:
            started = True
            selected.append(ln["text"])
            prev_bottom = ln["bottom"]
            continue

        if not started:
            continue

        # Stop if vertical gap is too large
        if prev_bottom is not None and (ln["top"] - prev_bottom) > max_gap:
            break

        # Prefer lines in the same horizontal area as anchor
        horizontal_overlap = not (
            ln["right"] < anchor["left"] - 100 or ln["left"] > anchor["right"] + 900
        )
        near_anchor_column = abs(ln["left"] - anchor["left"]) <= 220
        if not (horizontal_overlap or near_anchor_column):
            continue

        if stop_pattern.search(ln["text"]):
            break

        selected.append(ln["text"])
        prev_bottom = ln["bottom"]

    comp = normalize_text(" ".join(selected))
    comp = re.sub(
        r"(?i)^\s*(komposisi|composition|ingredients?)\s*[:\-]?\s*", "", comp
    ).strip()
    return comp
