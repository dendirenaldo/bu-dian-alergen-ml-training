"""OCR engine using Tesseract."""

import logging
from typing import Any

import numpy as np
import pytesseract

from app.core.preprocessing.text import normalize_text, clean_phrase

logger = logging.getLogger(__name__)


def _ocr_lines(
    image_for_ocr: np.ndarray,
    lang: str = "eng",
    min_conf: float = 35.0,
) -> list[dict[str, Any]]:
    """Extract text lines from an image using Tesseract OCR.

    Groups words by block/paragraph/line and returns structured line data.

    Args:
        image_for_ocr: Preprocessed image (numpy array).
        lang: Tesseract language code.
        min_conf: Minimum confidence threshold for word inclusion.

    Returns:
        List of line dicts with keys: text, norm, left, right, top, bottom,
        height, conf, block.
    """
    data = pytesseract.image_to_data(
        image_for_ocr, lang=lang, output_type=pytesseract.Output.DICT
    )
    grouped: dict[tuple, list[dict]] = {}

    for i in range(len(data["text"])):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            continue
        if conf < min_conf:
            continue

        block = int(data["block_num"][i])
        par = int(data["par_num"][i])
        line = int(data["line_num"][i])
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        key = (block, par, line)
        grouped.setdefault(key, []).append(
            {
                "word": word,
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
                "height": height,
                "conf": conf,
                "block": block,
            }
        )

    lines: list[dict[str, Any]] = []
    for key, words in grouped.items():
        words = sorted(words, key=lambda x: x["left"])
        text = normalize_text(" ".join(w["word"] for w in words))
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "norm": clean_phrase(text),
                "left": min(w["left"] for w in words),
                "right": max(w["right"] for w in words),
                "top": min(w["top"] for w in words),
                "bottom": max(w["bottom"] for w in words),
                "height": max(w["height"] for w in words),
                "conf": float(np.mean([w["conf"] for w in words])),
                "block": words[0]["block"],
            }
        )

    lines.sort(key=lambda x: (x["top"], x["left"]))
    return lines
