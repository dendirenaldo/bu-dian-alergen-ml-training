"""Bold text detection for allergen identification."""

import logging

import cv2
import numpy as np
import pytesseract

from app.core.preprocessing.text import normalize_text, clean_phrase, split_composition_items, token_overlap_score

logger = logging.getLogger(__name__)


def extract_bold_phrases(path: str, min_conf: float = 45.0) -> list[str]:
    """Extract bold text phrases from an image using ink density analysis.

    Identifies bold words by analyzing adaptive thresholding ink ratios
    and grouping them into phrases by block/paragraph/line.

    Args:
        path: Image file path.
        min_conf: Minimum Tesseract confidence threshold.

    Returns:
        List of unique bold phrase strings.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    data = pytesseract.image_to_data(gray, lang="eng", output_type=pytesseract.Output.DICT)

    words: list[dict] = []
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

        l = int(data["left"][i])
        t = int(data["top"][i])
        w = max(1, int(data["width"][i]))
        h = max(1, int(data["height"][i]))
        roi = bin_inv[t : t + h, l : l + w]
        ink_ratio = float(np.mean(roi > 0)) if roi.size else 0.0

        words.append(
            {
                "word": word,
                "ink": ink_ratio,
                "h": h,
                "left": l,
                "block": int(data["block_num"][i]),
                "par": int(data["par_num"][i]),
                "line": int(data["line_num"][i]),
            }
        )

    if not words:
        return []

    ink_thr = float(np.percentile([w["ink"] for w in words], 75))
    h_thr = float(np.percentile([w["h"] for w in words], 60))
    bold_words = [w for w in words if w["ink"] >= ink_thr and w["h"] >= h_thr]

    grouped: dict[tuple, list[dict]] = {}
    for w in bold_words:
        grouped.setdefault((w["block"], w["par"], w["line"]), []).append(w)

    phrases: list[str] = []
    for _, ws in grouped.items():
        ws = sorted(ws, key=lambda x: x["left"])
        phrase = normalize_text(" ".join(x["word"] for x in ws))
        if len(clean_phrase(phrase)) >= 2:
            phrases.append(phrase)

    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        k = clean_phrase(p)
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out


def detect_bold_composition_items(
    path: str, composition_text: str
) -> tuple[str, list[str]]:
    """Detect which composition items appear in bold text.

    Cross-references composition items against bold phrases extracted
    from the image to identify potentially highlighted allergens.

    Args:
        path: Image file path.
        composition_text: Extracted composition text string.

    Returns:
        Tuple of (label, list_of_bold_items) where label is 'unsafe' or 'safe'.
    """
    bold_phrases = extract_bold_phrases(path)
    composition_items = split_composition_items(composition_text)

    bold_items: list[str] = []
    for item in composition_items:
        item_clean = clean_phrase(item)
        if not item_clean:
            continue
        for bp in bold_phrases:
            bp_clean = clean_phrase(bp)
            if not bp_clean:
                continue
            overlap = token_overlap_score(item, bp)
            contains = item_clean in bp_clean or bp_clean in item_clean
            if overlap >= 0.6 or contains:
                bold_items.append(item)
                break

    uniq: list[str] = []
    seen: set[str] = set()
    for x in bold_items:
        k = clean_phrase(x)
        if k and k not in seen:
            seen.add(k)
            uniq.append(x)

    label = "unsafe" if uniq else "safe"
    return label, uniq
