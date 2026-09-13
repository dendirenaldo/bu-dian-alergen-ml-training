"""Knowledge Base alergen — port Notebook V5 Cell 29.

8 kategori kanonis + fuzzy konservatif (>=92) + evidence non-kanonis
(sulfit) + warning patterns. Murni diagnostik: TIDAK pernah menggantikan
Gold Label. Tanpa dependensi TF sehingga bisa di-test ringan.

``rapidfuzz`` opsional: bila tidak terinstal, fallback ke ``difflib``
dengan ambang yang sama (92). Tambahkan ``rapidfuzz`` ke requirements
untuk parity penuh (sudah ditambahkan).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:  # parity penuh bila tersedia
    from rapidfuzz import fuzz as _rf_fuzz

    def _ratio(a: str, b: str) -> float:
        return float(_rf_fuzz.ratio(a, b))

    _FUZZY_BACKEND = "rapidfuzz"
except ImportError:  # fallback stdlib agar test tetap jalan tanpa dep baru

    import difflib

    def _ratio(a: str, b: str) -> float:
        return float(difflib.SequenceMatcher(None, a, b).ratio() * 100.0)

    _FUZZY_BACKEND = "difflib"

FUZZY_THRESHOLD = 92.0

ALLERGEN_KB: dict[str, dict[str, list[str]]] = {
    "Gluten": {
        "canonical": [
            "wheat", "wheat flour", "gandum", "tepung terigu",
            "terigu", "barley", "rye", "malt", "gluten",
            "tepung gandum",
        ],
        "synonyms": [
            "wheat flour", "tepung gandum", "tepung terigu",
            "gandum", "gluten",
        ],
    },
    "Dairy": {
        "canonical": [
            "milk", "susu", "milk powder", "susu bubuk", "whey",
            "casein", "caseinate", "cheese", "keju", "cream",
            "butter", "yogurt",
        ],
        "synonyms": [
            "susu", "susu bubuk", "whey", "kasein", "keju",
            "mentega",
        ],
    },
    "Egg": {
        "canonical": [
            "egg", "telur", "egg powder", "telur bubuk",
            "albumen", "ovalbumin",
        ],
        "synonyms": ["telur", "telur bubuk", "albumen"],
    },
    "Soy": {
        "canonical": [
            "soy", "soybean", "kedelai", "soya",
            "soy lecithin", "lesitin kedelai",
        ],
        "synonyms": [
            "kedelai", "soya", "lesitin kedelai",
            "soy lecithin",
        ],
    },
    "Fish": {
        "canonical": [
            "fish", "ikan", "tuna", "salmon",
            "sardine", "sarden", "anchovy", "teri",
        ],
        "synonyms": ["ikan", "tuna", "salmon", "sarden", "teri"],
    },
    "Crustacean": {
        "canonical": [
            "shrimp", "udang", "prawn", "crab",
            "kepiting", "lobster", "crustacea", "krustasea",
        ],
        "synonyms": ["udang", "kepiting", "lobster", "krustasea"],
    },
    "Sesame": {
        "canonical": [
            "sesame", "wijen", "sesame seed", "biji wijen",
        ],
        "synonyms": ["wijen", "biji wijen"],
    },
    "Tree Nut": {
        "canonical": [
            "almond", "kacang almond", "cashew", "kacang mete",
            "kacang mede", "hazelnut", "walnut", "kenari",
            "pistachio",
        ],
        "synonyms": [
            "almond", "kacang almond", "kacang mete",
            "kacang mede", "hazelnut", "walnut", "kenari",
            "pistachio",
        ],
    },
}

NON_CANONICAL_ALLERGEN_TERMS = ["sulfit", "sulfite", "sulfites"]

WARNING_PATTERNS = [
    r"\bmengandung\s+alergen\b",
    r"\bcontains?\s+(milk|wheat|egg|soy|fish|sesame|nut|shellfish)\b",
    r"\bdapat\s+mengandung\b",
    r"\bmay\s+contain\b",
    r"\bdiproduksi.*(?:susu|kedelai|telur|gandum|ikan|wijen|kacang|udang|krustasea|crustacea)\b",
    r"\bmemproses.*(?:susu|kedelai|telur|gandum|ikan|wijen|kacang|udang|krustasea|crustacea)\b",
]


def clean_kb_text(text: Any) -> str:
    """Normalisasi teks untuk KB (port notebook)."""
    text = str(text or "").lower()
    text = text.replace("®", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _term_present(norm_text: str, term: str) -> bool:
    term_norm = clean_kb_text(term)
    if not term_norm:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)", norm_text))


def rule_based_kb_label(text: Any) -> dict:
    """Label otomatis KB untuk satu teks (diagnostik, bukan ground truth).

    Returns:
        dict dengan kunci: unsafe, binary_label, label, allergen_categories,
        matched_terms, evidence_methods, confidence, noncanonical_evidence,
        warning_evidence.
    """
    raw = str(text or "")
    norm = clean_kb_text(raw)

    hits: list[tuple[str, str]] = []
    methods: list[str] = []
    noncanonical_hits: list[str] = []

    # Exact / synonym per kategori (satu hit per kategori).
    for category, info in ALLERGEN_KB.items():
        terms = sorted(set(info["canonical"] + info["synonyms"]), key=len, reverse=True)
        for term in terms:
            if _term_present(norm, term):
                hits.append((category, term))
                methods.append("exact/synonym")
                break

    # Fuzzy konservatif per kategori yang belum kena.
    candidate_phrases = re.split(r"[,;.()\n]+", raw)
    candidate_phrases = [clean_kb_text(x) for x in candidate_phrases if clean_kb_text(x)]
    for category, info in ALLERGEN_KB.items():
        if any(h[0] == category for h in hits):
            continue
        best_score = 0.0
        best_term = None
        for phrase in candidate_phrases:
            if len(phrase) < 4:
                continue
            for term in set(info["canonical"] + info["synonyms"]):
                term_norm = clean_kb_text(term)
                if len(term_norm) < 5:
                    continue
                score = _ratio(phrase, term_norm)
                if score > best_score:
                    best_score = score
                    best_term = term
        if best_term is not None and best_score >= FUZZY_THRESHOLD:
            hits.append((category, best_term))
            methods.append(f"fuzzy_{best_score:.0f}")

    for term in NON_CANONICAL_ALLERGEN_TERMS:
        if _term_present(norm, term):
            noncanonical_hits.append(term)

    warning_hit = any(re.search(p, norm, re.I) for p in WARNING_PATTERNS)
    unsafe = bool(hits or noncanonical_hits or warning_hit)

    if hits:
        confidence = 1.00
    elif noncanonical_hits:
        confidence = 0.95
    elif warning_hit:
        confidence = 0.85
    else:
        confidence = 0.00

    return {
        "unsafe": unsafe,
        "binary_label": int(unsafe),
        "label": "unsafe" if unsafe else "safe",
        "allergen_categories": sorted({h[0] for h in hits}),
        "matched_terms": [h[1] for h in hits],
        "evidence_methods": methods,
        "confidence": confidence,
        "noncanonical_evidence": noncanonical_hits,
        "warning_evidence": bool(warning_hit),
        "fuzzy_backend": _FUZZY_BACKEND,
    }


def kb_label_frame(texts: list[str]) -> list[dict]:
    """Batch helper: label KB untuk list teks."""
    return [rule_based_kb_label(t) for t in texts]
