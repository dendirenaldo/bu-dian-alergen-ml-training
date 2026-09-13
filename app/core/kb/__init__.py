"""Knowledge Base alergen (re-export)."""

from app.core.kb.rules import (
    ALLERGEN_KB,
    FUZZY_THRESHOLD,
    NON_CANONICAL_ALLERGEN_TERMS,
    WARNING_PATTERNS,
    clean_kb_text,
    kb_label_frame,
    rule_based_kb_label,
)

__all__ = [
    "ALLERGEN_KB",
    "FUZZY_THRESHOLD",
    "NON_CANONICAL_ALLERGEN_TERMS",
    "WARNING_PATTERNS",
    "clean_kb_text",
    "kb_label_frame",
    "rule_based_kb_label",
]
