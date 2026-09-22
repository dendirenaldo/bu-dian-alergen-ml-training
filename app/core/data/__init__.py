"""Paket data V5."""

from app.core.data.frozen import (
    FROZEN_SHA_ENV,
    frozen_list_sha256,
    load_frozen_products,
    resolve_frozen_products,
)
from app.core.data.gold_merge import (
    attach_kb_columns,
    build_model_source_from_single,
    merge_ocr_with_gold,
    normalize_key,
    normalize_product_name,
)
from app.core.data.leakage_split import (
    audit_final_pool_no_leakage,
    audit_split_no_leakage,
    build_split_manifest,
    choose_components_exact_deterministic,
    normalize_ocr_text_v5,
    normalize_product_key,
    run_v5_split,
)

from app.core.data.synthetic import build_final_training_pool, generate_synthetic_dataset

__all__ = [
    "attach_kb_columns",
    "FROZEN_SHA_ENV",
    "frozen_list_sha256",
    "load_frozen_products",
    "resolve_frozen_products",
    "audit_final_pool_no_leakage",
    "audit_split_no_leakage",
    "build_final_training_pool",
    "build_model_source_from_single",
    "build_split_manifest",
    "choose_components_exact_deterministic",
    "generate_synthetic_dataset",
    "merge_ocr_with_gold",
    "normalize_key",
    "normalize_ocr_text_v5",
    "normalize_product_key",
    "normalize_product_name",
    "run_v5_split",
]
