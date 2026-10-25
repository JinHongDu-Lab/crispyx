"""Constants shared across benchmarking tools.

This module centralizes all constants used by the benchmarking infrastructure
to avoid duplication and ensure consistency.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ============================================================================
# Cache Version
# ============================================================================
# Increment this version when the cache format changes to force re-computation
CACHE_VERSION = "1.0.0"


# ============================================================================
# DE Metric Keys
# ============================================================================

DE_METRIC_KEYS: Tuple[str, ...] = (
    "effect_max_abs_diff",
    "effect_pearson_corr_mean",
    "effect_pearson_corr_std",
    "effect_spearman_corr_mean",
    "effect_spearman_corr_std",
    "effect_top_k_overlap",
    "effect_top_100_overlap_mean",
    "effect_top_100_overlap_std",
    "effect_top_500_overlap_mean",
    "effect_top_500_overlap_std",
    "statistic_max_abs_diff",
    "statistic_pearson_corr_mean",
    "statistic_pearson_corr_std",
    "statistic_spearman_corr_mean",
    "statistic_spearman_corr_std",
    "statistic_top_k_overlap",
    "pvalue_max_abs_diff",
    "pvalue_log_pearson_corr_mean",
    "pvalue_log_pearson_corr_std",
    "pvalue_log_spearman_corr_mean",
    "pvalue_log_spearman_corr_std",
    "pvalue_top_k_overlap",
    "pvalue_top_100_overlap_mean",
    "pvalue_top_100_overlap_std",
    "pvalue_top_500_overlap_mean",
    "pvalue_top_500_overlap_std",
    "pvalue_method_a_auroc",
    "pvalue_method_b_auroc",
    "statistic_type_mismatch",
    "n_perturbations",
)

# Top-k values for overlap metrics
TOP_K_VALUES: Tuple[int, ...] = (50, 100, 500)


# ============================================================================
# Shrinkage Metadata
# ============================================================================

# Mapping of method names to their LFC shrinkage type
# With standalone lfcShrink methods, these are separate benchmark steps
SHRINKAGE_METADATA: Dict[str, str] = {
    # Standalone lfcShrink methods (run after base NB-GLM fitting)
    "crispyx_de_lfcshrink": "apeglm",           # standalone lfcShrink with method='full'
    "crispyx_de_lfcshrink_pydeseq2": "apeglm",  # PyDESeq2-parity variant
    "pertpy_de_lfcshrink": "apeglm",            # PyDESeq2 standalone lfcShrink
}

# Set of methods that produce LFC shrinkage outputs (derived from SHRINKAGE_METADATA)
LFCSHRINK_METHODS = set(SHRINKAGE_METADATA.keys())

# All NB-GLM methods for explicit heatmap collection (base methods)
NB_GLM_METHODS = [
    "crispyx_de_nb_glm",
    "crispyx_de_nb_glm_pydeseq2",    # Optional: PyDESeq2-parity variant
    "crispyx_de_lfcshrink",          # Standalone lfcShrink
    "crispyx_de_lfcshrink_pydeseq2", # Optional: PyDESeq2-parity variant
    "edger_de_glm",
    "pertpy_de_pydeseq2",
    "pertpy_de_lfcshrink",           # PyDESeq2 standalone lfcShrink
]

# All DE methods for heatmap collection (includes t-test, Wilcoxon, and NB-GLM)
ALL_DE_METHODS_FOR_HEATMAP = [
    # t-test
    "crispyx_de_t_test",
    "scanpy_de_t_test",
    # Wilcoxon
    "crispyx_de_wilcoxon",
    "scanpy_de_wilcoxon",
    # NB-GLM base
    "crispyx_de_nb_glm",
    "crispyx_de_nb_glm_pydeseq2",  # Optional: PyDESeq2-parity variant
    "edger_de_glm",
    "pertpy_de_pydeseq2",
    # NB-GLM lfcShrink (standalone)
    "crispyx_de_lfcshrink",
    "crispyx_de_lfcshrink_pydeseq2",  # Optional: PyDESeq2-parity variant
    "pertpy_de_lfcshrink",
]

# Methods that produce shrunk outputs (standalone lfcShrink methods)
# These are loaded directly by their result_path
METHODS_WITH_SHRUNK_OUTPUT = [
    "crispyx_de_lfcshrink",
    "crispyx_de_lfcshrink_pydeseq2",  # Optional: PyDESeq2-parity variant
    "pertpy_de_lfcshrink",
]


# ============================================================================
# Heatmap Method Order
# ============================================================================

# Order for methods in overlap heatmaps
# Ordered by: t-test, Wilcoxon, NB-GLM (base), NB-GLM (lfcShrink)
# CRISPYx variants grouped adjacently for comparison
HEATMAP_METHOD_ORDER = [
    # t-test
    "crispyx_de_t_test",
    "scanpy_de_t_test",
    # Wilcoxon
    "crispyx_de_wilcoxon",
    "scanpy_de_wilcoxon",
    # NB-GLM (no shrinkage) - edgeR first as reference
    "edger_de_glm",
    "pertpy_de_pydeseq2",
    "crispyx_de_nb_glm",
    "crispyx_de_nb_glm_pydeseq2",  # NB-GLM with PyDESeq2-parity settings
    # NB-GLM (standalone lfcShrink methods)
    "pertpy_de_lfcshrink",
    "crispyx_de_lfcshrink",
    "crispyx_de_lfcshrink_pydeseq2",  # lfcShrink with PyDESeq2-parity base
]


# ============================================================================
# Method Display Names
# ============================================================================

# Centralized method display names for consistent formatting across all sections
# Keys are internal method names, values are display names
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    # t-test
    "crispyx_de_t_test": "t-test",
    "scanpy_de_t_test": "t-test",
    # Wilcoxon
    "crispyx_de_wilcoxon": "Wilcoxon",
    "scanpy_de_wilcoxon": "Wilcoxon",
    # NB-GLM base methods
    "crispyx_de_nb_glm": "NB-GLM",
    "crispyx_de_nb_glm_pydeseq2": "NB-GLM (pydeseq2)",
    "edger_de_glm": "NB-GLM",
    "pertpy_de_pydeseq2": "NB-GLM",
    # NB-GLM lfcShrink methods
    "crispyx_de_lfcshrink": "lfcShrink",
    "crispyx_de_lfcshrink_pydeseq2": "lfcShrink (pydeseq2)",  # PyDESeq2-parity variant
    "pertpy_de_lfcshrink": "lfcShrink",
    # QC
    "crispyx_qc_filtered": "QC filter",
    "scanpy_qc_filtered": "QC filter",
    # Pseudobulk
    "crispyx_pb_avg_log": "pseudobulk (avg log)",
    "crispyx_pb_avg": "pseudobulk (avg)",
    "crispyx_pb_pseudobulk": "pseudobulk",
}

# Accuracy reruns of references on the accuracy subset (plan 2026-10-25 §3.2),
# and crispyx's reruns on a control-capped subset.
for _base in ("scanpy_de_t_test", "scanpy_de_wilcoxon", "pertpy_de_pydeseq2", "edger_de_glm",
              "crispyx_de_t_test", "crispyx_de_wilcoxon", "crispyx_de_nb_glm"):
    METHOD_DISPLAY_NAMES[f"{_base}_subset"] = f"{METHOD_DISPLAY_NAMES[_base]} (subset)"
    METHOD_DISPLAY_NAMES[f"{_base}_subset_ctrl"] = f"{METHOD_DISPLAY_NAMES[_base]} (subset, capped control)"
del _base

# Order for DE method display in tables: t-test, Wilcoxon, NB-GLM
DE_METHOD_DISPLAY_ORDER = [
    "t-test",
    "Wilcoxon",
    "NB-GLM",
]

# Order for category subsections in Performance and Accuracy
CATEGORY_DISPLAY_ORDER = [
    "Preprocessing / QC",
    "DE: t-test",
    "DE: Wilcoxon",
    "DE: NB GLM",
    "Other",
]


# ============================================================================
# Standard Columns
# ============================================================================

# Standard DE result columns
STANDARD_DE_COLUMNS = ["perturbation", "gene", "effect_size", "statistic", "pvalue"]

# Status order for sorting benchmark results
STATUS_ORDER = ["success", "recovered", "memory_limit", "timeout", "error", "unknown"]
