"""Streamlined CRISPR screen analysis toolkit with Scanpy-style entry points."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# OpenMP conflict guard — must precede any Numba import
# ---------------------------------------------------------------------------
# anndata 0.11.x eagerly imports torch when it is installed, loading PyTorch's
# libiomp5 (Intel OMP) into the same process as Numba's libomp (LLVM OMP).
# The dual OMP runtimes crash Numba prange on macOS (Segfault 11).  Using
# Numba's "workqueue" backend avoids OMP entirely while preserving parallelism.
# anndata 0.12+ fixes this but requires Python >=3.11 (crispyx supports 3.10).
# Override with NUMBA_THREADING_LAYER=omp if you manage the OMP conflict yourself.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crispyx")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

from .batch import BatchReducer, BatchStatistic, batch_process
from .data import (
    AnnData,
    OverlapResult,
    compute_overlap,
    convert_to_csc,
    convert_to_csr,
    detect_gene_symbol_column,
    detect_perturbation_column,
    ensure_gene_symbol_column,
    infer_columns,
    load_obs,
    load_var,
    normalize_total_log1p,
    normalise_perturbation_labels,
    read_h5ad_ondisk,
    read_backed,
    resolve_data_path,
    standardise_gene_names,
    write_obs,
    write_var,
)
from .de import (
    RankGenesGroupsResult,
    nb_glm_test,
    shrink_lfc,
    t_test,
    wilcoxon_test,
)
from ._preflight import estimate_disk_usage
from .profiling import (
    Profiler,
    MemoryProfiler,
    TimingProfiler,
    plot_benchmark_comparison,
)
from .plotting import (
    materialize_rank_genes_groups,
    plot_ma,
    plot_overlap_heatmap,
    plot_pca,
    plot_pca_loadings,
    plot_pca_variance_ratio,
    plot_qc_perturbation_counts,
    plot_qc_summary,
    plot_top_genes_bar,
    plot_umap,
    plot_volcano,
    rank_genes_groups_df,
)
from .pseudobulk import (
    PseudobulkMethod,
    aggregate_pseudobulk,
    compute_average_log_expression,
    compute_normalized_effects,
    compute_pseudobulk_effects,
    compute_pseudobulk_expression,
)
from .qc import (
    filter_cells_by_gene_count,
    filter_genes_by_cell_count,
    filter_perturbations_by_cell_count,
    quality_control_summary,
)

# ---------------------------------------------------------------------------
# Scanpy-style namespace singletons: cx.pp, cx.pb, cx.tl, cx.pl
# ---------------------------------------------------------------------------

from ._namespaces import (
    _PlottingNamespace,
    _PreprocessingNamespace,
    _PseudobulkNamespace,
    _ToolsNamespace,
)

pp = _PreprocessingNamespace()
pb = _PseudobulkNamespace()
tl = _ToolsNamespace()
pl = _PlottingNamespace()

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "__version__",
    # Namespace singletons
    "pp",
    "pb",
    "tl",
    "pl",
    # Quality control
    "filter_cells_by_gene_count",
    "filter_genes_by_cell_count",
    "filter_perturbations_by_cell_count",
    "quality_control_summary",
    # Pseudo-bulk
    "PseudobulkMethod",
    "aggregate_pseudobulk",
    "compute_average_log_expression",
    "compute_normalized_effects",
    "compute_pseudobulk_effects",
    "compute_pseudobulk_expression",
    # Generic batch statistics
    "BatchReducer",
    "BatchStatistic",
    "batch_process",
    # Differential expression
    "RankGenesGroupsResult",
    "t_test",
    "wilcoxon_test",
    "nb_glm_test",
    "shrink_lfc",
    # Data utilities
    "AnnData",
    "ensure_gene_symbol_column",
    "read_h5ad_ondisk",
    "read_backed",
    "resolve_data_path",
    "normalize_total_log1p",
    "convert_to_csc",
    "convert_to_csr",
    "load_obs",
    "load_var",
    "write_obs",
    "write_var",
    "standardise_gene_names",
    "normalise_perturbation_labels",
    "detect_perturbation_column",
    "detect_gene_symbol_column",
    "infer_columns",
    "OverlapResult",
    "compute_overlap",
    # Disk-usage estimation
    "estimate_disk_usage",
    # Profiling
    "Profiler",
    "MemoryProfiler",
    "TimingProfiler",
    "plot_benchmark_comparison",
    # Plotting
    "materialize_rank_genes_groups",
    "rank_genes_groups_df",
    "plot_pca",
    "plot_pca_variance_ratio",
    "plot_pca_loadings",
    "plot_umap",
    "plot_volcano",
    "plot_ma",
    "plot_top_genes_bar",
    "plot_qc_perturbation_counts",
    "plot_qc_summary",
    "plot_overlap_heatmap",
]
