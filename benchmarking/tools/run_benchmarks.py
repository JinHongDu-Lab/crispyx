"""Utilities to benchmark streaming CRISPR screen analysis methods."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

# ============================================================================
# Early Thread Configuration
# ============================================================================
# Set thread limits BEFORE importing any numerical libraries (numpy, numba, etc.)
# This prevents Numba from initializing with os.cpu_count() threads and later
# failing when we try to change NUMBA_NUM_THREADS.
# Default to a reasonable value; will be overridden by config if available.
if 'NUMBA_NUM_THREADS' not in os.environ:
    # Use a conservative default that can be increased later
    # Config-based n_cores will override this via set_thread_env_vars
    _default_threads = str(min(os.cpu_count() or 8, 32))
    os.environ['NUMBA_NUM_THREADS'] = _default_threads
    os.environ.setdefault('OMP_NUM_THREADS', _default_threads)
    os.environ.setdefault('MKL_NUM_THREADS', _default_threads)

import numpy as np
import pandas as pd
import yaml
import anndata as ad
from tqdm import tqdm

# Ensure the local package is importable when the project has not been installed.
# Path: benchmarking/tools/run_benchmarks.py -> parents[2] = project root
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .env_config import (
    EnvironmentConfig,
    configure_r_environment,
    set_thread_env_vars,
)
from .generate_demo_dataset import write_demo_dataset
from .generate_results import evaluate_benchmarks
from .profiling import (
    MemoryTracker,
    get_peak_memory_bytes,
    get_peak_memory_mb,
    get_current_rss_bytes,
    sample_subprocess_memory,
)
from crispyx.data import (
    read_backed,
    resolve_control_label,
    calculate_adaptive_qc_thresholds,
    standardize_dataset,
    normalize_total_log1p,
)
from crispyx.de import t_test, wilcoxon_test, nb_glm_test, shrink_lfc
from .comparison import DE_METRIC_KEYS, compute_de_comparison_metrics
from crispyx.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)
from crispyx.qc import quality_control_summary

# Import constants from centralized module
from .constants import CACHE_VERSION, STANDARD_DE_COLUMNS, STATUS_ORDER

# Import cache utilities
from .cache import (
    get_expected_output_path,
    is_scalar_na,
    make_json_serializable,
    save_method_result,
    load_method_result,
    load_cached_results,
    validate_and_recover_cache_result,
    save_cache_config,
    load_cache_config,
    invalidate_cache,
    check_output_exists,
)


# ============================================================================
# Numba Warm-up for Fair Benchmarking
# ============================================================================

def _warmup_numba_jit():
    """Trigger Numba JIT compilation before timed benchmarking.
    
    This ensures fair comparison by excluding one-time compilation overhead
    from benchmark timings. The warm-up calls a minimal subset of Numba
    functions used by nb_glm_test with tiny test data.
    """
    import numpy as np
    from crispyx.glm import (
        _nb_map_grid_search_numba,
        _nb_ll_for_alpha,
    )
    
    # Create minimal test data (10 cells, 5 genes)
    Y_test = np.random.randint(0, 10, size=(10, 5)).astype(np.float64)
    mu_test = np.ones((10, 5), dtype=np.float64) * 2.0
    log_trend_test = np.zeros(5, dtype=np.float64)
    log_alpha_grid = np.linspace(-8, 2, 5)  # 5 grid points
    prior_var = 0.25
    
    # Trigger JIT compilation
    _nb_map_grid_search_numba(Y_test, mu_test, log_trend_test, log_alpha_grid, prior_var)


# ============================================================================
# Reference Method Runners
# ============================================================================

import importlib
from scipy.stats import norm, rankdata

from crispyx.data import (
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
)
from crispyx.de import _tie_correction


def _normalize_total(matrix: np.ndarray, target_sum: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
    import scipy.sparse as sp
    
    # Handle both sparse and dense matrices
    if sp.issparse(matrix):
        library_size = np.asarray(matrix.sum(axis=1)).flatten()
    else:
        library_size = matrix.sum(axis=1)
    
    scale = np.divide(
        target_sum,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float64),
        where=library_size > 0,
    )
    
    if sp.issparse(matrix):
        # For sparse matrices, use multiply to keep sparse format
        # multiply() returns COO, so convert back to CSR for compatibility
        normalised = matrix.multiply(scale[:, None]).tocsr()
    else:
        normalised = matrix * scale[:, None]
    
    return normalised, library_size


def _log1p(matrix: np.ndarray) -> np.ndarray:
    import scipy.sparse as sp
    
    if sp.issparse(matrix):
        # For sparse matrices, apply log1p and keep sparse
        return matrix.log1p()
    else:
        return np.log1p(matrix)


def _filter_cells(matrix: np.ndarray, min_genes: int) -> np.ndarray:
    import scipy.sparse as sp
    
    if sp.issparse(matrix):
        expressed = np.asarray((matrix > 0).sum(axis=1)).flatten()
    else:
        expressed = (matrix > 0).sum(axis=1)
    return expressed >= min_genes


def _filter_genes(matrix: np.ndarray, min_cells: int) -> np.ndarray:
    import scipy.sparse as sp
    
    if sp.issparse(matrix):
        expressed = np.asarray((matrix > 0).sum(axis=0)).flatten()
    else:
        expressed = (matrix > 0).sum(axis=0)
    return expressed >= min_cells


def _subprocess_worker(pipe, func, args, kwargs):  # type: ignore[override]
    try:
        result = func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - exercised in error paths
        exc_type = exc.__class__
        pipe.send(("error", exc_type.__module__, exc_type.__name__, exc.args, traceback.format_exc()))
    else:
        pipe.send(("ok", result))
    finally:
        pipe.close()


def _run_in_subprocess(func, *args, **kwargs):
    """Execute ``func`` in a fresh process and return its result."""

    available = mp.get_all_start_methods()
    method = 'fork' if 'fork' in available else 'spawn'
    ctx = mp.get_context(method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    process = ctx.Process(target=_subprocess_worker, args=(child_conn, func, args, kwargs))
    process.start()
    message = parent_conn.recv()
    process.join()

    status = message[0]
    if status == 'ok':
        return message[1]

    _, module, name, exc_args, formatted = message
    try:
        exc_module = importlib.import_module(module)
        exc_type = getattr(exc_module, name)
    except Exception:  # pragma: no cover - defensive
        raise RuntimeError(formatted)
    raise exc_type(*exc_args)


def _load_into_memory(path: Path, use_cache: bool = True):
    """Load dataset into memory, keeping sparse format if possible.
    
    Parameters
    ----------
    path : Path
        Path to the h5ad file
    use_cache : bool
        If True, cache the loaded data to avoid reloading for subsequent calls.
        Cache is keyed by absolute path.
    
    Returns
    -------
    ad.AnnData
        Dataset with matrix in memory (sparse if original was sparse)
    """
    import anndata as ad
    import scipy.sparse as sp
    global _SCANPY_REFERENCE_CACHE
    
    cache_key = str(path.resolve())
    
    if use_cache and cache_key in _SCANPY_REFERENCE_CACHE:
        return _SCANPY_REFERENCE_CACHE[cache_key]
    
    adata = ad.read_h5ad(str(path))
    
    # Keep sparse matrices sparse to save memory
    # Only convert to dense if the matrix is already dense
    if sp.issparse(adata.X):
        # Ensure CSR format for Scanpy compatibility
        if not sp.isspmatrix_csr(adata.X):
            adata.X = adata.X.tocsr()
        # Ensure proper dtype
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)
    else:
        # Already dense, just ensure proper type
        adata.X = np.asarray(adata.X, dtype=np.float64)
    
    if use_cache:
        _SCANPY_REFERENCE_CACHE[cache_key] = adata
    
    return adata


def _normalise_metric_array(values: object, length: int) -> np.ndarray:
    if length == 0:
        return np.array([], dtype=float)
    if values is None:
        return np.full(length, np.nan, dtype=float)
    array = np.asarray(values)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=float)
    array = array.reshape(-1)
    if array.size >= length:
        return array[:length].astype(float)
    padded = np.full(length, np.nan, dtype=float)
    padded[: array.size] = array.astype(float)
    return padded


def _stream_results_to_frame(results: Mapping[str, object]) -> pd.DataFrame:
    columns = ["perturbation", "gene", "effect_size", "statistic", "pvalue"]
    frames: list[pd.DataFrame] = []
    for label, result in results.items():
        genes = getattr(result, "genes", None)
        if genes is None:
            continue
        gene_index = pd.Index(genes).astype(str)
        n_genes = len(gene_index)
        if n_genes == 0:
            continue
        data = {
            "perturbation": [str(label)] * n_genes,
            "gene": gene_index,
            "effect_size": _normalise_metric_array(getattr(result, "effect_size", None), n_genes),
            "statistic": _normalise_metric_array(getattr(result, "statistic", None), n_genes),
            "pvalue": _normalise_metric_array(getattr(result, "pvalue", None), n_genes),
        }
        frames.append(pd.DataFrame(data))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns)


def run_scanpy_qc(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run Scanpy QC pipeline and save filtered dataset."""
    import gc
    import time
    
    # Capture function start time (before imports)
    t_func_start = time.perf_counter()
    
    import scanpy as sc
    import scipy.sparse as sp
    
    # Phase 1: Load data
    t_load_start = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load_end = time.perf_counter()
    gc.collect()
    
    # Phase 2: Process
    t_process_start = time.perf_counter()
    
    # Ensure CSR
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
        
    # Filter cells
    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
    # Filter perturbations
    if min_cells_per_perturbation > 0:
        labels = adata.obs[perturbation_column].astype(str)
        counts = labels.value_counts()
        keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
        adata = adata[keep].copy()
        
    # Filter genes
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    
    t_process_end = time.perf_counter()
    gc.collect()
        
    # Phase 3: Save result
    t_save_start = time.perf_counter()
    output_path = output_dir / "scanpy_qc_filtered.h5ad"
    adata.write_h5ad(output_path)
    t_save_end = time.perf_counter()
    
    return {
        "result_path": str(output_path),
        "cells_kept": adata.n_obs,
        "genes_kept": adata.n_vars,
        "import_seconds": t_load_start - t_func_start,
        "load_seconds": t_load_end - t_load_start,
        "process_seconds": t_process_end - t_process_start,
        "save_seconds": t_save_end - t_save_start,
    }

def run_preprocess(
    input_path: Path,
    output_path: Path,
    *,
    chunk_size: int | None = None,
) -> Dict[str, Any]:
    """Normalize QC-filtered data (total-count normalization + log1p).

    Runs as a timed benchmark step so its wall-clock time is captured.
    Input must be the crispyx QC-filtered h5ad produced by crispyx_qc_filtered.

    The output is regenerated whenever the input file is newer than the output
    file (stale-cache detection).  This prevents a prior preprocessing result
    from silently masking a freshly-generated QC-filtered file.
    """
    import time

    t_start = time.perf_counter()
    needs_run = not output_path.exists()
    if not needs_run and input_path.exists():
        # Re-run when the QC-filtered input has been updated more recently than
        # the previously cached preprocessed output.
        if input_path.stat().st_mtime > output_path.stat().st_mtime:
            needs_run = True
    if needs_run:
        normalize_total_log1p(
            input_path,
            output_path,
            chunk_size=chunk_size,
            verbose=True,
        )
    return {
        "result_path": str(output_path),
        "process_seconds": time.perf_counter() - t_start,
    }


def run_preprocess_csc(
    input_path: Path,
    output_path: Path,
    *,
    chunk_size: int | None = None,
) -> Dict[str, Any]:
    """Convert the preprocessed CSR file to CSC for efficient Wilcoxon column access.

    If ``input_path`` is already stored as CSC the conversion is skipped and
    the original path is returned unchanged.  Stale-cache detection mirrors
    ``run_preprocess``: if ``input_path`` is newer than ``output_path`` the
    conversion is re-run.
    """
    import time

    from crispyx.data import convert_to_csc, get_matrix_storage_format

    t_start = time.perf_counter()

    # If source is already CSC, nothing to do.
    if get_matrix_storage_format(input_path) == "csc":
        return {
            "result_path": str(input_path),
            "process_seconds": time.perf_counter() - t_start,
            "was_already_csc": True,
        }

    needs_run = not output_path.exists()
    if not needs_run and input_path.exists():
        if input_path.stat().st_mtime > output_path.stat().st_mtime:
            needs_run = True
    if needs_run:
        convert_to_csc(
            input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            verbose=True,
        )
    return {
        "result_path": str(output_path),
        "process_seconds": time.perf_counter() - t_start,
        "was_already_csc": False,
    }


def run_preprocess_csr(
    input_path: Path,
    output_path: Path,
    *,
    chunk_size: int | None = None,
) -> Dict[str, Any]:
    """Convert standardized dataset to CSR for efficient NB-GLM row access.

    If ``input_path`` is already CSR the conversion is skipped.  A symlink
    from ``output_path`` → ``input_path`` is created so downstream steps can
    always reference ``output_path``.
    """
    import time

    from crispyx.data import convert_to_csr, get_matrix_storage_format

    t_start = time.perf_counter()

    if get_matrix_storage_format(input_path) == "csr":
        # Ensure output_path exists even when we skip conversion.
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.symlink_to(input_path.resolve())
        return {
            "result_path": str(output_path),
            "process_seconds": time.perf_counter() - t_start,
            "was_already_csr": True,
        }

    needs_run = not output_path.exists()
    if not needs_run and input_path.exists():
        if input_path.stat().st_mtime > output_path.stat().st_mtime:
            needs_run = True
    if needs_run:
        convert_to_csr(
            input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            verbose=True,
        )
    return {
        "result_path": str(output_path),
        "process_seconds": time.perf_counter() - t_start,
        "was_already_csr": False,
    }


def run_wilcoxon_with_csc(
    input_path: Path,
    csc_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    gene_name_column: str | None = None,
    data_name: str = "de_wilcoxon",
    n_jobs: int | None = None,
    memory_limit_gb: float | None = None,
    csc_chunk_size: int | None = None,
) -> Dict[str, Any]:
    """Ensure CSC format then run Wilcoxon DE, combining both in one timed step.

    CSR column slicing is O(total_nnz) per gene chunk for large datasets;
    CSC reduces this to O(nnz_in_chunk) total.  On large screens (Feng-gwsf)
    this gives ~18x speedup.  On small datasets (<2 GB) the conversion overhead
    dominates (~few seconds) while Wilcoxon itself is already fast, so the net
    effect on total runtime is negligible.

    Both conversion time and DE time are included in the benchmark measurement
    because the conversion is an implementation detail of the Wilcoxon pipeline:
    one combined number is more informative than split accounting where the
    "wilcoxon" time would look artificially short.

    ``csc_chunk_size`` is the cell/row chunk size used for CSC conversion (from QC).
    Gene chunk size for Wilcoxon is computed independently by ``wilcoxon_test``
    via ``calculate_optimal_gene_chunk_size``, which applies a tighter cap
    (max 512) appropriate for column-oriented operations with large control groups.

    Returns a dict so that :func:`_summarise_wilcoxon_csc` can expose the
    timing breakdown (``csc_conversion_seconds``, ``wilcoxon_seconds``,
    ``was_already_csc``) as extra benchmark columns.
    """
    import time

    # Step 1: Ensure CSC format (cheap no-op when already CSC or cached).
    t_csc_start = time.perf_counter()
    csc_result = run_preprocess_csc(input_path, csc_path, chunk_size=csc_chunk_size)
    csc_seconds = time.perf_counter() - t_csc_start
    was_already_csc = csc_result.get("was_already_csc", False)
    data_path = Path(csc_result["result_path"])

    # Step 2: Run Wilcoxon on the CSC file.
    # chunk_size=None: wilcoxon_test auto-calculates gene chunk size via
    # calculate_optimal_gene_chunk_size (max_chunk=512), independent of the
    # QC cell-chunk size. This avoids the O(n_ctrl * chunk_size) dense
    # allocation exploding for datasets with large control groups like Feng-gwsnf.
    t_de_start = time.perf_counter()
    de_result = wilcoxon_test(
        data=data_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        output_dir=output_dir,
        data_name=data_name,
        n_jobs=n_jobs,
        memory_limit_gb=memory_limit_gb,
        chunk_size=None,  # auto-calculated per calculate_optimal_gene_chunk_size
    )
    de_seconds = time.perf_counter() - t_de_start

    return {
        "de_result": de_result,
        "csc_conversion_seconds": 0.0 if was_already_csc else csc_seconds,
        "wilcoxon_seconds": de_seconds,
        "was_already_csc": was_already_csc,
    }


def run_scanpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    method: str,
    output_dir: Path,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """Run Scanpy DE on filtered dataset."""
    import gc
    import time
    
    # Capture function start time (before imports)
    t_func_start = time.perf_counter()
    
    import scanpy as sc
    
    # Phase 1: Load data
    t_load_start = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load_end = time.perf_counter()
    gc.collect()
    
    # Phase 2: Process
    t_process_start = time.perf_counter()
    
    # Preprocessing
    if preprocess:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    # DE
    sc.tl.rank_genes_groups(
        adata,
        groupby=perturbation_column,
        method=method,
        reference=control_label,
        n_genes=adata.n_vars,
        tie_correct=True,  # Match crispyx's default tie correction
    )
    
    df = sc.get.rank_genes_groups_df(adata, None)
    
    # Rename 'group' to perturbation_column if present
    if "group" in df.columns:
        df = df.rename(columns={"group": perturbation_column})
    
    t_process_end = time.perf_counter()
    gc.collect()
        
    # Phase 3: Save result
    t_save_start = time.perf_counter()
    output_path = output_dir / f"scanpy_de_{method.replace('-', '_')}.csv"
    df.to_csv(output_path, index=False)
    t_save_end = time.perf_counter()
    
    return {
        "result_path": str(output_path),
        "groups": len(df[perturbation_column].unique()) if not df.empty else 0,
        "import_seconds": t_load_start - t_func_start,
        "load_seconds": t_load_end - t_load_start,
        "process_seconds": t_process_end - t_process_start,
        "save_seconds": t_save_end - t_save_start,
    }

def run_edger_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run edgeR DE on filtered dataset."""
    import gc
    import os
    import time
    
    # Capture function start time (before imports)
    t_func_start = time.perf_counter()
    
    from .env_config import get_global_env_config
    
    # Configure R
    env_config = get_global_env_config()
    r_home = env_config.r_home if env_config else None
    # Let auto-detection find R_HOME instead of using hardcoded path
    configure_r_environment(r_home)
    
    import scanpy as sc
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    
    # Phase 1: Load data
    t_load_start = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load_end = time.perf_counter()
    gc.collect()
    
    # Phase 2: Process
    t_process_start = time.perf_counter()
    
    groups = adata.obs[perturbation_column].astype(str).values
    unique_groups = [g for g in np.unique(groups) if g != control_label]
    
    if not unique_groups:
        raise ValueError("No perturbation groups found")
        
    # R setup
    n_threads = n_jobs if n_jobs is not None and n_jobs > 0 else 1
    ro.r('library(edgeR)')
    ro.r(f'''
    tryCatch({{
        library(RhpcBLASctl)
        blas_set_num_threads({n_threads})
        omp_set_num_threads({n_threads})
    }}, error=function(e) {{ }})
    ''')
    
    # Convert data
    counts = adata.X.T
    if hasattr(counts, 'toarray'):
        counts = counts.toarray()
        
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['counts'] = counts
        ro.globalenv['groups'] = ro.StrVector(groups)
        ro.globalenv['gene_names'] = ro.StrVector(adata.var_names)
        
        ro.r('''
        rownames(counts) <- gene_names
        y <- DGEList(counts=counts, group=groups)
        y <- calcNormFactors(y)
        design <- model.matrix(~0 + groups)
        colnames(design) <- gsub("groups", "", colnames(design))
        y <- estimateDisp(y, design)
        fit <- glmQLFit(y, design)
        ''')
        
        all_results = []
        for group in unique_groups:
            ro.globalenv['target_group'] = group
            ro.globalenv['control'] = control_label
            
            ro.r('''
            contrast_vec <- makeContrasts(
                contrasts = paste0(target_group, "-", control),
                levels = design
            )
            lrt <- glmQLFTest(fit, contrast=contrast_vec)
            ''')
            
            genes = np.array(ro.r('rownames(lrt$table)'))
            logFC = np.array(ro.r('lrt$table$logFC'))
            logCPM = np.array(ro.r('lrt$table$logCPM'))
            F_stat = np.array(ro.r('lrt$table$F'))
            PValue = np.array(ro.r('lrt$table$PValue'))
            FDR = np.array(ro.r('p.adjust(lrt$table$PValue, method="BH")'))
            
            all_results.append(pd.DataFrame({
                'gene': genes,
                'logFC': logFC,
                'logCPM': logCPM,
                'F': F_stat,
                'PValue': PValue,
                'FDR': FDR,
                'perturbation': group
            }))
            
    final_results = pd.concat(all_results, ignore_index=True)
    t_process_end = time.perf_counter()
    gc.collect()
    
    # Phase 3: Save result
    t_save_start = time.perf_counter()
    output_path = output_dir / "edger_de_glm.csv"
    final_results.to_csv(output_path, index=False)
    t_save_end = time.perf_counter()
    
    return {
        "result_path": str(output_path),
        "groups": len(unique_groups),
        "import_seconds": t_load_start - t_func_start,
        "load_seconds": t_load_end - t_load_start,
        "process_seconds": t_process_end - t_process_start,
        "save_seconds": t_save_end - t_save_start,
    }

def run_pertpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    backend: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run Pertpy DE on filtered dataset."""
    import gc
    import io
    import time
    
    # Capture function start time (before imports)
    t_func_start = time.perf_counter()
    
    from contextlib import redirect_stdout, redirect_stderr
    import pertpy as pt
    import scanpy as sc
    
    # Phase 1: Load data
    t_load_start = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load_end = time.perf_counter()
    gc.collect()
    
    # Phase 2: Process (with suppressed output)
    t_process_start = time.perf_counter()
    
    # Resolve runner
    module = getattr(pt, "tools", None)
    candidate_modules = [
        getattr(module, "differential_expression", None),
        module,
    ]
    try:
        candidate_modules.append(import_module("pertpy.tools._differential_gene_expression"))
    except Exception:
        pass
        
    runner = None
    for candidate in candidate_modules:
        if candidate:
            runner = _resolve_pertpy_runner(candidate, backend) or _resolve_pertpy_class_runner(candidate, backend)
            if runner: break
            
    if not runner:
        raise ValueError(f"Pertpy runner {backend} not found")
        
    # Run with suppressed stdout/stderr to hide PyDESeq2 verbose output
    kwargs = {"groupby": perturbation_column, "control": control_label}
    if n_jobs: kwargs["n_cpus"] = n_jobs
    
    # Suppress PyDESeq2 verbose output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            result = runner(adata, **kwargs)
        except TypeError:
            # Fallback for different API
            kwargs = {"group_key": perturbation_column, "control": control_label}
            if n_jobs: kwargs["n_cpus"] = n_jobs
            result = runner(adata, **kwargs)
    
    t_de_end = time.perf_counter()
    
    # Convert result to DataFrame (can be slow for some backends)
    t_convert_start = time.perf_counter()
    df = _convert_reference_result_to_dataframe(result)
    t_convert_end = time.perf_counter()
    
    # Rename 'contrast' to perturbation_column if present (PyDESeq2)
    if df is not None and "contrast" in df.columns and perturbation_column not in df.columns:
        df = df.rename(columns={"contrast": perturbation_column})
    
    t_process_end = time.perf_counter()
    gc.collect()
        
    # Phase 3: Save result
    t_save_start = time.perf_counter()
    output_path = output_dir / f"pertpy_de_{backend}.csv"
    if df is not None:
        df.to_csv(output_path, index=False)
    t_save_end = time.perf_counter()
    
    return {
        "result_path": str(output_path),
        "groups": len(df[perturbation_column].unique()) if df is not None and "perturbation" in df.columns else 0,
        "import_seconds": t_load_start - t_func_start,
        "load_seconds": t_load_end - t_load_start,
        "process_seconds": t_process_end - t_process_start,
        "de_seconds": t_de_end - t_process_start,
        "convert_seconds": t_convert_end - t_convert_start,
        "save_seconds": t_save_end - t_save_start,
    }


def run_pydeseq2_integrated(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run PyDESeq2 with base DE and apeGLM LFC shrinkage, saving both outputs.
    
    This function calls PyDESeq2 directly and applies apeGLM LFC shrinkage,
    saving both non-shrunk and shrunk results for comparison.
    
    Returns timing breakdown with separate base_seconds (DE fitting) and 
    shrinkage_seconds (lfcShrink) components for detailed performance analysis.
    Both result_path (base) and shrunk_result_path are returned.
    """
    import gc
    import io
    import time
    from contextlib import redirect_stdout, redirect_stderr
    
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    
    t_func_start = time.perf_counter()
    
    # Load data
    t_load_start = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load_end = time.perf_counter()
    gc.collect()
    
    # Get unique perturbations (excluding control)
    perturbations = [p for p in adata.obs[perturbation_column].unique() if p != control_label]
    
    t_process_start = time.perf_counter()
    all_base_results = []
    all_shrunk_results = []
    
    # Track timing for base DE and shrinkage separately
    total_base_seconds = 0.0
    total_shrinkage_seconds = 0.0
    
    for pert in perturbations:
        # Subset to control + current perturbation
        mask = adata.obs[perturbation_column].isin([control_label, pert])
        adata_subset = adata[mask].copy()
        
        # PyDESeq2 requires dense matrix with integer counts
        if sp.issparse(adata_subset.X):
            adata_subset.X = np.asarray(adata_subset.X.todense())
        # Ensure integer counts (int32 saves ~50% memory vs int64)
        adata_subset.X = np.round(adata_subset.X).astype(np.int32)
        
        # Reorder categories so perturbation is first (becomes reference)
        # This allows us to shrink the control coefficient and then negate
        try:
            adata_subset.obs[perturbation_column] = adata_subset.obs[perturbation_column].cat.reorder_categories([pert, control_label])
        except Exception:
            pass  # Skip if reordering fails
        
        # Run PyDESeq2
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                # Time base DE fitting
                t_base_start = time.perf_counter()
                dds = DeseqDataSet(
                    adata=adata_subset,
                    design_factors=perturbation_column,
                    refit_cooks=True,
                    n_cpus=n_jobs or 1,
                )
                dds.deseq2()
                
                # The coefficient is control vs perturbation (since pert is reference)
                # We need to use control as the treatment coefficient
                coeff_name = f"{perturbation_column}[T.{control_label}]"
                
                # Create contrast for control vs perturbation (will negate later)
                stat_res = DeseqStats(
                    dds,
                    contrast=[perturbation_column, control_label, pert],
                    alpha=0.05,
                    n_cpus=n_jobs or 1,
                )
                stat_res.summary()
                t_base_end = time.perf_counter()
                total_base_seconds += (t_base_end - t_base_start)
                
                # Save base (non-shrunk) results before applying shrinkage
                base_results_df = stat_res.results_df.copy()
                base_results_df["log2FoldChange"] = -base_results_df["log2FoldChange"]
                if "stat" in base_results_df.columns:
                    base_results_df["stat"] = -base_results_df["stat"]
                base_results_df["gene"] = base_results_df.index
                base_results_df[perturbation_column] = pert
                base_results_df = base_results_df.reset_index(drop=True)
                all_base_results.append(base_results_df)
                
                # Apply apeGLM LFC shrinkage (timed separately)
                t_shrink_start = time.perf_counter()
                shrinkage_applied = False
                try:
                    stat_res.lfc_shrink(coeff=coeff_name)
                    shrinkage_applied = True
                except Exception as e:
                    # Try to find available coefficients
                    try:
                        if hasattr(dds, 'varm') and 'LFC' in dds.varm:
                            available_coeffs = list(dds.varm['LFC'].columns)
                            # Find the non-intercept coefficient
                            matching = [c for c in available_coeffs if c != 'Intercept']
                            if matching:
                                stat_res.lfc_shrink(coeff=matching[0])
                                shrinkage_applied = True
                    except Exception:
                        pass  # Fall back to unshrunken if shrinkage fails
                t_shrink_end = time.perf_counter()
                total_shrinkage_seconds += (t_shrink_end - t_shrink_start)
                
                # Extract shrunk results
                shrunk_results_df = stat_res.results_df.copy()
                
                # Negate LFC to get perturbation vs control (we computed control vs pert)
                shrunk_results_df["log2FoldChange"] = -shrunk_results_df["log2FoldChange"]
                # Also negate the stat to match
                if "stat" in shrunk_results_df.columns:
                    shrunk_results_df["stat"] = -shrunk_results_df["stat"]
                
                shrunk_results_df["gene"] = shrunk_results_df.index
                shrunk_results_df[perturbation_column] = pert
                shrunk_results_df["shrinkage_applied"] = shrinkage_applied
                shrunk_results_df = shrunk_results_df.reset_index(drop=True)
                
                all_shrunk_results.append(shrunk_results_df)
                
            except Exception as e:
                # Skip this perturbation if it fails
                continue
    
    t_process_end = time.perf_counter()
    
    # Helper to format dataframe
    def format_df(results_list):
        if results_list:
            df = pd.concat(results_list, ignore_index=True)
            df = df.rename(columns={
                "log2FoldChange": "log_fc",
                "lfcSE": "log_fc_se", 
                "stat": "statistic",
                "pvalue": "pvalue",
                "padj": "pvalue_adj",
            })
            return df
        return pd.DataFrame()
    
    base_df = format_df(all_base_results)
    shrunk_df = format_df(all_shrunk_results)
    
    # Save both results
    t_save_start = time.perf_counter()
    base_output_path = output_dir / "pertpy_de_pydeseq2.csv"
    shrunk_output_path = output_dir / "pertpy_de_pydeseq2_shrunk.csv"
    base_df.to_csv(base_output_path, index=False)
    shrunk_df.to_csv(shrunk_output_path, index=False)
    t_save_end = time.perf_counter()
    
    gc.collect()
    
    return {
        "result_path": str(base_output_path),
        "shrunk_result_path": str(shrunk_output_path),
        "groups": len(perturbations),
        "import_seconds": t_load_start - t_func_start,
        "load_seconds": t_load_end - t_load_start,
        "process_seconds": t_process_end - t_process_start,
        "base_seconds": total_base_seconds,
        "shrinkage_seconds": total_shrinkage_seconds,
        "save_seconds": t_save_end - t_save_start,
    }


def run_nb_glm_base(
    path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    n_jobs: int | None = None,
    size_factor_method: str = "sparse",
    scale_size_factors: bool = False,
    gene_name_column: str | None = None,
    use_control_cache: bool = True,
    size_factor_scope: str = "global",
    se_method: str = "sandwich",
    dispersion_scope: str = "global",
    data_name: str = "de_nb_glm",
    profiling: bool = True,
    memory_limit_gb: float | None = None,
    max_dense_fraction: float = 0.3,
) -> Dict[str, Any]:
    """Run NB-GLM fitting only (no shrinkage).
    
    This function runs nb_glm_test without LFC shrinkage, saving the base results.
    Shrinkage is handled by a separate run_lfcshrink step with depends_on.
    
    Parameters
    ----------
    path
        Path to input h5ad file
    perturbation_column
        Column name for perturbation labels
    control_label
        Label for control cells
    output_dir
        Directory for output files
    n_jobs
        Number of parallel jobs
    size_factor_method
        Method for size factor calculation. Default "sparse" is optimal for
        single-cell data. The "deseq2" method (genes expressed in ALL cells)
        provides no benefit for sparse scRNA-seq since it falls back to "sparse"
        when too few genes qualify.
    scale_size_factors
        Whether to scale size factors
    gene_name_column
        Column name for gene names (passed to nb_glm_test)
    use_control_cache
        If True, precompute control cell statistics once and reuse across
        all perturbation comparisons.
    size_factor_scope
        Scope for size factor computation: "global" or "per_comparison".
    se_method
        Method for SE computation: "sandwich" (robust) or "fisher" (PyDESeq2 parity).
    dispersion_scope
        Scope for dispersion estimation: "global" or "per_comparison".
    data_name
        Base name for output files (e.g., "de_nb_glm" -> crispyx_de_nb_glm.h5ad).
    profiling
        If True, enable profiling in nb_glm_test.
    memory_limit_gb
        Optional memory limit in GB. Used for adaptive memory management:
        - Triggers streaming mode for global dispersion if matrix too large
        - Limits parallel worker count based on estimated memory per worker
    max_dense_fraction
        Maximum fraction of available memory to use for dense matrices.
        Default is 0.3 (30%). Reduces risk of OOM during global dispersion.
        
    Returns
    -------
    Dict with result_path, timing, and memory metrics
    """
    import time
    from typing import Literal, cast
    
    # Cast string parameters to their literal types
    size_factor_method_lit = cast(Literal["sparse", "deseq2"], size_factor_method)
    size_factor_scope_lit = cast(Literal["global", "per_comparison"], size_factor_scope)
    se_method_lit = cast(Literal["sandwich", "fisher"], se_method)
    dispersion_scope_lit = cast(Literal["global", "per_comparison"], dispersion_scope)
    
    # Run NB-GLM base fitting with profiling (no shrinkage)
    t_base_start = time.perf_counter()
    base_result = nb_glm_test(
        data=path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        output_dir=output_dir,
        data_name=data_name,
        n_jobs=n_jobs,
        lfc_shrinkage_type="none",  # No shrinkage - handled by separate step
        size_factor_method=size_factor_method_lit,
        scale_size_factors=scale_size_factors,
        gene_name_column=gene_name_column,
        use_control_cache=use_control_cache,
        size_factor_scope=size_factor_scope_lit,
        se_method=se_method_lit,
        dispersion_scope=dispersion_scope_lit,
        profiling=profiling,
        memory_limit_gb=memory_limit_gb,
        max_dense_fraction=max_dense_fraction,
    )
    t_base_end = time.perf_counter()
    base_seconds_wall = t_base_end - t_base_start
    
    # Get base result path
    base_result_path = getattr(base_result, "result_path", None)
    if base_result_path is None:
        # Construct expected path
        base_result_path = output_dir / f"crispyx_{data_name}.h5ad"
    
    # Read profiling data from base h5ad
    base_peak_memory_mb = None
    if profiling and base_result_path and Path(base_result_path).exists():
        try:
            base_adata = ad.read_h5ad(str(base_result_path))
            prof_data = base_adata.uns.get("profiling")
            if isinstance(prof_data, dict) and prof_data.get("profiling_enabled"):
                base_peak_memory_mb = prof_data.get("fit_peak_memory_mb")
                # Use internal timing if available (more accurate)
                if "fit_seconds" in prof_data:
                    base_seconds_wall = prof_data["fit_seconds"]
        except Exception:
            pass  # Fall back to wall-clock timing
    
    # Get group/gene counts from base result
    n_groups = len(list(base_result.keys())) if hasattr(base_result, "keys") else 0
    n_genes = 0
    if n_groups > 0:
        first_key = list(base_result.keys())[0]
        first_group = base_result[first_key]
        if hasattr(first_group, "genes"):
            n_genes = len(first_group.genes)
    
    return {
        "result_path": str(base_result_path) if base_result_path else None,
        "groups": n_groups,
        "genes": n_genes,
        "elapsed_seconds": base_seconds_wall,
        "peak_memory_mb": base_peak_memory_mb,
        "profiling_enabled": profiling,
    }


def run_lfcshrink(
    base_result_path: Path,
    *,
    output_dir: Path,
    data_name: str = "de_nb_glm_shrunk",
    prior_scale_mode: str = "global",
    profiling: bool = True,
    memory_limit_gb: float | None = None,
) -> Dict[str, Any]:
    """Run lfcShrink on existing NB-GLM results.
    
    This function applies apeGLM LFC shrinkage to existing NB-GLM results.
    Memory baseline is measured after loading the base result from disk.
    
    Parameters
    ----------
    base_result_path
        Path to the base NB-GLM h5ad file (from run_nb_glm_base)
    output_dir
        Directory for output files
    data_name
        Name for output file (e.g., "de_nb_glm_shrunk" -> crispyx_de_nb_glm_shrunk.h5ad)
    prior_scale_mode
        Prior scale estimation mode: "global" or "per_comparison"
    profiling
        If True, enable profiling in shrink_lfc
    memory_limit_gb
        Optional memory budget in GB passed to shrink_lfc.
        
    Returns
    -------
    Dict with result_path, timing, and memory metrics
    """
    import time
    
    t_start = time.perf_counter()
    shrunk_result = shrink_lfc(
        data=base_result_path,
        output_dir=output_dir,
        data_name=data_name,
        profiling=profiling,
        prior_scale_mode=prior_scale_mode,
        memory_limit_gb=memory_limit_gb,
        method="stats",  # Use stats-based shrinkage (Gaussian approx) - more accurate than "full"
        # Note: "full" re-fits with NB likelihood using stored dispersion, which can
        # diverge from PyDESeq2 due to different dispersion estimation methods.
        # "stats" uses Gaussian approximation which is robust and matches pertpy well.
    )
    t_end = time.perf_counter()
    elapsed_seconds = t_end - t_start
    
    # Get shrunk output path from result
    shrunk_result_path = getattr(shrunk_result, "result_path", None)
    
    # Read profiling data from shrunk h5ad
    shrinkage_peak_memory_mb = None
    if profiling and shrunk_result_path and Path(shrunk_result_path).exists():
        try:
            shrunk_adata = ad.read_h5ad(str(shrunk_result_path))
            prof_data = shrunk_adata.uns.get("profiling")
            if isinstance(prof_data, dict) and prof_data.get("profiling_enabled"):
                shrinkage_peak_memory_mb = prof_data.get("shrinkage_peak_memory_mb")
                # Use internal timing if available
                if "shrinkage_seconds" in prof_data:
                    elapsed_seconds = prof_data["shrinkage_seconds"]
        except Exception:
            pass  # Fall back to wall-clock timing
    
    # Get group/gene counts from shrunk result
    n_groups = 0
    n_genes = 0
    if shrunk_result_path and Path(shrunk_result_path).exists():
        try:
            shrunk_adata = ad.read_h5ad(str(shrunk_result_path))
            n_groups = shrunk_adata.n_obs
            n_genes = shrunk_adata.n_vars
        except Exception:
            pass
    
    return {
        "result_path": str(shrunk_result_path) if shrunk_result_path else None,
        "groups": n_groups,
        "genes": n_genes,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mb": shrinkage_peak_memory_mb,
        "shrinkage_type": "apeglm",
        "prior_scale_mode": prior_scale_mode,
        "profiling_enabled": profiling,
    }


def run_pydeseq2_base(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run PyDESeq2 base DE fitting only (no shrinkage).
    
    This function runs PyDESeq2 without LFC shrinkage, saving base results.
    Shrinkage is handled by a separate run_pydeseq2_lfcshrink step.
    
    Returns timing and memory metrics for the base fitting only.
    """
    import gc
    import io
    import time
    from contextlib import redirect_stdout, redirect_stderr
    
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    
    t_func_start = time.perf_counter()
    
    # Load data
    t_load_start = time.perf_counter()
    adata = sc.read_h5ad(str(dataset_path))
    t_load_end = time.perf_counter()
    gc.collect()
    
    # Get unique perturbations (excluding control)
    perturbations = [p for p in adata.obs[perturbation_column].unique() if p != control_label]
    
    t_process_start = time.perf_counter()
    all_base_results = []
    
    for pert in perturbations:
        # Subset to control + current perturbation
        mask = adata.obs[perturbation_column].isin([control_label, pert])
        adata_subset = adata[mask].copy()
        
        # PyDESeq2 requires dense matrix with integer counts
        if sp.issparse(adata_subset.X):
            adata_subset.X = np.asarray(adata_subset.X.todense())
        # Ensure integer counts (int32 saves ~50% memory vs int64)
        adata_subset.X = np.round(adata_subset.X).astype(np.int32)
        
        # Reorder categories so perturbation is first (becomes reference)
        try:
            adata_subset.obs[perturbation_column] = adata_subset.obs[perturbation_column].cat.reorder_categories([pert, control_label])
        except Exception:
            pass
        
        # Run PyDESeq2
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                dds = DeseqDataSet(
                    adata=adata_subset,
                    design_factors=perturbation_column,
                    refit_cooks=True,
                    n_cpus=n_jobs or 1,
                )
                dds.deseq2()
                
                stat_res = DeseqStats(
                    dds,
                    contrast=[perturbation_column, control_label, pert],
                    alpha=0.05,
                    n_cpus=n_jobs or 1,
                )
                stat_res.summary()
                
                # Save base results
                base_results_df = stat_res.results_df.copy()
                base_results_df["log2FoldChange"] = -base_results_df["log2FoldChange"]
                if "stat" in base_results_df.columns:
                    base_results_df["stat"] = -base_results_df["stat"]
                base_results_df["gene"] = base_results_df.index
                base_results_df[perturbation_column] = pert
                # Store dds and stat_res references for lfcShrink step
                base_results_df["_coeff_name"] = f"{perturbation_column}[T.{control_label}]"
                base_results_df = base_results_df.reset_index(drop=True)
                all_base_results.append(base_results_df)
                
            except Exception:
                continue
    
    t_process_end = time.perf_counter()
    
    # Format and save base results
    if all_base_results:
        base_df = pd.concat(all_base_results, ignore_index=True)
        base_df = base_df.rename(columns={
            "log2FoldChange": "log_fc",
            "lfcSE": "log_fc_se", 
            "stat": "statistic",
            "pvalue": "pvalue",
            "padj": "pvalue_adj",
        })
        # Drop internal column before saving
        if "_coeff_name" in base_df.columns:
            base_df = base_df.drop(columns=["_coeff_name"])
    else:
        base_df = pd.DataFrame()
    
    t_save_start = time.perf_counter()
    base_output_path = output_dir / "pertpy_de_pydeseq2.csv"
    base_df.to_csv(base_output_path, index=False)
    t_save_end = time.perf_counter()
    
    gc.collect()
    
    return {
        "result_path": str(base_output_path),
        "groups": len(perturbations),
        "import_seconds": t_load_start - t_func_start,
        "load_seconds": t_load_end - t_load_start,
        "process_seconds": t_process_end - t_process_start,
        "save_seconds": t_save_end - t_save_start,
    }


def run_pydeseq2_lfcshrink(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run PyDESeq2 lfcShrink only.
    
    This function runs PyDESeq2 DE fitting followed by lfcShrink,
    but only reports timing for the shrinkage step. The base fitting
    is necessary to get the DeseqStats object for shrinkage.
    
    Memory baseline is measured after the base fitting completes,
    so only shrinkage memory is counted.
    """
    import gc
    import io
    import time
    from contextlib import redirect_stdout, redirect_stderr
    
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    
    t_func_start = time.perf_counter()
    
    # Load data
    adata = sc.read_h5ad(str(dataset_path))
    gc.collect()
    
    # Get unique perturbations (excluding control)
    perturbations = [p for p in adata.obs[perturbation_column].unique() if p != control_label]
    
    all_shrunk_results = []
    total_shrinkage_seconds = 0.0
    
    for pert in perturbations:
        # Subset to control + current perturbation
        mask = adata.obs[perturbation_column].isin([control_label, pert])
        adata_subset = adata[mask].copy()
        
        # PyDESeq2 requires dense matrix with integer counts
        if sp.issparse(adata_subset.X):
            adata_subset.X = np.asarray(adata_subset.X.todense())
        # Ensure integer counts (int32 saves ~50% memory vs int64)
        adata_subset.X = np.round(adata_subset.X).astype(np.int32)
        
        try:
            adata_subset.obs[perturbation_column] = adata_subset.obs[perturbation_column].cat.reorder_categories([pert, control_label])
        except Exception:
            pass
        
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                # Run base fitting (not timed for this method)
                dds = DeseqDataSet(
                    adata=adata_subset,
                    design_factors=perturbation_column,
                    refit_cooks=True,
                    n_cpus=n_jobs or 1,
                )
                dds.deseq2()
                
                coeff_name = f"{perturbation_column}[T.{control_label}]"
                
                stat_res = DeseqStats(
                    dds,
                    contrast=[perturbation_column, control_label, pert],
                    alpha=0.05,
                    n_cpus=n_jobs or 1,
                )
                stat_res.summary()
                
                # Time only the shrinkage step
                t_shrink_start = time.perf_counter()
                shrinkage_applied = False
                try:
                    stat_res.lfc_shrink(coeff=coeff_name)
                    shrinkage_applied = True
                except Exception:
                    try:
                        if hasattr(dds, 'varm') and 'LFC' in dds.varm:
                            available_coeffs = list(dds.varm['LFC'].columns)
                            matching = [c for c in available_coeffs if c != 'Intercept']
                            if matching:
                                stat_res.lfc_shrink(coeff=matching[0])
                                shrinkage_applied = True
                    except Exception:
                        pass
                t_shrink_end = time.perf_counter()
                total_shrinkage_seconds += (t_shrink_end - t_shrink_start)
                
                # Extract shrunk results
                shrunk_results_df = stat_res.results_df.copy()
                shrunk_results_df["log2FoldChange"] = -shrunk_results_df["log2FoldChange"]
                if "stat" in shrunk_results_df.columns:
                    shrunk_results_df["stat"] = -shrunk_results_df["stat"]
                shrunk_results_df["gene"] = shrunk_results_df.index
                shrunk_results_df[perturbation_column] = pert
                shrunk_results_df["shrinkage_applied"] = shrinkage_applied
                shrunk_results_df = shrunk_results_df.reset_index(drop=True)
                all_shrunk_results.append(shrunk_results_df)
                
            except Exception:
                continue
    
    # Format and save shrunk results
    if all_shrunk_results:
        shrunk_df = pd.concat(all_shrunk_results, ignore_index=True)
        shrunk_df = shrunk_df.rename(columns={
            "log2FoldChange": "log_fc",
            "lfcSE": "log_fc_se", 
            "stat": "statistic",
            "pvalue": "pvalue",
            "padj": "pvalue_adj",
        })
    else:
        shrunk_df = pd.DataFrame()
    
    shrunk_output_path = output_dir / "pertpy_de_pydeseq2_shrunk.csv"
    shrunk_df.to_csv(shrunk_output_path, index=False)
    
    gc.collect()
    
    return {
        "result_path": str(shrunk_output_path),
        "groups": len(perturbations),
        "elapsed_seconds": total_shrinkage_seconds,
        "shrinkage_type": "apeglm",
    }


# End of scanpy validation utilities
# ============================================================================


@dataclass
class BenchmarkMethod:
    """Description of a method that should be benchmarked."""

    name: str
    description: str
    function: Callable[..., Any]
    kwargs: Dict[str, Any]
    summary: Callable[[Any, Dict[str, Any]], Dict[str, Any]]
    depends_on: Optional[str] = None  # Name of method that must run first


@dataclass
class QCParams:
    """Quality control filtering parameters."""

    min_genes: int = 5
    min_cells_per_perturbation: int = 5
    min_cells_per_gene: int = 5
    chunk_size: int = 2048


@dataclass
class ResourceLimits:
    """Resource constraints for benchmark execution."""

    time_limit: int = 300  # seconds, 0 = no limit
    memory_limit: float = 4.0  # GB, 0 = no limit


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    dataset_path: Path
    dataset_name: str
    output_dir: Path
    perturbation_column: str = "perturbation"
    control_label: Optional[str] = None
    gene_name_column: Optional[str] = "gene_symbols"
    qc_params: Optional[QCParams] = field(default_factory=QCParams)  # None = adaptive
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    methods_to_run: Optional[List[str]] = None
    optional_methods: Optional[List[str]] = None  # Methods enabled only when explicitly listed
    show_progress: bool = True
    quiet: bool = False
    n_cores: Optional[int] = None
    force_restandardize: bool = False
    adaptive_qc_mode: str = "conservative"
    skip_existing: bool = True  # Skip methods with existing output files
    environment_config: Optional[EnvironmentConfig] = None
    chunk_size: Optional[int] = None  # Override chunk size for all operations
    use_docker: bool = False  # Run benchmarks inside Docker containers
    docker_image: str = "crispyx-benchmark:latest"  # Docker image name

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base_output_dir: Optional[Path] = None
    ) -> "BenchmarkConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        # Resolve dataset_path relative to REPO_ROOT if not absolute
        dataset_path = Path(data["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = REPO_ROOT / dataset_path
        
        dataset_name = data.get("dataset_name") or dataset_path.stem

        if base_output_dir:
            output_dir = base_output_dir / dataset_name
        else:
            output_dir = Path(
                data.get("output_dir", f"benchmarking/results/{dataset_name}")
            )
        
        # Resolve output_dir relative to REPO_ROOT if not absolute
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir

        qc_data = data.get("qc_params", {})
        # Allow qc_params to be null for adaptive calculation
        if qc_data is None:
            qc_params = None
        else:
            qc_params = QCParams(
                min_genes=qc_data.get("min_genes", 5),
                min_cells_per_perturbation=qc_data.get("min_cells_per_perturbation", 5),
                min_cells_per_gene=qc_data.get("min_cells_per_gene", 5),
                chunk_size=qc_data.get("chunk_size", 2048),
            )

        limits_data = data.get("resource_limits", {})
        resource_limits = ResourceLimits(
            time_limit=limits_data.get("time_limit", 300),
            memory_limit=limits_data.get("memory_limit", 4.0),
        )

        parallel_data = data.get("parallel_config", {})
        n_cores = parallel_data.get("n_cores")

        # Parse environment configuration
        env_data = data.get("environment_config", {})
        environment_config = None
        if env_data:
            environment_config = EnvironmentConfig(
                r_home=env_data.get("r_home"),
                default_n_cores=env_data.get("default_n_cores"),
            )

        force_restandardize = data.get("force_restandardize", False)
        adaptive_qc_mode = data.get("adaptive_qc_mode", "conservative")
        skip_existing = data.get("skip_existing", True)
        chunk_size = data.get("chunk_size")

        # Parse docker configuration
        docker_data = data.get("docker_config", {})
        use_docker = docker_data.get("enabled", False) if docker_data else False
        docker_image = docker_data.get("image", "crispyx-benchmark:latest") if docker_data else "crispyx-benchmark:latest"

        return cls(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            output_dir=output_dir,
            perturbation_column=data.get("perturbation_column", "perturbation"),
            control_label=data.get("control_label"),
            gene_name_column=data.get("gene_name_column", "gene_symbols"),
            qc_params=qc_params,
            resource_limits=resource_limits,
            methods_to_run=data.get("methods_to_run"),
            optional_methods=data.get("optional_methods"),
            show_progress=data.get("show_progress", True),
            quiet=data.get("quiet", False),
            n_cores=n_cores,
            force_restandardize=force_restandardize,
            adaptive_qc_mode=adaptive_qc_mode,
            skip_existing=skip_existing,
            environment_config=environment_config,
            chunk_size=chunk_size,
            use_docker=use_docker,
            docker_image=docker_image,
        )

    @classmethod
    def from_yaml(
        cls, yaml_path: Path
    ) -> Union["BenchmarkConfig", List["BenchmarkConfig"]]:
        """Load configuration(s) from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Multi-dataset mode
        if "datasets" in data:
            shared = data.get("shared_config", {})
            base_output = Path(shared.get("output_dir", "benchmarking/results"))
            # Resolve base_output relative to REPO_ROOT if not absolute
            if not base_output.is_absolute():
                base_output = REPO_ROOT / base_output

            configs = []
            for dataset_data in data["datasets"]:
                # Merge shared config with dataset-specific config
                merged = {**shared, **dataset_data}
                configs.append(cls.from_dict(merged, base_output))
            return configs

        # Single dataset mode
        return cls.from_dict(data)


@dataclass
class DifferentialComparisonSummary:
    """Summary statistics for comparing two DE methods."""

    test_type: str
    method_b_tool: str
    metrics: Dict[str, Optional[float]]
    method_a_result_path: str | None
    method_b_result_path: str | None
    error: Optional[str] = None
    method_a_runtime_seconds: Optional[float] = None
    method_b_runtime_seconds: Optional[float] = None
    method_a_peak_memory_mb: Optional[float] = None
    method_a_avg_memory_mb: Optional[float] = None
    method_a_peak_memory_absolute_mb: Optional[float] = None
    method_a_avg_memory_absolute_mb: Optional[float] = None
    method_b_peak_memory_mb: Optional[float] = None
    method_b_avg_memory_mb: Optional[float] = None
    method_b_peak_memory_absolute_mb: Optional[float] = None
    method_b_avg_memory_absolute_mb: Optional[float] = None

    @property
    def effect_max_abs_diff(self) -> Optional[float]:
        return self.metrics.get("effect_max_abs_diff")

    @property
    def statistic_max_abs_diff(self) -> Optional[float]:
        return self.metrics.get("statistic_max_abs_diff")

    @property
    def pvalue_max_abs_diff(self) -> Optional[float]:
        return self.metrics.get("pvalue_max_abs_diff")


# ============================================================================
# Cache function references removed - now imported from .cache module
# The following functions are now imported from benchmarking.tools.cache:
#   - get_expected_output_path (was _get_expected_output_path)
#   - is_scalar_na (was _is_scalar_na)
#   - make_json_serializable (was _make_json_serializable)
#   - save_method_result (was _save_method_result)
#   - load_method_result (was _load_method_result)
#   - load_cached_results (was _load_cached_results)
#   - validate_and_recover_cache_result (was _validate_and_recover_cache_result)
#   - save_cache_config (was _save_cache_config)
#   - load_cache_config (was _load_cache_config)
#   - invalidate_cache (was _invalidate_cache)
#   - check_output_exists (was _check_output_exists)
# ============================================================================


def _normalise_path(path: str | Path | None, context: Mapping[str, Any]) -> str | None:
    """Return ``path`` relative to the output directory or repository root."""

    if not path:
        return None

    path_obj = Path(path)
    candidates: Iterable[Path] = []
    output_dir = context.get("output_dir")
    if output_dir:
        candidates = [Path(str(output_dir))]
    repo_candidates = list(candidates) + [REPO_ROOT]

    for base in repo_candidates:
        try:
            return str(path_obj.resolve().relative_to(base.resolve()))
        except Exception:
            continue
    return str(path_obj)


def _percentage(part: float, total: float) -> float | None:
    """Return the percentage contribution of ``part`` to ``total``."""

    if total in (0, None):
        return None
    try:
        return (float(part) / float(total)) * 100.0
    except ZeroDivisionError:  # pragma: no cover - defensive
        return None


def _format_timing_summary(timings: Mapping[str, float]) -> str | None:
    """Return a compact human-readable summary for ``timings``."""

    if not timings:
        return None
    parts = [f"{name}={value:.3f}s" for name, value in sorted(timings.items())]
    return "; ".join(parts)


def _method_sort_key(method: BenchmarkMethod) -> tuple[int, int, str]:
    """Return a stable sort key that groups methods by task prefix.

    Within DE methods, t-test and wilcoxon run before nb-glm tests.
    """

    # Extract task order from method name
    name = method.name
    if "_qc_" in name or "_preprocess" in name:
        task_order = 0
    elif "_pb_" in name:
        task_order = 1
    elif "_de_" in name:
        task_order = 2
    else:
        task_order = 3

    # Sub-order within DE: t-test/wilcoxon first, then nb-glm/lfcshrink last
    if task_order == 2:
        if "_t_test" in name or "_wilcoxon" in name:
            sub_order = 0
        elif "_nb_glm" in name or "_lfcshrink" in name:
            sub_order = 2
        else:
            sub_order = 1  # edger, pydeseq2, etc.
    else:
        sub_order = 0

    return (task_order, sub_order, name)


def _postprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with normalised column names, ordering, and sorting."""

    if df.empty:
        return df

    table = df.copy()

    rename_map = {
        "elapsed_seconds": "runtime_seconds",
        "max_memory_mb": "peak_memory_mb",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})

    numeric_columns = [
        "runtime_seconds",
        "peak_memory_mb",
        "cells_total",
        "cells_kept",
        "cells_removed",
        "cells_kept_pct",
        "genes_total",
        "genes_kept",
        "genes_removed",
        "genes_kept_pct",
        "rows",
        "columns",
        "groups",
        "genes",
        "method_a_peak_memory_mb",
        "method_a_avg_memory_mb",
        "method_b_peak_memory_mb",
        "method_b_avg_memory_mb",
    ]
    numeric_columns.extend(
        key for key in DE_METRIC_KEYS if key not in numeric_columns
    )
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    # Sort by method name
    table = table.sort_values(["method"], kind="stable") if "method" in table.columns else table

    # Logical column grouping for results CSV
    preferred_order = [
        # 1. Method identification
        "method",
        "description",
        "status",
        # 2. Performance metrics
        "runtime_seconds",
        "peak_memory_mb",
        # 3. Data dimensions
        "cells_total",
        "cells_kept",
        "cells_kept_pct",
        "cells_removed",
        "genes_total",
        "genes_kept",
        "genes_kept_pct",
        "genes_removed",
        "rows",
        "columns",
        "groups",
        "genes",
        # 4. Comparison metadata
        "comparison_category",
        "test_type",
        "method_b_tool",
        # 5. Comparison metrics - Effect size
        "effect_pearson_corr",
        "effect_spearman_corr",
        "effect_top_k_overlap",
        "effect_max_abs_diff",
        # 6. Comparison metrics - Statistics
        "statistic_pearson_corr",
        "statistic_spearman_corr",
        "statistic_top_k_overlap",
        "statistic_max_abs_diff",
        # 7. Comparison metrics - P-values
        "pvalue_pearson_corr",
        "pvalue_spearman_corr",
        "pvalue_top_k_overlap",
        "pvalue_max_abs_diff",
        "pvalue_method_a_auroc",
        "pvalue_method_b_auroc",
        # 8. Detailed comparison data
        "method_a_peak_memory_mb",
        "method_a_avg_memory_mb",
        "method_b_peak_memory_mb",
        "method_b_avg_memory_mb",
        "method_a_timing_breakdown",
        "method_b_timing_breakdown",
        "runtime_diff_seconds",
        "runtime_diff_pct",
        "memory_diff_mb",
        "memory_diff_pct",
        # 9. File paths
        "result_path",
        "method_a_result_path",
        "method_b_result_path",
        # 10. Errors
        "error",
    ]
    ordered_columns = [col for col in preferred_order if col in table.columns]
    remaining_columns = [col for col in table.columns if col not in ordered_columns]
    table = table[ordered_columns + remaining_columns]

    # Drop internal tracking columns
    internal_columns = ["_loaded_from_cache"]
    table = table.drop(columns=[c for c in internal_columns if c in table.columns], errors="ignore")

    table = table.dropna(axis=1, how="all")
    return table.reset_index(drop=True)


def _compute_aggregate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return aggregate benchmark statistics derived from ``df``."""

    summary: Dict[str, Any] = {
        "total_methods": int(len(df)),
        "status_counts": {},
        "success_count": 0,
        "timeout_count": 0,
        "memory_limit_count": 0,
        "error_count": 0,
        "non_success_count": int(len(df)),
        "success_rate": None,
        "average_runtime_seconds": None,
        "categories": [],
        "dependency_errors": [],
        "other_errors": [],
        "error_details": [],
    }

    if df.empty:
        return summary

    status_series = df.get("status")
    if status_series is not None:
        filled_status = status_series.fillna("unknown").astype(str)
    else:
        filled_status = pd.Series(["unknown"] * len(df))

    status_counts = Counter(filled_status)
    ordered_status_counts: Dict[str, int] = {}
    for status in STATUS_ORDER:
        count = int(status_counts.get(status, 0))
        if count:
            ordered_status_counts[status] = count
    for status, count in status_counts.items():
        status = str(status)
        if status not in ordered_status_counts:
            ordered_status_counts[status] = int(count)

    summary["status_counts"] = ordered_status_counts
    # Count both 'success' and 'recovered' as successful completions
    summary["success_count"] = ordered_status_counts.get("success", 0) + ordered_status_counts.get("recovered", 0)
    summary["recovered_count"] = ordered_status_counts.get("recovered", 0)
    summary["timeout_count"] = ordered_status_counts.get("timeout", 0)
    summary["memory_limit_count"] = ordered_status_counts.get("memory_limit", 0)
    summary["error_count"] = ordered_status_counts.get("error", 0)
    summary["non_success_count"] = summary["total_methods"] - summary["success_count"]

    if summary["total_methods"]:
        summary["success_rate"] = summary["success_count"] / summary["total_methods"]

    runtime_series = df.get("runtime_seconds")
    if runtime_series is not None:
        runtimes = pd.to_numeric(runtime_series, errors="coerce").dropna()
        if not runtimes.empty:
            summary["average_runtime_seconds"] = float(runtimes.mean())

    # Remove category-based summaries since categories no longer exist
    summary["categories"] = []

    if "error" in df.columns:
        error_rows = df[df["error"].notna()]
        details = []
        for _, row in error_rows.iterrows():
            details.append(
                {
                    "method": str(row.get("method", "")),
                    "error": str(row["error"]),
                }
            )
        summary["error_details"] = details

        dependency_keywords = ("importerror", "modulenotfounderror", "no module named")
        dependency_errors = {
            detail["error"]
            for detail in details
            if any(keyword in detail["error"].lower() for keyword in dependency_keywords)
        }
        other_errors = {
            detail["error"]
            for detail in details
            if detail["error"] not in dependency_errors
        }
        summary["dependency_errors"] = sorted(dependency_errors)
        summary["other_errors"] = sorted(other_errors)

    return summary


def _anndata_to_de_dict_legacy(adata: Any) -> Dict[str, Any]:
    """Convert AnnData with DE results to a dictionary of result objects (legacy version - not used).
    
    This function has been replaced by the updated _anndata_to_de_dict with proper
    sparse matrix handling. Kept for reference only.
    """
    from types import SimpleNamespace
    
    result_dict = {}
    
    # crispyx stores results in layers: logfoldchange, statistic, pvalue
    # with obs representing perturbations
    if "logfoldchange" in adata.layers and "pvalue" in adata.layers:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=adata.layers["logfoldchange"][idx, :],
                statistic=adata.layers.get("statistic", [None] * adata.n_vars)[idx, :] if "statistic" in adata.layers else None,
                pvalue=adata.layers["pvalue"][idx, :],
            )
    
    return result_dict


def _anndata_to_de_dict(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format.
    
    Sparse Matrix Handling
    ----------------------
    Properly extracts 1D dense arrays from sparse matrices in adata.X and adata.layers.
    Uses np.asarray() to convert sparse row slices to dense without full matrix densification.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing DE results in layers (crispyx format)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping perturbation names to SimpleNamespace objects with
        genes, effect_size, statistic, and pvalue attributes
    """
    from types import SimpleNamespace
    
    def _extract_row(matrix, idx: int) -> np.ndarray | None:
        """Extract row from matrix (sparse or dense) as 1D dense array."""
        if matrix is None:
            return None
        if idx >= matrix.shape[0]:
            return None
        try:
            row = matrix[idx, :]
            # Convert to dense and flatten to 1D
            # For sparse matrices, this extracts only one row (memory efficient)
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    # crispyx stores results in layers with obs=perturbations
    # logfoldchange layer is optional (not present for t-test which stores effect in .X)
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            # Determine which statistic layer to use
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            # Determine which effect size layer/matrix to use
            if "logfoldchange" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif "logfoldchanges" in adata.layers:
                # crispyx t-test/Wilcoxon store LFCs in "logfoldchanges" (plural)
                effect_size_values = _extract_row(adata.layers["logfoldchanges"], idx)
            elif adata.X is not None:
                # t_test stores effect sizes in .X matrix
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    # Fallback for Scanpy format
    elif "rank_genes_groups" in adata.uns:
        rgg = adata.uns["rank_genes_groups"]
        # Get group names from structured array
        if hasattr(rgg["names"], "dtype") and hasattr(rgg["names"].dtype, "names"):
            groups = list(rgg["names"].dtype.names)
        else:
            groups = []
        
        for group in groups:
            # Use names from rank_genes_groups if available, otherwise fallback
            if "names" in rgg:
                genes = rgg["names"][group]
            else:
                genes = adata.uns.get("genes", adata.var_names.to_numpy())

            stream_result_dict[group] = SimpleNamespace(
                genes=genes,
                effect_size=rgg["logfoldchanges"][group] if "logfoldchanges" in rgg else None,
                statistic=rgg["scores"][group] if "scores" in rgg else None,
                pvalue=rgg["pvals"][group] if "pvals" in rgg else None,
            )
    
    return stream_result_dict


def _anndata_to_de_dict_raw(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format, using RAW (unshrunken) LFCs.
    
    This is identical to _anndata_to_de_dict but uses 'logfoldchange_raw' layer
    instead of 'logfoldchange' for effect sizes. This allows comparison of
    raw LFCs between crispyx and PyDESeq2 (before any shrinkage).
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing DE results in layers (crispyx format)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping perturbation names to SimpleNamespace objects with
        genes, effect_size (raw), statistic, and pvalue attributes
    """
    from types import SimpleNamespace
    
    def _extract_row(matrix, idx: int) -> np.ndarray | None:
        """Extract row from matrix (sparse or dense) as 1D dense array."""
        if matrix is None:
            return None
        if idx >= matrix.shape[0]:
            return None
        try:
            row = matrix[idx, :]
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            # Determine which statistic layer to use
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            # Use RAW logfoldchange (before shrinkage) for effect sizes
            if "logfoldchange_raw" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange_raw"], idx)
            elif "logfoldchange" in adata.layers:
                # Fallback to shrunken if raw not available
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif adata.X is not None:
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    
    return stream_result_dict


def _make_gene_index_unique(gene_index: pd.Index) -> pd.Index:
    """Append positional suffix to duplicate gene names (e.g. MATR3, MATR3-1).

    Prevents many-to-many merges in ``compute_de_comparison_metrics`` when
    datasets contain duplicate gene symbols (e.g. Feng gwsf/gwsnf/ts).
    """
    if not gene_index.duplicated().any():
        return gene_index
    counts: dict[str, int] = {}
    unique: list[str] = []
    for g in gene_index:
        if g in counts:
            counts[g] += 1
            unique.append(f"{g}-{counts[g]}")
        else:
            counts[g] = 0
            unique.append(g)
    return pd.Index(unique)


def _streaming_de_to_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a streaming differential expression mapping to a tidy DataFrame."""

    frames = []
    _gene_cache: dict[int, pd.Index] = {}
    for perturbation, entry in result.items():
        genes = getattr(entry, "genes", None)
        if genes is None:
            continue
        genes_id = id(genes)
        if genes_id in _gene_cache:
            gene_index = _gene_cache[genes_id]
        else:
            gene_index = _make_gene_index_unique(pd.Index(genes).astype(str))
            _gene_cache[genes_id] = gene_index
        n_rows = len(gene_index)
        frame = pd.DataFrame(
            {
                "perturbation": [str(perturbation)] * n_rows,
                "gene": gene_index,
                "effect_size": getattr(entry, "effect_size", pd.NA),
                "statistic": getattr(entry, "statistic", pd.NA),
                "pvalue": getattr(entry, "pvalue", pd.NA),
            }
        )
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=STANDARD_DE_COLUMNS)


def _standardise_de_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return ``df`` with standard differential expression column names."""

    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_DE_COLUMNS)

    result = df.copy()
    lower_to_original = {col.lower(): col for col in result.columns}

    def _resolve_column(candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                return original
        return None

    rename: Dict[str, str] = {}
    perturbation_col = _resolve_column(["perturbation", "group", "cluster", "label", "contrast"])
    gene_col = _resolve_column(["gene", "genes", "name", "names", "feature", "variable"])
    effect_col = _resolve_column(["effect_size", "logfoldchange", "logfoldchanges", "logfc", "lfc", "log_fc", "coefficient"])
    stat_col = _resolve_column(["statistic", "statistics", "stat", "score", "scores", "wald_statistic", "zscore", "t_stat", "t_value", "t_statistic", "f", "f_value", "u_stat"])
    pvalue_col = _resolve_column(["pvalue", "p_value", "pval", "pvals", "pvalue_raw", "pvalue_adj", "pvals_adj"])

    if perturbation_col is not None:
        rename[perturbation_col] = "perturbation"
    if gene_col is not None:
        rename[gene_col] = "gene"
    if effect_col is not None:
        rename[effect_col] = "effect_size"
    if stat_col is not None:
        rename[stat_col] = "statistic"
    if pvalue_col is not None:
        rename[pvalue_col] = "pvalue"

    result = result.rename(columns=rename)

    for column in STANDARD_DE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[STANDARD_DE_COLUMNS]
    result["perturbation"] = result["perturbation"].astype(str).str.strip()
    result["gene"] = result["gene"].astype(str).str.strip()
    # Deduplicate gene names within each perturbation to prevent many-to-many
    # merges when datasets contain duplicate gene symbols (e.g. Feng datasets).
    if result.duplicated(subset=["perturbation", "gene"]).any():
        cumcounts = result.groupby(["perturbation", "gene"], sort=False).cumcount()
        mask = cumcounts > 0
        if mask.any():
            result = result.copy()
            result.loc[mask, "gene"] = result.loc[mask, "gene"] + "-" + cumcounts[mask].astype(str)
    result["effect_size"] = pd.to_numeric(result["effect_size"], errors="coerce")
    result["statistic"] = pd.to_numeric(result["statistic"], errors="coerce")
    result["pvalue"] = pd.to_numeric(result["pvalue"], errors="coerce")
    return result


def _compare_de_frames(
    streaming: pd.DataFrame, reference: pd.DataFrame
) -> Dict[str, Optional[float]]:
    """Return comparison metrics between streaming and reference DE results."""

    return compute_de_comparison_metrics(streaming, reference)





def _resolve_pertpy_runner(module: Any, method: str) -> Optional[Callable[..., Any]]:
    """Best-effort resolution of a Pertpy differential expression runner."""

    candidates = [
        method,
        f"{method}_de",
        f"run_{method}",
        f"run_{method}_de",
        method.lower(),
        method.upper(),
    ]
    for name in candidates:
        runner = getattr(module, name, None)
        if callable(runner):
            return runner
    return None


def _resolve_pertpy_class_runner(module: Any, method: str) -> Optional[Callable[..., Any]]:
    """Return a callable wrapper for class-based Pertpy differential expression APIs."""

    class_aliases = {
        "edger": ["EdgeR"],
        "pydeseq2": ["PyDESeq2"],
        "statsmodels": ["Statsmodels"],
        "ttest": ["TTest"],
        "wilcoxon": ["WilcoxonTest"],
    }

    method_key = method.lower()
    candidate_names = list(class_aliases.get(method_key, []))
    candidate_names.extend(
        name
        for name in {
            method,
            method.capitalize(),
            method.upper(),
            method.title(),
        }
        if isinstance(name, str)
    )
    # Preserve order while removing duplicates
    seen: set[str] = set()
    deduped: list[str] = []
    for name in candidate_names:
        if not isinstance(name, str):
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    candidate_names = deduped

    for name in candidate_names:
        cls = getattr(module, name, None)
        if cls is None or not isinstance(cls, type):
            continue
        compare_groups = getattr(cls, "compare_groups", None)
        if compare_groups is None or not callable(compare_groups):
            continue

        current_cls = cls

        def runner(
            adata,
            *,
            groupby=None,
            group_key=None,
            control=None,
            reference=None,
            **kwargs,
        ):
            _ = kwargs  # Allow compatibility with legacy keyword arguments
            column = groupby or group_key
            baseline = control if control is not None else reference
            if column is None:
                raise TypeError("Pertpy runner requires a groupby column")
            if baseline is None:
                raise TypeError("Pertpy runner requires a control/reference label")

            obs_column = adata.obs[column]
            groups_to_compare = [value for value in obs_column.unique().tolist() if value != baseline]
            if not groups_to_compare:
                raise ValueError("No perturbation groups available for comparison")

            return current_cls.compare_groups(
                adata,
                column=column,
                baseline=baseline,
                groups_to_compare=groups_to_compare,
            )

        return runner
    return None


def _convert_reference_result_to_dataframe(result: Any) -> Optional[pd.DataFrame]:
    """Normalise a Pertpy reference result to a ``DataFrame`` when possible."""

    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, Mapping):
        frames = []
        for perturbation, value in result.items():
            if isinstance(value, pd.DataFrame):
                frame = value.copy()
                if "perturbation" not in frame.columns:
                    frame["perturbation"] = str(perturbation)
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()
    if hasattr(result, "to_df"):
        return result.to_df()
    return None


def _get_performance_emoji(pct: Optional[float], is_lower_better: bool = True) -> str:
    """Return emoji indicator for performance comparison.
    
    Parameters
    ----------
    pct : float | None
        Percentage value (method_a / method_b * 100)
    is_lower_better : bool
        If True, lower percentage means method_a is better (faster/less memory)
        
    Returns
    -------
    str
        ✅ if method_a is significantly better (>10% improvement)
        ⚠️ if similar (within ±10%)
        ❌ if method_a is significantly worse (>10% worse)
    """
    if pct is None or pd.isna(pct):
        return ""
    
    if is_lower_better:
        if pct < 90:  # method_a uses <90% of method_b's resources = better
            return "✅"
        elif pct > 110:  # method_a uses >110% of method_b's resources = worse
            return "❌"
        else:
            return "⚠️"
    else:
        if pct > 110:
            return "✅"
        elif pct < 90:
            return "❌"
        else:
            return "⚠️"


def _get_accuracy_emoji(corr: Optional[float]) -> str:
    """Return emoji indicator for accuracy/correlation.
    
    Parameters
    ----------
    corr : float | None
        Correlation value (0-1)
        
    Returns
    -------
    str
        ✅ if excellent (>0.95)
        ⚠️ if good (0.8-0.95)
        ❌ if poor (<0.8)
    """
    if corr is None or pd.isna(corr):
        return ""
    
    if corr >= 0.95:
        return "✅"
    elif corr >= 0.8:
        return "⚠️"
    else:
        return "❌"


def _format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    """Format mean ± std as a two-line string for markdown tables.
    
    Parameters
    ----------
    mean : float | None
        Mean value
    std : float | None
        Standard deviation value
        
    Returns
    -------
    str
        Formatted string like "0.950<br><small>±0.023</small>" or "-" if no data
    """
    if mean is None or pd.isna(mean):
        return "-"
    
    if std is None or pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    
    return f"{mean:.3f}<br><small>±{std:.3f}</small>"


def _get_method_category(method_name: str) -> tuple[str, str, int]:
    """Get category and test type for a method.
    
    Returns
    -------
    tuple[str, str, int]
        (category, test_type, sort_order)
    """
    if not isinstance(method_name, str):
        return ("other", "unknown", 99)
    
    if "_qc_" in method_name or method_name.endswith("_qc_filtered"):
        return ("Preprocessing / QC", "qc", 0)
    elif "_pb_avg" in method_name:
        return ("Preprocessing / QC", "pseudobulk_avg", 1)
    elif "_pb_pseudobulk" in method_name:
        return ("Preprocessing / QC", "pseudobulk", 2)
    elif "_de_t_test" in method_name:
        return ("DE: t-test", "t_test", 10)
    elif "_de_wilcoxon" in method_name:
        return ("DE: Wilcoxon", "wilcoxon", 20)
    elif "_de_nb_glm" in method_name:
        return ("DE: NB GLM", "nb_glm", 30)
    elif "_de_glm" in method_name:  # edger
        return ("DE: NB GLM", "edger", 32)
    elif "_de_pydeseq2_shrunk" in method_name:
        return ("DE: NB GLM", "pydeseq2_shrunk", 34)
    elif "_de_pydeseq2" in method_name:
        return ("DE: NB GLM", "pydeseq2", 33)
    else:
        return ("Other", "unknown", 99)


def _format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format percentage value."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}%"


def _format_diff(value: Optional[float], unit: str = "s", decimals: int = 1) -> str:
    """Format difference value with sign."""
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}{unit}"


def _format_method_name(method: str) -> str:
    """Format method name for display with proper capitalization."""
    # Check for specific tool names first
    if "edger_" in method.lower():
        return "NB-GLM"
    # Check for shrunk PyDESeq2 before regular PyDESeq2
    if "pertpy_" in method.lower() and "pydeseq2_shrunk" in method.lower():
        return "NB-GLM (lfcShrink)"
    if "pertpy_" in method.lower() and "pydeseq2" in method.lower():
        return "NB-GLM"
    
    # Remove common prefixes
    name = method.replace("crispyx_", "").replace("scanpy_", "").replace("pertpy_", "").replace("edger_", "")
    
    # Apply specific formatting
    name = name.replace("_", " ")
    
    # Proper capitalizations - lowercase except for specific terms
    replacements = {
        "de t test": "t-test",
        "de wilcoxon": "Wilcoxon",
        "de nb glm joint": "NB-GLM (joint)",
        "de nb glm": "NB-GLM",
        "de glm": "NB-GLM",
        "de pydeseq2": "NB-GLM",
        "qc filtered": "QC filter",
        "pb avg log": "pseudobulk (avg log)",
        "pb avg": "pseudobulk (avg)",
        "pb pseudobulk": "pseudobulk",
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
            break
    
    return name


def _get_method_package(method: str) -> str:
    """Get the package name for a method."""
    if method.startswith("crispyx_"):
        return "crispyx"
    elif method.startswith("scanpy_"):
        return "scanpy"
    elif method.startswith("edger_"):
        return "edgeR"
    elif method.startswith("pertpy_"):
        return "pertpy"
    return ""


def _format_full_method_name(method: str) -> str:
    """Format full method name including package (e.g., 'edgeR NB-GLM')."""
    package = _get_method_package(method)
    name = _format_method_name(method)
    if package:
        return f"{package} {name}"
    return name


def _is_crispyx_method(method: str) -> bool:
    """Check if method is a crispyx method."""
    return method.startswith("crispyx_")


def _generate_improved_markdown(
    perf_df: pd.DataFrame,
    perf_comp_results: List[Dict[str, Any]],
    accuracy_results: List[Dict[str, Any]],
    overlap_heatmaps: Optional[Dict[str, Path]] = None,
) -> str:
    """Generate improved markdown with categorized tables and emoji indicators."""
    
    md = "# Benchmark Results\n\n"
    
    # =========================================================================
    # Section 1: Performance by Category
    # =========================================================================
    md += "## 1. Performance\n\n"
    
    if not perf_df.empty:
        # Add category info to each method
        perf_df = perf_df.copy()
        perf_df["_category"] = perf_df["method"].apply(lambda x: _get_method_category(x)[0])
        perf_df["_sort_order"] = perf_df["method"].apply(lambda x: _get_method_category(x)[2])
        perf_df = perf_df.sort_values("_sort_order")
        
        # Format method names for display
        perf_df["Package"] = perf_df["method"].apply(_get_method_package)
        perf_df["Method"] = perf_df["method"].apply(_format_method_name)
        
        # Group by category
        categories = perf_df["_category"].unique()
        
        for category in categories:
            cat_df = perf_df[perf_df["_category"] == category].copy()
            
            md += f"### {category}\n\n"
            
            # Select columns based on category
            if "Preprocessing" in category or "QC" in category:
                cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "cells_kept", "genes_kept"]
            else:
                cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "groups", "genes"]
            
            # Keep only columns that exist
            cols = [c for c in cols if c in cat_df.columns]
            display_df = cat_df[cols].copy()
            
            # Rename columns for display
            rename_map = {
                "status": "Status",
                "elapsed_seconds": "Time (s)",
                "peak_memory_mb": "Memory (MB)",
                "cells_kept": "Cells",
                "genes_kept": "Genes",
                "groups": "Groups",
                "genes": "Genes",
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
            
            # Format numeric columns - integers for counts, decimals for time/memory
            for col in display_df.columns:
                if col in ["Cells", "Genes", "Groups"]:
                    # Format as integers
                    display_df[col] = display_df[col].apply(
                        lambda x: int(x) if pd.notna(x) else x
                    )
                elif col in display_df.select_dtypes(include=["number"]).columns:
                    display_df[col] = display_df[col].round(2)
            
            md += _frame_to_markdown_table(display_df)
            md += "\n\n"
    
    # =========================================================================
    # Section 2: Performance Comparison (crispyx as baseline)
    # =========================================================================
    md += "## 2. Performance Comparison\n\n"
    
    if perf_comp_results:
        # Separate crispyx comparisons from other comparisons
        crispyx_comps = []
        other_comps = []
        
        for comp in perf_comp_results:
            comparison = comp["comparison"]
            method_a = comparison.split(" vs ")[0]
            if _is_crispyx_method(method_a):
                crispyx_comps.append(comp)
            else:
                other_comps.append(comp)
        
        # Section 2a: crispyx as baseline
        if crispyx_comps:
            md += "### crispyx vs Reference Tools\n\n"
            md += "_crispyx as baseline. Negative values = crispyx is faster/uses less memory._\n\n"
            
            # Group by category
            comp_by_category: Dict[str, List[Dict[str, Any]]] = {}
            for comp in crispyx_comps:
                comparison = comp["comparison"]
                method_a = comparison.split(" vs ")[0]
                category = _get_method_category(method_a)[0]
                
                if category not in comp_by_category:
                    comp_by_category[category] = []
                comp_by_category[category].append(comp)
            
            for category, comps in comp_by_category.items():
                md += f"#### {category}\n\n"
                
                rows = []
                for comp in comps:
                    comparison = comp["comparison"]
                    parts = comparison.split(" vs ")
                    method_a = _format_method_name(parts[0])
                    method_b = _format_full_method_name(parts[1])
                    
                    time_pct = comp.get("time_pct")
                    mem_pct = comp.get("mem_pct")
                    time_diff = comp.get("time_diff_s")
                    mem_diff = comp.get("mem_diff_mb")
                    
                    time_emoji = _get_performance_emoji(time_pct, is_lower_better=True)
                    mem_emoji = _get_performance_emoji(mem_pct, is_lower_better=True)
                    
                    rows.append({
                        "crispyx method": method_a,
                        "compared to": method_b,
                        "Time Δ": _format_diff(time_diff, "s"),
                        "Time %": _format_pct(time_pct),
                        "": time_emoji,
                        "Mem Δ": _format_diff(mem_diff, " MB"),
                        "Mem %": _format_pct(mem_pct),
                        " ": mem_emoji,
                    })
                
                md += _frame_to_markdown_table(pd.DataFrame(rows))
                md += "\n\n"
        
        # Section 2b: Other comparisons (e.g., edgeR vs PyDESeq2)
        if other_comps:
            md += "### Tool Comparisons\n\n"
            md += "_Comparisons between external tools._\n\n"
            
            rows = []
            for comp in other_comps:
                comparison = comp["comparison"]
                parts = comparison.split(" vs ")
                method_a = _format_method_name(parts[0])
                method_b = _format_method_name(parts[1])
                
                time_pct = comp.get("time_pct")
                mem_pct = comp.get("mem_pct")
                time_diff = comp.get("time_diff_s")
                mem_diff = comp.get("mem_diff_mb")
                
                # For non-crispyx comparisons, interpret: method_a is the one we want to be faster
                time_emoji = _get_performance_emoji(time_pct, is_lower_better=True)
                mem_emoji = _get_performance_emoji(mem_pct, is_lower_better=True)
                
                rows.append({
                    "package A": _get_method_package(parts[0]),
                    "method A": method_a,
                    "package B": _get_method_package(parts[1]),
                    "method B": method_b,
                    "Time Δ (A-B)": _format_diff(time_diff, "s"),
                    "Time % (A/B)": _format_pct(time_pct),
                    "": time_emoji,
                    "Mem Δ (A-B)": _format_diff(mem_diff, " MB"),
                    "Mem % (A/B)": _format_pct(mem_pct),
                    " ": mem_emoji,
                })
            
            md += _frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n\n"
    
    # =========================================================================
    # Section 3: Accuracy Comparison
    # =========================================================================
    md += "## 3. Accuracy\n\n"
    md += "_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_\n\n"
    
    if accuracy_results:
        # Separate crispyx comparisons from other comparisons
        crispyx_accs = []
        other_accs = []
        
        for acc in accuracy_results:
            comparison = acc["comparison"]
            method_a = comparison.split(" vs ")[0]
            if _is_crispyx_method(method_a):
                crispyx_accs.append(acc)
            else:
                other_accs.append(acc)
        
        # Section 3a: crispyx accuracy comparisons
        if crispyx_accs:
            # Group by category
            acc_by_category: Dict[str, List[Dict[str, Any]]] = {}
            
            for acc in crispyx_accs:
                comparison = acc["comparison"]
                method_a = comparison.split(" vs ")[0]
                category = _get_method_category(method_a)[0]
                
                if category not in acc_by_category:
                    acc_by_category[category] = []
                acc_by_category[category].append(acc)
            
            for category, accs in acc_by_category.items():
                md += f"### {category}\n\n"
                
                # Check if this is QC comparison
                is_qc = "QC" in category or "Preprocessing" in category
                
                if is_qc:
                    # QC comparisons - show cell/gene differences
                    rows = []
                    for acc in accs:
                        comparison = acc["comparison"]
                        parts = comparison.split(" vs ")
                        method_a = _format_method_name(parts[0])
                        method_b = _format_full_method_name(parts[1])
                        
                        cells_diff = acc.get("cells_diff", 0)
                        genes_diff = acc.get("genes_diff", 0)
                        
                        # Perfect match = ✅
                        cells_emoji = "✅" if cells_diff == 0 else ("⚠️" if abs(cells_diff) < 10 else "❌")
                        genes_emoji = "✅" if genes_diff == 0 else ("⚠️" if abs(genes_diff) < 10 else "❌")
                        
                        rows.append({
                            "crispyx method": method_a,
                            "compared to": method_b,
                            "Cells Δ": f"{int(cells_diff):+d}" if pd.notna(cells_diff) else "-",
                            "": cells_emoji,
                            "Genes Δ": f"{int(genes_diff):+d}" if pd.notna(genes_diff) else "-",
                            " ": genes_emoji,
                        })
                    
                    md += _frame_to_markdown_table(pd.DataFrame(rows))
                else:
                    # DE comparisons - show correlations (Pearson and Spearman)
                    rows = []
                    for acc in accs:
                        comparison = acc["comparison"]
                        parts = comparison.split(" vs ")
                        method_a = _format_method_name(parts[0])
                        method_b = _format_full_method_name(parts[1])
                        
                        effect_p_mean = acc.get("effect_pearson_corr_mean")
                        effect_p_std = acc.get("effect_pearson_corr_std")
                        effect_s_mean = acc.get("effect_spearman_corr_mean")
                        effect_s_std = acc.get("effect_spearman_corr_std")
                        stat_p_mean = acc.get("statistic_pearson_corr_mean")
                        stat_p_std = acc.get("statistic_pearson_corr_std")
                        stat_s_mean = acc.get("statistic_spearman_corr_mean")
                        stat_s_std = acc.get("statistic_spearman_corr_std")
                        pval_p_mean = acc.get("pvalue_log_pearson_corr_mean")
                        pval_p_std = acc.get("pvalue_log_pearson_corr_std")
                        pval_s_mean = acc.get("pvalue_log_spearman_corr_mean")
                        pval_s_std = acc.get("pvalue_log_spearman_corr_std")
                        
                        rows.append({
                            "crispyx method": method_a,
                            "compared to": method_b,
                            "Eff ρ": _format_mean_std(effect_p_mean, effect_p_std),
                            "": _get_accuracy_emoji(effect_p_mean),
                            "Eff ρₛ": _format_mean_std(effect_s_mean, effect_s_std),
                            " ": _get_accuracy_emoji(effect_s_mean),
                            "Stat ρ": _format_mean_std(stat_p_mean, stat_p_std),
                            "  ": _get_accuracy_emoji(stat_p_mean),
                            "Stat ρₛ": _format_mean_std(stat_s_mean, stat_s_std),
                            "   ": _get_accuracy_emoji(stat_s_mean),
                            "log-Pval ρ": _format_mean_std(pval_p_mean, pval_p_std),
                            "    ": _get_accuracy_emoji(pval_p_mean),
                            "log-Pval ρₛ": _format_mean_std(pval_s_mean, pval_s_std),
                            "     ": _get_accuracy_emoji(pval_s_mean),
                        })
                    
                    md += _frame_to_markdown_table(pd.DataFrame(rows))
                
                md += "\n\n"
        
        # Section 3b: Other accuracy comparisons (e.g., edgeR vs PyDESeq2)
        if other_accs:
            md += "### Tool Comparisons\n\n"
            
            rows = []
            for acc in other_accs:
                comparison = acc["comparison"]
                parts = comparison.split(" vs ")
                method_a = _format_method_name(parts[0])
                method_b = _format_method_name(parts[1])
                
                effect_p_mean = acc.get("effect_pearson_corr_mean")
                effect_p_std = acc.get("effect_pearson_corr_std")
                effect_s_mean = acc.get("effect_spearman_corr_mean")
                effect_s_std = acc.get("effect_spearman_corr_std")
                stat_p_mean = acc.get("statistic_pearson_corr_mean")
                stat_p_std = acc.get("statistic_pearson_corr_std")
                stat_s_mean = acc.get("statistic_spearman_corr_mean")
                stat_s_std = acc.get("statistic_spearman_corr_std")
                pval_p_mean = acc.get("pvalue_log_pearson_corr_mean")
                pval_p_std = acc.get("pvalue_log_pearson_corr_std")
                pval_s_mean = acc.get("pvalue_log_spearman_corr_mean")
                pval_s_std = acc.get("pvalue_log_spearman_corr_std")
                
                rows.append({
                    "package A": _get_method_package(parts[0]),
                    "method A": method_a,
                    "package B": _get_method_package(parts[1]),
                    "method B": method_b,
                    "Eff ρ": _format_mean_std(effect_p_mean, effect_p_std),
                    "": _get_accuracy_emoji(effect_p_mean),
                    "Eff ρₛ": _format_mean_std(effect_s_mean, effect_s_std),
                    " ": _get_accuracy_emoji(effect_s_mean),
                    "Stat ρ": _format_mean_std(stat_p_mean, stat_p_std),
                    "  ": _get_accuracy_emoji(stat_p_mean),
                    "Stat ρₛ": _format_mean_std(stat_s_mean, stat_s_std),
                    "   ": _get_accuracy_emoji(stat_s_mean),
                    "log-Pval ρ": _format_mean_std(pval_p_mean, pval_p_std),
                    "    ": _get_accuracy_emoji(pval_p_mean),
                    "log-Pval ρₛ": _format_mean_std(pval_s_mean, pval_s_std),
                    "     ": _get_accuracy_emoji(pval_s_mean),
                })
            
            md += _frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n\n"
    
    # =========================================================================
    # Section 4: Gene Set Overlap
    # =========================================================================
    md += "## 4. Gene Set Overlap\n\n"
    md += "_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_\n\n"
    
    if accuracy_results:
        # Build overlap tables for effect size and p-value
        for metric_type, metric_label in [("effect", "Effect Size"), ("pvalue", "P-value")]:
            md += f"### {metric_label} Overlap\n\n"
            
            rows = []
            for acc in accuracy_results:
                comparison = acc["comparison"]
                parts = comparison.split(" vs ")
                method_a = _format_method_name(parts[0])
                method_b = _format_full_method_name(parts[1])
                
                # Get overlap metrics for different k values
                k50 = acc.get(f"{metric_type}_top_k_overlap")
                k100_mean = acc.get(f"{metric_type}_top_100_overlap_mean")
                k100_std = acc.get(f"{metric_type}_top_100_overlap_std")
                k500_mean = acc.get(f"{metric_type}_top_500_overlap_mean")
                k500_std = acc.get(f"{metric_type}_top_500_overlap_std")
                
                # Skip if no overlap data
                if k50 is None and k100_mean is None and k500_mean is None:
                    continue
                
                def _get_overlap_emoji(val: Optional[float]) -> str:
                    if val is None or pd.isna(val):
                        return ""
                    if val >= 0.7:
                        return "✅"
                    elif val >= 0.5:
                        return "⚠️"
                    else:
                        return "❌"
                
                rows.append({
                    "crispyx method": method_a,
                    "compared to": method_b,
                    "Top-50": f"{k50:.3f}" if k50 is not None and pd.notna(k50) else "-",
                    "": _get_overlap_emoji(k50),
                    "Top-100": _format_mean_std(k100_mean, k100_std),
                    " ": _get_overlap_emoji(k100_mean),
                    "Top-500": _format_mean_std(k500_mean, k500_std),
                    "  ": _get_overlap_emoji(k500_mean),
                })
            
            if rows:
                md += _frame_to_markdown_table(pd.DataFrame(rows))
            else:
                md += "_No overlap data available._\n"
            md += "\n\n"
    else:
        md += "_No overlap data available._\n\n"
    
    # Embed top-100 overlap heatmaps if available
    if overlap_heatmaps:
        md += "### Overlap Heatmaps (Top-100)\n\n"
        
        for metric, label in [("effect", "Effect Size"), ("pvalue", "P-value")]:
            heatmap_key = f"{metric}_top_100_overlap.png"
            if heatmap_key in overlap_heatmaps:
                md += f"#### {label}\n\n"
                md += f"![{label} Top-100 Overlap]({heatmap_key})\n\n"
    
    # =========================================================================
    # Legend
    # =========================================================================
    md += "---\n\n"
    md += "**Legend:**\n"
    md += "- Performance: ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse\n"
    md += "- Accuracy: ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8\n"
    md += "- Overlap: ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5\n"
    md += "- ρ = Pearson correlation, ρₛ = Spearman correlation\n"
    md += "- Correlation and overlap values shown as mean±std across perturbations\n"
    md += "- log-Pval: correlations computed on -log₁₀(p) transformed values\n"
    md += "- Top-k overlap: fraction of top-k genes shared between methods\n"
    md += "- lfcShrink column: shrinkage type used (apeglm, ashr, normal) or blank if none\n"
    md += "- NB-GLM sf=per: per-comparison size factor estimation (matches PyDESeq2 behavior)\n"
    md += "- NB-GLM (base): global size factor estimation across all comparisons\n"
    md += "\n**Note on lfcShrink methods:**\n"
    md += "- crispyx uses `method='full'` for benchmarking, which re-fits the full NB likelihood with L-BFGS-B optimization per gene. This matches PyDESeq2's apeglm shrinkage exactly.\n"
    md += "- For production use, `method='stats'` (default in `shrink_lfc`) is ~35× faster by using pre-computed MLE statistics with Newton-Raphson optimization, but results are approximate.\n"
    md += "\n**Note on overlap exclusions:**\n"
    md += "- P-value overlap: NB-GLM sf=per and lfcShrink versions are excluded because size factor estimation and LFC shrinkage only affect effect sizes, not test statistics or p-values.\n"
    md += "- Effect size overlap: NB-GLM sf=per (without lfcShrink) is excluded because per-comparison size factors only affect LFCs when lfcShrink is applied.\n"
    return md


def _evaluate_benchmarks_legacy(output_dir: Path) -> None:
    """Legacy evaluate benchmark results function - replaced by generate_results.evaluate_benchmarks."""
    # This function is deprecated and retained for reference only.
    # Use the new evaluate_benchmarks from .generate_results module instead.
    from .comparison import compute_de_comparison_metrics
    
    results_dir = output_dir / ".benchmark_cache"
    if not results_dir.exists():
        return
        
    # Load all results
    results = []
    for f in results_dir.glob("*.json"):
        if f.name == "config.json": continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                if "summary" in data:
                    # Flatten summary into main dict
                    summary = data.pop("summary")
                    data.update(summary)
                results.append(data)
        except Exception:
            pass
            
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # Rename max_memory_mb to peak_memory_mb if needed
    if "max_memory_mb" in df.columns and "peak_memory_mb" not in df.columns:
        df = df.rename(columns={"max_memory_mb": "peak_memory_mb"})
    
    # Generate Performance Table with timing breakdown
    # spawn_overhead_seconds captures subprocess startup time
    # de_seconds and convert_seconds are specific to pertpy methods
    perf_cols = [
        "method", "status", "elapsed_seconds", "spawn_overhead_seconds",
        "import_seconds", "load_seconds", "process_seconds", 
        "de_seconds", "convert_seconds", "save_seconds",
        "peak_memory_mb", "avg_memory_mb",
        "cells_kept", "genes_kept", "groups"
    ]
    perf_df = df[[c for c in perf_cols if c in df.columns]].copy()
    if "method" in perf_df.columns:
        perf_df = perf_df.sort_values("method")
    
    # Generate Accuracy Table
    accuracy_results = []
    
    # Generate Performance Comparison Table
    perf_comp_results = []
    
    # Collect DE results for heatmap generation
    de_results_for_heatmaps: Dict[str, pd.DataFrame] = {}
    
    # Define comparisons
    # Format: (method_a, method_b, comparison_type)
    # For de_lfcshrink comparisons, the code will automatically use shrunk_result_path
    # instead of result_path for methods that have integrated shrinkage
    comparisons = [
        # QC
        ("crispyx_qc_filtered", "scanpy_qc_filtered", "qc"),
        # DE GLM - independent vs external tools (base LFCs)
        ("crispyx_de_nb_glm", "edger_de_glm", "de"),
        ("crispyx_de_nb_glm", "pertpy_de_pydeseq2", "de"),
        # DE GLM - independent shrunk vs PyDESeq2 shrunk (uses shrunk_result_path)
        ("crispyx_de_nb_glm", "pertpy_de_pydeseq2", "de_lfcshrink"),
        # DE GLM - external tool comparison
        ("edger_de_glm", "pertpy_de_pydeseq2", "de"),
        # DE Tests
        ("crispyx_de_t_test", "scanpy_de_t_test", "de"),
        ("crispyx_de_wilcoxon", "scanpy_de_wilcoxon", "de"),
    ]
    
    # Add conditional comparisons for optional methods (only if they were run)
    optional_method_results = df[df["method"] == "crispyx_de_nb_glm_pydeseq2"]
    if not optional_method_results.empty and optional_method_results.iloc[0]["status"] in ("success", "recovered", "skipped_existing"):
        # PyDESeq2-parity method vs PyDESeq2 (for parity testing)
        comparisons.extend([
            ("crispyx_de_nb_glm_pydeseq2", "pertpy_de_pydeseq2", "de"),
            ("crispyx_de_nb_glm_pydeseq2", "pertpy_de_pydeseq2", "de_lfcshrink"),
            # Also compare default vs PyDESeq2-parity variants
            ("crispyx_de_nb_glm", "crispyx_de_nb_glm_pydeseq2", "de"),
        ])
    
    # Helper to load DE result DataFrame
    def _load_de_result(method_name: str, result_path_val: str, use_raw_lfc: bool = False) -> Optional[pd.DataFrame]:
        """Load and standardize a DE result from file.
        
        Parameters
        ----------
        method_name : str
            Name of the method (for error messages)
        result_path_val : str
            Relative path to the result file
        use_raw_lfc : bool
            If True and the file is h5ad, use logfoldchange_raw layer instead of
            logfoldchange for effect sizes. This is for fair comparison of raw
            (unshrunken) LFCs between crispyx and PyDESeq2.
        """
        if pd.isna(result_path_val):
            return None
        
        result_path = output_dir / str(result_path_val)
        if not result_path.exists():
            return None
            
        try:
            if str(result_path).endswith('.h5ad'):
                import anndata as ad
                adata = ad.read_h5ad(str(result_path))
                # Use raw LFC extractor if requested
                if use_raw_lfc:
                    result_dict = _anndata_to_de_dict_raw(adata)
                else:
                    result_dict = _anndata_to_de_dict(adata)
                return _streaming_de_to_frame(result_dict)
            else:
                result_df = pd.read_csv(result_path)
                return _standardise_de_dataframe(result_df)
        except Exception as e:
            print(f"Warning: Could not load DE result for {method_name}: {e}")
            return None
    
    # Explicitly collect DE methods (base and shrunk) for heatmaps
    de_methods = [
        "crispyx_de_t_test", "scanpy_de_t_test",
        "crispyx_de_wilcoxon", "scanpy_de_wilcoxon",
        "crispyx_de_nb_glm", "crispyx_de_nb_glm_pydeseq2", "edger_de_glm", "pertpy_de_pydeseq2",
    ]
    methods_with_shrunk = ["crispyx_de_nb_glm", "crispyx_de_nb_glm_pydeseq2", "pertpy_de_pydeseq2"]
    
    for method_name in de_methods:
        method_res = df[df["method"] == method_name]
        if method_res.empty:
            continue
        method_status = method_res.iloc[0]["status"]
        # Include success, recovered, and skipped_existing (cached results with valid output)
        if method_status not in ("success", "recovered", "skipped_existing"):
            continue
        # Load base result
        result_path_val = method_res.iloc[0].get("result_path")
        if method_name not in de_results_for_heatmaps and not pd.isna(result_path_val):
            method_df = _load_de_result(method_name, result_path_val)
            if method_df is not None:
                de_results_for_heatmaps[method_name] = method_df
        # Load shrunk result if available
        if method_name in methods_with_shrunk:
            shrunk_path_val = method_res.iloc[0].get("shrunk_result_path")
            shrunk_method_name = f"{method_name}_shrunk"
            if shrunk_method_name not in de_results_for_heatmaps and not pd.isna(shrunk_path_val):
                # Handle paths that may be relative to project root or to output_dir
                shrunk_path_str = str(shrunk_path_val)
                if shrunk_path_str.startswith("/"):
                    shrunk_path = Path(shrunk_path_str)
                elif shrunk_path_str.startswith("benchmarking/"):
                    # Path from project root
                    shrunk_path = Path(shrunk_path_str)
                else:
                    shrunk_path = output_dir / shrunk_path_str
                if shrunk_path.exists():
                    try:
                        if str(shrunk_path).endswith('.h5ad'):
                            import anndata as ad
                            adata = ad.read_h5ad(str(shrunk_path))
                            result_dict = _anndata_to_de_dict(adata)
                            shrunk_df = _streaming_de_to_frame(result_dict)
                        else:
                            shrunk_df = pd.read_csv(shrunk_path)
                            shrunk_df = _standardise_de_dataframe(shrunk_df)
                        if shrunk_df is not None:
                            de_results_for_heatmaps[shrunk_method_name] = shrunk_df
                    except Exception as e:
                        print(f"Warning: Could not load shrunk DE result for {shrunk_method_name}: {e}")
    
    for method_a_name, method_b_name, comp_type in comparisons:
        # Check if both exist and have valid results (success or skipped_existing with output)
        method_a_res = df[df["method"] == method_a_name]
        method_b_res = df[df["method"] == method_b_name]
        
        if method_a_res.empty or method_b_res.empty:
            continue
        
        valid_statuses = ("success", "recovered", "skipped_existing")
        if method_a_res.iloc[0]["status"] not in valid_statuses or method_b_res.iloc[0]["status"] not in valid_statuses:
            continue
            
        # Performance Comparison
        a_row = method_a_res.iloc[0]
        b_row = method_b_res.iloc[0]
        
        a_time = a_row.get("elapsed_seconds", np.nan)
        b_time = b_row.get("elapsed_seconds", np.nan)
        a_mem = a_row.get("peak_memory_mb", np.nan)
        b_mem = b_row.get("peak_memory_mb", np.nan)
        
        comp = {
            "comparison": f"{method_a_name} vs {method_b_name}",
            "method_a_time_s": a_time,
            "method_b_time_s": b_time,
            "time_diff_s": a_time - b_time if pd.notna(a_time) and pd.notna(b_time) else None,
            "time_pct": (a_time / b_time * 100) if pd.notna(a_time) and pd.notna(b_time) and b_time > 0 else None,
            "method_a_mem_mb": a_mem,
            "method_b_mem_mb": b_mem,
            "mem_diff_mb": a_mem - b_mem if pd.notna(a_mem) and pd.notna(b_mem) else None,
            "mem_pct": (a_mem / b_mem * 100) if pd.notna(a_mem) and pd.notna(b_mem) and b_mem > 0 else None,
        }
        perf_comp_results.append(comp)
        
        # Load result files
        try:
            # For de_lfcshrink comparisons, use shrunk_result_path if available
            if comp_type == "de_lfcshrink":
                method_a_path_val = method_a_res.iloc[0].get("shrunk_result_path", method_a_res.iloc[0]["result_path"])
                method_b_path_val = method_b_res.iloc[0].get("shrunk_result_path", method_b_res.iloc[0]["result_path"])
                # Handle case where shrunk_result_path is NaN - fall back to result_path
                if pd.isna(method_a_path_val):
                    method_a_path_val = method_a_res.iloc[0]["result_path"]
                if pd.isna(method_b_path_val):
                    method_b_path_val = method_b_res.iloc[0]["result_path"]
            else:
                method_a_path_val = method_a_res.iloc[0]["result_path"]
                method_b_path_val = method_b_res.iloc[0]["result_path"]
            
            if pd.isna(method_a_path_val) or pd.isna(method_b_path_val):
                print(f"Skipping comparison {method_a_name} vs {method_b_name}: missing result path")
                continue
                
            method_a_path = output_dir / str(method_a_path_val)
            method_b_path = output_dir / str(method_b_path_val)
            
            if comp_type == "qc":
                # Compare cell/gene counts
                acc = {
                    "comparison": f"{method_a_name} vs {method_b_name}",
                    "cells_diff": float(method_a_res.iloc[0]["cells_kept"] - method_b_res.iloc[0]["cells_kept"]),
                    "genes_diff": float(method_a_res.iloc[0]["genes_kept"] - method_b_res.iloc[0]["genes_kept"]),
                }
                accuracy_results.append(acc)
                
            elif comp_type == "de":
                # Load DE results using helper (also collect for heatmaps)
                method_a_df = _load_de_result(method_a_name, method_a_path_val)
                method_b_df = _load_de_result(method_b_name, method_b_path_val)
                
                # Store for heatmap generation
                if method_a_df is not None and method_a_name not in de_results_for_heatmaps:
                    de_results_for_heatmaps[method_a_name] = method_a_df
                if method_b_df is not None and method_b_name not in de_results_for_heatmaps:
                    de_results_for_heatmaps[method_b_name] = method_b_df
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name}: could not load results")
                    continue
                
                # Compute metrics
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                # Create new dict to avoid type errors
                acc = {"comparison": f"{method_a_name} vs {method_b_name}"}
                acc.update(metrics)
                accuracy_results.append(acc)
            
            elif comp_type == "de_raw":
                # Load DE results with RAW (unshrunken) LFCs for fair comparison
                # This compares crispyx raw LFCs with PyDESeq2 raw LFCs (before shrinkage)
                method_a_df = _load_de_result(method_a_name, method_a_path_val, use_raw_lfc=True)
                method_b_df = _load_de_result(method_b_name, method_b_path_val, use_raw_lfc=True)
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name}: could not load results")
                    continue
                
                # Compute metrics - no suffix needed, raw is the standard comparison
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name}"}
                acc.update(metrics)
                accuracy_results.append(acc)
            
            elif comp_type == "de_lfcshrink":
                # Load DE results comparing shrunken LFCs
                # crispyx uses its shrinkage, PyDESeq2 uses lfcShrink (apeGLM)
                method_a_df = _load_de_result(method_a_name, method_a_path_val, use_raw_lfc=False)
                method_b_df = _load_de_result(method_b_name, method_b_path_val, use_raw_lfc=False)
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name} (lfcShrink): could not load results")
                    continue
                
                # Compute metrics with (lfcShrink) suffix to indicate shrinkage comparison
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name} (lfcShrink)"}
                acc.update(metrics)
                accuracy_results.append(acc)
                
        except Exception as e:
            print(f"Error comparing {method_a_name} vs {method_b_name}: {e}")
    
    # Generate overlap heatmaps
    overlap_heatmaps: Dict[str, Path] = {}
    if de_results_for_heatmaps:
        try:
            from .visualization import generate_overlap_heatmaps
            overlap_heatmaps = generate_overlap_heatmaps(
                de_results_for_heatmaps,
                output_dir,
                k_values=(50, 100, 500),
            )
        except Exception as e:
            print(f"Warning: Could not generate overlap heatmaps: {e}")
            
    # Save tables
    perf_df.to_csv(output_dir / "performance_summary.csv", index=False)
    if accuracy_results:
        acc_df = pd.DataFrame(accuracy_results)
        acc_df.to_csv(output_dir / "accuracy_summary.csv", index=False)
        
    # Generate improved Markdown
    md = _generate_improved_markdown(
        perf_df, 
        perf_comp_results, 
        accuracy_results,
        overlap_heatmaps,
    )
        
    with open(output_dir / "results.md", "w") as f:
        f.write(md)


def _summarise_quality_control(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    total_cells = int(context.get("dataset_cells", 0))
    total_genes = int(context.get("dataset_genes", 0))
    kept_cells = int(getattr(result, "cell_mask").sum())
    kept_genes = int(getattr(result, "gene_mask").sum())
    removed_cells = max(total_cells - kept_cells, 0)
    removed_genes = max(total_genes - kept_genes, 0)
    result_path = _normalise_path(getattr(result, "filtered_path", None), context)
    return {
        "cells_total": total_cells,
        "cells_kept": kept_cells,
        "cells_removed": removed_cells,
        "cells_kept_pct": _percentage(kept_cells, total_cells) if total_cells else None,
        "genes_total": total_genes,
        "genes_kept": kept_genes,
        "genes_removed": removed_genes,
        "genes_kept_pct": _percentage(kept_genes, total_genes) if total_genes else None,
        "result_path": result_path,
    }


def _summarise_dataframe(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise tabular results, including on-disk export paths when available."""

    rows = cols = 0
    if hasattr(result, "shape"):
        try:
            rows, cols = result.shape
        except Exception:  # pragma: no cover - defensive fallback
            rows = cols = 0

    path_candidate: str | Path | None = None
    for attr in ("path", "filename", "result_path"):
        candidate = getattr(result, attr, None)
        if candidate:
            path_candidate = candidate
            break

    summary: Dict[str, Any] = {
        "rows": int(rows),
        "columns": int(cols),
    }

    if path_candidate:
        summary["result_path"] = _normalise_path(path_candidate, context)

    return summary


def _summarise_de_mapping(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    groups: list[str] = []
    if isinstance(result, Mapping):
        groups = list(result.keys())
    elif hasattr(result, "groups"):
        groups = list(getattr(result, "groups"))

    n_groups = len(groups)
    n_genes = 0
    output_path = getattr(result, "result_path", None)
    if groups:
        first_key = groups[0]
        try:
            first = result[first_key]
        except Exception:  # pragma: no cover - defensive
            first = None
        if first is not None:
            genes = getattr(first, "genes", None)
            if genes is not None:
                n_genes = int(len(genes))
            if output_path is None:
                output_path = getattr(first, "result_path", None)
    summary = {
        "groups": n_groups,
        "genes": n_genes,
    }
    if output_path:
        summary["result_path"] = _normalise_path(output_path, context)
    return summary


def _summarise_wilcoxon_csc(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise result from :func:`run_wilcoxon_with_csc`.

    Delegates DE metric extraction to :func:`_summarise_de_mapping` and adds
    ``csc_conversion_seconds``, ``wilcoxon_seconds``, and ``was_already_csc``
    as extra columns so the benchmark report shows the timing breakdown.
    """
    if not isinstance(result, dict) or "de_result" not in result:
        # Fallback for unexpected result shape
        return _summarise_de_mapping(result, context)

    de_result = result["de_result"]
    summary = _summarise_de_mapping(de_result, context)

    # Attach timing breakdown as extra reporting columns
    csc_secs = result.get("csc_conversion_seconds")
    de_secs = result.get("wilcoxon_seconds")
    was_already_csc = result.get("was_already_csc", False)

    if csc_secs is not None:
        summary["csc_conversion_seconds"] = round(csc_secs, 3)
    if de_secs is not None:
        summary["wilcoxon_seconds"] = round(de_secs, 3)
    summary["was_already_csc"] = was_already_csc

    return summary


def _summarise_runner_result(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise result from a standalone runner function."""
    # Handle dictionary result (new format)
    if isinstance(result, dict):
        summary = result.copy()
        if "result_path" in summary:
            summary["result_path"] = _normalise_path(summary["result_path"], context)
        return summary

    # Handle tuple result (old format - if any)
    # Result is tuple: (dataframe, path, error, runtime, peak_mem, avg_mem)
    if isinstance(result, tuple) and len(result) >= 6:
        df, path, error, runtime, peak_mem, avg_mem = result[:6]
        
        summary = {}
        if path:
            summary["result_path"] = _normalise_path(path, context)
            
        if error:
            summary["error"] = error
            
        # Pass through internal metrics
        if runtime is not None:
            summary["internal_runtime_seconds"] = runtime
        if peak_mem is not None:
            summary["internal_peak_memory_mb"] = peak_mem
        if avg_mem is not None:
            summary["internal_avg_memory_mb"] = avg_mem
            
        if df is not None and hasattr(df, "shape"):
            summary["rows"] = df.shape[0]
            summary["columns"] = df.shape[1]
            
            # Extract QC metrics if present in attrs
            if hasattr(df, "attrs"):
                for key in ["cells_kept", "genes_kept", "original_cells", "original_genes"]:
                    if key in df.attrs:
                        summary[key] = df.attrs[key]
                        
        return summary

    return {"error": "Invalid result format"}


def _check_cgroups_available() -> bool:
    """Check if cgroups v2 memory control is available via systemd-run.
    
    Returns True if systemd-run --scope --user with MemoryMax works.
    """
    try:
        result = subprocess.run(
            ["systemd-run", "--scope", "--user", "--property=MemoryMax=1G", "true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# Cache the cgroups availability check
_CGROUPS_AVAILABLE: bool | None = None


def _is_cgroups_available() -> bool:
    """Check cgroups availability with caching."""
    global _CGROUPS_AVAILABLE
    if _CGROUPS_AVAILABLE is None:
        _CGROUPS_AVAILABLE = _check_cgroups_available()
    return _CGROUPS_AVAILABLE


def _uses_mmap_backed_files(method: "BenchmarkMethod") -> bool:
    """Return True if ``method`` is known to use mmap-backed files.

    crispyx DE methods (wilcoxon, t-test, nb_glm) and QC/preprocess open
    large h5ad files via AnnData backing (mmap) and create numpy memmaps
    for result arrays.  Their virtual address space (VSZ) is typically 2–4×
    their actual RSS, making RLIMIT_AS an unreliable enforcement mechanism.
    """
    name = method.name.lower()
    return "crispyx" in name


# ============================================================================
# Docker-based Execution
# ============================================================================

# Cache Docker availability check
_DOCKER_AVAILABLE: bool | None = None


def _check_docker_available() -> bool:
    """Check if Docker is available and the benchmark image exists.
    
    Returns True if docker command works and crispyx-benchmark image exists.
    """
    try:
        # Check if docker command exists
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        
        # Check if docker daemon is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _is_docker_available() -> bool:
    """Check Docker availability with caching."""
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is None:
        _DOCKER_AVAILABLE = _check_docker_available()
    return _DOCKER_AVAILABLE


def _docker_image_exists(image_name: str = "crispyx-benchmark:latest") -> bool:
    """Check if the Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _build_docker_image(
    dockerfile_path: Path | None = None,
    image_name: str = "crispyx-benchmark:latest",
    quiet: bool = False,
) -> bool:
    """Build the Docker benchmark image.
    
    Parameters
    ----------
    dockerfile_path : Path | None
        Path to Dockerfile. If None, uses benchmarking/Dockerfile.
    image_name : str
        Name and tag for the image.
    quiet : bool
        If True, suppress build output.
    
    Returns
    -------
    bool
        True if build succeeded, False otherwise.
    """
    if dockerfile_path is None:
        dockerfile_path = REPO_ROOT / "benchmarking" / "Dockerfile"
    
    context_path = dockerfile_path.parent.parent  # Repository root
    
    cmd = [
        "docker", "build",
        "-t", image_name,
        "-f", str(dockerfile_path),
        str(context_path),
    ]
    
    if not quiet:
        print(f"Building Docker image: {image_name}")
        print(f"  Dockerfile: {dockerfile_path}")
        print(f"  Context: {context_path}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=quiet,
            timeout=1800,  # 30 minutes timeout for build
        )
        if result.returncode == 0:
            if not quiet:
                print(f"  ✓ Image built successfully: {image_name}")
            return True
        else:
            if not quiet:
                print(f"  ✗ Image build failed")
                if result.stderr:
                    print(result.stderr.decode()[-500:])
            return False
    except subprocess.TimeoutExpired:
        if not quiet:
            print("  ✗ Image build timed out (30 minutes)")
        return False
    except Exception as e:
        if not quiet:
            print(f"  ✗ Image build error: {e}")
        return False


class DockerRunner:
    """Execute benchmark methods inside Docker containers with memory limits.
    
    This class provides reliable cross-platform memory limiting by running
    benchmark methods inside Docker containers with --memory limits.
    
    Parameters
    ----------
    image_name : str
        Docker image name and tag.
    memory_limit_gb : float
        Memory limit in gigabytes.
    n_cores : int
        Number of CPU cores to use.
    workspace_root : Path
        Path to the workspace root (for volume mounts).
    quiet : bool
        If True, suppress verbose output.
    """
    
    def __init__(
        self,
        image_name: str = "crispyx-benchmark:latest",
        memory_limit_gb: float = 64.0,
        n_cores: int = 8,
        workspace_root: Path | None = None,
        quiet: bool = False,
    ):
        self.image_name = image_name
        self.memory_limit_gb = memory_limit_gb
        self.n_cores = n_cores
        self.workspace_root = workspace_root or REPO_ROOT
        self.quiet = quiet
        
        # Validate Docker is available
        if not _is_docker_available():
            raise RuntimeError(
                "Docker is not available. Please install Docker or use native execution."
            )
    
    def _get_function_info(self, method: BenchmarkMethod) -> Dict[str, str]:
        """Extract function module and name from a BenchmarkMethod."""
        func = method.function
        module_name = func.__module__
        func_name = func.__name__
        
        # Handle __main__ case - replace with actual module path
        if module_name == "__main__":
            module_name = "benchmarking.tools.run_benchmarks"
        
        return {"function_module": module_name, "function_name": func_name}
    
    def _convert_paths_to_container(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert host paths in kwargs to container paths."""
        result = {}
        for key, value in kwargs.items():
            if isinstance(value, Path):
                # Convert to container path
                try:
                    relative = value.relative_to(self.workspace_root)
                    result[key] = f"/workspace/{relative}"
                except ValueError:
                    # Path not under workspace, use as-is
                    result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def _normalize_container_paths(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert container paths back to host-relative paths.
        
        Paths returned from Docker containers use /workspace/ prefix.
        This method strips the prefix to make paths relative to workspace root.
        
        Parameters
        ----------
        output : Dict[str, Any]
            Output dictionary from Docker worker
            
        Returns
        -------
        Dict[str, Any]
            Output with normalized paths
        """
        result = output.copy()
        
        # Normalize paths in summary dict
        if "summary" in result and isinstance(result["summary"], dict):
            summary = result["summary"].copy()
            for key in ["result_path", "shrunk_result_path"]:
                if key in summary and isinstance(summary[key], str):
                    path = summary[key]
                    # Strip /workspace/ prefix if present
                    if path.startswith("/workspace/"):
                        summary[key] = path[len("/workspace/"):]
            result["summary"] = summary
        
        # Also check top-level result_path and shrunk_result_path
        for key in ["result_path", "shrunk_result_path"]:
            if key in result and isinstance(result[key], str):
                path = result[key]
                if path.startswith("/workspace/"):
                    result[key] = path[len("/workspace/"):]
        
        return result
    
    def run(
        self,
        method: BenchmarkMethod,
        context: Dict[str, Any],
        time_limit: int | None = None,
    ) -> Dict[str, Any]:
        """Run a benchmark method inside a Docker container.
        
        Parameters
        ----------
        method : BenchmarkMethod
            The benchmark method to run.
        context : Dict[str, Any]
            Context dictionary with dataset info.
        time_limit : int | None
            Time limit in seconds (None = no limit).
        
        Returns
        -------
        Dict[str, Any]
            Result dictionary with status, timing, memory, etc.
        """
        import tempfile
        import uuid
        
        # Create temporary directory for IPC
        run_id = str(uuid.uuid4())[:8]
        container_name = f"crispyx_{run_id}"
        temp_dir = Path(tempfile.mkdtemp(prefix=f"crispyx_docker_{run_id}_"))
        
        def _cleanup_container() -> None:
            """Stop and remove the container if it exists."""
            try:
                # Stop the container (with timeout to avoid hanging)
                subprocess.run(
                    ["docker", "stop", "-t", "5", container_name],
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, Exception):
                # Force kill if stop times out
                try:
                    subprocess.run(
                        ["docker", "kill", container_name],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass
            
            # Remove container (in case --rm didn't work due to crash)
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
        
        try:
            # Prepare input configuration
            func_info = self._get_function_info(method)
            container_kwargs = self._convert_paths_to_container(method.kwargs)
            
            input_config = {
                "method_name": method.name,
                "function_module": func_info["function_module"],
                "function_name": func_info["function_name"],
                "kwargs": container_kwargs,
                "context": context,
                "n_threads": self.n_cores,
            }
            
            # Write input JSON
            input_file = temp_dir / "input.json"
            with open(input_file, 'w') as f:
                json.dump(input_config, f, indent=2, default=str)
            
            output_file = temp_dir / "output.json"
            
            # Build docker run command
            memory_bytes = int(self.memory_limit_gb * 1024 * 1024 * 1024)

            # Detect symlinks inside the workspace that point to external
            # filesystems.  Docker's -v bind-mount does NOT follow symlinks,
            # so if e.g. "benchmarking/results" is a symlink to /data/... the
            # container would see a broken symlink at /workspace/benchmarking/results.
            # Fix: for every Path in method.kwargs, walk its components and
            # detect intermediate symlinks whose targets lie outside workspace_root.
            # Add each unique real target as an explicit bind mount (at its own
            # absolute path) so the container can dereference the symlink.
            extra_mounts: list[str] = []
            _seen_mounts: set[str] = set()
            for _kw_value in method.kwargs.values():
                if not isinstance(_kw_value, Path):
                    continue
                # Case 1: absolute path entirely outside workspace (e.g. /data1/origin/).
                # Mount it read-only at the same absolute path so the container can
                # resolve it without any path translation.
                try:
                    _kw_value.relative_to(self.workspace_root)
                except ValueError:
                    _real = _kw_value.resolve()
                    # Mount the closest existing ancestor (the file may not exist yet)
                    _mount_src = _real
                    while not _mount_src.exists() and _mount_src.parent != _mount_src:
                        _mount_src = _mount_src.parent
                    if _mount_src.exists():
                        _spec = f"{_mount_src}:{_mount_src}:ro"
                        if _spec not in _seen_mounts:
                            extra_mounts.extend(["-v", _spec])
                            _seen_mounts.add(_spec)
                    continue  # no symlink walk needed for external paths
                # Case 2: path inside workspace — detect symlinks that point outside.
                _current = Path("/")
                for _part in _kw_value.parts[1:]:
                    _current = _current / _part
                    try:
                        _current.relative_to(self.workspace_root)
                    except ValueError:
                        continue  # still building up to workspace_root, keep walking
                    if _current.is_symlink():
                        _real = _current.resolve()
                        try:
                            _real.relative_to(self.workspace_root)
                        except ValueError:
                            # Symlink target is outside workspace → expose it
                            _spec = f"{_real}:{_real}:rw"
                            if _spec not in _seen_mounts:
                                extra_mounts.extend(["-v", _spec])
                                _seen_mounts.add(_spec)
                        break  # no need to go deeper once symlink found

            cmd = [
                "docker", "run", "--rm",
                f"--name={container_name}",  # Named container for cleanup
                f"--memory={memory_bytes}",
                "--memory-swap=-1",  # Disable swap to enforce memory limit
                f"--cpus={self.n_cores}",
                # Mount workspace
                "-v", f"{self.workspace_root}:/workspace:rw",
                # Extra mounts for symlinked directories inside the workspace
                *extra_mounts,
                # Mount temp directory for IPC
                "-v", f"{temp_dir}:/ipc:rw",
                # Mount host timezone so container timestamps match host time
                "-v", "/etc/localtime:/etc/localtime:ro",
                # Environment variables
                "-e", f"NUMBA_NUM_THREADS={self.n_cores}",
                "-e", f"OMP_NUM_THREADS={self.n_cores}",
                "-e", f"MKL_NUM_THREADS={self.n_cores}",
                # Working directory
                "-w", "/workspace",
                # Image
                self.image_name,
                # Worker arguments
                "--input", "/ipc/input.json",
                "--output", "/ipc/output.json",
            ]
            
            if not self.quiet:
                print(f"  🐳 Running {method.name} in Docker container...")
            
            # Run container
            start_time = time.perf_counter()
            
            try:
                timeout = time_limit + 60 if time_limit else None  # Add buffer for container startup
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - start_time
                
                # Read output
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        output = json.load(f)
                    
                    # Normalize container paths to host-relative paths
                    output = self._normalize_container_paths(output)
                    
                    # Add elapsed time from parent (includes container overhead)
                    output["elapsed_seconds"] = elapsed
                    output["container_overhead_seconds"] = elapsed - (output.get("elapsed_seconds", elapsed) or elapsed)
                    
                    return output
                else:
                    # Container failed before writing output
                    error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                    if result.returncode == 137:
                        return {
                            "method": method.name,
                            "status": "memory_limit",
                            "elapsed_seconds": elapsed,
                            "error": f"Container killed (OOM): {error_msg[-500:]}",
                        }
                    else:
                        return {
                            "method": method.name,
                            "status": "error",
                            "elapsed_seconds": elapsed,
                            "error": f"Container failed (exit {result.returncode}): {error_msg[-500:]}",
                        }
                        
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start_time
                # Kill and cleanup container
                _cleanup_container()
                return {
                    "method": method.name,
                    "status": "timeout",
                    "elapsed_seconds": elapsed,
                    "error": f"Container timed out after {time_limit}s",
                }
            except Exception as e:
                # Handle any other errors (including keyboard interrupt)
                elapsed = time.perf_counter() - start_time
                _cleanup_container()
                return {
                    "method": method.name,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error": f"Unexpected error: {str(e)[-500:]}",
                }
                
        finally:
            # Always attempt container cleanup (in case of OOM or other failures)
            _cleanup_container()
            
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def cleanup_orphaned_containers(self) -> int:
        """Clean up any orphaned crispyx containers from previous runs.
        
        This is useful to call at the start of a benchmark session to ensure
        no stale containers are consuming memory.
        
        Returns
        -------
        int
            Number of containers cleaned up.
        """
        try:
            # Find all containers with crispyx_ prefix
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=crispyx_", "--format", "{{.Names}}"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                return 0
            
            container_names = result.stdout.decode().strip().split('\n')
            container_names = [n for n in container_names if n]  # Filter empty strings
            
            cleaned = 0
            for name in container_names:
                try:
                    # Force remove each container
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True,
                        timeout=10,
                    )
                    cleaned += 1
                    if not self.quiet:
                        print(f"  🧹 Cleaned up orphaned container: {name}")
                except Exception:
                    pass
            
            return cleaned
        except Exception:
            return 0
    
    def ensure_image(self) -> bool:
        """Ensure the Docker image exists, building if necessary.
        
        Returns
        -------
        bool
            True if image is available, False otherwise.
        """
        if _docker_image_exists(self.image_name):
            if not self.quiet:
                print(f"  ✓ Docker image available: {self.image_name}")
            return True
        
        if not self.quiet:
            print(f"  ⚠ Docker image not found: {self.image_name}")
            print("  Building image (this may take several minutes)...")
        
        return _build_docker_image(quiet=self.quiet)


def _worker(
    queue: mp.Queue,
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
    n_threads: int = 1,
    use_cgroups_memory: bool = False,
) -> None:
    """Execute ``method`` with optional resource limits and report the outcome.
    
    Memory Baseline Strategy
    ------------------------
    Captures baseline memory before method execution to ensure fair comparison
    with reference methods. Both peak and average memory are reported as deltas
    from this baseline.
    
    For streaming methods:
    - QC methods: Baseline captures environment + minimal data loading overhead
    - DE methods (t-test, wilcoxon): Baseline includes QC-filtered normalized data in memory
    - nb_glm: Baseline includes QC-filtered count data (not normalized)
    
    This matches reference method baselines which capture after _prepare_reference_anndata
    loads filtered datasets.
    
    Parameters
    ----------
    n_threads : int
        Number of threads to use for BLAS/OpenMP operations. Defaults to 1.
    """

    def _apply_resource_limit(limit_value: int | None, limit_type: int, label: str) -> None:
        """Apply ``resource.setrlimit`` defensively.

        Some platforms (notably macOS) raise ``ValueError`` when attempting to
        reduce both the soft and hard limits in a single call, reporting that
        the current limit exceeds the maximum. To avoid crashing the worker
        process, we first lower the soft limit while keeping the existing hard
        cap, then attempt to lower the hard limit. If the OS still rejects the
        change we fall back to best-effort limits and continue execution.
        """

        if not limit_value or limit_value <= 0:
            return

        desired = int(limit_value)

        try:
            current_soft, current_hard = resource.getrlimit(limit_type)

            # First, clamp the soft limit to the current hard ceiling to avoid
            # ``ValueError: current limit exceeds maximum limit``.
            soft_limit = min(
                desired,
                current_hard if current_hard != resource.RLIM_INFINITY else desired,
            )
            if current_soft != soft_limit:
                resource.setrlimit(limit_type, (soft_limit, current_hard))

            # If possible, also lower the hard limit so we get consistent
            # enforcement even if the OS ignores the soft cap.
            if current_hard == resource.RLIM_INFINITY or desired < current_hard:
                try:
                    resource.setrlimit(limit_type, (soft_limit, soft_limit))
                except ValueError:
                    print(
                        f"Warning: unable to reduce {label} hard limit; "
                        "soft limit applied only.",
                        file=sys.stderr,
                    )
        except (ValueError, OSError) as exc:
            print(
                f"Warning: could not apply {label} limit ({exc}); proceeding without it.",
                file=sys.stderr,
            )

    # Set thread limits for BLAS/OpenMP to control parallelism
    set_thread_env_vars(n_threads)

    # Apply resource limits
    # RLIMIT_AS limits *virtual address space* (VSZ), NOT physical RSS.
    # crispyx methods use mmap-backed h5ad files (10–40 GB mapped) plus
    # numpy memmaps for result arrays, Numba thread stacks, etc.  These
    # inflate VSZ far beyond actual RAM consumption, causing spurious
    # SIGKILL even when RSS is well within the budget.  For example,
    # Feng-gwsf runs at ~6 GB RSS but 42 GB VSZ.
    #
    # Skip RLIMIT_AS when:
    # (a) cgroups memory control is being used externally, or
    # (b) the method is a crispyx method that uses memmaps / backed files.
    #
    # Memory enforcement still works via parent-process RSS sampling
    # (sample_subprocess_memory) which tracks actual physical memory.
    skip_rlimit_as = use_cgroups_memory or _uses_mmap_backed_files(method)
    if not skip_rlimit_as:
        _apply_resource_limit(memory_limit, resource.RLIMIT_AS, "virtual memory")

    # RLIMIT_CPU is NOT used for any method.
    #
    # RLIMIT_CPU counts CPU time summed across ALL pthreads in the process.
    # Even "single-threaded" methods (scanpy, pertpy) use multi-threaded BLAS
    # (OpenBLAS / MKL) and may use n_cores threads. With 32 threads, the
    # effective wall-clock limit becomes time_limit / n_threads ≈ 675 s —
    # far below the intended 6-hour per-method budget.
    #
    # Wall-clock enforcement is already handled correctly by the parent process
    # via process.join(timeout=join_timeout) followed by process.terminate().
    # RLIMIT_CPU is therefore both redundant and incorrect as a wall-clock proxy.

    # Note: Numba JIT compilation is cached to disk (cache=True in decorators).
    # First run after code changes may be slower, but subsequent runs use cache.
    # No explicit warm-up needed - the cache handles this automatically.

    # Use MemoryTracker for memory measurement
    tracker = MemoryTracker(sample_interval=0.1)
    tracker.start()
    
    start = time.perf_counter()
    try:
        result = method.function(**method.kwargs)
        elapsed = time.perf_counter() - start
        
        # Stop memory tracking
        tracker.stop()
        
        # Use absolute peak memory (total RSS, not delta)
        peak_memory_mb = tracker.get_peak_absolute_mb()
        
        # Calculate average memory from samples (absolute values)
        avg_memory_mb = tracker.get_average_absolute_mb()
        
        summary = method.summary(result, context)
        queue.put(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": peak_memory_mb,
                "avg_memory_mb": avg_memory_mb,
                "summary": summary,
            }
        )
    except MemoryError as exc:
        # Stop memory tracking
        try:
            tracker.stop()
        except RuntimeError:
            pass  # Already stopped or never started
        
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "memory_limit",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
                "error": f"MemoryError: {exc}",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        # Stop memory tracking
        try:
            tracker.stop()
        except RuntimeError:
            pass  # Already stopped or never started
        
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "error",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
                "error": f"{exc}",
                "traceback": traceback.format_exc(),
            }
        )
    
    # Force immediate exit to avoid hanging on orphaned ThreadPoolExecutor threads.
    # Some libraries (e.g., PyDESeq2 via pydeseq2/anndata) create ThreadPoolExecutors
    # that don't shut down cleanly. Python's normal exit waits for all non-daemon
    # threads, causing 600+ second delays. os._exit() bypasses this.
    # 
    # We need a small delay to allow the multiprocessing Queue's background thread
    # to send the result data to the parent process before we force-exit.
    import time as _time
    _time.sleep(1.0)  # Allow queue to flush (increased from 0.5s for large results)
    
    import os as _os
    _os._exit(0)


def _run_with_limits(
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
    use_docker: bool = False,
    docker_runner: Optional[DockerRunner] = None,
) -> Dict[str, Any]:
    # If Docker mode is enabled, use DockerRunner
    if use_docker and docker_runner is not None:
        return docker_runner.run(method, context, time_limit)
    
    # Extract n_jobs/n_cores from method kwargs to set thread limits
    n_threads = method.kwargs.get('n_jobs') or method.kwargs.get('n_cores') or 1
    if n_threads is None or n_threads <= 0:
        n_threads = 1
    
    # Check if cgroups memory control is available
    # Use cgroups for crispyx methods that may use memmaps (joint NB-GLM)
    # to avoid RLIMIT_AS issues with virtual address space limits
    use_cgroups = _is_cgroups_available() and 'crispyx' in method.name.lower()
    

    # Use spawn context for R/rpy2 compatibility, to avoid OpenMP fork issues
    # triggered by some scanpy workflows, AND for crispyx methods that use Numba
    # (forking inherits the parent's already-initialized Numba threads).
    # This includes nb_glm, wilcoxon (uses _wilcoxon_sparse_batch_numba), and t_test.
    needs_spawn = (
        'pertpy' in method.name.lower() 
        or 'scanpy' in method.name.lower()
        or 'crispyx_de' in method.name.lower()  # All crispyx DE methods use Numba kernels
        or 'edger' in method.name.lower()  # R/rpy2 needs spawn context
    )
    mp_context = mp.get_context('spawn') if needs_spawn else mp
    
    # Track wall-clock time at parent process level for accuracy
    parent_start_time = time.perf_counter()
    
    queue = mp_context.Queue()
    process = mp_context.Process(
        target=_worker,
        args=(queue, method, context, memory_limit, time_limit, n_threads, use_cgroups),
        name=f"benchmark-{method.name}",
    )
    process.start()
    
    # For spawned processes, track memory from parent using psutil
    # This avoids the issue where spawn resets peak RSS counter
    parent_memory_samples = []
    parent_stop_event = threading.Event()
    parent_memory_thread = None
    if needs_spawn and process.pid:
        parent_memory_thread = threading.Thread(
            target=sample_subprocess_memory,
            args=(process.pid, parent_stop_event, parent_memory_samples, 0.1),
            daemon=True
        )
        parent_memory_thread.start()
    
    join_timeout = None
    if time_limit and time_limit > 0:
        join_timeout = time_limit + 5
    process.join(timeout=join_timeout)
    
    # Stop parent-process memory sampling
    parent_stop_event.set()
    if parent_memory_thread:
        parent_memory_thread.join(timeout=1.0)
    
    # Calculate actual wall-clock elapsed time from parent process
    parent_elapsed_time = time.perf_counter() - parent_start_time
    
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "status": "timeout",
            "elapsed_seconds": parent_elapsed_time,
            "peak_memory_mb": None,
            "avg_memory_mb": None,
            "summary": {},
            "error": f"Exceeded time limit of {time_limit} seconds",
        }

    # Check process exit code and read result from queue
    # Handle EOFError which occurs when subprocess crashes during queue.put()
    payload = None
    if process.exitcode != 0:
        # Process failed - try to get any partial result but expect failure
        try:
            if not queue.empty():
                payload = queue.get(timeout=2.0)
        except (EOFError, Exception):
            pass  # Expected - subprocess crashed
        
        if payload is None:
            payload = {
                "status": "error",
                "elapsed_seconds": parent_elapsed_time,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
                "error": f"Worker process crashed (exitcode={process.exitcode})",
            }
    else:
        # Process succeeded - read from queue
        try:
            if not queue.empty():
                payload = queue.get(timeout=5.0)
            else:
                raise queue.Empty()
        except queue.Empty:
            payload = {
                "status": "error",
                "elapsed_seconds": parent_elapsed_time,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
                "error": f"Process exited normally but no result in queue",
            }
        except EOFError:
            # Queue data was corrupted - process may have died during transmission
            payload = {
                "status": "error",
                "elapsed_seconds": parent_elapsed_time,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
                "error": f"Result transmission failed (corrupted queue data)",
            }

    payload.setdefault("summary", {})
    
    # For spawned processes, ALWAYS use parent-tracked memory (absolute values)
    # because subprocess getrusage peak RSS resets on spawn, making subprocess
    # delta-based values unreliable. Parent-tracked values are absolute (total RSS).
    if needs_spawn and parent_memory_samples:
        parent_peak_mb = max(parent_memory_samples) / (1024.0 * 1024.0)
        parent_avg_mb = np.mean(parent_memory_samples) / (1024.0 * 1024.0)
        
        # Always use parent-tracked values for spawned processes
        payload["peak_memory_mb"] = parent_peak_mb
        payload["avg_memory_mb"] = parent_avg_mb
    
    # Ensure memory fields are present
    payload.setdefault("peak_memory_mb", None)
    payload.setdefault("avg_memory_mb", None)
    
    # For spawned processes, calculate spawn overhead (subprocess startup time)
    # This is the time between parent starting the process and subprocess beginning work
    if needs_spawn:
        subprocess_elapsed = payload.get("elapsed_seconds", 0) or 0
        spawn_overhead = parent_elapsed_time - subprocess_elapsed
        payload["spawn_overhead_seconds"] = max(0, spawn_overhead)
    
    # Override elapsed_seconds with parent process timing for total wall time
    payload["elapsed_seconds"] = parent_elapsed_time
    
    return payload


def _load_dataset_context(path: Path) -> Dict[str, Any]:
    backed = read_backed(path)
    try:
        context = {
            "dataset_cells": backed.n_obs,
            "dataset_genes": backed.n_vars,
        }
    finally:
        backed.file.close()
    return context


def create_benchmark_suite(
    dataset_path: Path,
    output_dir: Path,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
    gene_name_column: str | None = "gene_symbols",
    qc_params: QCParams | None = None,
    n_cores: int | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
) -> Dict[str, BenchmarkMethod]:
    """Return the available benchmark methods for the provided dataset.
    
    Parameters
    ----------
    dataset_path
        Path to standardized dataset.
    output_dir
        Directory for benchmark outputs.
    perturbation_column
        Name of perturbation column (should be 'perturbation' after standardization).
    control_label
        Control label (should be 'control' after standardization or None for auto-detect).
    gene_name_column
        Gene name column or None to use var.index.
    qc_params
        QC parameters. If None, will be calculated adaptively.
    n_cores
        Number of cores to use for parallel DE methods. If None, auto-detects.
    chunk_size
        Optional fixed chunk size to use. Overrides qc_params.chunk_size if provided.
    memory_limit_gb
        Optional memory limit in GB for memory-adaptive methods like NB-GLM.
        If provided, methods will use streaming mode when matrix size exceeds
        a fraction of this limit.
    """
    import anndata as ad
    
    # Use provided QC params or calculate adaptively
    if qc_params is None:
        # Calculate adaptive QC parameters
        adata_temp = ad.read_h5ad(dataset_path, backed='r')
        try:
            adaptive_thresholds = calculate_adaptive_qc_thresholds(
                adata_temp, perturbation_column, mode='conservative', chunk_size=chunk_size
            )
            min_genes = adaptive_thresholds['min_genes']
            min_cells_per_perturbation = adaptive_thresholds['min_cells_per_perturbation']
            min_cells_per_gene = adaptive_thresholds['min_cells_per_gene']
            final_chunk_size = adaptive_thresholds['chunk_size']
        finally:
            adata_temp.file.close()
    else:
        min_genes = qc_params.min_genes
        min_cells_per_perturbation = qc_params.min_cells_per_perturbation
        min_cells_per_gene = qc_params.min_cells_per_gene
        final_chunk_size = chunk_size if chunk_size is not None else qc_params.chunk_size

    backed = read_backed(dataset_path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        detected_control = resolve_control_label(labels, control_label, verbose=False)
    finally:
        backed.file.close()

    shared_kwargs = {
        "perturbation_column": perturbation_column,
        "control_label": detected_control,
        "gene_name_column": gene_name_column,
    }

    # Create phase-based subdirectories for streaming outputs
    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    de_dir = output_dir / "de"
    de_dir.mkdir(parents=True, exist_ok=True)

    # preprocessed_path: QC-filtered then normalized — created by crispyx_preprocess
    # which depends on crispyx_qc_filtered. All t-test and Wilcoxon DE methods use it.
    qc_filtered_path = preprocessing_dir / "crispyx_qc_filtered.h5ad"
    preprocessed_path = preprocessing_dir / f"preprocessed_{dataset_path.name}"
    # preprocessed_csc_path: CSC conversion of preprocessed_path — created by
    # crispyx_preprocess_csc.  Used by crispyx_de_wilcoxon for fast column access.
    preprocessed_csc_path = preprocessing_dir / f"preprocessed_csc_{dataset_path.name}"
    # standardized_csr_path: CSR conversion of dataset_path — created by
    # crispyx_standardize_csr.  Used by crispyx_de_nb_glm for fast row access.
    # If the input is already CSR (e.g. from early conversion in
    # _run_single_benchmark), point directly at it.
    from crispyx.data import get_matrix_storage_format as _get_fmt
    if _get_fmt(dataset_path) == "csr":
        standardized_csr_path = dataset_path
    else:
        standardized_csr_path = preprocessing_dir / f"standardized_csr_{dataset_path.name}"

    methods = {
        "crispyx_qc_filtered": BenchmarkMethod(
            name="crispyx_qc_filtered",
            description="Streaming quality control filters",
            function=quality_control_summary,
            kwargs={
                "data": dataset_path,
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "chunk_size": final_chunk_size,
                "memory_limit_gb": memory_limit_gb,
                **shared_kwargs,
                "output_dir": preprocessing_dir,
                "output_path": preprocessing_dir / "crispyx_qc_filtered.h5ad",
            },
            summary=_summarise_quality_control,
        ),
        "crispyx_preprocess": BenchmarkMethod(
            name="crispyx_preprocess",
            description="Normalize QC-filtered data (total-count + log1p)",
            function=run_preprocess,
            kwargs={
                "input_path": qc_filtered_path,
                "output_path": preprocessed_path,
                "chunk_size": final_chunk_size,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_qc_filtered",
        ),
        "crispyx_preprocess_csc": BenchmarkMethod(
            name="crispyx_preprocess_csc",
            description="Convert preprocessed CSR to CSC for fast Wilcoxon column access",
            function=run_preprocess_csc,
            kwargs={
                "input_path": preprocessed_path,
                "output_path": preprocessed_csc_path,
                "chunk_size": final_chunk_size,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_preprocess",
        ),
        "crispyx_standardize_csr": BenchmarkMethod(
            name="crispyx_standardize_csr",
            description="Convert standardized dataset to CSR for fast NB-GLM row access",
            function=run_preprocess_csr,
            kwargs={
                "input_path": dataset_path,
                "output_path": standardized_csr_path,
                "chunk_size": final_chunk_size,
            },
            summary=_summarise_runner_result,
        ),
        "crispyx_pb_avg_log": BenchmarkMethod(
            name="crispyx_pb_avg_log",
            description="Average log-normalised expression per perturbation",
            function=compute_average_log_expression,
            kwargs={
                "data": dataset_path,
                **shared_kwargs,
                "output_dir": preprocessing_dir,
                "output_path": preprocessing_dir / "crispyx_pb_avg_log.h5ad",
            },
            summary=_summarise_dataframe,
        ),
        "crispyx_pb_pseudobulk": BenchmarkMethod(
            name="crispyx_pb_pseudobulk",
            description="Pseudo-bulk log fold-change per perturbation",
            function=compute_pseudobulk_expression,
            kwargs={
                "data": dataset_path,
                **shared_kwargs,
                "output_dir": preprocessing_dir,
                "output_path": preprocessing_dir / "crispyx_pb_pseudobulk.h5ad",
            },
            summary=_summarise_dataframe,
        ),
        "crispyx_de_t_test": BenchmarkMethod(
            name="crispyx_de_t_test",
            description="t-test differential expression test",
            function=t_test,
            kwargs={
                "data": preprocessed_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "output_path": de_dir / "crispyx_de_t_test.h5ad",
                "n_jobs": n_cores,
                "memory_limit_gb": memory_limit_gb,
            },
            summary=_summarise_de_mapping,
            depends_on="crispyx_preprocess",
        ),
        "crispyx_de_wilcoxon": BenchmarkMethod(
            name="crispyx_de_wilcoxon",
            description="Wilcoxon DE with CSR\u2192CSC conversion (conversion + DE timed together)",
            function=run_wilcoxon_with_csc,
            kwargs={
                "input_path": preprocessed_path,
                "csc_path": preprocessed_csc_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "output_path": de_dir / "crispyx_de_wilcoxon.h5ad",
                "n_jobs": n_cores,
                "memory_limit_gb": memory_limit_gb,
                # csc_chunk_size: cell-row chunk size for CSC conversion (from QC)
                # wilcoxon gene_chunk_size is calculated independently by wilcoxon_test
                "csc_chunk_size": final_chunk_size,
            },
            summary=_summarise_wilcoxon_csc,
            depends_on="crispyx_preprocess",
        ),
        "crispyx_de_nb_glm": BenchmarkMethod(
            name="crispyx_de_nb_glm",
            description="NB-GLM base fitting (no shrinkage)",
            function=run_nb_glm_base,
            kwargs={
                "path": standardized_csr_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "n_jobs": n_cores,
                "size_factor_method": "deseq2",
                "scale_size_factors": False,
                "use_control_cache": True,
                "size_factor_scope": "global",
                "output_path": de_dir / "crispyx_de_nb_glm.h5ad",
                "memory_limit_gb": memory_limit_gb,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_standardize_csr",
        ),
        # NB-GLM with PyDESeq2-parity configuration (for benchmarking parity)
        "crispyx_de_nb_glm_pydeseq2": BenchmarkMethod(
            name="crispyx_de_nb_glm_pydeseq2",
            description="NB-GLM with PyDESeq2-parity settings (fisher SE, per-comparison SF/disp)",
            function=run_nb_glm_base,
            kwargs={
                "path": standardized_csr_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "n_jobs": n_cores,
                "size_factor_method": "deseq2",
                "scale_size_factors": False,
                "use_control_cache": False,  # per-comparison requires fresh computation
                "size_factor_scope": "per_comparison",
                "se_method": "fisher",
                "dispersion_scope": "per_comparison",
                "output_path": de_dir / "crispyx_de_nb_glm_pydeseq2_nb_glm.h5ad",
                "memory_limit_gb": memory_limit_gb,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_standardize_csr",
        ),
        # lfcShrink with global prior scale (faster)
        "crispyx_de_lfcshrink": BenchmarkMethod(
            name="crispyx_de_lfcshrink",
            description="apeGLM LFC shrinkage (global prior scale)",
            function=run_lfcshrink,
            kwargs={
                "base_result_path": de_dir / "crispyx_de_nb_glm.h5ad",
                "output_dir": de_dir,
                "data_name": "crispyx_de_nb_glm_shrunk",
                "prior_scale_mode": "global",
                "memory_limit_gb": memory_limit_gb,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_de_nb_glm",
        ),
        # lfcShrink with PyDESeq2-parity configuration (uses PyDESeq2-parity NB-GLM base)
        "crispyx_de_lfcshrink_pydeseq2": BenchmarkMethod(
            name="crispyx_de_lfcshrink_pydeseq2",
            description="apeGLM LFC shrinkage with PyDESeq2-parity base (per-comparison prior scale)",
            function=run_lfcshrink,
            kwargs={
                "base_result_path": de_dir / "crispyx_de_nb_glm_pydeseq2_nb_glm.h5ad",
                "output_dir": de_dir,
                "data_name": "crispyx_de_nb_glm_shrunk_pydeseq2",
                "prior_scale_mode": "per_comparison",
                "memory_limit_gb": memory_limit_gb,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_de_nb_glm_pydeseq2",
        ),
        "scanpy_qc_filtered": BenchmarkMethod(
            name="scanpy_qc_filtered",
            description="Quality control using Scanpy",
            function=run_scanpy_qc,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": preprocessing_dir,
            },
            summary=_summarise_runner_result,
        ),
        "scanpy_de_t_test": BenchmarkMethod(
            name="scanpy_de_t_test",
            description="Differential expression using Scanpy (t-test)",
            function=run_scanpy_de,
            kwargs={
                "dataset_path": preprocessed_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "method": "t-test",
                "output_dir": de_dir,
                "preprocess": False,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_preprocess",
        ),
        "scanpy_de_wilcoxon": BenchmarkMethod(
            name="scanpy_de_wilcoxon",
            description="Differential expression using Scanpy (Wilcoxon)",
            function=run_scanpy_de,
            kwargs={
                "dataset_path": preprocessed_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "method": "wilcoxon",
                "output_dir": de_dir,
                "preprocess": False,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_preprocess",
        ),
        "edger_de_glm": BenchmarkMethod(
            name="edger_de_glm",
            description="Differential expression using edgeR (GLM)",
            function=run_edger_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "output_dir": de_dir,
                "n_jobs": n_cores,
            },
            summary=_summarise_runner_result,
        ),
        "pertpy_de_pydeseq2": BenchmarkMethod(
            name="pertpy_de_pydeseq2",
            description="PyDESeq2 base fitting (no shrinkage)",
            function=run_pydeseq2_base,
            kwargs={
                "dataset_path": qc_filtered_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "output_dir": de_dir,
                "n_jobs": n_cores,
            },
            summary=_summarise_runner_result,
            depends_on="crispyx_qc_filtered",
        ),
        "pertpy_de_lfcshrink": BenchmarkMethod(
            name="pertpy_de_lfcshrink",
            description="PyDESeq2 apeGLM LFC shrinkage",
            function=run_pydeseq2_lfcshrink,
            kwargs={
                "dataset_path": qc_filtered_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "output_dir": de_dir,
                "n_jobs": n_cores,
            },
            summary=_summarise_runner_result,
            depends_on="pertpy_de_pydeseq2",
        ),
    }

    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streamlined CRISPR analysis methods")
    default_output = Path(__file__).resolve().parent / "results"
    
    # Config file option (new)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file (overrides individual arguments)",
    )
    
    # Single dataset arguments (backward compatible)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "demo_benchmark.h5ad",
        help="Path to an AnnData .h5ad file to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to store benchmark outputs and summaries",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of methods to run. Defaults to all available methods.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Maximum number of CPU seconds allowed per method (0 disables the limit)",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=4.0,
        help="Maximum memory per method in gigabytes (0 disables the limit)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for streaming operations (overrides config and adaptive calculation)",
    )
    
    # Environment configuration arguments
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel operations (overrides config file)",
    )
    parser.add_argument(
        "--r-home",
        type=str,
        default=None,
        help="Path to R installation directory (overrides config and auto-detection)",
    )
    
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help=(
            "Generate the synthetic demo dataset at --data-path before running the benchmarks. "
            "This is useful when bootstrapping a fresh checkout."
        ),
    )
    parser.add_argument(
        "--demo-seed",
        type=int,
        default=0,
        help=(
            "Random seed used when generating the demo dataset (set to -1 to sample a random seed)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and non-essential output",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Run benchmarks inside Docker containers for reliable memory limiting and reproducibility",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="crispyx-benchmark:latest",
        help="Docker image name for benchmark execution (default: crispyx-benchmark:latest)",
    )
    parser.add_argument(
        "--build-docker",
        action="store_true",
        help="Build the Docker image before running benchmarks (requires --use-docker)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-run methods. When used with --methods, only clears cache for specified methods (preserving others for reports). Otherwise clears entire cache.",
    )
    parser.add_argument(
        "--clear-results", "--clean",
        action="store_true",
        dest="clear_results",
        help="Clear the entire output directory before running benchmarks",
    )
    parser.add_argument(
        "--regenerate-report",
        action="store_true",
        help="Skip benchmark execution and only regenerate reports from existing cache",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip methods with existing output files (default: True). Use --no-skip-existing to force re-run of specified methods.",
    )
    return parser.parse_args()


def _format_summary_markdown(summary: Dict[str, Any]) -> str:
    """Return a narrative Markdown summary for ``summary`` statistics."""

    lines: list[str] = ["## Benchmark summary", ""]

    total_methods = summary.get("total_methods", 0)
    success_count = summary.get("success_count", 0)
    recovered_count = summary.get("recovered_count", 0)
    timeout_count = summary.get("timeout_count", 0)
    memory_limit_count = summary.get("memory_limit_count", 0)
    error_count = summary.get("error_count", 0)
    non_success = summary.get("non_success_count", 0)
    success_rate = summary.get("success_rate")
    average_runtime = summary.get("average_runtime_seconds")

    totals_line = f"- **Methods executed:** {total_methods}"
    lines.append(totals_line)

    success_line = f"- **Succeeded:** {success_count}"
    if success_rate is not None:
        success_line += f" ({success_rate * 100:.1f}% success rate)"
    lines.append(success_line)

    # Note any recovered methods (completed despite initial timeout/error)
    if recovered_count:
        lines.append(f"  - Recovered (completed after timeout): {recovered_count}")

    if non_success:
        lines.append(f"- **Did not succeed:** {non_success}")
    if timeout_count:
        lines.append(f"  - Timeouts: {timeout_count}")
    if memory_limit_count:
        lines.append(f"  - Memory limit exceeded: {memory_limit_count}")
    if error_count:
        lines.append(f"  - Errors: {error_count}")

    if average_runtime is not None:
        lines.append(f"- **Average runtime:** {average_runtime:.3f}s")

    # Remove category-based runtime summary since categories no longer exist

    dependency_errors = summary.get("dependency_errors", [])
    other_errors = summary.get("other_errors", [])
    if dependency_errors or other_errors:
        lines.append("- **Notable issues:**")
        if dependency_errors:
            lines.append("  - Dependency errors detected:")
            for message in dependency_errors:
                lines.append(f"    - {message}")
        if other_errors:
            lines.append("  - Other errors recorded:")
            for message in other_errors:
                lines.append(f"    - {message}")
    else:
        lines.append("- **Notable issues:** None")

    return "\n".join(lines).strip() + "\n"


def _frame_to_markdown_table(table: pd.DataFrame) -> str:
    """Render ``table`` as a Markdown table suitable for GitHub."""

    if table.empty:
        return "| |\n|---|\n"

    formatted = table.copy()
    numeric_cols = formatted.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        formatted[numeric_cols] = formatted[numeric_cols].round(3)

    headers = list(formatted.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = []
    for _, row in formatted.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"


def _extract_task_and_test_type(method_name: str) -> tuple[str, str]:
    """Extract task type and test type from method name.
    
    Returns
    -------
    tuple[str, str]
        (task, test_type) where task is 'preprocessing' or 'differential_expression'
        and test_type is like 'qc', 't_test', 'wilcoxon', 'nb_glm'
    """
    # Handle NaN or non-string values
    if not isinstance(method_name, str):
        return ("other", "unknown")
    
    if "_qc_" in method_name or method_name.endswith("_qc_filtered"):
        return ("preprocessing", "qc")
    elif "_pb_avg" in method_name:
        return ("preprocessing", "pb_avg")
    elif "_pb_pseudobulk" in method_name:
        return ("preprocessing", "pb_pseudobulk")
    elif "_de_t_test" in method_name:
        return ("differential_expression", "t_test")
    elif "_de_wilcoxon" in method_name:
        return ("differential_expression", "wilcoxon")
    elif "_de_nb_glm" in method_name or "_de_glm" in method_name:
        return ("differential_expression", "nb_glm")
    elif "_de_pydeseq2" in method_name:
        return ("differential_expression", "pydeseq2")
    else:
        return ("other", "unknown")


def _dataframe_to_markdown(
    df: pd.DataFrame, summary: Optional[Dict[str, Any]] = None
) -> str:
    """Render benchmark results as task-based Markdown tables with comparisons."""

    computed_summary = summary or _compute_aggregate_statistics(df)
    narrative = _format_summary_markdown(computed_summary)

    if df.empty:
        return narrative + "\n| |\n|---|\n"

    sections: list[str] = []
    
    # Separate crispyx methods from reference methods
    crispyx_methods = df[df["method"].str.startswith("crispyx_", na=False)].copy()
    reference_methods = df[~df["method"].str.startswith("crispyx_", na=False)].copy()
    
    # Group by task (preprocessing vs differential_expression)
    for task in ["preprocessing", "differential_expression"]:
        task_crispyx = crispyx_methods[
            crispyx_methods["method"].apply(lambda x: _extract_task_and_test_type(x)[0] == task)
        ].copy()
        task_reference = reference_methods[
            reference_methods["method"].apply(lambda x: _extract_task_and_test_type(x)[0] == task)
        ].copy()
        
        if task_crispyx.empty and task_reference.empty:
            continue
        
        # Create task header
        task_title = "Preprocessing" if task == "preprocessing" else "Differential Expression"
        sections.append(f"## {task_title}\n")
        
        # Table 1: crispyx basic info
        if not task_crispyx.empty:
            sections.append("### crispyx Methods\n")
            
            # Select relevant columns for crispyx table
            crispyx_cols = ["method", "description", "status", "runtime_seconds", "peak_memory_mb"]
            if task == "preprocessing":
                crispyx_cols.extend(["cells_kept", "genes_kept", "result_path"])
            else:  # differential_expression
                crispyx_cols.extend(["groups", "genes", "result_path"])
            
            # Keep only columns that exist
            crispyx_table_cols = [col for col in crispyx_cols if col in task_crispyx.columns]
            crispyx_table = task_crispyx[crispyx_table_cols].copy()
            
            # Sort by test type
            crispyx_table["_sort_key"] = crispyx_table["method"].apply(
                lambda x: _extract_task_and_test_type(x)[1]
            )
            crispyx_table = crispyx_table.sort_values("_sort_key").drop(columns=["_sort_key"])
            
            # Drop all-NA columns
            crispyx_table = crispyx_table.dropna(axis=1, how="all")
            
            sections.append(_frame_to_markdown_table(crispyx_table) + "\n")
        
        # Table 2: Reference comparisons
        if not task_reference.empty:
            sections.append("### Reference Comparisons\n")
            
            # Select relevant columns for comparison table
            comparison_cols = [
                "method", "reference_tool", "test_type", "status",
                "runtime_seconds", "peak_memory_mb"
            ]
            
            # Add comparison-specific metrics
            if task == "preprocessing":
                # Add performance difference columns for preprocessing
                perf_diff_cols = [
                    "runtime_diff_seconds", "runtime_diff_pct",
                    "memory_diff_mb", "memory_diff_pct"
                ]
                comparison_cols.extend([col for col in perf_diff_cols if col in task_reference.columns])
                
                comparison_specific = [
                    "normalization_max_abs_diff", "log1p_max_abs_diff",
                    "avg_log_effect_max_abs_diff", "pseudobulk_effect_max_abs_diff"
                ]
            else:  # differential_expression
                # For DE comparisons, add performance difference columns and correlation metrics
                perf_diff_cols = [
                    "runtime_diff_seconds", "runtime_diff_pct",
                    "method_a_peak_memory_mb", "method_a_avg_memory_mb",
                    "method_b_peak_memory_mb", "method_b_avg_memory_mb",
                    "memory_diff_mb", "memory_diff_pct"
                ]
                comparison_cols.extend([col for col in perf_diff_cols if col in task_reference.columns])
                
                comparison_specific = [
                    "effect_pearson_corr", "effect_spearman_corr", "effect_top_k_overlap",
                    "statistic_pearson_corr", "statistic_spearman_corr", "statistic_top_k_overlap",
                    "pvalue_pearson_corr", "pvalue_spearman_corr", "pvalue_top_k_overlap"
                ]
            comparison_cols.extend([col for col in comparison_specific if col in task_reference.columns])
            
            # Add result paths
            comparison_cols.extend(["method_a_result_path", "method_b_result_path"])
            
            # Keep only columns that exist
            comparison_table_cols = [col for col in comparison_cols if col in task_reference.columns]
            comparison_table = task_reference[comparison_table_cols].copy()
            
            # Sort by test type then by method
            comparison_table["_sort_key"] = comparison_table["method"].apply(
                lambda x: (_extract_task_and_test_type(x)[1], x)
            )
            comparison_table = comparison_table.sort_values("_sort_key").drop(columns=["_sort_key"])
            
            # Drop all-NA columns
            comparison_table = comparison_table.dropna(axis=1, how="all")
            
            sections.append(_frame_to_markdown_table(comparison_table) + "\n")
    
    tables = "\n".join(sections)
    return narrative + "\n" + tables



def _run_single_benchmark(
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict]:
    """Run benchmarks for a single dataset and return results DataFrame and metadata.
    
    Returns
    -------
    tuple
        (results_dataframe, metadata_dict)
    """
    
    # Standardize dataset (uses cache if available)
    if not config.quiet:
        print(f"\nStandardizing dataset: {config.dataset_path.name}")
    
    standardized_path = standardize_dataset(
        dataset_path=config.dataset_path,
        perturbation_column=config.perturbation_column,
        control_label=config.control_label,
        gene_name_column=config.gene_name_column,
        output_dir=config.output_dir,
        force=config.force_restandardize,
    )

    # Ensure a CSR copy exists for efficient row-wise access (adaptive QC,
    # NB-GLM, size factors).  On CSC or dense standardized files this avoids
    # the O(total_nnz)-per-row-slice pathology that caused multi-hour hangs.
    from crispyx.data import get_matrix_storage_format

    standardized_csr_path = standardized_path  # default: already CSR
    if get_matrix_storage_format(standardized_path) != "csr":
        preprocessing_dir = config.output_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        csr_path = preprocessing_dir / f"standardized_csr_{standardized_path.stem.removeprefix('standardized_')}.h5ad"
        csr_result = run_preprocess_csr(
            standardized_path, csr_path,
            chunk_size=config.chunk_size,
        )
        standardized_csr_path = Path(csr_result["result_path"])
        if not config.quiet:
            print(f"  ✓ CSR-converted standardized file: {standardized_csr_path.name}")

    # Use CSR path for adaptive QC (row-wise matrix access) and context loading.
    context = _load_dataset_context(standardized_csr_path)
    context["dataset_path"] = str(standardized_path)
    context["output_dir"] = config.output_dir
    
    # Track whether QC params are adaptive or user-specified
    adaptive_qc = config.qc_params is None
    qc_params_used = None
    
    # Calculate or use provided QC params (this happens inside create_benchmark_suite)
    # We need to extract them for logging
    if adaptive_qc:
        import anndata as ad
        if not config.quiet:
            print(f"\nCalculating adaptive QC parameters (mode: {config.adaptive_qc_mode})...")
        
        adata_temp = ad.read_h5ad(standardized_csr_path, backed='r')
        try:
            qc_params_used = calculate_adaptive_qc_thresholds(
                adata_temp, "perturbation", mode=config.adaptive_qc_mode, chunk_size=config.chunk_size
            )
            
            if not config.quiet:
                print(f"  ✓ min_genes: {qc_params_used['min_genes']}")
                print(f"  ✓ min_cells_per_perturbation: {qc_params_used['min_cells_per_perturbation']}")
                print(f"  ✓ min_cells_per_gene: {qc_params_used['min_cells_per_gene']}")
                print(f"  ✓ chunk_size: {qc_params_used['chunk_size']}")
        finally:
            adata_temp.file.close()
    else:
        qc_params_used = {
            "min_genes": config.qc_params.min_genes,
            "min_cells_per_perturbation": config.qc_params.min_cells_per_perturbation,
            "min_cells_per_gene": config.qc_params.min_cells_per_gene,
            "chunk_size": config.chunk_size if config.chunk_size is not None else config.qc_params.chunk_size,
        }
    
    # Check cache validity and invalidate if needed
    cached_config = load_cache_config(config.output_dir)
    should_invalidate = False
    invalidate_reason = ""
    
    if config.force_restandardize:
        should_invalidate = True
        invalidate_reason = "force_restandardize=True"
    elif cached_config is not None:
        # Check if cache version changed
        if cached_config.get("cache_version") != CACHE_VERSION:
            should_invalidate = True
            cached_version = cached_config.get("cache_version", "unknown")
            invalidate_reason = f"cache version changed ({cached_version} -> {CACHE_VERSION})"
        # Check if QC params changed
        elif cached_config.get("qc_params") != qc_params_used:
            should_invalidate = True
            invalidate_reason = "QC parameters changed"
        # Check if standardized dataset path changed
        elif cached_config.get("standardized_dataset_path") != str(standardized_path):
            should_invalidate = True
            invalidate_reason = "standardized dataset path changed"
    
    if should_invalidate:
        invalidate_cache(config.output_dir, invalidate_reason)
    
    # Save current config to cache
    save_cache_config(config.output_dir, qc_params_used, str(standardized_path))

    available_methods = create_benchmark_suite(
        dataset_path=standardized_csr_path,
        output_dir=config.output_dir,
        perturbation_column="perturbation",  # Always 'perturbation' after standardization
        control_label="control",  # Always 'control' after standardization
        gene_name_column=config.gene_name_column,
        qc_params=config.qc_params,  # Will use adaptive if None
        n_cores=config.n_cores,
        chunk_size=config.chunk_size,
        memory_limit_gb=config.resource_limits.memory_limit if config.resource_limits.memory_limit > 0 else None,
    )
    
    # Define optional methods (excluded by default, only run when explicitly requested)
    # These methods are useful for specific validation but not needed for standard benchmarks
    # Methods with _pydeseq2 suffix are specifically designed to match PyDESeq2's exact behavior
    OPTIONAL_METHODS = {
        "crispyx_preprocess_csc",  # CSR→CSC conversion; auto-included when crispyx_de_wilcoxon is selected
        "crispyx_standardize_csr",  # CSC/dense→CSR conversion; auto-included when crispyx_de_nb_glm is selected
        "crispyx_de_lfcshrink",  # LFC shrinkage step (runs separately from NB-GLM base)
        "crispyx_de_nb_glm_pydeseq2",  # NB-GLM with PyDESeq2-parity settings for benchmarking
        "crispyx_de_lfcshrink_pydeseq2",  # LFC shrinkage with PyDESeq2-parity base
        "pertpy_de_lfcshrink",  # PyDESeq2 LFC shrinkage step
    }
    
    methods_to_run = config.methods_to_run
    if methods_to_run:
        # When --methods is specified, allow any available method including optional ones
        selected_names: list[str] = []
        for name in methods_to_run:
            if name in available_methods:
                selected_names.append(name)
                continue

            # Allow shorthand prefixes (e.g., "scanpy" -> all scanpy_* methods)
            prefix_matches = [
                method_name
                for method_name in available_methods
                if method_name.startswith(f"{name}_")
            ]

            if prefix_matches:
                selected_names.extend(prefix_matches)
                continue

            raise ValueError(
                f"Unknown method '{name}'. Available methods: {sorted(available_methods)}"
            )
    else:
        # Default: run all methods except optional ones
        ordered_methods = sorted(available_methods.values(), key=_method_sort_key)
        selected_names = [
            method.name for method in ordered_methods
            if method.name not in OPTIONAL_METHODS
        ]
        
        # Add optional methods if explicitly enabled in config
        if config.optional_methods:
            for opt_name in config.optional_methods:
                if opt_name in available_methods and opt_name not in selected_names:
                    selected_names.append(opt_name)
                elif opt_name not in available_methods:
                    import warnings
                    warnings.warn(f"Unknown optional method '{opt_name}' will be skipped")

    # Auto-include dependencies: if a selected method has a depends_on that is
    # not yet in selected_names but exists in available_methods, add it now.
    # This ensures e.g. crispyx_preprocess_csc is included whenever
    # crispyx_de_wilcoxon is selected, without running it when wilcoxon is absent.
    def _resolve_dependencies(names: list[str], methods: Dict[str, BenchmarkMethod]) -> list[str]:
        """Return names extended with any missing depends_on prerequisites."""
        resolved: list[str] = list(names)
        resolved_set: set[str] = set(names)
        # Iterate until stable (handles transitive chains)
        changed = True
        while changed:
            changed = False
            for name in list(resolved):
                method = methods.get(name)
                if method and method.depends_on:
                    dep = method.depends_on
                    if dep in methods and dep not in resolved_set:
                        resolved.append(dep)
                        resolved_set.add(dep)
                        changed = True
        return resolved

    selected_names = _resolve_dependencies(selected_names, available_methods)

    # Sort methods topologically to respect dependencies
    # Methods with depends_on must run after their dependency
    def _topological_sort_methods(names: list[str], methods: Dict[str, BenchmarkMethod]) -> list[str]:
        """Sort method names so dependencies run before dependents."""
        # Build dependency graph
        in_degree = {name: 0 for name in names}
        dependents = {name: [] for name in names}
        
        for name in names:
            method = methods.get(name)
            if method and method.depends_on:
                dep = method.depends_on
                if dep in names:
                    in_degree[name] += 1
                    dependents[dep].append(name)
                elif dep not in methods:
                    # Dependency not in available methods - warn but continue
                    pass
        
        # Kahn's algorithm for topological sort
        queue = [name for name in names if in_degree[name] == 0]
        sorted_names = []
        
        while queue:
            # Sort queue to maintain deterministic order
            queue.sort(key=lambda n: names.index(n))
            current = queue.pop(0)
            sorted_names.append(current)
            
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles (shouldn't happen with proper config)
        if len(sorted_names) != len(names):
            # Some methods couldn't be sorted - return original order with warning
            missing = set(names) - set(sorted_names)
            import warnings
            warnings.warn(f"Dependency cycle detected for methods: {missing}. Using original order.")
            return names
        
        return sorted_names
    
    selected_names = _topological_sort_methods(selected_names, available_methods)

    rows = []
    
    # Calculate memory limit in bytes
    memory_limit_bytes = None
    if config.resource_limits.memory_limit > 0:
        memory_limit_bytes = int(config.resource_limits.memory_limit * 1024 * 1024 * 1024)

    # Initialize Docker runner if Docker mode is enabled
    docker_runner = None
    if config.use_docker:
        if not _is_docker_available():
            raise RuntimeError(
                "Docker mode requested but Docker is not available. "
                "Please install Docker or remove --use-docker flag."
            )
        docker_runner = DockerRunner(
            image_name=config.docker_image,
            memory_limit_gb=config.resource_limits.memory_limit,
            n_cores=config.n_cores or 8,
            workspace_root=REPO_ROOT,
            quiet=config.quiet,
        )
        # Ensure Docker image is available
        if not docker_runner.ensure_image():
            raise RuntimeError(
                f"Docker image {config.docker_image} is not available and could not be built. "
                "Please build the image manually: docker build -t crispyx-benchmark benchmarking/"
            )
        # Clean up any orphaned containers from previous runs
        cleaned = docker_runner.cleanup_orphaned_containers()
        if cleaned > 0 and not config.quiet:
            print(f"  🧹 Cleaned up {cleaned} orphaned container(s) from previous runs")
        if not config.quiet:
            print(f"  🐳 Docker mode enabled with image: {config.docker_image}")
            print(f"      Memory limit: {config.resource_limits.memory_limit:.1f} GB")

    # Create progress bar for benchmark execution
    show_progress = config.show_progress and not config.quiet
    if show_progress:
        method_iterator = tqdm(
            selected_names,
            desc="Running benchmarks",
            unit="method",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    else:
        method_iterator = selected_names

    # Track which methods were actually executed (not loaded from cache)
    actually_executed: set[str] = set()

    for name in method_iterator:
        if name not in available_methods:
            raise ValueError(f"Unknown method '{name}'. Available methods: {sorted(available_methods)}")
        method = available_methods[name]
        
        # Check if output already exists (for resuming interrupted benchmarks)
        if config.skip_existing and check_output_exists(method.name, config.output_dir):
            # Try to load cached benchmark metadata
            cached_result = load_method_result(method.name, config.output_dir)
            
            if cached_result is not None:
                # Use cached result - mark as loaded from cache for accurate summary
                cached_result["_loaded_from_cache"] = True
                rows.append(cached_result)
                
                # Display cached stats
                runtime = cached_result.get("elapsed_seconds") or cached_result.get("runtime_seconds")
                memory = cached_result.get("max_memory_mb") or cached_result.get("peak_memory_mb")
                
                if not config.quiet:
                    stats_str = ""
                    if runtime is not None:
                        stats_str += f"{runtime:.1f}s"
                    if memory is not None:
                        if stats_str:
                            stats_str += ", "
                        stats_str += f"{memory:.1f}MB"
                    
                    if stats_str:
                        print(f"  ⏭️  Skipping {method.name} (cached: {stats_str})")
                    else:
                        print(f"  ⏭️  Skipping {method.name} (cached)")
                
                # Update progress bar with cached stats
                if show_progress:
                    postfix_dict = {"status": "cached"}
                    if memory is not None:
                        postfix_dict["memory"] = f"{memory:.1f}MB"
                    if runtime is not None:
                        postfix_dict["time"] = f"{runtime:.1f}s"
                    method_iterator.set_postfix(postfix_dict, refresh=False)  # type: ignore
            else:
                # No cache, but output exists - create minimal row with basic metadata
                if not config.quiet:
                    print(f"  ⏭️  Skipping {method.name} (output exists, recovering metadata)")
                
                existing_path = get_expected_output_path(method.name, config.output_dir)
                row = {
                    "method": method.name,
                    "description": method.description,
                    "status": "skipped_existing",
                    "elapsed_seconds": None,
                    "peak_memory_mb": None,
                    "result_path": _normalise_path(existing_path, context) if existing_path else None,
                }
                
                # Try to extract metadata and performance stats from the output file
                if existing_path and existing_path.exists():
                    try:
                        if existing_path.suffix == ".h5ad":
                            import anndata as ad
                            adata_meta = ad.read_h5ad(existing_path, backed='r')
                            row["genes"] = adata_meta.n_vars
                            row["cells"] = adata_meta.n_obs
                            
                            # Try to extract performance stats from uns if available
                            if hasattr(adata_meta, 'uns'):
                                uns = adata_meta.uns
                                # Check for de_results groups
                                if 'de_results' in uns:
                                    row["groups"] = len(uns.get('de_results', {}))
                                # Check for benchmark metadata stored in uns
                                if 'benchmark_metadata' in uns:
                                    meta = uns['benchmark_metadata']
                                    if 'elapsed_seconds' in meta:
                                        row["elapsed_seconds"] = float(meta['elapsed_seconds'])
                                    if 'peak_memory_mb' in meta:
                                        row["peak_memory_mb"] = float(meta['peak_memory_mb'])
                                # Also check for timing info stored directly
                                for key in ['elapsed_seconds', 'runtime_seconds', 'total_seconds']:
                                    if key in uns and row.get("elapsed_seconds") is None:
                                        row["elapsed_seconds"] = float(uns[key])
                                for key in ['peak_memory_mb', 'max_memory_mb', 'memory_mb']:
                                    if key in uns and row.get("peak_memory_mb") is None:
                                        row["peak_memory_mb"] = float(uns[key])
                            
                            adata_meta.file.close()
                        elif existing_path.suffix == ".csv":
                            df_meta = pd.read_csv(existing_path, nrows=0)
                            row["columns"] = list(df_meta.columns)
                    except Exception:
                        pass  # Silently ignore metadata extraction errors
                
                rows.append(row)
                
                # Save minimal cache entry to avoid repeated "no metrics" warnings
                save_method_result(method.name, row, config.output_dir)
                
                # Update progress bar for skipped items (show recovered stats if available)
                if show_progress:
                    mem_str = f"{row['peak_memory_mb']:.0f}MB" if row.get('peak_memory_mb') else "--"
                    time_str = f"{row['elapsed_seconds']:.1f}s" if row.get('elapsed_seconds') else "--"
                    method_iterator.set_postfix(  # type: ignore
                        status="skipped",
                        memory=mem_str,
                        time=time_str,
                        refresh=False
                    )
            continue
        
        # Update progress bar with current method name
        if show_progress:
            method_iterator.set_description(f"Running {method.name}")  # type: ignore
        
        result = _run_with_limits(
            method, context, memory_limit_bytes, config.resource_limits.time_limit,
            use_docker=config.use_docker, docker_runner=docker_runner,
        )
        
        # Update progress bar with result status
        if show_progress:
            status = result.get("status", "unknown")
            mem_mb = result.get("peak_memory_mb") or 0
            elapsed = result.get("elapsed_seconds") or 0
            method_iterator.set_postfix(  # type: ignore
                status=status, 
                memory=f"{mem_mb:.0f}MB",
                time=f"{elapsed:.1f}s",
                refresh=False
            )
        
        row = {
            "method": method.name,
            "description": method.description,
            "status": result.get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "spawn_overhead_seconds": result.get("spawn_overhead_seconds"),
            "peak_memory_mb": result.get("peak_memory_mb"),
            "avg_memory_mb": result.get("avg_memory_mb"),
        }
        summary = result.get("summary", {})
        if summary:
            row.update(summary)
        if result.get("error"):
            row["error"] = result["error"]
        
        rows.append(row)
        actually_executed.add(method.name)
        
        # Save result to cache immediately after completion
        save_method_result(method.name, row, config.output_dir)
    
    # Load any cached results that weren't re-run and merge with new results
    cached_results = load_cached_results(config.output_dir)
    methods_in_rows = {row["method"] for row in rows}
    
    # Add cached results for methods that weren't processed this run
    for cached_row in cached_results:
        method_name = cached_row.get("method")
        if method_name not in methods_in_rows:
            cached_row["_loaded_from_cache"] = True
            rows.append(cached_row)
    
    # Calculate accurate counts based on what was actually executed
    executed_count = len(actually_executed)
    cached_count = len(rows) - executed_count
    
    # Display merge summary
    if not config.quiet and len(rows) > 0:
        print(f"\n📊 Benchmark Result Summary:")
        if executed_count > 0:
            print(f"   ✅ Newly executed: {executed_count} method(s)")
        if cached_count > 0:
            print(f"   💾 Loaded from cache: {cached_count} method(s)")
        print(f"   📋 Total results: {len(rows)} method(s)")
    
    # Create metadata dict
    metadata = {
        "qc_params_used": qc_params_used,
        "adaptive_qc": adaptive_qc,
        "adaptive_qc_mode": config.adaptive_qc_mode if adaptive_qc else None,
        "standardized_dataset_path": str(standardized_path),
        "original_dataset_path": str(config.dataset_path),
    }

    return _postprocess_results(pd.DataFrame(rows)), metadata


def main() -> None:
    args = parse_args()
    
    # Initialize global environment configuration from CLI args
    from .env_config import set_global_env_config, EnvironmentConfig
    
    env_config = EnvironmentConfig(
        r_home=args.r_home if hasattr(args, 'r_home') else None,
        default_n_cores=args.n_cores if hasattr(args, 'n_cores') else None,
    )
    set_global_env_config(env_config)
    
    # Load configuration from YAML or command-line arguments
    if args.config:
        config_result = BenchmarkConfig.from_yaml(args.config)
        configs = config_result if isinstance(config_result, list) else [config_result]
        
        # Override environment config from YAML with CLI args if provided
        for config in configs if isinstance(configs, list) else [configs]:
            if config.environment_config is None:
                config.environment_config = env_config
            else:
                # CLI args take precedence over YAML
                if args.r_home:
                    config.environment_config.r_home = args.r_home
                if args.n_cores:
                    config.environment_config.default_n_cores = args.n_cores
            
            # Update global config with the final merged config
            set_global_env_config(config.environment_config)
            
            # Also update n_cores in config if specified via CLI
            if args.n_cores and config.n_cores is None:
                config.n_cores = args.n_cores

            # Update chunk_size if specified via CLI
            if hasattr(args, 'chunk_size') and args.chunk_size is not None:
                config.chunk_size = args.chunk_size

            # Allow CLI to override the methods list when using a config file
            if args.methods is not None:
                config.methods_to_run = args.methods
            
            # Allow CLI to override skip_existing
            if hasattr(args, 'skip_existing') and args.skip_existing is not None:
                config.skip_existing = args.skip_existing
            
            # Handle Docker flags from CLI (CLI takes precedence)
            if hasattr(args, 'use_docker') and args.use_docker:
                config.use_docker = True
            if hasattr(args, 'docker_image') and args.docker_image != "crispyx-benchmark:latest":
                config.docker_image = args.docker_image
    else:
        # Traditional CLI mode - create config from arguments
        dataset_path = args.data_path
        output_dir: Path = args.output_dir
        
        if args.generate_demo:
            seed = None if args.demo_seed == -1 else args.demo_seed
            generated = write_demo_dataset(dataset_path, seed=seed)
            print(f"Generated demo dataset at {generated}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_path}' was not found. "
                "Generate it with --generate-demo or 'python benchmarking/generate_demo_dataset.py', "
                "or supply --data-path to an existing .h5ad file."
            )
        
        # Set memory limit for resource limits
        if args.memory_limit and args.memory_limit > 0:
            memory_limit = args.memory_limit
        else:
            memory_limit = 0
            
        # Determine skip_existing from CLI (default True if not specified)
        skip_existing_value = args.skip_existing if args.skip_existing is not None else True
        
        # Create single config from CLI args
        configs = [BenchmarkConfig(
            dataset_path=dataset_path,
            dataset_name=dataset_path.stem,
            output_dir=output_dir,
            qc_params=QCParams(),
            resource_limits=ResourceLimits(
                time_limit=args.time_limit,
                memory_limit=memory_limit,
            ),
            methods_to_run=args.methods,
            quiet=args.quiet,
            n_cores=args.n_cores if hasattr(args, 'n_cores') else None,
            environment_config=env_config,
            chunk_size=args.chunk_size if hasattr(args, 'chunk_size') else None,
            use_docker=args.use_docker if hasattr(args, 'use_docker') else False,
            docker_image=args.docker_image if hasattr(args, 'docker_image') else "crispyx-benchmark:latest",
            skip_existing=skip_existing_value,
        )]
    
    # Handle --build-docker flag
    if hasattr(args, 'build_docker') and args.build_docker:
        if not any(c.use_docker for c in configs):
            print("Warning: --build-docker specified but --use-docker is not enabled.")
        print("Building Docker image...")
        if not _build_docker_image(quiet=args.quiet):
            print("Error: Failed to build Docker image.")
            sys.exit(1)
    
    # Handle --clear-results flag (clear entire output directory)
    if hasattr(args, 'clear_results') and args.clear_results:
        for config in configs:
            if config.output_dir.exists():
                import shutil
                print(f"Clearing output directory: {config.output_dir}")
                shutil.rmtree(config.output_dir)
    
    # Handle --force flag (force re-run of methods, selectively clearing cache)
    # When --methods is specified, only clear cache for those specific methods
    # When --methods is NOT specified, clear entire cache to force re-run of all methods
    if hasattr(args, 'force') and args.force:
        for config in configs:
            cache_dir = config.output_dir / ".benchmark_cache"
            if cache_dir.exists():
                if args.methods is not None and len(args.methods) > 0:
                    # Selective cache clearing: only clear cache for specified methods
                    # This preserves cached results for other methods (used in reports)
                    methods_to_clear = set(args.methods)
                    cleared_count = 0
                    for cache_file in cache_dir.glob("*.json"):
                        # Skip config.json and temp files
                        if cache_file.name in ("config.json",) or cache_file.name.startswith("."):
                            continue
                        # Extract method name from cache filename (e.g., "crispyx_de_nb_glm.json")
                        method_name = cache_file.stem
                        # Check if this method should be cleared
                        # Also check for prefix matches (e.g., "crispyx" matches "crispyx_de_nb_glm")
                        should_clear = method_name in methods_to_clear or any(
                            method_name.startswith(f"{prefix}_") for prefix in methods_to_clear
                        )
                        if should_clear:
                            try:
                                cache_file.unlink()
                                cleared_count += 1
                            except Exception as exc:
                                print(f"Warning: Failed to clear cache for {method_name}: {exc}")
                    if cleared_count > 0:
                        print(f"Cleared cache for {cleared_count} method(s) in {cache_dir}")
                else:
                    # No specific methods specified: clear entire cache
                    import shutil
                    print(f"Clearing entire benchmark cache: {cache_dir}")
                    shutil.rmtree(cache_dir)
            # When force is specified, disable skip_existing unless explicitly set
            if args.skip_existing is None:
                config.skip_existing = False
    
    # Handle --regenerate-report flag (skip execution, only regenerate reports)
    if hasattr(args, 'regenerate_report') and args.regenerate_report:
        for i, config in enumerate(configs):
            if len(configs) > 1:
                print(f"\n{'='*60}")
                print(f"Dataset {i+1}/{len(configs)}: {config.dataset_name}")
                print(f"{'='*60}")
            
            cache_dir = config.output_dir / ".benchmark_cache"
            if not cache_dir.exists():
                print(f"Warning: No cache found at {cache_dir}, skipping {config.dataset_name}")
                continue
            
            print(f"Regenerating reports from cache: {config.output_dir}")
            evaluate_benchmarks(config.output_dir)
            print(f"  Report: {config.output_dir / 'benchmark_report.md'}")
        return
    
    # Run benchmarks for each configuration
    for i, config in enumerate(configs):
        if len(configs) > 1:
            print(f"\n{'='*60}")
            print(f"Dataset {i+1}/{len(configs)}: {config.dataset_name}")
            print(f"{'='*60}")
        
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run benchmarks with standardization and adaptive QC
        df, metadata = _run_single_benchmark(config)
        
        # Save results
        summary = _compute_aggregate_statistics(df)
        
        # Add metadata to summary
        summary.update(metadata)
        
        csv_path = config.output_dir / "results.csv"
        md_path = config.output_dir / "results.md"
        summary_json_path = config.output_dir / "summary.json"

        df.to_csv(csv_path, index=False)
        md_path.write_text(_dataframe_to_markdown(df, summary))
        summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        
       # Run evaluation (generate comparison tables and visualizations)
        evaluate_benchmarks(config.output_dir)

        if not config.quiet:
            print(f"\nBenchmark complete for {config.dataset_name}")
            print(f"  Results: {csv_path}")
            print(f"  Summary: {md_path}")
            print(f"  JSON: {summary_json_path}")
            if metadata.get("adaptive_qc"):
                print(f"\n  Adaptive QC parameters saved to summary.json")


if __name__ == "__main__":
    main()
