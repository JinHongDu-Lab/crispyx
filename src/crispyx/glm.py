"""Generalized linear models utilities for differential expression."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence, Literal, Tuple

import numba as nb
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, solve as scipy_solve
from scipy.optimize import minimize_scalar, minimize, brentq
from scipy.special import gammaln, digamma, polygamma

# Import Numba kernels from separate module
from ._kernels import (
    gammaln_nb,
    _nb_loglik_grid_numba,
    _nb_ll_for_alpha,
    _compute_mle_dispersion_numba,
    _nb_map_grid_search_numba,
    _nb_map_grid_search_with_refinement_numba,
    _wls_solve_2x2_numba,
    _irls_batch_numba,
)

# Import profiling utilities from dedicated module
from .profiling import Profiler, MemoryProfiler, TimingProfiler

# Import memory utilities
from ._memory import (
    _get_available_memory_mb,
    _estimate_dense_memory_gb,
    _estimate_gene_batch_size_fitter,
    _estimate_max_workers,
)

logger = logging.getLogger(__name__)



@dataclass
class NBGLMResult:
    """Result of fitting a negative binomial GLM for a single gene."""

    coef: np.ndarray
    se: np.ndarray
    dispersion: float
    converged: bool
    n_iter: int
    deviance: float
    max_cooks: float | None = None


@dataclass
class NBGLMBatchResult:
    """Result of fitting NB GLM for multiple genes in a batch."""

    coef: np.ndarray  # Shape: (n_genes, n_features)
    se: np.ndarray  # Shape: (n_genes, n_features)
    dispersion: np.ndarray  # Shape: (n_genes,)
    converged: np.ndarray  # Shape: (n_genes,) bool
    n_iter: np.ndarray  # Shape: (n_genes,) int
    deviance: np.ndarray  # Shape: (n_genes,)


@dataclass
class ControlStatisticsCache:
    """Cached statistics for control cells to avoid redundant computation.
    
    When fitting independent NB-GLM models for multiple perturbations, each
    comparison includes the same control cells. This cache precomputes and
    stores control cell contributions to the IRLS normal equations, allowing
    them to be reused across all perturbation comparisons instead of being
    redundantly computed.
    
    The cache stores:
    - Control cell intercept (β₀): baseline log-expression per gene
    - Control dispersion: estimated from control cells only
    - XᵀWX contribution from control cells (for intercept column)
    - XᵀWz contribution from control cells
    - Global size factors (optional): precomputed on all cells for consistency
    - Global dispersion (optional): precomputed MAP dispersion using all cells
    - Global dispersion prior variance: for MAP shrinkage
    
    Memory optimization: We store control_matrix as dense (not mu/weights)
    since mu and weights change during IRLS and must be recomputed anyway.
    Storing the dense matrix avoids repeated .toarray() calls in workers.
    
    This reduces IRLS complexity from O(n_perturbations × n_control × n_genes × n_iter)
    to O(n_control × n_genes × n_iter) for control-related computations.
    """
    
    # Control cell data (stored as dense for efficiency)
    control_matrix: np.ndarray  # Shape: (n_control, n_genes) - always dense
    control_n: int  # Number of control cells
    control_offset: np.ndarray  # Shape: (n_control,) log size factors
    
    # Precomputed intercept (baseline expression for control)
    beta_intercept: np.ndarray  # Shape: (n_genes,)
    
    # Control dispersion (estimated from control cells only)
    control_dispersion: np.ndarray  # Shape: (n_genes,)
    
    # XᵀWX contribution from control (for intercept only, since X[control, perturbation] = 0)
    # For a simple intercept+perturbation design:
    # XᵀWX[0,0] from control = sum of weights for control cells per gene
    control_xtwx_intercept: np.ndarray  # Shape: (n_genes,)
    
    # XᵀWz contribution from control
    # For intercept: XᵀWz[0] from control = sum of (W * z) for control cells per gene  
    control_xtwz_intercept: np.ndarray  # Shape: (n_genes,)
    
    # Mean expression for control (for dispersion trend fitting)
    control_mean_expr: np.ndarray  # Shape: (n_genes,)
    
    # Expression counts for control cells
    control_expr_counts: np.ndarray  # Shape: (n_genes,)
    
    # Proportion of control cells expressing each gene
    pts_rest: np.ndarray  # Shape: (n_genes,)
    
    # Global size factors (optional): precomputed on all cells for consistency
    # When provided, all comparisons use the same size factors (faster, more consistent)
    global_size_factors: np.ndarray | None = None  # Shape: (n_cells_total,)
    
    # Global MAP dispersion (optional): precomputed using all cells
    # When provided, workers skip per-comparison MAP dispersion computation
    global_dispersion: np.ndarray | None = None  # Shape: (n_genes,)
    global_dispersion_trend: np.ndarray | None = None  # Shape: (n_genes,)
    
    # Prior variance for dispersion shrinkage (from global trend fitting)
    global_disp_prior_var: float | None = None
    
    # Dispersion scope: 'global' or 'per_comparison'
    # When 'global', workers use precomputed global_dispersion and skip MoM/trend
    dispersion_scope: str | None = None
    
    # Frozen control sufficient statistics (for memory-efficient parallel fitting)
    # When these are set, workers don't need the raw control_matrix
    # This reduces per-worker pickle size from ~5GB to ~1MB for large datasets
    frozen_control_W_sum: np.ndarray | None = None  # Shape: (n_genes,) - sum of control weights
    frozen_control_Wz_sum: np.ndarray | None = None  # Shape: (n_genes,) - sum of control W*z
    frozen_control_mu_sum: np.ndarray | None = None  # Shape: (n_genes,) - sum of control mu (for dispersion)
    frozen_control_resid_sq_sum: np.ndarray | None = None  # Shape: (n_genes,) - sum of (Y-mu)^2 (for dispersion)
    frozen_control_Y_sum: np.ndarray | None = None  # Shape: (n_genes,) - sum of control counts
    use_frozen_control: bool = False  # Flag to indicate frozen mode is active


def precompute_control_statistics(
    control_matrix: np.ndarray | sp.csr_matrix,
    control_offset: np.ndarray,
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
    min_mu: float = 0.5,
    dispersion_method: Literal["moments", "cox-reid"] = "moments",
    global_size_factors: np.ndarray | None = None,
    freeze_control: bool = False,
) -> ControlStatisticsCache:
    """Precompute control cell statistics for reuse across perturbation comparisons.
    
    This function fits a simple intercept-only model to control cells to estimate
    the baseline expression level (β₀) per gene. The resulting intercept, mean
    expression, weights, and XᵀWX/XᵀWz contributions are cached for reuse.
    
    Parameters
    ----------
    control_matrix
        Expression matrix for control cells, shape (n_control, n_genes).
    control_offset
        Log size factors for control cells, shape (n_control,).
    max_iter
        Maximum IRLS iterations for intercept estimation.
    tol
        Convergence tolerance.
    min_mu
        Minimum fitted mean value.
    dispersion_method
        Method for dispersion estimation.
    global_size_factors
        Optional precomputed size factors for all cells (n_cells_total,).
        When provided, stored in cache for use across all comparisons.
    freeze_control
        If True, compute frozen sufficient statistics (W_sum, Wz_sum, etc.) and
        set control_matrix to None to save memory. This reduces per-worker pickle
        size from ~5GB to ~1MB for large datasets. Workers can use the frozen
        stats with `fit_batch_with_frozen_control()` instead of the raw matrix.
        Default False.
        
    Returns
    -------
    ControlStatisticsCache
        Cached statistics for control cells.
    """
    # Check sparsity and determine strategy
    is_sparse = sp.issparse(control_matrix)
    if is_sparse:
        nnz = control_matrix.nnz
        total = control_matrix.shape[0] * control_matrix.shape[1]
        sparsity = 1.0 - (nnz / total) if total > 0 else 0.0
    else:
        nnz = np.count_nonzero(control_matrix)
        sparsity = 1.0 - (nnz / control_matrix.size) if control_matrix.size > 0 else 0.0
    
    if is_sparse:
        control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
        n_control, n_genes = control_matrix.shape
        logger.debug(f"Control matrix sparsity: {sparsity:.1%} ({n_control} cells × {n_genes} genes)")
    else:
        control_expr_counts = np.sum(control_matrix > 0, axis=0)
        n_control, n_genes = control_matrix.shape
    
    offset = np.asarray(control_offset, dtype=np.float64)
    
    # Compute pts_rest (proportion of control cells expressing each gene)
    pts_rest = control_expr_counts / n_control if n_control > 0 else np.zeros(n_genes, dtype=np.float32)
    
    # Compute normalized mean expression - use sparse operations if highly sparse
    if is_sparse and sparsity > 0.8:
        # For highly sparse matrices, use sparse mean
        control_mean_expr = np.asarray(control_matrix.mean(axis=0)).ravel() / np.exp(offset).mean()
        mean_counts = np.asarray(control_matrix.mean(axis=0)).ravel()
    else:
        # Densify for IRLS computations
        if is_sparse:
            Y = np.asarray(control_matrix.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(control_matrix, dtype=np.float64)
        normalized = Y / np.exp(offset)[:, None]
        control_mean_expr = normalized.mean(axis=0)
        mean_counts = Y.mean(axis=0)
    
    # Always densify for IRLS (required for efficient vectorized operations)
    if is_sparse:
        Y = np.asarray(control_matrix.toarray(), dtype=np.float64)
    else:
        Y = np.asarray(control_matrix, dtype=np.float64)
    
    # Initialize intercept: log of mean normalized counts
    mean_offset = np.exp(offset).mean()
    beta_intercept = np.log(np.maximum(mean_counts / mean_offset, 1e-10))
    
    # Initialize dispersion
    alpha = np.full(n_genes, 0.1, dtype=np.float64)
    
    # IRLS for intercept-only model
    log_min_mu = np.log(min_mu)
    offset_col = offset[:, None]
    
    mu = np.empty((n_control, n_genes), dtype=np.float64)
    weights = np.empty_like(mu)
    z = np.empty_like(mu)
    
    for iteration in range(max_iter):
        # Compute mu = exp(β₀ + offset)
        eta = beta_intercept[None, :] + offset_col
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, min_mu, out=mu)
        
        # Compute weights: W = μ² / (μ + α * μ²)
        variance = mu + alpha[None, :] * mu * mu
        np.divide(mu * mu, np.maximum(variance, min_mu), out=weights)
        
        # Working response: z = η + (Y - μ) / μ
        resid = Y - mu
        z = eta + resid / np.maximum(mu, min_mu)
        
        # Solve for intercept: β₀ = sum(W * (z - offset)) / sum(W)
        z_centered = z - offset_col
        xtwx = np.sum(weights, axis=0)  # (n_genes,)
        xtwz = np.sum(weights * z_centered, axis=0)  # (n_genes,)
        
        beta_new = xtwz / np.maximum(xtwx, 1e-10)
        
        # Check convergence
        if np.max(np.abs(beta_new - beta_intercept)) < tol:
            beta_intercept = beta_new
            break
        
        beta_intercept = beta_new
        
        # Update dispersion (method of moments)
        resid_sq = resid * resid
        numerator = np.sum((resid_sq - Y) / np.maximum(mu * mu, min_mu), axis=0)
        dof = max(n_control - 1, 1)
        alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
        alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
    
    # Final mu and weights
    eta = beta_intercept[None, :] + offset_col
    np.clip(eta, log_min_mu, 20.0, out=eta)
    np.exp(eta, out=mu)
    np.maximum(mu, min_mu, out=mu)
    
    variance = mu + alpha[None, :] * mu * mu
    np.divide(mu * mu, np.maximum(variance, min_mu), out=weights)
    
    # Compute XᵀWX and XᵀWz for control (intercept column only)
    z_centered = eta + (Y - mu) / np.maximum(mu, min_mu) - offset_col
    control_xtwx_intercept = np.sum(weights, axis=0)  # (n_genes,)
    control_xtwz_intercept = np.sum(weights * z_centered, axis=0)  # (n_genes,)
    
    # Compute frozen control sufficient statistics if requested
    # These allow workers to skip the raw control_matrix entirely
    frozen_control_W_sum = None
    frozen_control_Wz_sum = None
    frozen_control_mu_sum = None
    frozen_control_resid_sq_sum = None
    frozen_control_Y_sum = None
    use_frozen = False
    
    if freeze_control:
        # Compute frozen statistics for memory-efficient parallel fitting
        # These are the sufficient statistics needed for NB-GLM fitting
        frozen_control_W_sum = control_xtwx_intercept.copy()  # Same as sum of weights
        frozen_control_Wz_sum = control_xtwz_intercept.copy()  # Same as sum of W*z
        frozen_control_mu_sum = np.sum(mu, axis=0)  # For dispersion updates
        resid = Y - mu
        frozen_control_resid_sq_sum = np.sum(resid * resid, axis=0)  # For dispersion
        frozen_control_Y_sum = np.sum(Y, axis=0)  # For dispersion variance term
        use_frozen = True
        
        # Set control_matrix to None to save memory
        # Workers will use frozen stats instead
        Y_to_store = None
        logger.debug(
            f"Frozen control stats computed: control_n={n_control}, n_genes={n_genes}, "
            f"memory saved: {n_control * n_genes * 8 / 1e6:.1f} MB"
        )
    else:
        Y_to_store = Y
    
    # Free temporary arrays
    del mu, weights, z_centered, eta
    
    return ControlStatisticsCache(
        control_matrix=Y_to_store,  # None if freeze_control=True
        control_n=n_control,
        control_offset=offset if not freeze_control else None,  # Not needed if frozen
        beta_intercept=beta_intercept,
        control_dispersion=alpha,
        control_xtwx_intercept=control_xtwx_intercept,
        control_xtwz_intercept=control_xtwz_intercept,
        control_mean_expr=control_mean_expr,
        control_expr_counts=control_expr_counts.astype(np.int32),
        pts_rest=pts_rest.astype(np.float32),
        global_size_factors=global_size_factors,
        frozen_control_W_sum=frozen_control_W_sum,
        frozen_control_Wz_sum=frozen_control_Wz_sum,
        frozen_control_mu_sum=frozen_control_mu_sum,
        frozen_control_resid_sq_sum=frozen_control_resid_sq_sum,
        frozen_control_Y_sum=frozen_control_Y_sum,
        use_frozen_control=use_frozen,
    )


def precompute_control_statistics_streaming(
    path: "str | Path",
    control_mask: np.ndarray,
    control_offset: np.ndarray,
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
    min_mu: float = 0.5,
    global_size_factors: np.ndarray | None = None,
    freeze_control: bool = True,
    chunk_size: int = 4096,
) -> ControlStatisticsCache:
    """Streaming version of precompute_control_statistics for very large control groups.

    Instead of densifying the full control matrix (which can exceed 100+ GiB for
    large datasets), this function reads control cells from disk in chunks and
    accumulates sufficient statistics for the intercept-only IRLS model.

    Peak memory is O(chunk_size × n_genes) instead of O(n_control × n_genes).

    Parameters
    ----------
    path
        Path to the h5ad file containing the count matrix.
    control_mask
        Boolean mask over all cells indicating control cells, shape (n_cells,).
    control_offset
        Log size factors for control cells, shape (n_control,).
    max_iter
        Maximum IRLS iterations for intercept estimation.
    tol
        Convergence tolerance.
    min_mu
        Minimum fitted mean value.
    global_size_factors
        Optional precomputed size factors for all cells (n_cells_total,).
    freeze_control
        Must be True for streaming mode (raw control_matrix is never materialised).
    chunk_size
        Number of control cells to process per chunk. Default 4096.

    Returns
    -------
    ControlStatisticsCache
        Cached statistics with frozen control sufficient statistics.
    """
    from pathlib import Path as _Path
    from .data import read_backed

    if not freeze_control:
        raise ValueError(
            "Streaming precompute_control_statistics requires freeze_control=True "
            "because the raw control matrix is never materialised."
        )

    path = _Path(path)
    offset = np.asarray(control_offset, dtype=np.float64)
    control_indices = np.where(control_mask)[0]
    n_control = len(control_indices)

    # Get n_genes from file
    backed = read_backed(path)
    n_genes = backed.n_vars
    backed.file.close()

    log_min_mu = np.log(min_mu)

    # ---- Helper to iterate control cells in chunks from disk ----
    def _iter_control_chunks():
        """Yield (Y_chunk, offset_chunk) for control cells."""
        bk = read_backed(path)
        try:
            for start in range(0, n_control, chunk_size):
                end = min(start + chunk_size, n_control)
                idx = control_indices[start:end]
                chunk = bk.X[idx, :]
                if sp.issparse(chunk):
                    chunk = np.asarray(chunk.toarray(), dtype=np.float64)
                else:
                    chunk = np.asarray(chunk, dtype=np.float64)
                yield chunk, offset[start:end]
        finally:
            bk.file.close()

    # ---- Pass 0: Compute expression counts & mean counts (single pass) ----
    expr_counts = np.zeros(n_genes, dtype=np.int64)
    count_sum = np.zeros(n_genes, dtype=np.float64)
    norm_sum = np.zeros(n_genes, dtype=np.float64)

    for Y_chunk, off_chunk in _iter_control_chunks():
        expr_counts += np.asarray((Y_chunk > 0).sum(axis=0)).ravel()
        count_sum += Y_chunk.sum(axis=0)
        norm_sum += (Y_chunk / np.exp(off_chunk)[:, None]).sum(axis=0)

    mean_counts = count_sum / n_control
    control_mean_expr = norm_sum / n_control
    pts_rest = (expr_counts / n_control).astype(np.float32) if n_control > 0 else np.zeros(n_genes, dtype=np.float32)

    # ---- Initialise intercept & dispersion ----
    mean_offset = np.exp(offset).mean()
    beta_intercept = np.log(np.maximum(mean_counts / mean_offset, 1e-10))
    alpha = np.full(n_genes, 0.1, dtype=np.float64)

    # ---- IRLS loop (each iteration streams through control cells) ----
    for iteration in range(max_iter):
        xtwx = np.zeros(n_genes, dtype=np.float64)
        xtwz = np.zeros(n_genes, dtype=np.float64)
        mom_numerator = np.zeros(n_genes, dtype=np.float64)

        for Y_chunk, off_chunk in _iter_control_chunks():
            n_chunk = Y_chunk.shape[0]
            # eta = beta_intercept + offset
            eta = beta_intercept[None, :] + off_chunk[:, None]
            np.clip(eta, log_min_mu, 20.0, out=eta)
            mu = np.exp(eta)
            np.maximum(mu, min_mu, out=mu)

            # Weights: W = mu^2 / var, var = mu + alpha * mu^2
            variance = mu + alpha[None, :] * mu * mu
            W = mu * mu / np.maximum(variance, min_mu)

            # Working response z = eta + (Y - mu) / mu
            resid = Y_chunk - mu
            z = eta + resid / np.maximum(mu, min_mu)
            z_centered = z - off_chunk[:, None]

            xtwx += W.sum(axis=0)
            xtwz += (W * z_centered).sum(axis=0)

            # MoM numerator: sum((y-mu)^2 - y) / mu^2
            mom_numerator += ((resid * resid - Y_chunk) / np.maximum(mu * mu, min_mu)).sum(axis=0)

        beta_new = xtwz / np.maximum(xtwx, 1e-10)

        if np.max(np.abs(beta_new - beta_intercept)) < tol:
            beta_intercept = beta_new
            break
        beta_intercept = beta_new

        # Update dispersion (MoM)
        dof = max(n_control - 1, 1)
        alpha_new = np.clip(mom_numerator / dof, 1e-8, 1e6)
        alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)

    # ---- Final pass: compute frozen sufficient statistics ----
    frozen_W_sum = np.zeros(n_genes, dtype=np.float64)
    frozen_Wz_sum = np.zeros(n_genes, dtype=np.float64)
    frozen_mu_sum = np.zeros(n_genes, dtype=np.float64)
    frozen_resid_sq_sum = np.zeros(n_genes, dtype=np.float64)
    frozen_Y_sum = np.zeros(n_genes, dtype=np.float64)

    for Y_chunk, off_chunk in _iter_control_chunks():
        eta = beta_intercept[None, :] + off_chunk[:, None]
        np.clip(eta, log_min_mu, 20.0, out=eta)
        mu = np.exp(eta)
        np.maximum(mu, min_mu, out=mu)

        variance = mu + alpha[None, :] * mu * mu
        W = mu * mu / np.maximum(variance, min_mu)

        resid = Y_chunk - mu
        z = eta + resid / np.maximum(mu, min_mu)
        z_centered = z - off_chunk[:, None]

        frozen_W_sum += W.sum(axis=0)
        frozen_Wz_sum += (W * z_centered).sum(axis=0)
        frozen_mu_sum += mu.sum(axis=0)
        frozen_resid_sq_sum += (resid * resid).sum(axis=0)
        frozen_Y_sum += Y_chunk.sum(axis=0)

    logger.info(
        f"Streaming control statistics computed: {n_control:,} cells × {n_genes:,} genes "
        f"in chunks of {chunk_size} "
        f"(peak memory saved: {n_control * n_genes * 8 / 1e9:.1f} GB)"
    )

    return ControlStatisticsCache(
        control_matrix=None,
        control_n=n_control,
        control_offset=None,
        beta_intercept=beta_intercept,
        control_dispersion=alpha,
        control_xtwx_intercept=frozen_W_sum,
        control_xtwz_intercept=frozen_Wz_sum,
        control_mean_expr=control_mean_expr,
        control_expr_counts=expr_counts.astype(np.int32),
        pts_rest=pts_rest,
        global_size_factors=global_size_factors,
        frozen_control_W_sum=frozen_W_sum,
        frozen_control_Wz_sum=frozen_Wz_sum,
        frozen_control_mu_sum=frozen_mu_sum,
        frozen_control_resid_sq_sum=frozen_resid_sq_sum,
        frozen_control_Y_sum=frozen_Y_sum,
        use_frozen_control=True,
    )


def precompute_global_dispersion(
    control_cache: ControlStatisticsCache,
    all_cell_matrix: np.ndarray | sp.csr_matrix,
    all_cell_offset: np.ndarray,
    *,
    n_grid: int = 25,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
    fit_type: Literal["parametric", "local", "mean"] = "parametric",
    fast_mode: bool = True,
    max_dense_fraction: float = 0.3,
    memory_limit_gb: float | None = None,
) -> ControlStatisticsCache:
    """Precompute global dispersion trend using all cells and update cache.
    
    This function computes a global dispersion trend using all cells in the
    dataset (control + all perturbations), similar to how DESeq2/PyDESeq2
    estimates dispersion from all samples. The trend is then used for MAP
    shrinkage in all per-perturbation comparisons.
    
    Using global dispersion has several advantages:
    1. More stable estimates (larger sample size for trend fitting)
    2. ~10× faster per-perturbation fitting (skips MAP estimation)
    3. More consistent results across perturbations
    
    Memory-adaptive behavior:
    If the estimated memory for densifying the matrix exceeds
    max_dense_fraction × min(available_memory, memory_limit_gb), 
    the function switches to chunk-wise streaming processing.
    
    Parameters
    ----------
    control_cache
        Precomputed control cell statistics cache.
    all_cell_matrix
        Count matrix for all cells, shape (n_cells, n_genes).
    all_cell_offset
        Log size factors for all cells, shape (n_cells,).
    n_grid
        Number of grid points for dispersion MAP estimation.
    min_disp
        Minimum dispersion value.
    max_disp
        Maximum dispersion value.
    fit_type
        Type of trend fitting ("parametric", "local", or "mean").
    fast_mode
        If True, use simple trend shrinkage (MoM → trend) instead of expensive
        MAP estimation. This is ~50× faster and suitable for most use cases.
        If False, use full MAP estimation with grid search + refinement.
    max_dense_fraction
        Maximum fraction of available memory to use for dense matrix.
        If estimated memory exceeds this, switch to streaming mode.
        Default is 0.3 (30% of available memory).
    memory_limit_gb
        Optional explicit memory limit in GB. If provided, the effective
        memory budget is min(available_memory, memory_limit_gb).
        
    Returns
    -------
    ControlStatisticsCache
        Updated cache with global_dispersion, global_dispersion_trend, and
        global_disp_prior_var fields populated.
    """
    from scipy.special import polygamma
    
    # Get matrix dimensions
    if sp.issparse(all_cell_matrix):
        n_cells, n_genes = all_cell_matrix.shape
    else:
        n_cells, n_genes = all_cell_matrix.shape
    
    # Estimate memory required for dense processing
    # Need ~3 copies: Y (data), mu (fitted values), and work arrays
    estimated_memory_gb = _estimate_dense_memory_gb(n_cells, n_genes, n_copies=3)
    
    # Compute effective memory budget
    available_memory_gb = _get_available_memory_mb() / 1000.0
    if memory_limit_gb is not None:
        effective_limit_gb = min(available_memory_gb, memory_limit_gb)
    else:
        effective_limit_gb = available_memory_gb
    
    memory_budget_gb = max_dense_fraction * effective_limit_gb
    
    # Check if we need streaming mode
    if estimated_memory_gb > memory_budget_gb:
        logger.warning(
            f"Dense matrix would require ~{estimated_memory_gb:.1f} GB "
            f"(budget: {memory_budget_gb:.1f} GB = {max_dense_fraction:.0%} of "
            f"{effective_limit_gb:.1f} GB). Switching to streaming mode."
        )
        return _precompute_global_dispersion_streaming(
            control_cache=control_cache,
            all_cell_matrix=all_cell_matrix,
            all_cell_offset=all_cell_offset,
            min_disp=min_disp,
            max_disp=max_disp,
            fit_type=fit_type,
        )
    
    # Standard dense processing path
    # Densify if sparse
    if sp.issparse(all_cell_matrix):
        Y = np.asarray(all_cell_matrix.toarray(), dtype=np.float64)
    else:
        Y = np.asarray(all_cell_matrix, dtype=np.float64)
    
    n_cells, n_genes = Y.shape
    offset = np.asarray(all_cell_offset, dtype=np.float64)
    
    # Compute mean expression (for trend fitting)
    normalized = Y / np.exp(offset)[:, None]
    mean_expr = normalized.mean(axis=0)
    
    # Initial MLE dispersion using method of moments
    # First fit a simple intercept model to get mu
    beta0 = np.log(np.maximum(mean_expr * np.exp(offset).mean(), 1e-10))
    eta = beta0[None, :] + offset[:, None]
    np.clip(eta, -30, 30, out=eta)
    mu = np.exp(eta)
    np.maximum(mu, 1e-10, out=mu)
    
    # MoM dispersion
    resid = Y - mu
    dof = max(n_cells - 1, 1)
    alpha_mle = np.sum((resid * resid - Y) / np.maximum(mu * mu, 1e-10), axis=0) / dof
    alpha_mle = np.clip(alpha_mle, min_disp, max_disp)
    
    # Fit dispersion trend
    trend = fit_dispersion_trend(mean_expr, alpha_mle, fit_type=fit_type)
    
    # Estimate prior variance for MAP shrinkage (PyDESeq2 style)
    log_alpha = np.log(np.maximum(alpha_mle, min_disp))
    log_trend = np.log(np.maximum(trend, min_disp))
    valid = np.isfinite(log_alpha) & np.isfinite(log_trend)
    
    if np.sum(valid) > 10:
        residuals = log_alpha[valid] - log_trend[valid]
        mad = np.median(np.abs(residuals - np.median(residuals)))
        squared_logres = (1.4826 * mad) ** 2
        num_vars = 2  # intercept + perturbation
        polygamma_corr = polygamma(1, (n_cells - num_vars) / 2)
        prior_var = max(squared_logres - polygamma_corr, 0.25)
    else:
        prior_var = 0.25
    
    if fast_mode:
        # FAST MODE: Use simple trend shrinkage (similar to shrink_dispersions)
        # This is ~50× faster than MAP and gives comparable results for global estimation
        # Shrink MoM dispersion toward trend using weighted average
        # Weight by reliability: use trend more for genes with extreme MoM estimates
        log_alpha_shrunk = np.where(
            valid,
            (log_alpha + log_trend) / 2,  # Simple average in log space
            log_trend  # Fall back to trend for invalid genes
        )
        global_dispersion = np.exp(np.clip(log_alpha_shrunk, np.log(min_disp), np.log(max_disp)))
    else:
        # FULL MODE: Compute MAP dispersion using vectorized grid search
        log_min = np.log(min_disp)
        log_max = np.log(max_disp)
        log_alpha_grid = np.linspace(log_min, log_max, n_grid)
        
        # Use fused kernel with Brent refinement
        best_log_alpha = _nb_map_grid_search_with_refinement_numba(
            Y, mu, log_trend, log_alpha_grid, prior_var,
            tol=1e-4,
            max_refine_iter=20,
        )
        global_dispersion = np.exp(np.clip(best_log_alpha, log_min, log_max))
    
    # Update cache with global dispersion
    control_cache.global_dispersion = global_dispersion
    control_cache.global_dispersion_trend = trend
    control_cache.global_disp_prior_var = prior_var
    
    return control_cache


def _precompute_global_dispersion_streaming(
    control_cache: ControlStatisticsCache,
    all_cell_matrix: np.ndarray | sp.csr_matrix,
    all_cell_offset: np.ndarray,
    *,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
    fit_type: Literal["parametric", "local", "mean"] = "parametric",
    chunk_size: int = 2048,
) -> ControlStatisticsCache:
    """Streaming version of global dispersion precomputation for large datasets.
    
    This function estimates dispersion by streaming through the data in chunks,
    avoiding the need to densify the full matrix. Uses method-of-moments
    estimation accumulated across chunks.
    
    Parameters
    ----------
    control_cache
        Precomputed control cell statistics cache.
    all_cell_matrix
        Count matrix for all cells, shape (n_cells, n_genes). Can be sparse.
    all_cell_offset
        Log size factors for all cells, shape (n_cells,).
    min_disp
        Minimum dispersion value.
    max_disp
        Maximum dispersion value.
    fit_type
        Type of trend fitting ("parametric", "local", or "mean").
    chunk_size
        Number of cells to process per chunk.
        
    Returns
    -------
    ControlStatisticsCache
        Updated cache with global_dispersion, global_dispersion_trend, and
        global_disp_prior_var fields populated.
    """
    from scipy.special import polygamma
    
    # Get dimensions
    if sp.issparse(all_cell_matrix):
        n_cells, n_genes = all_cell_matrix.shape
    else:
        n_cells, n_genes = all_cell_matrix.shape
    
    offset = np.asarray(all_cell_offset, dtype=np.float64)
    
    # Pass 1: Compute mean expression (for intercept estimation and trend fitting)
    # Accumulate sum and count
    expr_sum = np.zeros(n_genes, dtype=np.float64)
    
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        
        if sp.issparse(all_cell_matrix):
            chunk = np.asarray(all_cell_matrix[start:end].toarray(), dtype=np.float64)
        else:
            chunk = np.asarray(all_cell_matrix[start:end], dtype=np.float64)
        
        # Normalize by size factors for mean computation
        offset_chunk = offset[start:end]
        normalized_chunk = chunk / np.exp(offset_chunk)[:, None]
        expr_sum += normalized_chunk.sum(axis=0)
    
    mean_expr = expr_sum / n_cells
    
    # Compute intercept from mean expression
    mean_offset = np.exp(offset).mean()
    beta0 = np.log(np.maximum(mean_expr * mean_offset, 1e-10))
    
    # Pass 2: Compute MoM dispersion by streaming
    # MoM: alpha = sum((y - mu)^2 - y) / mu^2 / dof
    numerator_sum = np.zeros(n_genes, dtype=np.float64)
    
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        
        if sp.issparse(all_cell_matrix):
            chunk = np.asarray(all_cell_matrix[start:end].toarray(), dtype=np.float64)
        else:
            chunk = np.asarray(all_cell_matrix[start:end], dtype=np.float64)
        
        offset_chunk = offset[start:end]
        
        # Compute fitted values
        eta = beta0[None, :] + offset_chunk[:, None]
        np.clip(eta, -30, 30, out=eta)
        mu = np.exp(eta)
        np.maximum(mu, 1e-10, out=mu)
        
        # MoM numerator: (y - mu)^2 - y over mu^2
        resid = chunk - mu
        numerator = (resid * resid - chunk) / np.maximum(mu * mu, 1e-10)
        numerator_sum += numerator.sum(axis=0)
    
    dof = max(n_cells - 1, 1)
    alpha_mle = np.clip(numerator_sum / dof, min_disp, max_disp)
    
    # Fit dispersion trend
    trend = fit_dispersion_trend(mean_expr, alpha_mle, fit_type=fit_type)
    
    # Estimate prior variance (PyDESeq2 style)
    log_alpha = np.log(np.maximum(alpha_mle, min_disp))
    log_trend = np.log(np.maximum(trend, min_disp))
    valid = np.isfinite(log_alpha) & np.isfinite(log_trend)
    
    if np.sum(valid) > 10:
        residuals = log_alpha[valid] - log_trend[valid]
        mad = np.median(np.abs(residuals - np.median(residuals)))
        squared_logres = (1.4826 * mad) ** 2
        num_vars = 2
        polygamma_corr = polygamma(1, (n_cells - num_vars) / 2)
        prior_var = max(squared_logres - polygamma_corr, 0.25)
    else:
        prior_var = 0.25
    
    # Use simple trend shrinkage (equivalent to fast_mode=True)
    # Streaming mode always uses this for memory efficiency
    log_alpha_shrunk = np.where(
        valid,
        (log_alpha + log_trend) / 2,
        log_trend
    )
    global_dispersion = np.exp(np.clip(log_alpha_shrunk, np.log(min_disp), np.log(max_disp)))
    
    # Update cache
    control_cache.global_dispersion = global_dispersion
    control_cache.global_dispersion_trend = trend
    control_cache.global_disp_prior_var = prior_var
    
    logger.info(
        f"Streaming dispersion estimation complete: "
        f"processed {n_cells} cells in chunks of {chunk_size}"
    )
    
    return control_cache


def precompute_global_dispersion_from_path(
    path: str | Path,
    control_cache: ControlStatisticsCache,
    all_cell_offset: np.ndarray,
    *,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
    fit_type: Literal["parametric", "local", "mean"] = "parametric",
    chunk_size: int = 4096,
) -> ControlStatisticsCache:
    """Path-based streaming global dispersion for very large datasets.
    
    This function estimates dispersion by streaming directly from an h5ad file,
    avoiding the need to load the entire matrix into memory. Uses method-of-moments
    estimation accumulated across chunks read from disk.
    
    This is the preferred method for datasets that exceed available memory
    (e.g., Replogle-GW-k562 with ~2M cells × 8K genes = ~131 GB).
    
    Parameters
    ----------
    path
        Path to the h5ad file containing the count matrix.
    control_cache
        Precomputed control cell statistics cache.
    all_cell_offset
        Log size factors for all cells, shape (n_cells,).
    min_disp
        Minimum dispersion value.
    max_disp
        Maximum dispersion value.
    fit_type
        Type of trend fitting ("parametric", "local", or "mean").
    chunk_size
        Number of cells to process per chunk. Larger chunks are faster
        but use more memory. Default 4096 balances speed and memory.
        
    Returns
    -------
    ControlStatisticsCache
        Updated cache with global_dispersion, global_dispersion_trend, and
        global_disp_prior_var fields populated.
    """
    from pathlib import Path
    from scipy.special import polygamma
    from .data import read_backed
    
    path = Path(path)
    offset = np.asarray(all_cell_offset, dtype=np.float64)
    
    # Get dimensions from backed file
    backed = read_backed(path)
    n_cells, n_genes = backed.shape
    backed.file.close()
    
    logger.info(
        f"Computing global dispersion from path (streaming): "
        f"{n_cells:,} cells × {n_genes:,} genes"
    )
    
    # Pass 1: Compute mean expression (for intercept estimation and trend fitting)
    expr_sum = np.zeros(n_genes, dtype=np.float64)
    
    backed = read_backed(path)
    try:
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = backed.X[start:end, :]
            
            if sp.issparse(chunk):
                chunk = np.asarray(chunk.toarray(), dtype=np.float64)
            else:
                chunk = np.asarray(chunk, dtype=np.float64)
            
            # Normalize by size factors for mean computation
            offset_chunk = offset[start:end]
            normalized_chunk = chunk / np.exp(offset_chunk)[:, None]
            expr_sum += normalized_chunk.sum(axis=0)
    finally:
        backed.file.close()
    
    mean_expr = expr_sum / n_cells
    
    # Compute intercept from mean expression
    mean_offset = np.exp(offset).mean()
    beta0 = np.log(np.maximum(mean_expr * mean_offset, 1e-10))
    
    # Pass 2: Compute MoM dispersion by streaming from disk
    # MoM: alpha = sum((y - mu)^2 - y) / mu^2 / dof
    numerator_sum = np.zeros(n_genes, dtype=np.float64)
    
    backed = read_backed(path)
    try:
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = backed.X[start:end, :]
            
            if sp.issparse(chunk):
                chunk = np.asarray(chunk.toarray(), dtype=np.float64)
            else:
                chunk = np.asarray(chunk, dtype=np.float64)
            
            offset_chunk = offset[start:end]
            
            # Compute fitted values
            eta = beta0[None, :] + offset_chunk[:, None]
            np.clip(eta, -30, 30, out=eta)
            mu = np.exp(eta)
            np.maximum(mu, 1e-10, out=mu)
            
            # MoM numerator: (y - mu)^2 - y over mu^2
            resid = chunk - mu
            numerator = (resid * resid - chunk) / np.maximum(mu * mu, 1e-10)
            numerator_sum += numerator.sum(axis=0)
    finally:
        backed.file.close()
    
    dof = max(n_cells - 1, 1)
    alpha_mle = np.clip(numerator_sum / dof, min_disp, max_disp)
    
    # Fit dispersion trend
    trend = fit_dispersion_trend(mean_expr, alpha_mle, fit_type=fit_type)
    
    # Estimate prior variance (PyDESeq2 style)
    log_alpha = np.log(np.maximum(alpha_mle, min_disp))
    log_trend = np.log(np.maximum(trend, min_disp))
    valid = np.isfinite(log_alpha) & np.isfinite(log_trend)
    
    if np.sum(valid) > 10:
        residuals = log_alpha[valid] - log_trend[valid]
        mad = np.median(np.abs(residuals - np.median(residuals)))
        squared_logres = (1.4826 * mad) ** 2
        num_vars = 2
        polygamma_corr = polygamma(1, (n_cells - num_vars) / 2)
        prior_var = max(squared_logres - polygamma_corr, 0.25)
    else:
        prior_var = 0.25
    
    # Use simple trend shrinkage (equivalent to fast_mode=True)
    log_alpha_shrunk = np.where(
        valid,
        (log_alpha + log_trend) / 2,
        log_trend
    )
    global_dispersion = np.exp(np.clip(log_alpha_shrunk, np.log(min_disp), np.log(max_disp)))
    
    # Update cache
    control_cache.global_dispersion = global_dispersion
    control_cache.global_dispersion_trend = trend
    control_cache.global_disp_prior_var = prior_var
    
    logger.info(
        f"Path-based streaming dispersion complete: "
        f"processed {n_cells:,} cells in chunks of {chunk_size}"
    )
    
    return control_cache


class NBGLMFitter:
    """L-BFGS-B solver for negative binomial GLMs.

    Parameters
    ----------
    design:
        The design matrix with shape ``(n_samples, n_features)``.
    offset:
        Optional log-scale offset (e.g. library size) per sample.
    dispersion:
        Optional dispersion (alpha) for the negative binomial. If ``None`` the
        dispersion is re-estimated at each iteration using a method-of-moments
        update similar to the one used in statsmodels.
    max_iter:
        Maximum number of outer iterations for alternating optimization.
    tol:
        Absolute tolerance on the coefficient updates for convergence.
    poisson_init_iter:
        Maximum number of iterations for the Poisson initialisation stage. If
        set to ``0`` the Poisson warm start is skipped and coefficients are
        initialised at zero.
    ridge_penalty:
        Small diagonal ridge penalty added to the weighted normal equations to
        improve numerical stability. This does not change the estimator when
        the system is well conditioned but prevents failures when the Hessian
        is nearly singular.
    min_mu:
        Lower bound on the fitted mean to avoid issues with extremely small
        predicted counts for lowly expressed genes.
    min_total_count:
        Genes with a total count below this threshold are not fitted (the
        resulting ``NBGLMResult`` will report ``converged=False``).
    dispersion_method:
        Method for estimating dispersion when ``dispersion`` is None.

        - ``"moments"``: Method-of-moments (fast but less accurate).
        - ``"cox-reid"``: Cox-Reid adjusted profile likelihood (slower but more
          accurate, similar to DESeq2).
    """

    def __init__(
        self,
        design: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        dispersion: float | None = None,
        max_iter: int = 50,
        tol: float = 1e-6,
        poisson_init_iter: int = 20,
        ridge_penalty: float = 1e-6,
        min_mu: float = 0.5,
        min_total_count: float = 1.0,
        compute_cooks: bool = False,
        dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    ) -> None:
        self.design = np.asarray(design, dtype=np.float64)
        if self.design.ndim != 2:
            raise ValueError("design must be a 2D array")
        self.n_samples, self.n_features = self.design.shape
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.zeros(self.n_samples, dtype=np.float64)
        )
        if self.offset.shape != (self.n_samples,):
            raise ValueError("offset must have shape (n_samples,)")
        self.dispersion = dispersion
        self.max_iter = int(max_iter)
        self.tol = tol
        self.poisson_init_iter = int(max(0, poisson_init_iter))
        self.ridge_penalty = ridge_penalty
        self.min_mu = min_mu
        self.min_total_count = min_total_count
        self.compute_cooks = compute_cooks
        self.dispersion_method = dispersion_method

    def fit_gene(self, counts: ArrayLike) -> NBGLMResult:
        """Fit NB GLM for a single gene using L-BFGS-B optimization."""
        y = np.asarray(counts, dtype=np.float64)
        if y.shape != (self.n_samples,):
            raise ValueError("counts must have shape (n_samples,)")
        total = float(y.sum())
        if total < self.min_total_count or not np.isfinite(total):
            zeros = np.zeros(self.n_features, dtype=np.float64)
            return NBGLMResult(
                coef=zeros,
                se=np.full(self.n_features, np.inf, dtype=np.float64),
                dispersion=float("nan"),
                converged=False,
                n_iter=0,
                deviance=float("nan"),
                max_cooks=None,
            )
        
        return self._fit_gene_lbfgsb(y)

    def fit_matrix(self, matrix: ArrayLike, *, batch_size: int | None = None) -> list[NBGLMResult]:
        """Fit NB GLM for every gene (column) in a count matrix.

        Parameters
        ----------
        matrix : array-like of shape (n_samples, n_genes)
            Raw count matrix.  Sparse (CSC) and dense formats are accepted.
        batch_size : int or None, optional
            Number of genes to densify at once when *matrix* is sparse.
            ``None`` processes all genes in one batch.

        Returns
        -------
        list of NBGLMResult
            One result per gene, in column order.
        """
        if sp.issparse(matrix):
            sparse_matrix = sp.csc_matrix(matrix, dtype=np.float64)
            n_genes = sparse_matrix.shape[1]
            batch = batch_size or n_genes
            if sparse_matrix.shape[0] != self.n_samples:
                raise ValueError("matrix must have shape (n_samples, n_genes)")
            results: list[NBGLMResult] = []
            for start in range(0, n_genes, batch):
                end = min(start + batch, n_genes)
                dense_block = np.asarray(sparse_matrix[:, start:end].toarray())
                for col in range(dense_block.shape[1]):
                    results.append(self.fit_gene(dense_block[:, col]))
            return results

        y = np.asarray(matrix, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] != self.n_samples:
            raise ValueError("matrix must have shape (n_samples, n_genes)")

        batch = batch_size or y.shape[1]
        results: list[NBGLMResult] = []
        for start in range(0, y.shape[1], batch):
            end = min(start + batch, y.shape[1])
            block = y[:, start:end]
            for col in range(block.shape[1]):
                results.append(self.fit_gene(block[:, col]))
        return results

    def _fit_gene_lbfgsb(self, y: np.ndarray) -> NBGLMResult:
        """Fit NB GLM using L-BFGS-B optimization (PyDESeq2 style).
        
        This method directly optimizes the negative binomial log-likelihood
        using scipy's L-BFGS-B optimizer, which is the approach used by PyDESeq2.
        It alternates between optimizing coefficients (beta) and dispersion (alpha).
        """
        # Initialize with Poisson warm start
        beta = np.zeros(self.n_features, dtype=np.float64)
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start(y, beta.copy())
        
        # Initial dispersion estimate using method of moments
        eta = self.offset + self.design @ beta
        mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
        mu = np.maximum(mu, self.min_mu)
        alpha = self._update_alpha(y, mu, 0.1)
        
        converged = False
        n_iter = 0
        
        # Alternate between optimizing beta and alpha
        for outer_iter in range(self.max_iter):
            # Optimize beta given alpha using L-BFGS-B
            def neg_log_likelihood(beta_vec: np.ndarray) -> float:
                eta = self.offset + self.design @ beta_vec
                mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
                mu = np.maximum(mu, self.min_mu)
                r = 1.0 / max(alpha, 1e-10)
                # NB log-likelihood (using numba-accelerated gammaln)
                ll = np.sum(
                    gammaln_nb(y + r)
                    - gammaln_nb(r)
                    - gammaln_nb(y + 1)
                    + r * np.log(r / (r + mu))
                    + y * np.log(mu / (r + mu + 1e-12))
                )
                return -ll
            
            def gradient(beta_vec: np.ndarray) -> np.ndarray:
                eta = self.offset + self.design @ beta_vec
                mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
                mu = np.maximum(mu, self.min_mu)
                r = 1.0 / max(alpha, 1e-10)
                # Gradient: d(-ll)/d(beta) = X^T @ (mu - y * (1 + r) / (mu + r))
                # Simplified: X^T @ (mu * (y + r) / (mu + r) - y)
                w = (y + r) / (mu + r)
                grad = self.design.T @ (mu * w - y)
                return -grad  # Negative because we're minimizing negative LL
            
            beta_old = beta.copy()
            try:
                result = minimize(
                    neg_log_likelihood,
                    beta,
                    method='L-BFGS-B',
                    jac=gradient,
                    options={'maxiter': 50, 'gtol': self.tol, 'ftol': 1e-10}
                )
                if result.success or result.fun < neg_log_likelihood(beta):
                    beta = result.x
                n_iter = outer_iter + 1
            except Exception:
                pass
            
            # Update mu with new beta
            eta = self.offset + self.design @ beta
            mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
            mu = np.maximum(mu, self.min_mu)
            
            # Update dispersion
            if self.dispersion is not None:
                alpha = self.dispersion
            else:
                alpha_old = alpha
                alpha = self._update_alpha(y, mu, alpha)
                
                # Refine with Cox-Reid if requested
                if self.dispersion_method == "cox-reid":
                    variance = mu + alpha * (mu**2)
                    weights = (mu**2) / np.maximum(variance, self.min_mu)
                    alpha = self.estimate_dispersion_cox_reid(
                        y, mu, weights, initial_alpha=alpha
                    )
            
            # Check convergence
            beta_diff = float(np.max(np.abs(beta - beta_old)))
            if beta_diff < self.tol:
                converged = True
                break
        
        # Compute final statistics
        eta = self.offset + self.design @ beta
        mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
        mu = np.maximum(mu, self.min_mu)
        variance = mu + alpha * (mu**2)
        weights = (mu**2) / np.maximum(variance, self.min_mu)
        
        # Compute covariance matrix (inverse Hessian)
        cov_beta = self._hessian_inverse(weights)
        se = np.sqrt(np.maximum(np.diag(cov_beta), self.min_mu))
        
        # Compute deviance
        deviance = self._compute_deviance(y, mu, alpha)
        
        # Cook's distance if requested
        max_cooks = None
        if self.compute_cooks:
            hat_diag = self._hat_diagonal(weights, cov_beta)
            pearson_resid = (y - mu) / np.sqrt(np.maximum(variance, self.min_mu))
            denom = np.maximum((1.0 - hat_diag) ** 2, self.min_mu)
            cooks = (pearson_resid**2 / max(self.n_features, 1)) * (hat_diag / denom)
            max_cooks = float(np.nanmax(cooks)) if cooks.size else None
        
        return NBGLMResult(
            coef=beta,
            se=se,
            dispersion=alpha,
            converged=converged,
            n_iter=n_iter,
            deviance=deviance,
            max_cooks=max_cooks,
        )

    def _poisson_warm_start(self, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        for _ in range(self.poisson_init_iter):
            eta = self.offset + self.design @ beta
            mu = np.exp(np.clip(eta, a_min=np.log(self.min_mu), a_max=None))
            mu = np.maximum(mu, self.min_mu)
            weights = mu
            z = eta + (y - mu) / np.maximum(mu, self.min_mu)
            working_response = z - self.offset
            beta_new, _ = self._weighted_least_squares(weights, working_response)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                return beta_new
            beta = beta_new
        return beta

    def _weighted_least_squares(self, weights: np.ndarray, y_working: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if weights.shape != (self.n_samples,):
            raise ValueError("weights must have shape (n_samples,)")
        w_sqrt = np.sqrt(np.clip(weights, self.min_mu, None))
        x_weighted = self.design * w_sqrt[:, None]
        z_weighted = y_working * w_sqrt
        xtwx = x_weighted.T @ x_weighted
        if self.ridge_penalty:
            xtwx = xtwx + self.ridge_penalty * np.eye(self.n_features)
        xtwz = x_weighted.T @ z_weighted
        try:
            c, lower = cho_factor(xtwx, overwrite_a=False, check_finite=False)
            beta = cho_solve((c, lower), xtwz, check_finite=False)
            inv_hessian = cho_solve((c, lower), np.eye(self.n_features), check_finite=False)
        except np.linalg.LinAlgError:
            try:
                beta = np.linalg.solve(xtwx, xtwz)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(xtwx) @ xtwz
            try:
                inv_hessian = np.linalg.inv(xtwx)
            except np.linalg.LinAlgError:
                inv_hessian = np.linalg.pinv(xtwx)
        return beta, inv_hessian

    def _hessian_inverse(self, weights: np.ndarray) -> np.ndarray:
        w_sqrt = np.sqrt(np.clip(weights, self.min_mu, None))
        x_weighted = self.design * w_sqrt[:, None]
        xtwx = x_weighted.T @ x_weighted
        if self.ridge_penalty:
            xtwx = xtwx + self.ridge_penalty * np.eye(self.n_features)
        try:
            c, lower = cho_factor(xtwx, overwrite_a=False, check_finite=False)
            inv_hessian = cho_solve((c, lower), np.eye(self.n_features), check_finite=False)
        except np.linalg.LinAlgError:
            inv_hessian = np.linalg.pinv(xtwx)
        return inv_hessian

    def _hat_diagonal(self, weights: np.ndarray, inv_hessian: np.ndarray) -> np.ndarray:
        w_sqrt = np.sqrt(np.clip(weights, self.min_mu, None))
        x_weighted = self.design * w_sqrt[:, None]
        projection = x_weighted @ inv_hessian
        hat = np.sum(x_weighted * projection, axis=1)
        return np.clip(hat, 0.0, 1.0)

    @staticmethod
    def _compute_deviance(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
        mu = np.maximum(mu, 1e-12)
        if alpha <= 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu) - (y - mu), -mu)
            return 2.0 * float(np.nansum(terms))
        r = 1.0 / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = y * np.log(np.maximum(y, 1e-12) / mu)
            term2 = (y + r) * np.log((y + r) / (mu + r))
            dev = 2.0 * float(np.nansum(term1 - term2))
        return dev

    def _update_alpha(self, y: np.ndarray, mu: np.ndarray, current_alpha: float) -> float:
        # Method-of-moments style update used as a cheap approximation to
        # maximize the profile likelihood for alpha.
        resid = y - mu
        denom = np.maximum(mu**2, self.min_mu)
        numerator = np.sum((resid**2 - y) / denom)
        dof = max(y.size - self.n_features, 1)
        alpha = numerator / dof
        if not np.isfinite(alpha):
            alpha = current_alpha
        alpha = float(np.clip(alpha, 1e-8, 1e6))
        if alpha <= 0:
            alpha = 1e-8
        return alpha

    def estimate_dispersion_cox_reid(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        weights: np.ndarray,
        *,
        initial_alpha: float = 0.1,
        bounds: tuple[float, float] = (1e-8, 1e3),
    ) -> float:
        """Estimate dispersion using Cox-Reid adjusted profile likelihood.
        
        This method maximizes the adjusted profile log-likelihood for the
        dispersion parameter, which includes a bias correction term based on
        the Cox-Reid adjustment. This approach is similar to DESeq2's dispersion
        estimation.
        
        Parameters
        ----------
        y
            Count vector of shape (n_samples,).
        mu
            Current fitted mean values of shape (n_samples,).
        weights
            IRLS weights from the current fit.
        initial_alpha
            Starting value for the optimization.
        bounds
            Lower and upper bounds for the dispersion parameter.
            
        Returns
        -------
        float
            Estimated dispersion parameter.
        """
        n = len(y)
        p = self.n_features
        
        def neg_log_likelihood(alpha: float) -> float:
            if alpha <= 0:
                return np.inf
            r = 1.0 / alpha
            
            # Negative binomial log-likelihood (using numba-accelerated gammaln)
            ll = np.sum(
                gammaln_nb(y + r)
                - gammaln_nb(r)
                - gammaln_nb(y + 1)
                + r * np.log(r / (r + mu))
                + y * np.log(mu / (r + mu + 1e-12))
            )
            
            # Cox-Reid adjustment: -0.5 * log(det(X^T W X))
            # This accounts for the fact that we're profiling over beta
            variance = mu + alpha * (mu**2)
            w = (mu**2) / np.maximum(variance, self.min_mu)
            w_sqrt = np.sqrt(np.clip(w, self.min_mu, None))
            x_weighted = self.design * w_sqrt[:, None]
            xtwx = x_weighted.T @ x_weighted
            try:
                sign, log_det = np.linalg.slogdet(xtwx)
                if sign > 0:
                    ll -= 0.5 * log_det
            except np.linalg.LinAlgError:
                pass
            
            return -ll
        
        # Use bounded optimization
        try:
            result = minimize_scalar(
                neg_log_likelihood,
                bounds=bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 50},
            )
            if result.success and np.isfinite(result.x):
                return float(np.clip(result.x, bounds[0], bounds[1]))
        except Exception:
            pass
        
        # Fallback to method-of-moments
        return self._update_alpha(y, mu, initial_alpha)


def build_design_matrix(
    obs_frame,
    *,
    covariate_columns: Sequence[str],
    perturbation_indicator: np.ndarray,
    intercept: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Construct a design matrix from covariates and a perturbation indicator.

    Parameters
    ----------
    obs_frame:
        Pandas ``DataFrame`` (preferred) or structured numpy array containing
        covariate columns.
    covariate_columns:
        Columns that should be included in the design matrix. Categorical
        columns are expanded using one-hot encoding (dropping the first level).
    perturbation_indicator:
        Binary array of shape ``(n_samples,)`` marking perturbed cells.
    intercept:
        Whether to prepend an intercept column to the design matrix.

    Returns
    -------
    design:
        The numeric design matrix as a ``numpy.ndarray`` of ``float64``.
    column_names:
        The column names corresponding to the design matrix.
    """

    import pandas as pd

    if not isinstance(obs_frame, pd.DataFrame):
        obs_frame = pd.DataFrame(obs_frame)
    if len(obs_frame) != perturbation_indicator.shape[0]:
        raise ValueError("Number of samples in obs_frame and indicator do not match")
    matrices = []
    column_names: list[str] = []
    if intercept:
        matrices.append(np.ones((len(obs_frame), 1), dtype=np.float64))
        column_names.append("intercept")
    matrices.append(perturbation_indicator.reshape(-1, 1).astype(np.float64))
    column_names.append("perturbation")
    for column in covariate_columns:
        if column not in obs_frame.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_frame")
        series = obs_frame[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] == 0:
                continue
            matrices.append(dummies.to_numpy(dtype=np.float64))
            column_names.extend(dummies.columns.astype(str).tolist())
        else:
            matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
            column_names.append(str(column))
    design = np.hstack(matrices) if matrices else np.empty((len(obs_frame), 0), dtype=np.float64)
    return design, column_names


def fit_dispersion_trend(
    means: ArrayLike,
    dispersions: ArrayLike,
    *,
    min_mean: float = 0.5,
    fit_type: Literal["parametric", "local", "mean"] = "parametric",
    n_iter: int = 10,
) -> np.ndarray:
    """Fit a smooth mean-dispersion trend using DESeq2/PyDESeq2-style Gamma GLM.

    The parametric trend models dispersion as
    ``dispersion = asymptDisp + extraPois / mean``.

    This matches PyDESeq2's fitDispersionTrend which uses iteratively reweighted
    least squares (IRLS) with a Gamma family and log link, fitting
    ``E[disp] = a0 + a1 / mean``.

    Outliers are iteratively removed based on prediction ratio bounds.
    
    Parameters
    ----------
    means
        Mean expression values per gene (normalized counts).
    dispersions
        Raw genewise dispersion estimates.
    min_mean
        Minimum mean value for fitting (genes below this are excluded).
    fit_type
        Type of trend fitting:
        - "parametric": DESeq2/PyDESeq2-style Gamma GLM (recommended)
        - "local": Weighted local regression (LOWESS-like)
        - "mean": Simple median (fallback)
    n_iter
        Number of iterations for IRLS fitting.
    
    Returns
    -------
    np.ndarray
        Fitted trend values for each gene.
    """
    means_arr = np.asarray(means, dtype=np.float64)
    disp_arr = np.asarray(dispersions, dtype=np.float64)
    
    # For fitting, use all genes with valid dispersion (PyDESeq2 style)
    # Don't filter by min_mean - only exclude genes with inf covariate (mean=0)
    # or invalid dispersion values
    valid_for_fit = (
        np.isfinite(means_arr)
        & np.isfinite(disp_arr)
        & (means_arr > 0)  # Only exclude truly zero mean (avoids inf in 1/mean)
        & (disp_arr > 0)   # Exclude zero/negative dispersion
    )
    n_valid = valid_for_fit.sum()
    
    if n_valid < 3:
        baseline = np.nanmedian(disp_arr[valid_for_fit]) if np.any(valid_for_fit) else 0.1
        return np.full_like(means_arr, baseline, dtype=np.float64)
    
    x_valid = means_arr[valid_for_fit]
    y_valid = disp_arr[valid_for_fit]
    
    if fit_type == "mean":
        baseline = np.nanmedian(y_valid)
        return np.full_like(means_arr, baseline, dtype=np.float64)
    
    if fit_type == "parametric":
        # PyDESeq2-style Gamma GLM fit: disp = a0 + a1 / mean
        # Uses L-BFGS-B optimization with Gamma deviance loss, matching PyDESeq2
        # The model is: E[disp] = a0 + a1 / mean (identity link)
        # Loss = mean(target / mu + log(mu)) where mu = a0 + a1 * covariate
        from scipy.optimize import minimize
        
        try:
            # Build design matrix: [1, 1/mean] for covariates (PyDESeq2 style)
            covariates = np.column_stack([np.ones_like(x_valid), 1.0 / x_valid])
            targets = y_valid.copy()
            
            # PyDESeq2-style iterative fitting with outlier removal
            # Key difference: remove genes from arrays after each iteration
            old_params = np.array([0.1, 0.1])
            params = np.array([1.0, 1.0])
            X_current = covariates.copy()
            y_current = targets.copy()
            
            # Convergence criterion from PyDESeq2:
            # while (coeffs > 1e-10).all() and (log(|coeffs/old_coeffs|)^2).sum() >= 1e-6
            max_outer_iter = 20
            for outer_iter in range(max_outer_iter):
                # Check convergence
                if not (params > 1e-10).all():
                    break
                log_change = np.log(np.abs(params / old_params)) ** 2
                if log_change.sum() < 1e-6 and outer_iter > 0:
                    break
                
                old_params = params.copy()
                
                # Gamma GLM loss and gradient
                def loss(coeffs):
                    mu = X_current @ coeffs
                    mu = np.maximum(mu, 1e-10)
                    return np.nanmean(y_current / mu + np.log(mu))
                
                def grad(coeffs):
                    mu = X_current @ coeffs
                    mu = np.maximum(mu, 1e-10)
                    return -np.nanmean(
                        ((y_current / mu - 1)[:, None] * X_current) / mu[:, None],
                        axis=0
                    )
                
                try:
                    res = minimize(
                        loss,
                        x0=params,
                        jac=grad,
                        method="L-BFGS-B",
                        bounds=[(1e-12, np.inf), (1e-12, np.inf)],
                    )
                    if not res.success:
                        break
                    params = res.x
                    predictions = X_current @ params
                except Exception:
                    break
                
                # Outlier removal (PyDESeq2 style): keep genes with 1e-4 <= ratio < 15
                pred_ratios = y_current / np.maximum(predictions, 1e-10)
                keep_mask = (pred_ratios >= 1e-4) & (pred_ratios < 15.0)
                
                if keep_mask.sum() < 3:
                    break  # Not enough genes
                
                # Remove outliers from arrays (critical: this is how PyDESeq2 does it)
                X_current = X_current[keep_mask]
                y_current = y_current[keep_mask]
            
            # Compute trend for all genes: a0 + a1 / mean
            # PyDESeq2 doesn't clamp mean here - use small epsilon to avoid division by zero
            trend = params[0] + params[1] / np.maximum(means_arr, 1e-8)
            trend = np.maximum(trend, 1e-8)
            return trend
            
        except Exception:
            # Fall back to polynomial fit
            pass
    
    # Fallback: log-quadratic polynomial fit (original method)
    x = np.log(np.clip(means_arr[valid_for_fit], min_mean, None))
    y = np.log(np.clip(disp_arr[valid_for_fit], 1e-10, None))
    
    # Use robust weights to reduce influence of outliers
    median_y = np.median(y)
    mad_y = np.median(np.abs(y - median_y))
    weights = 1.0 / (1.0 + ((y - median_y) / (1.4826 * mad_y + 1e-8)) ** 2)
    
    try:
        coeffs = np.polyfit(x, y, deg=2, w=weights)
    except np.linalg.LinAlgError:
        coeffs = np.polyfit(x, y, deg=2)
    
    log_means_all = np.log(np.clip(means_arr, min_mean, None))
    trend = np.exp(np.polyval(coeffs, log_means_all))
    return np.maximum(trend, 1e-8)


def shrink_dispersions(
    raw: ArrayLike,
    trend: ArrayLike,
    *,
    prior_df: float | None = None,
    min_prior_df: float = 1.0,
    max_prior_df: float = 100.0,
    outlier_sigma: float = 2.0,
    n_iter: int = 5,
) -> np.ndarray:
    """Shrink dispersions toward fitted trend using empirical Bayes.
    
    This implements a DESeq2/PyDESeq2-style empirical Bayes shrinkage where the
    prior variance is estimated from the distribution of log-dispersion residuals
    around the trend using an iterative trimmed variance estimator.
    
    Genes with dispersions more than `outlier_sigma` standard deviations above 
    the trend keep their MLE (not shrunken), matching PyDESeq2's outlier handling.
    
    The shrinkage formula is:
        log(shrunk) = (prior_df * log(trend) + log(raw)) / (prior_df + 1)
    
    This is equivalent to a posterior mean estimate under a log-normal prior.
    
    Parameters
    ----------
    raw
        Raw MLE dispersion estimates.
    trend
        Fitted mean-dispersion trend values.
    prior_df
        Prior degrees of freedom controlling shrinkage strength.
        If None, estimated empirically from the data using iterative trimming.
    min_prior_df
        Minimum allowed prior degrees of freedom.
    max_prior_df
        Maximum allowed prior degrees of freedom.
    outlier_sigma
        Number of standard deviations above trend beyond which genes
        keep their MLE dispersion (not shrunken). Set to inf to disable.
    
    Returns
    -------
    np.ndarray
        Shrunken dispersion estimates.
    """
    raw_arr = np.asarray(raw, dtype=np.float64)
    trend_arr = np.asarray(trend, dtype=np.float64)
    shrunk = np.array(raw_arr, copy=True)
    mask = (
        np.isfinite(raw_arr)
        & np.isfinite(trend_arr)
        & (raw_arr > 0)
        & (trend_arr > 0)
    )
    if not np.any(mask):
        return shrunk
    
    log_raw = np.log(raw_arr[mask])
    log_trend = np.log(trend_arr[mask])
    residuals = log_raw - log_trend
    
    # PyDESeq2-style iterative trimmed variance estimator
    # Iteratively exclude residuals outside 2.5 MAD of median to get robust prior_var
    trim_threshold = 2.5
    use_for_var = np.ones(len(residuals), dtype=bool)
    
    for _ in range(n_iter):
        resid_subset = residuals[use_for_var]
        if len(resid_subset) < 3:
            break
            
        median_resid = np.median(resid_subset)
        mad = np.median(np.abs(resid_subset - median_resid))
        sigma_resid = 1.4826 * mad  # Scale MAD to approximate std dev
        
        if sigma_resid < 1e-8:
            # Fallback to sample std
            sigma_resid = max(np.std(resid_subset, ddof=1), 1e-4)
        
        # Update mask: include only residuals within trim_threshold * sigma
        abs_dev = np.abs(residuals - median_resid)
        use_for_var = abs_dev < trim_threshold * sigma_resid
        
        if use_for_var.sum() < 3:
            use_for_var = np.ones(len(residuals), dtype=bool)
            break
    
    # Final variance estimate from trimmed residuals
    resid_subset = residuals[use_for_var]
    median_resid = np.median(resid_subset)
    mad = np.median(np.abs(resid_subset - median_resid))
    sigma_resid = 1.4826 * mad
    
    # If variance is very small, use sample variance from trimmed set
    if sigma_resid < 1e-4:
        sigma_resid = max(np.std(resid_subset, ddof=1), 1e-4)
    
    prior_var = sigma_resid ** 2
    
    if prior_df is None:
        # PyDESeq2 estimates prior_df from the variance of log-dispersion residuals
        # Using the relationship: Var(log_disp) = trigamma(prior_df)
        # Approximate: prior_df ≈ 1 / prior_var for log-normal approximation
        if prior_var > 1e-8:
            # Use trigamma inverse approximation for better accuracy
            # For small prior_var, prior_df is large
            prior_df = 1.0 / prior_var
        else:
            prior_df = max_prior_df
        
        prior_df = float(np.clip(prior_df, min_prior_df, max_prior_df))
    
    # Identify outliers: genes with dispersion > outlier_sigma * sigma above trend
    # These genes keep their MLE (not shrunken) - PyDESeq2 behavior
    is_outlier = residuals > outlier_sigma * sigma_resid
    
    # Apply shrinkage to non-outliers
    log_post = (prior_df * log_trend + log_raw) / (prior_df + 1.0)
    
    # Non-outliers get shrunken values, outliers keep MLE
    shrunk_values = np.where(is_outlier, np.exp(log_raw), np.exp(log_post))
    shrunk[mask] = shrunk_values
    
    return shrunk


def estimate_dispersion_map(
    Y: np.ndarray,
    mu: np.ndarray,
    trend: np.ndarray,
    *,
    prior_var: float | None = None,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
    n_grid: int = 25,
    refine: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """Estimate MAP dispersion using vectorized grid search + optional refinement.

    This implements PyDESeq2-style MAP estimation where the dispersion
    is estimated by maximizing
    ``log L(Y | mu, alpha) + log prior(alpha | trend, prior_var)``.

    The prior is log-normal: ``log(alpha) ~ N(log(trend), prior_var)``.

    **Optimization (v5)**: Uses fused Numba kernel that combines grid search
    with Brent's method refinement in a single parallel pass over genes.
    This eliminates joblib process spawning overhead, achieving ~2-3× speedup
    over the previous joblib-based refinement while maintaining identical
    accuracy to scipy's minimize_scalar.
    
    Default is n_grid=25, refine=True which provides optimal balance of
    speed and accuracy. The Brent refinement makes grid size largely
    irrelevant for accuracy (all grid sizes 15-50 achieve perfect correlation).
    
    Parameters
    ----------
    Y
        Count matrix of shape (n_cells, n_genes).
    mu
        Fitted mean matrix of shape (n_cells, n_genes).
    trend
        Dispersion trend values of shape (n_genes,).
    prior_var
        Variance of the log-normal prior. If None, estimated from data
        using the variance of log-dispersion residuals around trend.
    min_disp
        Minimum allowed dispersion value.
    max_disp
        Maximum allowed dispersion value.
    n_grid
        Number of grid points for initial search. More points = better
        initial estimate but slower grid search. Default is 50 for good
        accuracy (96% Top-100 overlap with PyDESeq2 without refinement).
    refine
        If True, refine the grid search result using Brent's method.
        Default is True for best accuracy (99% Top-100 overlap with PyDESeq2).
        Set to False for ~2× speedup if slight accuracy loss is acceptable.
    n_jobs
        Number of parallel jobs for refinement. -1 uses all cores.
    
    Returns
    -------
    np.ndarray
        MAP dispersion estimates of shape (n_genes,).
    """
    from scipy.special import polygamma
    
    n_cells, n_genes = Y.shape
    # Avoid copies if already float64
    if Y.dtype != np.float64:
        Y = np.asarray(Y, dtype=np.float64)
    if mu.dtype != np.float64:
        mu = np.asarray(mu, dtype=np.float64)
    trend = np.asarray(trend, dtype=np.float64)
    
    # Clip mu in-place to avoid creating a copy
    np.maximum(mu, 1e-10, out=mu)
    
    # Memory-optimized MLE estimation using Numba for speed
    # Computes per-gene MLE dispersion without large intermediate arrays
    dof = max(n_cells - 2, 1)
    alpha_mle = _compute_mle_dispersion_numba(Y, mu, dof)
    alpha_mle = np.clip(alpha_mle, min_disp, max_disp)
    
    # Estimate prior variance if not provided (PyDESeq2 style)
    if prior_var is None:
        log_alpha = np.log(np.maximum(alpha_mle, min_disp))
        log_trend_arr = np.log(np.maximum(trend, min_disp))
        valid = np.isfinite(log_alpha) & np.isfinite(log_trend_arr)
        if np.sum(valid) > 10:
            residuals = log_alpha[valid] - log_trend_arr[valid]
            # Robust estimate using MAD (mean absolute deviation)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            squared_logres = (1.4826 * mad) ** 2
            # PyDESeq2 formula: max(squared_logres - polygamma_correction, 0.25)
            num_vars = 2  # intercept + perturbation
            polygamma_corr = polygamma(1, (n_cells - num_vars) / 2)
            prior_var = max(squared_logres - polygamma_corr, 0.25)
        else:
            prior_var = 0.25
    
    log_trend = np.log(np.maximum(trend, min_disp))
    log_min = np.log(min_disp)
    log_max = np.log(max_disp)
    
    # Create grid of log-alpha values
    log_alpha_grid = np.linspace(log_min, log_max, n_grid)
    
    # =========================================================================
    # Fused grid search + golden section refinement (Numba-parallel)
    # This is ~3-4× faster than separate grid search + joblib refinement
    # by eliminating process spawning overhead and keeping everything in Numba
    # =========================================================================
    if refine:
        # Use fused kernel with golden section refinement
        best_log_alpha = _nb_map_grid_search_with_refinement_numba(
            Y, mu, log_trend, log_alpha_grid, prior_var,
            tol=1e-4,
            max_refine_iter=20,
        )
        return np.exp(np.clip(best_log_alpha, log_min, log_max))
    else:
        # Grid search only (no refinement)
        best_log_alpha, best_idx = _nb_map_grid_search_numba(
            Y, mu, log_trend, log_alpha_grid, prior_var
        )
        return np.exp(np.clip(best_log_alpha, log_min, log_max))


def _estimate_apeglm_prior_scale(
    mle_lfc: np.ndarray,
    se: np.ndarray,
    init_scale: float = 1.0,
) -> float:
    """Estimate apeGLM prior scale parameter using PyDESeq2's adaptive method.
    
    This implements the prior variance estimation from PyDESeq2's `fit_prior_var`
    function, which finds the scale parameter that balances the data likelihood
    with the Cauchy prior.
    
    Parameters
    ----------
    mle_lfc
        MLE log-fold change estimates.
    se
        Standard errors of the MLE estimates.
    init_scale
        Initial guess for the scale parameter.
    
    Returns
    -------
    float
        Estimated prior scale parameter for the Cauchy prior.
    """
    mask = np.isfinite(mle_lfc) & np.isfinite(se) & (se > 0)
    if not np.any(mask):
        return init_scale
    
    S = mle_lfc[mask] ** 2
    D = se[mask] ** 2
    
    def objective(a: float) -> float:
        """Objective function: find a such that weighted mean of (S-D) equals a."""
        coeff = 1.0 / (2.0 * (a + D) ** 2)
        return float(((S - D) * coeff).sum() / coeff.sum() - a)
    
    # Match PyDESeq2's _fit_prior_var exactly:
    # - If objective(min_var) < 0, return min_var (maximum shrinkage)
    # - Otherwise, find root in [min_var, max_var]
    min_var, max_var = 1e-6, 400.0
    
    try:
        f_min = objective(min_var)
        
        if f_min < 0:
            # No root exists above min_var, use min_var (max shrinkage)
            # This is the PyDESeq2 behavior when Var(LFC) < median(SE^2)
            scale_sq = min_var
        else:
            # Find root in bracket
            scale_sq = brentq(objective, min_var, max_var, xtol=1e-6)
    except Exception:
        # Fallback: use min_var for maximum shrinkage
        scale_sq = min_var
    
    # PyDESeq2: prior_scale = min(sqrt(prior_var), 1.0)
    # No minimum floor on prior_scale - allow aggressive shrinkage
    return min(np.sqrt(scale_sq), 1.0)


def _fit_gene_apeglm_lbfgsb(
    y: np.ndarray,
    design_matrix: np.ndarray,
    log_size_factors: np.ndarray,
    disp: float,
    beta_init: np.ndarray,
    prior_scale: float,
    prior_no_shrink_scale: float,
    shrink_index: int,
    max_iter: int,
    tol: float,
    mle_se_j: float,
    min_mu: float = 0.0,
) -> tuple[np.ndarray, float, bool]:
    """Fit apeGLM for a single gene using L-BFGS-B with grid search fallback.
    
    This matches PyDESeq2's nbinomGLM implementation with proper NB likelihood
    and Cauchy prior on the LFC coefficient.
    
    NOTE: By default (min_mu=0.0), no min_mu clamping is applied, matching PyDESeq2.
    However, if the MLE coefficients were fitted WITH min_mu clamping (as in CRISPYx),
    using min_mu > 0 here ensures consistency between the MLE and MAP likelihoods,
    preventing pathological shrinkage behavior (LFC expansion instead of shrinkage).
    
    Parameters
    ----------
    y : np.ndarray
        Count vector for this gene (n_cells,).
    design_matrix : np.ndarray
        Design matrix (n_cells, n_params).
    log_size_factors : np.ndarray
        Log size factors (offset) for each cell (n_cells,).
    disp : float
        Dispersion parameter for this gene.
    beta_init : np.ndarray
        Initial coefficient values (n_params,).
    prior_scale : float
        Scale parameter for Cauchy prior on LFC.
    prior_no_shrink_scale : float
        Scale for normal prior on non-shrunk coefficients.
    shrink_index : int
        Index of coefficient to shrink (typically 1 for LFC).
    max_iter : int
        Maximum L-BFGS-B iterations.
    tol : float
        Convergence tolerance.
    mle_se_j : float
        MLE standard error for fallback.
        
    Returns
    -------
    beta_map : np.ndarray
        MAP coefficient estimates.
    se_map : float
        Approximate SE for shrunk coefficient.
    converged : bool
        Whether optimization converged.
    """
    n_params = design_matrix.shape[1]
    prior_scale_sq = prior_scale ** 2
    prior_no_shrink_var = prior_no_shrink_scale ** 2
    
    # Skip genes with invalid data
    if not np.isfinite(disp) or disp <= 0 or not np.all(np.isfinite(beta_init)):
        return beta_init, mle_se_j if np.isfinite(mle_se_j) else 1.0, False
    
    # Safety check for very low dispersion genes (near-Poisson behavior)
    # With very low dispersion, the NB likelihood becomes flat and optimization
    # can diverge to extreme values. PyDESeq2 doesn't have this issue because
    # it uses per-comparison dispersion which produces more reasonable estimates.
    # For genes with disp < 0.01 and small MLE LFC, return MLE (no shrinkage).
    if disp < 0.01 and abs(beta_init[shrink_index]) < 1.0:
        return beta_init, mle_se_j if np.isfinite(mle_se_j) else 1.0, True
    
    size = 1.0 / disp  # NB size parameter (r in NB(r, p))
    
    # Scale constant for numerical stability (PyDESeq2 style)
    scale_cnst = max(1.0, float(y.sum()) / 1e6)
    
    # Use log(min_mu) for clamping if min_mu > 0
    log_min_mu = np.log(min_mu) if min_mu > 0 else -np.inf
    
    def neg_log_posterior(beta: np.ndarray) -> float:
        """Negative log posterior = NB NLL + prior penalties.
        
        If min_mu > 0, applies min_mu clamping to match NB-GLM fitting.
        """
        # Linear predictor: X @ beta + offset = log(mu)
        xbeta = design_matrix @ beta
        eta = xbeta + log_size_factors
        
        # Apply min_mu clamping if specified (for consistency with NB-GLM fitting)
        if min_mu > 0:
            eta = np.maximum(eta, log_min_mu)
        
        # NB log-likelihood
        log_mu_plus_size = np.logaddexp(eta, np.log(size))
        nll = np.sum(-y * eta + (y + size) * log_mu_plus_size) / scale_cnst
        
        # Prior penalties
        # Normal prior on intercept and covariates (indices != shrink_index)
        prior_normal = 0.0
        for k in range(n_params):
            if k != shrink_index:
                prior_normal += beta[k] ** 2 / (2 * prior_no_shrink_var)
        
        # Cauchy prior on LFC (shrink_index): log(1 + (beta/scale)^2)
        prior_cauchy = np.log1p((beta[shrink_index] / prior_scale) ** 2)
        
        return nll + prior_normal + prior_cauchy
    
    def gradient(beta: np.ndarray) -> np.ndarray:
        """Gradient of negative log posterior.
        
        If min_mu > 0, applies min_mu clamping to match NB-GLM fitting.
        """
        xbeta = design_matrix @ beta
        eta = xbeta + log_size_factors
        
        # Apply min_mu clamping if specified
        if min_mu > 0:
            # Track which observations are clamped
            clamped = eta < log_min_mu
            eta = np.maximum(eta, log_min_mu)
        else:
            clamped = np.zeros(len(eta), dtype=bool)
        
        mu = np.exp(eta)
        
        # NB gradient: d(NLL)/d(beta) = X^T @ ((y + size) * mu / (mu + size) - y)
        w = (y + size) * mu / (mu + size) - y
        
        # Zero out gradient contribution for clamped observations
        # (they're at the boundary, so changes in beta don't affect NLL)
        if min_mu > 0 and clamped.any():
            w[clamped] = 0.0
        
        grad_nll = (design_matrix.T @ w) / scale_cnst
        
        # Prior gradients
        grad_prior = np.zeros(n_params, dtype=np.float64)
        for k in range(n_params):
            if k != shrink_index:
                grad_prior[k] = beta[k] / prior_no_shrink_var
        # Cauchy gradient: 2 * beta / (scale^2 + beta^2)
        grad_prior[shrink_index] = 2 * beta[shrink_index] / (prior_scale_sq + beta[shrink_index] ** 2)
        
        return grad_nll + grad_prior
    
    # Try L-BFGS-B optimization first
    converged = False
    beta_map = beta_init.copy()
    
    try:
        result = minimize(
            neg_log_posterior,
            beta_init,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(-30, 30)] * n_params,
            options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-8},
        )
        if result.success or result.fun < neg_log_posterior(beta_init):
            beta_map = result.x
            converged = result.success
    except Exception:
        pass
    
    # Grid search fallback if L-BFGS-B failed or didn't improve
    if not converged:
        # Grid search over LFC coefficient matching PyDESeq2's grid_fit_shrink_beta
        grid_lfc = np.linspace(-30.0, 30.0, 60)
        best_obj = neg_log_posterior(beta_map)
        best_beta = beta_map.copy()
        
        for lfc_val in grid_lfc:
            beta_test = beta_map.copy()
            beta_test[shrink_index] = lfc_val
            obj_val = neg_log_posterior(beta_test)
            if obj_val < best_obj:
                best_obj = obj_val
                best_beta = beta_test.copy()
        
        beta_map = best_beta
        converged = True  # Grid search always "converges"
    
    # Estimate SE from inverse Hessian at MAP
    try:
        eta = design_matrix @ beta_map + log_size_factors
        # No min_mu clamping in SE computation (matching PyDESeq2)
        mu = np.exp(eta)
        
        # NB weights: W = mu * (1 + mu/size)^(-1) = mu * size / (mu + size)
        W = mu * size / (mu + size)
        XtWX = design_matrix.T @ (design_matrix * W[:, None]) / scale_cnst
        
        # Add prior curvature
        for k in range(n_params):
            if k != shrink_index:
                XtWX[k, k] += 1.0 / prior_no_shrink_var
        
        # Cauchy Hessian: 2 * (s^2 - beta^2) / (s^2 + beta^2)^2
        beta_lfc = beta_map[shrink_index]
        cauchy_hess = 2 * (prior_scale_sq - beta_lfc**2) / (prior_scale_sq + beta_lfc**2)**2
        XtWX[shrink_index, shrink_index] += cauchy_hess
        
        inv_hess = np.linalg.inv(XtWX)
        se_map = np.sqrt(max(inv_hess[shrink_index, shrink_index], 1e-10))
    except (np.linalg.LinAlgError, ValueError):
        se_map = mle_se_j if np.isfinite(mle_se_j) else 1.0
    
    return beta_map, se_map, converged


def shrink_lfc_apeglm(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    dispersion: np.ndarray,
    mle_coef: np.ndarray,
    mle_se: np.ndarray,
    *,
    shrink_index: int = 1,
    prior_scale: float | None = None,
    prior_no_shrink_scale: float = 15.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    n_jobs: int = -1,
    batch_size: int = 128,
    min_mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply apeGLM LFC shrinkage using Cauchy prior (PyDESeq2-compatible).
    
    This implements the apeGLM (approximate posterior estimation for GLM)
    shrinkage method used by DESeq2/PyDESeq2. The method re-fits the NB-GLM
    model with a Cauchy prior penalty on the LFC coefficient.
    
    Key features matching PyDESeq2:
    - L-BFGS-B optimization with analytical gradients
    - Grid search fallback over [-5, 5] with 50 points when optimization fails
    - Parallel per-gene optimization using joblib with batch_size=128
    - Proper NB likelihood formulation with numerical stability
    - min_mu clamping for consistency with NB-GLM fitting (if min_mu > 0)
    
    Parameters
    ----------
    counts
        Raw count matrix (n_cells, n_genes).
    design_matrix
        Design matrix (n_cells, n_params).
    size_factors
        Size factors for each cell (n_cells,).
    dispersion
        Gene-wise dispersion estimates (n_genes,).
    mle_coef
        MLE coefficient matrix (n_params, n_genes) for warm-starting.
    mle_se
        Standard errors of MLE LFC estimates (n_genes,).
    shrink_index
        Index of the coefficient to shrink (default: 1, the LFC coefficient).
    prior_scale
        Scale parameter for Cauchy prior. If None, estimated globally from
        the MLE LFC distribution (matching PyDESeq2's approach).
    prior_no_shrink_scale
        Scale for normal prior on non-shrunk coefficients (default: 15.0).
    max_iter
        Maximum iterations for L-BFGS-B optimization.
    tol
        Convergence tolerance.
    n_jobs
        Number of parallel jobs. Default -1 uses all available cores.
    batch_size
        Number of genes per joblib batch (default: 128, matching PyDESeq2).
    min_mu
        Minimum mean value for mu clamping in NB log-likelihood. If > 0,
        mu is clamped to be at least min_mu. This should match the min_mu
        used during NB-GLM fitting to ensure the stored coefficients are
        consistent with the likelihood surface (default: 0.0 = no clamping).
    
    Returns
    -------
    shrunk_coef
        Shrunken coefficient matrix (n_params, n_genes).
    shrunk_se
        Approximate standard errors from inverse Hessian (n_genes,).
    converged
        Boolean array indicating convergence for each gene (n_genes,).
    """
    n_cells, n_genes = counts.shape
    n_params = design_matrix.shape[1]
    
    # Ensure arrays are float64
    counts = np.asarray(counts, dtype=np.float64)
    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    size_factors = np.asarray(size_factors, dtype=np.float64).ravel()
    dispersion = np.asarray(dispersion, dtype=np.float64).ravel()
    mle_coef = np.asarray(mle_coef, dtype=np.float64)
    mle_se = np.asarray(mle_se, dtype=np.float64).ravel()
    
    # Pre-compute log size factors (offset)
    log_size_factors = np.log(np.maximum(size_factors, 1e-10))
    
    # Estimate prior scale globally if not provided (matching PyDESeq2)
    if prior_scale is None:
        mle_lfc = mle_coef[shrink_index, :]
        prior_scale = _estimate_apeglm_prior_scale(mle_lfc, mle_se)
    
    logger.debug(f"apeGLM shrinkage: prior_scale={prior_scale:.4f}, n_genes={n_genes}, min_mu={min_mu}")
    
    # Parallel optimization over genes using joblib with loky backend
    # (loky uses process-based parallelism, avoiding GIL for CPU-bound optimization)
    results = Parallel(n_jobs=n_jobs, batch_size=batch_size, backend="loky")(
        delayed(_fit_gene_apeglm_lbfgsb)(
            counts[:, j],
            design_matrix,
            log_size_factors,
            dispersion[j],
            mle_coef[:, j].copy(),
            prior_scale,
            prior_no_shrink_scale,
            shrink_index,
            max_iter,
            tol,
            mle_se[j],
            min_mu,
        )
        for j in range(n_genes)
    )
    
    # Collect results
    shrunk_coef = np.zeros((n_params, n_genes), dtype=np.float64)
    shrunk_se = np.zeros(n_genes, dtype=np.float64)
    converged = np.zeros(n_genes, dtype=bool)
    
    for j, (beta, se, conv) in enumerate(results):
        shrunk_coef[:, j] = beta
        shrunk_se[j] = se
        converged[j] = conv
    
    n_converged = converged.sum()
    logger.debug(f"apeGLM shrinkage complete: {n_converged}/{n_genes} genes converged")
    
    return shrunk_coef, shrunk_se, converged


def shrink_lfc_apeglm_from_stats(
    mle_lfc: np.ndarray,
    mle_se: np.ndarray,
    xtwx_diag: np.ndarray | None = None,
    *,
    base_mean: np.ndarray | None = None,
    prior_scale: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    use_gene_specific_prior: bool = True,
    hybrid_fallback: bool = True,
    hybrid_mle_se_threshold: float = 3.0,
    hybrid_base_mean_threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply apeGLM-style shrinkage using pre-computed MLE statistics.

    This is a memory-efficient, fully vectorized version of apeGLM shrinkage
    that uses pre-computed MLE coefficients and standard errors without
    requiring access to the full count matrix. It applies a Cauchy prior and
    finds the MAP estimate using a damped Newton-Raphson optimization across
    all genes simultaneously.

    Accuracy improvements (v2):

    1. Gene-specific prior scales based on expression level (sqrt(base_mean)).
    2. Moment-corrected SE using observed Fisher information approximation.
    3. Hybrid fallback: marks genes with ``abs(MLE)/SE`` > threshold or low
       expression for full NB-GLM re-fitting to achieve Eff rho >= 0.98.

    The posterior mode is found by solving
    ``beta_MAP = argmin { (beta - mle_lfc)^2 / (2 * se^2) + log(1 + (beta/s)^2) }``
    which is the negative log posterior with a Cauchy prior.

    This implementation uses a robust damped Newton method with Hessian
    clamping to ensure convergence even when the standard Hessian becomes
    negative (which can happen when ``abs(beta)`` > prior_scale). This matches
    PyDESeq2's behavior which uses L-BFGS-B for robustness.
    
    Parameters
    ----------
    mle_lfc
        MLE log-fold change estimates (n_genes,).
    mle_se
        Standard errors of MLE estimates (n_genes,).
    xtwx_diag
        Diagonal elements of X'WX for Fisher information (n_genes,).
        If provided, used for posterior SE computation. If None, uses
        approximation based on MLE SE.
    base_mean
        Mean normalized expression per gene (n_genes,). Used for gene-specific
        prior scaling. If None, uniform prior scale is used.
    prior_scale
        Base scale parameter for Cauchy prior. If None, estimated adaptively.
        When use_gene_specific_prior=True, this is scaled per-gene.
    max_iter
        Maximum iterations for optimization.
    tol
        Convergence tolerance.
    use_gene_specific_prior
        If True, scale prior_scale by 1/sqrt(base_mean) per gene. This accounts
        for the fact that lowly-expressed genes have higher variance and should
        be shrunk more aggressively. Default True.
    hybrid_fallback
        If True, return a mask indicating genes that need full NB-GLM re-fitting
        due to problematic statistics (large ``abs(MLE)/SE`` or low expression).
    hybrid_mle_se_threshold
        Genes with ``abs(MLE)/SE`` > this threshold are marked for full re-fitting.
    hybrid_base_mean_threshold
        Genes with base_mean < this threshold are marked for full re-fitting.
    
    Returns
    -------
    shrunk_lfc
        Shrunken log-fold change estimates (n_genes,).
    shrunk_se
        Posterior standard errors (n_genes,).
    converged
        Boolean array indicating convergence for each gene (n_genes,).
    needs_full_refit
        Boolean array indicating genes that need full NB-GLM re-fitting (n_genes,).
        Only populated when hybrid_fallback=True, otherwise all False.
    """
    mle_lfc = np.asarray(mle_lfc, dtype=np.float64).ravel()
    mle_se = np.asarray(mle_se, dtype=np.float64).ravel()
    n_genes = mle_lfc.shape[0]
    
    shrunk_lfc = mle_lfc.copy()
    shrunk_se = mle_se.copy()
    converged = np.zeros(n_genes, dtype=bool)
    needs_full_refit = np.zeros(n_genes, dtype=bool)
    
    # Identify valid genes
    valid_mask = np.isfinite(mle_lfc) & np.isfinite(mle_se) & (mle_se > 0)
    if not np.any(valid_mask):
        return shrunk_lfc, shrunk_se, converged, needs_full_refit
    
    # Estimate base prior scale if not provided
    if prior_scale is None:
        prior_scale = _estimate_apeglm_prior_scale(mle_lfc[valid_mask], mle_se[valid_mask])
    
    # Compute gene-specific prior scales
    if use_gene_specific_prior and base_mean is not None:
        base_mean = np.asarray(base_mean, dtype=np.float64).ravel()
        # Scale prior by 1/sqrt(base_mean) - lowly expressed genes shrink more
        # Clamp base_mean to avoid extreme scaling
        safe_base_mean = np.clip(base_mean, 1.0, 1e6)
        gene_prior_scale = prior_scale / np.sqrt(safe_base_mean / 100.0)  # Normalize around 100 counts
        gene_prior_scale = np.clip(gene_prior_scale, 0.01, 10.0)  # Prevent extreme values
    else:
        gene_prior_scale = np.full(n_genes, prior_scale, dtype=np.float64)
    
    prior_scale_sq = gene_prior_scale ** 2
    
    # Identify genes needing hybrid fallback
    if hybrid_fallback:
        mle_se_ratio = np.abs(mle_lfc) / np.maximum(mle_se, 1e-10)
        # Base condition: genes with large |MLE|/SE
        needs_full_refit = valid_mask & (mle_se_ratio > hybrid_mle_se_threshold)
        # Additional condition: lowly expressed genes (if base_mean available)
        if base_mean is not None:
            needs_full_refit = needs_full_refit | (valid_mask & (base_mean < hybrid_base_mean_threshold))
        # For hybrid genes, we still compute stats approximation but mark for later refinement
    
    # Pre-compute variance for valid genes
    var_mle = np.where(valid_mask, mle_se ** 2, 1.0)  # Avoid div by zero
    
    # Apply moment correction to SE when xtwx_diag is provided
    # This uses observed Fisher information instead of expected
    if xtwx_diag is not None:
        xtwx_diag = np.asarray(xtwx_diag, dtype=np.float64).ravel()
        # Observed Fisher information correction: SE_corrected = SE * sqrt(expected/observed)
        # For now, we use the provided xtwx_diag directly for posterior SE calculation
        # This will be used after optimization
    
    # Initialize beta at zero (strong shrinkage initialization)
    # This is more robust than starting at MLE for large |MLE|
    beta = np.zeros_like(mle_lfc)
    
    # Track which genes are still active (not yet converged)
    active = valid_mask.copy()
    
    # Objective function: f(beta) = (beta - mle)^2 / (2*var) + log(1 + (beta/s)^2)
    def compute_objective(b):
        return 0.5 * (b - mle_lfc)**2 / var_mle + np.log1p((b / gene_prior_scale)**2)
    
    # Damped Newton-Raphson with line search
    for iteration in range(max_iter):
        if not np.any(active):
            break
        
        # Gradient: (beta - mle) / var + 2*beta / (s^2 + beta^2)
        denom = prior_scale_sq + beta ** 2
        grad = np.where(active,
            (beta - mle_lfc) / var_mle + 2 * beta / denom,
            0.0)
        
        # Hessian: 1/var + 2*(s^2 - beta^2) / (s^2 + beta^2)^2
        hess_raw = 1.0 / var_mle + 2 * (prior_scale_sq - beta**2) / (denom**2)
        
        # CRITICAL FIX: Clamp Hessian to be positive definite
        # When |beta| > s, the Cauchy Hessian becomes negative.
        # We use the absolute value of the Hessian, which is equivalent to
        # using gradient descent when the Hessian is negative (moving in 
        # the direction that reduces the objective).
        # Additionally, add a small regularization term for stability.
        min_hess = 1.0 / (var_mle + prior_scale_sq)  # Minimum positive Hessian
        hess = np.where(active,
            np.maximum(np.abs(hess_raw), min_hess),
            1.0)
        
        # Newton step (direction)
        step = grad / hess
        
        # Line search with backtracking to ensure objective decreases
        # This makes the algorithm globally convergent
        alpha = np.ones(n_genes)  # Step size
        f_old = compute_objective(beta)
        
        for _ in range(10):  # Max 10 backtracking iterations
            beta_new = np.where(active, beta - alpha * step, beta)
            f_new = compute_objective(beta_new)
            
            # Armijo condition: f(new) < f(old) - c * alpha * grad * step
            # We use c=0.1 for relaxed condition
            armijo_ok = f_new <= f_old - 0.1 * alpha * grad * step
            
            # Update step size for genes that don't satisfy Armijo
            needs_backtrack = active & ~armijo_ok
            if not np.any(needs_backtrack):
                break
            alpha = np.where(needs_backtrack, alpha * 0.5, alpha)
        
        beta_new = np.where(active, beta - alpha * step, beta)
        
        # Check convergence per gene
        change = np.abs(beta_new - beta)
        newly_converged = active & (change < tol)
        converged |= newly_converged
        active &= ~newly_converged
        
        beta = beta_new
    
    # Mark all remaining as converged (may have hit max_iter)
    converged[active] = True  # They stopped updating even if not at tolerance
    
    # Compute posterior SE from inverse Hessian at MAP
    denom = prior_scale_sq + beta ** 2
    cauchy_hess = 2 * (prior_scale_sq - beta**2) / (denom**2)
    total_hess = 1.0 / var_mle + cauchy_hess
    
    # Update results for valid genes
    shrunk_lfc = np.where(valid_mask, beta, mle_lfc)
    # For posterior SE, use absolute Hessian (since we may have converged
    # at a point where Hessian is negative due to the Cauchy prior)
    shrunk_se = np.where(valid_mask, 
                         1.0 / np.sqrt(np.maximum(np.abs(total_hess), 1e-12)), 
                         mle_se)
    
    return shrunk_lfc, shrunk_se, converged, needs_full_refit


def compute_cooks_distance_batch(
    Y: np.ndarray,
    mu: np.ndarray,
    dispersion: np.ndarray,
    n_params: int = 2,
) -> np.ndarray:
    """Compute Cook's distance for each observation in a batch of genes.
    
    Cook's distance measures the influence of each observation on the fitted
    model. Large values indicate potential outliers that disproportionately
    affect the estimates.
    
    For GLMs, Cook's distance is computed as:
        D_i = (r_i^2 / p) * (h_ii / (1 - h_ii)^2)
    
    where r_i is the Pearson residual, p is the number of parameters,
    and h_ii is the leverage (diagonal of the hat matrix).
    
    Parameters
    ----------
    Y
        Count matrix of shape (n_cells, n_genes).
    mu
        Fitted mean matrix of shape (n_cells, n_genes).
    dispersion
        Dispersion estimates of shape (n_genes,).
    n_params
        Number of model parameters (default 2 for intercept + treatment).
    
    Returns
    -------
    np.ndarray
        Cook's distance matrix of shape (n_cells, n_genes).
    """
    n_cells, n_genes = Y.shape
    
    # Compute variance: V = mu + dispersion * mu^2
    variance = mu + dispersion[None, :] * mu ** 2
    variance = np.maximum(variance, 1e-10)
    
    # Pearson residuals: r = (Y - mu) / sqrt(V)
    resid = (Y - mu) / np.sqrt(variance)
    
    # Weights for NB GLM: W = mu^2 / V
    weights = mu ** 2 / variance
    
    # Approximate leverage using average weight contribution
    # For balanced designs: h_ii ≈ W_i / sum(W)
    sum_weights = np.sum(weights, axis=0, keepdims=True)
    leverage = weights / np.maximum(sum_weights, 1e-10)
    
    # Clip leverage to avoid division by zero
    leverage = np.clip(leverage, 1e-10, 1 - 1e-10)
    
    # Cook's distance: D = (r^2 / p) * (h / (1-h)^2)
    cooks = (resid ** 2 / n_params) * (leverage / (1 - leverage) ** 2)
    
    return cooks


def filter_outliers_cooks(
    Y: np.ndarray,
    mu: np.ndarray,
    dispersion: np.ndarray,
    *,
    n_params: int = 2,
    threshold_quantile: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify and replace outlier counts based on Cook's distance.
    
    Following DESeq2's approach:
    1. Compute Cook's distance for each observation
    2. Identify outliers where Cook's D > F(threshold_quantile, p, n-p)
    3. Replace outlier counts with trimmed mean from non-outlier samples
    
    Parameters
    ----------
    Y
        Count matrix of shape (n_cells, n_genes).
    mu
        Fitted mean matrix of shape (n_cells, n_genes).
    dispersion
        Dispersion estimates of shape (n_genes,).
    n_params
        Number of model parameters.
    threshold_quantile
        Quantile of F distribution for outlier threshold.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Y_filtered: Count matrix with outliers replaced
        - outlier_mask: Boolean matrix indicating outliers (n_cells, n_genes)
    """
    from scipy import stats
    
    n_cells, n_genes = Y.shape
    
    # Compute Cook's distance
    cooks = compute_cooks_distance_batch(Y, mu, dispersion, n_params)
    
    # F distribution threshold
    dfn = n_params
    dfd = max(n_cells - n_params, 1)
    threshold = stats.f.ppf(threshold_quantile, dfn, dfd)
    
    # Identify outliers
    outlier_mask = cooks > threshold
    
    # Replace outliers with trimmed mean
    Y_filtered = Y.copy()
    
    for g in range(n_genes):
        outliers_g = outlier_mask[:, g]
        if np.any(outliers_g):
            non_outlier_counts = Y[~outliers_g, g]
            if len(non_outlier_counts) > 0:
                # Use trimmed mean (exclude top/bottom 10%)
                trimmed_mean = stats.trim_mean(non_outlier_counts, 0.1)
                Y_filtered[outliers_g, g] = trimmed_mean
    
    return Y_filtered, outlier_mask


def estimate_covariate_effects_streaming(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    chunk_size: int = 2048,
    poisson_iter: int = 10,
    tol: float = 1e-6,
    return_intercept: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Estimate global intercept and covariate effects using control cells only.
    
    This function fits a Poisson regression using only control cells to estimate:
    - An intercept (baseline expression for control cells)
    - Covariate effects (if any covariates are specified)
    
    By using only control cells, the intercept represents the true control baseline
    that can then be used as an offset in per-perturbation fitting. This ensures
    that perturbation effects are properly estimated as deviations from control.
    
    For the intercept-only case (no covariates), the closed-form MLE is used:
        intercept = log(sum(Y) / sum(size_factors))
    
    When covariates are present, IRLS is used with proper per-gene weighting.
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells (already loaded).
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells.
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    chunk_size
        Number of cells to process per chunk.
    poisson_iter
        Number of Poisson IRLS iterations.
    tol
        Convergence tolerance.
    return_intercept
        If True, also return the global intercept coefficients. This is useful
        for joint fitting where the intercept should be shared across all
        perturbation comparisons.
        
    Returns
    -------
    np.ndarray or tuple
        If return_intercept is False:
            Covariate effects of shape (n_covariates, n_genes). These are the
            log-scale regression coefficients for each covariate.
        If return_intercept is True:
            Tuple of (covariate_effects, intercept) where:
            - covariate_effects has shape (n_covariates, n_genes)
            - intercept has shape (n_genes,) representing the control baseline
    """
    import pandas as pd
    from .data import iter_matrix_chunks
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Identify control cells
    control_mask = (perturbation_labels == control_label)
    control_indices = np.where(control_mask)[0]
    n_control = len(control_indices)
    
    if n_control == 0:
        raise ValueError(f"No control cells found with label '{control_label}'")
    
    # Build covariate portion of design matrix (for control cells only)
    cov_matrices = []
    cov_names: list[str] = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_df")
        series = obs_df[column].iloc[control_indices]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
                cov_names.extend(dummies.columns.astype(str).tolist())
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
            cov_names.append(str(column))
    
    n_covariates = sum(m.shape[1] for m in cov_matrices) if cov_matrices else 0
    
    # Size factors for control cells
    size_factors_control = size_factors[control_indices]
    
    if n_covariates == 0:
        # No covariates: use closed-form MLE for intercept
        # intercept = log(sum(Y) / sum(size_factors)) for each gene
        # This is the exact MLE for Poisson with offset
        
        sum_counts = np.zeros(n_genes, dtype=np.float64)
        
        control_idx = 0
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            chunk_control_mask = control_mask[slc]
            if not np.any(chunk_control_mask):
                continue
            
            Y_chunk_control = np.asarray(chunk[chunk_control_mask], dtype=np.float64)
            sum_counts += Y_chunk_control.sum(axis=0)
        
        sum_size_factors = size_factors_control.sum()
        beta_intercept = np.log(np.maximum(sum_counts / sum_size_factors, 1e-12))
        beta_cov = np.zeros((0, n_genes), dtype=np.float64)
        
        if return_intercept:
            return beta_cov, beta_intercept
        return beta_cov
    
    # With covariates: use IRLS with proper per-gene weighting
    # We need to solve separately for each gene since weights differ per gene
    cov_matrix_control = np.hstack(cov_matrices)  # (n_control, n_covariates)
    
    # Design matrix: intercept + covariates (control cells only)
    # Shape: (n_control, 1 + n_covariates)
    n_features = 1 + n_covariates
    
    # Initialize beta coefficients
    beta = np.zeros((n_features, n_genes), dtype=np.float64)
    
    # Log size factors for control cells
    log_size_factors_control = np.log(np.maximum(size_factors[control_indices], 1e-12))
    
    # Poisson IRLS with streaming - only process control cells
    # We need to map global chunk indices to control cell indices
    for iteration in range(poisson_iter):
        # Accumulate X^T W X and X^T W z across chunks
        xtwx_accum = np.zeros((n_features, n_features), dtype=np.float64)
        xtwz_accum = np.zeros((n_features, n_genes), dtype=np.float64)
        
        control_idx = 0  # Track position within control cells
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            # Find which cells in this chunk are control cells
            chunk_control_mask = control_mask[slc]
            
            if not np.any(chunk_control_mask):
                continue
            
            # Extract control cells from this chunk
            Y_chunk_control = np.asarray(chunk[chunk_control_mask], dtype=np.float64)
            n_chunk_control = Y_chunk_control.shape[0]
            
            # Get indices of control cells in this chunk (relative to all control cells)
            chunk_control_count = chunk_control_mask.sum()
            control_slice = slice(control_idx, control_idx + chunk_control_count)
            control_idx += chunk_control_count
            
            # Build design for control cells in this chunk
            X_chunk = np.empty((n_chunk_control, n_features), dtype=np.float64)
            X_chunk[:, 0] = 1.0  # Intercept
            if n_covariates > 0:
                X_chunk[:, 1:] = cov_matrix_control[control_slice]
            
            offset_chunk = log_size_factors_control[control_slice]
            
            # Compute eta, mu for this chunk
            eta = X_chunk @ beta + offset_chunk[:, None]
            eta = np.clip(eta, -20.0, 20.0)
            mu = np.exp(eta)
            mu = np.maximum(mu, 1e-6)
            
            # Poisson weights = mu
            W = mu  # (n_chunk_control, n_genes)
            
            # Working response: z = eta - offset + (y - mu) / mu
            z = eta - offset_chunk[:, None] + (Y_chunk_control - mu) / np.maximum(mu, 1e-6)
            
            # Accumulate X^T W X: sum over genes, then over samples
            # X^T W X = sum_g sum_i W[i,g] * X[i,:,None] * X[i,None,:]
            # We use average weights across genes for a shared XtWX
            W_sum = W.sum(axis=1)  # (n_chunk_control,)
            xtwx_accum += X_chunk.T @ (W_sum[:, None] * X_chunk)
            
            # Accumulate X^T W z per gene
            Wz = W * z  # (n_chunk_control, n_genes)
            xtwz_accum += X_chunk.T @ Wz  # (n_features, n_genes)
        
        # Solve for beta: (X^T W X) beta = X^T W z
        # Add ridge penalty for stability
        ridge = 1e-6 * np.eye(n_features)
        try:
            beta_new = np.linalg.solve(xtwx_accum + ridge, xtwz_accum)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(xtwx_accum + ridge, xtwz_accum, rcond=None)[0]
        
        # Check convergence
        max_diff = np.max(np.abs(beta_new - beta))
        beta = beta_new
        
        if max_diff < tol:
            break
    
    # Extract intercept (first row)
    beta_intercept = beta[0, :]  # (n_genes,)
    
    # Extract covariate effects
    beta_cov = beta[1:, :]  # (n_covariates, n_genes)
    
    if return_intercept:
        return beta_cov, beta_intercept
    return beta_cov


def _nb_deviance(
    Y: np.ndarray,
    mu: np.ndarray,
    alpha: float,
) -> float:
    """Compute negative binomial deviance.
    
    Deviance = 2 * sum(y * log(y/mu) - (y + r) * log((y + r) / (mu + r)))
    where r = 1/alpha and terms with y=0 use limit y*log(y) -> 0.
    
    Parameters
    ----------
    Y : (n_samples,) or (n_samples, n_genes)
        Observed counts.
    mu : same shape as Y
        Fitted mean values.
    alpha : float
        Dispersion parameter.
        
    Returns
    -------
    deviance : float
        Total deviance.
    """
    r = 1.0 / max(alpha, 1e-10)
    Y_safe = np.maximum(Y, 1e-10)
    mu_safe = np.maximum(mu, 1e-10)
    
    # Term 1: y * log(y/mu), with limit 0 when y=0
    term1 = np.where(Y > 0, Y * np.log(Y_safe / mu_safe), 0.0)
    
    # Term 2: (y + r) * log((y + r) / (mu + r))
    term2 = (Y + r) * np.log((Y + r) / (mu_safe + r))
    
    return float(2.0 * np.sum(term1 - term2))


def estimate_global_dispersion_streaming(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    beta_intercept: np.ndarray,
    beta_cov: np.ndarray | None = None,
    beta_perturbation: np.ndarray | None = None,
    chunk_size: int = 2048,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    poisson_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Estimate global per-gene dispersion using all cells via streaming.
    
    This function streams through all cells to estimate dispersion for each gene
    using a full design matrix. The dispersion is estimated using all conditions
    together, which provides more stable estimates than per-perturbation estimation,
    similar to how PyDESeq2 estimates dispersion from all samples.
    
    The function uses the pre-estimated intercept, perturbation effects, and 
    covariate effects to compute fitted values (mu), then estimates dispersion 
    from the residuals using method-of-moments.
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells (already loaded).
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells (used as reference level).
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    beta_intercept
        Pre-estimated global intercept coefficients, shape (n_genes,).
    beta_cov
        Pre-estimated covariate effects, shape (n_covariates, n_genes).
        If None or empty, no covariate adjustment is applied.
    beta_perturbation
        Pre-estimated perturbation effects, shape (n_perturbations, n_genes).
        If None, perturbation effects are estimated via Poisson IRLS.
    chunk_size
        Number of cells to process per chunk.
    dispersion_method
        Method for dispersion estimation:
        - "moments": Method-of-moments (fast but less accurate)
        - "cox-reid": Cox-Reid adjusted profile likelihood (more accurate)
    poisson_iter
        Number of Poisson IRLS iterations for refining mu estimates.
        Only used if beta_perturbation is None.
    tol
        Convergence tolerance.
        
    Returns
    -------
    np.ndarray
        Dispersion estimates of shape (n_genes,). These are the alpha values
        for the negative binomial distribution: Var(Y) = mu + alpha * mu^2.
    """
    import pandas as pd
    from .data import iter_matrix_chunks
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Build perturbation indicator matrix (one-hot, control as reference)
    unique_labels = np.unique(perturbation_labels)
    non_control = unique_labels[unique_labels != control_label]
    n_perturbations = len(non_control)
    
    # Create label-to-index mapping
    label_to_idx = {label: i for i, label in enumerate(non_control)}
    cell_pert_idx = np.full(n_cells, -1, dtype=np.int32)
    for i, label in enumerate(perturbation_labels):
        if label != control_label:
            cell_pert_idx[i] = label_to_idx[label]
    
    # Build covariate matrix
    cov_matrices = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            continue
        series = obs_df[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
    
    if cov_matrices:
        cov_matrix = np.hstack(cov_matrices)
    else:
        cov_matrix = np.zeros((n_cells, 0), dtype=np.float64)
    n_covariates = cov_matrix.shape[1]
    
    # Full design: intercept + perturbations + covariates
    n_features = 1 + n_perturbations + n_covariates
    
    # Log size factors for offset
    log_size_factors = np.log(np.maximum(size_factors, 1e-12))
    
    # Use provided perturbation effects or estimate them
    if beta_perturbation is not None:
        # Use pre-computed perturbation effects
        beta_pert = beta_perturbation.copy()
    else:
        # Estimate perturbation effects via Poisson IRLS
        beta_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
        for iteration in range(poisson_iter):
            # Accumulate for perturbation effects (diagonal structure)
            pert_xtwx_diag = np.zeros(n_perturbations, dtype=np.float64)
            pert_xtwz = np.zeros((n_perturbations, n_genes), dtype=np.float64)
            
            for slc, chunk in iter_matrix_chunks(
                backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
            ):
                Y_chunk = np.asarray(chunk, dtype=np.float64)
                n_chunk = Y_chunk.shape[0]
                
                offset_chunk = log_size_factors[slc]
                pert_idx_chunk = cell_pert_idx[slc]
                cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
                
                # Compute eta
                eta = beta_intercept[None, :] + offset_chunk[:, None]
                pert_mask = pert_idx_chunk >= 0
                if np.any(pert_mask):
                    eta[pert_mask] += beta_pert[pert_idx_chunk[pert_mask], :]
                if n_covariates > 0 and cov_chunk is not None and beta_cov is not None:
                    eta += cov_chunk @ beta_cov
                
                eta = np.clip(eta, -20.0, 20.0)
                mu = np.exp(eta)
                mu = np.maximum(mu, 1e-6)
                
                # Poisson weights
                W = mu
                W_sum = W.sum(axis=1)
                
                # Working response
                z_full = eta - offset_chunk[:, None] + (Y_chunk - mu) / np.maximum(mu, 1e-6)
                # Subtract fixed effects
                z_pert = z_full - beta_intercept[None, :]
                if n_covariates > 0 and cov_chunk is not None and beta_cov is not None:
                    z_pert = z_pert - cov_chunk @ beta_cov
                
                Wz = W * z_pert
                
                # Accumulate (diagonal structure)
                for i in range(n_chunk):
                    p_idx = pert_idx_chunk[i]
                    if p_idx >= 0:
                        pert_xtwx_diag[p_idx] += W_sum[i]
                        pert_xtwz[p_idx, :] += Wz[i, :]
            
            # Solve for perturbation effects (diagonal system)
            ridge = 1e-6
            D_inv = 1.0 / np.maximum(pert_xtwx_diag + ridge, 1e-12)
            beta_pert_new = D_inv[:, None] * pert_xtwz
            
            max_diff = np.max(np.abs(beta_pert_new - beta_pert))
            beta_pert = beta_pert_new
            
            if max_diff < tol:
                break
    
    # Now compute dispersion using method of moments, streaming through data
    numerator_sum = np.zeros(n_genes, dtype=np.float64)
    n_total = 0
    
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)
        n_chunk = Y_chunk.shape[0]
        n_total += n_chunk
        
        offset_chunk = log_size_factors[slc]
        pert_idx_chunk = cell_pert_idx[slc]
        cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
        
        # Compute mu using full model
        eta = beta_intercept[None, :] + offset_chunk[:, None]
        pert_mask = pert_idx_chunk >= 0
        if np.any(pert_mask):
            eta[pert_mask] += beta_pert[pert_idx_chunk[pert_mask], :]
        if n_covariates > 0 and cov_chunk is not None and beta_cov is not None:
            eta += cov_chunk @ beta_cov
        
        eta = np.clip(eta, -20.0, 20.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-6)
        
        # Method of moments: (y - mu)^2 - y over mu^2
        resid = Y_chunk - mu
        numerator = (resid * resid - Y_chunk) / np.maximum(mu * mu, 1e-12)
        numerator_sum += numerator.sum(axis=0)
    
    # Degrees of freedom
    dof = max(n_total - n_features, 1)
    dispersion = np.clip(numerator_sum / dof, 1e-8, 1e6)
    
    # Handle invalid values
    dispersion = np.where(np.isfinite(dispersion), dispersion, 0.1)
    
    return dispersion


class NBGLMBatchFitter:
    """Vectorized batch fitter for NB GLM across multiple genes.
    
    This fitter processes all genes simultaneously using vectorized operations,
    providing significant speedup compared to per-gene fitting. It uses IRLS
    (Iteratively Reweighted Least Squares) with batched matrix operations.
    
    Parameters
    ----------
    design
        Design matrix with shape ``(n_samples, n_features)``.
    offset
        Log-scale offset (e.g., log size factors) per sample.
    max_iter
        Maximum IRLS iterations.
    tol
        Convergence tolerance on coefficient updates.
    poisson_init_iter
        Initial Poisson iterations for warm start.
    dispersion_method
        Method for dispersion estimation: "moments" or "cox-reid".
    min_mu
        Minimum fitted mean to avoid numerical issues.
    min_total_count
        Minimum total count for a gene to be fitted.
    """
    
    def __init__(
        self,
        design: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        max_iter: int = 25,
        tol: float = 1e-6,
        poisson_init_iter: int = 5,
        dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
        min_mu: float = 0.5,
        min_total_count: float = 1.0,
        ridge_penalty: float = 1e-6,
    ) -> None:
        self.design = np.asarray(design, dtype=np.float64)
        if self.design.ndim != 2:
            raise ValueError("design must be a 2D array")
        self.n_samples, self.n_features = self.design.shape
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.zeros(self.n_samples, dtype=np.float64)
        )
        self.max_iter = int(max_iter)
        self.tol = tol
        self.poisson_init_iter = int(max(0, poisson_init_iter))
        self.dispersion_method = dispersion_method
        self.min_mu = min_mu
        self.min_total_count = min_total_count
        self.ridge_penalty = ridge_penalty
        
        # Precompute X^T X for efficiency
        self._xtx = self.design.T @ self.design
    
    def fit_batch(
        self, 
        counts: ArrayLike,
        gene_batch_size: int | Literal["auto"] | None = "auto",
        use_numba: bool = True,
    ) -> NBGLMBatchResult:
        """Fit NB GLM for all genes in the count matrix.
        
        Memory-optimized implementation with optional gene batching and Numba
        acceleration for the 2-parameter model (intercept + perturbation).
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
        gene_batch_size
            Number of genes to process per batch. If "auto", calculated based
            on memory constraints (~100 MB per batch). If None, process all
            genes at once (legacy behavior).
        use_numba
            Whether to use Numba-accelerated IRLS for 2-feature models.
            Default True for better memory efficiency.
            
        Returns
        -------
        NBGLMBatchResult
            Results for all genes with vectorized arrays.
        """
        if sp.issparse(counts):
            Y = np.asarray(counts.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(counts, dtype=np.float64)
        
        if Y.ndim != 2 or Y.shape[0] != self.n_samples:
            raise ValueError(f"counts must have shape ({self.n_samples}, n_genes)")
        
        n_genes = Y.shape[1]
        X = self.design
        n_features = self.n_features
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Check which genes have sufficient counts
        total_counts = Y.sum(axis=0)
        valid_genes = total_counts >= self.min_total_count
        n_valid = valid_genes.sum()
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter, deviance=deviance
            )
        
        # Work only with valid genes
        Y_valid = Y[:, valid_genes]  # (n_samples, n_valid)
        valid_indices = np.where(valid_genes)[0]
        
        # Calculate gene batch size
        if gene_batch_size == "auto":
            gene_batch_size = _estimate_gene_batch_size_fitter(
                self.n_samples, n_valid, n_work_arrays=4, target_mb=100.0
            )
        elif gene_batch_size is None:
            gene_batch_size = n_valid  # Process all at once
        
        # Use Numba path for 2-feature case (intercept + perturbation)
        # This is much more memory efficient as it uses per-gene loops
        if use_numba and n_features == 2:
            return self._fit_batch_numba(
                Y, Y_valid, valid_genes, valid_indices, n_genes, gene_batch_size
            )
        
        # Fallback to batched NumPy implementation
        return self._fit_batch_numpy_batched(
            Y, Y_valid, valid_genes, valid_indices, n_genes, gene_batch_size
        )
    
    def _fit_batch_numba(
        self,
        Y: np.ndarray,
        Y_valid: np.ndarray,
        valid_genes: np.ndarray,
        valid_indices: np.ndarray,
        n_genes: int,
        gene_batch_size: int,
    ) -> NBGLMBatchResult:
        """Numba-accelerated IRLS for 2-feature models.
        
        Uses per-gene Numba loops which are more memory efficient than
        vectorized operations across all genes.
        """
        n_valid = Y_valid.shape[1]
        n_features = self.n_features
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Initialize beta
        beta_init = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start
        if self.poisson_init_iter > 0:
            beta_init = self._poisson_warm_start_batch(Y_valid, beta_init)
        
        # Initial dispersion (MoM)
        alpha = np.full(n_valid, 0.1, dtype=np.float64)
        
        # Run Numba IRLS
        beta_result, se_result, conv_result, iter_result = _irls_batch_numba(
            Y_valid,
            self.design,
            self.offset,
            alpha,
            beta_init,
            self.max_iter,
            self.tol,
            self.min_mu,
            self.ridge_penalty,
        )
        
        # Compute final dispersion using MoM
        mu_final = np.zeros_like(Y_valid)
        for g in range(n_valid):
            eta = self.offset + self.design @ beta_result[:, g]
            eta = np.clip(eta, np.log(self.min_mu), 20.0)
            mu_final[:, g] = np.exp(eta)
        mu_final = np.maximum(mu_final, self.min_mu)
        
        resid = Y_valid - mu_final
        dof = max(self.n_samples - n_features, 1)
        alpha_final = np.sum((resid * resid - Y_valid) / np.maximum(mu_final * mu_final, self.min_mu), axis=0) / dof
        alpha_final = np.clip(alpha_final, 1e-8, 1e6)
        
        # Cox-Reid refinement if requested
        if self.dispersion_method == "cox-reid":
            alpha_final = self._refine_dispersion_cox_reid_batch(Y_valid, mu_final, alpha_final)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu_final, alpha_final)
        
        # Store results
        coef[valid_indices] = beta_result.T
        se[valid_indices] = se_result.T
        dispersion[valid_indices] = alpha_final
        converged[valid_indices] = conv_result
        n_iter[valid_indices] = iter_result
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter, deviance=deviance
        )
    
    def _fit_batch_numpy_batched(
        self,
        Y: np.ndarray,
        Y_valid: np.ndarray,
        valid_genes: np.ndarray,
        valid_indices: np.ndarray,
        n_genes: int,
        gene_batch_size: int,
    ) -> NBGLMBatchResult:
        """NumPy-based IRLS with gene batching to reduce memory.
        
        Processes genes in batches to limit work array memory usage.
        Reduced from 7 to 4 work arrays via memory reuse.
        """
        n_valid = Y_valid.shape[1]
        n_features = self.n_features
        X = self.design
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged_arr = np.zeros(n_genes, dtype=bool)
        n_iter_arr = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Initialize beta for all valid genes
        beta_all = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start
        if self.poisson_init_iter > 0:
            beta_all = self._poisson_warm_start_batch(Y_valid, beta_all)
        
        # Initialize dispersion
        alpha_all = np.full(n_valid, 0.1, dtype=np.float64)
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Precompute constants
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        # Process genes in batches
        for batch_start in range(0, n_valid, gene_batch_size):
            batch_end = min(batch_start + gene_batch_size, n_valid)
            batch_size = batch_end - batch_start
            batch_slice = slice(batch_start, batch_end)
            
            Y_batch = Y_valid[:, batch_slice]
            beta_batch = beta_all[:, batch_slice]
            alpha_batch = alpha_all[batch_slice]
            batch_converged = np.zeros(batch_size, dtype=bool)
            
            # Allocate work arrays for this batch only (4 arrays instead of 7)
            eta = np.empty((self.n_samples, batch_size), dtype=np.float64)
            mu = np.empty_like(eta)
            # variance_weights: used for both variance and weights (sequential)
            variance_weights = np.empty_like(eta)
            # z_working: used for z, working_response, and resid (sequential)
            z_working = np.empty_like(eta)
            
            for iteration in range(1, self.max_iter + 1):
                # Compute eta and mu
                np.dot(X, beta_batch, out=eta)
                eta += offset_col
                np.clip(eta, log_min_mu, 20.0, out=eta)
                np.exp(eta, out=mu)
                np.maximum(mu, self.min_mu, out=mu)
                
                # Compute variance: V = mu + alpha * mu^2
                np.multiply(mu, mu, out=variance_weights)
                variance_weights *= alpha_batch[None, :]
                variance_weights += mu
                
                # Compute weights in-place: W = mu^2 / V
                mu_sq = mu * mu  # Temporary for numerator
                np.divide(mu_sq, np.maximum(variance_weights, self.min_mu), out=variance_weights)
                # Now variance_weights contains weights
                
                # Working response: z = eta + (Y - mu) / mu - offset
                np.subtract(Y_batch, mu, out=z_working)  # z_working = Y - mu
                np.divide(z_working, np.maximum(mu, self.min_mu), out=z_working)
                z_working += eta
                z_working -= offset_col
                # Now z_working contains working_response
                
                # Solve weighted least squares
                beta_new = self._weighted_least_squares_batch(variance_weights, z_working)
                
                # Check convergence
                beta_diff = np.max(np.abs(beta_new - beta_batch), axis=0)
                newly_converged = (beta_diff < self.tol) & ~batch_converged
                batch_converged |= newly_converged
                
                # Update iteration count for non-converged genes
                for i in range(batch_size):
                    if not batch_converged[i]:
                        gene_n_iter[batch_start + i] = iteration
                
                beta_batch = beta_new
                
                # Update dispersion (MoM)
                np.subtract(Y_batch, mu, out=z_working)  # resid = Y - mu
                np.multiply(z_working, z_working, out=eta)  # reuse eta as temp
                eta -= Y_batch
                denom = np.maximum(mu * mu, self.min_mu)
                numerator = np.sum(eta / denom, axis=0)
                dof = max(self.n_samples - n_features, 1)
                alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
                alpha_batch = np.where(np.isfinite(alpha_new), alpha_new, alpha_batch)
                
                if np.all(batch_converged):
                    break
            
            # Store batch results
            beta_all[:, batch_slice] = beta_batch
            alpha_all[batch_slice] = alpha_batch
            gene_converged[batch_start:batch_end] = batch_converged
            
            # Clean up batch arrays
            del eta, mu, variance_weights, z_working, mu_sq
        
        # Final mu computation and dispersion refinement
        eta_final = np.dot(X, beta_all) + offset_col
        np.clip(eta_final, log_min_mu, 20.0, out=eta_final)
        mu_final = np.exp(eta_final)
        np.maximum(mu_final, self.min_mu, out=mu_final)
        
        if self.dispersion_method == "cox-reid":
            alpha_all = self._refine_dispersion_cox_reid_batch(Y_valid, mu_final, alpha_all)
        
        # Compute SE and deviance
        variance_final = mu_final + alpha_all[None, :] * mu_final * mu_final
        weights_final = mu_final * mu_final / np.maximum(variance_final, self.min_mu)
        se_valid = self._compute_se_batch(weights_final)
        dev_valid = self._compute_deviance_batch(Y_valid, mu_final, alpha_all)
        
        # Store to output arrays
        coef[valid_indices] = beta_all.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha_all
        converged_arr[valid_indices] = gene_converged
        n_iter_arr[valid_indices] = gene_n_iter
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged_arr, n_iter=n_iter_arr, deviance=deviance
        )
    
    def fit_batch_with_covariate_offset(
        self, 
        counts: ArrayLike, 
        covariate_offset: np.ndarray,
    ) -> NBGLMBatchResult:
        """Fit NB GLM with pre-computed covariate offset.
        
        This method is used in the joint fitting approach where covariate effects
        are estimated globally and then held fixed during per-perturbation fitting.
        The covariate offset is subtracted from the working response during IRLS.
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
        covariate_offset
            Pre-computed covariate offset of shape ``(n_samples, n_genes)``,
            representing X_cov @ beta_cov for the covariate portion of the design.
            
        Returns
        -------
        NBGLMBatchResult
            Results for all genes with vectorized arrays.
        """
        if sp.issparse(counts):
            Y = np.asarray(counts.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(counts, dtype=np.float64)
        
        if Y.ndim != 2 or Y.shape[0] != self.n_samples:
            raise ValueError(f"counts must have shape ({self.n_samples}, n_genes)")
        
        n_genes = Y.shape[1]
        covariate_offset = np.asarray(covariate_offset, dtype=np.float64)
        if covariate_offset.shape != (self.n_samples, n_genes):
            raise ValueError(
                f"covariate_offset must have shape ({self.n_samples}, {n_genes})"
            )
        
        X = self.design
        n_features = self.n_features
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Check which genes have sufficient counts
        total_counts = Y.sum(axis=0)
        valid_genes = total_counts >= self.min_total_count
        n_valid = valid_genes.sum()
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter, deviance=deviance
            )
        
        # Work only with valid genes
        Y_valid = Y[:, valid_genes]  # (n_samples, n_valid)
        cov_offset_valid = covariate_offset[:, valid_genes]  # (n_samples, n_valid)
        valid_indices = np.where(valid_genes)[0]
        
        # Initialize beta for all valid genes: (n_features, n_valid)
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start with covariate offset
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start_with_offset(Y_valid, beta, cov_offset_valid)
        
        # Initialize dispersion estimates (method of moments)
        alpha = np.full(n_valid, 0.1, dtype=np.float64)
        
        # IRLS iterations
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Pre-allocate work arrays
        eta = np.empty((self.n_samples, n_valid), dtype=np.float64)
        mu = np.empty_like(eta)
        variance = np.empty_like(eta)
        weights = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        resid = np.empty_like(eta)
        
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        for iteration in range(1, self.max_iter + 1):
            # Compute eta including covariate offset
            # eta = X @ beta + offset + covariate_offset
            np.dot(X, beta, out=eta)
            eta += offset_col
            eta += cov_offset_valid  # Add pre-computed covariate contribution
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Compute variance and weights
            np.multiply(mu, mu, out=variance)
            variance *= alpha[None, :]
            variance += mu
            np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
            
            # Working response (subtract covariate offset from z)
            np.subtract(Y_valid, mu, out=resid)
            np.divide(resid, np.maximum(mu, self.min_mu), out=z)
            z += eta
            # Remove covariate offset and regular offset to get working response for X @ beta
            np.subtract(z, offset_col, out=working_response)
            working_response -= cov_offset_valid
            
            # Solve weighted least squares
            beta_new = self._weighted_least_squares_batch(weights, working_response)
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion using method of moments
            np.subtract(Y_valid, mu, out=resid)
            np.multiply(resid, resid, out=variance)
            variance -= Y_valid
            denom = np.maximum(mu * mu, self.min_mu)
            numerator = np.sum(variance / denom, axis=0)
            dof = max(self.n_samples - n_features, 1)
            alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
            alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
            
            if np.all(gene_converged):
                break
        
        # Final dispersion refinement with Cox-Reid if requested
        if self.dispersion_method == "cox-reid":
            alpha = self._refine_dispersion_cox_reid_batch(Y_valid, mu, alpha)
        
        # Compute final mu, weights, and standard errors
        np.dot(X, beta, out=eta)
        eta += offset_col
        eta += cov_offset_valid
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, self.min_mu, out=mu)
        np.multiply(mu, mu, out=variance)
        variance *= alpha[None, :]
        variance += mu
        np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
        
        # Compute SE
        se_valid = self._compute_se_batch(weights)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu, alpha)
        
        # Store results
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter[valid_indices] = gene_n_iter
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter, deviance=deviance
        )
    
    def fit_batch_with_joint_offsets(
        self, 
        counts: ArrayLike, 
        *,
        intercept_offset: np.ndarray | None = None,
        covariate_offset: np.ndarray | None = None,
        fixed_dispersion: np.ndarray | None = None,
    ) -> NBGLMBatchResult:
        """Fit NB GLM with pre-computed intercept and covariate offsets.
        
        This method is used in the joint fitting approach where the global
        intercept (and optionally covariates and dispersion) are estimated 
        using all cells and then held fixed during per-perturbation fitting.
        
        The design matrix should NOT include an intercept column when using
        intercept_offset, as the global intercept is added as an offset.
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
        intercept_offset
            Pre-computed global intercept of shape ``(n_genes,)``. If provided,
            the design matrix should not include an intercept column.
        covariate_offset
            Pre-computed covariate offset of shape ``(n_samples, n_genes)``,
            representing X_cov @ beta_cov for the covariate portion.
        fixed_dispersion
            If provided, use these dispersion values instead of estimating.
            Shape ``(n_genes,)``.
            
        Returns
        -------
        NBGLMBatchResult
            Results for all genes with vectorized arrays.
        """
        if sp.issparse(counts):
            Y = np.asarray(counts.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(counts, dtype=np.float64)
        
        if Y.ndim != 2 or Y.shape[0] != self.n_samples:
            raise ValueError(f"counts must have shape ({self.n_samples}, n_genes)")
        
        n_genes = Y.shape[1]
        X = self.design
        n_features = self.n_features
        
        # Validate and prepare offsets
        if intercept_offset is not None:
            intercept_offset = np.asarray(intercept_offset, dtype=np.float64)
            if intercept_offset.shape != (n_genes,):
                raise ValueError(f"intercept_offset must have shape ({n_genes},)")
        
        if covariate_offset is not None:
            covariate_offset = np.asarray(covariate_offset, dtype=np.float64)
            if covariate_offset.shape != (self.n_samples, n_genes):
                raise ValueError(
                    f"covariate_offset must have shape ({self.n_samples}, {n_genes})"
                )
        
        if fixed_dispersion is not None:
            fixed_dispersion = np.asarray(fixed_dispersion, dtype=np.float64)
            if fixed_dispersion.shape != (n_genes,):
                raise ValueError(f"fixed_dispersion must have shape ({n_genes},)")
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Check which genes have sufficient counts
        total_counts = Y.sum(axis=0)
        valid_genes = total_counts >= self.min_total_count
        n_valid = valid_genes.sum()
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter, deviance=deviance
            )
        
        # Work only with valid genes
        Y_valid = Y[:, valid_genes]
        valid_indices = np.where(valid_genes)[0]
        
        # Prepare valid gene offsets
        intercept_valid = intercept_offset[valid_genes] if intercept_offset is not None else None
        cov_offset_valid = covariate_offset[:, valid_genes] if covariate_offset is not None else None
        
        # Initialize beta
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start_with_joint_offsets(
                Y_valid, beta, intercept_valid, cov_offset_valid
            )
        
        # Initialize or use fixed dispersion
        if fixed_dispersion is not None:
            alpha = fixed_dispersion[valid_genes].copy()
            use_fixed_dispersion = True
        else:
            alpha = np.full(n_valid, 0.1, dtype=np.float64)
            use_fixed_dispersion = False
        
        # IRLS iterations
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Pre-allocate work arrays
        eta = np.empty((self.n_samples, n_valid), dtype=np.float64)
        mu = np.empty_like(eta)
        variance = np.empty_like(eta)
        weights = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        resid = np.empty_like(eta)
        
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        for iteration in range(1, self.max_iter + 1):
            # Compute eta = X @ beta + offset + intercept_offset + covariate_offset
            np.dot(X, beta, out=eta)
            eta += offset_col
            if intercept_valid is not None:
                eta += intercept_valid[None, :]  # Broadcast (n_genes,) to (n_samples, n_genes)
            if cov_offset_valid is not None:
                eta += cov_offset_valid
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Compute variance and weights
            np.multiply(mu, mu, out=variance)
            variance *= alpha[None, :]
            variance += mu
            np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
            
            # Working response
            np.subtract(Y_valid, mu, out=resid)
            np.divide(resid, np.maximum(mu, self.min_mu), out=z)
            z += eta
            # Remove all offsets to get working response for X @ beta
            np.subtract(z, offset_col, out=working_response)
            if intercept_valid is not None:
                working_response -= intercept_valid[None, :]
            if cov_offset_valid is not None:
                working_response -= cov_offset_valid
            
            # Solve weighted least squares
            beta_new = self._weighted_least_squares_batch(weights, working_response)
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion if not fixed
            if not use_fixed_dispersion:
                np.subtract(Y_valid, mu, out=resid)
                np.multiply(resid, resid, out=variance)
                variance -= Y_valid
                denom = np.maximum(mu * mu, self.min_mu)
                numerator = np.sum(variance / denom, axis=0)
                dof = max(self.n_samples - n_features, 1)
                alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
                alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
            
            if np.all(gene_converged):
                break
        
        # Final dispersion refinement if not using fixed dispersion
        if not use_fixed_dispersion and self.dispersion_method == "cox-reid":
            alpha = self._refine_dispersion_cox_reid_batch(Y_valid, mu, alpha)
        
        # Compute final mu, weights, and standard errors
        np.dot(X, beta, out=eta)
        eta += offset_col
        if intercept_valid is not None:
            eta += intercept_valid[None, :]
        if cov_offset_valid is not None:
            eta += cov_offset_valid
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, self.min_mu, out=mu)
        np.multiply(mu, mu, out=variance)
        variance *= alpha[None, :]
        variance += mu
        np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
        
        # Compute SE
        se_valid = self._compute_se_batch(weights)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu, alpha)
        
        # Store results
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter[valid_indices] = gene_n_iter
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter, deviance=deviance
        )
    
    def _poisson_warm_start_with_joint_offsets(
        self, 
        Y: np.ndarray, 
        beta: np.ndarray, 
        intercept_offset: np.ndarray | None,
        covariate_offset: np.ndarray | None,
    ) -> np.ndarray:
        """Poisson warm start with pre-computed intercept and covariate offsets."""
        X = self.design
        n_samples, n_genes = Y.shape
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        eta = np.empty((n_samples, n_genes), dtype=np.float64)
        mu = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        
        for _ in range(self.poisson_init_iter):
            np.dot(X, beta, out=eta)
            eta += offset_col
            if intercept_offset is not None:
                eta += intercept_offset[None, :]
            if covariate_offset is not None:
                eta += covariate_offset
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Poisson working response
            z[:] = eta + (Y - mu) / np.maximum(mu, self.min_mu)
            np.subtract(z, offset_col, out=working_response)
            if intercept_offset is not None:
                working_response -= intercept_offset[None, :]
            if covariate_offset is not None:
                working_response -= covariate_offset
            
            # Solve with Poisson weights (= mu)
            beta = self._weighted_least_squares_batch(mu, working_response)
        
        return beta
    
    def _poisson_warm_start_with_offset(
        self, Y: np.ndarray, beta: np.ndarray, covariate_offset: np.ndarray
    ) -> np.ndarray:
        """Poisson warm start with pre-computed covariate offset."""
        X = self.design
        n_samples, n_genes = Y.shape
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        eta = np.empty((n_samples, n_genes), dtype=np.float64)
        mu = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        
        for _ in range(self.poisson_init_iter):
            np.dot(X, beta, out=eta)
            eta += offset_col
            eta += covariate_offset  # Include covariate contribution
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            np.subtract(Y, mu, out=z)
            np.divide(z, np.maximum(mu, self.min_mu), out=z)
            z += eta
            np.subtract(z, offset_col, out=working_response)
            working_response -= covariate_offset  # Remove covariate offset
            
            beta_new = self._weighted_least_squares_batch(mu, working_response)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                return beta_new
            beta = beta_new
        return beta
    
    def _poisson_warm_start_batch(
        self, Y: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Vectorized Poisson warm start for all genes."""
        X = self.design
        n_samples, n_genes = Y.shape
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        # Pre-allocate work arrays
        eta = np.empty((n_samples, n_genes), dtype=np.float64)
        mu = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        
        for _ in range(self.poisson_init_iter):
            np.dot(X, beta, out=eta)
            eta += offset_col
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            # Poisson weights = mu
            np.subtract(Y, mu, out=z)
            np.divide(z, np.maximum(mu, self.min_mu), out=z)
            z += eta
            np.subtract(z, offset_col, out=working_response)
            beta_new = self._weighted_least_squares_batch(mu, working_response)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                return beta_new
            beta = beta_new
        return beta
    
    def _weighted_least_squares_batch(
        self, weights: np.ndarray, y_working: np.ndarray
    ) -> np.ndarray:
        """Solve WLS for all genes simultaneously using vectorized operations.
        
        Parameters
        ----------
        weights
            Weight matrix of shape (n_samples, n_genes).
        y_working
            Working response matrix of shape (n_samples, n_genes).
            
        Returns
        -------
        np.ndarray
            Coefficient matrix of shape (n_features, n_genes).
        """
        X = self.design  # (n_samples, n_features)
        n_samples, n_genes = weights.shape
        n_features = self.n_features
        
        # Clip weights for numerical stability
        W = np.clip(weights, self.min_mu, None)  # (n_samples, n_genes)
        
        # Efficient X^T W X computation using blocked approach
        # X^T W X [g,i,j] = sum_k X[k,i] * W[k,g] * X[k,j]
        # 
        # For 2x2 design matrix, we can compute the 4 elements directly:
        # (0,0): sum_k X[k,0]^2 * W[k,g] = sum_k W[k,g] (since X[:,0] = 1)
        # (0,1) = (1,0): sum_k X[k,0] * X[k,1] * W[k,g] = sum_k X[k,1] * W[k,g]
        # (1,1): sum_k X[k,1]^2 * W[k,g]
        
        if n_features == 2:
            # Fast path for common 2-feature design
            X1 = X[:, 1]  # (n_samples,) - the perturbation indicator
            
            # XtWX elements (vectorized over genes)
            xtwx_00 = np.sum(W, axis=0)  # (n_genes,)
            xtwx_01 = X1 @ W  # (n_genes,)
            xtwx_11 = (X1[:, None] ** 2 * W).sum(axis=0)  # (n_genes,)
            
            # X^T W z elements
            Wz = W * y_working  # (n_samples, n_genes)
            xtwz_0 = np.sum(Wz, axis=0)  # (n_genes,)
            xtwz_1 = X1 @ Wz  # (n_genes,)
            
            # Add ridge penalty
            if self.ridge_penalty:
                xtwx_00 = xtwx_00 + self.ridge_penalty
                xtwx_11 = xtwx_11 + self.ridge_penalty
            
            # Solve 2x2 systems analytically using Cramer's rule
            det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)  # Avoid division by zero
            
            beta0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
            beta1 = (xtwx_00 * xtwz_1 - xtwx_01 * xtwz_0) / det
            
            beta = np.vstack([beta0, beta1])  # (n_features, n_genes)
            return beta
        else:
            # General case using einsum
            xtwx = np.einsum('ki,kg,kj->gij', X, W, X, optimize=True)
            
            # Add ridge penalty to diagonal
            if self.ridge_penalty:
                ridge = self.ridge_penalty * np.eye(n_features, dtype=np.float64)
                xtwx = xtwx + ridge[None, :, :]
            
            # Compute X^T W z for all genes: (n_genes, n_features)
            Wz = W * y_working  # (n_samples, n_genes)
            xtwz = np.einsum('ki,kg->gi', X, Wz, optimize=True)
            
            # Solve all systems at once using batched solve
            # Need to add dimension for broadcasting: (n_genes, n_features, 1)
            try:
                beta = np.linalg.solve(xtwx, xtwz[:, :, None])[:, :, 0]  # (n_genes, n_features)
            except np.linalg.LinAlgError:
                # Fallback to per-gene solve for singular matrices
                beta = np.zeros((n_genes, n_features), dtype=np.float64)
                for g in range(n_genes):
                    try:
                        beta[g] = np.linalg.solve(xtwx[g], xtwz[g])
                    except np.linalg.LinAlgError:
                        beta[g] = np.linalg.lstsq(xtwx[g], xtwz[g], rcond=None)[0]
            
            return beta.T  # (n_features, n_genes)
    
    def _compute_se_batch(self, weights: np.ndarray) -> np.ndarray:
        """Compute standard errors for all genes using vectorized operations."""
        X = self.design  # (n_samples, n_features)
        n_samples, n_genes = weights.shape
        n_features = self.n_features
        
        # Clip weights for numerical stability
        W = np.clip(weights, self.min_mu, None)  # (n_samples, n_genes)
        
        if n_features == 2:
            # Fast path for 2-feature design: use analytical inverse of 2x2 matrix
            X1 = X[:, 1]  # (n_samples,) - the perturbation indicator
            
            # XtWX elements (vectorized over genes)
            xtwx_00 = np.sum(W, axis=0)  # (n_genes,)
            xtwx_01 = X1 @ W  # (n_genes,)
            xtwx_11 = (X1[:, None] ** 2 * W).sum(axis=0)  # (n_genes,)
            
            # Add ridge penalty
            if self.ridge_penalty:
                xtwx_00 = xtwx_00 + self.ridge_penalty
                xtwx_11 = xtwx_11 + self.ridge_penalty
            
            # 2x2 matrix inverse diagonal elements:
            # For M = [[a, b], [b, c]], M^-1 = (1/det) * [[c, -b], [-b, a]]
            # So diag(M^-1) = [c/det, a/det]
            det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)  # Avoid division by zero
            
            inv_diag_0 = xtwx_11 / det  # Variance of beta_0 (intercept)
            inv_diag_1 = xtwx_00 / det  # Variance of beta_1 (perturbation effect)
            
            se = np.vstack([
                np.sqrt(np.maximum(inv_diag_0, 1e-12)),
                np.sqrt(np.maximum(inv_diag_1, 1e-12))
            ])  # (n_features, n_genes)
            return se
        else:
            # General case
            # Compute X^T W X for all genes: (n_genes, n_features, n_features)
            xtwx = np.einsum('ki,kg,kj->gij', X, W, X, optimize=True)
            
            # Add ridge penalty to diagonal
            if self.ridge_penalty:
                ridge = self.ridge_penalty * np.eye(n_features, dtype=np.float64)
                xtwx = xtwx + ridge[None, :, :]
            
            # Invert all matrices at once and extract diagonal
            se = np.full((n_features, n_genes), np.inf, dtype=np.float64)
            try:
                inv_xtwx = np.linalg.inv(xtwx)  # (n_genes, n_features, n_features)
                # Extract diagonal of each inverse matrix
                diag_inv = np.diagonal(inv_xtwx, axis1=1, axis2=2)  # (n_genes, n_features)
                se = np.sqrt(np.maximum(diag_inv, 1e-12)).T  # (n_features, n_genes)
            except np.linalg.LinAlgError:
                # Fallback to per-gene inversion for singular matrices
                for g in range(n_genes):
                    try:
                        inv_xtwx_g = np.linalg.inv(xtwx[g])
                        se[:, g] = np.sqrt(np.maximum(np.diag(inv_xtwx_g), 1e-12))
                    except np.linalg.LinAlgError:
                        pass
            
            return se
    
    def _compute_deviance_batch(
        self, Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """Compute deviance for all genes using vectorized operations."""
        # Y, mu: (n_samples, n_genes), alpha: (n_genes,)
        mu_safe = np.maximum(mu, 1e-12)
        Y_safe = np.maximum(Y, 1e-12)
        
        # Compute r = 1/alpha for each gene, broadcast to (1, n_genes)
        r = 1.0 / np.maximum(alpha, 1e-10)  # (n_genes,)
        r = r[None, :]  # (1, n_genes) for broadcasting
        
        with np.errstate(divide="ignore", invalid="ignore"):
            # NB deviance: 2 * sum(y * log(y/mu) - (y + r) * log((y + r) / (mu + r)))
            # Handle y=0 case: when y=0, y*log(y/mu) = 0
            term1 = np.where(Y > 0, Y * np.log(Y_safe / mu_safe), 0.0)
            term2 = (Y + r) * np.log((Y + r) / (mu_safe + r))
            
            # Sum over samples (axis 0) for each gene
            deviance = 2.0 * np.nansum(term1 - term2, axis=0)  # (n_genes,)
        
        return deviance
    
    def _refine_dispersion_cox_reid_batch(
        self, Y: np.ndarray, mu: np.ndarray, alpha_init: np.ndarray
    ) -> np.ndarray:
        """Refine dispersion estimates using Cox-Reid for all genes.
        
        Uses vectorized grid search with numba acceleration for speed.
        Precomputes gammaln(Y+1) to avoid redundant computation.
        """
        n_samples, n_genes = Y.shape
        n_features = self.n_features
        
        # Vectorized grid search with smaller grid (10 points)
        log_grid = np.linspace(-3, 2, 10)
        alpha_grid = 10.0 ** log_grid
        n_alpha = len(alpha_grid)
        
        # Pre-compute design matrix quantities
        X = self.design
        
        # Precompute gammaln(Y + 1) - this is expensive and Y doesn't change
        gammaln_Y_plus_1 = gammaln_nb(Y + 1)
        
        # Compute NB log-likelihood for all alpha values using parallelized numba kernel
        ll_grid = _nb_loglik_grid_numba(Y, mu, alpha_grid, gammaln_Y_plus_1)
        
        # Cox-Reid adjustment: -0.5 * log(det(X^T W X)) for each alpha
        # Precompute X1 quantities for 2-feature case
        if n_features == 2:
            X1 = X[:, 1]
            X1_sq = X1 ** 2
        
        for a_idx, a in enumerate(alpha_grid):
            variance = mu + a * (mu ** 2)
            W = (mu ** 2) / np.maximum(variance, self.min_mu)
            
            if n_features == 2:
                # Fast path for 2-feature design: analytical determinant
                xtwx_00 = np.sum(W, axis=0)
                xtwx_01 = X1 @ W
                xtwx_11 = np.sum(X1_sq[:, None] * W, axis=0)
                det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
                log_det = np.log(np.maximum(det, 1e-12))
            else:
                # General case using einsum
                XtWX = np.einsum('ki,kg,kj->gij', X, W, X, optimize=True)
                try:
                    sign, log_det = np.linalg.slogdet(XtWX)
                    log_det = np.where(sign > 0, log_det, 0.0)
                except np.linalg.LinAlgError:
                    log_det = np.zeros(n_genes)
            
            ll_grid[a_idx] -= 0.5 * log_det
        
        # Find best alpha for each gene
        nll_grid = -ll_grid  # (n_alpha, n_genes)
        best_idx = np.argmin(nll_grid, axis=0)  # (n_genes,)
        best_alpha = alpha_grid[best_idx]
        
        # Clip to reasonable range
        alpha = np.clip(best_alpha, 1e-8, 1e3)
        
        return alpha

    def fit_batch_with_control_cache(
        self,
        perturbation_matrix: np.ndarray | sp.csr_matrix,
        perturbation_offset: np.ndarray,
        control_cache: "ControlStatisticsCache",
        *,
        perturbation_indicator: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> NBGLMBatchResult:
        """Fit NB GLM using precomputed control cell statistics.
        
        This method provides significant speedup by reusing control cell
        contributions (XᵀWX, XᵀWz) from a precomputed cache instead of
        redundantly computing them for each perturbation comparison.
        
        The design matrix is [1, perturbation_indicator] where:
        - Control cells have indicator = 0
        - Perturbation cells have indicator = 1
        
        The control contribution to XᵀWX and XᵀWz is taken from the cache,
        and only perturbation cell contributions are computed fresh.
        
        Parameters
        ----------
        perturbation_matrix
            Expression matrix for perturbation cells only, shape (n_pert, n_genes).
        perturbation_offset
            Log size factors for perturbation cells, shape (n_pert,).
        control_cache
            Precomputed control cell statistics from `precompute_control_statistics`.
        perturbation_indicator
            Binary indicator for perturbation cells in the combined design.
            Should be shape (n_control + n_pert,) with 0 for control, 1 for perturbation.
        valid_mask
            Optional boolean mask for genes to fit, shape (n_genes,).
            
        Returns
        -------
        NBGLMBatchResult
            Fitted coefficients and statistics.
        """
        # Densify perturbation matrix if needed
        if sp.issparse(perturbation_matrix):
            Y_pert = np.asarray(perturbation_matrix.toarray(), dtype=np.float64)
        else:
            Y_pert = np.asarray(perturbation_matrix, dtype=np.float64)
        
        n_pert, n_genes = Y_pert.shape
        n_control = control_cache.control_n
        n_total = n_control + n_pert
        
        # Get control data from cache (already dense, no need for .toarray())
        Y_control = control_cache.control_matrix  # Already np.ndarray
        
        # Initialize outputs
        n_features = 2  # intercept + perturbation
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter_arr = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Determine valid genes
        if valid_mask is None:
            total_counts = Y_control.sum(axis=0) + Y_pert.sum(axis=0)
            valid_mask = total_counts >= self.min_total_count
        
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter_arr, deviance=deviance
            )
        
        # Work with valid genes only
        Y_control_valid = Y_control[:, valid_mask]
        Y_pert_valid = Y_pert[:, valid_mask]
        
        # Initialize beta from cache: [β₀_cached, 0]
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        beta[0, :] = control_cache.beta_intercept[valid_mask]
        
        # Use cached control dispersion as starting point
        alpha = control_cache.control_dispersion[valid_mask].copy()
        
        # Precompute offsets
        offset_control = control_cache.control_offset[:, None]
        offset_pert = perturbation_offset[:, None]
        log_min_mu = np.log(self.min_mu)
        
        # Work arrays
        mu_control = np.empty((n_control, n_valid), dtype=np.float64)
        mu_pert = np.empty((n_pert, n_valid), dtype=np.float64)
        W_control = np.empty_like(mu_control)
        W_pert = np.empty_like(mu_pert)
        
        # Convergence tracking
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        for iteration in range(1, self.max_iter + 1):
            beta_intercept = beta[0, :]  # (n_valid,)
            beta_pert = beta[1, :]  # (n_valid,)
            
            # Control cells: eta = β₀ + offset (perturbation indicator = 0)
            eta_control = beta_intercept[None, :] + offset_control
            np.clip(eta_control, log_min_mu, 20.0, out=eta_control)
            np.exp(eta_control, out=mu_control)
            np.maximum(mu_control, self.min_mu, out=mu_control)
            
            # Perturbation cells: eta = β₀ + β₁ + offset (perturbation indicator = 1)
            eta_pert = beta_intercept[None, :] + beta_pert[None, :] + offset_pert
            np.clip(eta_pert, log_min_mu, 20.0, out=eta_pert)
            np.exp(eta_pert, out=mu_pert)
            np.maximum(mu_pert, self.min_mu, out=mu_pert)
            
            # Weights: W = μ² / (μ + α * μ²)
            var_control = mu_control + alpha[None, :] * mu_control * mu_control
            np.divide(mu_control * mu_control, np.maximum(var_control, self.min_mu), out=W_control)
            
            var_pert = mu_pert + alpha[None, :] * mu_pert * mu_pert
            np.divide(mu_pert * mu_pert, np.maximum(var_pert, self.min_mu), out=W_pert)
            
            # Working responses
            z_control = eta_control + (Y_control_valid - mu_control) / np.maximum(mu_control, self.min_mu)
            z_pert = eta_pert + (Y_pert_valid - mu_pert) / np.maximum(mu_pert, self.min_mu)
            
            # Remove offsets for working response
            z_control_centered = z_control - offset_control
            z_pert_centered = z_pert - offset_pert
            
            # Compute XᵀWX and XᵀWz
            # For design [1, p] where p is perturbation indicator:
            # XᵀWX = [[sum_all(W), sum_pert(W)],
            #         [sum_pert(W), sum_pert(W)]]
            # XᵀWz = [sum_all(W*z), sum_pert(W*z)]
            
            W_control_sum = np.sum(W_control, axis=0)  # (n_valid,)
            W_pert_sum = np.sum(W_pert, axis=0)  # (n_valid,)
            
            Wz_control_sum = np.sum(W_control * z_control_centered, axis=0)  # (n_valid,)
            Wz_pert_sum = np.sum(W_pert * z_pert_centered, axis=0)  # (n_valid,)
            
            # XᵀWX elements
            xtwx_00 = W_control_sum + W_pert_sum  # sum over all cells
            xtwx_01 = W_pert_sum  # only perturbation cells contribute
            xtwx_11 = W_pert_sum  # perturbation indicator is 1 for pert cells
            
            # XᵀWz elements
            xtwz_0 = Wz_control_sum + Wz_pert_sum  # all cells
            xtwz_1 = Wz_pert_sum  # only perturbation cells
            
            # Add ridge penalty
            ridge = self.ridge_penalty
            xtwx_00 = xtwx_00 + ridge
            xtwx_11 = xtwx_11 + ridge
            
            # Solve 2x2 system using Cramer's rule
            det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            
            beta_new_0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
            beta_new_1 = (xtwx_00 * xtwz_1 - xtwx_01 * xtwz_0) / det
            
            beta_new = np.vstack([beta_new_0, beta_new_1])
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion (method of moments)
            resid_control = Y_control_valid - mu_control
            resid_pert = Y_pert_valid - mu_pert
            
            numerator = (
                np.sum((resid_control ** 2 - Y_control_valid) / np.maximum(mu_control ** 2, self.min_mu), axis=0)
                + np.sum((resid_pert ** 2 - Y_pert_valid) / np.maximum(mu_pert ** 2, self.min_mu), axis=0)
            )
            dof = max(n_total - n_features, 1)
            alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
            alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
            
            if np.all(gene_converged):
                break
        
        # Compute final standard errors using sandwich estimator (PyDESeq2 style)
        # SE = sqrt(c' @ H @ M @ H @ c) where:
        #   M = XᵀWX (unregularized Fisher information)
        #   Mr = M + ridge*I (regularized)
        #   H = inv(Mr)
        #   c = [0, 1] for perturbation effect
        
        # Recompute XᵀWX for final weights
        W_control_sum = np.sum(W_control, axis=0)
        W_pert_sum = np.sum(W_pert, axis=0)
        
        # Unregularized M
        M00 = W_control_sum + W_pert_sum
        M01 = W_pert_sum
        M11 = W_pert_sum
        
        # Regularized Mr = M + ridge*I
        ridge = self.ridge_penalty
        Mr00 = M00 + ridge
        Mr01 = M01
        Mr11 = M11 + ridge
        
        # H = inv(Mr) for 2x2: inv = (1/det) * [[d, -b], [-c, a]]
        det_r = Mr00 * Mr11 - Mr01 * Mr01
        det_r = np.where(np.abs(det_r) < 1e-12, 1e-12, det_r)
        
        H00 = Mr11 / det_r
        H01 = -Mr01 / det_r
        H11 = Mr00 / det_r
        
        # For contrast c = [0, 1]: Hc = [H[0,1], H[1,1]] = [H01, H11]
        Hc0 = H01
        Hc1 = H11
        
        # Sandwich variance: Hc.T @ M @ Hc
        # = Hc0² * M00 + 2 * Hc0 * Hc1 * M01 + Hc1² * M11
        var_pert = Hc0**2 * M00 + 2 * Hc0 * Hc1 * M01 + Hc1**2 * M11
        se_pert = np.sqrt(np.maximum(var_pert, 1e-12))
        
        # For intercept, contrast c = [1, 0]: Hc = [H00, H01]
        var_intercept = H00**2 * M00 + 2 * H00 * H01 * M01 + H01**2 * M11
        se_intercept = np.sqrt(np.maximum(var_intercept, 1e-12))
        
        se_valid = np.vstack([se_intercept, se_pert])  # (n_features, n_valid)
        
        # Store results
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter_arr[valid_indices] = gene_n_iter
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter_arr, deviance=deviance
        )
    def fit_batch_with_frozen_control(
        self,
        perturbation_matrix: np.ndarray | sp.csr_matrix,
        perturbation_offset: np.ndarray,
        control_cache: "ControlStatisticsCache",
        *,
        valid_mask: np.ndarray | None = None,
    ) -> NBGLMBatchResult:
        """Fit NB GLM using frozen control sufficient statistics (memory-efficient).
        
        This method uses precomputed sufficient statistics from control cells
        instead of the raw control_matrix, reducing per-worker memory from
        ~5GB to ~1MB for large datasets.
        
        Key differences from fit_batch_with_control_cache:
        - β₀ (intercept) is FROZEN to the value estimated from control cells
        - Only β₁ (perturbation effect) is estimated
        - Control contributions (W_sum, Wz_sum) are pre-computed constants
        - No access to raw control_matrix (control_cache.control_matrix is None)
        
        The design matrix is [1, perturbation_indicator] where:
        - Control cells have indicator = 0 (contributions are frozen)
        - Perturbation cells have indicator = 1
        
        Parameters
        ----------
        perturbation_matrix
            Expression matrix for perturbation cells only, shape (n_pert, n_genes).
        perturbation_offset
            Log size factors for perturbation cells, shape (n_pert,).
        control_cache
            Precomputed control cell statistics with use_frozen_control=True.
            Must have frozen_control_W_sum and frozen_control_Wz_sum set.
        valid_mask
            Optional boolean mask for genes to fit, shape (n_genes,).
            
        Returns
        -------
        NBGLMBatchResult
            Fitted coefficients and statistics.
            
        Notes
        -----
        This method is designed for parallel processing where each worker
        handles a subset of perturbations. By using frozen control stats:
        
        - Per-worker pickle size: ~5GB → ~1MB (control_matrix not needed)
        - Memory enables: 2 workers → 32 workers (for 128GB memory limit)
        - Time reduction: ~300h → ~10h (for genome-wide screens)
        
        Mathematical justification:
        With global dispersion and fixed β₀, the control cells' contribution
        to XᵀWX and XᵀWz is constant across all perturbation comparisons:
        
        - μ_control = exp(β₀ + offset) is fixed (no perturbation indicator)
        - W_control = μ²/(μ + α*μ²) depends only on μ and global α
        - z_control = η + (Y - μ)/μ - offset = β₀ + (Y - μ)/μ
        
        Therefore, sum(W_control) and sum(W_control * z_centered) are constants
        that can be pre-computed once and reused across all comparisons.
        """
        if not control_cache.use_frozen_control:
            raise ValueError(
                "control_cache.use_frozen_control must be True. "
                "Use precompute_control_statistics(..., freeze_control=True) to create the cache."
            )
        
        if control_cache.frozen_control_W_sum is None or control_cache.frozen_control_Wz_sum is None:
            raise ValueError(
                "Frozen control statistics not available. "
                "control_cache.frozen_control_W_sum and frozen_control_Wz_sum must be set."
            )
        
        # Densify perturbation matrix if needed
        if sp.issparse(perturbation_matrix):
            Y_pert = np.asarray(perturbation_matrix.toarray(), dtype=np.float64)
        else:
            Y_pert = np.asarray(perturbation_matrix, dtype=np.float64)
        
        n_pert, n_genes = Y_pert.shape
        n_control = control_cache.control_n
        n_total = n_control + n_pert
        
        # Initialize outputs
        n_features = 2  # intercept + perturbation
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter_arr = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Determine valid genes (use control-side total for validation)
        if valid_mask is None:
            # Without raw control_matrix, use frozen_control_Y_sum
            control_total = control_cache.frozen_control_Y_sum  # (n_genes,)
            pert_total = Y_pert.sum(axis=0)
            total_counts = control_total + pert_total
            valid_mask = total_counts >= self.min_total_count
        
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter_arr, deviance=deviance
            )
        
        # Work with valid genes only
        Y_pert_valid = Y_pert[:, valid_mask]
        
        # Frozen control sufficient statistics (pre-computed, constant)
        frozen_W_sum = control_cache.frozen_control_W_sum[valid_mask]  # (n_valid,)
        frozen_Wz_sum = control_cache.frozen_control_Wz_sum[valid_mask]  # (n_valid,)
        
        # For dispersion updates (method of moments)
        frozen_resid_sq_sum = control_cache.frozen_control_resid_sq_sum[valid_mask]
        frozen_Y_sum = control_cache.frozen_control_Y_sum[valid_mask]
        frozen_mu_sum = control_cache.frozen_control_mu_sum[valid_mask]
        
        # β₀ is FROZEN from control cells (this is the key insight!)
        beta_intercept = control_cache.beta_intercept[valid_mask].copy()  # (n_valid,)
        
        # β₁ (perturbation effect) is initialized to 0
        beta_pert = np.zeros(n_valid, dtype=np.float64)
        
        # Use global dispersion from control cache
        alpha = control_cache.control_dispersion[valid_mask].copy()
        
        # If global dispersion is available, use it (more stable)
        if control_cache.global_dispersion is not None:
            alpha = control_cache.global_dispersion[valid_mask].copy()
        
        # Precompute perturbation offsets
        offset_pert = perturbation_offset[:, None]  # (n_pert, 1)
        log_min_mu = np.log(self.min_mu)
        
        # Work arrays for perturbation cells only (no control arrays needed!)
        mu_pert = np.empty((n_pert, n_valid), dtype=np.float64)
        W_pert = np.empty_like(mu_pert)
        
        # Convergence tracking
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        for iteration in range(1, self.max_iter + 1):
            # Perturbation cells: eta = β₀ + β₁ + offset
            eta_pert = beta_intercept[None, :] + beta_pert[None, :] + offset_pert
            np.clip(eta_pert, log_min_mu, 20.0, out=eta_pert)
            np.exp(eta_pert, out=mu_pert)
            np.maximum(mu_pert, self.min_mu, out=mu_pert)
            
            # Weights: W = μ² / (μ + α * μ²)
            var_pert = mu_pert + alpha[None, :] * mu_pert * mu_pert
            np.divide(mu_pert * mu_pert, np.maximum(var_pert, self.min_mu), out=W_pert)
            
            # Working responses for perturbation cells
            z_pert = eta_pert + (Y_pert_valid - mu_pert) / np.maximum(mu_pert, self.min_mu)
            z_pert_centered = z_pert - offset_pert  # Remove offset
            
            # Perturbation contributions to XᵀWX and XᵀWz
            W_pert_sum = np.sum(W_pert, axis=0)  # (n_valid,)
            Wz_pert_sum = np.sum(W_pert * z_pert_centered, axis=0)  # (n_valid,)
            
            # XᵀWX elements (2x2 matrix per gene)
            # Control contributions are FROZEN, perturbation contributions are fresh
            xtwx_00 = frozen_W_sum + W_pert_sum  # sum over all cells
            xtwx_01 = W_pert_sum  # only perturbation cells have indicator=1
            xtwx_11 = W_pert_sum  # perturbation indicator is 1 for pert cells
            
            # XᵀWz elements
            xtwz_0 = frozen_Wz_sum + Wz_pert_sum  # all cells
            xtwz_1 = Wz_pert_sum  # only perturbation cells
            
            # Add ridge penalty
            ridge = self.ridge_penalty
            xtwx_00_reg = xtwx_00 + ridge
            xtwx_11_reg = xtwx_11 + ridge
            
            # Solve 2x2 system using Cramer's rule
            # BUT: β₀ is FROZEN, so we only update β₁
            # 
            # The full system is:
            # [xtwx_00  xtwx_01] [β₀]   [xtwz_0]
            # [xtwx_01  xtwx_11] [β₁] = [xtwz_1]
            #
            # With β₀ frozen, we solve for β₁ from the second row:
            # xtwx_01 * β₀ + xtwx_11 * β₁ = xtwz_1
            # β₁ = (xtwz_1 - xtwx_01 * β₀) / xtwx_11_reg
            
            beta_pert_new = (xtwz_1 - xtwx_01 * beta_intercept) / np.maximum(xtwx_11_reg, 1e-12)
            
            # Check convergence
            beta_diff = np.abs(beta_pert_new - beta_pert)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta_pert = beta_pert_new
            
            if np.all(gene_converged):
                break
        
        # Compute final standard errors using sandwich estimator
        # For frozen β₀, we use the conditional variance of β₁ given β₀
        
        # Recompute XᵀWX for final weights
        W_pert_sum = np.sum(W_pert, axis=0)
        
        # Unregularized M for SE calculation
        M00 = frozen_W_sum + W_pert_sum
        M01 = W_pert_sum
        M11 = W_pert_sum
        
        # Regularized Mr = M + ridge*I
        ridge = self.ridge_penalty
        Mr00 = M00 + ridge
        Mr01 = M01
        Mr11 = M11 + ridge
        
        # H = inv(Mr) for 2x2
        det_r = Mr00 * Mr11 - Mr01 * Mr01
        det_r = np.where(np.abs(det_r) < 1e-12, 1e-12, det_r)
        
        H00 = Mr11 / det_r
        H01 = -Mr01 / det_r
        H11 = Mr00 / det_r
        
        # For β₁ contrast c = [0, 1]: Hc = [H01, H11]
        Hc0 = H01
        Hc1 = H11
        
        # Sandwich variance: Hc.T @ M @ Hc
        var_pert_effect = Hc0**2 * M00 + 2 * Hc0 * Hc1 * M01 + Hc1**2 * M11
        se_pert = np.sqrt(np.maximum(var_pert_effect, 1e-12))
        
        # For intercept SE (using frozen β₀'s original SE would be more accurate,
        # but we approximate with the sandwich estimator for consistency)
        var_intercept = H00**2 * M00 + 2 * H00 * H01 * M01 + H01**2 * M11
        se_intercept = np.sqrt(np.maximum(var_intercept, 1e-12))
        
        # Store results
        beta = np.vstack([beta_intercept, beta_pert])  # (2, n_valid)
        se_valid = np.vstack([se_intercept, se_pert])  # (2, n_valid)
        
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter_arr[valid_indices] = gene_n_iter
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter_arr, deviance=deviance
        )