"""Numba-accelerated kernels for NB-GLM differential expression.

This module contains all Numba JIT-compiled functions used by the GLM module.
Separating these kernels improves code organization and allows for easier
maintenance of the performance-critical code paths.
"""

from __future__ import annotations

import ctypes
import math

import numba as nb
import numpy as np
from numba.extending import get_cython_function_address

from ._irls import solve_spd_2x2

_solve_spd_2x2 = nb.njit(cache=True)(solve_spd_2x2)

# =============================================================================
# Numba-accelerated gammaln using scipy's cython implementation
# =============================================================================

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
_functype = ctypes.CFUNCTYPE(_dble, _dble)
_gammaln_float64 = _functype(_addr)


@nb.vectorize([nb.float64(nb.float64)], nopython=True)
def gammaln_nb(x):
    """Numba-accelerated gammaln using scipy's cython implementation."""
    return _gammaln_float64(x)


# =============================================================================
# Grid search kernels for dispersion estimation
# =============================================================================

@nb.njit(parallel=True)
def _nb_loglik_grid_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    alpha_grid: np.ndarray,
    gammaln_Y_plus_1: np.ndarray,
) -> np.ndarray:
    """Compute NB log-likelihood for all alpha values in grid (parallelized).
    
    Parameters
    ----------
    Y : (n_samples, n_genes)
    mu : (n_samples, n_genes)
    alpha_grid : (n_alpha,)
    gammaln_Y_plus_1 : (n_samples, n_genes) precomputed
    
    Returns
    -------
    ll_grid : (n_alpha, n_genes) log-likelihood for each alpha and gene
    """
    n_samples, n_genes = Y.shape
    n_alpha = alpha_grid.shape[0]
    ll_grid = np.zeros((n_alpha, n_genes), dtype=np.float64)
    
    for a_idx in nb.prange(n_alpha):
        alpha = alpha_grid[a_idx]
        r = 1.0 / alpha
        log_r = np.log(r)
        gammaln_r = math.lgamma(r)
        
        for g in range(n_genes):
            ll = 0.0
            for i in range(n_samples):
                y_ig = Y[i, g]
                mu_ig = mu[i, g]
                ll += (
                    math.lgamma(y_ig + r)
                    - gammaln_r
                    - gammaln_Y_plus_1[i, g]
                    + r * (log_r - np.log(r + mu_ig + 1e-12))
                    + y_ig * np.log(mu_ig / (r + mu_ig + 1e-12) + 1e-12)
                )
            ll_grid[a_idx, g] = ll
    
    return ll_grid


# =============================================================================
# Dispersion estimation kernels
# =============================================================================

@nb.njit(cache=True)
def _nb_ll_for_alpha(Y_g, mu_g, alpha):
    """Compute NB log-likelihood for a single gene at a given alpha.
    
    Vectorized across cells using Numba-compatible operations.
    """
    r = 1.0 / alpha
    log_r = np.log(r)
    gammaln_r = math.lgamma(r)
    n_cells = Y_g.shape[0]
    
    ll = 0.0
    for i in range(n_cells):
        y = Y_g[i]
        mu = mu_g[i]
        ll += (
            math.lgamma(y + r)
            - gammaln_r
            - math.lgamma(y + 1.0)
            + r * (log_r - np.log(r + mu + 1e-12))
            + y * np.log(mu / (r + mu + 1e-12) + 1e-12)
        )
    return ll


@nb.njit(parallel=True, cache=True)
def _compute_mle_dispersion_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    dof: float,
) -> np.ndarray:
    """Compute MLE dispersion per gene without large intermediate arrays.
    
    Memory-optimized: computes per-gene without creating (n_cells, n_genes) intermediates.
    Uses Numba parallel for speed.
    """
    n_cells, n_genes = Y.shape
    alpha_mle = np.zeros(n_genes, dtype=np.float64)
    
    for g in nb.prange(n_genes):
        acc = 0.0
        for i in range(n_cells):
            y_val = Y[i, g]
            mu_val = mu[i, g]
            resid = y_val - mu_val
            variance = resid * resid - y_val
            denom = max(mu_val * mu_val, 1e-10)
            acc += variance / denom
        alpha_mle[g] = acc / dof
    
    return alpha_mle


@nb.njit(cache=True)
def _neg_half_log_det_numba(xtwx: np.ndarray) -> float:
    """``-0.5 * log det(xtwx)`` from a Cholesky factorisation in place.

    Zero for an empty matrix (no Cox-Reid adjustment), or one that is not
    positive definite (as the gene-wise Cox-Reid step treats it).
    """
    log_det = 0.0
    for j in range(xtwx.shape[0]):
        d = xtwx[j, j]
        for k in range(j):
            d -= xtwx[j, k] * xtwx[j, k]
        if d <= 0.0:
            return 0.0
        ljj = math.sqrt(d)
        xtwx[j, j] = ljj
        log_det += math.log(ljj)
        for i in range(j + 1, xtwx.shape[0]):
            v = xtwx[i, j]
            for k in range(j):
                v -= xtwx[i, k] * xtwx[j, k]
            xtwx[i, j] = v / ljj
    return -log_det  # -0.5 * (2 * sum(log diag(L)))


@nb.njit(parallel=True, cache=True)
def _nb_map_grid_search_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    log_trend: np.ndarray,
    log_alpha_grid: np.ndarray,
    prior_var: float,
    x_indptr: np.ndarray,
    x_indices: np.ndarray,
    x_data: np.ndarray,
    n_features: int,
) -> tuple:
    """Vectorized grid search for MAP dispersion across all genes.
    
    For each gene, evaluates the posterior (log-likelihood + log-prior) at each
    grid point and finds the best grid point. Returns the best log-alpha and
    the indices of adjacent grid points for refinement.
    
    Memory optimization: computes gammaln(Y + 1) once per gene instead of
    n_grid times, saving significant computation time.
    
    Parameters
    ----------
    Y : (n_cells, n_genes)
        Count matrix.
    mu : (n_cells, n_genes)
        Fitted mean matrix.
    log_trend : (n_genes,)
        Log of dispersion trend values.
    log_alpha_grid : (n_grid,)
        Grid of log-dispersion values to search.
    prior_var : float
        Variance of log-normal prior.
    x_indptr, x_indices, x_data, n_features
        The design matrix in CSR form, for the Cox-Reid adjustment
        (``n_features=0`` for none).
        
    Returns
    -------
    best_log_alpha : (n_genes,)
        Best log-alpha value for each gene.
    best_idx : (n_genes,)
        Index of best grid point.
    """
    n_cells, n_genes = Y.shape
    n_grid = log_alpha_grid.shape[0]
    
    best_log_alpha = np.zeros(n_genes, dtype=np.float64)
    best_idx = np.zeros(n_genes, dtype=np.int64)
    
    # Parallelize over genes
    for g in nb.prange(n_genes):
        best_posterior = -np.inf
        best_k = 0
        log_trend_g = log_trend[g]
        Y_g = Y[:, g]
        mu_g = mu[:, g]
        xtwx = np.empty((n_features, n_features))
        
        # Precompute gammaln(Y_g + 1) for this gene - only done once, not n_grid times!
        gammaln_y_plus_1 = 0.0
        for i in range(n_cells):
            gammaln_y_plus_1 += math.lgamma(Y_g[i] + 1.0)
        
        for k in range(n_grid):
            log_alpha = log_alpha_grid[k]
            posterior = _nb_posterior_with_cache_numba(
                Y_g, mu_g, log_alpha, log_trend_g, prior_var, gammaln_y_plus_1,
                x_indptr, x_indices, x_data, xtwx,
            )
            
            if posterior > best_posterior:
                best_posterior = posterior
                best_k = k
                best_log_alpha[g] = log_alpha
        
        best_idx[g] = best_k
    
    return best_log_alpha, best_idx


@nb.njit(cache=True)
def _brent_minimize_numba(
    Y_g: np.ndarray,
    mu_g: np.ndarray,
    log_trend_g: float,
    prior_var: float,
    gammaln_y_plus_1_sum: float,
    x_indptr: np.ndarray,
    x_indices: np.ndarray,
    x_data: np.ndarray,
    xtwx: np.ndarray,
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> float:
    """Brent's method for minimizing -posterior in [a, b].
    
    This is a Numba-compatible implementation of scipy's minimize_scalar (bounded).
    Returns the log_alpha that maximizes the posterior.
    """
    # Golden ratio
    golden = 0.3819660112501051  # (3 - sqrt(5)) / 2
    
    # Initial setup
    x = w = v = a + golden * (b - a)
    fx = fw = fv = -_nb_posterior_with_cache_numba(
        Y_g, mu_g, x, log_trend_g, prior_var, gammaln_y_plus_1_sum, x_indptr, x_indices, x_data, xtwx
    )
    
    d = 0.0  # Distance to next point
    e = 0.0  # Distance moved on the step before last
    
    for _ in range(max_iter):
        midpoint = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1
        
        # Check for convergence
        if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
            return x
        
        # Try parabolic interpolation
        if abs(e) > tol1:
            # Fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            
            if q > 0:
                p = -p
            else:
                q = -q
            
            r = e
            e = d
            
            # Check if parabolic step is acceptable
            if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
                # Take parabolic step
                d = p / q
                u = x + d
                
                # f must not be evaluated too close to a or b
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if x < midpoint else -tol1
            else:
                # Take golden section step
                e = (b if x < midpoint else a) - x
                d = golden * e
        else:
            # Take golden section step
            e = (b if x < midpoint else a) - x
            d = golden * e
        
        # f must not be evaluated too close to x
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d > 0 else -tol1)
        
        fu = -_nb_posterior_with_cache_numba(
            Y_g, mu_g, u, log_trend_g, prior_var, gammaln_y_plus_1_sum, x_indptr, x_indices, x_data, xtwx
        )
        
        # Update a, b, v, w, x
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    
    return x


@nb.njit(cache=True)
def _nb_posterior_with_cache_numba(
    Y_g: np.ndarray,
    mu_g: np.ndarray,
    log_alpha: float,
    log_trend_g: float,
    prior_var: float,
    gammaln_y_plus_1_sum: float,
    x_indptr: np.ndarray,
    x_indices: np.ndarray,
    x_data: np.ndarray,
    xtwx: np.ndarray,
) -> float:
    """Cox-Reid adjusted NB log-likelihood plus log-prior for one gene.

    Uses the precomputed ``gammaln(Y+1)`` sum. ``XᵀWX``, ``W = mu / (1 +
    alpha mu)``, is accumulated into the work buffer ``xtwx`` (``(0, 0)``
    for no adjustment) in the same pass over the cells as the likelihood;
    the design ``X`` is given in CSR form, so a one-hot batch design costs a
    few products per cell rather than ``n_features**2``.
    """
    n_cells = Y_g.shape[0]
    alpha = math.exp(log_alpha)
    r = 1.0 / alpha
    log_r = math.log(r)
    gammaln_r = math.lgamma(r)
    cox_reid = xtwx.shape[0] > 0
    xtwx[:, :] = 0.0

    ll = 0.0
    for i in range(n_cells):
        y = Y_g[i]
        mu_i = mu_g[i]
        r_plus_mu = r + mu_i + 1e-12
        ll += (
            math.lgamma(y + r)
            - gammaln_r
            + r * (log_r - math.log(r_plus_mu))
            + y * math.log(mu_i / r_plus_mu + 1e-12)
        )
        if cox_reid:
            w = mu_i / (1.0 + alpha * mu_i)
            for a in range(x_indptr[i], x_indptr[i + 1]):
                wa = w * x_data[a]
                ja = x_indices[a]
                for b in range(x_indptr[i], x_indptr[i + 1]):
                    xtwx[ja, x_indices[b]] += wa * x_data[b]
    ll -= gammaln_y_plus_1_sum

    log_prior = -0.5 * (log_alpha - log_trend_g) ** 2 / prior_var
    return ll + log_prior + _neg_half_log_det_numba(xtwx)


@nb.njit(parallel=True, cache=True)
def _nb_map_grid_search_with_refinement_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    log_trend: np.ndarray,
    log_alpha_grid: np.ndarray,
    prior_var: float,
    x_indptr: np.ndarray,
    x_indices: np.ndarray,
    x_data: np.ndarray,
    n_features: int,
    tol: float = 1e-5,
    max_refine_iter: int = 50,
) -> np.ndarray:
    """Fused grid search + Brent's method refinement for MAP dispersion.
    
    This kernel combines grid search and refinement in a single parallelized
    pass, avoiding joblib overhead for per-gene refinement. Uses Brent's method
    (quadratic interpolation) for refinement which is more accurate than
    golden section search.
    
    Parameters
    ----------
    Y : (n_cells, n_genes)
        Count matrix.
    mu : (n_cells, n_genes)
        Fitted mean matrix.
    log_trend : (n_genes,)
        Log of dispersion trend values.
    log_alpha_grid : (n_grid,)
        Grid of log-dispersion values to search.
    prior_var : float
        Variance of log-normal prior.
    x_indptr, x_indices, x_data, n_features
        The design matrix in CSR form, for the Cox-Reid adjustment
        (``n_features=0`` for none).
    tol : float
        Tolerance for Brent convergence.
    max_refine_iter : int
        Maximum refinement iterations.
        
    Returns
    -------
    best_log_alpha : (n_genes,)
        Refined log-alpha value for each gene.
    """
    n_cells, n_genes = Y.shape
    n_grid = log_alpha_grid.shape[0]
    
    best_log_alpha = np.zeros(n_genes, dtype=np.float64)
    
    # Parallelize over genes
    for g in nb.prange(n_genes):
        best_posterior = -np.inf
        best_k = 0
        log_trend_g = log_trend[g]
        Y_g = Y[:, g]
        mu_g = mu[:, g]
        xtwx = np.empty((n_features, n_features))
        
        # Precompute gammaln(Y_g + 1) for this gene - reused in refinement
        gammaln_y_plus_1_sum = 0.0
        for i in range(n_cells):
            gammaln_y_plus_1_sum += math.lgamma(Y_g[i] + 1.0)
        
        # Stage 1: Grid search
        for k in range(n_grid):
            log_alpha = log_alpha_grid[k]
            posterior = _nb_posterior_with_cache_numba(
                Y_g, mu_g, log_alpha, log_trend_g, prior_var, gammaln_y_plus_1_sum,
                x_indptr, x_indices, x_data, xtwx,
            )
            
            if posterior > best_posterior:
                best_posterior = posterior
                best_k = k
        
        best_grid_log_alpha = log_alpha_grid[best_k]
        
        # Stage 2: Brent's method refinement (if not at boundary)
        # We want to MAXIMIZE the posterior, so we use Brent to find minimum of -posterior
        if best_k > 0 and best_k < n_grid - 1:
            # Bracket: [grid[best_k-1], grid[best_k+1]]
            a = log_alpha_grid[best_k - 1]
            b = log_alpha_grid[best_k + 1]
            
            if b - a > tol:
                # Use Brent's method for refinement (more accurate than golden section)
                refined_log_alpha = _brent_minimize_numba(
                    Y_g, mu_g, log_trend_g, prior_var, gammaln_y_plus_1_sum,
                    x_indptr, x_indices, x_data, xtwx, a, b, tol=tol, max_iter=max_refine_iter
                )
                
                # Final sanity check: ensure refinement is actually better than grid
                refined_posterior = _nb_posterior_with_cache_numba(
                    Y_g, mu_g, refined_log_alpha, log_trend_g, prior_var, gammaln_y_plus_1_sum,
                    x_indptr, x_indices, x_data, xtwx,
                )
                if refined_posterior >= best_posterior:
                    best_log_alpha[g] = refined_log_alpha
                else:
                    best_log_alpha[g] = best_grid_log_alpha
            else:
                best_log_alpha[g] = best_grid_log_alpha
        else:
            best_log_alpha[g] = best_grid_log_alpha
    
    return best_log_alpha


# =============================================================================
# Wilcoxon rank-sum test kernels (zero-separated ranking of sparse data)
# =============================================================================

# Threshold for zero-separation optimization: if >= this fraction of values are zero,
# use the optimized zero-separated ranking. Otherwise use standard full ranking.
_ZERO_PARTITION_THRESHOLD = 0.5


@nb.njit(parallel=True, cache=True)
def _presort_control_nonzeros(control_dense: np.ndarray):
    """Pre-sort non-zero control values per gene for reuse across groups.

    Returns a flat array of sorted non-zeros with per-gene offsets and counts.
    Sorting control non-zeros once per gene chunk instead of once per group
    gives ~n_groups× speedup for the dominant zero-separation path.
    """
    n_control, n_genes = control_dense.shape

    # Pass 1: count non-zeros per gene (parallel)
    n_nonzero = np.empty(n_genes, dtype=np.int64)
    n_zeros = np.empty(n_genes, dtype=np.int64)
    for g in nb.prange(n_genes):
        nz = 0
        for i in range(n_control):
            if control_dense[i, g] != 0.0:
                nz += 1
        n_nonzero[g] = nz
        n_zeros[g] = n_control - nz

    # Prefix sum for offsets (sequential, only n_genes iterations)
    offsets = np.empty(n_genes + 1, dtype=np.int64)
    offsets[0] = 0
    for g in range(n_genes):
        offsets[g + 1] = offsets[g] + n_nonzero[g]

    total = offsets[n_genes]
    flat = np.empty(total, dtype=np.float64)

    # Pass 2: extract and sort non-zeros (parallel)
    for g in nb.prange(n_genes):
        start = offsets[g]
        nz = n_nonzero[g]
        idx = 0
        for i in range(n_control):
            if control_dense[i, g] != 0.0:
                flat[start + idx] = control_dense[i, g]
                idx += 1
        # Sort this gene's non-zeros
        if nz > 1:
            tmp = flat[start:start + nz].copy()
            tmp.sort()
            flat[start:start + nz] = tmp

    return flat, offsets, n_nonzero, n_zeros


@nb.njit(parallel=True, cache=True)
def _compute_ctrl_tie_sums(
    ctrl_sorted_flat: np.ndarray,
    ctrl_offsets: np.ndarray,
    ctrl_n_nonzero: np.ndarray,
) -> np.ndarray:
    """Compute per-gene tie-correction sums for pre-sorted control non-zeros.

    For each gene g, computes ``sum(t^3 - t)`` over all non-zero tie groups
    in the control distribution.  Called once per gene chunk, the result is
    passed to ``_wilcoxon_single_pert_presorted`` so that the per-pert binary
    search path can adjust the tie correction without re-walking the full
    control array.

    Parameters
    ----------
    ctrl_sorted_flat : (sum_ctrl_nnz,)
        Sorted control non-zeros from ``_presort_control_nonzeros``.
    ctrl_offsets : (n_genes + 1,)
        Per-gene start offsets into ``ctrl_sorted_flat``.
    ctrl_n_nonzero : (n_genes,)
        Number of non-zero control values per gene.

    Returns
    -------
    ctrl_tie_sums : (n_genes,)
        Per-gene ``sum(t^3 - t)`` for the control non-zero tie groups.
    """
    n_genes = ctrl_n_nonzero.shape[0]
    ctrl_tie_sums = np.zeros(n_genes, dtype=np.float64)

    for g in nb.prange(n_genes):
        n_nz = ctrl_n_nonzero[g]
        if n_nz < 2:
            continue
        start = ctrl_offsets[g]
        i = 0
        while i < n_nz:
            v = ctrl_sorted_flat[start + i]
            tie_start = i
            while i < n_nz - 1 and ctrl_sorted_flat[start + i + 1] == v:
                i += 1
            tie_count = i - tie_start + 1
            if tie_count > 1:
                t = float(tie_count)
                ctrl_tie_sums[g] += t ** 3 - t
            i += 1

    return ctrl_tie_sums


@nb.njit(cache=True)
def _all_tied(n_zeros: int, ctrl_sorted: np.ndarray, pert_sorted: np.ndarray) -> bool:
    """Whether every value of a gene -- ``n_zeros`` zeros plus the sorted
    control and perturbation non-zeros -- is the same, so that the rank-sum
    test is undefined (its tie-corrected variance is zero)."""
    n_nonzero = ctrl_sorted.shape[0] + pert_sorted.shape[0]
    if n_nonzero == 0:
        return True
    if n_zeros > 0:
        return False
    if ctrl_sorted.shape[0] == 0:
        return pert_sorted[0] == pert_sorted[-1]
    if pert_sorted.shape[0] == 0:
        return ctrl_sorted[0] == ctrl_sorted[-1]
    return min(ctrl_sorted[0], pert_sorted[0]) == max(ctrl_sorted[-1], pert_sorted[-1])


@nb.njit(cache=True)
def _rank_sum_pert_bsearch_numba(
    ctrl_sorted: np.ndarray,
    pert_sorted: np.ndarray,
    n_zeros: int,
    ctrl_tie_sum: float,
) -> tuple:
    """Binary-search Wilcoxon rank sum for pert non-zeros vs pre-sorted ctrl.

    An O(n_pert_nz * log(n_ctrl_nz)) binary-search pass rather than an
    O(n_ctrl_nz + n_pert_nz) merge walk.  For CRISPR datasets where
    n_ctrl_nz >> n_pert_nz (e.g. 140 K ctrl vs 11 pert non-zeros), this is
    ~750x faster per gene per perturbation.

    Parameters
    ----------
    ctrl_sorted : (n_ctrl_nz,)
        Sorted control non-zero values for one gene (slice of ctrl_sorted_flat).
    pert_sorted : (n_pert_nz,)
        Sorted pert non-zero values for one gene.
    n_zeros : int
        Total number of zeros (ctrl + pert) for this gene.  Zeros occupy ranks
        1..n_zeros; non-zero ranks start at n_zeros + 1.
    ctrl_tie_sum : float
        Pre-computed ``sum(t^3 - t)`` for ctrl non-zero tie groups (from
        ``_compute_ctrl_tie_sums``).  Used as the starting point for the tie
        correction adjustment so ctrl values not present in pert are never
        re-visited.

    Returns
    -------
    rank_sum : float
        Sum of ranks of the pert non-zero values (the zero contribution
        ``n_pert_zeros * zero_avg_rank`` is added by the caller).
    tie_corr : float
        Tie correction factor: 1 - sum(t^3 - t) / (n_total^3 - n_total).
    """
    n_ctrl_nz = ctrl_sorted.shape[0]
    n_pert_nz = pert_sorted.shape[0]
    n_total = n_ctrl_nz + n_pert_nz + n_zeros
    n_total_f = float(n_total)

    # Start tie_sum with ctrl non-zero groups and zero group
    tie_sum = ctrl_tie_sum
    if n_zeros > 1:
        tz = float(n_zeros)
        tie_sum += tz ** 3 - tz

    rank_sum = 0.0

    # Walk through sorted pert values; group consecutive ties together
    j = 0
    while j < n_pert_nz:
        v = pert_sorted[j]

        # Count consecutive pert values equal to v
        n_pert_eq = 1
        while j + n_pert_eq < n_pert_nz and pert_sorted[j + n_pert_eq] == v:
            n_pert_eq += 1

        # Binary search: count ctrl values < v (lo_c) and == v (n_ctrl_eq)
        lo_c = np.searchsorted(ctrl_sorted, v, side='left')
        hi_c = np.searchsorted(ctrl_sorted, v, side='right')
        n_ctrl_eq = hi_c - lo_c

        # Values below v in the combined non-zero sorted sequence:
        #   lo_c ctrl values + j pert values all have value < v
        n_below = lo_c + j
        n_eq = n_ctrl_eq + n_pert_eq

        # Average rank for this tied group (1-indexed; zeros occupy 1..n_zeros)
        avg_rank = float(n_zeros) + float(n_below) + float(n_eq + 1) * 0.5
        rank_sum += float(n_pert_eq) * avg_rank

        # Adjust tie correction.  ctrl_tie_sum already accounts for ctrl-only
        # tie groups (those with count >= 2).  We only need to patch in the
        # new combined contribution for any value that appears in pert.
        if n_ctrl_eq > 1:
            # Replace ctrl-only contribution with combined contribution
            old_ctrl = float(n_ctrl_eq) ** 3 - float(n_ctrl_eq)
            new_comb = float(n_eq) ** 3 - float(n_eq)
            tie_sum += new_comb - old_ctrl
        elif n_ctrl_eq == 1:
            # Was a singleton in ctrl (not in ctrl_tie_sum); add combined
            t = float(n_eq)  # n_eq >= 2 since n_ctrl_eq=1 and n_pert_eq>=1
            tie_sum += t ** 3 - t
        elif n_pert_eq > 1:
            # Pure pert tie with no ctrl match
            t = float(n_pert_eq)
            tie_sum += t ** 3 - t

        j += n_pert_eq

    denom = n_total_f ** 3 - n_total_f
    if denom > 0.0:
        tie_corr = 1.0 - tie_sum / denom
    else:
        tie_corr = 1.0

    return rank_sum, tie_corr


@nb.njit(parallel=False, cache=True)
def _wilcoxon_single_pert_presorted(
    control_dense: np.ndarray,
    ctrl_sorted_flat: np.ndarray,
    ctrl_offsets: np.ndarray,
    ctrl_n_nonzero: np.ndarray,
    ctrl_n_zeros: np.ndarray,
    ctrl_tie_sums: np.ndarray,
    pert_dense: np.ndarray,
    valid_genes: np.ndarray,
    tie_correct: bool,
    zero_threshold: float,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Single-perturbation Wilcoxon kernel, called from the prange in
    :func:`_wilcoxon_batch_perts_presorted_numba` (so not itself parallel:
    Numba does not support nested parallel launches).

    Ranks the perturbation's non-zeros against the pre-sorted control
    non-zeros with :func:`_rank_sum_pert_bsearch_numba`. A gene whose values
    are all tied has no rank test, whatever ``tie_correct``: its U, z, p and
    effect are NaN.
    """
    n_control = control_dense.shape[0]
    n_pert = pert_dense.shape[0]
    n_genes = pert_dense.shape[1]
    n_total = n_control + n_pert

    n_control_f = float(n_control)
    n_pert_f = float(n_pert)
    n_total_f = float(n_total)

    for g in range(n_genes):
        if not valid_genes[g]:
            u_stat_out[g] = 0.0
            z_score_out[g] = 0.0
            pvalue_out[g] = 1.0
            effect_out[g] = 0.5  # the AUC of no shift
            continue

        pert_col = pert_dense[:, g]

        # Count pert zeros
        n_pert_zeros = 0
        for i in range(n_pert):
            if pert_col[i] == 0.0:
                n_pert_zeros += 1

        n_zeros = ctrl_n_zeros[g] + n_pert_zeros

        tied = n_zeros == n_total
        rank_sum = 0.0
        tie_corr = 1.0
        if not tied:
            # --- Binary-search ranking (always) ---
            # O(n_pert_nz * log(n_ctrl_nz)) — works for any zero fraction.
            # O(n_pert_nz * log(n_ctrl_nz)) — binary search is always
            # faster than O(n_total * log(n_total)) argsort for dense genes.
            n_ctrl_nz = ctrl_n_nonzero[g]
            n_pert_nonzero = n_pert - n_pert_zeros

            pert_nonzero = np.empty(n_pert_nonzero, dtype=np.float64)
            idx = 0
            for i in range(n_pert):
                if pert_col[i] != 0.0:
                    pert_nonzero[idx] = pert_col[i]
                    idx += 1
            pert_sorted = np.sort(pert_nonzero)

            start = ctrl_offsets[g]
            ctrl_sorted = ctrl_sorted_flat[start : start + n_ctrl_nz]

            tied = _all_tied(n_zeros, ctrl_sorted, pert_sorted)
            rank_sum_nz, tie_corr = _rank_sum_pert_bsearch_numba(
                ctrl_sorted, pert_sorted, n_zeros, ctrl_tie_sums[g]
            )

            if not tie_correct:
                tie_corr = 1.0

            zero_avg_rank = (float(n_zeros) + 1.0) / 2.0
            rank_sum = float(n_pert_zeros) * zero_avg_rank + rank_sum_nz

        if tied:
            # Every value tied: the rank test is undefined, not "no change",
            # with or without the tie correction.
            u_stat_out[g] = np.nan
            z_score_out[g] = np.nan
            pvalue_out[g] = np.nan
            effect_out[g] = np.nan
            continue

        # Statistics
        expected = n_pert_f * (n_total_f + 1.0) / 2.0
        u_stat = rank_sum - n_pert_f * (n_pert_f + 1.0) / 2.0

        std = math.sqrt(tie_corr * n_pert_f * n_control_f * (n_total_f + 1.0) / 12.0)

        if std > 0.0:
            z = (rank_sum - expected) / std
            abs_z = abs(z)
            pval = math.erfc(abs_z / math.sqrt(2.0))
        else:
            # An empty group: no comparison was made.
            z = np.nan
            pval = np.nan

        if n_pert_f > 0 and n_control_f > 0:
            effect = u_stat / (n_pert_f * n_control_f)  # AUC
        else:
            effect = 0.5

        if math.isnan(z):
            u_stat = np.nan
            effect = np.nan
        u_stat_out[g] = u_stat
        z_score_out[g] = z
        pvalue_out[g] = pval
        effect_out[g] = effect


@nb.njit(parallel=True, cache=True)
def _wilcoxon_batch_perts_presorted_numba(
    control_dense: np.ndarray,
    ctrl_sorted_flat: np.ndarray,
    ctrl_offsets: np.ndarray,
    ctrl_n_nonzero: np.ndarray,
    ctrl_n_zeros: np.ndarray,
    ctrl_tie_sums: np.ndarray,
    all_pert_stacked: np.ndarray,
    pert_row_offsets: np.ndarray,
    valid_masks: np.ndarray,
    tie_correct: bool,
    zero_threshold: float,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Batch Wilcoxon test: single prange over perturbations.

    A single ``prange`` over all perturbation groups of a gene chunk, rather
    than one parallel launch per perturbation.  Each parallel thread handles
    one perturbation and iterates over genes sequentially.

    Parameters
    ----------
    control_dense : (n_control, n_valid_genes)
        Dense control expression for this gene chunk.
    ctrl_sorted_flat : (sum_ctrl_nnz,)
        Pre-sorted control non-zeros (from ``_presort_control_nonzeros``).
    ctrl_offsets : (n_valid_genes + 1,)
        Offsets into ``ctrl_sorted_flat`` per gene.
    ctrl_n_nonzero : (n_valid_genes,)
        Number of non-zero control values per gene.
    ctrl_n_zeros : (n_valid_genes,)
        Number of zero control values per gene.
    ctrl_tie_sums : (n_valid_genes,)
        Per-gene ``sum(t^3 - t)`` for ctrl non-zero tie groups (from
        ``_compute_ctrl_tie_sums``).  Passed through to
        ``_wilcoxon_single_pert_presorted`` for the binary-search path.
    all_pert_stacked : (total_pert_cells, n_valid_genes)
        Pre-stacked dense pert matrix (all groups concatenated row-wise).
    pert_row_offsets : (n_perts + 1,)
        Row offsets into ``all_pert_stacked`` for each perturbation.
    valid_masks : (n_perts, n_valid_genes)
        Boolean validity mask per (pert, gene).
    tie_correct : bool
    zero_threshold : float
    u_stat_out : (n_perts, n_valid_genes)
    z_score_out : (n_perts, n_valid_genes)
    pvalue_out : (n_perts, n_valid_genes)
    effect_out : (n_perts, n_valid_genes)
    """
    n_perts = pert_row_offsets.shape[0] - 1

    for p_idx in nb.prange(n_perts):
        p_start = pert_row_offsets[p_idx]
        p_end = pert_row_offsets[p_idx + 1]
        pert_dense = all_pert_stacked[p_start:p_end, :]

        _wilcoxon_single_pert_presorted(
            control_dense,
            ctrl_sorted_flat,
            ctrl_offsets,
            ctrl_n_nonzero,
            ctrl_n_zeros,
            ctrl_tie_sums,
            pert_dense,
            valid_masks[p_idx],
            tie_correct,
            zero_threshold,
            u_stat_out[p_idx],
            z_score_out[p_idx],
            pvalue_out[p_idx],
            effect_out[p_idx],
        )


@nb.njit(parallel=False, cache=True)
def _wilcoxon_stratified_single_pert(
    ctrl_flat: np.ndarray,
    ctrl_starts: np.ndarray,
    ctrl_n_nonzero: np.ndarray,
    ctrl_n_zeros: np.ndarray,
    ctrl_tie_sums: np.ndarray,
    all_pert_stacked: np.ndarray,
    seg_offsets: np.ndarray,
    seg_batch: np.ndarray,
    seg_lo: int,
    seg_hi: int,
    valid_genes: np.ndarray,
    tie_correct: bool,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Batch-stratified (van Elteren) Wilcoxon rank-sum for a single perturbation.

    Ranks perturbation vs control **within each batch (stratum)** separately,
    then combines the per-stratum rank statistics.  For each stratum ``b`` with
    ``n1`` perturbation and ``n0`` control cells (``N = n1 + n0``):

    * ``R_b`` : rank sum of the perturbation cells (ranked within the stratum)
    * ``E_b = n1 * (N + 1) / 2`` : expected rank sum under H0
    * ``Var_b = c_b * n1 * n0 * (N + 1) / 12`` : variance with tie correction
      factor ``c_b``

    The strata are combined with unit weights (equivalent to summing the
    Mann-Whitney ``U`` statistics across strata)::

        Z = sum_b (R_b - E_b) / sqrt(sum_b Var_b)

    which is the stratified Wilcoxon / van Elteren test.  Strata where either
    ``n1`` or ``n0`` is zero contribute nothing (they carry no information about
    the comparison).  The combined common-language effect size is
    ``sum_b U_b / sum_b (n1_b * n0_b)``, the stratified AUC.

    Notes
    -----
    Reuses :func:`_rank_sum_pert_bsearch_numba` for the per-stratum rank sum,
    so the pre-sorted control non-zeros are looked up per ``(batch, gene)`` via
    ``ctrl_starts`` / ``ctrl_n_nonzero`` instead of the single pooled control.
    """
    n_genes = all_pert_stacked.shape[1]
    sqrt2 = math.sqrt(2.0)

    for g in range(n_genes):
        if not valid_genes[g]:
            u_stat_out[g] = 0.0
            z_score_out[g] = 0.0
            pvalue_out[g] = 1.0
            effect_out[g] = 0.5  # the AUC of no shift
            continue

        num = 0.0        # sum_b (R_b - E_b)
        var = 0.0        # sum_b Var_b
        u_sum = 0.0      # sum_b U_b
        n1n0_sum = 0.0   # sum_b n1_b * n0_b
        tied = True      # every contributing stratum has a single value

        for s in range(seg_lo, seg_hi):
            b = seg_batch[s]
            r0 = seg_offsets[s]
            r1 = seg_offsets[s + 1]
            n1 = r1 - r0
            if n1 == 0:
                continue

            n_ctrl_nz_b = ctrl_n_nonzero[b, g]
            n_ctrl_z_b = ctrl_n_zeros[b, g]
            n0 = n_ctrl_nz_b + n_ctrl_z_b
            if n0 == 0:
                continue

            N = n1 + n0

            # Count pert zeros and gather pert non-zeros for this gene/stratum.
            n_pert_zeros = 0
            for i in range(r0, r1):
                if all_pert_stacked[i, g] == 0.0:
                    n_pert_zeros += 1
            n_pert_nz = n1 - n_pert_zeros
            pert_nonzero = np.empty(n_pert_nz, dtype=np.float64)
            idx = 0
            for i in range(r0, r1):
                v = all_pert_stacked[i, g]
                if v != 0.0:
                    pert_nonzero[idx] = v
                    idx += 1
            pert_sorted = np.sort(pert_nonzero)

            n_zeros = n_ctrl_z_b + n_pert_zeros
            start = ctrl_starts[b, g]
            ctrl_sorted = ctrl_flat[start : start + n_ctrl_nz_b]

            if not _all_tied(n_zeros, ctrl_sorted, pert_sorted):
                tied = False
            rank_sum_nz, tie_corr = _rank_sum_pert_bsearch_numba(
                ctrl_sorted, pert_sorted, n_zeros, ctrl_tie_sums[b, g]
            )
            if not tie_correct:
                tie_corr = 1.0

            n1f = float(n1)
            n0f = float(n0)
            nf = float(N)
            zero_avg_rank = (float(n_zeros) + 1.0) / 2.0
            rank_sum = float(n_pert_zeros) * zero_avg_rank + rank_sum_nz

            expected_b = n1f * (nf + 1.0) / 2.0
            var_b = tie_corr * n1f * n0f * (nf + 1.0) / 12.0
            u_b = rank_sum - n1f * (n1f + 1.0) / 2.0

            num += rank_sum - expected_b
            var += var_b
            u_sum += u_b
            n1n0_sum += n1f * n0f

        if var > 0.0 and not tied:
            z = num / math.sqrt(var)
            pval = math.erfc(abs(z) / sqrt2)
        else:
            # No stratum to compare in, or every value tied within each:
            # the rank test is undefined (with or without the tie
            # correction), not "no change".
            z = np.nan
            pval = np.nan

        if n1n0_sum > 0.0:
            effect = u_sum / n1n0_sum  # AUC
        else:
            effect = 0.5

        if math.isnan(z):
            u_sum = np.nan
            effect = np.nan
        u_stat_out[g] = u_sum
        z_score_out[g] = z
        pvalue_out[g] = pval
        effect_out[g] = effect


@nb.njit(parallel=True, cache=True)
def _wilcoxon_stratified_batch_perts_numba(
    ctrl_flat: np.ndarray,
    ctrl_starts: np.ndarray,
    ctrl_n_nonzero: np.ndarray,
    ctrl_n_zeros: np.ndarray,
    ctrl_tie_sums: np.ndarray,
    all_pert_stacked: np.ndarray,
    seg_offsets: np.ndarray,
    seg_batch: np.ndarray,
    pert_ptr: np.ndarray,
    valid_masks: np.ndarray,
    tie_correct: bool,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Batch-stratified Wilcoxon test: single ``prange`` over perturbations.

    Parameters
    ----------
    ctrl_flat : (sum_ctrl_nnz,)
        Sorted control non-zeros for every ``(batch, gene)`` cell, concatenated
        in ``(batch, gene)`` order.
    ctrl_starts, ctrl_n_nonzero, ctrl_n_zeros, ctrl_tie_sums : (n_batches, n_valid_genes)
        Per ``(batch, gene)`` global start offset into ``ctrl_flat``, number of
        non-zero / zero control values, and the ``sum(t^3 - t)`` tie sum of the
        control non-zeros.
    all_pert_stacked : (total_pert_cells, n_valid_genes)
        Dense perturbation expression, rows concatenated in
        ``(perturbation, batch)`` order.
    seg_offsets : (n_segments + 1,)
        Row offsets into ``all_pert_stacked`` delimiting each
        ``(perturbation, batch)`` segment.
    seg_batch : (n_segments,)
        Batch index for each segment.
    pert_ptr : (n_perts + 1,)
        Segment range ``[pert_ptr[p], pert_ptr[p + 1])`` belonging to
        perturbation ``p``.
    valid_masks : (n_perts, n_valid_genes)
    tie_correct : bool
    u_stat_out, z_score_out, pvalue_out, effect_out : (n_perts, n_valid_genes)
    """
    n_perts = pert_ptr.shape[0] - 1

    for p_idx in nb.prange(n_perts):
        _wilcoxon_stratified_single_pert(
            ctrl_flat,
            ctrl_starts,
            ctrl_n_nonzero,
            ctrl_n_zeros,
            ctrl_tie_sums,
            all_pert_stacked,
            seg_offsets,
            seg_batch,
            pert_ptr[p_idx],
            pert_ptr[p_idx + 1],
            valid_masks[p_idx],
            tie_correct,
            u_stat_out[p_idx],
            z_score_out[p_idx],
            pvalue_out[p_idx],
            effect_out[p_idx],
        )


# Mirrors crispyx._irls.{EPS, ETA_MIN, ETA_MAX}; numba folds module-level
# floats in at compile time, so they cannot be imported as names here.
EPS_K = 1e-10
ETA_MIN_K = -30.0
ETA_MAX_K = 20.0


@nb.njit(parallel=True, cache=True)
def _irls_batch_numba(
    Y: np.ndarray,
    X: np.ndarray,
    offset: np.ndarray,
    alpha: np.ndarray,
    beta_init: np.ndarray,
    max_iter: int,
    tol: float,
    min_mu: float,
    ridge: float,
) -> tuple:
    """Numba-accelerated IRLS for batch of genes.
    
    Memory-optimized: processes each gene independently without large work arrays.
    Uses parallel loop over genes for speed.
    
    Parameters
    ----------
    Y : (n_samples, n_genes)
        Count matrix.
    X : (n_samples, n_features)
        Design matrix.
    offset : (n_samples,)
        Log size factors.
    alpha : (n_genes,)
        Dispersion estimates.
    beta_init : (n_features, n_genes)
        Initial coefficients.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
    min_mu : float
        Minimum mu value.
    ridge : float
        Ridge penalty.
        
    Returns
    -------
    beta : (n_features, n_genes)
        Fitted coefficients.
    converged : (n_genes,)
        Convergence flags.
    n_iter : (n_genes,)
        Number of iterations.
    """
    n_samples, n_genes = Y.shape
    n_features = X.shape[1]
    
    beta = np.copy(beta_init)
    converged = np.zeros(n_genes, dtype=nb.boolean)
    n_iter = np.zeros(n_genes, dtype=np.int32)
    
    # ``min_mu`` floors the fitted mean; clipping eta at log(min_mu) keeps
    # eta == log(mu), which the working response below relies on.  An unfloored
    # fit (min_mu == 0) has no such bound, so fall back to the generic clip.
    if min_mu > 0.0:
        log_min_mu = max(np.log(min_mu), ETA_MIN_K)
    else:
        log_min_mu = ETA_MIN_K
    
    # Parallel loop over genes
    for g in nb.prange(n_genes):
        alpha_g = alpha[g]
        beta_g = beta[:, g].copy()
        gene_converged = False
        
        # Per-gene work arrays (small, stack-allocated)
        mu_g = np.zeros(n_samples, dtype=np.float64)
        W_g = np.zeros(n_samples, dtype=np.float64)
        z_g = np.zeros(n_samples, dtype=np.float64)
        
        for iteration in range(max_iter):
            # Compute eta and mu
            for i in range(n_samples):
                eta_i = offset[i]
                for f in range(n_features):
                    eta_i += X[i, f] * beta_g[f]
                eta_i = min(max(eta_i, log_min_mu), ETA_MAX_K)
                mu_i = np.exp(eta_i)
                mu_i = max(mu_i, min_mu)
                mu_g[i] = mu_i
                
                # Weight: W = mu^2 / (mu + alpha * mu^2).  The guards are
                # division guards only -- flooring these at ``min_mu`` would
                # overstate the leverage of every low-count cell.
                var_i = mu_i + alpha_g * mu_i * mu_i
                W_g[i] = (mu_i * mu_i) / max(var_i, EPS_K)
                
                # Working response: z = eta + (y - mu) / mu
                z_g[i] = eta_i + (Y[i, g] - mu_i) / max(mu_i, EPS_K) - offset[i]
            
            # Solve WLS: (X'WX + ridge*I) beta_new = X'Wz
            # For 2-feature case, accumulate the 2x2 system directly
            if n_features == 2:
                xtwx_00 = ridge
                xtwx_01 = 0.0
                xtwx_11 = ridge
                xtwz_0 = 0.0
                xtwz_1 = 0.0
                
                for i in range(n_samples):
                    w_i = W_g[i]
                    z_i = z_g[i]
                    x1_i = X[i, 1]
                    
                    xtwx_00 += w_i
                    xtwx_01 += w_i * x1_i
                    xtwx_11 += w_i * x1_i * x1_i
                    xtwz_0 += w_i * z_i
                    xtwz_1 += w_i * z_i * x1_i
                
                beta_new_0, beta_new_1 = _solve_spd_2x2(xtwx_00, xtwx_01, xtwx_11, xtwz_0, xtwz_1)
                
                # Check convergence
                diff = max(abs(beta_new_0 - beta_g[0]), abs(beta_new_1 - beta_g[1]))
                beta_g[0] = beta_new_0
                beta_g[1] = beta_new_1
                
                if diff < tol:
                    gene_converged = True
                    n_iter[g] = iteration + 1
                    break
            else:
                # General case: would need matrix operations
                # For now, fallback to simpler convergence check
                gene_converged = True
                n_iter[g] = iteration + 1
                break
        
        if not gene_converged:
            n_iter[g] = max_iter
        
        beta[:, g] = beta_g
        converged[g] = gene_converged
    
    return beta, converged, n_iter


# =============================================================================
# Vectorized row-median for size factor computation
# =============================================================================

@nb.njit(cache=True)
def _median_sorted(arr: np.ndarray) -> float:
    """Compute median of a sorted array."""
    n = len(arr)
    if n == 0:
        return np.nan
    mid = n // 2
    if n % 2 == 0:
        return (arr[mid - 1] + arr[mid]) / 2.0
    return arr[mid]


@nb.njit(parallel=True, cache=True)
def _compute_row_medians_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    geo_means: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Compute median of ratios for each row of a CSR matrix.
    
    For each row, computes: median(data[j] / geo_means[indices[j]])
    where geo_means[indices[j]] > 0 and the ratio is finite and positive.
    
    Parameters
    ----------
    data : (nnz,)
        CSR data array.
    indices : (nnz,)
        CSR column indices.
    indptr : (n_rows + 1,)
        CSR row pointers.
    geo_means : (n_cols,)
        Geometric means for each column (gene).
    n_rows : int
        Number of rows.
        
    Returns
    -------
    medians : (n_rows,)
        Median ratio for each row. NaN if no valid ratios.
    """
    medians = np.full(n_rows, np.nan, dtype=np.float64)
    
    for row in nb.prange(n_rows):
        start = indptr[row]
        end = indptr[row + 1]
        
        if start == end:
            continue
        
        # Count valid ratios first
        n_valid = 0
        for j in range(start, end):
            col = indices[j]
            if geo_means[col] > 0:
                ratio = data[j] / geo_means[col]
                if np.isfinite(ratio) and ratio > 0:
                    n_valid += 1
        
        if n_valid == 0:
            continue
        
        # Allocate and fill valid ratios
        ratios = np.empty(n_valid, dtype=np.float64)
        idx = 0
        for j in range(start, end):
            col = indices[j]
            if geo_means[col] > 0:
                ratio = data[j] / geo_means[col]
                if np.isfinite(ratio) and ratio > 0:
                    ratios[idx] = ratio
                    idx += 1
        
        # Sort and compute median
        ratios.sort()
        medians[row] = _median_sorted(ratios)
    
    return medians


# =============================================================================
# Residual effects: one pass of sufficient statistics over CSR rows
# =============================================================================

@nb.njit(inline="always")
def _row_totals(indptr, indices, data, is_mito, r, base):
    total = 0.0
    mito_counts = 0.0
    for j in range(indptr[r] - base, indptr[r + 1] - base):
        total += data[j]
        if is_mito[indices[j]]:
            mito_counts += data[j]
    return total, mito_counts


@nb.njit(parallel=True, nogil=True, cache=True)
def _residual_effects_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    groups: np.ndarray,
    target_rows: np.ndarray,
    target_bounds: np.ndarray,
    edge_groups: np.ndarray,
    edges: np.ndarray,
    control_rows: np.ndarray,
    control_bounds: np.ndarray,
    category_columns: np.ndarray,
    continuous: np.ndarray,
    continuous_columns: np.ndarray,
    total_column: int,
    mito_column: int,
    is_mito: np.ndarray,
    sums: np.ndarray,
    control_xty: np.ndarray,
    control_sumsq: np.ndarray,
    totals: np.ndarray,
    mito: np.ndarray,
) -> None:
    """Accumulate :func:`crispyx.residual_effects`' statistics for a row chunk.

    Each row is a cell of raw counts; its total and mitochondrial share are
    written to ``totals`` / ``mito``. A perturbed cell with a non-zero total
    adds its ``log1p(CP10k)`` values to row ``groups[r]`` of ``sums``; a
    control cell (group 0) adds ``x * y`` to ``control_xty[w]`` (genes x
    design columns) and ``y * y`` to ``control_sumsq[w]``, where ``x`` is the cell's
    design row -- 1 in column 0 (intercept), 1 in each ``category_columns[r]``
    column (``-1`` for a reference level), ``continuous[r]`` in
    ``continuous_columns``, and the log total / mitochondrial share in
    ``total_column`` / ``mito_column`` (``-1`` when not in the design). ``x``
    is never formed.

    Work is split into segments run in parallel, without locks: segment
    ``w`` handles ``target_rows[target_bounds[w]:target_bounds[w + 1]]``
    and ``control_rows[control_bounds[w]:control_bounds[w + 1]]``, the
    latter into its own ``control_xty[w]`` and ``control_sumsq[w]``. Target
    rows are ordered by group, so only a segment's first and last group can
    be shared with another segment: rows of ``edge_groups[w, k]`` go to
    ``edges[w, k]`` (zeroed here) instead of ``sums``, for the caller to add
    in afterwards; every other group belongs to one segment alone. The
    controls' sum is the intercept column of ``control_xty``, so they do not
    touch ``sums``.
    """
    n_categories = category_columns.shape[1]
    n_continuous = continuous_columns.shape[0]
    base = indptr[0]
    for w in nb.prange(target_bounds.shape[0] - 1):
        edges[w] = 0.0
        for i in range(target_bounds[w], target_bounds[w + 1]):
            r = target_rows[i]
            total, mito_counts = _row_totals(indptr, indices, data, is_mito, r, base)
            totals[r] = total
            mito[r] = mito_counts / total if total > 0 else 0.0
            if total <= 0:
                continue
            scale = 1e4 / total
            group = groups[r]
            if group == edge_groups[w, 0]:
                acc = edges[w, 0]
            elif group == edge_groups[w, 1]:
                acc = edges[w, 1]
            else:
                acc = sums[group]
            for j in range(indptr[r] - base, indptr[r + 1] - base):
                acc[indices[j]] += math.log1p(data[j] * scale)
        xty = control_xty[w]
        sumsq = control_sumsq[w]
        for i in range(control_bounds[w], control_bounds[w + 1]):
            r = control_rows[i]
            total, mito_counts = _row_totals(indptr, indices, data, is_mito, r, base)
            totals[r] = total
            mito[r] = mito_counts / total if total > 0 else 0.0
            if total <= 0:
                continue
            scale = 1e4 / total
            log_total = math.log(total)
            share = mito_counts / total
            for j in range(indptr[r] - base, indptr[r + 1] - base):
                gene = indices[j]
                y = math.log1p(data[j] * scale)
                sumsq[gene] += y * y
                xty[gene, 0] += y
                for c in range(n_categories):
                    column = category_columns[r, c]
                    if column >= 0:
                        xty[gene, column] += y
                for c in range(n_continuous):
                    xty[gene, continuous_columns[c]] += continuous[r, c] * y
                if total_column >= 0:
                    xty[gene, total_column] += log_total * y
                if mito_column >= 0:
                    xty[gene, mito_column] += share * y


# =============================================================================
# Two-group DESeq2 against a control summary
# =============================================================================
#
# A comparison of one perturbation with the controls under the design
# [intercept, perturbation] fits the two groups' means separately, so every
# control term of DESeq2's objectives -- the NB log-likelihood, the Cox-Reid
# term (log det X'WX = log W_control + log W_perturbed) and the intercept's
# equation -- depends on the controls and the dispersion only. The controls
# are summarised once (glm.ControlSummary): a per-gene histogram of their
# non-zero counts, and per size-factor bin the first three moments of the
# size factors and of the counts weighted by them. Terms smooth in the size
# factor are evaluated per bin to second order around the bin centre; the
# perturbed cells, a few hundred, are summed exactly.
#
# Per cell, PyDESeq2's NB negative log-likelihood less the terms free of the
# dispersion alpha is
#     log1p(alpha*mu)/alpha + y*log1p(alpha*mu) - G(y, 1/alpha),
#     G(y, r) = lgamma(y + r) - lgamma(r) - y*log(r),
# written so that no term of order r*log(r) cancels when alpha is tiny.


@nb.njit(cache=True)
def _lgamma_ratio_term(y, r):
    """G(y, r) = lgamma(y + r) - lgamma(r) - y log r, stable for large r."""
    if y == 0.0:
        return 0.0
    if r < 10.0:
        return math.lgamma(y + r) - math.lgamma(r) - y * math.log(r)
    # Stirling series for lgamma at x = y + r and at r, differenced.
    x = y + r
    out = (x - 0.5) * math.log1p(y / r) - y
    out += 1.0 / (12.0 * x) - 1.0 / (12.0 * r)
    out -= 1.0 / (360.0 * x ** 3) - 1.0 / (360.0 * r ** 3)
    out += 1.0 / (1260.0 * x ** 5) - 1.0 / (1260.0 * r ** 5)
    return out


@nb.njit(cache=True)
def _control_lin_terms(g, alpha, m, min_mu, centre, s_mom, y_mom, hist_ptr, hist_val, hist_cnt):
    """Controls' NB negative log-likelihood (less alpha-free terms) and sum
    of weights mu/(1 + alpha mu) for gene ``g``, with mu = max(s m, min_mu)."""
    nll = 0.0
    weight = 0.0
    floor_log = math.log1p(alpha * min_mu)
    floor_w = min_mu / (1.0 + alpha * min_mu)
    am = alpha * m
    for b in range(centre.shape[0]):
        n_b = s_mom[0, b]
        if n_b == 0.0:
            continue
        c = centre[b]
        y0 = y_mom[0, g, b]
        if c * m < min_mu:
            nll += n_b * floor_log / alpha + y0 * floor_log
            weight += n_b * floor_w
            continue
        v = 1.0 + am * c
        f0 = math.log1p(am * c)
        f1 = am / v
        f2 = -(am * am) / (v * v)
        nll += (n_b * f0 + s_mom[1, b] * f1 + 0.5 * s_mom[2, b] * f2) / alpha
        nll += y0 * f0 + y_mom[1, g, b] * f1 + 0.5 * y_mom[2, g, b] * f2
        weight += n_b * c * m / v + s_mom[1, b] * m / (v * v) - s_mom[2, b] * alpha * m * m / (v * v * v)
    r = 1.0 / alpha
    for k in range(hist_ptr[g], hist_ptr[g + 1]):
        nll -= hist_cnt[k] * _lgamma_ratio_term(hist_val[k], r)
    return nll, weight


@nb.njit(cache=True)
def _cells_lin_terms(y, s, m, alpha, min_mu):
    """The same two quantities summed exactly over cells (counts ``y``,
    size factors ``s``)."""
    nll = 0.0
    weight = 0.0
    r = 1.0 / alpha
    for i in range(y.shape[0]):
        mu = s[i] * m
        if mu < min_mu:
            mu = min_mu
        f0 = math.log1p(alpha * mu)
        nll += f0 / alpha + y[i] * f0 - _lgamma_ratio_term(y[i], r)
        weight += mu / (1.0 + alpha * mu)
    return nll, weight


@nb.njit(cache=True)
def _two_group_objective(log_alpha, g, m_c, m_t, y_t, s_t, min_mu, centre, s_mom, y_mom,
                         hist_ptr, hist_val, hist_cnt, log_prior_mean, prior_var):
    """DESeq2's dispersion objective (to minimise): NB negative log-likelihood
    with the Cox-Reid term, plus the log-normal prior when ``prior_var`` > 0."""
    alpha = math.exp(log_alpha)
    nll_c, w_c = _control_lin_terms(g, alpha, m_c, min_mu, centre, s_mom, y_mom, hist_ptr, hist_val, hist_cnt)
    nll_t, w_t = _cells_lin_terms(y_t, s_t, m_t, alpha, min_mu)
    out = nll_c + nll_t + 0.5 * (math.log(w_c) + math.log(w_t))
    if prior_var > 0.0:
        d = log_alpha - log_prior_mean
        out += d * d / (2.0 * prior_var)
    return out


@nb.njit(parallel=True, cache=True)
def _two_group_dispersions(genes, m_c, m_t, y_t, s_t, min_mu, centre, s_mom, y_mom, hist_ptr, hist_val,
                           hist_cnt, log_lo, log_hi, log_prior_mean, prior_var, n_grid, tol):
    """Minimise the dispersion objective over log alpha in [log_lo, log_hi]
    for each gene in ``genes``: a grid, then golden-section search in the
    bracket around the grid's best point. ``y_t`` is (cells, genes)."""
    out = np.empty(genes.shape[0])
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    step = (log_hi - log_lo) / (n_grid - 1)
    for i in nb.prange(genes.shape[0]):
        g = genes[i]
        yg = np.ascontiguousarray(y_t[:, g])
        prior_mean = log_prior_mean[i] if prior_var > 0.0 else 0.0
        best = 0
        best_val = np.inf
        for k in range(n_grid):
            val = _two_group_objective(log_lo + k * step, g, m_c[g], m_t[g], yg, s_t, min_mu, centre, s_mom,
                                       y_mom, hist_ptr, hist_val, hist_cnt, prior_mean, prior_var)
            if val < best_val:
                best_val = val
                best = k
        lo = log_lo + max(best - 1, 0) * step
        hi = log_lo + min(best + 1, n_grid - 1) * step
        x1 = hi - invphi * (hi - lo)
        x2 = lo + invphi * (hi - lo)
        f1 = _two_group_objective(x1, g, m_c[g], m_t[g], yg, s_t, min_mu, centre, s_mom, y_mom,
                                  hist_ptr, hist_val, hist_cnt, prior_mean, prior_var)
        f2 = _two_group_objective(x2, g, m_c[g], m_t[g], yg, s_t, min_mu, centre, s_mom, y_mom,
                                  hist_ptr, hist_val, hist_cnt, prior_mean, prior_var)
        while hi - lo > tol:
            if f1 < f2:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - invphi * (hi - lo)
                f1 = _two_group_objective(x1, g, m_c[g], m_t[g], yg, s_t, min_mu, centre, s_mom, y_mom,
                                          hist_ptr, hist_val, hist_cnt, prior_mean, prior_var)
            else:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + invphi * (hi - lo)
                f2 = _two_group_objective(x2, g, m_c[g], m_t[g], yg, s_t, min_mu, centre, s_mom, y_mom,
                                          hist_ptr, hist_val, hist_cnt, prior_mean, prior_var)
        x = 0.5 * (lo + hi)
        if best_val < min(f1, f2):  # the optimum at a bound of the range
            x = log_lo + best * step
        out[i] = math.exp(x)
    return out


@nb.njit(cache=True)
def _control_score(g, a, alpha, min_mu, centre, s_mom, s_log, y_mom):
    """Controls' part of PyDESeq2's IRLS fixed-point equation for the
    intercept ``a``, and its derivative. A cell with s*exp(a) >= min_mu adds
    (y - mu)/(1 + alpha mu); a cell whose mean IRLS holds at min_mu adds
    W (log(min_mu/s) - a + (y - min_mu)/min_mu), W = min_mu/(1 + alpha min_mu)."""
    k = math.exp(a)
    score = 0.0
    slope = 0.0
    w_floor = min_mu / (1.0 + alpha * min_mu)
    ak = alpha * k
    for b in range(centre.shape[0]):
        n_b = s_mom[0, b]
        if n_b == 0.0:
            continue
        c = centre[b]
        y0 = y_mom[0, g, b]
        if c * k < min_mu:
            score += w_floor * (n_b * (math.log(min_mu) - a - 1.0) - s_log[b] + y0 / min_mu)
            slope -= n_b * w_floor
            continue
        y1 = y_mom[1, g, b]
        y2 = y_mom[2, g, b]
        v = 1.0 + ak * c
        # sum y/(1 + alpha mu): f(s) = 1/(1 + alpha k s)
        score += y0 / v - y1 * ak / (v * v) + y2 * ak * ak / (v * v * v)
        # sum mu/(1 + alpha mu): g(s) = k s/(1 + alpha k s)
        score -= n_b * k * c / v + s_mom[1, b] * k / (v * v) - s_mom[2, b] * alpha * k * k / (v * v * v)
        # d/da per cell: -mu (1 + alpha y)/(1 + alpha mu)^2; h(s) = k s/(1 + alpha k s)^2
        h0 = k * c / (v * v)
        h1 = k * (1.0 - ak * c) / (v * v * v)
        h2 = -alpha * k * k * (4.0 - 2.0 * ak * c) / (v * v * v * v)
        slope -= n_b * h0 + s_mom[1, b] * h1 + 0.5 * s_mom[2, b] * h2
        slope -= alpha * (y0 * h0 + y1 * h1 + 0.5 * y2 * h2)
    return score, slope


@nb.njit(cache=True)
def _cells_score(y, s, b, alpha, min_mu):
    """The perturbed cells' part of the same equation, summed exactly."""
    k = math.exp(b)
    score = 0.0
    slope = 0.0
    w_floor = min_mu / (1.0 + alpha * min_mu)
    for i in range(y.shape[0]):
        mu = s[i] * k
        if mu < min_mu:
            score += w_floor * (math.log(min_mu / s[i]) - b + (y[i] - min_mu) / min_mu)
            slope -= w_floor
        else:
            v = 1.0 + alpha * mu
            score += (y[i] - mu) / v
            slope -= mu * (1.0 + alpha * y[i]) / (v * v)
    return score, slope


@nb.njit(cache=True)
def _control_weight(a, alpha, centre, s_mom):
    """Sum over controls of mu/(1 + alpha mu) with the unthresholded mu = s exp(a)."""
    k = math.exp(a)
    out = 0.0
    for b in range(centre.shape[0]):
        n_b = s_mom[0, b]
        if n_b == 0.0:
            continue
        c = centre[b]
        v = 1.0 + alpha * k * c
        out += n_b * k * c / v + s_mom[1, b] * k / (v * v) - s_mom[2, b] * alpha * k * k / (v * v * v)
    return out


@nb.njit(parallel=True, cache=True)
def _two_group_lfc(genes, dispersion, a0, b0, y_t, s_t, min_mu, centre, s_mom, s_log, y_mom, ridge,
                   max_iter, tol):
    """PyDESeq2's IRLS solution for the control intercept a and the perturbed
    group's log mean b (LFC = b - a) at the given dispersions -- damped Newton
    on the fixed-point equations X'(W (z - X beta)) = ridge beta -- and the
    Wald standard error of b - a from the unthresholded means."""
    n = genes.shape[0]
    a_out = np.full(n, np.nan)
    b_out = np.full(n, np.nan)
    se_out = np.full(n, np.nan)
    converged = np.zeros(n, dtype=np.bool_)
    for i in nb.prange(n):
        g = genes[i]
        alpha = dispersion[i]
        yg = np.ascontiguousarray(y_t[:, g])
        a = a0[i]
        b = b0[i]
        for _ in range(max_iter):
            sc, dc = _control_score(g, a, alpha, min_mu, centre, s_mom, s_log, y_mom)
            st, dt = _cells_score(yg, s_t, b, alpha, min_mu)
            f1 = sc - ridge * (2.0 * a - b)
            f2 = st - ridge * (b - a)
            j11 = dc - 2.0 * ridge
            j22 = dt - ridge
            det = j11 * j22 - ridge * ridge
            if det == 0.0:
                break
            da = -(j22 * f1 - ridge * f2) / det
            db = -(j11 * f2 - ridge * f1) / det
            norm0 = abs(f1) + abs(f2)
            step = 1.0
            na = a
            nb_ = b
            for _h in range(30):
                na = min(max(a + step * da, -30.0), 30.0)
                nb_ = min(max(b + step * db, -30.0), 30.0)
                sc2, _dc = _control_score(g, na, alpha, min_mu, centre, s_mom, s_log, y_mom)
                st2, _dt = _cells_score(yg, s_t, nb_, alpha, min_mu)
                if abs(sc2 - ridge * (2.0 * na - nb_)) + abs(st2 - ridge * (nb_ - na)) <= norm0:
                    break
                step *= 0.5
            moved = max(abs(na - a), abs(nb_ - b))
            a = na
            b = nb_
            if moved < tol:
                converged[i] = True
                break
        k_t = math.exp(b)
        w_t = 0.0
        for j in range(yg.shape[0]):
            mu = s_t[j] * k_t
            w_t += mu / (1.0 + alpha * mu)
        w_c = _control_weight(a, alpha, centre, s_mom)
        # M = [[w_c + w_t, w_t], [w_t, w_t]]; H = (M + ridge I)^-1;
        # se^2 = (H M H)[1, 1], PyDESeq2's wald_test
        m11 = w_c + w_t + ridge
        m22 = w_t + ridge
        dm = m11 * m22 - w_t * w_t
        c1 = -w_t / dm
        c2 = m11 / dm
        var = c1 * c1 * (w_c + w_t) + 2.0 * c1 * c2 * w_t + c2 * c2 * w_t
        a_out[i] = a
        b_out[i] = b
        se_out[i] = math.sqrt(var) if var > 0.0 else np.nan
    return a_out, b_out, se_out, converged


@nb.njit(cache=True)
def _accumulate_control_summary(data, indices, indptr, bin_of_row, s_rows, centre, y_mom, small_hist,
                                big_genes, big_values, n_big):
    """Add a CSR block of control rows to the summary: count-weighted
    size-factor moments per (gene, bin), counts 1..small_hist.shape[1]-1 into
    the dense histogram, larger counts into the overflow buffers (returns
    their new length)."""
    vmax = small_hist.shape[1] - 1
    for r in range(indptr.shape[0] - 1):
        b = bin_of_row[r]
        d = s_rows[r] - centre[b]
        for j in range(indptr[r], indptr[r + 1]):
            y = data[j]
            if y == 0.0:
                continue
            g = indices[j]
            y_mom[0, g, b] += y
            y_mom[1, g, b] += y * d
            y_mom[2, g, b] += y * d * d
            v = int(y)
            if v <= vmax:
                small_hist[g, v] += 1.0
            else:
                big_genes[n_big] = g
                big_values[n_big] = y
                n_big += 1
    return n_big
