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


@nb.njit(parallel=True, cache=True)
def _nb_map_grid_search_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    log_trend: np.ndarray,
    log_alpha_grid: np.ndarray,
    prior_var: float,
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
    
    # Precompute alpha and log(alpha) values for grid
    alpha_grid = np.exp(log_alpha_grid)
    r_grid = 1.0 / alpha_grid
    log_r_grid = np.log(r_grid)
    gammaln_r_grid = np.empty(n_grid, dtype=np.float64)
    for k in range(n_grid):
        gammaln_r_grid[k] = math.lgamma(r_grid[k])
    
    # Parallelize over genes
    for g in nb.prange(n_genes):
        best_posterior = -np.inf
        best_k = 0
        log_trend_g = log_trend[g]
        Y_g = Y[:, g]
        mu_g = mu[:, g]
        
        # Precompute gammaln(Y_g + 1) for this gene - only done once, not n_grid times!
        gammaln_y_plus_1 = 0.0
        for i in range(n_cells):
            gammaln_y_plus_1 += math.lgamma(Y_g[i] + 1.0)
        
        for k in range(n_grid):
            log_alpha = log_alpha_grid[k]
            r = r_grid[k]
            log_r = log_r_grid[k]
            gammaln_r = gammaln_r_grid[k]
            
            # Compute NB log-likelihood inline for speed
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
            # Subtract gammaln(Y+1) term (precomputed)
            ll -= gammaln_y_plus_1
            
            # Add log-prior: -0.5 * (log_alpha - log_trend)^2 / prior_var
            log_prior = -0.5 * (log_alpha - log_trend_g) ** 2 / prior_var
            posterior = ll + log_prior
            
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
    fx = fw = fv = -_nb_posterior_with_cache_numba(Y_g, mu_g, x, log_trend_g, prior_var, gammaln_y_plus_1_sum)
    
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
        
        fu = -_nb_posterior_with_cache_numba(Y_g, mu_g, u, log_trend_g, prior_var, gammaln_y_plus_1_sum)
        
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
) -> float:
    """Compute NB posterior (log-likelihood + log-prior) for a single gene.
    
    Uses precomputed gammaln(Y+1) sum for efficiency.
    """
    n_cells = Y_g.shape[0]
    alpha = math.exp(log_alpha)
    r = 1.0 / alpha
    log_r = math.log(r)
    gammaln_r = math.lgamma(r)
    
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
    ll -= gammaln_y_plus_1_sum
    
    # Add log-prior
    log_prior = -0.5 * (log_alpha - log_trend_g) ** 2 / prior_var
    
    return ll + log_prior


@nb.njit(parallel=True, cache=True)
def _nb_map_grid_search_with_refinement_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    log_trend: np.ndarray,
    log_alpha_grid: np.ndarray,
    prior_var: float,
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
    
    # Precompute alpha and log(alpha) values for grid
    alpha_grid = np.exp(log_alpha_grid)
    r_grid = 1.0 / alpha_grid
    log_r_grid = np.log(r_grid)
    gammaln_r_grid = np.empty(n_grid, dtype=np.float64)
    for k in range(n_grid):
        gammaln_r_grid[k] = math.lgamma(r_grid[k])
    
    # Parallelize over genes
    for g in nb.prange(n_genes):
        best_posterior = -np.inf
        best_k = 0
        log_trend_g = log_trend[g]
        Y_g = Y[:, g]
        mu_g = mu[:, g]
        
        # Precompute gammaln(Y_g + 1) for this gene - reused in refinement
        gammaln_y_plus_1_sum = 0.0
        for i in range(n_cells):
            gammaln_y_plus_1_sum += math.lgamma(Y_g[i] + 1.0)
        
        # Stage 1: Grid search
        for k in range(n_grid):
            log_alpha = log_alpha_grid[k]
            r = r_grid[k]
            log_r = log_r_grid[k]
            gammaln_r = gammaln_r_grid[k]
            
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
            ll -= gammaln_y_plus_1_sum
            
            log_prior = -0.5 * (log_alpha - log_trend_g) ** 2 / prior_var
            posterior = ll + log_prior
            
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
                    a, b, tol=tol, max_iter=max_refine_iter
                )
                
                # Final sanity check: ensure refinement is actually better than grid
                refined_posterior = _nb_posterior_with_cache_numba(Y_g, mu_g, refined_log_alpha, log_trend_g, prior_var, gammaln_y_plus_1_sum)
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
# IRLS batch processing kernels
# =============================================================================

@nb.njit(cache=True)
def _wls_solve_2x2_numba(
    W: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    ridge: float,
) -> tuple:
    """Solve weighted least squares for 2-parameter model (intercept + perturbation).
    
    Optimized for the common case of [1, perturbation_indicator] design matrix.
    Uses direct 2x2 matrix inversion which is faster than general solve.
    
    Parameters
    ----------
    W : (n_samples,)
        IRLS weights for current gene.
    z : (n_samples,)
        Working response for current gene.
    X : (n_samples, 2)
        Design matrix [1, perturbation_indicator].
    ridge : float
        Ridge penalty for regularization.
        
    Returns
    -------
    beta : (2,)
        Fitted coefficients [intercept, perturbation_effect].
    se : (2,)
        Standard errors.
    """
    n_samples = W.shape[0]
    
    # Compute X'WX elements directly (2x2 matrix)
    xtwx_00 = 0.0  # sum(W)
    xtwx_01 = 0.0  # sum(W * x1)
    xtwx_11 = 0.0  # sum(W * x1^2)
    xtwz_0 = 0.0   # sum(W * z)
    xtwz_1 = 0.0   # sum(W * z * x1)
    
    for i in range(n_samples):
        w_i = W[i]
        z_i = z[i]
        x1_i = X[i, 1]  # perturbation indicator
        
        xtwx_00 += w_i
        xtwx_01 += w_i * x1_i
        xtwx_11 += w_i * x1_i * x1_i
        xtwz_0 += w_i * z_i
        xtwz_1 += w_i * z_i * x1_i
    
    # Add ridge penalty
    xtwx_00 += ridge
    xtwx_11 += ridge
    
    # Direct 2x2 inverse
    det = xtwx_00 * xtwx_11 - xtwx_01 * xtwx_01
    if abs(det) < 1e-12:
        det = 1e-12
    
    inv_00 = xtwx_11 / det
    inv_01 = -xtwx_01 / det
    inv_11 = xtwx_00 / det
    
    # beta = inv(X'WX) @ X'Wz
    beta_0 = inv_00 * xtwz_0 + inv_01 * xtwz_1
    beta_1 = inv_01 * xtwz_0 + inv_11 * xtwz_1
    
    # SE = sqrt(diag(inv(X'WX)))
    se_0 = np.sqrt(max(inv_00, 1e-12))
    se_1 = np.sqrt(max(inv_11, 1e-12))
    
    beta = np.array([beta_0, beta_1])
    se = np.array([se_0, se_1])
    
    return beta, se


# =============================================================================
# Wilcoxon rank-sum test kernels
# =============================================================================

@nb.njit(parallel=True, cache=True)
def _rankdata_2d_numba(arr: np.ndarray, ranks_out: np.ndarray) -> None:
    """Compute average ranks for each column of a 2D array in parallel.
    
    Ranks are computed along axis=0 (within each column), matching
    scipy.stats.rankdata(arr, axis=0, method='average').
    
    Parameters
    ----------
    arr : (n_samples, n_genes)
        Input matrix to rank.
    ranks_out : (n_samples, n_genes)
        Output array for ranks (modified in-place).
    """
    n_samples, n_genes = arr.shape
    
    for g in nb.prange(n_genes):
        col = arr[:, g].copy()
        order = np.argsort(col)
        
        # Assign ranks with tie averaging
        i = 0
        while i < n_samples:
            j = i
            while j < n_samples - 1 and col[order[j + 1]] == col[order[i]]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks_out[order[k], g] = avg_rank
            i = j + 1


@nb.njit(parallel=True, cache=True)
def _tie_correction_numba(ranks: np.ndarray, correction_out: np.ndarray) -> None:
    """Compute tie correction factors for Wilcoxon test in parallel.
    
    The tie correction factor is: 1 - sum(t^3 - t) / (n^3 - n)
    where t is the count of each tied group.
    
    Parameters
    ----------
    ranks : (n_samples, n_genes)
        Rank matrix.
    correction_out : (n_genes,)
        Output array for correction factors (modified in-place).
    """
    n_samples, n_genes = ranks.shape
    size = float(n_samples)
    denom = size ** 3 - size
    
    for g in nb.prange(n_genes):
        if denom <= 0 or n_samples < 2:
            correction_out[g] = 1.0
            continue
        
        # Sort column to find tie groups
        col = ranks[:, g].copy()
        col.sort()
        
        # Count ties and accumulate t^3 - t
        tie_sum = 0.0
        i = 0
        while i < n_samples:
            j = i
            while j < n_samples - 1 and col[j + 1] == col[j]:
                j += 1
            count = float(j - i + 1)
            if count > 1:
                tie_sum += count ** 3 - count
            i = j + 1
        
        correction_out[g] = 1.0 - tie_sum / denom


@nb.njit(parallel=True, cache=True)
def _compute_rank_sums_batch_numba(
    ranks: np.ndarray,
    pert_indices_flat: np.ndarray,
    pert_offsets: np.ndarray,
    pert_counts: np.ndarray,
    n_pert_groups: int,
    control_n: int,
    tie_correction: np.ndarray,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
) -> None:
    """Compute Wilcoxon statistics for multiple perturbation groups sharing control.
    
    This batched version computes statistics for all perturbations at once,
    reusing the pre-computed ranks matrix. Each perturbation group's cells
    are specified by slices into pert_indices_flat.
    
    Parameters
    ----------
    ranks : (n_total_cells, n_genes)
        Pre-computed ranks for all cells (control + all perturbation cells).
        Control cells are at indices [0:control_n).
    pert_indices_flat : (sum of all pert cell counts,)
        Flattened array of cell indices for all perturbation groups.
    pert_offsets : (n_pert_groups,)
        Start offset into pert_indices_flat for each perturbation group.
    pert_counts : (n_pert_groups,)
        Number of cells in each perturbation group.
    n_pert_groups : int
        Number of perturbation groups.
    control_n : int
        Number of control cells.
    tie_correction : (n_genes,)
        Tie correction factors (computed from all-cell ranks).
    u_stat_out : (n_pert_groups, n_genes)
        Output U-statistics.
    z_score_out : (n_pert_groups, n_genes)
        Output z-scores.
    pvalue_out : (n_pert_groups, n_genes)
        Output p-values.
    """
    n_genes = ranks.shape[1]
    control_n_f = float(control_n)
    
    # Parallelize over perturbation groups
    for p_idx in nb.prange(n_pert_groups):
        n_pert = pert_counts[p_idx]
        n_pert_f = float(n_pert)
        n_total = n_pert_f + control_n_f
        
        # Get slice of cell indices for this perturbation
        start_idx = pert_offsets[p_idx]
        end_idx = start_idx + n_pert
        
        expected = n_pert_f * (n_total + 1.0) / 2.0
        
        for g in range(n_genes):
            # Sum ranks for perturbation cells
            rank_sum = 0.0
            for i in range(start_idx, end_idx):
                cell_idx = pert_indices_flat[i]
                rank_sum += ranks[cell_idx, g]
            
            # U-statistic
            u_stat = rank_sum - n_pert_f * (n_pert_f + 1.0) / 2.0
            u_stat_out[p_idx, g] = u_stat
            
            # Standard deviation with tie correction
            std = math.sqrt(
                tie_correction[g] * n_pert_f * control_n_f * (n_total + 1.0) / 12.0
            )
            
            if std > 0:
                z = (rank_sum - expected) / std
                z_score_out[p_idx, g] = z
                abs_z = abs(z)
                pvalue_out[p_idx, g] = math.erfc(abs_z / math.sqrt(2.0))
            else:
                z_score_out[p_idx, g] = 0.0
                pvalue_out[p_idx, g] = 1.0


# =============================================================================
# Optimized Wilcoxon kernels for sparse data with zero-separation
# =============================================================================

# Threshold for zero-separation optimization: if >= this fraction of values are zero,
# use the optimized zero-separated ranking. Otherwise use standard full ranking.
_ZERO_PARTITION_THRESHOLD = 0.5


@nb.njit(cache=True)
def _merge_sorted_with_ranks_numba(
    sorted_a: np.ndarray,
    sorted_b: np.ndarray,
    ranks_a_out: np.ndarray,
    ranks_b_out: np.ndarray,
    zero_offset: int,
) -> float:
    """Merge two sorted arrays and compute average ranks with tie handling.
    
    Computes ranks as if the arrays were concatenated and ranked together,
    using average rank for ties. Also computes tie correction factor.
    
    Parameters
    ----------
    sorted_a : (n_a,)
        First sorted array (e.g., control non-zero values).
    sorted_b : (n_b,)
        Second sorted array (e.g., perturbation non-zero values).
    ranks_a_out : (n_a,)
        Output ranks for sorted_a elements.
    ranks_b_out : (n_b,)
        Output ranks for sorted_b elements.
    zero_offset : int
        Number of zeros (their ranks are 1..zero_offset, avg = (zero_offset+1)/2).
        Non-zero ranks start at zero_offset + 1.
        
    Returns
    -------
    float
        Tie correction factor: 1 - sum(t^3 - t) / (n^3 - n) where t is tie group size.
    """
    n_a = sorted_a.shape[0]
    n_b = sorted_b.shape[0]
    n_total = n_a + n_b + zero_offset
    
    if n_a == 0 and n_b == 0:
        # Only zeros - tie correction for all-same values
        if zero_offset > 0:
            denom = float(zero_offset) ** 3 - float(zero_offset)
            if denom > 0:
                tie_sum = float(zero_offset) ** 3 - float(zero_offset)
                return 1.0 - tie_sum / denom
        return 1.0
    
    # Merge and track which array each element came from
    # merged_vals[i] = value, merged_src[i] = 0 for a, 1 for b
    # merged_orig_idx[i] = original index in source array
    merged_len = n_a + n_b
    merged_vals = np.empty(merged_len, dtype=np.float64)
    merged_src = np.empty(merged_len, dtype=np.int32)
    merged_orig_idx = np.empty(merged_len, dtype=np.int64)
    
    # Standard merge
    i, j, k = 0, 0, 0
    while i < n_a and j < n_b:
        if sorted_a[i] <= sorted_b[j]:
            merged_vals[k] = sorted_a[i]
            merged_src[k] = 0
            merged_orig_idx[k] = i
            i += 1
        else:
            merged_vals[k] = sorted_b[j]
            merged_src[k] = 1
            merged_orig_idx[k] = j
            j += 1
        k += 1
    while i < n_a:
        merged_vals[k] = sorted_a[i]
        merged_src[k] = 0
        merged_orig_idx[k] = i
        i += 1
        k += 1
    while j < n_b:
        merged_vals[k] = sorted_b[j]
        merged_src[k] = 1
        merged_orig_idx[k] = j
        j += 1
        k += 1
    
    # Assign ranks with tie averaging
    # Ranks for non-zeros start at (zero_offset + 1)
    tie_sum = 0.0
    
    # Add zero-group tie contribution
    if zero_offset > 1:
        t = float(zero_offset)
        tie_sum += t ** 3 - t
    
    pos = 0
    while pos < merged_len:
        # Find all elements with same value (tie group)
        tie_start = pos
        while pos < merged_len - 1 and merged_vals[pos + 1] == merged_vals[tie_start]:
            pos += 1
        tie_end = pos
        
        # Tie group size
        tie_count = tie_end - tie_start + 1
        if tie_count > 1:
            t = float(tie_count)
            tie_sum += t ** 3 - t
        
        # Average rank for this tie group
        # Ranks are 1-based: first non-zero gets rank (zero_offset + 1)
        first_rank = zero_offset + tie_start + 1
        last_rank = zero_offset + tie_end + 1
        avg_rank = (first_rank + last_rank) / 2.0
        
        # Assign to output arrays
        for idx in range(tie_start, tie_end + 1):
            if merged_src[idx] == 0:
                ranks_a_out[merged_orig_idx[idx]] = avg_rank
            else:
                ranks_b_out[merged_orig_idx[idx]] = avg_rank
        
        pos += 1
    
    # Compute tie correction
    n_total_f = float(n_total)
    denom = n_total_f ** 3 - n_total_f
    if denom > 0:
        return 1.0 - tie_sum / denom
    return 1.0


@nb.njit(parallel=True, cache=True)
def _wilcoxon_sparse_batch_numba(
    control_dense: np.ndarray,
    pert_dense: np.ndarray,
    valid_genes: np.ndarray,
    tie_correct: bool,
    zero_threshold: float,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Compute Wilcoxon statistics for all genes using zero-separation optimization.
    
    For sparse data, separates zeros from non-zeros:
    - Zeros form a tied group with known average rank = (n_zeros + 1) / 2
    - Only non-zero values need sorting/ranking
    - Reduces computational work by 10-100x for sparse genes
    
    Falls back to standard full ranking when zero fraction < threshold.
    
    Parameters
    ----------
    control_dense : (n_control, n_genes)
        Dense control expression matrix.
    pert_dense : (n_pert, n_genes)
        Dense perturbation expression matrix.
    valid_genes : (n_genes,)
        Boolean mask of genes to process.
    tie_correct : bool
        Whether to apply tie correction.
    zero_threshold : float
        Minimum fraction of zeros to use zero-separation (e.g., 0.5).
    u_stat_out : (n_genes,)
        Output U-statistics.
    z_score_out : (n_genes,)
        Output z-scores.
    pvalue_out : (n_genes,)
        Output p-values.
    effect_out : (n_genes,)
        Output effect sizes.
    """
    n_control = control_dense.shape[0]
    n_pert = pert_dense.shape[0]
    n_genes = control_dense.shape[1]
    n_total = n_control + n_pert
    
    n_control_f = float(n_control)
    n_pert_f = float(n_pert)
    n_total_f = float(n_total)
    
    for g in nb.prange(n_genes):
        if not valid_genes[g]:
            u_stat_out[g] = 0.0
            z_score_out[g] = 0.0
            pvalue_out[g] = 1.0
            effect_out[g] = 0.0
            continue
        
        # Extract gene column
        ctrl_col = control_dense[:, g]
        pert_col = pert_dense[:, g]
        
        # Count zeros
        n_ctrl_zeros = 0
        n_pert_zeros = 0
        for i in range(n_control):
            if ctrl_col[i] == 0.0:
                n_ctrl_zeros += 1
        for i in range(n_pert):
            if pert_col[i] == 0.0:
                n_pert_zeros += 1
        
        n_zeros = n_ctrl_zeros + n_pert_zeros
        zero_frac = float(n_zeros) / n_total_f
        
        # Decide whether to use zero-separation
        use_zero_sep = zero_frac >= zero_threshold
        
        if use_zero_sep and n_zeros < n_total:
            # Zero-separation path: only sort non-zeros
            n_ctrl_nonzero = n_control - n_ctrl_zeros
            n_pert_nonzero = n_pert - n_pert_zeros
            
            # Extract non-zeros
            ctrl_nonzero = np.empty(n_ctrl_nonzero, dtype=np.float64)
            pert_nonzero = np.empty(n_pert_nonzero, dtype=np.float64)
            
            idx = 0
            for i in range(n_control):
                if ctrl_col[i] != 0.0:
                    ctrl_nonzero[idx] = ctrl_col[i]
                    idx += 1
            
            idx = 0
            for i in range(n_pert):
                if pert_col[i] != 0.0:
                    pert_nonzero[idx] = pert_col[i]
                    idx += 1
            
            # Sort non-zeros
            ctrl_sorted = np.sort(ctrl_nonzero)
            pert_sorted = np.sort(pert_nonzero)
            
            # Get ranks via merge
            ctrl_ranks = np.empty(n_ctrl_nonzero, dtype=np.float64)
            pert_ranks = np.empty(n_pert_nonzero, dtype=np.float64)
            
            tie_corr = _merge_sorted_with_ranks_numba(
                ctrl_sorted, pert_sorted, ctrl_ranks, pert_ranks, n_zeros
            )
            
            if not tie_correct:
                tie_corr = 1.0
            
            # Compute rank sum for perturbation
            # Zero cells in perturbation get average rank (n_zeros + 1) / 2
            zero_avg_rank = (float(n_zeros) + 1.0) / 2.0
            rank_sum = float(n_pert_zeros) * zero_avg_rank
            for i in range(n_pert_nonzero):
                rank_sum += pert_ranks[i]
        
        else:
            # Standard full ranking path
            combined = np.empty(n_total, dtype=np.float64)
            for i in range(n_pert):
                combined[i] = pert_col[i]
            for i in range(n_control):
                combined[n_pert + i] = ctrl_col[i]
            
            # Sort and get order
            order = np.argsort(combined)
            
            # Assign ranks with tie averaging
            ranks = np.empty(n_total, dtype=np.float64)
            tie_sum = 0.0
            pos = 0
            while pos < n_total:
                tie_start = pos
                while pos < n_total - 1 and combined[order[pos + 1]] == combined[order[tie_start]]:
                    pos += 1
                tie_end = pos
                
                tie_count = tie_end - tie_start + 1
                if tie_count > 1:
                    t = float(tie_count)
                    tie_sum += t ** 3 - t
                
                avg_rank = (tie_start + tie_end + 2) / 2.0  # 1-based
                for idx in range(tie_start, tie_end + 1):
                    ranks[order[idx]] = avg_rank
                
                pos += 1
            
            # Tie correction
            denom = n_total_f ** 3 - n_total_f
            if denom > 0 and tie_correct:
                tie_corr = 1.0 - tie_sum / denom
            else:
                tie_corr = 1.0
            
            # Rank sum for perturbation (first n_pert elements)
            rank_sum = 0.0
            for i in range(n_pert):
                rank_sum += ranks[i]
        
        # Compute statistics
        expected = n_pert_f * (n_total_f + 1.0) / 2.0
        u_stat = rank_sum - n_pert_f * (n_pert_f + 1.0) / 2.0
        
        std = math.sqrt(tie_corr * n_pert_f * n_control_f * (n_total_f + 1.0) / 12.0)
        
        if std > 0.0:
            z = (rank_sum - expected) / std
            abs_z = abs(z)
            pval = math.erfc(abs_z / math.sqrt(2.0))
        else:
            z = 0.0
            pval = 1.0
        
        # Effect size: U / (n1 * n2) - 0.5
        if n_pert_f > 0 and n_control_f > 0:
            effect = u_stat / (n_pert_f * n_control_f) - 0.5
        else:
            effect = 0.0
        
        u_stat_out[g] = u_stat
        z_score_out[g] = z
        pvalue_out[g] = pval
        effect_out[g] = effect


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


@nb.njit(parallel=True, cache=True)
def _wilcoxon_presorted_ctrl_numba(
    control_dense: np.ndarray,
    ctrl_sorted_flat: np.ndarray,
    ctrl_offsets: np.ndarray,
    ctrl_n_nonzero: np.ndarray,
    ctrl_n_zeros: np.ndarray,
    pert_dense: np.ndarray,
    valid_genes: np.ndarray,
    tie_correct: bool,
    zero_threshold: float,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Wilcoxon test reusing pre-sorted control non-zeros.

    For the zero-separation path (majority of genes in sparse data),
    skips the O(n_ctrl * log(n_ctrl)) sort per group, using the
    pre-sorted array from ``_presort_control_nonzeros`` instead.
    Falls back to full sorting for the rare low-zero-fraction genes.
    """
    n_control = control_dense.shape[0]
    n_pert = pert_dense.shape[0]
    n_genes = pert_dense.shape[1]
    n_total = n_control + n_pert

    n_control_f = float(n_control)
    n_pert_f = float(n_pert)
    n_total_f = float(n_total)

    for g in nb.prange(n_genes):
        if not valid_genes[g]:
            u_stat_out[g] = 0.0
            z_score_out[g] = 0.0
            pvalue_out[g] = 1.0
            effect_out[g] = 0.0
            continue

        pert_col = pert_dense[:, g]

        # Count pert zeros
        n_pert_zeros = 0
        for i in range(n_pert):
            if pert_col[i] == 0.0:
                n_pert_zeros += 1

        n_zeros = ctrl_n_zeros[g] + n_pert_zeros

        if n_zeros < n_total:
            # --- Zero-separation with pre-sorted control (always) ---
            # Removing the zero_threshold gate: the merge walk is
            # O(n_ctrl_nz + n_pert_nz) which is always better than
            # the O(n_total * log(n_total)) argsort fallback.
            n_ctrl_nz = ctrl_n_nonzero[g]
            n_pert_nonzero = n_pert - n_pert_zeros

            # Extract and sort pert non-zeros (typically very few values)
            pert_nonzero = np.empty(n_pert_nonzero, dtype=np.float64)
            idx = 0
            for i in range(n_pert):
                if pert_col[i] != 0.0:
                    pert_nonzero[idx] = pert_col[i]
                    idx += 1
            pert_sorted = np.sort(pert_nonzero)

            # Use pre-sorted control non-zeros (no sort needed!)
            start = ctrl_offsets[g]
            ctrl_sorted = ctrl_sorted_flat[start:start + n_ctrl_nz]

            ctrl_ranks = np.empty(n_ctrl_nz, dtype=np.float64)
            pert_ranks = np.empty(n_pert_nonzero, dtype=np.float64)

            tie_corr = _merge_sorted_with_ranks_numba(
                ctrl_sorted, pert_sorted, ctrl_ranks, pert_ranks, n_zeros
            )

            if not tie_correct:
                tie_corr = 1.0

            zero_avg_rank = (float(n_zeros) + 1.0) / 2.0
            rank_sum = float(n_pert_zeros) * zero_avg_rank
            for i in range(n_pert_nonzero):
                rank_sum += pert_ranks[i]

        else:
            # All values are zero: rank_sum equals expected, U = expected.
            rank_sum = n_pert_f * (n_total_f + 1.0) / 2.0
            tie_corr = 0.0  # std will be 0 → z = 0, p = 1

        # Statistics
        expected = n_pert_f * (n_total_f + 1.0) / 2.0
        u_stat = rank_sum - n_pert_f * (n_pert_f + 1.0) / 2.0

        std = math.sqrt(tie_corr * n_pert_f * n_control_f * (n_total_f + 1.0) / 12.0)

        if std > 0.0:
            z = (rank_sum - expected) / std
            abs_z = abs(z)
            pval = math.erfc(abs_z / math.sqrt(2.0))
        else:
            z = 0.0
            pval = 1.0

        if n_pert_f > 0 and n_control_f > 0:
            effect = u_stat / (n_pert_f * n_control_f) - 0.5
        else:
            effect = 0.0

        u_stat_out[g] = u_stat
        z_score_out[g] = z
        pvalue_out[g] = pval
        effect_out[g] = effect


@nb.njit(parallel=True, cache=True)
def _wilcoxon_all_perts_numba(
    control_dense: np.ndarray,
    all_pert_dense: np.ndarray,
    pert_masks: np.ndarray,
    pert_counts: np.ndarray,
    valid_masks: np.ndarray,
    tie_correct: bool,
    zero_threshold: float,
    u_stat_out: np.ndarray,
    z_score_out: np.ndarray,
    pvalue_out: np.ndarray,
    effect_out: np.ndarray,
) -> None:
    """Compute Wilcoxon statistics for all perturbations and genes in parallel.
    
    This is the main optimized kernel that replaces ThreadPoolExecutor.
    Parallelizes over genes (inner loop) for better cache locality.
    
    Parameters
    ----------
    control_dense : (n_control, n_chunk_genes)
        Dense control expression matrix for this chunk.
    all_pert_dense : (n_total_cells, n_chunk_genes)
        Dense expression matrix for all cells.
    pert_masks : (n_perts, n_total_cells)
        Boolean masks for each perturbation (row = perturbation, col = cell).
    pert_counts : (n_perts,)
        Number of cells in each perturbation.
    valid_masks : (n_perts, n_chunk_genes)
        Boolean masks for valid genes per perturbation.
    tie_correct : bool
        Whether to apply tie correction.
    zero_threshold : float
        Minimum fraction of zeros to use zero-separation.
    u_stat_out : (n_perts, n_chunk_genes)
        Output U-statistics.
    z_score_out : (n_perts, n_chunk_genes)
        Output z-scores.
    pvalue_out : (n_perts, n_chunk_genes)
        Output p-values.
    effect_out : (n_perts, n_chunk_genes)
        Output effect sizes.
    """
    n_perts = pert_counts.shape[0]
    n_control = control_dense.shape[0]
    n_chunk_genes = control_dense.shape[1]
    n_total_cells = all_pert_dense.shape[0]
    
    n_control_f = float(n_control)
    
    # Process each perturbation sequentially, genes in parallel within
    for p_idx in range(n_perts):
        n_pert = pert_counts[p_idx]
        n_pert_f = float(n_pert)
        n_total = n_pert + n_control
        n_total_f = float(n_total)
        
        # Extract perturbation cells for this group
        pert_dense = np.empty((n_pert, n_chunk_genes), dtype=np.float64)
        cell_idx = 0
        for i in range(n_total_cells):
            if pert_masks[p_idx, i]:
                for g in range(n_chunk_genes):
                    pert_dense[cell_idx, g] = all_pert_dense[i, g]
                cell_idx += 1
        
        # Process genes in parallel
        for g in nb.prange(n_chunk_genes):
            if not valid_masks[p_idx, g]:
                u_stat_out[p_idx, g] = 0.0
                z_score_out[p_idx, g] = 0.0
                pvalue_out[p_idx, g] = 1.0
                effect_out[p_idx, g] = 0.0
                continue
            
            # Extract gene column
            ctrl_col = control_dense[:, g]
            pert_col = pert_dense[:, g]
            
            # Count zeros
            n_ctrl_zeros = 0
            n_pert_zeros = 0
            for i in range(n_control):
                if ctrl_col[i] == 0.0:
                    n_ctrl_zeros += 1
            for i in range(n_pert):
                if pert_col[i] == 0.0:
                    n_pert_zeros += 1
            
            n_zeros = n_ctrl_zeros + n_pert_zeros
            zero_frac = float(n_zeros) / n_total_f
            
            # Decide whether to use zero-separation
            use_zero_sep = zero_frac >= zero_threshold
            
            if use_zero_sep and n_zeros < n_total:
                # Zero-separation path
                n_ctrl_nonzero = n_control - n_ctrl_zeros
                n_pert_nonzero = n_pert - n_pert_zeros
                
                ctrl_nonzero = np.empty(n_ctrl_nonzero, dtype=np.float64)
                pert_nonzero = np.empty(n_pert_nonzero, dtype=np.float64)
                
                idx = 0
                for i in range(n_control):
                    if ctrl_col[i] != 0.0:
                        ctrl_nonzero[idx] = ctrl_col[i]
                        idx += 1
                
                idx = 0
                for i in range(n_pert):
                    if pert_col[i] != 0.0:
                        pert_nonzero[idx] = pert_col[i]
                        idx += 1
                
                ctrl_sorted = np.sort(ctrl_nonzero)
                pert_sorted = np.sort(pert_nonzero)
                
                ctrl_ranks = np.empty(n_ctrl_nonzero, dtype=np.float64)
                pert_ranks = np.empty(n_pert_nonzero, dtype=np.float64)
                
                tie_corr = _merge_sorted_with_ranks_numba(
                    ctrl_sorted, pert_sorted, ctrl_ranks, pert_ranks, n_zeros
                )
                
                if not tie_correct:
                    tie_corr = 1.0
                
                zero_avg_rank = (float(n_zeros) + 1.0) / 2.0
                rank_sum = float(n_pert_zeros) * zero_avg_rank
                for i in range(n_pert_nonzero):
                    rank_sum += pert_ranks[i]
            
            else:
                # Standard full ranking
                combined = np.empty(n_total, dtype=np.float64)
                for i in range(n_pert):
                    combined[i] = pert_col[i]
                for i in range(n_control):
                    combined[n_pert + i] = ctrl_col[i]
                
                order = np.argsort(combined)
                ranks = np.empty(n_total, dtype=np.float64)
                tie_sum = 0.0
                pos = 0
                while pos < n_total:
                    tie_start = pos
                    while pos < n_total - 1 and combined[order[pos + 1]] == combined[order[tie_start]]:
                        pos += 1
                    tie_end = pos
                    
                    tie_count = tie_end - tie_start + 1
                    if tie_count > 1:
                        t = float(tie_count)
                        tie_sum += t ** 3 - t
                    
                    avg_rank = (tie_start + tie_end + 2) / 2.0
                    for idx in range(tie_start, tie_end + 1):
                        ranks[order[idx]] = avg_rank
                    
                    pos += 1
                
                denom = n_total_f ** 3 - n_total_f
                if denom > 0 and tie_correct:
                    tie_corr = 1.0 - tie_sum / denom
                else:
                    tie_corr = 1.0
                
                rank_sum = 0.0
                for i in range(n_pert):
                    rank_sum += ranks[i]
            
            # Statistics
            expected = n_pert_f * (n_total_f + 1.0) / 2.0
            u_stat = rank_sum - n_pert_f * (n_pert_f + 1.0) / 2.0
            
            std = math.sqrt(tie_corr * n_pert_f * n_control_f * (n_total_f + 1.0) / 12.0)
            
            if std > 0.0:
                z = (rank_sum - expected) / std
                abs_z = abs(z)
                pval = math.erfc(abs_z / math.sqrt(2.0))
            else:
                z = 0.0
                pval = 1.0
            
            if n_pert_f > 0 and n_control_f > 0:
                effect = u_stat / (n_pert_f * n_control_f) - 0.5
            else:
                effect = 0.0
            
            u_stat_out[p_idx, g] = u_stat
            z_score_out[p_idx, g] = z
            pvalue_out[p_idx, g] = pval
            effect_out[p_idx, g] = effect


@nb.njit(cache=True)
def _rank_sum_pert_bsearch_numba(
    ctrl_sorted: np.ndarray,
    pert_sorted: np.ndarray,
    n_zeros: int,
    ctrl_tie_sum: float,
) -> tuple:
    """Binary-search Wilcoxon rank sum for pert non-zeros vs pre-sorted ctrl.

    Replaces the O(n_ctrl_nz + n_pert_nz) merge walk in
    ``_merge_sorted_with_ranks_numba`` with an O(n_pert_nz * log(n_ctrl_nz))
    binary-search pass.  For CRISPR datasets where n_ctrl_nz >> n_pert_nz
    (e.g. 140 K ctrl vs 11 pert non-zeros), this is ~750x faster per gene
    per perturbation.

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
    """Non-parallel single-perturbation Wilcoxon kernel for use inside prange.

    Identical logic to ``_wilcoxon_presorted_ctrl_numba`` but without
    ``parallel=True`` so it can be safely called from within a prange loop
    (Numba does not support nested parallel launches).

    The zero-separation path now uses ``_rank_sum_pert_bsearch_numba`` instead
    of ``_merge_sorted_with_ranks_numba``, giving O(n_pert_nz * log(n_ctrl_nz))
    complexity instead of O(n_ctrl_nz + n_pert_nz) (~750x speedup for typical
    CRISPR datasets with large control groups).
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
            effect_out[g] = 0.0
            continue

        pert_col = pert_dense[:, g]

        # Count pert zeros
        n_pert_zeros = 0
        for i in range(n_pert):
            if pert_col[i] == 0.0:
                n_pert_zeros += 1

        n_zeros = ctrl_n_zeros[g] + n_pert_zeros

        if n_zeros < n_total:
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

            rank_sum_nz, tie_corr = _rank_sum_pert_bsearch_numba(
                ctrl_sorted, pert_sorted, n_zeros, ctrl_tie_sums[g]
            )

            if not tie_correct:
                tie_corr = 1.0

            zero_avg_rank = (float(n_zeros) + 1.0) / 2.0
            rank_sum = float(n_pert_zeros) * zero_avg_rank + rank_sum_nz

        else:
            # All values are zero: rank_sum equals expected, U = expected.
            rank_sum = n_pert_f * (n_total_f + 1.0) / 2.0
            tie_corr = 0.0  # std will be 0 → z = 0, p = 1

        # Statistics
        expected = n_pert_f * (n_total_f + 1.0) / 2.0
        u_stat = rank_sum - n_pert_f * (n_pert_f + 1.0) / 2.0

        std = math.sqrt(tie_corr * n_pert_f * n_control_f * (n_total_f + 1.0) / 12.0)

        if std > 0.0:
            z = (rank_sum - expected) / std
            abs_z = abs(z)
            pval = math.erfc(abs_z / math.sqrt(2.0))
        else:
            z = 0.0
            pval = 1.0

        if n_pert_f > 0 and n_control_f > 0:
            effect = u_stat / (n_pert_f * n_control_f) - 0.5
        else:
            effect = 0.0

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

    Replaces 2254 sequential ``_wilcoxon_presorted_ctrl_numba`` calls per
    gene chunk with a single ``prange`` over all perturbation groups.  Each
    parallel thread handles one perturbation and iterates over genes
    sequentially, eliminating ~2253 redundant Numba thread-pool launches.

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
    ``sum_b U_b / sum_b (n1_b * n0_b) - 0.5``.

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
            effect_out[g] = 0.0
            continue

        num = 0.0        # sum_b (R_b - E_b)
        var = 0.0        # sum_b Var_b
        u_sum = 0.0      # sum_b U_b
        n1n0_sum = 0.0   # sum_b n1_b * n0_b

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

        if var > 0.0:
            z = num / math.sqrt(var)
            pval = math.erfc(abs(z) / sqrt2)
        else:
            z = 0.0
            pval = 1.0

        if n1n0_sum > 0.0:
            effect = u_sum / n1n0_sum - 0.5
        else:
            effect = 0.0

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
    se : (n_features, n_genes)
        Standard errors.
    converged : (n_genes,)
        Convergence flags.
    n_iter : (n_genes,)
        Number of iterations.
    """
    n_samples, n_genes = Y.shape
    n_features = X.shape[1]
    
    beta = np.copy(beta_init)
    se = np.full((n_features, n_genes), np.inf, dtype=np.float64)
    converged = np.zeros(n_genes, dtype=nb.boolean)
    n_iter = np.zeros(n_genes, dtype=np.int32)
    
    log_min_mu = np.log(min_mu)
    
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
                eta_i = min(max(eta_i, log_min_mu), 20.0)
                mu_i = np.exp(eta_i)
                mu_i = max(mu_i, min_mu)
                mu_g[i] = mu_i
                
                # Weight: W = mu^2 / (mu + alpha * mu^2)
                var_i = mu_i + alpha_g * mu_i * mu_i
                W_g[i] = (mu_i * mu_i) / max(var_i, min_mu)
                
                # Working response: z = eta + (y - mu) / mu
                z_g[i] = eta_i + (Y[i, g] - mu_i) / max(mu_i, min_mu) - offset[i]
            
            # Solve WLS: beta_new = (X'WX + ridge*I)^{-1} X'Wz
            # For 2-feature case, use direct formula
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
                
                det = xtwx_00 * xtwx_11 - xtwx_01 * xtwx_01
                if abs(det) < 1e-12:
                    det = 1e-12
                
                beta_new_0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
                beta_new_1 = (-xtwx_01 * xtwz_0 + xtwx_00 * xtwz_1) / det
                
                # Check convergence
                diff = max(abs(beta_new_0 - beta_g[0]), abs(beta_new_1 - beta_g[1]))
                beta_g[0] = beta_new_0
                beta_g[1] = beta_new_1
                
                if diff < tol:
                    gene_converged = True
                    n_iter[g] = iteration + 1
                    
                    # Compute SE
                    inv_00 = xtwx_11 / det
                    inv_11 = xtwx_00 / det
                    se[0, g] = np.sqrt(max(inv_00, 1e-12))
                    se[1, g] = np.sqrt(max(inv_11, 1e-12))
                    break
            else:
                # General case: would need matrix operations
                # For now, fallback to simpler convergence check
                gene_converged = True
                n_iter[g] = iteration + 1
                break
        
        if not gene_converged:
            n_iter[g] = max_iter
            # Compute final SE even if not converged
            if n_features == 2:
                xtwx_00 = ridge
                xtwx_01 = 0.0
                xtwx_11 = ridge
                for i in range(n_samples):
                    w_i = W_g[i]
                    x1_i = X[i, 1]
                    xtwx_00 += w_i
                    xtwx_01 += w_i * x1_i
                    xtwx_11 += w_i * x1_i * x1_i
                det = xtwx_00 * xtwx_11 - xtwx_01 * xtwx_01
                if abs(det) < 1e-12:
                    det = 1e-12
                se[0, g] = np.sqrt(max(xtwx_11 / det, 1e-12))
                se[1, g] = np.sqrt(max(xtwx_00 / det, 1e-12))
        
        beta[:, g] = beta_g
        converged[g] = gene_converged
    
    return beta, se, converged, n_iter


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
