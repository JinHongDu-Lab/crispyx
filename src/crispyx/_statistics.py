"""Statistical utilities for differential expression testing.

This module provides functions for tie correction, p-value adjustment,
and batched SE/dispersion computation.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np


def _low_expr_in_both_mask(
    *,
    pert_expr_counts: np.ndarray,
    control_expr_counts: np.ndarray,
    pert_mean: np.ndarray,
    control_mean: np.ndarray,
    n_pert_cells: int,
    n_control_cells: int,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_pct_both: float | None = None,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
) -> np.ndarray:
    """Return a boolean mask flagging genes that are jointly low-expressed.

    A gene is flagged (True = drop / exclude from DE testing) only when **both**
    the perturbation group and the control group satisfy their respective
    low-expression criteria.

    **Control side**: flagged when *both* the fraction of expressing cells is
    below ``min_pct_ctrl`` *and* the mean expression is below ``min_mean_ctrl``.

    **Perturbed side**: flagged when *both* the fraction of expressing cells is
    below ``min_pct_pert`` *and* the mean expression is below ``min_mean_pert``.
    The dual condition makes the filter more robust than a pct-only check: a
    doublet / ambient-RNA artefact with high counts in 1–2 cells has a high mean
    despite tiny pct, so it passes the mean check and is correctly retained.

    Parameters
    ----------
    pert_expr_counts, control_expr_counts
        Number of cells with non-zero expression per gene, shape (n_genes,).
    pert_mean, control_mean
        Mean expression per gene, shape (n_genes,).
    n_pert_cells, n_control_cells
        Number of cells in each group.
    min_pct_ctrl
        Minimum fraction of expressing cells for the *control* side. Default 0.01.
    min_pct_pert
        Minimum fraction of expressing cells for the *perturbed* side.
        Default 0.002 (lower than ctrl; induction from near-zero baseline is valid).
    min_pct_both
        Deprecated. If not ``None``, overrides both ``min_pct_ctrl`` and
        ``min_pct_pert`` and emits a ``DeprecationWarning``.
    min_mean_ctrl
        Minimum mean expression for the *control* side. Default 0.05.
    min_mean_pert
        Minimum mean expression for the *perturbed* side. Default 0.005.

    Returns
    -------
    ndarray of bool, shape (n_genes,)
        ``True`` for genes that should be dropped from this comparison.
    """
    if min_pct_both is not None:
        warnings.warn(
            "min_pct_both is deprecated; use min_pct_ctrl and min_pct_pert instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        min_pct_ctrl = float(min_pct_both)
        min_pct_pert = float(min_pct_both)

    if (
        (min_pct_ctrl <= 0.0 and min_pct_pert <= 0.0
         and min_mean_ctrl <= 0.0 and min_mean_pert <= 0.0)
        or n_pert_cells == 0
        or n_control_cells == 0
    ):
        return np.zeros(pert_expr_counts.shape[0], dtype=bool)

    pct_p = pert_expr_counts.astype(np.float64, copy=False) / float(n_pert_cells)
    pct_c = control_expr_counts.astype(np.float64, copy=False) / float(n_control_cells)

    # Control side: pct AND mean check
    low_c = (pct_c < min_pct_ctrl) & (control_mean < min_mean_ctrl)

    # Perturbed side: pct AND mean check (dual condition)
    low_p = (pct_p < min_pct_pert) & (pert_mean < min_mean_pert)

    return low_p & low_c


def _tie_correction(ranks: np.ndarray) -> np.ndarray:
    """Compute tie correction factors for each column of ``ranks``.

    Parameters
    ----------
    ranks : ndarray of shape (n_obs, n_genes)
        Rank matrix (1-based, with ties averaged).

    Returns
    -------
    ndarray of shape (n_genes,)
        Correction factor per gene in ``[0, 1]``.  A value of 1 means
        no ties; lower values indicate more ties.
    """

    n_genes = ranks.shape[1]
    correction = np.ones(n_genes, dtype=np.float64)
    for idx in range(n_genes):
        column = np.sort(np.ravel(ranks[:, idx]))
        size = float(column.size)
        if size < 2:
            continue
        boundaries = np.concatenate(
            (
                np.array([True]),
                column[1:] != column[:-1],
                np.array([True]),
            )
        )
        indices = np.flatnonzero(boundaries)
        counts = np.diff(indices).astype(np.float64)
        denom = size**3 - size
        if denom <= 0:
            continue
        correction[idx] = 1.0 - float(np.sum(counts**3 - counts)) / denom
    return correction


def _adjust_pvalue_matrix(
    matrix: np.ndarray,
    method: Literal["benjamini-hochberg", "bonferroni"],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return a p-value matrix adjusted per row using the requested method.

    Parameters
    ----------
    matrix
        P-value matrix to correct.
    method
        Correction method to apply.
    out
        Optional array to write results into. If provided, must be the same shape
        as ``matrix``.
    """

    adjusted = out if out is not None else np.ones_like(matrix)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        valid = np.isfinite(row)
        if not np.any(valid):
            adjusted[row_idx] = row
            continue
        values = row[valid]
        if method == "bonferroni":
            n_tests = values.size
            scaled = np.minimum(values * n_tests, 1.0)
            result = np.empty_like(values)
            result[:] = scaled
        elif method == "benjamini-hochberg":
            order = np.argsort(values)
            ordered = values[order]
            n_tests = ordered.size
            ranks = np.arange(1, n_tests + 1, dtype=np.float64)
            adj = ordered * n_tests / ranks
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            adj = np.clip(adj, 0.0, 1.0)
            result = np.empty_like(ordered)
            result[order] = adj
        else:  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported correction method: {method}")
        new_row = np.array(row, copy=True)
        new_row[valid] = result
        adjusted[row_idx] = new_row
    return adjusted


def _compute_se_batched(
    Y_control: np.ndarray,
    Y_pert: np.ndarray,
    control_offset: np.ndarray,
    pert_offset: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    dispersion: np.ndarray,
    *,
    gene_batch_size: int = 5000,
    ridge: float = 1e-6,
    se_method: Literal["sandwich", "fisher"] = "sandwich",
) -> np.ndarray:
    """Compute SE using batched processing to reduce peak memory.
    
    Instead of building a full (n_cells, n_genes) matrix for mu and W,
    this function processes genes in batches and accumulates the XᵀWX sums.
    
    Parameters
    ----------
    Y_control
        Control expression matrix, shape (n_control, n_genes).
    Y_pert
        Perturbation expression matrix, shape (n_pert, n_genes).
    control_offset
        Log size factors for control cells, shape (n_control,).
    pert_offset
        Log size factors for perturbation cells, shape (n_pert,).
    beta0
        Intercept coefficients, shape (n_genes,).
    beta1
        Perturbation effect coefficients, shape (n_genes,).
    dispersion
        Dispersion values, shape (n_genes,).
    gene_batch_size
        Number of genes to process per batch.
    ridge
        Ridge penalty for regularization.
    se_method
        Method for computing standard errors:
        - "sandwich": Sandwich estimator SE = sqrt(c' @ H @ M @ H @ c). More
          robust to model misspecification. This is the default.
        - "fisher": Standard Fisher information SE = sqrt(diag(inv(X'WX + ridge*I))).
          Matches PyDESeq2's approach for better parity.
        
    Returns
    -------
    np.ndarray
        Standard errors for perturbation effect, shape (n_genes,).
    """
    n_control = Y_control.shape[0]
    n_pert = Y_pert.shape[0]
    n_genes = Y_control.shape[1]
    
    # Output SE array
    se = np.full(n_genes, np.nan, dtype=np.float64)
    
    # Process genes in batches
    for g_start in range(0, n_genes, gene_batch_size):
        g_end = min(g_start + gene_batch_size, n_genes)
        batch_genes = g_end - g_start
        
        # Extract batch data
        beta0_batch = beta0[g_start:g_end]
        beta1_batch = beta1[g_start:g_end]
        disp_batch = dispersion[g_start:g_end]
        
        # Compute mu for control cells: mu = exp(beta0 + offset)
        # Shape: (n_control, batch_genes)
        eta_ctrl = beta0_batch[None, :] + control_offset[:, None]
        np.clip(eta_ctrl, -30, 30, out=eta_ctrl)
        mu_ctrl = np.exp(eta_ctrl)
        np.maximum(mu_ctrl, 1e-10, out=mu_ctrl)
        
        # Compute mu for perturbation cells: mu = exp(beta0 + beta1 + offset)
        # Shape: (n_pert, batch_genes)
        eta_pert = beta0_batch[None, :] + beta1_batch[None, :] + pert_offset[:, None]
        np.clip(eta_pert, -30, 30, out=eta_pert)
        mu_pert = np.exp(eta_pert)
        np.maximum(mu_pert, 1e-10, out=mu_pert)
        
        # Compute weights: W = mu / (1 + mu * disp)
        # Control weights
        W_ctrl = mu_ctrl / (1 + mu_ctrl * disp_batch[None, :])
        # Perturbation weights
        W_pert = mu_pert / (1 + mu_pert * disp_batch[None, :])
        
        # Fisher information (XᵀWX) - unregularized
        # For design [1, p] where p is perturbation indicator (0 for ctrl, 1 for pert):
        # M00 = sum_all(W), M01 = M10 = sum_pert(W), M11 = sum_pert(W)
        M00 = W_ctrl.sum(axis=0) + W_pert.sum(axis=0)  # (batch_genes,)
        M01 = W_pert.sum(axis=0)  # (batch_genes,) - only pert contributes
        M11 = W_pert.sum(axis=0)  # same as M01 since indicator is 1
        
        # Free work arrays
        del eta_ctrl, eta_pert, mu_ctrl, mu_pert, W_ctrl, W_pert
        
        # Regularized: Mr = M + ridge*I
        Mr00 = M00 + ridge
        Mr01 = M01
        Mr11 = M11 + ridge
        
        # Inverse: H = inv(Mr) for 2x2 matrix
        det_r = Mr00 * Mr11 - Mr01 ** 2
        np.maximum(det_r, 1e-12, out=det_r)
        
        H00 = Mr11 / det_r
        H01 = -Mr01 / det_r
        H11 = Mr00 / det_r
        
        # Compute SE based on method
        if se_method == "fisher":
            # Fisher information SE: SE = sqrt(diag(inv(X'WX + ridge*I)))
            # For perturbation effect, this is just H11
            var_pert = H11
        else:
            # Sandwich SE for perturbation effect (contrast [0, 1])
            # SE = sqrt(c' @ H @ M @ H @ c) where c = [0, 1]
            var_pert = H01**2 * M00 + 2 * H01 * H11 * M01 + H11**2 * M11
        
        se[g_start:g_end] = np.sqrt(np.maximum(var_pert, 1e-12))
    
    return se


def _compute_mom_dispersion_batched(
    Y_control: np.ndarray,
    Y_pert: np.ndarray,
    control_offset: np.ndarray,
    pert_offset: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    converged: np.ndarray,
    *,
    gene_batch_size: int = 5000,
) -> np.ndarray:
    """Compute MoM dispersion using batched processing to reduce peak memory.
    
    Instead of building full (n_cells, n_genes) matrices for mu and residuals,
    this function processes genes in batches.
    
    Parameters
    ----------
    Y_control
        Control expression matrix, shape (n_control, n_genes).
    Y_pert
        Perturbation expression matrix, shape (n_pert, n_genes).
    control_offset
        Log size factors for control cells, shape (n_control,).
    pert_offset
        Log size factors for perturbation cells, shape (n_pert,).
    beta0
        Intercept coefficients, shape (n_genes,).
    beta1
        Perturbation effect coefficients, shape (n_genes,).
    converged
        Convergence mask, shape (n_genes,).
    gene_batch_size
        Number of genes to process per batch.
        
    Returns
    -------
    np.ndarray
        MoM dispersion estimates, shape (n_genes,).
    """
    n_control = Y_control.shape[0]
    n_pert = Y_pert.shape[0]
    n_genes = Y_control.shape[1]
    n_total = n_control + n_pert
    dof = max(n_total - 2, 1)
    
    # Output dispersion array
    disp = np.full(n_genes, np.nan, dtype=np.float64)
    
    # Process genes in batches
    for g_start in range(0, n_genes, gene_batch_size):
        g_end = min(g_start + gene_batch_size, n_genes)
        
        # Extract batch data
        beta0_batch = beta0[g_start:g_end]
        beta1_batch = beta1[g_start:g_end]
        conv_batch = converged[g_start:g_end]
        Y_ctrl_batch = Y_control[:, g_start:g_end]
        Y_pert_batch = Y_pert[:, g_start:g_end]
        
        # Compute mu for control cells: mu = exp(beta0 + offset)
        eta_ctrl = beta0_batch[None, :] + control_offset[:, None]
        np.clip(eta_ctrl, -30, 30, out=eta_ctrl)
        mu_ctrl = np.exp(eta_ctrl)
        np.maximum(mu_ctrl, 1e-10, out=mu_ctrl)
        # Zero out non-converged genes
        mu_ctrl[:, ~conv_batch] = 1e-10
        
        # Compute mu for perturbation cells: mu = exp(beta0 + beta1 + offset)
        eta_pert = beta0_batch[None, :] + beta1_batch[None, :] + pert_offset[:, None]
        np.clip(eta_pert, -30, 30, out=eta_pert)
        mu_pert = np.exp(eta_pert)
        np.maximum(mu_pert, 1e-10, out=mu_pert)
        mu_pert[:, ~conv_batch] = 1e-10
        
        # Compute MoM dispersion
        # Formula: alpha = sum((Y - mu)^2 - Y) / mu^2) / dof
        resid_ctrl = Y_ctrl_batch - mu_ctrl
        resid_pert = Y_pert_batch - mu_pert
        
        numerator = (
            np.sum((resid_ctrl * resid_ctrl - Y_ctrl_batch) / np.maximum(mu_ctrl * mu_ctrl, 1e-10), axis=0)
            + np.sum((resid_pert * resid_pert - Y_pert_batch) / np.maximum(mu_pert * mu_pert, 1e-10), axis=0)
        )
        
        disp[g_start:g_end] = np.clip(numerator / dof, 1e-8, 1e6)
    
    return disp


def _nonestimable_glm_mask(
    *,
    pert_expr_counts: np.ndarray,
    control_expr_counts: np.ndarray,
    min_cells_ctrl: int = 1,
    min_cells_pert: int = 1,
) -> np.ndarray:
    """Flag (gene, perturbation) pairs whose GLM effect is not estimable.

    A log-link GLM estimates the perturbation effect as
    ``log(mean_perturbed) - log(mean_control)``.  When one arm carries no
    counts at all that quantity is infinite: the likelihood has no interior
    maximum and the coefficient runs to the boundary, so whatever the fitter
    reports is set by where it was stopped, not by the data.  On one such gene
    the fitted effect moved by a factor of four with the ``min_mu`` floor
    alone, and the Wald statistic went from "no evidence" to "wildly
    significant" on identical counts.  (The first of those is the
    Hauck-Donner effect: as the coefficient diverges its standard error grows
    faster still, so the Wald statistic tends to zero on what is the strongest
    possible signal.)

    Such pairs are therefore reported as untested -- ``NaN`` effect, statistic
    and p-value -- exactly as genes with no counts anywhere already are.  This
    is not the same as reporting an effect of zero, which would say "no
    change" about a gene the perturbation may have silenced completely.  The
    observation itself is preserved in ``pts`` and ``pts_rest``, which are
    populated for every gene, so a pair expressed in 0% of perturbed and 95%
    of control cells stays visible there.

    This complements :func:`_low_expr_in_both_mask`, which drops pairs that are
    jointly low in *both* arms.  A pair that is absent from one arm and
    abundant in the other passes that filter -- correctly, since it carries
    real signal -- and is caught here instead.

    The two thresholds are separate because the informative direction depends
    on the screen.  In a knockdown screen (CRISPRi) the expected hit is
    abundant in control and depleted in the perturbed arm, so the useful
    setting is a demanding ``min_cells_ctrl`` -- evidence the gene is really
    expressed at baseline -- beside a permissive ``min_cells_pert``.  An
    activation screen (CRISPRa) wants the reverse.  Both default to 1, which
    is symmetric and is also the exact boundary between an effect that exists
    and one that does not; a threshold below 1 does not describe a weaker
    filter, it readmits the artefact.

    Parameters
    ----------
    pert_expr_counts, control_expr_counts
        Number of cells with non-zero expression per gene, shape (n_genes,).
    min_cells_ctrl
        Minimum expressing cells required in the *control* arm. Default 1.
    min_cells_pert
        Minimum expressing cells required in the *perturbed* arm. Default 1.

    Returns
    -------
    ndarray of bool, shape (n_genes,)
        ``True`` for pairs that should not be given an effect estimate.
    """
    control_floor = max(int(min_cells_ctrl), 0)
    pert_floor = max(int(min_cells_pert), 0)
    flagged = np.zeros(np.shape(pert_expr_counts), dtype=bool)
    if control_floor:
        flagged |= np.asarray(control_expr_counts) < control_floor
    if pert_floor:
        flagged |= np.asarray(pert_expr_counts) < pert_floor
    return flagged
