"""Size factor computation utilities for normalization.

This module provides functions for computing library size factors
using median-of-ratios and DESeq2-style methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp

from .data import iter_matrix_chunks, read_backed

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def _validate_size_factors(
    size_factors: "ArrayLike", n_cells: int, *, scale: bool = True
) -> np.ndarray:
    """Validate and optionally scale user-provided size factors.
    
    Parameters
    ----------
    size_factors
        User-provided size factors.
    n_cells
        Expected number of cells.
    scale
        If True, rescale so geometric mean equals 1.
        
    Returns
    -------
    np.ndarray
        Validated size factors.
    """
    size_factors_arr = np.asarray(size_factors, dtype=np.float64).reshape(-1)
    if size_factors_arr.shape[0] != n_cells:
        raise ValueError(
            f"Provided size_factors have length {size_factors_arr.shape[0]} but expected {n_cells}"
        )
    mask = np.isfinite(size_factors_arr) & (size_factors_arr > 0)
    if not np.any(mask):
        raise ValueError("Provided size_factors contain no positive finite values")
    size_factors_arr[~mask] = np.nanmedian(size_factors_arr[mask])
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors_arr, 1e-12, None))))
        return size_factors_arr / scale_factor
    return size_factors_arr


def _median_of_ratios_size_factors(
    path: str | Path, *, chunk_size: int = 2048, scale: bool = True
) -> np.ndarray:
    """Compute median-of-ratios size factors with vectorized row-median computation.
    
    Optimized implementation using Numba-accelerated parallel row-median kernel
    for significant speedup on large datasets (5× faster than per-row Python loop
    with the default chunk_size=2048).
    
    Parameters
    ----------
    path
        Path to h5ad file with count data.
    chunk_size
        Number of cells to process per chunk. Larger values reduce I/O overhead
        but use more memory. Default 2048 balances speed and memory usage.
    scale
        If True, rescale size factors so their geometric mean equals 1.
        
    Returns
    -------
    np.ndarray
        Size factors for each cell.
    """
    from ._kernels import _compute_row_medians_csr
    
    backed = read_backed(path)
    n_cells = backed.n_obs
    n_genes = backed.n_vars
    log_sum = np.zeros(n_genes, dtype=np.float64)
    log_count = np.zeros(n_genes, dtype=np.int64)
    try:
        for _, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = sp.csr_matrix(block)
            csc = csr.tocsc()
            counts_per_gene = np.diff(csc.indptr)
            if not counts_per_gene.any():
                continue
            gene_indices = np.repeat(np.arange(n_genes, dtype=np.int64), counts_per_gene)
            data = np.log(np.clip(csc.data, 1e-12, None))
            np.add.at(log_sum, gene_indices, data)
            np.add.at(log_count, gene_indices, 1)
    finally:
        backed.file.close()
    geo_means = np.zeros(n_genes, dtype=np.float64)
    valid_geo = log_count > 0
    geo_means[valid_geo] = np.exp(log_sum[valid_geo] / log_count[valid_geo])

    size_factors = np.full(n_cells, np.nan, dtype=np.float64)
    backed = read_backed(path)
    try:
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = sp.csr_matrix(block, dtype=np.float64)
            # Vectorized row-median computation using Numba kernel
            chunk_medians = _compute_row_medians_csr(
                csr.data.astype(np.float64),
                csr.indices.astype(np.int64),
                csr.indptr.astype(np.int64),
                geo_means,
                csr.shape[0],
            )
            size_factors[slc] = chunk_medians
    finally:
        backed.file.close()

    valid_sf = np.isfinite(size_factors) & (size_factors > 0)
    if not np.any(valid_sf):
        return np.ones(n_cells, dtype=np.float64)
    fallback = np.nanmedian(size_factors[valid_sf])
    size_factors[~valid_sf] = fallback
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors, 1e-12, None))))
        return size_factors / scale_factor
    return size_factors


def _deseq2_style_size_factors(
    path: str | Path,
    *,
    chunk_size: int = 256,
    scale: bool = True,
    early_termination_threshold: int = 10,
) -> np.ndarray:
    """Compute DESeq2-style size factors using only genes expressed in all cells.
    
    This method computes size factors in the same way as DESeq2/PyDESeq2:
    1. Find genes expressed (count > 0) in ALL cells
    2. Compute geometric mean of counts per gene across all cells
    3. For each cell, compute ratios of counts to geometric means
    4. Take median of ratios as the size factor
    
    This approach works well for bulk RNA-seq where most genes are expressed in all
    samples, but may have issues with very sparse single-cell data where few genes
    are expressed in all cells.
    
    **Optimization (2026-01-13)**: The all_expressed check is now vectorized using
    sparse column operations (O(n_genes) instead of O(n_cells × n_genes)), with
    early termination when too few genes remain after initial chunks.
    
    Parameters
    ----------
    path
        Path to h5ad file with count data.
    chunk_size
        Number of cells to process per chunk.
    scale
        If True, rescale size factors so their geometric mean equals 1.
    early_termination_threshold
        If fewer than this many genes remain after processing a chunk,
        fall back to sparse method immediately. Default 10.
        
    Returns
    -------
    np.ndarray
        Size factors for each cell.
    """
    from scipy.stats import gmean
    
    backed = read_backed(path)
    n_cells = backed.n_obs
    n_genes = backed.n_vars
    
    # First pass: find genes expressed in all cells
    # VECTORIZED: Use sparse column counts instead of O(n_cells × n_genes) loop
    # A gene is expressed in ALL cells of a chunk iff its nnz == chunk_n_rows
    all_expressed = np.ones(n_genes, dtype=bool)
    cells_processed = 0
    try:
        for _, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = sp.csr_matrix(block)
            chunk_n_rows = csr.shape[0]
            cells_processed += chunk_n_rows
            
            # VECTORIZED: Count non-zeros per gene (column) in O(nnz + n_genes) time
            # Convert to CSC for efficient column access
            csc = csr.tocsc()
            nnz_per_gene = np.diff(csc.indptr)  # O(n_genes)
            
            # Gene is NOT expressed in all cells of this chunk if nnz < chunk_n_rows
            genes_missing_in_chunk = nnz_per_gene < chunk_n_rows
            all_expressed[genes_missing_in_chunk] = False
            
            # EARLY TERMINATION: If too few genes remain, fall back immediately
            n_remaining = all_expressed.sum()
            if n_remaining < early_termination_threshold:
                logger.warning(
                    f"Only {n_remaining} genes expressed in all cells after "
                    f"{cells_processed:,}/{n_cells:,} cells. Falling back to sparse method."
                )
                backed.file.close()
                return _median_of_ratios_size_factors(path, chunk_size=chunk_size, scale=scale)
    finally:
        backed.file.close()
    
    n_all_expressed = np.sum(all_expressed)
    if n_all_expressed < 10:
        logger.warning(
            f"Only {n_all_expressed} genes expressed in all cells. "
            "Consider using size_factor_method='sparse' for sparse data."
        )
    if n_all_expressed == 0:
        logger.warning(
            "No genes expressed in all cells. Falling back to sparse method."
        )
        return _median_of_ratios_size_factors(path, chunk_size=chunk_size, scale=scale)
    
    # Collect counts for all-expressed genes
    all_expressed_idx = np.where(all_expressed)[0]

    # Memory check: estimate size of counts_filtered array
    counts_filtered_gb = n_cells * n_all_expressed * 8 / 1e9
    use_streaming = counts_filtered_gb > 4.0  # >4 GB → stream to avoid OOM

    if use_streaming:
        logger.info(
            f"Large counts_filtered ({n_cells:,} × {n_all_expressed:,} = "
            f"{counts_filtered_gb:.1f} GB). Using streaming DESeq2 size factors."
        )

        # Streaming Pass 2: compute geometric means via log-sum accumulation.
        # All selected genes have count > 0 in every cell, so log is finite.
        log_sum = np.zeros(n_all_expressed, dtype=np.float64)
        backed = read_backed(path)
        try:
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=True
            ):
                block_arr = np.asarray(block, dtype=np.float64)
                block_filtered = block_arr[:, all_expressed_idx]
                log_sum += np.log(np.maximum(block_filtered, 1e-300)).sum(axis=0)
        finally:
            backed.file.close()
        geo_means = np.exp(log_sum / n_cells)

        # Streaming Pass 3: compute per-cell median of ratios
        size_factors = np.full(n_cells, np.nan, dtype=np.float64)
        backed = read_backed(path)
        try:
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=True
            ):
                block_arr = np.asarray(block, dtype=np.float64)
                block_filtered = block_arr[:, all_expressed_idx]
                ratios = block_filtered / geo_means
                size_factors[slc] = np.median(ratios, axis=1)
        finally:
            backed.file.close()
    else:
        counts_filtered = np.zeros((n_cells, n_all_expressed), dtype=np.float64)
        backed = read_backed(path)
        try:
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=True
            ):
                block_arr = np.asarray(block)
                counts_filtered[slc.start:slc.stop, :] = block_arr[:, all_expressed_idx]
        finally:
            backed.file.close()

        # Compute geometric means per gene
        geo_means = gmean(counts_filtered, axis=0)

        # Compute ratios and take median
        ratios = counts_filtered / geo_means
        size_factors = np.median(ratios, axis=1)
    
    # Handle any invalid values
    valid_sf = np.isfinite(size_factors) & (size_factors > 0)
    if not np.all(valid_sf):
        fallback = np.nanmedian(size_factors[valid_sf])
        size_factors[~valid_sf] = fallback
    
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors, 1e-12, None))))
        return size_factors / scale_factor
    return size_factors


def _compute_subset_size_factors(
    X: np.ndarray | sp.spmatrix,
    cell_mask: np.ndarray,
    *,
    scale: bool = True,
) -> np.ndarray:
    """Compute DESeq2-style size factors for a subset of cells.
    
    Uses only genes expressed in ALL cells of the subset (like DESeq2/PyDESeq2).
    
    Parameters
    ----------
    X
        Full count matrix (n_cells, n_genes).
    cell_mask
        Boolean mask indicating which cells to include in the subset.
    scale
        If True, scale so geometric mean of size factors equals 1.
        
    Returns
    -------
    np.ndarray
        Size factors for cells in the subset (length = sum(cell_mask)).
    """
    from scipy.stats import gmean
    
    # Get subset matrix
    if sp.issparse(X):
        X_sub = X[cell_mask, :].toarray()
    else:
        X_sub = np.asarray(X[cell_mask, :])
    
    n_cells, n_genes = X_sub.shape
    
    # Find genes expressed in ALL cells of subset (DESeq2 style)
    min_per_gene = X_sub.min(axis=0)
    all_expressed = min_per_gene > 0
    n_reference = all_expressed.sum()
    
    if n_reference < 5:
        # Fall back to median-of-ratios with non-zero filtering
        # Compute geometric means using only non-zero values
        log_X = np.log(np.maximum(X_sub, 1e-12))
        log_X[X_sub == 0] = np.nan
        geo_means = np.exp(np.nanmean(log_X, axis=0))
        
        # Compute ratios
        size_factors = np.zeros(n_cells, dtype=np.float64)
        for i in range(n_cells):
            ratios = X_sub[i, :] / geo_means
            valid = (X_sub[i, :] > 0) & np.isfinite(ratios) & (ratios > 0)
            if np.sum(valid) > 0:
                size_factors[i] = np.median(ratios[valid])
            else:
                size_factors[i] = np.nan
    else:
        # Use only genes expressed in all cells (DESeq2 style)
        X_ref = X_sub[:, all_expressed]
        geo_means = gmean(X_ref, axis=0)
        ratios = X_ref / geo_means
        size_factors = np.median(ratios, axis=1)
    
    # Handle invalid values
    valid_sf = np.isfinite(size_factors) & (size_factors > 0)
    if not np.all(valid_sf):
        if np.any(valid_sf):
            fallback = np.nanmedian(size_factors[valid_sf])
            size_factors[~valid_sf] = fallback
        else:
            size_factors[:] = 1.0
    
    if scale:
        scale_factor = np.exp(np.mean(np.log(np.clip(size_factors, 1e-12, None))))
        return size_factors / scale_factor
    return size_factors
