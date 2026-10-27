"""Size factors for count models: library size, causarray's geometric means
and DESeq2's median of ratios, each against the screen's controls where a
reference is needed (see docs/nb_glm.rst, "Size factors")."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import iter_matrix_chunks, read_backed

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

def _validate_size_factors(size_factors: "ArrayLike", n_cells: int) -> np.ndarray:
    """User-provided size factors: one per cell, non-positive or non-finite
    ones replaced by the median of the others, at geometric mean 1."""
    size_factors_arr = np.asarray(size_factors, dtype=np.float64).reshape(-1)
    if size_factors_arr.shape[0] != n_cells:
        raise ValueError(
            f"Provided size_factors have length {size_factors_arr.shape[0]} but expected {n_cells}"
        )
    if not np.any(np.isfinite(size_factors_arr) & (size_factors_arr > 0)):
        raise ValueError("Provided size_factors contain no positive finite values")
    return _unit_geometric_mean(size_factors_arr)


def _unit_geometric_mean(size_factors: np.ndarray) -> np.ndarray:
    """Replace non-positive or non-finite factors by the median of the others
    and rescale to geometric mean 1 (which no test depends on: a constant
    factor is absorbed by the intercept)."""
    valid = np.isfinite(size_factors) & (size_factors > 0)
    if not valid.any():
        return np.ones(size_factors.shape[0], dtype=np.float64)
    size_factors = size_factors.copy()
    size_factors[~valid] = np.median(size_factors[valid])
    return size_factors / np.exp(np.mean(np.log(size_factors)))


def _library_size_factors(path: str | Path, *, chunk_size: int = 2048) -> np.ndarray:
    """Each cell's total count, at geometric mean 1, in one pass over ``path``."""
    backed = read_backed(path)
    totals = np.zeros(backed.n_obs, dtype=np.float64)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            totals[slc] = np.asarray(block.sum(axis=1)).ravel()
    finally:
        backed.file.close()
    return _unit_geometric_mean(totals)


def _control_statistics(
    path: str | Path, is_control: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per gene over the control cells: the sum of counts, the number of
    cells with a count, and the sum of the log non-zero counts. Reads only
    the control rows (contiguous in a file sorted by perturbation)."""
    control_rows = np.flatnonzero(is_control)
    backed = read_backed(path)
    n_genes = backed.n_vars
    total = np.zeros(n_genes, dtype=np.float64)
    nnz = np.zeros(n_genes, dtype=np.int64)
    log_sum = np.zeros(n_genes, dtype=np.float64)
    try:
        for start in range(0, control_rows.size, chunk_size):
            csc = sp.csc_matrix(backed.X[control_rows[start:start + chunk_size], :], dtype=np.float64)
            csc.sum_duplicates()
            csc.eliminate_zeros()
            counted = np.diff(csc.indptr)
            gene_of = np.repeat(np.arange(n_genes), counted)
            nnz += counted
            total += np.bincount(gene_of, weights=csc.data, minlength=n_genes)
            log_sum += np.bincount(gene_of, weights=np.log(csc.data), minlength=n_genes)
    finally:
        backed.file.close()
    return total, nnz, log_sum


def _row_median_ratios(
    path: str | Path, geo_means: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's median ratio to ``geo_means`` over the genes it counts with
    a non-zero reference (NaN without one), and its total count; one pass."""
    from ._kernels import _compute_row_medians_csr

    backed = read_backed(path)
    medians = np.full(backed.n_obs, np.nan, dtype=np.float64)
    totals = np.zeros(backed.n_obs, dtype=np.float64)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            csr = sp.csr_matrix(block, dtype=np.float64)
            medians[slc] = _compute_row_medians_csr(
                csr.data, csr.indices.astype(np.int64), csr.indptr.astype(np.int64), geo_means, csr.shape[0]
            )
            totals[slc] = np.asarray(csr.sum(axis=1)).ravel()
    finally:
        backed.file.close()
    return medians, totals


def _control_geomeans_size_factors(
    path: str | Path,
    labels: np.ndarray,
    control_label: str,
    *,
    min_mean: float = 2.0,
    min_genes: int = 100,
    chunk_size: int = 2048,
    control_statistics: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """causarray's ``comp_size_factor(method="geomeans")`` with the controls as reference.

    Reference genes are those with a mean count of at least ``min_mean`` over
    the controls (the ``min_genes`` highest when fewer pass); a gene's
    reference is the geometric mean of the controls' non-zero counts. A
    cell's factor is the median of its ratios to the reference over the
    reference genes it counts. A cell counting none of them gets its total
    count, put on the others' scale by their median ratio of factor to total.

    Restricting the ratios to well-counted genes is what lets a median of
    ratios track sequencing depth in sparse data: over every expressed gene
    the ratios are mostly of counts 1 and 2 and follow only part of a depth
    difference. Reads the control rows, then every row once;
    ``control_statistics`` (from :func:`_control_statistics`) skips the first.
    """
    is_control = np.asarray(labels) == control_label
    n_control = int(is_control.sum())
    if control_statistics is None:
        control_statistics = _control_statistics(path, is_control, chunk_size)
    control_sum, control_nnz, control_log_sum = control_statistics
    n_genes = control_sum.size

    mean = control_sum / max(n_control, 1)
    keep = mean >= min_mean
    if keep.sum() < min(min_genes, n_genes):
        keep = np.zeros(n_genes, dtype=bool)
        keep[np.argsort(mean)[::-1][:min_genes]] = True
    keep &= control_nnz > 0
    geo_means = np.zeros(n_genes, dtype=np.float64)  # 0 leaves a gene out of the ratios
    geo_means[keep] = np.exp(control_log_sum[keep] / control_nnz[keep])

    size_factors, totals = _row_median_ratios(path, geo_means, chunk_size)
    found = np.isfinite(size_factors) & (size_factors > 0)
    missing = ~found & (totals > 0)
    if missing.any() and found.any():
        size_factors[missing] = totals[missing] * np.median(size_factors[found] / totals[found])
    logger.info("Control geometric-mean size factors: %d reference genes", int(keep.sum()))
    return _unit_geometric_mean(size_factors)


def _control_reference_size_factors(
    path: str | Path,
    labels: np.ndarray,
    control_label: str,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    """DESeq2 median-of-ratios size factors against a reference made of the controls.

    DESeq2 computes size factors once for all samples of a model and takes
    each sample's median ratio to a reference of per-gene geometric means;
    ``estimateSizeFactors(geoMeans=...)`` lets the caller supply that
    reference. Here the reference comes from the control cells, so a control
    cell's factor does not depend on which perturbations the file holds or are
    compared, and every comparison sees the same control factors:

    1. reference genes are those with a count in every control cell; their
       geometric means over the controls are the reference;
    2. a control cell's factor is the median of its ratios to the reference
       (DESeq2's ratio method on the controls alone);
    3. a perturbed cell's factor uses the reference genes with a count in
       every cell of its perturbation -- DESeq2 likewise drops a gene that is
       zero in any sample compared -- or, when fewer than three remain, the
       reference genes this cell expresses.

    Meant for counts where many genes are counted in every control
    (pseudo-bulk, bulk). With fewer than five such genes -- most single-cell
    data -- the factors are :func:`_control_geomeans_size_factors` instead;
    a median of ratios over every expressed gene would track only part of
    each cell's depth.

    Reads the control rows, then every row twice: for each cell's median over
    the reference and each perturbation's number of cells counting each
    reference gene (a perturbations x reference genes table, the only state
    beyond a block), then for the medians over each perturbation's genes.

    Parameters
    ----------
    path
        h5ad file of raw counts.
    labels
        Each cell's perturbation label, in row order.
    control_label
        The label of the control cells.
    chunk_size
        Rows per chunk.
    """
    from ._kernels import _compute_row_medians_csr

    labels = np.asarray(labels)
    is_control = labels == control_label
    n_control = int(is_control.sum())
    statistics = _control_statistics(path, is_control, chunk_size)
    _, control_nnz, control_log_sum = statistics
    in_every_control = control_nnz == n_control
    if int(in_every_control.sum()) < 5:
        logger.info(
            "Only %d genes counted in every control: size_factor_method='deseq2' falls back to 'geomeans'",
            int(in_every_control.sum()),
        )
        return _control_geomeans_size_factors(
            path, labels, control_label, chunk_size=chunk_size, control_statistics=statistics
        )
    geo_means = np.zeros(control_nnz.size, dtype=np.float64)  # 0 marks a gene outside the reference
    geo_means[in_every_control] = np.exp(control_log_sum[in_every_control] / n_control)
    reference = np.flatnonzero(in_every_control)
    codes, uniques = pd.factorize(labels)
    n_rows_of = np.bincount(codes, minlength=len(uniques))

    # Pass 1: each cell's median over the reference genes it counts (the
    # controls' factors, and the fallback), and how many cells of each label
    # count each reference gene.
    size_factors = np.full(labels.size, np.nan, dtype=np.float64)
    counted = np.zeros((len(uniques), reference.size), dtype=np.int32)  # labels x reference genes
    backed = read_backed(path)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            csr = sp.csr_matrix(block, dtype=np.float64)
            csr.sum_duplicates()
            csr.eliminate_zeros()
            size_factors[slc] = _compute_row_medians_csr(
                csr.data, csr.indices.astype(np.int64), csr.indptr.astype(np.int64), geo_means, csr.shape[0]
            )
            present = csr[:, reference]
            present.data[:] = 1.0
            n_rows = present.shape[0]
            onehot = sp.csr_matrix(
                (np.ones(n_rows), (codes[slc], np.arange(n_rows))), shape=(len(uniques), n_rows)
            )
            add = (onehot @ present).tocoo()
            counted[add.row, add.col] += add.data.astype(np.int32)
    finally:
        backed.file.close()

    # Pass 2: the cells of a perturbation with at least three reference genes
    # counted in all its cells, over those genes. The controls already have
    # theirs: every reference gene is counted in every control.
    own_genes = counted == n_rows_of[:, None]
    has_own = own_genes.sum(axis=1) >= 3
    has_own[codes[np.argmax(is_control)]] = False
    if has_own.any():
        inverse_reference = 1.0 / geo_means[reference]
        rows_per_block = max(1, (1 << 22) // reference.size)  # bounds the dense ratios
        backed = read_backed(path)
        try:
            for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
                block_codes = codes[slc]
                rows = np.flatnonzero(has_own[block_codes])
                if rows.size == 0:
                    continue
                counts = sp.csr_matrix(block, dtype=np.float64)[rows][:, reference]
                for start in range(0, rows.size, rows_per_block):
                    part = slice(start, start + rows_per_block)
                    ratios = counts[part].toarray() * inverse_reference
                    ratios[~own_genes[block_codes[rows[part]]]] = np.nan
                    size_factors[slc.start + rows[part]] = np.nanmedian(ratios, axis=1)
        finally:
            backed.file.close()
    logger.info("Control-reference size factors: %d genes counted in every control", reference.size)
    return _unit_geometric_mean(size_factors)
