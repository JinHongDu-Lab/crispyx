"""Size factors for count models: library size, causarray's geometric means
and DESeq2's median of ratios, each against the screen's controls where a
reference is needed (see docs/differential_expression.rst, "Size factors")."""

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


def _control_geomeans_size_factors(
    path: str | Path,
    labels: np.ndarray,
    control_label: str,
    *,
    min_mean: float = 2.0,
    min_genes: int = 100,
    chunk_size: int = 2048,
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
    difference. Two passes over ``path``.
    """
    from ._kernels import _compute_row_medians_csr

    labels = np.asarray(labels)
    is_control = labels == control_label
    n_control = int(is_control.sum())
    backed = read_backed(path)
    n_cells, n_genes = backed.n_obs, backed.n_vars
    control_sum = np.zeros(n_genes, dtype=np.float64)
    control_nnz = np.zeros(n_genes, dtype=np.int64)
    control_log_sum = np.zeros(n_genes, dtype=np.float64)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            rows = is_control[slc]
            if not rows.any():
                continue
            csc = sp.csr_matrix(block, dtype=np.float64)[rows].tocsc()
            nnz = np.diff(csc.indptr)
            gene_of = np.repeat(np.arange(n_genes), nnz)
            control_nnz += nnz
            control_sum += np.bincount(gene_of, weights=csc.data, minlength=n_genes)
            control_log_sum += np.bincount(gene_of, weights=np.log(csc.data), minlength=n_genes)
    finally:
        backed.file.close()

    mean = control_sum / max(n_control, 1)
    keep = mean >= min_mean
    if keep.sum() < min(min_genes, n_genes):
        keep = np.zeros(n_genes, dtype=bool)
        keep[np.argsort(mean)[::-1][:min_genes]] = True
    keep &= control_nnz > 0
    geo_means = np.zeros(n_genes, dtype=np.float64)  # 0 leaves a gene out of the ratios
    geo_means[keep] = np.exp(control_log_sum[keep] / control_nnz[keep])

    size_factors = np.full(n_cells, np.nan, dtype=np.float64)
    totals = np.zeros(n_cells, dtype=np.float64)
    backed = read_backed(path)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            csr = sp.csr_matrix(block, dtype=np.float64)
            size_factors[slc] = _compute_row_medians_csr(
                csr.data, csr.indices.astype(np.int64), csr.indptr.astype(np.int64), geo_means, csr.shape[0]
            )
            totals[slc] = np.asarray(csr.sum(axis=1)).ravel()
    finally:
        backed.file.close()

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

    Two passes over ``path``: the controls' rows, then every row.

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
    backed = read_backed(path)
    n_cells, n_genes = backed.n_obs, backed.n_vars
    control_nnz = np.zeros(n_genes, dtype=np.int64)
    control_log_sum = np.zeros(n_genes, dtype=np.float64)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            rows = is_control[slc]
            if not rows.any():
                continue
            csc = sp.csr_matrix(block, dtype=np.float64)[rows].tocsc()
            nnz = np.diff(csc.indptr)
            control_nnz += nnz
            control_log_sum += np.bincount(
                np.repeat(np.arange(n_genes), nnz), weights=np.log(csc.data), minlength=n_genes
            )
    finally:
        backed.file.close()

    in_every_control = control_nnz == n_control
    if int(in_every_control.sum()) < 5:
        logger.info(
            "Only %d genes counted in every control: size_factor_method='deseq2' falls back to 'geomeans'",
            int(in_every_control.sum()),
        )
        return _control_geomeans_size_factors(path, labels, control_label, chunk_size=chunk_size)
    geo_means = np.zeros(n_genes, dtype=np.float64)  # 0 marks a gene outside the reference
    geo_means[in_every_control] = np.exp(control_log_sum[in_every_control] / n_control)
    reference = np.flatnonzero(geo_means > 0)

    size_factors = np.full(n_cells, np.nan, dtype=np.float64)
    # The reference genes' counts, kept to give each perturbation its own gene set.
    reference_counts = np.zeros((n_cells, reference.size), dtype=np.float64)
    backed = read_backed(path)
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            csr = sp.csr_matrix(block, dtype=np.float64)
            size_factors[slc] = _compute_row_medians_csr(
                csr.data, csr.indices.astype(np.int64), csr.indptr.astype(np.int64), geo_means, csr.shape[0]
            )
            reference_counts[slc] = csr[:, reference].toarray()
    finally:
        backed.file.close()

    for label in np.unique(labels[~is_control]):
        rows = np.flatnonzero(labels == label)
        counts = reference_counts[rows]
        keep = (counts > 0).all(axis=0)
        if keep.sum() >= 3:
            size_factors[rows] = np.median(counts[:, keep] / geo_means[reference[keep]], axis=1)
    logger.info("Control-reference size factors: %d genes counted in every control", reference.size)
    return _unit_geometric_mean(size_factors)
