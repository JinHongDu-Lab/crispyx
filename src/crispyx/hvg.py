"""Highly variable gene (HVG) selection for streaming, disk-backed AnnData.

Producer half of the ``var["highly_variable"]`` contract that ``dimred.py``'s
PCA entry points already consume. Reimplemented as a streaming, format-aware
pass rather than a thin ``scanpy.pp.highly_variable_genes`` wrapper because
every scanpy flavor computes gene moments via ``_get_mean_var(X)`` against a
fully realized ``X`` -- there is no bounded-memory, chunked-from-disk code
path, the same reason ``normalize_total_log1p`` and the QC/DE streaming paths
in this package are crispyx's own reimplementations.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _messages
from ._checkpoint import _create_progress_context
from .data import (
    AnnData,
    _looks_like_counts,
    _detect_backed_sparse_format,
    calculate_optimal_chunk_size,
    calculate_optimal_gene_chunk_size,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    load_obs,
    load_var,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_path_to_update,
    write_var,
)


def _stream_reduce(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    codes: np.ndarray,
    n_batches: int,
    chunk_size: int,
    show_progress: bool,
    desc: str,
    block_fn: Callable[[np.ndarray | sp.spmatrix, np.ndarray, slice], tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``matrix`` off disk and sum ``block_fn``'s two outputs over each batch's cells.

    ``codes`` holds each cell's batch (``-1``: a cell not used).
    ``block_fn(block, codes, genes)`` gets a chunk's used rows, their batch
    codes and the chunk's genes, and returns two arrays shaped like the
    block (sparse when it is); their rows are summed per batch here, giving
    two ``(n_batches, n_vars)`` arrays. Column chunks for CSC (each holds
    every cell of its genes), row chunks otherwise; ``convert_to_dense=False``
    throughout, so a chunk's memory is that of its non-zeros.
    """
    matrix = backed.X if layer is None else backed.layers[layer]
    by_gene = storage_format == "csc"
    out_a = np.zeros((n_batches, backed.n_vars))
    out_b = np.zeros((n_batches, backed.n_vars))
    n_chunks = -(-(backed.n_vars if by_gene else backed.n_obs) // chunk_size)
    chunks = iter_matrix_chunks(
        backed, axis=int(by_gene), matrix=matrix, chunk_size=chunk_size, convert_to_dense=False,
    )
    with _create_progress_context(n_chunks, f"{desc} (CSC)" if by_gene else desc, show_progress, unit="chunk") as bar:
        for slc, block in chunks:
            bar.update(1)
            cells, genes = (slice(None), slc) if by_gene else (slc, slice(None))
            block_codes = codes[cells]
            used = block_codes >= 0
            if not used.any():
                continue
            if not used.all():
                block, block_codes = block[used], block_codes[used]
            indicator = sp.csr_matrix(
                (np.ones(block_codes.size), (block_codes, np.arange(block_codes.size))),
                shape=(n_batches, block_codes.size),
            )
            x, x2 = block_fn(block, block_codes, genes)
            out_a[:, genes] += _dense(indicator @ x)
            out_b[:, genes] += _dense(indicator @ x2)
    return out_a, out_b


def _dense(values) -> np.ndarray:
    return values.toarray() if sp.issparse(values) else np.asarray(values)


def _stream_gene_moments(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    codes: np.ndarray,
    n_batches: int,
    transform: Literal["identity", "expm1"],
    chunk_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sum, sum of squares)`` per batch and gene and the cells per batch.

    ``expm1`` is applied to a per-block copy, and only to ``block.data`` when
    sparse (``expm1(0) == 0``), so no dense transformed copy is made.
    """

    def _values_and_squares(block, _codes, _genes):
        if transform == "expm1":
            if sp.issparse(block):
                block = block.copy()
                block.data = np.expm1(block.data)
            else:
                block = np.expm1(block)
        if not np.issubdtype(block.dtype, np.floating):
            block = block.astype(np.float64)  # an integer count's square can overflow its dtype
        return block, block.multiply(block) if sp.issparse(block) else np.square(block)

    sums, sqsums = _stream_reduce(
        backed,
        storage_format=storage_format,
        layer=layer,
        codes=codes,
        n_batches=n_batches,
        chunk_size=chunk_size,
        show_progress=show_progress,
        desc="Computing gene moments",
        block_fn=_values_and_squares,
    )
    return sums, sqsums, np.bincount(codes[codes >= 0], minlength=n_batches)


def _mean_var(sums: np.ndarray, sqsums: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and unbiased (``ddof=1``) variance from sums over ``n`` cells (last axis: genes)."""
    n = np.asarray(n, dtype=np.float64)[..., None]
    mean = sums / n
    return mean, (sqsums - n * np.square(mean)) / np.maximum(n - 1, 1)


def _check_looks_like_counts(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
) -> None:
    """Warn if the first non-empty chunk doesn't look like raw counts.

    Mirrors scanpy's own ``flavor="seurat_v3"`` sanity check
    (``check_nonnegative_integers``): only the first chunk is inspected,
    matching the same "peek, don't scan" cost the rest of this module pays
    for correctness checks (e.g. ``de.py``'s count-likeness guard).
    """
    matrix = backed.X if layer is None else backed.layers[layer]
    axis = 1 if storage_format == "csc" else 0
    for _, block in iter_matrix_chunks(
        backed, axis=axis, matrix=matrix, chunk_size=100, convert_to_dense=False,
    ):
        values = block.data if sp.issparse(block) else np.asarray(block)
        if values.size == 0:
            continue
        if not _looks_like_counts(values):
            _messages.emit(
                "flavor='seurat_v3' expects raw count data, but non-integer or "
                "negative values were found. Results may be unreliable.",
                UserWarning,
            )
        break


def _top_n_mask(scores: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Boolean mask selecting the top ``n_top_genes`` by ``scores`` (NaN-safe).

    NaN scores are always excluded. Ties at the cutoff are all included
    (same ``>=`` semantics scanpy's own ``_nth_highest``/``_subset_genes``
    use for the dispersion-based flavors), so more than ``n_top_genes``
    entries can come back true. Use :func:`_seurat_v3_select` for
    ``seurat_v3``, whose selection semantics are different.
    """
    finite = scores[~np.isnan(scores)]
    if finite.size == 0 or n_top_genes <= 0:
        return np.zeros(scores.shape, dtype=bool)
    n_top = min(n_top_genes, finite.size)
    cutoff = finite.min() if n_top >= finite.size else np.sort(finite)[::-1][n_top - 1]
    return np.nan_to_num(scores, nan=-np.inf) >= cutoff


def _seurat_v3_select(
    norm: np.ndarray, n_top_genes: int, *, paper: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scanpy's ``seurat_v3`` selection over batches: ``(highly_variable,
    nbatches, median_rank)`` from the ``(n_batches, n_vars)`` normalised variances.

    Genes are ranked within each batch (``argsort(argsort(-x))``: ties broken
    by position, so exactly ``n_top_genes`` are chosen even when low-count
    genes share a normalised variance); a gene's ``nbatches`` counts the
    batches where it ranks in the top ``n_top_genes`` and its median rank is
    taken over those. ``seurat_v3`` sorts by median rank, then batches;
    ``seurat_v3_paper`` by batches, then median rank. With one batch both
    are the top ``n_top_genes``.
    """
    ranks = np.argsort(np.argsort(-norm, axis=1), axis=1).astype(np.float32)
    nbatches = (ranks < n_top_genes).sum(axis=0)
    ranks[ranks >= n_top_genes] = np.nan
    median_rank = np.ma.median(np.ma.masked_invalid(ranks), axis=0).filled(np.nan)
    order = pd.DataFrame({"rank": median_rank, "n": nbatches}).sort_values(
        ["n", "rank"] if paper else ["rank", "n"], ascending=[False, True] if paper else [True, False],
        na_position="last",
    ).index
    highly_variable = np.zeros(norm.shape[1], dtype=bool)
    highly_variable[order[:n_top_genes]] = True
    return highly_variable, nbatches, median_rank


def _seurat_v3_flavor(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    codes: np.ndarray,
    n_batches: int,
    span: float,
    chunk_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sums, sqsums, n_cells, norm_gene_var)`` per batch via the vst method (Stuart et al. 2019).

    Reproduces ``scanpy.pp.highly_variable_genes(..., flavor="seurat_v3")``
    per batch: fit a degree-2 LOESS of ``log10(variance)`` on ``log10(mean)``
    from raw counts (pass 1), then compute the variance of each gene after
    clipping to a regularized standard deviation (pass 2), via the
    closed-form identity ``Var(clip(x)) = (n*mean^2 + sum(clip(x)^2) -
    2*mean*sum(clip(x))) / ((n-1)*reg_std^2)``, which makes pass 2 chunk- and
    sparse-friendly. Two passes whatever the number of batches: each pass
    accumulates every batch's sums at once.
    """
    from skmisc.loess import loess

    _check_looks_like_counts(backed, storage_format=storage_format, layer=layer)
    moments = dict(storage_format=storage_format, layer=layer, codes=codes, n_batches=n_batches,
                   chunk_size=chunk_size, show_progress=show_progress)
    sums, sqsums, n = _stream_gene_moments(backed, transform="identity", **moments)
    mean, var = _mean_var(sums, sqsums, n)

    reg_std = np.ones_like(mean)
    for b in range(n_batches):
        not_const = var[b] > 0
        if not_const.any():
            model = loess(
                np.log10(mean[b, not_const]), np.log10(var[b, not_const]),
                span=span, degree=2, surface="interpolate",
            )
            model.fit()
            reg_std[b, not_const] = np.sqrt(10.0 ** model.outputs.fitted_values)
    clip_val = reg_std * np.sqrt(n)[:, None] + mean  # >= 0: clipping keeps zeros

    def _clipped(block, block_codes, genes):
        clip = clip_val[:, genes]
        if sp.issparse(block):
            block = block.tocoo(copy=True)
            block.data = np.minimum(block.data, clip[block_codes[block.row], block.col])
            return block, block.multiply(block)
        clipped = np.minimum(block, clip[block_codes])
        return clipped, np.square(clipped)

    clipped_sum, clipped_sqsum = _stream_reduce(desc="Clipping counts", block_fn=_clipped, backed=backed, **moments)
    n = n[:, None]
    norm_gene_var = (n * np.square(mean) + clipped_sqsum - 2 * clipped_sum * mean) / ((n - 1) * np.square(reg_std))
    return sums, sqsums, n.ravel(), norm_gene_var


def _mean_dispersion_flavor(
    mean: np.ndarray,
    var: np.ndarray,
    *,
    n_top_genes: int | None,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(variances_norm, highly_variable)`` for the mean/dispersion flavor.

    Reproduces ``scanpy.pp.highly_variable_genes(..., flavor="seurat")``
    (Satija et al. 2015) applied to precomputed moments instead of a
    materialized, log1p-normalized expression matrix: bin genes by
    ``log1p(mean)`` into ``n_bins`` equal-width bins, z-score each gene's
    ``log(dispersion)`` against its bin's mean/std, then either take the top
    ``n_top_genes`` by normalized dispersion or threshold on
    ``min_mean``/``max_mean``/``min_disp``.
    """
    mean_safe = mean.copy()
    mean_safe[mean_safe == 0] = 1e-12
    dispersion = var / mean_safe
    # Constant genes (var == 0) have dispersion == 0; NaN them out before
    # logging rather than letting log(0) run, then suppress the resulting
    # (expected, propagates as NaN) "invalid value" warning from log(NaN).
    dispersion = np.where(dispersion == 0, np.nan, dispersion)
    with np.errstate(invalid="ignore"):
        log_disp = np.log(dispersion)
    log_mean = np.log1p(mean_safe)

    bins = pd.DataFrame({"log_mean": log_mean, "log_disp": log_disp})
    bins["mean_bin"] = pd.cut(bins["log_mean"], bins=n_bins)
    grouped = bins.groupby("mean_bin", observed=True)["log_disp"]
    bin_stats = grouped.agg(avg="mean", dev="std")

    # A bin with a single gene has an undefined (NaN) std; per Seurat/scanpy
    # convention, that gene's normalized dispersion is set to exactly 1 by
    # zeroing the bin average and using its own dispersion as the "spread".
    one_gene_per_bin = bin_stats["dev"].isnull()
    if one_gene_per_bin.any():
        bin_stats.loc[one_gene_per_bin, "dev"] = bin_stats.loc[one_gene_per_bin, "avg"]
        bin_stats.loc[one_gene_per_bin, "avg"] = 0.0

    aligned = bin_stats.loc[bins["mean_bin"]]
    variances_norm = (bins["log_disp"].to_numpy() - aligned["avg"].to_numpy()) / aligned["dev"].to_numpy()

    if n_top_genes is not None:
        highly_variable = _top_n_mask(variances_norm, n_top_genes)
    else:
        variances_norm_filled = np.nan_to_num(variances_norm)
        highly_variable = (
            (log_mean > min_mean)
            & (log_mean < max_mean)
            & (variances_norm_filled > min_disp)
        )

    return variances_norm, highly_variable


def _mean_dispersion_batches(
    sums: np.ndarray,
    sqsums: np.ndarray,
    n: np.ndarray,
    *,
    n_top_genes: int | None,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Scanpy's ``flavor="seurat"`` with ``batch_key``: each batch on the
    genes expressed in it (zeros for the others), averaged over batches. The
    top ``n_top_genes`` by the number of batches selecting them, then by mean
    normalised dispersion; or the thresholds applied to the averages."""
    mean, var = _mean_var(sums, sqsums, n)
    norm = np.zeros(mean.shape)
    log_mean = np.zeros(mean.shape)
    selected = np.zeros(mean.shape, dtype=bool)
    for b, expressed in enumerate(sums > 0):
        norm[b, expressed], selected[b, expressed] = _mean_dispersion_flavor(
            mean[b, expressed], var[b, expressed], n_top_genes=n_top_genes,
            min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, n_bins=n_bins,
        )
        log_mean[b, expressed] = np.log1p(mean[b, expressed])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a gene with no dispersion in any batch
        variances_norm = np.nanmean(norm, axis=0)
    nbatches = selected.sum(axis=0)
    if n_top_genes is not None:
        order = pd.DataFrame({"n": nbatches, "norm": variances_norm}).sort_values(
            ["n", "norm"], ascending=False, na_position="last",
        ).index
        highly_variable = np.zeros(mean.shape[1], dtype=bool)
        highly_variable[order[:n_top_genes]] = True
    else:
        log_mean = log_mean.mean(axis=0)
        highly_variable = (
            (log_mean > min_mean) & (log_mean < max_mean) & (np.nan_to_num(variances_norm) > min_disp)
        )
    extra = {"highly_variable_nbatches": nbatches, "highly_variable_intersection": nbatches == len(n)}
    return variances_norm, highly_variable, extra


def _batch_codes(
    path: Path, batch_key: str | None, cell_mask: np.ndarray | None, n_obs: int,
) -> tuple[np.ndarray, int]:
    """Each cell's batch code (``-1``: not used) and the number of batches."""
    used = np.ones(n_obs, dtype=bool) if cell_mask is None else cell_mask
    codes = np.full(n_obs, -1, dtype=np.int64)
    if batch_key is None:
        codes[used] = 0
        return codes, 1
    labels = load_obs(path)[batch_key][used]
    if labels.isna().any():
        raise ValueError(f"obs[{batch_key!r}] has missing values")
    codes[used], names = pd.factorize(labels, sort=True)
    sizes = np.bincount(codes[used], minlength=len(names))
    if (sizes < 2).any():
        raise ValueError(f"batch {names[np.argmax(sizes < 2)]!r} has fewer than 2 cells")
    return codes, len(names)


def _resolve_cell_mask(
    path: Path,
    *,
    n_obs: int,
    perturbation_column: str | None,
    control_label: str | None,
    cell_mask: "Literal['control'] | np.ndarray | None",
    verbose: int | bool,
) -> np.ndarray | None:
    """Resolve ``cell_mask`` to a plain boolean array (or ``None``) from ``obs`` alone.

    Implements the ``cell_mask="control"`` default (Item 2): control cells
    only, so PCA embeddings built on the resulting HVG set reflect baseline
    cell-state heterogeneity rather than perturbation effects. Reads only
    ``obs`` (via ``load_obs``) -- ``.X`` is never touched here.
    """
    if cell_mask is None:
        return None

    if isinstance(cell_mask, str):
        if cell_mask != "control":
            raise ValueError(f"cell_mask must be 'control', an array, or None; got {cell_mask!r}.")
        if perturbation_column is None:
            raise ValueError(
                "cell_mask='control' requires perturbation_column; pass it explicitly, "
                "or set cell_mask=None to use all cells."
            )
        obs = load_obs(path)
        if perturbation_column not in obs.columns:
            raise KeyError(
                f"Perturbation column {perturbation_column!r} was not found in adata.obs. "
                f"Available columns: {list(obs.columns)}"
            )
        labels = obs[perturbation_column].astype(str).to_numpy()
        resolved_control_label = resolve_control_label(labels, control_label, verbose=verbose)
        mask = labels == resolved_control_label
        if not mask.any():
            raise ValueError(
                f"No cells found with label {resolved_control_label!r} in "
                f"obs[{perturbation_column!r}]; pass cell_mask=None to use all cells."
            )
        return mask

    mask = np.asarray(cell_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != n_obs:
        raise ValueError(
            f"cell_mask must be a 1D boolean array of length n_obs ({n_obs}); "
            f"got shape {mask.shape}."
        )
    if not mask.any():
        raise ValueError(
            "cell_mask selects zero cells; pass a mask with at least one True entry, "
            "or cell_mask=None to use all cells."
        )
    return mask


def highly_variable_genes(
    data: str | Path | AnnData | ad.AnnData,
    *,
    flavor: Literal["seurat_v3", "seurat_v3_paper", "mean_dispersion"] = "seurat_v3",
    n_top_genes: int | None = 2000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_bins: int = 20,
    span: float = 0.3,
    layer: str | None = None,
    batch_key: str | None = None,
    perturbation_column: str | None = None,
    control_label: str | None = None,
    cell_mask: "Literal['control'] | np.ndarray | None" = "control",
    gene_name_column: str | None = None,
    chunk_size: int | None = None,
    inplace: bool = True,
    verbose: int | bool = True,
) -> pd.DataFrame | None:
    """Select highly variable genes from a backed AnnData, streamed off disk.

    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx AnnData / backed anndata.AnnData.
    flavor
        HVG selection method:

        - ``"seurat_v3"`` (default; Stuart et al. 2019, scanpy's
          ``flavor="seurat_v3"``): ranks genes by standardized variance fit
          via LOESS. Expects **raw counts**; requires ``n_top_genes``.
        - ``"seurat_v3_paper"`` (scanpy's): as ``"seurat_v3"``, but with
          ``batch_key`` genes are ranked by the number of batches selecting
          them first, then by median rank (``"seurat_v3"``: the reverse).
        - ``"mean_dispersion"`` (Satija et al. 2015, scanpy's
          ``flavor="seurat"``): bins genes by mean expression and
          z-normalizes dispersion within each bin. Expects
          **log1p-normalized data**; needs no extra dependency.
    n_top_genes
        Number of top genes to select by normalized variance/dispersion.
        Required for the ``seurat_v3`` flavors. For ``flavor="mean_dispersion"``,
        if ``None``, ``min_mean``/``max_mean``/``min_disp`` thresholds are
        used instead.
    min_mean, max_mean, min_disp
        ``mean_dispersion``-only threshold cutoffs used when ``n_top_genes``
        is ``None``.
    n_bins
        ``mean_dispersion``-only: number of equal-width bins used to
        normalize dispersion by mean expression level.
    span
        ``seurat_v3``-only: LOESS smoothing span (fraction of genes used to
        estimate the variance at each point).
    layer
        If provided, use ``adata.layers[layer]`` instead of ``adata.X``.
    batch_key
        ``obs`` column of batches (e.g. samples): genes are selected within
        each batch and combined as Scanpy does, so variation between batches
        does not dominate the set. Each batch needs two cells (within
        ``cell_mask``). Two passes over the matrix however many batches.
    perturbation_column
        Column in ``obs`` containing perturbation labels. Required when
        ``cell_mask="control"`` (its default); ignored when ``cell_mask`` is
        ``None`` or an explicit array.
    control_label
        Label identifying control cells. If ``None``, auto-detected from
        ``perturbation_column``. Only used when ``cell_mask="control"``.
    cell_mask
        Which cells to compute gene moments over:

        - ``"control"`` (default): control cells only, resolved from
          ``perturbation_column``/``control_label``. This is a
          CRISPR/Perturb-seq-specific default -- selecting HVGs from all
          cells lets on-target perturbation effects dominate the variable
          gene set, structuring downstream PCA around *which perturbation a
          cell received* rather than baseline cell-state heterogeneity.
        - ``None``: use all cells (the scanpy/Seurat default).
        - An explicit boolean array: a custom cell subset.
    gene_name_column
        Column in var containing gene names, used only to sanity-check that
        gene identifiers look like symbols, not Ensembl IDs.
    chunk_size
        Number of cells (or, for CSC-stored data, genes) to process per
        chunk. If ``None``, automatically calculated based on available
        memory.
    inplace
        If ``True`` on a backed AnnData, write ``var["highly_variable"]``
        (bool), ``var["means"]``, ``var["variances"]``,
        ``var["variances_norm"]`` back to the file and return ``None``. If
        ``False``, return a DataFrame with those same columns, indexed by
        gene, without modifying the file. With ``batch_key``, ``means`` and
        ``variances`` are over all cells used, ``variances_norm`` is the mean
        over batches, and ``highly_variable_nbatches`` is added, with
        ``highly_variable_rank`` (``seurat_v3`` flavors) or
        ``highly_variable_intersection`` (``mean_dispersion``), Scanpy's
        columns.
    verbose
        Print progress and a completion summary.

    Returns
    -------
    pd.DataFrame or None
        DataFrame indexed by gene when ``inplace=False``, else ``None``.

    Raises
    ------
    ValueError
        If ``cell_mask="control"`` is requested without
        ``perturbation_column``, if the resolved/explicit cell mask selects
        zero cells, if a ``seurat_v3`` flavor is used without
        ``n_top_genes``, or if a batch has fewer than two cells.
    """
    if flavor not in ("seurat_v3", "seurat_v3_paper", "mean_dispersion"):
        raise ValueError(
            f"Unknown flavor {flavor!r}; expected 'seurat_v3', 'seurat_v3_paper' or 'mean_dispersion'."
        )
    if flavor != "mean_dispersion" and n_top_genes is None:
        raise ValueError(f"n_top_genes is required for flavor={flavor!r}.")

    path = resolve_path_to_update(data, "var") if inplace else resolve_data_path(data)
    _messages.print_reading(verbose, "pp.highly_variable_genes", path)

    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_obs, n_vars = backed.n_obs, backed.n_vars

        cell_mask_arr = _resolve_cell_mask(
            path,
            n_obs=n_obs,
            perturbation_column=perturbation_column,
            control_label=control_label,
            cell_mask=cell_mask,
            verbose=verbose,
        )

        matrix = backed.X if layer is None else backed.layers[layer]
        detected_format = _detect_backed_sparse_format(matrix)
        storage_format = detected_format if detected_format is not None else "dense"
        if chunk_size is None:
            chunk_size = (
                calculate_optimal_gene_chunk_size(n_obs, n_vars)
                if storage_format == "csc"
                else calculate_optimal_chunk_size(n_obs, n_vars)
            )

        codes, n_batches = _batch_codes(path, batch_key, cell_mask_arr, n_obs)
        common = dict(
            storage_format=storage_format, layer=layer, codes=codes, n_batches=n_batches,
            chunk_size=chunk_size, show_progress=bool(verbose),
        )
        if flavor == "mean_dispersion":
            sums, sqsums, n = _stream_gene_moments(backed, transform="expm1", **common)
            thresholds = dict(n_top_genes=n_top_genes, min_mean=min_mean, max_mean=max_mean,
                              min_disp=min_disp, n_bins=n_bins)
            if batch_key is None:
                variances_norm, highly_variable = _mean_dispersion_flavor(
                    *_mean_var(sums[0], sqsums[0], n[0]), **thresholds,
                )
                extra = {}
            else:
                variances_norm, highly_variable, extra = _mean_dispersion_batches(sums, sqsums, n, **thresholds)
        else:
            sums, sqsums, n, norm = _seurat_v3_flavor(backed, span=span, **common)
            highly_variable, nbatches, rank = _seurat_v3_select(
                norm, n_top_genes, paper=flavor == "seurat_v3_paper",
            )
            variances_norm = norm.mean(axis=0)
            extra = {} if batch_key is None else {
                "highly_variable_rank": rank, "highly_variable_nbatches": nbatches,
            }
    finally:
        backed.file.close()

    mean, var = _mean_var(sums.sum(axis=0), sqsums.sum(axis=0), n.sum())
    var_df = load_var(path)
    kept = int(highly_variable.sum())
    _messages.print_done(
        verbose, "pp.highly_variable_genes", f"{kept}/{n_vars} genes marked highly variable",
    )

    columns = {
        "means": mean,
        "variances": var,
        "variances_norm": variances_norm,
        "highly_variable": highly_variable,
        **extra,
    }
    if inplace:
        # A previous run's batch columns describe another selection.
        stale = {"highly_variable_nbatches", "highly_variable_rank", "highly_variable_intersection"} - set(extra)
        write_var(path, var_df.drop(columns=list(stale & set(var_df.columns))).assign(**columns))
        return None
    return pd.DataFrame(columns, index=var_df.index)
