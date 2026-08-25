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
from tqdm.auto import tqdm

from . import _messages
from .data import (
    AnnData,
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
    write_var,
)


def _stream_reduce(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    cell_mask: np.ndarray | None,
    chunk_size: int,
    show_progress: bool,
    desc: str,
    block_fn: Callable[[np.ndarray | sp.spmatrix, slice], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``matrix`` off disk, applying ``block_fn`` per chunk and combining results.

    Shared dispatch/looping core for :func:`_stream_gene_moments` and
    :func:`_stream_clipped_sums`: column-chunked for CSC (each chunk already
    holds every cell for its genes, so ``block_fn``'s per-gene outputs are
    assigned directly by ``slc``), row-chunked with a running accumulator
    otherwise (``block_fn``'s outputs are summed across cell chunks).
    ``convert_to_dense=False`` throughout so a chunk's memory is ``O(nnz in
    that chunk)``, not ``O(chunk_size x n_genes)`` -- for the CSC path in
    particular, a "column chunk" spans every cell, so staying sparse is not
    optional.
    """
    matrix = backed.X if layer is None else backed.layers[layer]

    if storage_format == "csc":
        n_chunks = (backed.n_vars + chunk_size - 1) // chunk_size
        out_a = np.empty(backed.n_vars)
        out_b = np.empty(backed.n_vars)
        for slc, block in tqdm(
            iter_matrix_chunks(backed, axis=1, matrix=matrix, chunk_size=chunk_size, convert_to_dense=False),
            total=n_chunks,
            desc=f"{desc} (CSC)",
            disable=not show_progress,
        ):
            if cell_mask is not None:
                block = block[cell_mask]
            a, b = block_fn(block, slc)
            out_a[slc] = a
            out_b[slc] = b
        return out_a, out_b

    n_chunks = (backed.n_obs + chunk_size - 1) // chunk_size
    out_a = np.zeros(backed.n_vars)
    out_b = np.zeros(backed.n_vars)
    for slc, block in tqdm(
        iter_matrix_chunks(backed, axis=0, matrix=matrix, chunk_size=chunk_size, convert_to_dense=False),
        total=n_chunks,
        desc=desc,
        disable=not show_progress,
    ):
        local_mask = None if cell_mask is None else cell_mask[slc]
        if local_mask is not None:
            if not np.any(local_mask):
                continue
            block = block[local_mask]
        a, b = block_fn(block, slc)
        out_a += a
        out_b += b
    return out_a, out_b


def _stream_gene_moments(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    cell_mask: np.ndarray | None,
    transform: Literal["identity", "expm1"],
    chunk_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(mean, var, n_cells_used)`` per gene, streamed off disk.

    ``expm1``/masking are applied to a per-block copy (never the shared
    backed slice) and, for the sparse case, only to ``block.data`` -- both
    operations are sparsity-preserving (``expm1(0) == 0``; masking rows
    never touches nonzero *values*), so no dense transformed copy of a
    chunk is ever materialized. Variance uses the unbiased (``ddof=1``)
    estimator, computed from the per-gene sum/sum-of-squares totals after
    streaming -- these totals are already summed over every cell in both
    the CSC and row-chunked dispatch, so the same closed-form pass works
    for either.
    """
    n_used = int(cell_mask.sum()) if cell_mask is not None else backed.n_obs

    def _sum_and_sqsum(block, _slc: slice) -> tuple[np.ndarray, np.ndarray]:
        if transform == "expm1":
            if sp.issparse(block):
                block = block.copy()
                block.data = np.expm1(block.data)
            else:
                block = np.expm1(block)
        if sp.issparse(block):
            s = np.asarray(block.sum(axis=0)).ravel()
            ss = np.asarray(block.multiply(block).sum(axis=0)).ravel()
        else:
            s = block.sum(axis=0)
            ss = np.square(block).sum(axis=0)
        return s, ss

    s_acc, ss_acc = _stream_reduce(
        backed,
        storage_format=storage_format,
        layer=layer,
        cell_mask=cell_mask,
        chunk_size=chunk_size,
        show_progress=show_progress,
        desc="Computing gene moments",
        block_fn=_sum_and_sqsum,
    )
    mean = s_acc / n_used
    var = (ss_acc - n_used * np.square(mean)) / max(n_used - 1, 1)
    return mean, var, n_used


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
        looks_like_counts = not np.any(values < 0) and np.all(np.isclose(values, np.round(values)))
        if not looks_like_counts:
            warnings.warn(
                "flavor='seurat_v3' expects raw count data, but non-integer or "
                "negative values were found. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        break


def _top_n_mask(scores: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Boolean mask selecting the top ``n_top_genes`` by ``scores`` (NaN-safe).

    NaN scores are always excluded. Ties at the cutoff are all included
    (same ``>=`` semantics scanpy's own ``_nth_highest``/``_subset_genes``
    use for the dispersion-based flavors), so more than ``n_top_genes``
    entries can come back true. Use :func:`_rank_top_n_mask` for
    ``seurat_v3``, whose selection semantics are different.
    """
    finite = scores[~np.isnan(scores)]
    if finite.size == 0 or n_top_genes <= 0:
        return np.zeros(scores.shape, dtype=bool)
    n_top = min(n_top_genes, finite.size)
    cutoff = finite.min() if n_top >= finite.size else np.sort(finite)[::-1][n_top - 1]
    return np.nan_to_num(scores, nan=-np.inf) >= cutoff


def _rank_top_n_mask(scores: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Boolean mask selecting exactly the top ``n_top_genes`` by ``scores``.

    Unlike :func:`_top_n_mask`'s ``>=`` cutoff, this always returns exactly
    ``min(n_top_genes, scores.size)`` entries -- ties are broken by argsort
    order rather than all being included. This is ``scanpy``'s own
    ``flavor="seurat_v3"`` selection mechanism (``argsort(argsort(-x))``
    ranks, then ``rank < n_top_genes``), and matters in practice: real
    low-count genes can share an identical normalized variance (e.g.
    several genes each detected in a single cell with the same count), and
    a ``>=`` cutoff would let every tied gene at the boundary through,
    silently returning more genes than requested.
    """
    n_top = min(n_top_genes, scores.size)
    ranks = np.argsort(np.argsort(-scores))
    return ranks < n_top


def _stream_clipped_sums(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    cell_mask: np.ndarray | None,
    clip_val: np.ndarray,
    chunk_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(clipped_sum, clipped_sqsum)`` per gene -- ``seurat_v3``'s second pass.

    Clips each entry to its own gene's ``clip_val`` directly on
    ``block.data`` (sparse) or the dense block, never materializing a dense
    clipped copy of a chunk. This is safe because ``clip_val`` is always
    non-negative (``reg_std * sqrt(n) + mean``, both terms >= 0), so
    clipping a zero entry is a no-op and sparsity is preserved. Dispatches
    the same way as :func:`_stream_gene_moments`.
    """

    def _clip_and_sum(block, slc: slice) -> tuple[np.ndarray, np.ndarray]:
        clip_local = clip_val[slc] if storage_format == "csc" else clip_val
        if sp.issparse(block):
            block = block.tocsc() if storage_format == "csc" else block.tocsr()
            block = block.copy()
            if storage_format == "csc":
                col_sizes = np.diff(block.indptr)
                per_element_clip = np.repeat(clip_local, col_sizes)
            else:
                per_element_clip = clip_local[block.indices]
            block.data = np.minimum(block.data, per_element_clip)
            s = np.asarray(block.sum(axis=0)).ravel()
            ss = np.asarray(block.multiply(block).sum(axis=0)).ravel()
        else:
            clipped = np.minimum(block, clip_local[None, :])
            s = clipped.sum(axis=0)
            ss = np.square(clipped).sum(axis=0)
        return s, ss

    return _stream_reduce(
        backed,
        storage_format=storage_format,
        layer=layer,
        cell_mask=cell_mask,
        chunk_size=chunk_size,
        show_progress=show_progress,
        desc="Clipping counts",
        block_fn=_clip_and_sum,
    )


def _seurat_v3_flavor(
    backed: ad.AnnData,
    *,
    storage_format: str,
    layer: str | None,
    cell_mask: np.ndarray | None,
    span: float,
    chunk_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mean, variances, norm_gene_var)`` via the vst method (Stuart et al. 2019).

    Reproduces ``scanpy.pp.highly_variable_genes(..., flavor="seurat_v3")``:
    fit a degree-2 LOESS of ``log10(variance)`` on ``log10(mean)`` from raw
    counts (pass 1), then compute the variance of each gene after clipping
    to a regularized standard deviation (pass 2), via the closed-form
    identity ``Var(clip(x)) = (n*mean^2 + sum(clip(x)^2) - 2*mean*sum(clip(x)))
    / ((n-1)*reg_std^2)`` -- exactly what makes pass 2 chunk- and
    sparse-friendly instead of needing a materialized standardized array.
    Two full passes are inherent to the method: the clip threshold is a
    global, all-genes-dependent quantity from the pass-1 LOESS fit, so pass
    2 cannot start until pass 1 has finished for every gene.
    """
    from skmisc.loess import loess

    _check_looks_like_counts(backed, storage_format=storage_format, layer=layer)

    mean, var, n_used = _stream_gene_moments(
        backed,
        storage_format=storage_format,
        layer=layer,
        cell_mask=cell_mask,
        transform="identity",
        chunk_size=chunk_size,
        show_progress=show_progress,
    )

    not_const = var > 0
    estimated_log_var = np.zeros(mean.shape[0], dtype=np.float64)
    if not_const.any():
        model = loess(
            np.log10(mean[not_const]), np.log10(var[not_const]),
            span=span, degree=2, surface="interpolate",
        )
        model.fit()
        estimated_log_var[not_const] = model.outputs.fitted_values
    reg_std = np.sqrt(10.0 ** estimated_log_var)

    clip_val = reg_std * np.sqrt(n_used) + mean

    clipped_sum, clipped_sqsum = _stream_clipped_sums(
        backed,
        storage_format=storage_format,
        layer=layer,
        cell_mask=cell_mask,
        clip_val=clip_val,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )

    norm_gene_var = (
        n_used * np.square(mean) + clipped_sqsum - 2 * clipped_sum * mean
    ) / ((n_used - 1) * np.square(reg_std))

    return mean, var, norm_gene_var


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
    flavor: Literal["seurat_v3", "mean_dispersion"] = "seurat_v3",
    n_top_genes: int | None = 2000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_bins: int = 20,
    span: float = 0.3,
    layer: str | None = None,
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
        Path to h5ad file, or a crispyx/anndata AnnData object.
    flavor
        HVG selection method:

        - ``"seurat_v3"`` (default; Stuart et al. 2019, scanpy's
          ``flavor="seurat_v3"``): ranks genes by standardized variance fit
          via LOESS. Expects **raw counts**; requires ``n_top_genes``.
        - ``"mean_dispersion"`` (Satija et al. 2015, scanpy's
          ``flavor="seurat"``): bins genes by mean expression and
          z-normalizes dispersion within each bin. Expects
          **log1p-normalized data**; needs no extra dependency.
    n_top_genes
        Number of top genes to select by normalized variance/dispersion.
        Required for ``flavor="seurat_v3"``. For ``flavor="mean_dispersion"``,
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
        gene, without modifying the file.
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
        zero cells, or if ``flavor="seurat_v3"`` is used without
        ``n_top_genes``.
    """
    if flavor not in ("seurat_v3", "mean_dispersion"):
        raise ValueError(f"Unknown flavor {flavor!r}; expected 'seurat_v3' or 'mean_dispersion'.")
    if flavor == "seurat_v3" and n_top_genes is None:
        raise ValueError("n_top_genes is required for flavor='seurat_v3'.")

    path = resolve_data_path(data)
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

        if flavor == "seurat_v3":
            mean, var, variances_norm = _seurat_v3_flavor(
                backed,
                storage_format=storage_format,
                layer=layer,
                cell_mask=cell_mask_arr,
                span=span,
                chunk_size=chunk_size,
                show_progress=bool(verbose),
            )
            highly_variable = _rank_top_n_mask(variances_norm, n_top_genes)
        else:
            mean, var, _n_used = _stream_gene_moments(
                backed,
                storage_format=storage_format,
                layer=layer,
                cell_mask=cell_mask_arr,
                transform="expm1",
                chunk_size=chunk_size,
                show_progress=bool(verbose),
            )
            variances_norm, highly_variable = _mean_dispersion_flavor(
                mean,
                var,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                n_bins=n_bins,
            )
    finally:
        backed.file.close()

    var_df = load_var(path)
    kept = int(highly_variable.sum())
    _messages.print_done(
        verbose, "pp.highly_variable_genes", f"{kept}/{n_vars} genes marked highly variable",
    )

    if inplace:
        var_df["means"] = mean
        var_df["variances"] = var
        var_df["variances_norm"] = variances_norm
        var_df["highly_variable"] = highly_variable
        write_var(path, var_df)
        return None

    return pd.DataFrame(
        {
            "means": mean,
            "variances": var,
            "variances_norm": variances_norm,
            "highly_variable": highly_variable,
        },
        index=var_df.index,
    )
