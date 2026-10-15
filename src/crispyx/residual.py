"""Covariate-adjusted perturbation effects: control-fitted linear model residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence, get_args

import anndata as ad
import h5py
import numba
import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import _messages
from ._grouping import cell_labels
from ._checkpoint import _create_progress_context
from ._disk import estimate_bytes, warn_if_disk_space_low
from ._kernels import _residual_effects_rows
from ._memory import _resolve_memory_limit_bytes, _resolve_n_jobs
from ._provenance import KEY as _PROVENANCE_KEY, call_fingerprint, reusable, stamp
from ._statistics import _adjust_pvalue_matrix
from .data import (
    AnnData,
    _ensure_csr,
    _replace_on_success,
    _validate_counts_chunk,
    calculate_optimal_chunk_size,
    chunk_starts,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    scratch_copy_bytes,
    stream_on_fast_axis,
    validate_format_mismatch_policy,
)

ResidualNull = Literal["model", "robust"]

#: Covariates computed from each cell's counts during the pass, not read from obs.
COMPUTED_COVARIATES = ("log_total_counts", "mito_fraction")

#: Arguments that decide how a call runs but not what it writes.
_OPERATIONAL_ARGS = frozenset({
    "data", "chunk_size", "n_jobs", "memory_limit_gb", "format_mismatch_policy", "data_name",
    "output_path", "verbose", "force",
})

_NAME = "residual_effects"


def residual_effects(
    data: str | Path | AnnData | ad.AnnData,
    perturbation_column: str | pd.Series,
    control_label: str | None = None,
    *,
    covariates: Sequence[str],
    null: ResidualNull = "model",
    min_cells: int = 10,
    min_control_mean: float = 0.1,
    perturbations: Iterable[str] | None = None,
    gene_name_column: str | None = None,
    chunk_size: int | None = None,
    n_jobs: int | None = None,
    memory_limit_gb: float | None = None,
    format_mismatch_policy: Literal["auto", "warn", "convert", "off"] = "auto",
    data_name: str | None = None,
    output_path: str | Path | None = None,
    verbose: int | bool = True,
    force: bool = False,
) -> AnnData:
    """Perturbation effects adjusted for covariates by a model fitted on controls.

    The method of Feng et al. (2026, Cell Genomics,
    doi:10.1016/j.xgen.2025.101076). Each gene's
    ``log1p(CP10k)`` expression is fitted to the covariates by least squares
    over the control cells only; a perturbation's effect is the mean, over its
    cells, of observed minus fitted expression. Differences in donor, lane or
    batch between a perturbation's cells and the controls are thereby removed
    rather than read as an effect.

    Everything comes from one pass over the rows of the raw counts. The mean
    residual of perturbation ``t`` is ``mean_t(y) - mean_t(x) @ B``, so only
    per-perturbation sums and the controls' cross-products are kept, never a
    per-cell residual; memory is one perturbations x genes array.

    Parameters
    ----------
    data
        Path to an h5ad file of **raw counts**, :class:`crispyx.AnnData`, or an
        ``anndata.AnnData`` opened with ``backed='r'``. The function normalises
        (``log1p`` of counts per 10,000) itself and refuses input with
        non-integer or negative values in any chunk it reads.
    perturbation_column
        Column in ``adata.obs`` naming each cell's perturbation, or a
        ``pd.Series`` of labels indexed by ``obs_names``. Cells whose label is
        missing (NaN) are not used, which restricts a file to a subset of cells
        without writing a copy; row chunks without a used cell are not read.
    control_label
        Label of the control cells the model is fitted on. Inferred from common
        patterns when omitted.
    covariates
        Terms of the model besides the intercept; required, with no default.
        Each is an ``obs`` column (one-hot coded, first level dropped, when
        categorical, string or boolean; used as is when numeric) or one of two
        values computed from the counts:

        * ``"log_total_counts"``: natural log of the cell's total counts;
        * ``"mito_fraction"``: share of the cell's counts in genes whose symbol
          starts with ``MT-``.

        Adjust for how cells were processed (donor, lane, batch), which no
        perturbation changes. Avoid what the perturbation itself can change
        (depth, mitochondrial share, cell cycle): adjusting for it removes part
        of the effect.

        ``[]`` gives the plain difference from the controls. Interactions
        (lane x donor) and other scores are ``obs`` columns made beforehand.
        Perturbed cells in a categorical level no control shares cannot be
        adjusted and are left out (counted at ``verbose=1``); numeric
        covariates must be finite in used cells, and an ``obs`` column named
        like a computed value raises. Collinear
        covariates are fine unless a perturbation's effect would then depend
        on how the fit splits them, which raises.
    null
        Spread the z-score divides by:

        * ``"model"`` (the paper's test): ``sigma_g * sqrt(1/n_t + h_t)``,
          with ``sigma_g`` the residual standard deviation of the control fit
          and ``h_t = mean_t(x) (X^T X)^+ mean_t(x)^T`` the control fit's own
          error at the perturbation's mean design, which every perturbation
          shares. ``h_t`` is negligible with many controls (the paper's
          ``sigma_g / sqrt(n_t)``) and keeps z calibrated with few. Treats
          cells as independent; cells sharing a guide or donor are not, so it
          calls too many hits.
        * ``"robust"``: ``z = (effect * sqrt(n_t) - m_g) / s_g``, with ``m_g``
          the median over perturbations of ``effect * sqrt(n_t)`` and ``s_g``
          1.4826 x their median absolute deviation. It assumes most
          perturbations leave gene ``g`` unchanged -- true for genome-wide
          screens, false for a focused panel where every target is active.

        ``h_t`` is written to ``obs`` (``leverage``) and ``sigma_g``, ``m_g``
        and ``s_g`` to ``var``, so either z can be recomputed.
    min_cells
        Perturbations with fewer usable cells are left out (listed at
        ``verbose=1``).
    min_control_mean
        Genes are tested when the controls' mean ``log1p(CP10k)`` exceeds this
        (Feng et al. used 0.1). Untested genes keep their effect but have NaN
        z-scores and p-values.
    perturbations
        Subset of perturbation labels to report. All non-control labels when
        omitted. The fit uses the controls either way.
    gene_name_column
        Column in ``adata.var`` with gene symbols. Uses ``var_names`` when
        omitted.
    chunk_size
        Cells read per chunk. Auto-selected when omitted.
    n_jobs
        Threads for the per-chunk computation (a chunk's perturbations are
        split between them); the next chunk is read meanwhile. All cores when
        omitted. Results do not depend on it.
    memory_limit_gb
        Memory budget. Three perturbations x genes float64 arrays
        (accumulator, effect, robust spread) and one genes x design-columns
        array per thread must fit in it, else the call refuses before
        reading the matrix. Detected from the system when omitted.
    format_mismatch_policy
        How to handle a source stored as CSC, whose row-(cell-)streaming here
        is O(total_nnz) per chunk and can be ~100x slower than CSR:

        * ``"auto"`` (default): measure the source's read throughput and
          convert only when the re-reads would cost more than one conversion.
          See :func:`crispyx.data.resolve_auto_format_mismatch_policy`.
        * ``"warn"``: proceed but log a single actionable warning.
        * ``"convert"``: stream from a temporary CSR copy written beside the
          output (about the source's size in free disk), removed before
          returning.
        * ``"off"``: proceed silently with no warning.
    data_name
        Custom stem for the output filename.
    output_path
        Exact output h5ad path. Defaults to ``<stem>_cx_residual_effects.h5ad``
        next to the input.
    verbose
        ``1`` / ``True`` prints progress and a summary.
    force
        Recompute even when an existing output was made by the same call.

    Returns
    -------
    AnnData
        On-disk result with perturbations in observations and genes in
        variables. ``X`` is the effect: the mean residual, a difference of mean
        ``log1p(CP10k)`` values (natural log) -- not a log2 fold change.
        Layers ``z_score``, ``pvalue`` (two-sided normal) and ``pvalue_adj``
        (Benjamini-Hochberg within each perturbation). ``obs`` holds ``n_cells``,
        ``n_genes_tested`` and ``leverage``; ``var`` holds ``control_mean``,
        ``residual_sd``, ``robust_center``, ``robust_sd`` and ``tested``; ``uns`` records the null, the covariates
        and the design's column names.
    """
    # A one-shot iterable would be fingerprinted by its repr and then used up.
    if perturbations is not None:
        perturbations = [str(p) for p in perturbations]
    covariates = list(covariates)
    call_args = dict(locals())  # for the result fingerprint, before any other local
    if null not in get_args(ResidualNull):
        raise ValueError(f"{_NAME}: null must be 'model' or 'robust'; received {null!r}.")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")
    if min_control_mean < 0:
        raise ValueError("min_control_mean must be non-negative")
    validate_format_mismatch_policy(format_mismatch_policy)
    covariates = [str(c) for c in covariates]
    if len(set(covariates)) != len(covariates):
        raise ValueError(f"{_NAME}: covariates are listed more than once: {covariates}")

    path = resolve_data_path(data)
    resolved_output = resolve_output_path(
        path, suffix=_NAME, data_name=data_name, output_path=output_path,
    )
    fingerprint = call_fingerprint(path, call_args, _OPERATIONAL_ARGS)
    if not force and reusable(resolved_output, _NAME, fingerprint, verbose):
        return AnnData(resolved_output)

    _messages.print_reading(verbose, _NAME, path)
    backed = read_backed(path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        labels, label_name = cell_labels(backed.obs, perturbation_column)
        labelled = labels.notna().to_numpy()
        values = labels.astype(str).to_numpy()
        control_label = resolve_control_label(values[labelled], control_label, verbose=verbose)
        groups, targets = _group_codes(
            values, labelled, control_label, perturbations, min_cells, verbose,
        )
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column).astype(str)
        groups = _drop_cells_without_controls(backed.obs, covariates, groups, verbose)
        design = _Design.build(backed.obs, covariates, groups, gene_symbols)
        workers = max(1, min(_resolve_n_jobs(n_jobs), numba.config.NUMBA_NUM_THREADS))
        p = len(design.columns)
        # three perturbations x genes arrays; per thread, the controls' genes x
        # design columns cross-products and sums of squares plus two edge rows
        # (and one more cross-product array when the threads' are summed)
        needed = 8 * n_vars * (3 * (len(targets) + 1) + (workers + 1) * (p + 1) + 2 * workers)
        budget = _resolve_memory_limit_bytes(memory_limit_gb)
        if needed > budget:
            raise MemoryError(
                f"{_NAME}: {len(targets):,} perturbations x {n_vars:,} genes with {p} design "
                f"columns on {workers} thread(s) need {needed / 1e9:.1f} GB; the budget is "
                f"{budget / 1e9:.1f} GB. Pass fewer perturbations, fewer threads (n_jobs) or "
                "fewer covariate levels, or raise memory_limit_gb."
            )
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(
                n_obs, n_vars, available_memory_gb=memory_limit_gb,
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    finally:
        backed.file.close()

    _messages.vprint(
        verbose, _NAME,
        f"{len(targets):,} perturbations, {int((groups == 0).sum()):,} control cells, "
        f"design: {len(design.columns)} columns",
    )
    # Row 0 (the controls) stays zero: their sum is control_xty's intercept column.
    sums = np.zeros((len(targets) + 1, n_vars))
    control_xty = np.zeros((workers, n_vars, p))
    control_sumsq = np.zeros((workers, n_vars))
    edges = np.zeros((workers, 2, n_vars))
    totals = np.zeros(n_obs)
    mito = np.zeros(n_obs)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with stream_on_fast_axis(
        path, axis=0, policy=format_mismatch_policy, fn_name=_NAME,
        scratch_dir=resolved_output.parent, chunk_size=chunk_size,
        memory_limit_gb=memory_limit_gb, verbose=verbose,
    ) as stream_path:
        streamed = read_backed(stream_path)
        try:
            selected = groups >= 0
            n_chunks = len(chunk_starts(n_obs, chunk_size, selected=selected))
            with _create_progress_context(n_chunks, _NAME, verbose, unit="chunk") as pbar:
                for slc, block in iter_matrix_chunks(
                    streamed, axis=0, chunk_size=chunk_size, convert_to_dense=False,
                    warn_slow_axis=False, selected=selected, read_ahead=True,
                ):
                    block = _ensure_csr(block)
                    _validate_counts_chunk(block, context=_NAME)
                    chunk_groups = groups[slc]
                    target_rows, target_bounds, edge_groups, control_rows, control_bounds = _segments(
                        chunk_groups, np.diff(block.indptr), workers,
                    )
                    _residual_effects_rows(
                        block.indptr, block.indices, block.data, chunk_groups,
                        target_rows, target_bounds, edge_groups, edges,
                        control_rows, control_bounds,
                        design.category_columns[slc], design.continuous[slc],
                        design.continuous_columns, design.total_column, design.mito_column,
                        design.is_mito, sums, control_xty, control_sumsq,
                        totals[slc], mito[slc],
                    )
                    for w, k in zip(*np.nonzero(edge_groups > 0)):
                        sums[edge_groups[w, k]] += edges[w, k]
                    pbar.update(1)
        finally:
            streamed.file.close()

    control_xty = control_xty.sum(axis=0)
    control_sumsq = control_sumsq.sum(axis=0)
    used = (groups >= 0) & (totals > 0)
    n_cells = np.bincount(groups[used], minlength=len(targets) + 1).astype(np.float64)
    keep = n_cells[1:] >= min_cells
    if not keep.all():
        _messages.vprint(
            verbose, _NAME,
            f"{int((~keep).sum())} perturbation(s) have fewer than {min_cells} cells with counts "
            f"and are left out: {[targets[i] for i in np.flatnonzero(~keep)][:10]}",
        )
    if not keep.any():
        raise ValueError(f"{_NAME}: no perturbation has at least {min_cells} usable cells.")
    group_design_sums, control_xtx = design.cross_products(groups, used, totals, mito, len(targets) + 1)
    n_controls = int(n_cells[0])
    rows = 1 + np.flatnonzero(keep)
    n_target = n_cells[rows]
    mean_design = group_design_sums[rows] / n_target[:, None]
    # Least squares through the pseudo-inverse of X^T X, after centring and
    # scaling the non-intercept columns over the controls (X T): a covariate
    # of large magnitude but modest spread is otherwise nearly collinear with
    # the intercept and lost below the cutoff. The null space of X T, mapped
    # back through T, is what collinearity leaves free.
    transform = _standardising_transform(control_xtx)
    # eigh, not svd(hermitian=True): the latter zeroes the eigenvector of an
    # eigenvalue that is exactly 0, which is the null space wanted here.
    eigenvalues, eigenvectors = np.linalg.eigh(transform.T @ control_xtx @ transform)
    order = np.argsort(eigenvalues)[::-1]
    singular, vt = np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order].T
    independent = singular > singular[0] * p * np.finfo(np.float64).eps
    rank = int(independent.sum())
    basis = transform @ vt[independent].T
    coefficients = basis @ ((basis.T @ control_xty.T) / singular[independent, None])
    # x̄ (X^T X)^+ x̄ per perturbation: the control fit's share of the
    # variance of its effect, which every perturbation's effect carries.
    leverage = np.square((mean_design @ basis) / np.sqrt(singular[independent])).sum(axis=1)
    if rank < p:
        null_space = (transform @ vt[~independent].T).T
        null_space /= np.linalg.norm(null_space, axis=1, keepdims=True)
        _check_identified(mean_design, null_space, design.columns, targets, keep, rank)
        _messages.vprint(
            verbose, _NAME,
            f"the design has rank {rank} < {p} columns (collinear covariates); every "
            "perturbation's design lies within the controls', so its effect is still unique.",
        )
    df = n_controls - rank
    if df <= 0:
        raise ValueError(
            f"{_NAME}: {n_controls} control cells cannot fit {rank} independent design columns."
        )
    rss = control_sumsq - np.einsum("pg,gp->g", coefficients, control_xty)
    residual_sd = np.sqrt(np.maximum(rss, 0.0) / df)
    control_mean = control_xty[:, 0] / n_cells[0]

    effect = sums[rows]
    del sums
    effect /= n_target[:, None]
    effect -= mean_design @ coefficients
    # Each null scales an effect to unit variance: "model" by the control
    # fit's noise, counting its own error (shared by every perturbation)
    # through the leverage; "robust" by the spread across perturbations,
    # centred on their median, which also absorbs that shared error.
    scaled = effect * np.sqrt(n_target)[:, None]
    robust_center = np.median(scaled, axis=0)
    scaled -= robust_center
    robust_sd = 1.4826 * np.median(np.abs(scaled, out=scaled), axis=0)
    del scaled
    if null == "model":
        spread, center, unit = residual_sd, 0.0, 1.0 / np.sqrt(1.0 / n_target + leverage)
    else:
        spread, center, unit = robust_sd, robust_center, np.sqrt(n_target)
    tested = (control_mean > min_control_mean) & (spread > 0)
    kept_targets = [targets[i] for i in np.flatnonzero(keep)]

    obs_index = pd.Index(kept_targets, name="perturbation").astype(str)
    obs = pd.DataFrame(
        {
            label_name: obs_index.to_list(),
            "n_cells": n_target.astype(np.int64),
            "n_genes_tested": np.full(len(kept_targets), int(tested.sum()), dtype=np.int64),
            "leverage": leverage,
        },
        index=obs_index,
    )
    var = pd.DataFrame(
        {
            "control_mean": control_mean,
            "residual_sd": residual_sd,
            "robust_center": robust_center,
            "robust_sd": robust_sd,
            "tested": tested,
        },
        index=pd.Index(gene_symbols, name=None),
    )
    uns = {
        "method": _NAME,
        "null": null,
        "perturbation_column": label_name,
        "control_label": control_label,
        "covariates": covariates,
        "design_columns": list(design.columns),
        "n_control_cells": n_controls,
        "scale": "log1p(counts per 10,000)",
        _PROVENANCE_KEY: stamp(_NAME, fingerprint),
    }
    _messages.print_disk_estimate(
        verbose, _NAME,
        warn_if_disk_space_low(
            estimate_bytes(max(len(kept_targets), 1), n_vars, 4, overhead=1.10),
            resolved_output, context=_NAME,
        ),
    )
    with _replace_on_success(resolved_output) as partial:
        ad.AnnData(obs=obs, var=var, uns=uns).write(partial)
        with h5py.File(partial, "r+") as handle:
            _dense_dataset(handle, "X", effect)
            layers = handle.require_group("layers")
            z_ds = _dense_dataset(layers, "z_score", shape=effect.shape)
            p_ds = _dense_dataset(layers, "pvalue", shape=effect.shape)
            padj_ds = _dense_dataset(layers, "pvalue_adj", shape=effect.shape)
            step = max(1, (1 << 22) // max(n_vars, 1))
            for start in range(0, len(kept_targets), step):
                stop = start + step
                z = (effect[start:stop] * unit[start:stop, None] - center) / np.where(tested, spread, np.nan)
                pvalue = 2.0 * ndtr(-np.abs(z))
                z_ds[start:stop] = z
                p_ds[start:stop] = pvalue
                padj_ds[start:stop] = _adjust_pvalue_matrix(pvalue, "benjamini-hochberg")

    if int(verbose) >= 1:
        print(
            f"[cx] {_NAME}: {len(kept_targets):,} perturbations x {n_vars:,} genes "
            f"({int(tested.sum()):,} tested), {null} null"
        )
    _messages.print_saving(verbose, _NAME, resolved_output)
    return AnnData(resolved_output)


def _segments(groups: np.ndarray, row_nnz: np.ndarray, workers: int):
    """Split a chunk's rows into ``workers`` segments for the kernel: the
    perturbation rows ordered by group and cut where their non-zeros
    balance, and the control rows in equal shares (each segment has its own
    control accumulators). A group may span segments, so each segment's first
    and last group (``edge_groups``, -1 when absent) are accumulated
    privately; a file sorted by perturbation still keeps every thread busy."""
    rows = np.flatnonzero(groups > 0)
    rows = rows[np.argsort(groups[rows], kind="stable")]
    cumulative = np.cumsum(row_nnz[rows])
    if rows.size:
        cuts = np.searchsorted(cumulative, cumulative[-1] * np.arange(1, workers) / workers, side="right")
    else:
        cuts = np.zeros(workers - 1, dtype=np.int64)
    target_bounds = np.concatenate([[0], cuts, [rows.size]]).astype(np.int64)
    lo, hi = target_bounds[:-1], target_bounds[1:]
    filled = hi > lo
    edge_groups = np.full((workers, 2), -1, dtype=np.int64)
    first = groups[rows[lo[filled]]]
    last = groups[rows[hi[filled] - 1]]
    edge_groups[filled, 0] = first
    edge_groups[filled, 1] = np.where(last != first, last, -1)
    controls = np.flatnonzero(groups == 0)
    control_bounds = np.linspace(0, controls.size, workers + 1).astype(np.int64)
    return rows.astype(np.int64), target_bounds, edge_groups, controls.astype(np.int64), control_bounds


def _standardising_transform(xtx: np.ndarray) -> np.ndarray:
    """``T`` such that ``X T`` is the intercept followed by the other columns
    centred and scaled to unit variance over the cells of ``X^T X``; a
    column constant among them is centred to zero."""
    n = xtx[0, 0]
    mean = xtx[0, 1:] / n
    sd = np.sqrt(np.maximum(np.diag(xtx)[1:] / n - mean ** 2, 0.0))
    scale = np.where(sd > 0, sd, 1.0)
    transform = np.eye(xtx.shape[0])
    transform[0, 1:] = -mean / scale
    transform[1:, 1:] = np.diag(1.0 / scale)
    return transform


def _check_identified(mean_design, null_space, columns, targets, keep, rank) -> None:
    """Raise when a perturbation's effect is not unique: its mean design row
    has a component in the null space of the controls' design, so the
    effect would depend on how the fit splits the collinear columns."""
    scale = np.maximum(np.linalg.norm(mean_design, axis=1), 1.0)
    free = np.linalg.norm(mean_design @ null_space.T, axis=1) > 1e-8 * scale
    if not free.any():
        return
    collinear = [columns[i] for i in np.flatnonzero(np.abs(null_space).max(axis=0) > 1e-8)]
    kept = np.flatnonzero(keep)
    names = [targets[kept[i]] for i in np.flatnonzero(free)]
    raise ValueError(
        f"{_NAME}: the controls' design has rank {rank} < {len(columns)} columns; "
        f"{collinear} are collinear among the controls but not among the cells of "
        f"{len(names)} perturbation(s), whose effects would therefore be arbitrary: "
        f"{names[:10]}. Drop or merge the collinear covariates, or restrict the run to "
        "cells whose combinations occur among the controls."
    )


def _dense_dataset(group: h5py.Group, name: str, data=None, shape=None) -> h5py.Dataset:
    """A float64 dataset tagged as an anndata dense array."""
    if data is not None:
        ds = group.create_dataset(name, data=np.asarray(data, dtype=np.float64))
    else:
        ds = group.create_dataset(name, shape=shape, dtype=np.float64)
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"
    return ds


def _group_codes(
    values: np.ndarray,
    labelled: np.ndarray,
    control_label: str,
    perturbations: Iterable[str] | None,
    min_cells: int,
    verbose: int | bool,
) -> tuple[np.ndarray, list[str]]:
    """Per-cell group: 0 for controls, 1.. for reported perturbations, -1 unused."""
    codes, uniques = pd.factorize(pd.Series(values[labelled]), sort=False)
    uniques = uniques.astype(str)
    counts = np.bincount(codes, minlength=len(uniques))
    if control_label not in set(uniques):
        raise ValueError(f"{_NAME}: control label {control_label!r} matches no labelled cell.")
    if perturbations is not None:
        requested = [str(p) for p in perturbations]
        missing = sorted(set(requested) - set(uniques))
        if missing:
            raise ValueError(f"Requested perturbation(s) not found: {missing[:5]}")
        wanted = set(requested)
    else:
        wanted = set(uniques)
    targets = [
        label for label, count in zip(uniques, counts)
        if label != control_label and label in wanted and count >= min_cells
    ]
    small = [
        label for label, count in zip(uniques, counts)
        if label != control_label and label in wanted and count < min_cells
    ]
    if small:
        _messages.vprint(
            verbose, _NAME,
            f"{len(small)} perturbation(s) have fewer than {min_cells} cells and are left out: {small[:10]}",
        )
    if not targets:
        raise ValueError(f"{_NAME}: no perturbation has at least {min_cells} cells.")
    position = {label: i + 1 for i, label in enumerate(targets)}
    position[control_label] = 0
    lookup = np.array([position.get(label, -1) for label in uniques], dtype=np.int64)
    groups = np.full(values.shape[0], -1, dtype=np.int64)
    groups[labelled] = lookup[codes]
    return groups, targets


def _drop_cells_without_controls(obs: pd.DataFrame, covariates: list[str], groups: np.ndarray,
                                 verbose: int | bool) -> np.ndarray:
    """``groups`` with the perturbed cells in a categorical covariate level
    that no control shares set to unused (-1): the control fit has no
    estimate for that level, so their residuals would be undefined. Their
    perturbations keep their other cells."""
    controls = groups == 0
    uncovered = np.zeros(len(groups), dtype=bool)
    for name in covariates:
        if name in COMPUTED_COVARIATES or name not in obs.columns:
            continue
        series = obs[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        values = series.astype(str).to_numpy()
        uncovered |= (groups > 0) & ~np.isin(values, np.unique(values[controls]))
    if not uncovered.any():
        return groups
    _messages.vprint(
        verbose, _NAME,
        f"{int(uncovered.sum()):,} perturbed cell(s) are in a covariate level with no control "
        "cells, which the control fit cannot adjust for, and are left out.",
    )
    return np.where(uncovered, -1, groups)


class _Design:
    """The model's design: column names and what the kernel needs per cell."""

    def __init__(self, columns, category_columns, continuous, continuous_columns,
                 total_column, mito_column, is_mito):
        self.columns = columns
        self.category_columns = category_columns
        self.continuous = continuous
        self.continuous_columns = continuous_columns
        self.total_column = total_column
        self.mito_column = mito_column
        self.is_mito = is_mito

    @classmethod
    def build(cls, obs: pd.DataFrame, covariates: list[str], groups: np.ndarray,
              gene_symbols: pd.Index) -> "_Design":
        used = groups >= 0
        columns = ["intercept"]
        category_codes, continuous_values, continuous_columns = [], [], []
        total_column = mito_column = -1
        is_mito = np.zeros(len(gene_symbols), dtype=np.bool_)
        for name in covariates:
            if name in COMPUTED_COVARIATES:
                if name in obs.columns:
                    raise ValueError(
                        f"{_NAME}: covariate {name!r} is computed from the counts, but adata.obs "
                        "also has a column of that name. Rename the column to use it instead."
                    )
                if name == "log_total_counts":
                    total_column = len(columns)
                else:
                    is_mito = np.asarray(pd.Index(gene_symbols).str.upper().str.startswith("MT-"))
                    if not is_mito.any():
                        raise ValueError(
                            f"{_NAME}: 'mito_fraction' needs genes whose symbol starts with 'MT-'; "
                            "there are none."
                        )
                    mito_column = len(columns)
                columns.append(name)
                continue
            if name not in obs.columns:
                raise KeyError(
                    f"Covariate '{name}' was not found in adata.obs. "
                    f"Available columns: {list(obs.columns)}"
                )
            series = obs[name]
            if series[used].isna().any():
                raise ValueError(f"{_NAME}: covariate {name!r} has missing values in used cells.")
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                if not np.isfinite(values[used]).all():
                    raise ValueError(f"{_NAME}: covariate {name!r} has infinite values in used cells.")
                continuous_columns.append(len(columns))
                continuous_values.append(values)
                columns.append(name)
                continue
            levels = pd.Index(pd.unique(series[used].astype(str))).sort_values()
            codes = levels.get_indexer(series.astype(str))
            offset = len(columns) - 1  # level 0 is the reference and gets no column
            category_codes.append(np.where(codes > 0, codes + offset, -1))
            columns.extend(f"{name}[{level}]" for level in levels[1:])
        n_obs = len(groups)
        category_columns = (
            np.column_stack(category_codes).astype(np.int64)
            if category_codes else np.empty((n_obs, 0), dtype=np.int64)
        )
        continuous = (
            np.column_stack(continuous_values)
            if continuous_values else np.empty((n_obs, 0), dtype=np.float64)
        )
        return cls(columns, np.ascontiguousarray(category_columns), np.ascontiguousarray(continuous),
                   np.asarray(continuous_columns, dtype=np.int64), total_column, mito_column, is_mito)

    def rows(self, cells: np.ndarray, totals: np.ndarray, mito: np.ndarray) -> np.ndarray:
        """The dense design rows of ``cells``."""
        x = np.zeros((cells.size, len(self.columns)))
        x[:, 0] = 1.0
        x[:, self.continuous_columns] = self.continuous[cells]
        if self.total_column >= 0:
            x[:, self.total_column] = np.log(totals[cells])
        if self.mito_column >= 0:
            x[:, self.mito_column] = mito[cells]
        codes = self.category_columns[cells]
        hit = codes >= 0
        x[np.nonzero(hit)[0], codes[hit]] = 1.0
        return x

    def cross_products(self, groups, used, totals, mito, n_groups):
        """Per-group sums of the design rows, and the controls' ``X^T X``."""
        cells = np.flatnonzero(used)
        cell_groups = groups[cells]
        p = len(self.columns)
        # one bincount per numeric column, one for all the one-hot columns
        group_sums = np.zeros((n_groups, p))
        group_sums[:, 0] = np.bincount(cell_groups, minlength=n_groups)
        numeric = [(c, self.continuous[cells, k]) for k, c in enumerate(self.continuous_columns)]
        if self.total_column >= 0:
            numeric.append((self.total_column, np.log(totals[cells])))
        if self.mito_column >= 0:
            numeric.append((self.mito_column, mito[cells]))
        for column, values in numeric:
            group_sums[:, column] = np.bincount(cell_groups, weights=values, minlength=n_groups)
        codes = self.category_columns[cells]
        hit = codes >= 0
        flat = (cell_groups[:, None] * p + codes)[hit]
        group_sums += np.bincount(flat, minlength=n_groups * p).reshape(n_groups, p)
        # X^T X over the controls, a bounded block of dense rows at a time
        controls = cells[cell_groups == 0]
        xtx = np.zeros((p, p))
        step = max(1, (1 << 23) // p)
        for start in range(0, controls.size, step):
            x = self.rows(controls[start:start + step], totals, mito)
            xtx += x.T @ x
        return group_sums, xtx


def _estimate_shape_for_residual_effects(
    path: Path,
    *,
    perturbation_column: str | pd.Series,
    control_label: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells: int = 10,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    format_mismatch_policy: str = "auto",
    **_ignored,
) -> dict[str, float]:
    validate_format_mismatch_policy(format_mismatch_policy)
    backed = read_backed(path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        labels, _ = cell_labels(backed.obs, perturbation_column)
        labelled = labels.notna().to_numpy()
        values = labels.astype(str).to_numpy()
        control_label = resolve_control_label(values[labelled], control_label, verbose=False)
        _, targets = _group_codes(values, labelled, control_label, perturbations, min_cells, False)
    finally:
        backed.file.close()
    # X and three layers, float64
    estimate = {"output": estimate_bytes(len(targets), n_vars, 4, overhead=1.10)}
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(n_obs, n_vars, available_memory_gb=memory_limit_gb)
    # the temporary CSR copy stream_on_fast_axis may write beside the output
    scratch = scratch_copy_bytes(path, axis=0, policy=format_mismatch_policy, chunk_size=chunk_size)
    if scratch is not None:
        estimate["scratch"] = scratch
    return estimate
