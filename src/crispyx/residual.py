"""Covariate-adjusted perturbation effects: control-fitted linear model residuals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Literal, Sequence, get_args

import anndata as ad
import h5py
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.special import ndtr

from . import _messages
from ._checkpoint import _create_progress_context
from ._disk import estimate_bytes, warn_if_disk_space_low
from ._kernels import _residual_effects_rows
from ._memory import _resolve_memory_limit_bytes, _resolve_n_jobs
from ._provenance import KEY as _PROVENANCE_KEY, call_fingerprint, reusable, stamp
from ._statistics import _adjust_pvalue_matrix
from .data import (
    AnnData,
    _decoded_block_reader,
    _detect_backed_sparse_format,
    _replace_on_success,
    calculate_optimal_chunk_size,
    ensure_gene_symbol_column,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    stream_on_fast_axis,
)

ResidualNull = Literal["model", "robust"]

#: Covariates computed from each cell's counts during the pass, not read from obs.
COMPUTED_COVARIATES = ("log_total_counts", "mito_fraction")

#: Arguments that decide how a call runs but not what it writes.
_OPERATIONAL_ARGS = frozenset({
    "data", "chunk_size", "n_jobs", "memory_limit_gb", "data_name", "output_path", "verbose", "force",
})

_NAME = "residual_effects"


def residual_effects(
    data: str | Path | AnnData | ad.AnnData,
    perturbation_column: str | pd.Series,
    control_label: str | None = None,
    *,
    covariates: Sequence[str] = COMPUTED_COVARIATES,
    null: ResidualNull = "model",
    min_cells: int = 10,
    min_control_mean: float = 0.1,
    perturbations: Iterable[str] | None = None,
    gene_name_column: str | None = None,
    chunk_size: int | None = None,
    n_jobs: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    verbose: int | bool = True,
    force: bool = False,
) -> AnnData:
    """Perturbation effects adjusted for covariates by a model fitted on controls.

    The estimator of Feng et al. (2025, Cell Genomics): every gene's
    ``log1p(CP10k)`` expression is regressed by ordinary least squares on the
    covariates over the control cells only; a perturbation's effect is the mean,
    over its cells, of observed minus predicted expression. Cell-line, donor,
    lane, depth or mitochondrial differences between a perturbation's cells and
    the controls are thereby removed rather than read as an effect.

    Everything is computed in one streaming pass over the rows of the raw count
    matrix, which collects per-perturbation sums of ``log1p(CP10k)``, the
    controls' cross-products with the covariates and their sums of squares. The
    mean residual of perturbation ``t`` is ``mean_t(y) - mean_t(x) @ B``, so no
    per-cell residual is ever formed and memory is one perturbations x genes
    array.

    Parameters
    ----------
    data
        Path to an h5ad file of **raw counts**, :class:`crispyx.AnnData`, or an
        ``anndata.AnnData`` opened with ``backed='r'``. The function normalises
        (``log1p`` of counts per 10,000) itself and refuses non-integer input.
    perturbation_column
        Column in ``adata.obs`` naming each cell's perturbation, or a
        ``pd.Series`` of labels indexed by ``obs_names``. Cells whose label is
        missing (NaN) are not used, which restricts a file to a subset of cells
        without writing a copy; row chunks without a used cell are not read.
    control_label
        Label of the control cells the model is fitted on. Inferred from common
        patterns when omitted.
    covariates
        Terms of the model, besides the intercept. Each is an ``obs`` column --
        one-hot coded (first level dropped) when categorical, string or boolean,
        entered as is when numeric -- or one of the two quantities computed from
        the counts during the pass:

        * ``"log_total_counts"``: natural log of the cell's total counts;
        * ``"mito_fraction"``: share of the cell's counts in genes whose symbol
          starts with ``MT-``.

        An ``obs`` column with one of these names raises instead of being used
        silently. Interactions (e.g. lane x donor) and other per-cell scores
        (cell cycle) are ``obs`` columns computed beforehand. Every level of a
        categorical covariate must occur among the controls. ``[]`` gives the
        unadjusted mean difference from the controls.
    null
        Spread the z-score divides by:

        * ``"model"`` (the paper's test): ``sigma_g / sqrt(n_t)``, with
          ``sigma_g`` the residual standard deviation of the control fit.
          Treats cells as independent; cells sharing a guide or donor are not,
          which makes it anti-conservative.
        * ``"robust"``: ``s_g / sqrt(n_t)``, with ``s_g`` 1.4826 x the median
          absolute deviation, over perturbations, of ``effect * sqrt(n_t)``. It
          assumes most perturbations leave gene ``g`` unchanged -- true for
          genome-wide screens, false for a focused panel where every target is
          active.

        Both spreads are written to ``var``, so either z can be recomputed.
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
        Memory budget. The perturbations x genes accumulator must fit in it
        three times over (accumulator, effect, robust spread), else the call
        refuses before reading the matrix. Detected from the system when omitted.
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
        (Benjamini-Hochberg within each perturbation). ``obs`` holds ``n_cells``
        and ``n_genes_tested``; ``var`` holds ``control_mean``, ``residual_sd``,
        ``robust_sd`` and ``tested``; ``uns`` records the null, the covariates
        and the design's column names.
    """
    call_args = dict(locals())  # for the result fingerprint, before any other local
    if null not in get_args(ResidualNull):
        raise ValueError(f"{_NAME}: null must be 'model' or 'robust'; received {null!r}.")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")
    if min_control_mean < 0:
        raise ValueError("min_control_mean must be non-negative")
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
        labels, label_name = _cell_labels(backed, perturbation_column)
        labelled = labels.notna().to_numpy()
        values = labels.astype(str).to_numpy()
        control_label = resolve_control_label(values[labelled], control_label, verbose=verbose)
        groups, targets = _group_codes(
            values, labelled, control_label, perturbations, min_cells, verbose,
        )
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column).astype(str)
        design = _Design.build(backed.obs, covariates, groups, gene_symbols)
        accumulator_bytes = (len(targets) + 1) * n_vars * 8
        budget = _resolve_memory_limit_bytes(memory_limit_gb)
        if 3 * accumulator_bytes > budget:
            raise MemoryError(
                f"{_NAME}: {len(targets):,} perturbations x {n_vars:,} genes need "
                f"{3 * accumulator_bytes / 1e9:.1f} GB (three float64 copies); the budget is "
                f"{budget / 1e9:.1f} GB. Pass fewer perturbations or raise memory_limit_gb."
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
    workers = max(1, min(_resolve_n_jobs(n_jobs), numba.config.NUMBA_NUM_THREADS))
    # Row 0 (the controls) stays zero: their sum is control_xty's intercept column.
    sums = np.zeros((len(targets) + 1, n_vars))
    control_xty = np.zeros((workers, n_vars, len(design.columns)))
    control_sumsq = np.zeros((workers, n_vars))
    totals = np.zeros(n_obs)
    mito = np.zeros(n_obs)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with stream_on_fast_axis(
        path, axis=0, policy="auto", fn_name=_NAME, scratch_dir=resolved_output.parent,
        chunk_size=chunk_size, memory_limit_gb=memory_limit_gb, verbose=verbose,
    ) as stream_path:
        streamed = read_backed(stream_path)
        try:
            matrix = streamed.X
            read_block = _decoded_block_reader(
                matrix, _detect_backed_sparse_format(matrix), 0, n_obs, n_vars,
            ) or (lambda start, end: matrix[start:end])
            starts = [
                start for start in range(0, n_obs, chunk_size)
                if (groups[start:start + chunk_size] >= 0).any()
            ]
            checked = False

            def read_csr(start):
                block = read_block(start, min(start + chunk_size, n_obs))
                return block.tocsr() if sparse.issparse(block) else sparse.csr_matrix(np.asarray(block))

            with _create_progress_context(len(starts), _NAME, verbose, unit="chunk") as pbar:
                for start, block in _read_ahead(read_csr, starts):
                    end = min(start + chunk_size, n_obs)
                    if not checked:
                        _check_counts(block.data)
                        checked = True
                    chunk_groups = groups[start:end]
                    target_rows, target_bounds, control_rows, control_bounds = _segments(
                        chunk_groups, np.diff(block.indptr), workers,
                    )
                    _residual_effects_rows(
                        block.indptr, block.indices, block.data, chunk_groups,
                        target_rows, target_bounds, control_rows, control_bounds,
                        design.category_columns[start:end], design.continuous[start:end],
                        design.continuous_columns, design.total_column, design.mito_column,
                        design.is_mito, sums, control_xty, control_sumsq,
                        totals[start:end], mito[start:end],
                    )
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
    group_design_sums, control_xtx = design.cross_products(groups, used, totals, mito, len(targets) + 1)
    n_controls = int(n_cells[0])
    coefficients, _, rank, _ = np.linalg.lstsq(control_xtx, control_xty.T, rcond=None)
    if rank < len(design.columns):
        _messages.vprint(
            verbose, _NAME,
            f"the design has rank {rank} < {len(design.columns)} columns (collinear covariates); "
            "fitted values and effects are still unique.",
        )
    df = n_controls - rank
    if df <= 0:
        raise ValueError(
            f"{_NAME}: {n_controls} control cells cannot fit {rank} independent design columns."
        )
    rss = control_sumsq - np.einsum("pg,gp->g", coefficients, control_xty)
    residual_sd = np.sqrt(np.maximum(rss, 0.0) / df)
    control_mean = control_xty[:, 0] / n_cells[0]

    rows = 1 + np.flatnonzero(keep)
    n_target = n_cells[rows]
    effect = sums[rows]
    del sums
    effect /= n_target[:, None]
    mean_design = group_design_sums[rows] / n_target[:, None]
    effect -= mean_design @ coefficients
    scaled = effect * np.sqrt(n_target)[:, None]
    scaled -= np.median(scaled, axis=0)
    robust_sd = 1.4826 * np.median(np.abs(scaled, out=scaled), axis=0)
    del scaled
    spread = residual_sd if null == "model" else robust_sd
    tested = (control_mean > min_control_mean) & (spread > 0)
    kept_targets = [targets[i] for i in np.flatnonzero(keep)]

    obs_index = pd.Index(kept_targets, name="perturbation").astype(str)
    obs = pd.DataFrame(
        {
            label_name: obs_index.to_list(),
            "n_cells": n_target.astype(np.int64),
            "n_genes_tested": np.full(len(kept_targets), int(tested.sum()), dtype=np.int64),
        },
        index=obs_index,
    )
    var = pd.DataFrame(
        {
            "control_mean": control_mean,
            "residual_sd": residual_sd,
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
                z = effect[start:stop] * np.sqrt(n_target[start:stop])[:, None] / np.where(tested, spread, np.nan)
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
    perturbation rows ordered by group and cut at group boundaries, balanced
    by non-zeros, so no two segments share a group; the control rows in
    equal shares (each segment has its own control accumulators)."""
    rows = np.flatnonzero(groups > 0)
    rows = rows[np.argsort(groups[rows], kind="stable")]
    cumulative = np.cumsum(row_nnz[rows])
    cuts = np.full(workers - 1, rows.size, dtype=np.int64)
    group_starts = np.flatnonzero(np.diff(groups[rows])) + 1
    if group_starts.size:
        # balance by non-zeros, then move each cut forward to where a group starts
        balanced = np.searchsorted(cumulative, cumulative[-1] * np.arange(1, workers) / workers)
        after = np.searchsorted(group_starts, balanced)
        found = after < group_starts.size
        cuts[found] = group_starts[after[found]]
    target_bounds = np.concatenate([[0], cuts, [rows.size]]).astype(np.int64)
    controls = np.flatnonzero(groups == 0)
    control_bounds = np.linspace(0, controls.size, workers + 1).astype(np.int64)
    return rows.astype(np.int64), target_bounds, controls.astype(np.int64), control_bounds


def _read_ahead(read, starts):
    """Yield ``(start, read(start))`` for each start, reading the next chunk on
    a background thread while the caller processes the current one. The pass
    is bound by reading, and the kernel releases the GIL, so the two overlap."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cx_read_ahead") as pool:
        pending = pool.submit(read, starts[0]) if starts else None
        for i, start in enumerate(starts):
            block = pending.result()
            pending = pool.submit(read, starts[i + 1]) if i + 1 < len(starts) else None
            yield start, block


def _dense_dataset(group: h5py.Group, name: str, data=None, shape=None) -> h5py.Dataset:
    """A float64 dataset tagged as an anndata dense array."""
    if data is not None:
        ds = group.create_dataset(name, data=np.asarray(data, dtype=np.float64))
    else:
        ds = group.create_dataset(name, shape=shape, dtype=np.float64)
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"
    return ds


def _cell_labels(backed: ad.AnnData, perturbation_column: str | pd.Series) -> tuple[pd.Series, str]:
    """Each cell's label in ``obs_names`` order (NaN = not used), and its name."""
    names = pd.Index(backed.obs_names).astype(str)
    if isinstance(perturbation_column, pd.Series):
        labels = perturbation_column.set_axis(perturbation_column.index.astype(str))
        if not labels.index.is_unique:
            raise ValueError("The labels Series must be indexed by cell, each cell once.")
        absent = names[~names.isin(labels.index)]
        if len(absent):
            raise ValueError(
                f"The labels Series has no entry for {len(absent)} of {len(names)} cells "
                f"(e.g. {absent[:3].tolist()}); index it by obs_names, with NaN for cells to leave out."
            )
        return labels.reindex(names), str(perturbation_column.name or "perturbation")
    if perturbation_column not in backed.obs.columns:
        raise KeyError(
            f"Column '{perturbation_column}' was not found in adata.obs. "
            f"Available columns: {list(backed.obs.columns)}"
        )
    return backed.obs[perturbation_column].set_axis(names), str(perturbation_column)


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


def _check_counts(values: np.ndarray) -> None:
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.rint(values)):
        raise ValueError(
            f"{_NAME} normalises raw counts itself, but X holds non-integer or negative "
            "values (already normalised?). Pass a file, or point X at a layer, of raw counts."
        )


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
        controls = groups == 0
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
                continuous_columns.append(len(columns))
                continuous_values.append(series.to_numpy(dtype=np.float64))
                columns.append(name)
                continue
            levels = pd.Index(pd.unique(series[used].astype(str))).sort_values()
            codes = levels.get_indexer(series.astype(str))
            without_controls = [
                level for i, level in enumerate(levels) if not (codes[controls] == i).any()
            ]
            if without_controls:
                raise ValueError(
                    f"{_NAME}: covariate {name!r} has level(s) with no control cells, whose "
                    f"effect the control fit cannot estimate: {without_controls[:10]}"
                )
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

    def values(self, column: int, cells: np.ndarray, totals: np.ndarray, mito: np.ndarray) -> np.ndarray:
        """Design column ``column`` for ``cells``."""
        if column == 0:
            return np.ones(cells.size)
        if column == self.total_column:
            return np.log(totals[cells])
        if column == self.mito_column:
            return mito[cells]
        where = np.flatnonzero(self.continuous_columns == column)
        if where.size:
            return self.continuous[cells, where[0]]
        return (self.category_columns[cells] == column).any(axis=1).astype(np.float64)

    def cross_products(self, groups, used, totals, mito, n_groups):
        """Per-group sums of the design rows, and the controls' ``X^T X``."""
        cells = np.flatnonzero(used)
        cell_groups = groups[cells]
        p = len(self.columns)
        group_sums = np.empty((n_groups, p))
        for column in range(p):
            group_sums[:, column] = np.bincount(
                cell_groups, weights=self.values(column, cells, totals, mito), minlength=n_groups,
            )
        controls = cells[cell_groups == 0]
        x = np.column_stack([self.values(column, controls, totals, mito) for column in range(p)])
        return group_sums, x.T @ x


def _estimate_shape_for_residual_effects(
    path: Path,
    *,
    perturbation_column: str | pd.Series,
    control_label: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells: int = 10,
    **_ignored,
) -> dict[str, float]:
    backed = read_backed(path)
    try:
        labels, _ = _cell_labels(backed, perturbation_column)
        labelled = labels.notna().to_numpy()
        values = labels.astype(str).to_numpy()
        control_label = resolve_control_label(values[labelled], control_label, verbose=False)
        _, targets = _group_codes(values, labelled, control_label, perturbations, min_cells, False)
        # X and three layers, float64
        return {"output": estimate_bytes(len(targets), backed.n_vars, 4, overhead=1.10)}
    finally:
        backed.file.close()
