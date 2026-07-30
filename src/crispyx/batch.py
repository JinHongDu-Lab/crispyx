"""Generic streaming statistics stratified by experimental batch."""

from __future__ import annotations

import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike

from ._grouping import resolve_group_reference_aliases
from .data import (
    AnnData,
    calculate_optimal_chunk_size,
    calculate_wilcoxon_chunk_size,
    ensure_gene_symbol_column,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
)


@dataclass(frozen=True)
class BatchStatistic:
    """A finalized within-batch statistic and its aggregation weight.

    Parameters
    ----------
    values
        One value per gene in the current gene chunk, so shape ``(width,)``.
        Converted to ``float64``.
    weight
        How much this batch counts when the per-batch statistics are combined.
        Either a scalar applied to every gene, or one value per gene with shape
        ``(width,)``. Must be finite and non-negative; a weight of zero drops
        the batch for the genes concerned.

    Notes
    -----
    Batches are combined per gene as ``sum(weight * values) / sum(weight)``. The
    weight is therefore what makes a cross-batch average cell-count weighted
    rather than a plain mean of batch means. Genes whose weights sum to zero are
    reported as ``NaN``.
    """

    values: ArrayLike
    weight: float | ArrayLike = 1.0


@dataclass(frozen=True)
class BatchReducer:
    """Callbacks defining a mergeable streaming statistic.

    A reducer must supply three callbacks, plus a fourth required only for
    ``mode="comparison"``:

    * ``initialize(width)`` → a fresh state object.
    * ``update(state, block)`` → ``None``, or a replacement state.
    * ``finalize(state)`` → :class:`BatchStatistic`, or a plain array.
    * ``compare(group_state, reference_state)`` → :class:`BatchStatistic`, or a
      plain array. Optional; omit it for ``mode="group"``.

    crispyx streams the matrix one gene chunk at a time and, inside a gene
    chunk, one cell chunk at a time. The callbacks therefore only ever see a
    slice of the data and must carry whatever running state the statistic needs.

    Shapes and types
    ----------------
    ``width`` : ``int``
        Number of genes in the current gene chunk. Equal to the run's
        ``chunk_size`` except for the last chunk, which may be shorter, so size
        state from ``width`` rather than assuming it is constant.
    ``block`` : ``np.ndarray``, shape ``(n_cells, width)``
        Dense array holding just the cells of one ``(group, batch)`` combination
        that fall inside the current cell chunk. Sparse input is densified
        first. ``n_cells`` is at least 1 and differs between calls; the dtype
        follows the stored matrix, so cast explicitly if the statistic needs
        ``float64``.
    ``state`` : any object
        Created fresh for each gene chunk and never shared across gene chunks.
        Within one gene chunk, one state is held per ``(group, batch)`` and is
        updated by every cell chunk containing that combination. ``update`` may
        mutate it in place and return ``None``, or return a replacement.
    ``finalize`` / ``compare`` return value
        One value per gene, so shape ``(width,)``. Returning a bare array is
        equivalent to ``BatchStatistic(values, weight=1.0)``.

    Examples
    --------
    A within-batch mean, weighted across batches by cell count::

        import numpy as np
        import crispyx as cx

        def initialize(width):
            # One accumulator per gene in this chunk, plus a scalar cell count.
            return {"total": np.zeros(width, dtype=np.float64), "n": 0}

        def update(state, block):
            # block is (n_cells, width); summing over cells leaves (width,).
            state["total"] += block.sum(axis=0)
            state["n"] += block.shape[0]
            # Returning None keeps the state that was mutated above.

        def finalize(state):
            # values must be (width,); the weight makes the cross-batch
            # average cell-count weighted.
            return cx.BatchStatistic(state["total"] / state["n"], weight=state["n"])

        reducer = cx.BatchReducer(initialize, update, finalize)

    Adding ``compare`` enables ``mode="comparison"``. It receives the two states
    accumulated independently for the group and for the reference within the
    same batch::

        def compare(group_state, reference_state):
            difference = (
                group_state["total"] / group_state["n"]
                - reference_state["total"] / reference_state["n"]
            )
            n_group, n_reference = group_state["n"], reference_state["n"]
            weight = n_group * n_reference / (n_group + n_reference)
            return cx.BatchStatistic(difference, weight=weight)

        reducer = cx.BatchReducer(initialize, update, finalize, compare)

    Notes
    -----
    The statistic must be expressible through state that merges across cell
    chunks. One that needs all of a group's cells at once, such as an exact
    median, cannot be written this way.
    """

    initialize: Callable[[int], Any]
    update: Callable[[Any, np.ndarray], Any | None]
    finalize: Callable[[Any], BatchStatistic | ArrayLike]
    compare: Callable[[Any, Any], BatchStatistic | ArrayLike] | None = None


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _safe_statistic_name(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("statistic_name must not be empty")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not safe:
        raise ValueError("statistic_name must contain at least one letter or number")
    return safe


def _as_dense(block: Any) -> np.ndarray:
    if sp.issparse(block):
        block = block.toarray()
    return np.asarray(block)


def _normalise_statistic(
    result: BatchStatistic | ArrayLike,
    width: int,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(result, BatchStatistic):
        values, weight = result.values, result.weight
    else:
        values, weight = result, 1.0

    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1 or values_arr.shape[0] != width:
        raise ValueError(
            f"Reducer output for {context} must have shape ({width},); "
            f"received {values_arr.shape}."
        )
    weight_arr = np.asarray(weight, dtype=np.float64)
    if weight_arr.ndim == 0:
        weight_arr = np.full(width, float(weight_arr), dtype=np.float64)
    elif weight_arr.ndim != 1 or weight_arr.shape[0] != width:
        raise ValueError(
            f"Reducer weight for {context} must be scalar or have shape ({width},); "
            f"received {weight_arr.shape}."
        )
    if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0):
        raise ValueError(f"Reducer weights for {context} must be finite and non-negative.")
    return values_arr, weight_arr


def _metadata_matches(adata: ad.AnnData, expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        actual = adata.uns.get(key)
        if isinstance(expected_value, list):
            if actual is None:
                return False
            actual_list = np.asarray(actual).tolist()
            if not isinstance(actual_list, list):
                return False
            if [str(x) for x in actual_list] != expected_value:
                return False
        elif expected_value is None:
            if actual not in (None, "", b""):
                return False
        elif str(actual) != str(expected_value):
            return False
    return True


def batch_process(
    data: str | Path | AnnData | ad.AnnData,
    reducer: BatchReducer,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str,
    mode: Literal["group", "comparison"] = "group",
    statistic_name: str,
    chunk_size: int | None = None,
    cell_chunk_size: int | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
    memory_limit_gb: float | None = None,
    force: bool = False,
) -> AnnData:
    """Compute a generic gene-wise statistic within experimental batches.

    Parameters follow the differential-expression API: ``groupby`` aliases
    ``perturbation_column`` and ``reference`` aliases ``control_label``.
    ``chunk_size`` controls genes per chunk, while ``cell_chunk_size`` controls
    rows supplied to the reducer at once.

    In ``"group"`` mode, ``finalize`` is called for every observed
    ``(group, batch)`` state. In ``"comparison"`` mode, ``compare`` is called
    for group/reference states from shared batches. Finalized batch statistics
    are combined as ``sum(weight * values) / sum(weight)``.

    Parameters
    ----------
    data
        Path to an existing h5ad file, :class:`crispyx.AnnData`, or a backed
        :class:`anndata.AnnData`. A normal in-memory ``anndata.AnnData`` is not
        accepted because the operation is designed to stream from disk. ``X``
        must be a numeric cells-by-genes matrix, and ``obs`` must contain the
        grouping and batch columns.
    reducer
        A :class:`BatchReducer` whose state can be merged across cell chunks.
        ``initialize(width)`` receives the current gene-chunk width;
        ``update(state, block)`` receives a dense cells-by-``width`` array; and
        ``finalize(state)`` must return one value per gene. Comparison mode also
        requires ``compare(group_state, reference_state)``.
    perturbation_column
        Column in ``adata.obs`` containing group labels.
    groupby
        Scanpy-compatible alias for ``perturbation_column``. The two arguments
        are mutually exclusive.
    control_label
        Reference group used in comparison mode. If omitted, crispyx infers a
        common control label.
    reference
        Scanpy-compatible alias for ``control_label``.
    gene_name_column
        Column in ``adata.var`` containing gene symbols. Uses ``var_names``
        when omitted.
    perturbations
        Ordered subset of groups to return. All applicable groups are used
        when omitted.
    batch_column
        Column in ``adata.obs`` containing experimental batch/stratum labels.
    mode
        ``"group"`` finalizes each group independently. ``"comparison"``
        compares each group with the reference within shared batches.
    statistic_name
        Stable identifier used in output naming and cache metadata.
    chunk_size
        Genes processed per chunk. Automatically selected when omitted.
    cell_chunk_size
        Cells supplied to each reducer update. Automatically selected when
        omitted.
    data_name
        Optional input stem override used to construct the output filename.
    output_path
        Exact output h5ad path. Takes precedence over naming arguments.
    output_dir
        Deprecated output directory override; use ``output_path`` instead.
    verbose
        Verbosity level. Level 1 prints run/reload and save summaries.
    memory_limit_gb
        Soft memory budget used for automatic chunk sizing.
    force
        Recompute even when a matching output already exists. Edits to the input
        file are detected automatically through its path and modification time,
        so a regenerated source invalidates a cached result on its own. Use
        ``force`` when a reducer's implementation changes without changing
        ``statistic_name``, since the reducer itself cannot be fingerprinted.

    Returns
    -------
    AnnData
        On-disk result with groups in observations, genes in variables, the
        corrected statistic in ``X``, and accumulated weights in
        ``layers['weight_sum']``.

    Notes
    -----
    The reducer sees the values stored in ``X`` exactly as provided. Normalize
    or log-transform the input beforehand when the statistic requires it. A
    statistic must be expressible through mergeable state; callbacks that need
    all cells simultaneously are not suitable for this streaming interface.

    Examples
    --------
    A streaming mean keeps only a sum and count. Returning the count as the
    weight produces a cell-count-weighted mean across biological batches::

        import numpy as np
        import crispyx as cx

        def initialize_mean(width):
            return {"sum": np.zeros(width), "count": 0}

        def update_mean(state, block):
            state["sum"] += block.sum(axis=0)
            state["count"] += block.shape[0]

        def finalize_mean(state):
            values = state["sum"] / state["count"]
            return cx.BatchStatistic(values, weight=state["count"])

        mean_reducer = cx.BatchReducer(
            initialize=initialize_mean,
            update=update_mean,
            finalize=finalize_mean,
        )
        result = cx.tl.batch_process(
            "screen.h5ad",
            mean_reducer,
            groupby="perturbation",
            batch_column="batch",
            statistic_name="mean",
        )

    Comparison mode adds a callback receiving the independently accumulated
    group and reference states. This example computes within-batch mean
    differences with the harmonic-count weighting used by crispyx pseudobulk::

        def compare_means(group_state, reference_state):
            n_group = group_state["count"]
            n_reference = reference_state["count"]
            difference = (
                group_state["sum"] / n_group
                - reference_state["sum"] / n_reference
            )
            weight = n_group * n_reference / (n_group + n_reference)
            return cx.BatchStatistic(difference, weight=weight)

        comparison_reducer = cx.BatchReducer(
            initialize=initialize_mean,
            update=update_mean,
            finalize=finalize_mean,
            compare=compare_means,
        )
        result = cx.tl.batch_process(
            "screen.h5ad",
            comparison_reducer,
            groupby="perturbation",
            reference="control",
            batch_column="batch",
            mode="comparison",
            statistic_name="mean_difference",
        )
    """
    if not isinstance(reducer, BatchReducer):
        raise TypeError("reducer must be a BatchReducer instance")
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="batch_process",
    )
    if mode not in ("group", "comparison"):
        raise ValueError("mode must be 'group' or 'comparison'")
    if mode == "group" and control_label is not None:
        raise TypeError(
            "batch_process() reference/control arguments are only valid when mode='comparison'."
        )
    if mode == "comparison" and reducer.compare is None:
        raise TypeError("reducer.compare is required when mode='comparison'.")

    statistic_name = _safe_statistic_name(statistic_name)
    path = resolve_data_path(data)
    suffix = f"batch_{statistic_name}_{mode}"
    resolved_output = resolve_output_path(
        path,
        suffix=suffix,
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
    )

    backed = read_backed(path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        if batch_column not in backed.obs.columns:
            raise KeyError(
                f"Batch column '{batch_column}' was not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column).astype(str)
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        observed = _unique_strings(labels)

        if mode == "comparison":
            control_label = resolve_control_label(labels, control_label)
            if control_label not in observed:
                raise ValueError(f"Reference group '{control_label}' contains no cells")
        if perturbations is None:
            groups = [g for g in observed if mode == "group" or g != control_label]
        else:
            groups = _unique_strings(perturbations)
            if mode == "comparison":
                groups = [g for g in groups if g != control_label]
        missing_groups = [group for group in groups if group not in observed]
        if missing_groups:
            raise ValueError(
                f"Perturbation(s) {missing_groups[:3]}"
                f"{'...' if len(missing_groups) > 3 else ''} contain no cells"
            )

        raw_batch = np.asarray(backed.obs[batch_column].to_numpy())
        batch_codes, batch_uniques = pd.factorize(raw_batch, sort=True)
        batch_codes = batch_codes.astype(np.int64)
        batch_ids = [str(x) for x in batch_uniques]
        if not batch_ids:
            raise ValueError(f"Batch column '{batch_column}' contains no usable batches.")
        n_missing_batch = int(np.sum(batch_codes < 0))
        if n_missing_batch:
            warnings.warn(
                f"{n_missing_batch} cells have a missing '{batch_column}' value and are "
                "excluded from batch_process().",
                stacklevel=2,
            )

        group_lookup = {group: idx for idx, group in enumerate(groups)}
        group_codes = np.asarray([group_lookup.get(label, -1) for label in labels], dtype=np.int64)
        reference_code = -2
        if mode == "comparison":
            group_codes[labels == control_label] = reference_code

        n_groups = len(groups)
        n_genes = backed.n_vars
        n_batches = len(batch_ids)
        group_batch_counts = np.zeros((n_groups, n_batches), dtype=np.int64)
        valid_group_cells = (group_codes >= 0) & (batch_codes >= 0)
        if valid_group_cells.any():
            np.add.at(
                group_batch_counts,
                (group_codes[valid_group_cells], batch_codes[valid_group_cells]),
                1,
            )
        reference_batch_counts = np.zeros(n_batches, dtype=np.int64)
        if mode == "comparison":
            ref_cells = (group_codes == reference_code) & (batch_codes >= 0)
            reference_batch_counts += np.bincount(
                batch_codes[ref_cells], minlength=n_batches
            ).astype(np.int64)

        if chunk_size is None:
            chunk_size = calculate_wilcoxon_chunk_size(
                backed.n_obs,
                n_genes,
                available_memory_gb=memory_limit_gb,
            )
        if cell_chunk_size is None:
            cell_chunk_size = calculate_optimal_chunk_size(
                backed.n_obs,
                min(n_genes, chunk_size),
                available_memory_gb=memory_limit_gb,
            )
        if chunk_size <= 0 or cell_chunk_size <= 0:
            raise ValueError("chunk_size and cell_chunk_size must be positive")

        expected_metadata = {
            "statistic_name": statistic_name,
            "mode": mode,
            "perturbation_column": perturbation_column,
            "control_label": control_label if mode == "comparison" else None,
            "batch_column": batch_column,
            "groups": groups,
            "batch_ids": batch_ids,
            # Identify the input itself, so that regenerating the source in place
            # invalidates the cache instead of silently returning stale values.
            "source_path": str(path.resolve()),
            "source_mtime_ns": int(path.stat().st_mtime_ns),
        }
        if resolved_output.exists() and not force:
            existing = ad.read_h5ad(resolved_output, backed="r")
            try:
                matches = _metadata_matches(existing, expected_metadata)
            finally:
                existing.file.close()
            if matches:
                if int(verbose) >= 1:
                    print(f"[cx] Loading existing result: {resolved_output}")
                    print("[cx] Pass force=True to rerun the analysis.")
                return AnnData(resolved_output)

        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        if int(verbose) >= 1:
            print(
                f"[cx] tl.batch_process: {n_groups} groups × {n_genes} genes, "
                f"stratified by '{batch_column}'"
            )

        with tempfile.TemporaryDirectory(prefix="cx_batch_") as tmpdir:
            values_path = Path(tmpdir) / "values.dat"
            weights_path = Path(tmpdir) / "weights.dat"
            values_mm = np.memmap(
                values_path, dtype=np.float64, mode="w+", shape=(max(n_groups, 1), n_genes)
            )
            weights_mm = np.memmap(
                weights_path, dtype=np.float64, mode="w+", shape=(max(n_groups, 1), n_genes)
            )
            values_mm[:] = np.nan
            weights_mm.fill(0)
            batches_used = np.zeros((n_groups, n_batches), dtype=bool)

            for gene_start in range(0, n_genes, chunk_size):
                gene_end = min(gene_start + chunk_size, n_genes)
                width = gene_end - gene_start
                states: dict[tuple[int, int], Any] = {}

                for cell_start in range(0, backed.n_obs, cell_chunk_size):
                    cell_end = min(cell_start + cell_chunk_size, backed.n_obs)
                    local_groups = group_codes[cell_start:cell_end]
                    local_batches = batch_codes[cell_start:cell_end]
                    usable = (local_batches >= 0) & (
                        (local_groups >= 0)
                        | ((mode == "comparison") & (local_groups == reference_code))
                    )
                    if not usable.any():
                        continue
                    block = _as_dense(backed.X[cell_start:cell_end, gene_start:gene_end])
                    pairs = np.unique(
                        np.column_stack((local_groups[usable], local_batches[usable])), axis=0
                    )
                    for group_code, batch_code in pairs:
                        key = (int(group_code), int(batch_code))
                        mask = (local_groups == group_code) & (local_batches == batch_code)
                        group_name = (
                            str(control_label)
                            if group_code == reference_code
                            else groups[int(group_code)]
                        )
                        context = f"group '{group_name}', batch '{batch_ids[int(batch_code)]}'"
                        try:
                            if key in states:
                                state = states[key]
                            else:
                                state = reducer.initialize(width)
                            replacement = reducer.update(state, block[mask])
                        except Exception as exc:
                            raise RuntimeError(f"Reducer failed for {context}") from exc
                        states[key] = state if replacement is None else replacement

                for group_index, group in enumerate(groups):
                    numerator = np.zeros(width, dtype=np.float64)
                    denominator = np.zeros(width, dtype=np.float64)
                    for batch_index, batch_id in enumerate(batch_ids):
                        if group_batch_counts[group_index, batch_index] == 0:
                            continue
                        group_key = (group_index, batch_index)
                        if group_key not in states:
                            continue
                        group_state = states[group_key]
                        context = f"group '{group}', batch '{batch_id}'"
                        try:
                            if mode == "group":
                                finalized = reducer.finalize(group_state)
                            else:
                                if reference_batch_counts[batch_index] == 0:
                                    continue
                                reference_key = (reference_code, batch_index)
                                if reference_key not in states:
                                    continue
                                reference_state = states[reference_key]
                                finalized = reducer.compare(group_state, reference_state)  # type: ignore[misc]
                            batch_values, batch_weights = _normalise_statistic(
                                finalized, width, context=context
                            )
                        except Exception as exc:
                            if isinstance(exc, (ValueError, TypeError)) and str(exc).startswith("Reducer"):
                                raise
                            raise RuntimeError(f"Reducer failed for {context}") from exc
                        positive = batch_weights > 0
                        if positive.any():
                            numerator[positive] += (
                                batch_values[positive] * batch_weights[positive]
                            )
                            denominator[positive] += batch_weights[positive]
                            batches_used[group_index, batch_index] = True
                    values_mm[group_index, gene_start:gene_end] = np.divide(
                        numerator,
                        denominator,
                        out=np.full(width, np.nan, dtype=np.float64),
                        where=denominator > 0,
                    )
                    weights_mm[group_index, gene_start:gene_end] = denominator

            untestable = np.all(np.asarray(weights_mm[:n_groups]) <= 0, axis=1)
            n_batches_used = batches_used.sum(axis=1).astype(np.int64)
            if untestable.any():
                examples = [groups[i] for i in np.flatnonzero(untestable)[:5]]
                warnings.warn(
                    f"{int(untestable.sum())} group(s) have no usable batch statistics; "
                    f"results are NaN. Examples: {examples}",
                    stacklevel=2,
                )

            obs_index = pd.Index(groups, name="perturbation")
            obs = pd.DataFrame(
                {
                    perturbation_column: groups,
                    "n_batches_used": n_batches_used,
                },
                index=obs_index,
            )
            var = pd.DataFrame(index=pd.Index(gene_symbols, name=backed.var_names.name))
            result = ad.AnnData(np.asarray(values_mm[:n_groups]), obs=obs, var=var)
            result.layers["weight_sum"] = np.asarray(weights_mm[:n_groups])
            result.uns.update(expected_metadata)
            result.uns["stratified"] = True
            result.uns["stratified_n_batches"] = int(n_batches)
            result.uns["stratified_n_untestable_perturbations"] = int(untestable.sum())
            result.uns["chunk_size"] = int(chunk_size)
            result.uns["cell_chunk_size"] = int(cell_chunk_size)
            result.write(resolved_output)
            values_mm._mmap.close()  # type: ignore[attr-defined]
            weights_mm._mmap.close()  # type: ignore[attr-defined]
    finally:
        backed.file.close()

    if int(verbose) >= 1:
        print(f"[cx] tl.batch_process: Saving → {resolved_output}")
    return AnnData(resolved_output)


__all__ = ["BatchReducer", "BatchStatistic", "batch_process"]
