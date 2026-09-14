"""Generic streaming statistics stratified by experimental batch."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike

from . import _messages
from ._checkpoint import (
    _create_progress_context,
    _find_last_completed_gene_chunk,
    _get_checkpoint_interval,
    _pack_bool_matrix,
    _read_checkpoint,
    _unpack_bool_matrix,
    _write_checkpoint_atomic,
)
from ._disk import estimate_bytes, warn_if_disk_space_low
from ._grouping import resolve_group_reference_aliases
from ._memory import _resolve_memory_limit_bytes
from .data import (
    AnnData,
    _update_h5ad_dataframe,
    calculate_optimal_chunk_size,
    calculate_wilcoxon_chunk_size,
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
        first. ``n_cells`` is at least 1 and differs between calls.

        How the cells of one combination are split across calls is an
        implementation detail -- it changes with ``cell_chunk_size``, with how
        the cells happen to be ordered in the file, and between crispyx
        releases -- so a reducer must not depend on it. The dtype likewise
        follows the stored matrix, which for a typical counts or normalized
        file is ``float32``: **accumulate in float64**, or the statistic's
        accuracy will vary with the block sizes crispyx happens to hand you
        (summing a few thousand ``float32`` rows at once loses roughly an
        order of magnitude of precision against summing a few hundred). One
        ``np.asarray(block, dtype=np.float64)`` at the top of ``update`` is
        enough; the state arrays below are float64 already.
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
            # Accumulate in float64: block follows the file's dtype, and how
            # many cells arrive per call is not part of the contract.
            block = np.asarray(block, dtype=np.float64)
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

    Multiple named channels from one pass
    --------------------------------------
    A statistic and something derived from the same state — e.g. a mean and
    its standard error, so a caller can form ``t = mean / se`` — can be
    packed into a single streaming pass instead of two. Set ``channels`` to
    the tuple of names ``finalize``/``compare`` will return, and have them
    return ``{name: BatchStatistic | ArrayLike, ...}`` with exactly those
    keys instead of one bare value::

        def compare_mean_se(group_state, reference_state):
            diff = group_state["mean"] - reference_state["mean"]
            se = np.sqrt(group_state["var"] / group_state["n"]
                         + reference_state["var"] / reference_state["n"])
            n_group, n_reference = group_state["n"], reference_state["n"]
            weight = n_group * n_reference / (n_group + n_reference)
            return {
                "mean_diff": cx.BatchStatistic(diff, weight=weight),
                "se": cx.BatchStatistic(se, weight=weight),
            }

        reducer = cx.BatchReducer(
            initialize, update, finalize, compare_mean_se,
            channels=("mean_diff", "se"),
        )

    Each channel is combined across batches independently (with its own
    weight) and written to ``result.layers[name]``; the first channel
    (``"mean_diff"`` above) is also copied into ``result.X``. Leave
    ``channels`` as ``None`` (the default) for the single-value form above.

    Notes
    -----
    The statistic must be expressible through state that merges across cell
    chunks. One that needs all of a group's cells at once, such as an exact
    median, cannot be written this way.
    """

    initialize: Callable[[int], Any]
    update: Callable[[Any, np.ndarray], Any | None]
    finalize: Callable[[Any], BatchStatistic | ArrayLike | dict[str, BatchStatistic | ArrayLike]]
    compare: (
        Callable[[Any, Any], BatchStatistic | ArrayLike | dict[str, BatchStatistic | ArrayLike]] | None
    ) = None
    channels: tuple[str, ...] | None = None


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
) -> tuple[np.ndarray, np.ndarray | float]:
    """Validate one finalized statistic into ``(values, weight)``.

    A scalar weight is returned as a Python float rather than broadcast to
    ``(width,)``. This runs once per ``(group, batch)`` pair per channel --
    hundreds of thousands of times per gene chunk on a screen with ~18k
    perturbations -- and every weight in the docs, the tests and the
    reference reducers is a scalar cell count, so broadcasting it here (and
    then validating and masking ``width`` copies of one number in the
    caller) was the single largest cost in the combine loop.
    """
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
        weight_value = float(weight_arr)
        if not math.isfinite(weight_value) or weight_value < 0:
            raise ValueError(f"Reducer weights for {context} must be finite and non-negative.")
        return values_arr, weight_value
    if weight_arr.ndim != 1 or weight_arr.shape[0] != width:
        raise ValueError(
            f"Reducer weight for {context} must be scalar or have shape ({width},); "
            f"received {weight_arr.shape}."
        )
    if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0):
        raise ValueError(f"Reducer weights for {context} must be finite and non-negative.")
    return values_arr, weight_arr


def _normalise_reducer_output(
    result: Any,
    width: int,
    channels: tuple[str, ...] | None,
    *,
    context: str,
) -> dict[str | None, tuple[np.ndarray, np.ndarray | float]]:
    """Validate and unpack one ``finalize``/``compare`` return value.

    Returns a mapping from channel name to ``(values, weight)`` -- with a
    single ``None`` key for the single-statistic form (``reducer.channels``
    unset).
    """
    if channels:
        if not isinstance(result, dict):
            raise TypeError(
                f"Reducer output for {context} must be a dict with keys {list(channels)} "
                f"(BatchReducer.channels is set); received {type(result).__name__}."
            )
        if set(result.keys()) != set(channels):
            raise ValueError(
                f"Reducer output for {context} must have exactly the keys {list(channels)}; "
                f"received {sorted(result.keys())}."
            )
        return {
            name: _normalise_statistic(result[name], width, context=f"{context}, channel '{name}'")
            for name in channels
        }
    if isinstance(result, dict):
        raise TypeError(
            f"Reducer output for {context} is a dict, but BatchReducer.channels was not "
            "set. Set channels=(...) on the BatchReducer to enable multi-channel output."
        )
    return {None: _normalise_statistic(result, width, context=context)}


def _auto_gene_chunk_size(
    n_obs: int,
    n_genes: int,
    *,
    n_groups: int,
    channels: tuple[str, ...] | None,
    memory_limit_gb: float | None,
) -> int:
    """Gene chunk width used when the caller does not pass ``chunk_size``.

    Shared with the disk-usage resolver, which has to see the same chunk
    count the run will stream to decide whether ``"auto"`` converts.

    :func:`calculate_wilcoxon_chunk_size` deliberately ignores ``n_groups``.
    Here every gene chunk also holds ``(n_groups, width)`` float64 numerator
    and denominator accumulators per output layer (so the chunk can be
    written as one block per layer); cap the width so they fit the same 15%
    per-chunk budget.
    """
    chunk_size = calculate_wilcoxon_chunk_size(
        n_obs, n_genes, available_memory_gb=memory_limit_gb,
    )
    n_layers = len(channels) if channels else 1
    accumulator_budget = 0.15 * _resolve_memory_limit_bytes(memory_limit_gb)
    accumulator_cap = int(accumulator_budget // (2 * n_layers * max(n_groups, 1) * 8))
    return max(1, min(chunk_size, accumulator_cap))


#: ``uns`` key written after the last gene chunk. Its absence marks an output
#: whose run never finished -- a killed job leaves a NaN-filled file carrying
#: complete metadata, which would otherwise read as a valid cached result.
_COMPLETE_KEY = "crispyx_run_complete"


def _run_is_complete(adata: ad.AnnData) -> bool:
    """Whether ``adata`` carries the marker written after the final gene chunk."""
    return bool(np.asarray(adata.uns.get(_COMPLETE_KEY, False)).item())


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
    verbose: int | bool = True,
    memory_limit_gb: float | None = None,
    force: bool = False,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    format_mismatch_policy: Literal["auto", "warn", "convert", "off"] = "auto",
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

    The matrix is streamed gene-chunk by gene-chunk in a single pass
    (:func:`crispyx.data.iter_matrix_chunks`, ``axis=1``) -- the same access
    pattern ``wilcoxon_test`` uses. This is native and cheap for a
    CSC-stored source; see ``format_mismatch_policy`` for a CSR-stored one.
    Each gene chunk's result is also the unit of resumable progress; see
    ``resume``.

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
        ``finalize(state)`` must return one value per gene, or -- with
        ``reducer.channels`` set -- a dict of named channels (see
        :class:`BatchReducer`). Comparison mode also requires
        ``compare(group_state, reference_state)``.
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
        Genes processed per chunk. Automatically selected when omitted, so
        that both the densified cell slabs and the per-chunk
        ``(n_groups, chunk_size)`` accumulators (two float64 arrays per output
        layer) fit the memory budget. Also the unit of resumable progress --
        see ``resume``.
    cell_chunk_size
        Cells densified at a time within a gene chunk, i.e. the largest block
        a reducer ``update`` receives. Each slab is a dense
        ``cell_chunk_size × chunk_size`` float64 array, so this is the
        per-chunk working set. Automatically selected when omitted.
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
    resume
        If True, resume an interrupted run from its checkpoint
        (``<output>.progress.json``), or -- if that checkpoint is missing or
        corrupted -- by scanning the partially-written output file itself for
        the last completed gene chunk. If False (default), always start over.
    checkpoint_interval
        Gene chunks between checkpoint writes. Auto-selected from the number
        of gene chunks when omitted.
    format_mismatch_policy
        How to handle a source stored as CSR, whose gene-(column-)streaming
        here re-reads the *whole* matrix once per gene chunk and holds it in
        memory -- typically ~100x more I/O than CSC:

        * ``"auto"`` (default): measure the source's read throughput and
          convert only when the re-reads would actually cost more than one
          conversion does -- roughly, a large file on a slow (networked)
          filesystem streamed in many chunks. A file that re-reads in
          milliseconds is streamed as-is, silently. See
          :func:`crispyx.data.resolve_auto_format_mismatch_policy`.
        * ``"convert"``: always convert the source to CSC in a temporary file
          beside the output (bounded-memory streaming via
          :func:`crispyx.data.convert_to_csc`, honouring ``memory_limit_gb``)
          and stream from that; the temporary file is removed before
          returning. Needs ~2x the source file's size in free disk space
          there; when that is not available the call falls back to
          ``"warn"`` behaviour with a warning naming the shortfall. Run
          ``cx.pp.convert_to_csc`` once instead if several steps will reuse
          the file.
        * ``"warn"``: proceed on the CSR source after one warning that
          quantifies the cost.
        * ``"off"``: proceed silently with no warning.

    Returns
    -------
    AnnData
        On-disk result with groups in observations, genes in variables, the
        corrected statistic in ``X``, and accumulated weights in
        ``layers['weight_sum']``. With ``reducer.channels`` set, every
        channel's combined values are additionally written to
        ``layers[name]`` (weights to ``layers[f"{name}_weight_sum"]``); the
        first channel's values are the ones copied into ``X``.

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
            return {"sum": np.zeros(width, dtype=np.float64), "count": 0}

        def update_mean(state, block):
            # Accumulate in float64: block follows the file's dtype, so
            # block.sum() on a float32 file would sum in float32.
            state["sum"] += np.asarray(block, dtype=np.float64).sum(axis=0)
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
    validate_format_mismatch_policy(format_mismatch_policy)
    channels = reducer.channels

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
    checkpoint_path = resolved_output.with_suffix(".progress.json")

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
        # Membership against a set, not the list: a screen has as many groups
        # as observed labels, so scanning the list per group is quadratic.
        observed_set = set(observed)
        missing_groups = [group for group in groups if group not in observed_set]
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
            _messages.warn(
                "tl.batch_process",
                f"{n_missing_batch} cells have a missing '{batch_column}' value and are excluded.",
                stacklevel=2,
            )

        reference_code = -2
        # Map cells to group codes through the distinct labels rather than
        # per cell: a dict lookup for every one of a few million cells costs
        # seconds, while there are only as many distinct labels as groups.
        group_lookup = {group: idx for idx, group in enumerate(groups)}
        label_codes, label_uniques = pd.factorize(labels)
        unique_group_codes = np.array(
            [group_lookup.get(str(label), -1) for label in label_uniques], dtype=np.int64
        )
        if mode == "comparison":
            unique_group_codes[label_uniques == control_label] = reference_code
        # A label pandas could not code (-1) indexes the appended sentinel.
        group_codes = np.append(unique_group_codes, -1)[label_codes]

        n_groups = len(groups)
        n_genes = backed.n_vars
        n_batches = len(batch_ids)

        if chunk_size is None and resume and not force and resolved_output.exists():
            # An auto-selected width depends on the memory budget, so resuming
            # under a different one would pick different gene-chunk
            # boundaries, fail the metadata match below, and overwrite the
            # partial output this call was asked to continue -- discarding
            # however many days of completed chunks it holds. Continue on the
            # width that output was written with. Skipped under force=True,
            # which reads nothing from the existing output -- including when
            # it is too damaged to open, the case force exists for.
            existing = ad.read_h5ad(resolved_output, backed="r")
            try:
                stored_chunk_size = existing.uns.get("chunk_size")
            finally:
                existing.file.close()
            if stored_chunk_size is not None:
                chunk_size = int(np.asarray(stored_chunk_size).item())
                _messages.vprint(
                    verbose, "tl.batch_process",
                    f"gene chunk_size={chunk_size} (from {resolved_output.name}, resuming)",
                )
        if chunk_size is None:
            chunk_size = _auto_gene_chunk_size(
                backed.n_obs, n_genes,
                n_groups=n_groups, channels=channels, memory_limit_gb=memory_limit_gb,
            )
            _messages.vprint(verbose, "tl.batch_process", f"gene chunk_size={chunk_size} (auto)")
        if cell_chunk_size is None:
            cell_chunk_size = calculate_optimal_chunk_size(
                backed.n_obs,
                min(n_genes, chunk_size),
                available_memory_gb=memory_limit_gb,
            )
            _messages.vprint(verbose, "tl.batch_process", f"cell_chunk_size={cell_chunk_size} (auto)")
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
            "channels": list(channels) if channels else None,
            "chunk_size": int(chunk_size),
            # Identify the input itself, so that regenerating the source in place
            # invalidates the cache instead of silently returning stale values.
            "source_path": str(path.resolve()),
            "source_mtime_ns": int(path.stat().st_mtime_ns),
        }
        # The output file is created -- with all of this metadata and a NaN
        # fill -- before the first gene chunk runs, so metadata alone cannot
        # tell a finished result from one whose run was killed. Only the
        # completion marker, written after the last chunk, can.
        if resolved_output.exists() and not force:
            existing = ad.read_h5ad(resolved_output, backed="r")
            try:
                matches = _run_is_complete(existing) and _metadata_matches(
                    existing, expected_metadata
                )
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

        n_gene_chunks = (n_genes + chunk_size - 1) // chunk_size
        eff_checkpoint_interval = _get_checkpoint_interval(n_gene_chunks, checkpoint_interval)
        layer_names: list[str | None] = list(channels) if channels else [None]
        # The weight layer the resume fallback scan keys off; it is written
        # last for every gene chunk so a chunk it reports complete has every
        # other dataset of that chunk already written.
        scan_layer: str | None = channels[0] if channels else None
        scan_weight_key = "weight_sum" if scan_layer is None else f"{scan_layer}_weight_sum"

        # ---- Resume bookkeeping: gene chunks complete strictly in order. ----
        last_completed_chunk = -1
        reuse_existing_output = False
        warned_about_overwrite = False
        checkpoint: dict[str, Any] | None = None
        recovered_via_scan = False
        if resume:
            checkpoint = _read_checkpoint(
                checkpoint_path,
                required_keys=("last_gene_chunk", "total_gene_chunks", "batches_used"),
            )
            if checkpoint is not None:
                last_completed_chunk = checkpoint.get("last_gene_chunk", -1)
                _messages.vprint(
                    verbose, "tl.batch_process",
                    f"resuming from gene chunk {last_completed_chunk + 1}/{n_gene_chunks}",
                )
            elif resolved_output.exists():
                # Checkpoint is missing or corrupted -- fall back to scanning
                # the (potentially partial) output file for the last chunk
                # every group's row was fully written for.
                last_completed_chunk = _find_last_completed_gene_chunk(
                    resolved_output, n_gene_chunks, chunk_size, n_genes,
                    weight_dataset=f"layers/{scan_weight_key}",
                )
                if last_completed_chunk >= 0:
                    recovered_via_scan = True
                    _messages.vprint(
                        verbose, "tl.batch_process",
                        f"checkpoint missing/corrupted; recovered progress through "
                        f"gene chunk {last_completed_chunk} by scanning {resolved_output.name}",
                    )
        if resume and last_completed_chunk >= 0 and resolved_output.exists():
            existing = ad.read_h5ad(resolved_output, backed="r")
            try:
                reuse_existing_output = (
                    _metadata_matches(existing, expected_metadata)
                    and existing.shape == (n_groups, n_genes)
                )
            finally:
                existing.file.close()
            if not reuse_existing_output:
                last_completed_chunk = -1
                # The checkpoint describes the output being discarded, so its
                # batches_used grid does not describe the run about to start:
                # keeping it would seed a from-scratch run with another run's
                # batch counts (the grid's shape matches whenever groups and
                # batches are unchanged, which is the common mismatch case).
                checkpoint = None
                warned_about_overwrite = True
                _messages.warn(
                    "tl.batch_process",
                    "Existing partial output does not match this call's parameters "
                    "or source; restarting from scratch, which overwrites it.",
                    stacklevel=2,
                )

        n_layer_arrays = 1 + 2 * len(channels) if channels else 2
        _output_disk_estimate = warn_if_disk_space_low(
            estimate_bytes(max(n_groups, 1), n_genes, overhead=1.10) * n_layer_arrays,
            resolved_output,
            context="tl.batch_process",
        )
        _messages.print_disk_estimate(verbose, "tl.batch_process", _output_disk_estimate)

        if not reuse_existing_output:
            # The run fills the output in place (that is what lets a killed
            # one resume), so starting over replaces whatever is there before
            # it has anything to put in its place: if this run is killed too,
            # neither result survives. Say so rather than let a result
            # disappear silently -- a complete output from a version before
            # the completion marker lands here, as does any rerun whose
            # parameters changed.
            if resolved_output.exists() and not force and not warned_about_overwrite:
                _messages.warn(
                    "tl.batch_process",
                    f"{resolved_output.name} exists but cannot be reused (it is "
                    "unfinished, was written by an earlier crispyx version, or "
                    "does not match this call's parameters), so it is overwritten "
                    "now and holds a usable result again only once this run "
                    "finishes. Copy it aside first if you still need it, or pass "
                    "resume=True to fill it in place from where the last run "
                    "stopped.",
                    stacklevel=2,
                )
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                except Exception:
                    pass
            obs = pd.DataFrame(
                {
                    perturbation_column: groups,
                    "n_batches_used": np.zeros(n_groups, dtype=np.int64),
                },
                index=pd.Index(groups, name="perturbation"),
            )
            var = pd.DataFrame(index=pd.Index(gene_symbols, name=backed.var_names.name))
            placeholder = ad.AnnData(
                sp.csr_matrix((n_groups, n_genes), dtype=np.float64), obs=obs, var=var,
            )
            placeholder.uns.update(expected_metadata)
            placeholder.uns["stratified"] = True
            placeholder.uns["stratified_n_batches"] = int(n_batches)
            placeholder.uns["cell_chunk_size"] = int(cell_chunk_size)
            placeholder.write(resolved_output)

            # HDF5 chunks aligned to the gene chunks, so each finished gene
            # chunk's (n_groups, width) block write -- and the resume scan's
            # column reads -- touch a few whole chunks rather than n_groups
            # strided row segments of a contiguous dataset.
            chunk_width = min(chunk_size, n_genes)
            chunk_rows = max(1, min(n_groups, (8 << 20) // (8 * max(chunk_width, 1))))
            hdf5_chunks = (chunk_rows, chunk_width) if n_groups and n_genes else None

            def _create_dense(group: h5py.Group, name: str, fill: float) -> None:
                if name in group:
                    del group[name]
                ds = group.create_dataset(
                    name, shape=(n_groups, n_genes), dtype="float64", fillvalue=fill,
                    chunks=hdf5_chunks,
                )
                ds.attrs["encoding-type"] = "array"
                ds.attrs["encoding-version"] = "0.2.0"

            with h5py.File(resolved_output, "r+") as f:
                _create_dense(f, "X", np.nan)
                layers_grp = f.require_group("layers")
                if channels:
                    for name in channels:
                        _create_dense(layers_grp, name, np.nan)
                        _create_dense(layers_grp, f"{name}_weight_sum", 0.0)
                else:
                    _create_dense(layers_grp, "weight_sum", 0.0)
            last_completed_chunk = -1

        # ---- Cells sorted by (group, batch) once. Each gene chunk then needs a
        # single O(nnz) row permutation, and the reducer is called once per
        # contiguous (group, batch) segment (per densified slab) instead of
        # once per pair per cell chunk with a mask scan and a scipy
        # fancy-index each time -- ~n_pairs calls per chunk, not ~n_cells.
        usable = (batch_codes >= 0) & (
            (group_codes >= 0) | ((mode == "comparison") & (group_codes == reference_code))
        )
        usable_rows = np.flatnonzero(usable)
        # Reference cells sort after every group (pair_group == n_groups).
        pair_group = np.where(
            group_codes[usable_rows] == reference_code, n_groups, group_codes[usable_rows]
        )
        pair_key = pair_group * n_batches + batch_codes[usable_rows]
        key_order = np.argsort(pair_key, kind="stable")
        order = usable_rows[key_order]
        sorted_key = pair_key[key_order]
        if order.size:
            seg_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_key)) + 1))
            seg_ends = np.concatenate((seg_starts[1:], [order.size]))
        else:
            seg_starts = seg_ends = np.empty(0, dtype=np.int64)
        seg_group = sorted_key[seg_starts] // n_batches
        seg_group = np.where(seg_group == n_groups, reference_code, seg_group)
        seg_batch = sorted_key[seg_starts] % n_batches

        def _pair_context(group_code: int, batch_code: int) -> str:
            group_name = str(control_label) if group_code == reference_code else groups[group_code]
            return f"group '{group_name}', batch '{batch_ids[batch_code]}'"

        with stream_on_fast_axis(
            path, axis=1, policy=format_mismatch_policy, fn_name="tl.batch_process",
            scratch_dir=resolved_output.parent, chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb, verbose=verbose, resumable=resume,
        ) as stream_path:
            stream_backed = backed if stream_path == path else read_backed(stream_path)
            try:
                batches_used = np.zeros((n_groups, n_batches), dtype=bool)
                if checkpoint is not None:
                    # Exact restore: which (group, batch) pairs already
                    # contributed usable weight in the chunks being skipped.
                    restored = _unpack_bool_matrix(
                        checkpoint.get("batches_used"), (n_groups, n_batches)
                    )
                    if restored is not None:
                        batches_used = restored
                    elif last_completed_chunk >= 0:
                        _messages.warn(
                            "tl.batch_process",
                            "The checkpoint's batches_used record could not be read; "
                            "obs['n_batches_used'] may undercount batches used only in "
                            "the skipped chunks. Pass force=True for an exact recount "
                            "if this matters.",
                            stacklevel=2,
                        )
                elif recovered_via_scan and last_completed_chunk >= 0:
                    _messages.warn(
                        "tl.batch_process",
                        "Recovered progress by scanning the output file (checkpoint "
                        "was missing/corrupted); obs['n_batches_used'] may undercount "
                        "batches used only in the recovered chunks. Pass force=True "
                        "for an exact recount if this matters.",
                        stacklevel=2,
                    )

                def _save_checkpoint(chunk_idx: int) -> None:
                    _write_checkpoint_atomic(checkpoint_path, {
                        "total_gene_chunks": n_gene_chunks,
                        "last_gene_chunk": chunk_idx,
                        "batches_used": _pack_bool_matrix(batches_used),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "method": "batch_process",
                        "statistic_name": statistic_name,
                        "mode": mode,
                    })

                with h5py.File(resolved_output, "r+") as out_f, _create_progress_context(
                    n_gene_chunks, "tl.batch_process", verbose, unit="gene chunk",
                    initial=last_completed_chunk + 1,
                ) as pbar:
                    out_X = out_f["X"]
                    out_layers = out_f["layers"]
                    current_chunk = last_completed_chunk + 1
                    for slc, block in iter_matrix_chunks(
                        stream_backed, axis=1, chunk_size=chunk_size, convert_to_dense=False,
                        start_chunk=current_chunk, warn_slow_axis=False,
                    ):
                        gene_start, gene_end = slc.start, slc.stop
                        width = gene_end - gene_start

                        # Row-gather per slab (CSR row indexing is O(nnz of the
                        # selected rows)) so the only per-chunk copies are the
                        # block itself and one densified slab at a time.
                        block = block.tocsr() if sp.issparse(block) else np.asarray(block)
                        states: dict[tuple[int, int], Any] = {}
                        for slab_start in range(0, order.size, cell_chunk_size):
                            slab_end = min(slab_start + cell_chunk_size, order.size)
                            dense = _as_dense(block[order[slab_start:slab_end]])
                            # Segments overlapping [slab_start, slab_end).
                            first = int(np.searchsorted(seg_ends, slab_start, side="right"))
                            last = int(np.searchsorted(seg_starts, slab_end, side="left"))
                            for s in range(first, last):
                                lo = max(int(seg_starts[s]), slab_start) - slab_start
                                hi = min(int(seg_ends[s]), slab_end) - slab_start
                                key = (int(seg_group[s]), int(seg_batch[s]))
                                try:
                                    state = states[key] if key in states else reducer.initialize(width)
                                    replacement = reducer.update(state, dense[lo:hi])
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Reducer failed for {_pair_context(*key)}"
                                    ) from exc
                                states[key] = state if replacement is None else replacement
                        del block

                        # Combine batches per group; states iterate in sorted
                        # (group, batch) order, so each group's batches are
                        # accumulated in ascending batch order.
                        numerators = {
                            name: np.zeros((n_groups, width), dtype=np.float64) for name in layer_names
                        }
                        denominators = {
                            name: np.zeros((n_groups, width), dtype=np.float64) for name in layer_names
                        }
                        for (group_index, batch_index), group_state in states.items():
                            if group_index == reference_code:
                                continue
                            context = _pair_context(group_index, batch_index)
                            try:
                                if mode == "group":
                                    finalized = reducer.finalize(group_state)
                                else:
                                    reference_state = states.get((reference_code, batch_index))
                                    if reference_state is None:
                                        continue
                                    finalized = reducer.compare(group_state, reference_state)  # type: ignore[misc]
                                per_channel = _normalise_reducer_output(
                                    finalized, width, channels, context=context
                                )
                            except Exception as exc:
                                if isinstance(exc, (ValueError, TypeError)) and str(exc).startswith("Reducer"):
                                    raise
                                raise RuntimeError(f"Reducer failed for {context}") from exc
                            for name, (batch_values, batch_weights) in per_channel.items():
                                if isinstance(batch_weights, float):
                                    # Scalar weight: the per-gene mask below
                                    # would be uniformly true, so the masked
                                    # gathers are pure overhead. Same result,
                                    # same NaN propagation from batch_values.
                                    if batch_weights > 0:
                                        numerators[name][group_index] += batch_values * batch_weights
                                        denominators[name][group_index] += batch_weights
                                        batches_used[group_index, batch_index] = True
                                    continue
                                positive = batch_weights > 0
                                if positive.any():
                                    numerators[name][group_index, positive] += (
                                        batch_values[positive] * batch_weights[positive]
                                    )
                                    denominators[name][group_index, positive] += batch_weights[positive]
                                    batches_used[group_index, batch_index] = True
                        del states

                        # One block write per layer per gene chunk. The
                        # numerator is divided in place (no third array); the
                        # scanned weight layer is written last (see scan_layer).
                        for name in layer_names:
                            values = numerators[name]
                            weights = denominators[name]
                            positive = weights > 0
                            np.divide(values, weights, out=values, where=positive)
                            values[~positive] = np.nan
                            if name is None:
                                out_X[:, gene_start:gene_end] = values
                            else:
                                out_layers[name][:, gene_start:gene_end] = values
                                if name == scan_layer:
                                    out_X[:, gene_start:gene_end] = values
                                if name != scan_layer:
                                    out_layers[f"{name}_weight_sum"][:, gene_start:gene_end] = weights
                        out_layers[scan_weight_key][:, gene_start:gene_end] = denominators[scan_layer]
                        del numerators, denominators

                        current_chunk += 1
                        if current_chunk % eff_checkpoint_interval == 0 or current_chunk == n_gene_chunks:
                            out_f.flush()
                            _save_checkpoint(current_chunk - 1)
                        pbar.update(1)

                    # Reduce the weight layer in gene-chunk slices rather
                    # than materialising it: at (18k groups x 35k genes) the
                    # layer is 5 GB, and reading it whole here has OOM-killed
                    # runs that had already completed every gene chunk --
                    # losing days of work at the last step. The slices follow
                    # the HDF5 chunking the output was created with.
                    weight_ds = out_layers[scan_weight_key]
                    untestable = np.ones(n_groups, dtype=bool)
                    for weight_start in range(0, n_genes, chunk_size):
                        weight_stop = min(weight_start + chunk_size, n_genes)
                        untestable &= np.all(
                            weight_ds[:, weight_start:weight_stop] <= 0, axis=1
                        )
            finally:
                if stream_backed is not backed:
                    stream_backed.file.close()

        n_batches_used = batches_used.sum(axis=1).astype(np.int64)
        if untestable.any():
            examples = [groups[i] for i in np.flatnonzero(untestable)[:5]]
            _messages.warn(
                "tl.batch_process",
                f"{int(untestable.sum())} group(s) have no usable batch statistics; "
                f"results are NaN. Examples: {examples}",
                stacklevel=2,
            )

        final_obs = pd.DataFrame(
            {perturbation_column: groups, "n_batches_used": n_batches_used},
            index=pd.Index(groups, name="perturbation"),
        )
        with h5py.File(resolved_output, "r+") as f:
            _update_h5ad_dataframe(f, "obs", final_obs)
            uns_grp = f.require_group("uns")

            def _write_uns_scalar(key: str, value: int) -> None:
                if key in uns_grp:
                    del uns_grp[key]
                ds = uns_grp.create_dataset(key, data=value)
                ds.attrs["encoding-type"] = "numeric-scalar"
                ds.attrs["encoding-version"] = "0.2.0"

            _write_uns_scalar(
                "stratified_n_untestable_perturbations", int(untestable.sum())
            )
            # Written last, after every chunk and every other dataset: this is
            # what marks the output complete for the cache check above.
            f.flush()
            _write_uns_scalar(_COMPLETE_KEY, 1)

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except Exception:
                pass
    finally:
        backed.file.close()

    if int(verbose) >= 1:
        print(f"[cx] tl.batch_process: Saving → {resolved_output}")
    return AnnData(resolved_output)


def _estimate_shape_for_batch_process(
    path: Path,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    batch_column: str,
    mode: Literal["group", "comparison"] = "group",
    perturbations: Iterable[str] | None = None,
    reducer: BatchReducer | None = None,
    format_mismatch_policy: str = "auto",
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    **_ignored,
) -> dict[str, float]:
    """Disk-usage resolver for :func:`batch_process`, used by
    ``crispyx.estimate_disk_usage()`` (see ``_preflight.py``). Reproduces the
    cheap, obs-only group-resolution preamble that :func:`batch_process`
    already runs before pre-sizing its output file's ``X``/``layers``, so the
    estimate can never disagree with the automatic warning emitted inside the
    real call. Results are written directly into the output file (no
    separate tempdir accumulator); a CSR source additionally gets a temporary
    CSC copy beside the output when the policy converts.
    """
    validate_format_mismatch_policy(format_mismatch_policy)
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="batch_process",
    )
    backed = read_backed(path)
    try:
        n_obs, n_genes = backed.n_obs, backed.n_vars
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        observed = _unique_strings(labels)
        if mode == "comparison":
            control_label = resolve_control_label(labels, control_label)
        if perturbations is None:
            groups = [g for g in observed if mode == "group" or g != control_label]
        else:
            groups = _unique_strings(perturbations)
            if mode == "comparison":
                groups = [g for g in groups if g != control_label]
    finally:
        backed.file.close()
    n_groups = max(len(groups), 1)
    channels = reducer.channels if isinstance(reducer, BatchReducer) else None
    n_layer_arrays = 1 + 2 * len(channels) if channels else 2
    if chunk_size is None:
        chunk_size = _auto_gene_chunk_size(
            n_obs, n_genes,
            n_groups=len(groups), channels=channels, memory_limit_gb=memory_limit_gb,
        )
    estimate = {
        "output": estimate_bytes(n_groups, n_genes, overhead=1.10) * n_layer_arrays,
    }
    scratch = scratch_copy_bytes(
        path, axis=1, policy=format_mismatch_policy, chunk_size=chunk_size
    )
    if scratch is not None:
        estimate["scratch"] = scratch
    return estimate


__all__ = ["BatchReducer", "BatchStatistic", "batch_process"]
