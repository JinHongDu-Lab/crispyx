"""Pseudo-bulk effect size estimators operating directly on ``.h5ad`` files."""

from __future__ import annotations

import hashlib
import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from ._disk import estimate_bytes, warn_if_disk_space_low
from ._grouping import resolve_group_reference_aliases
from .data import (
    AnnData,
    calculate_optimal_chunk_size,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
)


PseudobulkMethod = Literal["mean_log1p", "sum", "log_mean"]
# The subset compute_normalized_effects accepts. "sum" belongs to aggregate_pseudobulk,
# so keeping the public hint narrow stops a type checker from waving it through.
NormalizedEffectMethod = Literal["mean_log1p", "log_mean"]

# Output naming. "modern" is what compute_normalized_effects returns; the two
# deprecated wrappers request their historical names so existing code keeps working.
_NORMALIZED_LAYOUTS = {
    "modern": {
        "profile_layer": "perturbation_profile",
        "matched_layer": "control_profile_matched",
        "reference_uns": "control_profile",
        "suffix": "normalized_effects",
        "log_name": "pb.normalized_effects",
    },
    "mean_log1p": {
        "profile_layer": "perturbation_mean",
        "matched_layer": "control_mean_matched",
        "reference_uns": "control_mean",
        "suffix": "avg_log_effects",
        "log_name": "pb.compute_average_log_expression",
    },
    "log_mean": {
        "profile_layer": "perturbation_bulk",
        "matched_layer": "control_bulk_matched",
        "reference_uns": "control_bulk",
        "suffix": "pseudobulk_effects",
        "log_name": "pb.compute_pseudobulk_expression",
    },
}


def _resolve_candidates(
    labels: np.ndarray,
    control_label: str,
    perturbations: Iterable[str] | None,
) -> list[str]:
    if perturbations is None:
        unique = pd.Index(labels).unique().tolist()
    else:
        unique = [str(p) for p in perturbations]
    return [label for label in unique if label != control_label]


def _densify_block(block) -> np.ndarray:
    """Return ``block`` as a contiguous ``float64`` dense array."""
    if sparse.issparse(block):
        return np.asarray(block.toarray(), dtype=np.float64)
    return np.asarray(block, dtype=np.float64)


def _streaming_batch_corrected(
    backed,
    *,
    labels: np.ndarray,
    batch_labels: np.ndarray,
    candidates: list[str],
    control_label: str,
    n_genes: int,
    chunk_size: int,
    block_fn: Callable[[object], np.ndarray],
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """Compute batch-corrected effects in a single, bounded-memory pass.

    The per-``(perturbation, batch)`` sum accumulator -- the only quantity that
    scales with the number of batches -- is spilled to a disk-backed
    ``np.memmap`` so peak RAM stays ``O(chunk_size x n_genes + n_batches x
    n_genes + n_candidates x n_genes)`` regardless of the number of gem-groups.
    Control per-batch sums (``n_batches x n_genes``) are small and kept in RAM.

    For every ``(perturbation, batch)`` pair the within-batch effect is
    ``transform(pert_sum, n_pert) - transform(ctrl_sum, n_ctrl)``.  Pairs whose
    batch contains no control cells carry no contrast and are skipped.  The
    remaining within-batch effects are averaged per perturbation with
    harmonic-count weights ``w_b = n_pert_b * n_ctrl_b / (n_pert_b + n_ctrl_b)``.

    Parameters
    ----------
    backed
        Open, disk-backed AnnData handle to stream chunks from.  It must remain
        open for the duration of the call (chunks are read lazily).
    labels
        Per-cell perturbation labels, shape ``(n_obs,)``.
    batch_labels
        Per-cell batch (e.g. gem-group) labels, shape ``(n_obs,)``.  Factorised
        internally into contiguous integer codes.
    candidates
        Ordered list of perturbation labels to score (control excluded).  Row
        ``i`` of every returned matrix corresponds to ``candidates[i]``.
    control_label
        Label identifying the control group in ``labels``.
    n_genes
        Number of genes (columns) in the expression matrix.
    chunk_size
        Number of cells streamed per chunk.
    block_fn
        Maps a raw chunk to the dense per-cell accumuland (e.g. ``log1p`` of the
        normalised counts for average-log expression, or the normalised counts
        for pseudo-bulk).  Must return a ``(chunk_rows, n_genes)`` array.
    transform
        Maps ``(summed_accumuland, n_cells)`` to the per-batch expression used
        in the effect, where ``n_cells`` is broadcast as a column vector
        (shape ``(k, 1)``).  For average-log expression this is ``S / n``; for
        pseudo-bulk it is ``log1p(baseline_count * S / n)``.

    Returns
    -------
    effect_matrix : ndarray, shape (n_candidates, n_genes)
        Batch-corrected effect (harmonic-count weighted average of within-batch
        differences).  Equals ``pert_mean_corrected - ctrl_mean_matched``.
    pert_mean_corrected : ndarray, shape (n_candidates, n_genes)
        Batch-corrected per-perturbation expression: the harmonic-count weighted
        average of the within-batch perturbation expressions ``T_{p,b}``.
    ctrl_mean_matched : ndarray, shape (n_candidates, n_genes)
        Per-perturbation weight-matched control reference: the harmonic-count
        weighted average of the within-batch control expressions ``T_{c,b}``
        using each perturbation's own batch weights.
    pooled_ctrl_sum : ndarray, shape (n_genes,)
        Pooled control sum of the accumuland (for a global pooled reference).
    pooled_ctrl_count : int
        Pooled control cell count.
    batch_ids : list[str]
        Batch labels encountered, in first-appearance order.

    Raises
    ------
    ValueError
        If a candidate perturbation has no cells, or shares no batch with the
        control group (so no batch-corrected effect can be formed).
    """
    n_candidates = len(candidates)

    # Integer codes.  ``get_indexer`` returns -1 for control / excluded cells.
    pert_code = pd.Index(candidates).get_indexer(pd.Index(labels))
    batch_code, batch_uniques = pd.factorize(batch_labels, sort=False)
    batch_code = batch_code.astype(np.int64)
    n_batches = len(batch_uniques)
    batch_ids = [str(b) for b in batch_uniques]
    ctrl_mask_full = labels == control_label

    # Enumerate the present (perturbation, batch) pairs.
    cand_cell = pert_code >= 0
    pair_key_all = pert_code[cand_cell].astype(np.int64) * n_batches + batch_code[cand_cell]
    uniq_pairs = np.unique(pair_key_all)
    n_pairs = int(uniq_pairs.shape[0])
    pert_of_pair = (uniq_pairs // n_batches).astype(np.int64)
    batch_of_pair = (uniq_pairs % n_batches).astype(np.int64)

    # Accumulators: control in RAM (small), pair sums on disk (memmap).
    ctrl_sums = np.zeros((n_batches, n_genes), dtype=np.float64)
    ctrl_counts = np.zeros(n_batches, dtype=np.int64)
    pair_counts = np.zeros(n_pairs, dtype=np.int64)

    tmp = tempfile.NamedTemporaryFile(prefix="cx_pb_pairsums_", suffix=".dat", delete=False)
    tmp.close()
    warn_if_disk_space_low(
        estimate_bytes(max(n_pairs, 1), n_genes), tmp.name,
        context="pb batch-corrected accumulator",
    )
    pair_sums = np.memmap(tmp.name, dtype=np.float64, mode="w+", shape=(max(n_pairs, 1), n_genes))
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            acc = block_fn(block)
            bc = batch_code[slc]
            pc = pert_code[slc]
            cm = ctrl_mask_full[slc]

            # Control cells -> per-batch sums (RAM).
            if cm.any():
                sub = acc[cm]
                cb = bc[cm]
                ub, inv = np.unique(cb, return_inverse=True)
                indicator = sparse.csr_matrix(
                    (np.ones(cb.shape[0], dtype=np.float64), (inv, np.arange(cb.shape[0]))),
                    shape=(ub.shape[0], cb.shape[0]),
                )
                ctrl_sums[ub] += indicator @ sub
                ctrl_counts[ub] += np.bincount(inv, minlength=ub.shape[0]).astype(np.int64)

            # Perturbation cells -> per-(pert, batch) sums (memmap).
            pm = pc >= 0
            if pm.any():
                sub = acc[pm]
                keys = pc[pm].astype(np.int64) * n_batches + bc[pm]
                slots = np.searchsorted(uniq_pairs, keys)
                us, inv = np.unique(slots, return_inverse=True)
                indicator = sparse.csr_matrix(
                    (np.ones(keys.shape[0], dtype=np.float64), (inv, np.arange(keys.shape[0]))),
                    shape=(us.shape[0], keys.shape[0]),
                )
                pair_sums[us] = pair_sums[us] + (indicator @ sub)
                pair_counts[us] += np.bincount(inv, minlength=us.shape[0]).astype(np.int64)

        pooled_ctrl_sum = ctrl_sums.sum(axis=0)
        pooled_ctrl_count = int(ctrl_counts.sum())

        # Per-perturbation pooled counts (for the "no cells" guard).
        pooled_pert_counts = np.zeros(n_candidates, dtype=np.int64)
        np.add.at(pooled_pert_counts, pert_of_pair, pair_counts)

        # Weighted numerators for the batch-corrected perturbation expression and
        # the weight-matched control reference.
        pert_expr_num = np.zeros((n_candidates, n_genes), dtype=np.float64)
        ctrl_expr_num = np.zeros((n_candidates, n_genes), dtype=np.float64)
        weight_tot = np.zeros(n_candidates, dtype=np.float64)

        n_c_pair = ctrl_counts[batch_of_pair]
        pair_block = 4096
        for start in range(0, n_pairs, pair_block):
            end = min(start + pair_block, n_pairs)
            # Only pairs whose batch also contains control cells carry a contrast.
            n_p = pair_counts[start:end]
            n_c = n_c_pair[start:end]
            valid = (n_p > 0) & (n_c > 0)
            if not valid.any():
                continue
            Sp = np.asarray(pair_sums[start:end])[valid]
            n_pv = n_p[valid].astype(np.float64)
            n_cv = n_c[valid].astype(np.float64)
            p_idx = pert_of_pair[start:end][valid]
            b_idx = batch_of_pair[start:end][valid]
            w = (n_pv * n_cv) / (n_pv + n_cv)
            Tp = transform(Sp, n_pv[:, None])
            Tc = transform(ctrl_sums[b_idx], n_cv[:, None])
            np.add.at(pert_expr_num, p_idx, w[:, None] * Tp)
            np.add.at(ctrl_expr_num, p_idx, w[:, None] * Tc)
            np.add.at(weight_tot, p_idx, w)
    finally:
        pair_sums._mmap.close()  # type: ignore[attr-defined]
        os.unlink(tmp.name)

    no_cells = pooled_pert_counts == 0
    if no_cells.any():
        bad = candidates[int(np.argmax(no_cells))]
        raise ValueError(f"Perturbation '{bad}' contains no cells")
    zero = weight_tot <= 0
    if zero.any():
        bad = candidates[int(np.argmax(zero))]
        raise ValueError(
            f"Perturbation '{bad}' shares no batch with the control group; "
            "cannot compute a batch-corrected effect."
        )
    inv_w = 1.0 / weight_tot[:, None]
    pert_mean_corrected = pert_expr_num * inv_w
    ctrl_mean_matched = ctrl_expr_num * inv_w
    effect_matrix = pert_mean_corrected - ctrl_mean_matched
    return (
        effect_matrix,
        pert_mean_corrected,
        ctrl_mean_matched,
        pooled_ctrl_sum,
        pooled_ctrl_count,
        batch_ids,
    )


def _normalized_effects_impl(
    data: str | Path | AnnData | ad.AnnData,
    *,
    layout: str,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    method: NormalizedEffectMethod = "mean_log1p",
    baseline_count: float = 1.0,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
) -> AnnData:
    """Shared implementation. ``layout`` picks the output naming scheme.

    This is the one-command path from a cell-level ``.h5ad`` to an effect size: it
    normalises each cell's counts, aggregates per perturbation, and subtracts the
    reference, in a single streaming pass over the matrix.

    Unlike :func:`compute_pseudobulk_effects`, which works on whatever scale its
    input already carries, this function **normalises library size itself**. Do not
    pre-normalise the input as well.

    Parameters
    ----------
    data
        Path to an h5ad file, or a backed AnnData object.
    perturbation_column
        Column in ``adata.obs`` identifying perturbation groups.
    groupby
        Scanpy-compatible alias for ``perturbation_column``.
    control_label
        Label of the reference group. Inferred from common patterns when omitted.
    reference
        Scanpy-compatible alias for ``control_label``.
    method
        Which scale the averaging happens on. Both normalise library size first:

        * ``"mean_log1p"`` averages per-cell ``log1p`` values (mean of logs).
        * ``"log_mean"`` averages normalised counts, then takes
          ``log1p(baseline_count * mean)`` (log of mean).

        These are different estimators, not different spellings: averaging before
        or after the log gives different answers whenever expression varies within
        a group. ``"sum"`` is not accepted here; it belongs to
        :func:`aggregate_pseudobulk`.
    baseline_count
        Scaling applied inside the log for ``method="log_mean"``. Ignored by
        ``"mean_log1p"``. Must be positive.
    gene_name_column
        Column in ``adata.var`` with gene symbols. Uses ``var_names`` when omitted.
    perturbations
        Subset of perturbation labels to score. All non-reference groups when
        omitted.
    batch_column
        Column in ``adata.obs`` identifying each cell's batch. When given, effects
        are computed within each batch and combined with harmonic target/reference
        cell-count weights, ``w_b = n_tb * n_rb / (n_tb + n_rb)``, which removes
        batch-driven confounding. Batches without reference cells support no
        contrast and are skipped. When omitted, a single pooled effect is computed.
        There is no flag to disable the correction: supplying the column is the
        request for it.
    chunk_size
        Cells streamed per chunk. Auto-selected when omitted.
    memory_limit_gb
        Soft memory budget used to size the chunk. Only the chunk size is affected;
        results are identical regardless.
    data_name
        Custom stem for the output filename.
    output_path
        Exact output h5ad path. Takes precedence over ``output_dir`` / ``data_name``.
    output_dir
        Deprecated output directory override; use ``output_path``.
    verbose
        ``1`` / ``True`` prints a summary line.

    Returns
    -------
    AnnData
        On-disk result with perturbations in observations and the effect in ``X``.
        ``layers['perturbation_profile']`` holds the per-perturbation profile and
        ``uns['control_profile']`` the pooled reference profile. With
        ``batch_column``, ``X`` is the batch-corrected effect,
        ``layers['perturbation_profile']`` its batch-corrected profile, and
        ``layers['control_profile_matched']`` the weight-matched reference, so that
        ``X == perturbation_profile - control_profile_matched`` exactly.
        ``uns['batch_column']`` and ``uns['batch_ids']`` record the stratification.

    See Also
    --------
    compute_pseudobulk_effects : contrast on the input's own scale; also accepts a
        saved :func:`aggregate_pseudobulk` artifact.
    aggregate_pseudobulk : absolute profiles rather than a contrast.
    """
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="compute_normalized_effects",
    )
    if method not in ("mean_log1p", "log_mean"):
        raise ValueError(
            "method must be 'mean_log1p' or 'log_mean'; "
            f"received {method!r}. Use aggregate_pseudobulk for method='sum'."
        )
    if baseline_count <= 0:
        raise ValueError("baseline_count must be positive")
    naming = _NORMALIZED_LAYOUTS[layout]
    log_name = naming["log_name"]

    # The two methods differ only in where the log is applied.
    if method == "mean_log1p":
        def _per_cell(values: np.ndarray) -> np.ndarray:
            return np.log1p(values)

        def _finalise(mean: np.ndarray) -> np.ndarray:
            return mean
    else:
        def _per_cell(values: np.ndarray) -> np.ndarray:
            return values

        def _finalise(mean: np.ndarray) -> np.ndarray:
            return np.log1p(baseline_count * mean)

    path = resolve_data_path(data)
    if int(verbose) >= 1:
        print(f"[cx] {log_name}: Reading {path}")
    backed = read_backed(path)
    use_batch = batch_column is not None
    effect_matrix_np = np.empty((0, 0), dtype=np.float64)
    pert_corrected = np.empty((0, 0), dtype=np.float64)
    ctrl_matched = np.empty((0, 0), dtype=np.float64)
    pooled_ctrl_sum = np.empty(0, dtype=np.float64)
    pooled_ctrl_count = 0
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    try:
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(
                backed.n_obs, backed.n_vars, available_memory_gb=memory_limit_gb,
            )
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in "
                f"adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        batch_ids: list[str] = []
        if use_batch:
            if batch_column not in backed.obs.columns:
                raise KeyError(
                    f"Batch column '{batch_column}' was not found in adata.obs. "
                    f"Available columns: {list(backed.obs.columns)}"
                )
            batch_labels = backed.obs[batch_column].astype(str).to_numpy()

            def _block_fn(block) -> np.ndarray:
                normalised, _ = normalize_total_block(block)
                return _per_cell(_densify_block(normalised))

            def _transform(agg: np.ndarray, n: np.ndarray) -> np.ndarray:
                return _finalise(agg / n)

            (
                effect_matrix_np,
                pert_corrected,
                ctrl_matched,
                pooled_ctrl_sum,
                pooled_ctrl_count,
                batch_ids,
            ) = _streaming_batch_corrected(
                backed,
                labels=labels,
                batch_labels=batch_labels,
                candidates=candidates,
                control_label=control_label,
                n_genes=n_genes,
                chunk_size=chunk_size,
                block_fn=_block_fn,
                transform=_transform,
            )
        else:
            groups = [control_label] + candidates
            sums = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
            counts = {label: 0 for label in groups}
            for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
                slice_labels = labels[slc]
                normalised_block, _ = normalize_total_block(block)
                accumuland = _per_cell(normalised_block)
                for label in groups:
                    mask = slice_labels == label
                    if not np.any(mask):
                        continue
                    sums[label] += accumuland[mask].sum(axis=0)
                    counts[label] += int(mask.sum())
    finally:
        backed.file.close()

    matched_profile = None
    if use_batch:
        if pooled_ctrl_count == 0:
            raise ValueError("Control group contains no cells")
        control_profile = _finalise(pooled_ctrl_sum / pooled_ctrl_count)
        pert_profiles = list(pert_corrected)
        matched_profile = ctrl_matched
        effect_matrix = list(effect_matrix_np)
    else:
        if counts[control_label] == 0:
            raise ValueError("Control group contains no cells")
        control_profile = _finalise(sums[control_label] / counts[control_label])
        effect_matrix = []
        pert_profiles = []
        for label in candidates:
            if counts[label] == 0:
                raise ValueError(f"Perturbation '{label}' contains no cells")
            profile = _finalise(sums[label] / counts[label])
            pert_profiles.append(profile)
            effect_matrix.append(profile - control_profile)

    resolved_output = resolve_output_path(
        path,
        suffix=naming["suffix"],
        output_dir=output_dir,
        data_name=data_name,
        output_path=output_path,
    )
    if not effect_matrix:
        adata = ad.AnnData(
            np.zeros((0, gene_symbols.shape[0])),
            obs=pd.DataFrame(index=pd.Index([], name="perturbation")),
            var=pd.DataFrame(index=gene_symbols),
        )
        adata.uns[naming["reference_uns"]] = control_profile
        adata.uns["method"] = str(method)
        if method == "log_mean":
            adata.uns["baseline_count"] = float(baseline_count)
        if int(verbose) >= 1:
            print(f"[cx] {log_name}: 0 perturbations × {gene_symbols.shape[0]} genes")
            print(f"[cx] {log_name}: Saving → {resolved_output}")
        adata.write(resolved_output)
        return AnnData(resolved_output)

    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    adata = ad.AnnData(
        np.vstack(effect_matrix), obs=obs, var=pd.DataFrame(index=gene_symbols)
    )
    adata.layers[naming["profile_layer"]] = np.vstack(pert_profiles)
    adata.uns[naming["reference_uns"]] = control_profile
    adata.uns["method"] = str(method)
    if method == "log_mean":
        adata.uns["baseline_count"] = float(baseline_count)
    if use_batch:
        adata.layers[naming["matched_layer"]] = np.asarray(matched_profile)
        adata.uns["batch_column"] = str(batch_column)
        adata.uns["batch_ids"] = np.asarray(batch_ids, dtype=object)
    n_layers = 3 if use_batch else 2  # X + profile_layer (+ matched_layer if batch-corrected)
    warn_if_disk_space_low(
        estimate_bytes(len(candidates), n_genes, n_layers, overhead=1.10), resolved_output,
        context=log_name,
    )
    if int(verbose) >= 1:
        print(f"[cx] {log_name}: {len(candidates)} perturbations × {len(gene_symbols)} genes")
        print(f"[cx] {log_name}: Saving → {resolved_output}")
    adata.write(resolved_output)
    return AnnData(resolved_output)


def compute_normalized_effects(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    method: NormalizedEffectMethod = "mean_log1p",
    baseline_count: float = 1.0,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
) -> AnnData:
    """Compute library-size-normalised target-minus-reference effects from cells.

    This is the one-command path from a cell-level ``.h5ad`` to an effect size: it
    normalises each cell's counts, aggregates per perturbation, and subtracts the
    reference, in a single streaming pass over the matrix.

    Unlike :func:`compute_pseudobulk_effects`, which works on whatever scale its
    input already carries, this function **normalises library size itself**. Do not
    pre-normalise the input as well.

    Parameters
    ----------
    data
        Path to an h5ad file, or a backed AnnData object.
    perturbation_column
        Column in ``adata.obs`` identifying perturbation groups.
    groupby
        Scanpy-compatible alias for ``perturbation_column``.
    control_label
        Label of the reference group. Inferred from common patterns when omitted.
    reference
        Scanpy-compatible alias for ``control_label``.
    method
        Which scale the averaging happens on. Both normalise library size first:

        * ``"mean_log1p"`` averages per-cell ``log1p`` values (mean of logs).
        * ``"log_mean"`` averages normalised counts, then takes
          ``log1p(baseline_count * mean)`` (log of mean).

        These are different estimators, not different spellings: averaging before
        or after the log gives different answers whenever expression varies within
        a group. ``"sum"`` is not accepted here; it belongs to
        :func:`aggregate_pseudobulk`.
    baseline_count
        Scaling applied inside the log for ``method="log_mean"``. Ignored by
        ``"mean_log1p"``. Must be positive.
    gene_name_column
        Column in ``adata.var`` with gene symbols. Uses ``var_names`` when omitted.
    perturbations
        Subset of perturbation labels to score. All non-reference groups when
        omitted.
    batch_column
        Column in ``adata.obs`` identifying each cell's batch. When given, effects
        are computed within each batch and combined with harmonic target/reference
        cell-count weights, ``w_b = n_tb * n_rb / (n_tb + n_rb)``, which removes
        batch-driven confounding. Batches without reference cells support no
        contrast and are skipped. When omitted, a single pooled effect is computed.
        There is no flag to disable the correction: supplying the column is the
        request for it.
    chunk_size
        Cells streamed per chunk. Auto-selected when omitted.
    memory_limit_gb
        Soft memory budget used to size the chunk. Only the chunk size is affected;
        results are identical regardless.
    data_name
        Custom stem for the output filename.
    output_path
        Exact output h5ad path. Takes precedence over ``output_dir`` / ``data_name``.
    output_dir
        Deprecated output directory override; use ``output_path``.
    verbose
        ``1`` / ``True`` prints a summary line.

    Returns
    -------
    AnnData
        On-disk result with perturbations in observations and the effect in ``X``.
        ``layers['perturbation_profile']`` holds the per-perturbation profile,
        ``uns['control_profile']`` the pooled reference profile, and
        ``uns['method']`` the estimator used. With ``batch_column``, ``X`` is the
        batch-corrected effect, ``layers['perturbation_profile']`` its
        batch-corrected profile, and ``layers['control_profile_matched']`` the
        weight-matched reference, so that
        ``X == perturbation_profile - control_profile_matched`` exactly.
        ``uns['batch_column']`` and ``uns['batch_ids']`` record the stratification.
        ``method="log_mean"`` additionally records ``uns['baseline_count']``.

    See Also
    --------
    compute_pseudobulk_effects : contrast on the input's own scale; also accepts a
        saved :func:`aggregate_pseudobulk` artifact.
    aggregate_pseudobulk : absolute profiles rather than a contrast.
    """
    return _normalized_effects_impl(
        data,
        layout="modern",
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        method=method,
        baseline_count=baseline_count,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        batch_column=batch_column,
        chunk_size=chunk_size,
        memory_limit_gb=memory_limit_gb,
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
        verbose=verbose,
    )


def compute_average_log_expression(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
) -> AnnData:
    """Deprecated alias for ``compute_normalized_effects(method="mean_log1p")``.

    Retained for backward compatibility, including its original layer and ``uns``
    names. The output additionally carries ``uns['method']``, which earlier releases
    did not write; nothing else changes. Scheduled for removal in 0.1.0.
    """
    warnings.warn(
        "compute_average_log_expression() is deprecated and will be removed in "
        "crispyx 0.1.0. Use compute_normalized_effects(..., method='mean_log1p') "
        "(cx.pb.normalized_effects) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _normalized_effects_impl(
        data,
        layout="mean_log1p",
        perturbation_column=perturbation_column,
        control_label=control_label,
        method="mean_log1p",
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        batch_column=batch_column,
        chunk_size=chunk_size,
        memory_limit_gb=memory_limit_gb,
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
        verbose=verbose,
    )


def compute_pseudobulk_expression(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    baseline_count: float = 1.0,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
) -> AnnData:
    """Deprecated alias for ``compute_normalized_effects(method="log_mean")``.

    Retained for backward compatibility, including its original layer and ``uns``
    names. The output additionally carries ``uns['method']``, which earlier releases
    did not write; nothing else changes. Scheduled for removal in 0.1.0.
    """
    warnings.warn(
        "compute_pseudobulk_expression() is deprecated and will be removed in "
        "crispyx 0.1.0. Use compute_normalized_effects(..., method='log_mean') "
        "(cx.pb.normalized_effects) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _normalized_effects_impl(
        data,
        layout="log_mean",
        perturbation_column=perturbation_column,
        control_label=control_label,
        method="log_mean",
        baseline_count=baseline_count,
        gene_name_column=gene_name_column,
        perturbations=perturbations,
        batch_column=batch_column,
        chunk_size=chunk_size,
        memory_limit_gb=memory_limit_gb,
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
        verbose=verbose,
    )


def _normalise_groupby(groupby: str | Sequence[str]) -> list[str]:
    if isinstance(groupby, str):
        columns = [groupby]
    else:
        columns = [str(column) for column in groupby]
    if not columns or any(not column for column in columns):
        raise ValueError("groupby must contain at least one non-empty obs column name")
    if len(set(columns)) != len(columns):
        raise ValueError("groupby must not contain duplicate column names")
    return columns


def _matrix_block(matrix, start: int, end: int):
    return matrix[start:end]


def _validate_expression_block(block, *, context: str) -> bool:
    """Validate one expression block and report whether all values are integer-like."""
    values = block.data if sparse.issparse(block) else np.asarray(block)
    values = np.asarray(values)
    if values.size == 0:
        return True
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} contains non-finite values")
    if np.any(values < 0):
        raise ValueError(f"{context} contains negative values")
    return bool(np.all(np.abs(values - np.rint(values)) <= 1e-8))


def _group_seed(random_state: int, key: tuple[str, ...]) -> np.random.SeedSequence:
    digest = hashlib.blake2b(
        "\x1f".join(key).encode("utf-8"), digest_size=8
    ).digest()
    key_seed = int.from_bytes(digest, "little", signed=False)
    return np.random.SeedSequence(
        [int(random_state), key_seed & 0xFFFFFFFF, key_seed >> 32]
    )


def _metadata_value_equal(actual, expected) -> bool:
    if isinstance(expected, list):
        return [str(value) for value in np.asarray(actual).tolist()] == expected
    if isinstance(expected, bool):
        return bool(actual) is expected
    return actual == expected


def aggregate_pseudobulk(
    data: str | Path | AnnData | ad.AnnData,
    *,
    groupby: str | Sequence[str],
    method: PseudobulkMethod = "mean_log1p",
    layer: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells: int = 5,
    bootstrap_size: int | None = None,
    random_state: int = 0,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
    force: bool = False,
) -> AnnData:
    """Aggregate cells into absolute pseudo-bulk profiles.

    Each output observation represents one observed combination of the
    columns in ``groupby``. For example, ``groupby=["perturbation", "batch"]``
    retains repeated batch-level measurements for every perturbation instead
    of averaging those batches together.

    Parameters
    ----------
    data
        Path to an h5ad file, :class:`crispyx.AnnData`, or a backed AnnData.
    groupby
        One observation column, or an ordered sequence of columns defining a
        pseudo-bulk sample.
    method
        ``"sum"`` sums raw counts and therefore requires finite,
        non-negative, integer-valued input. ``"mean_log1p"`` computes the
        mean of per-cell log1p expression. Integer count input is transformed
        with ``log1p`` first; continuous input is treated as already
        log1p-transformed and averaged unchanged.
    layer
        Expression layer to aggregate. Uses ``X`` when omitted.
    gene_name_column
        Column in ``adata.var`` containing gene symbols. Uses ``var_names``
        when omitted.
    perturbations
        Optional subset of grouping values to retain. A profile is kept when any
        of its ``groupby`` values matches, so the argument selects on whichever
        column holds the labels regardless of its position, and every observed
        combination of the remaining columns is preserved. With
        ``groupby=["perturbation", "batch"]``, ``perturbations=["KO_A"]``
        therefore returns one profile per batch for ``KO_A``.
    min_cells
        Minimum number of source cells required for an observed group.
        Defaults to 5.
    bootstrap_size
        If provided, draw exactly this many cells once with replacement from
        every retained group before aggregation. Sampling is deterministic for
        a fixed ``random_state`` and independent of streaming chunk sizes.
    random_state
        Seed for bootstrap sampling.
    chunk_size
        Number of cells read per streaming block. Automatically selected when
        omitted.
    memory_limit_gb
        Soft memory budget used for automatic chunk sizing.
    data_name, output_path, output_dir
        Output naming controls. ``output_path`` takes precedence.
    verbose
        Level 1 reports input interpretation and save summaries.
    force
        Recompute an existing output instead of reloading a matching result.

    Returns
    -------
    AnnData
        On-disk result with absolute profiles in ``X``; grouping columns,
        ``n_cells``, and ``n_cells_aggregated`` in ``obs``; and a versioned
        ``uns['crispyx_pseudobulk']`` provenance marker.
    """
    if method not in ("mean_log1p", "sum"):
        raise ValueError("method must be 'mean_log1p' or 'sum'")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")
    if bootstrap_size is not None and bootstrap_size < 1:
        raise ValueError("bootstrap_size must be at least 1")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer")

    columns = _normalise_groupby(groupby)
    path = resolve_data_path(data)
    resolved_output = resolve_output_path(
        path,
        suffix=f"pseudobulk_{method}",
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
    )
    backed = read_backed(path)
    try:
        missing_columns = [column for column in columns if column not in backed.obs.columns]
        if missing_columns:
            raise KeyError(
                f"Grouping column(s) {missing_columns} were not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        if layer is not None and layer not in backed.layers:
            raise KeyError(
                f"Layer '{layer}' was not found. Available layers: {list(backed.layers.keys())}"
            )
        matrix = backed.X if layer is None else backed.layers[layer]
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column).astype(str)

        grouping = backed.obs[columns]
        usable = ~grouping.isna().any(axis=1).to_numpy()
        n_missing = int((~usable).sum())
        if n_missing:
            warnings.warn(
                f"{n_missing} cells have missing grouping values and are excluded.",
                stacklevel=2,
            )
        raw_codes = np.full(backed.n_obs, -1, dtype=np.int64)
        if usable.any():
            usable_index = np.flatnonzero(usable)
            multi = pd.MultiIndex.from_frame(grouping.iloc[usable_index].astype(str))
            usable_codes, uniques = pd.factorize(multi, sort=False)
            raw_codes[usable_index] = usable_codes.astype(np.int64)
            raw_keys = [tuple(str(value) for value in key) for key in uniques.tolist()]
        else:
            raw_keys = []
        raw_counts = np.bincount(
            raw_codes[raw_codes >= 0], minlength=len(raw_keys)
        ).astype(np.int64)
        requested = None if perturbations is None else {str(x) for x in perturbations}
        # Match a requested label against any grouping column, not just the first,
        # so that `groupby=["batch", "perturbation"]` filters on the perturbation
        # and every combination containing it is preserved.
        keep_raw = [
            index
            for index, key in enumerate(raw_keys)
            if raw_counts[index] >= min_cells
            and (requested is None or not requested.isdisjoint(key))
        ]
        if requested is not None:
            observed_values = {value for key in raw_keys for value in key}
            missing = sorted(requested - observed_values)
            if missing:
                raise ValueError(f"Requested perturbation(s) not found: {missing[:5]}")
        keys = [raw_keys[index] for index in keep_raw]
        counts = raw_counts[keep_raw]
        raw_to_output = np.full(len(raw_keys), -1, dtype=np.int64)
        raw_to_output[keep_raw] = np.arange(len(keep_raw), dtype=np.int64)
        group_codes = np.full(backed.n_obs, -1, dtype=np.int64)
        valid_raw = raw_codes >= 0
        group_codes[valid_raw] = raw_to_output[raw_codes[valid_raw]]

        multiplicity = np.zeros(backed.n_obs, dtype=np.int64)
        selected_cells = np.flatnonzero(group_codes >= 0)
        if bootstrap_size is None:
            multiplicity[selected_cells] = 1
            aggregated_counts = counts.copy()
        else:
            order = selected_cells[np.argsort(group_codes[selected_cells], kind="stable")]
            ordered_codes = group_codes[order]
            boundaries = np.searchsorted(
                ordered_codes, np.arange(len(keys) + 1), side="left"
            )
            for group_index, key in enumerate(keys):
                members = order[boundaries[group_index] : boundaries[group_index + 1]]
                rng = np.random.default_rng(_group_seed(int(random_state), key))
                draws = rng.choice(members, size=bootstrap_size, replace=True)
                unique_draws, draw_counts = np.unique(draws, return_counts=True)
                multiplicity[unique_draws] = draw_counts
            aggregated_counts = np.full(len(keys), bootstrap_size, dtype=np.int64)

        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(
                backed.n_obs,
                backed.n_vars,
                available_memory_gb=memory_limit_gb,
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        all_integer = True
        context = "X" if layer is None else f"layer '{layer}'"
        for start in range(0, backed.n_obs, chunk_size):
            end = min(start + chunk_size, backed.n_obs)
            local = multiplicity[start:end] > 0
            if not local.any():
                continue
            block = _matrix_block(matrix, start, end)[local]
            all_integer &= _validate_expression_block(block, context=context)
        if method == "sum" and not all_integer:
            raise ValueError(
                "method='sum' requires raw non-negative integer counts; the selected "
                f"{context} contains continuous values. Supply a raw-count layer."
            )
        input_scale = "counts" if all_integer else "log1p"
        if method == "mean_log1p" and all_integer and int(verbose) >= 1:
            print("[cx] pb.aggregate: detected count data; applying log1p before averaging")

        metadata = {
            "schema_version": 1,
            "is_pseudobulk": True,
            "groupby": columns,
            "method": method,
            "source_layer": "X" if layer is None else str(layer),
            "input_scale": input_scale,
            "min_cells": int(min_cells),
            "sampling": "none" if bootstrap_size is None else "bootstrap_with_replacement",
            "bootstrap_size": -1 if bootstrap_size is None else int(bootstrap_size),
            "random_state": int(random_state),
            "n_input_cells": int(backed.n_obs),
            "n_profiles": int(len(keys)),
            "source_path": str(path.resolve()),
            "source_mtime_ns": int(path.stat().st_mtime_ns),
        }
        if resolved_output.exists() and not force:
            existing = ad.read_h5ad(resolved_output, backed="r")
            try:
                existing_meta = existing.uns.get("crispyx_pseudobulk", {})
                if all(
                    _metadata_value_equal(existing_meta.get(key), value)
                    for key, value in metadata.items()
                ):
                    if int(verbose) >= 1:
                        print(f"[cx] Loading existing result: {resolved_output}")
                        print("[cx] Pass force=True to rerun the analysis.")
                    return AnnData(resolved_output)
            finally:
                existing.file.close()

        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        warn_if_disk_space_low(
            estimate_bytes(max(len(keys), 1), backed.n_vars), tempfile.gettempdir(),
            context="pb.aggregate accumulator",
        )
        warn_if_disk_space_low(
            estimate_bytes(max(len(keys), 1), backed.n_vars, overhead=1.10), resolved_output,
            context="pb.aggregate",
        )
        with tempfile.TemporaryDirectory(prefix="cx_pseudobulk_") as tmpdir:
            sums_path = Path(tmpdir) / "profiles.dat"
            profiles = np.memmap(
                sums_path,
                dtype=np.float64,
                mode="w+",
                shape=(max(len(keys), 1), backed.n_vars),
            )
            profiles.fill(0)
            for start in range(0, backed.n_obs, chunk_size):
                end = min(start + chunk_size, backed.n_obs)
                local_codes = group_codes[start:end]
                local_mult = multiplicity[start:end]
                local = (local_codes >= 0) & (local_mult > 0)
                if not local.any():
                    continue
                block = _densify_block(_matrix_block(matrix, start, end)[local])
                if method == "mean_log1p" and all_integer:
                    np.log1p(block, out=block)
                codes = local_codes[local]
                weights = local_mult[local].astype(np.float64)
                unique_codes, inverse = np.unique(codes, return_inverse=True)
                indicator = sparse.csr_matrix(
                    (weights, (inverse, np.arange(codes.size))),
                    shape=(unique_codes.size, codes.size),
                )
                profiles[unique_codes] = profiles[unique_codes] + indicator @ block
            if method == "mean_log1p" and len(keys):
                profiles[: len(keys)] /= aggregated_counts[:, None]

            obs = pd.DataFrame(
                {column: [key[i] for key in keys] for i, column in enumerate(columns)},
                index=pd.Index(
                    [f"pseudobulk_{i:08d}" for i in range(len(keys))],
                    name="pseudobulk_id",
                ),
            )
            obs["n_cells"] = counts
            obs["n_cells_aggregated"] = aggregated_counts
            obs["sampled_with_replacement"] = bootstrap_size is not None
            var = backed.var.copy()
            var.index = pd.Index(gene_symbols, name=backed.var_names.name)
            result = ad.AnnData(np.asarray(profiles[: len(keys)]), obs=obs, var=var)
            result.uns["crispyx_pseudobulk"] = metadata
            result.write(resolved_output)
            profiles._mmap.close()  # type: ignore[attr-defined]
    finally:
        backed.file.close()

    if int(verbose) >= 1:
        print(
            f"[cx] pb.aggregate: {len(keys)} profiles × {len(gene_symbols)} genes"
        )
        print(f"[cx] pb.aggregate: Saving → {resolved_output}")
    return AnnData(resolved_output)


def _has_pseudobulk_marker(path: Path) -> bool:
    backed = read_backed(path)
    try:
        marker = backed.uns.get("crispyx_pseudobulk", {})
        return bool(marker.get("is_pseudobulk", False))
    finally:
        backed.file.close()


def compute_pseudobulk_effects(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    batch_column: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    aggregate_batches: bool = False,
    method: PseudobulkMethod = "mean_log1p",
    layer: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells: int = 5,
    bootstrap_size: int | None = None,
    random_state: int = 0,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    bulk_output_path: str | Path | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: int | bool = False,
    force: bool = False,
) -> AnnData:
    """Compute target-minus-control effects from batch-level pseudo-bulk profiles.

    ``data`` may be a file created by :func:`aggregate_pseudobulk` or a
    cell-level h5ad file. Cell-level input is aggregated first using
    ``[perturbation_column, batch_column]``. By default, one effect is retained
    per perturbation and batch. Set ``aggregate_batches=True`` to combine
    within-batch effects using harmonic target/control cell-count weights.
    """
    if groupby is not None and perturbation_column is not None:
        raise TypeError(
            "compute_pseudobulk_effects() received both 'perturbation_column' "
            "and 'groupby'; pass only one."
        )
    perturbation_column = groupby if groupby is not None else perturbation_column
    if perturbation_column is None:
        raise TypeError(
            "compute_pseudobulk_effects() requires 'perturbation_column' or 'groupby'."
        )
    if reference is not None and control_label is not None:
        raise TypeError(
            "compute_pseudobulk_effects() received both 'control_label' and "
            "'reference'; pass only one."
        )
    control_label = reference if reference is not None else control_label
    source_path = resolve_data_path(data)
    resolved_output = resolve_output_path(
        source_path,
        suffix="pseudobulk_effects_by_batch" if not aggregate_batches else "pseudobulk_effects",
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
    )
    input_is_bulk = _has_pseudobulk_marker(source_path)

    def _run_from_bulk(bulk_path: Path) -> AnnData:
        bulk = read_backed(bulk_path)
        try:
            marker = bulk.uns.get("crispyx_pseudobulk", {})
            # This function contrasts on whatever scale it is given and never
            # normalises library size. Raw counts therefore produce an effect that
            # is confounded by sequencing depth -- a plausible-looking but wrong
            # answer, with nothing to signal it. aggregate_pseudobulk records the
            # scale it detected, so the mistake is detectable here for both a saved
            # artifact and cell-level input aggregated a moment ago.
            if str(marker.get("input_scale", "")) == "counts":
                warnings.warn(
                    "The input to compute_pseudobulk_effects() looks like raw counts, "
                    "and this function does not normalise library size, so the effect "
                    "will be confounded by sequencing depth. Normalise first with "
                    "cx.pp.normalize_total_log1p(), or use cx.pb.normalized_effects(), "
                    "which normalises internally.",
                    UserWarning,
                    stacklevel=3,
                )
            marker_groupby = [str(x) for x in np.asarray(marker.get("groupby", [])).tolist()]
            if perturbation_column not in bulk.obs.columns:
                raise KeyError(
                    f"Perturbation column '{perturbation_column}' is absent from the pseudo-bulk obs."
                )
            local_batch_column = batch_column
            if local_batch_column is None:
                other_columns = [column for column in marker_groupby if column != perturbation_column]
                if len(other_columns) == 1:
                    local_batch_column = other_columns[0]
                else:
                    raise TypeError(
                        "batch_column is required unless it can be inferred from exactly "
                        "two pseudo-bulk grouping columns."
                    )
            if local_batch_column not in bulk.obs.columns:
                raise KeyError(
                    f"Batch column '{local_batch_column}' is absent from the pseudo-bulk obs."
                )

            labels = bulk.obs[perturbation_column].astype(str).to_numpy()
            batches = bulk.obs[local_batch_column].astype(str).to_numpy()
            resolved_control = resolve_control_label(labels, control_label)
            requested = None if perturbations is None else {str(x) for x in perturbations}
            control_by_batch: dict[str, int] = {}
            for index in np.flatnonzero(labels == resolved_control):
                batch = batches[index]
                if batch in control_by_batch:
                    raise ValueError(
                        f"Pseudo-bulk input contains multiple control profiles for batch '{batch}'."
                    )
                control_by_batch[batch] = int(index)
            target_rows: list[int] = []
            control_rows: list[int] = []
            missing_batches: set[str] = set()
            for index, (label, batch) in enumerate(zip(labels, batches)):
                if label == resolved_control or (requested is not None and label not in requested):
                    continue
                control_index = control_by_batch.get(batch)
                if control_index is None:
                    missing_batches.add(batch)
                    continue
                target_rows.append(index)
                control_rows.append(control_index)
            if missing_batches:
                warnings.warn(
                    "Dropping target profiles without a same-batch control. "
                    f"Affected batches: {sorted(missing_batches)[:5]}",
                    stacklevel=2,
                )

            target_rows_arr = np.asarray(target_rows, dtype=np.int64)
            control_rows_arr = np.asarray(control_rows, dtype=np.int64)
            target_labels = labels[target_rows_arr]
            target_batches = batches[target_rows_arr]
            n_cells = (
                bulk.obs["n_cells"].to_numpy(dtype=np.int64)
                if "n_cells" in bulk.obs
                else np.ones(bulk.n_obs, dtype=np.int64)
            )
            n_cells_aggregated = (
                bulk.obs["n_cells_aggregated"].to_numpy(dtype=np.int64)
                if "n_cells_aggregated" in bulk.obs
                else n_cells
            )
            target_counts = n_cells[target_rows_arr]
            control_counts = n_cells[control_rows_arr]
            target_aggregated_counts = n_cells_aggregated[target_rows_arr]
            control_aggregated_counts = n_cells_aggregated[control_rows_arr]
            harmonic = (
                target_aggregated_counts
                * control_aggregated_counts
                / (target_aggregated_counts + control_aggregated_counts)
            )

            effect_metadata = {
                "schema_version": 1,
                "source_is_pseudobulk": bool(input_is_bulk),
                "source_path": str(source_path.resolve()),
                "source_mtime_ns": int(source_path.stat().st_mtime_ns),
                "perturbation_column": str(perturbation_column),
                "batch_column": str(local_batch_column),
                "reference": str(resolved_control),
                "effect": "target_minus_reference",
                "aggregate_batches": bool(aggregate_batches),
                "method": str(marker.get("method", method)),
                "n_missing_control_batches": int(len(missing_batches)),
            }
            if resolved_output.exists() and not force:
                existing = ad.read_h5ad(resolved_output, backed="r")
                try:
                    existing_meta = existing.uns.get("crispyx_pseudobulk_effects", {})
                    if all(
                        _metadata_value_equal(existing_meta.get(key), value)
                        for key, value in effect_metadata.items()
                    ):
                        if int(verbose) >= 1:
                            print(f"[cx] Loading existing result: {resolved_output}")
                            print("[cx] Pass force=True to rerun the analysis.")
                        return AnnData(resolved_output)
                finally:
                    existing.file.close()

            if aggregate_batches:
                output_labels = list(dict.fromkeys(target_labels.tolist()))
                label_lookup = {label: i for i, label in enumerate(output_labels)}
                output_codes = np.asarray([label_lookup[x] for x in target_labels], dtype=np.int64)
                n_output = len(output_labels)
            else:
                output_labels = target_labels.tolist()
                output_codes = np.arange(len(target_rows), dtype=np.int64)
                n_output = len(target_rows)

            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            warn_if_disk_space_low(
                estimate_bytes(max(n_output, 1), bulk.n_vars, 3), tempfile.gettempdir(),
                context="pb.effects accumulator",
            )
            with tempfile.TemporaryDirectory(prefix="cx_pb_effects_") as tmpdir:
                shape = (max(n_output, 1), bulk.n_vars)
                effect_mm = np.memmap(Path(tmpdir) / "effect.dat", dtype=np.float64, mode="w+", shape=shape)
                target_mm = np.memmap(Path(tmpdir) / "target.dat", dtype=np.float64, mode="w+", shape=shape)
                reference_mm = np.memmap(Path(tmpdir) / "reference.dat", dtype=np.float64, mode="w+", shape=shape)
                effect_mm.fill(np.nan)
                target_mm.fill(np.nan)
                reference_mm.fill(np.nan)
                gene_chunk = min(512, max(bulk.n_vars, 1))
                for start in range(0, bulk.n_vars, gene_chunk):
                    end = min(start + gene_chunk, bulk.n_vars)
                    source = np.asarray(bulk.X[:, start:end], dtype=np.float64)
                    target_values = source[target_rows_arr]
                    reference_values = source[control_rows_arr]
                    if aggregate_batches:
                        denominator = np.bincount(
                            output_codes, weights=harmonic, minlength=n_output
                        )
                        target_num = np.zeros((n_output, end - start), dtype=np.float64)
                        reference_num = np.zeros_like(target_num)
                        np.add.at(target_num, output_codes, harmonic[:, None] * target_values)
                        np.add.at(reference_num, output_codes, harmonic[:, None] * reference_values)
                        target_chunk = target_num / denominator[:, None]
                        reference_chunk = reference_num / denominator[:, None]
                    else:
                        target_chunk = target_values
                        reference_chunk = reference_values
                    target_mm[:n_output, start:end] = target_chunk
                    reference_mm[:n_output, start:end] = reference_chunk
                    effect_mm[:n_output, start:end] = target_chunk - reference_chunk

                if aggregate_batches:
                    n_batches = np.bincount(output_codes, minlength=n_output)
                    obs = pd.DataFrame(
                        {
                            perturbation_column: output_labels,
                            "n_batches_used": n_batches,
                        },
                        index=pd.Index(output_labels, name="perturbation"),
                    )
                else:
                    obs = pd.DataFrame(
                        {
                            perturbation_column: output_labels,
                            local_batch_column: target_batches,
                            "n_cells_target": target_counts,
                            "n_cells_reference": control_counts,
                            "n_cells_aggregated_target": target_aggregated_counts,
                            "n_cells_aggregated_reference": control_aggregated_counts,
                        },
                        index=pd.Index(
                            [f"effect_{i:08d}" for i in range(n_output)],
                            name="effect_id",
                        ),
                    )
                var = bulk.var.copy()
                result = ad.AnnData(np.asarray(effect_mm[:n_output]), obs=obs, var=var)
                result.layers["target_profile"] = np.asarray(target_mm[:n_output])
                result.layers["reference_profile"] = np.asarray(reference_mm[:n_output])
                result.uns["crispyx_pseudobulk_effects"] = effect_metadata
                result.write(resolved_output)
                effect_mm._mmap.close()  # type: ignore[attr-defined]
                target_mm._mmap.close()  # type: ignore[attr-defined]
                reference_mm._mmap.close()  # type: ignore[attr-defined]
        finally:
            bulk.file.close()
        if int(verbose) >= 1:
            print(f"[cx] pb.effects: Saving → {resolved_output}")
        return AnnData(resolved_output)

    if input_is_bulk:
        return _run_from_bulk(source_path)
    if batch_column is None:
        raise TypeError("batch_column is required for cell-level input")
    if bulk_output_path is not None:
        bulk_result = aggregate_pseudobulk(
            source_path,
            groupby=[perturbation_column, batch_column],
            method=method,
            layer=layer,
            gene_name_column=gene_name_column,
            perturbations=None,
            min_cells=min_cells,
            bootstrap_size=bootstrap_size,
            random_state=random_state,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            output_path=bulk_output_path,
            verbose=verbose,
            force=force,
        )
        bulk_result.close()
        return _run_from_bulk(Path(bulk_output_path))
    with tempfile.TemporaryDirectory(prefix="cx_pb_intermediate_") as tmpdir:
        temporary_bulk = Path(tmpdir) / "pseudobulk.h5ad"
        bulk_result = aggregate_pseudobulk(
            source_path,
            groupby=[perturbation_column, batch_column],
            method=method,
            layer=layer,
            gene_name_column=gene_name_column,
            min_cells=min_cells,
            bootstrap_size=bootstrap_size,
            random_state=random_state,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            output_path=temporary_bulk,
            verbose=verbose,
            force=True,
        )
        bulk_result.close()
        return _run_from_bulk(temporary_bulk)


# ---------------------------------------------------------------------------
# Disk-usage resolvers for crispyx.estimate_disk_usage() (see _preflight.py).
#
# Each resolver reads only cheap obs/uns metadata (backed mode; never streams
# X) to reproduce the same group/batch counts the real function computes in
# its own preamble, then feeds them into the same _disk.estimate_bytes
# formulas used by the automatic warnings above. This is intentionally a
# second read of the counting logic rather than a shared call, since the
# real functions already have these values in scope from their own preamble
# and re-deriving them via a second backed open would be wasted I/O; the
# requirement is one implementation of the counting *logic*, reproduced
# here, not one call site.
# ---------------------------------------------------------------------------


def _estimate_shape_for_normalized_effects(
    path: Path,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    batch_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    **_ignored,
) -> dict[str, float]:
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="compute_normalized_effects",
    )
    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        n_genes = backed.n_vars
        if batch_column is None:
            return {"output": estimate_bytes(len(candidates), n_genes, 2, overhead=1.10)}
        batch_labels = backed.obs[batch_column].astype(str).to_numpy()
        pert_code = pd.Index(candidates).get_indexer(pd.Index(labels))
        batch_code, batch_uniques = pd.factorize(batch_labels, sort=False)
        n_batches = len(batch_uniques)
        cand_cell = pert_code >= 0
        pair_key_all = pert_code[cand_cell].astype(np.int64) * n_batches + batch_code[cand_cell]
        n_pairs = int(np.unique(pair_key_all).shape[0]) if cand_cell.any() else 0
        return {
            "tempdir": estimate_bytes(max(n_pairs, 1), n_genes),
            "output": estimate_bytes(len(candidates), n_genes, 3, overhead=1.10),
        }
    finally:
        backed.file.close()


def _estimate_shape_for_aggregate_pseudobulk(
    path: Path,
    *,
    groupby: str | Sequence[str],
    min_cells: int = 5,
    perturbations: Iterable[str] | None = None,
    **_ignored,
) -> dict[str, float]:
    columns = _normalise_groupby(groupby)
    backed = read_backed(path)
    try:
        n_vars = backed.n_vars
        grouping = backed.obs[columns]
        usable = ~grouping.isna().any(axis=1).to_numpy()
        if not usable.any():
            return {"tempdir": estimate_bytes(1, n_vars), "output": estimate_bytes(1, n_vars, overhead=1.10)}
        multi = pd.MultiIndex.from_frame(grouping.iloc[np.flatnonzero(usable)].astype(str))
        codes, uniques = pd.factorize(multi, sort=False)
        counts = np.bincount(codes, minlength=len(uniques))
        raw_keys = [tuple(str(value) for value in key) for key in uniques.tolist()]
        requested = None if perturbations is None else {str(x) for x in perturbations}
        n_keep = sum(
            1
            for index, key in enumerate(raw_keys)
            if counts[index] >= min_cells and (requested is None or not requested.isdisjoint(key))
        )
        return {
            "tempdir": estimate_bytes(max(n_keep, 1), n_vars),
            "output": estimate_bytes(max(n_keep, 1), n_vars, overhead=1.10),
        }
    finally:
        backed.file.close()


def _estimate_shape_for_pseudobulk_effects(
    path: Path,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    batch_column: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    aggregate_batches: bool = False,
    min_cells: int = 5,
    perturbations: Iterable[str] | None = None,
    **_ignored,
) -> dict[str, float]:
    perturbation_column = groupby if groupby is not None else perturbation_column
    control_label = reference if reference is not None else control_label
    if perturbation_column is None:
        raise TypeError(
            "estimate_disk_usage('compute_pseudobulk_effects', ...) requires "
            "'perturbation_column' or 'groupby'."
        )
    backed = read_backed(path)
    try:
        n_vars = backed.n_vars
        is_bulk = bool(backed.uns.get("crispyx_pseudobulk", {}).get("is_pseudobulk", False))
        if is_bulk:
            labels = backed.obs[perturbation_column].astype(str).to_numpy()
            resolved_control = resolve_control_label(labels, control_label)
            target_mask = labels != resolved_control
            n_target = int(target_mask.sum())
            n_output = len(set(labels[target_mask].tolist())) if aggregate_batches else n_target
            return {"tempdir": estimate_bytes(max(n_output, 1), n_vars, 3)}
    finally:
        backed.file.close()
    if batch_column is None:
        raise TypeError(
            "estimate_disk_usage('compute_pseudobulk_effects', ...) requires "
            "'batch_column' for cell-level input."
        )
    # Cell-level input runs aggregate_pseudobulk into an intermediate file
    # first (usually the dominant cost), then a smaller effects pass.
    return _estimate_shape_for_aggregate_pseudobulk(
        path,
        groupby=[perturbation_column, batch_column],
        min_cells=min_cells,
        perturbations=perturbations,
    )


# docs/api.rst documents this module with ``automodule :members:``, which follows
# ``__all__``. compute_average_log_expression and compute_pseudobulk_expression are
# deliberately absent: they remain importable for backward compatibility but must not
# appear in the rendered API, so that only compute_normalized_effects is discoverable.
__all__ = [
    "NormalizedEffectMethod",
    "PseudobulkMethod",
    "aggregate_pseudobulk",
    "compute_normalized_effects",
    "compute_pseudobulk_effects",
]
