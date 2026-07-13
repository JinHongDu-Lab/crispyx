"""Pseudo-bulk effect size estimators operating directly on ``.h5ad`` files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse

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
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    verbose: int | bool = False,
) -> AnnData:
    """Compute average log-normalised expression per perturbation relative to control.

    For each perturbation group, computes the per-gene mean of log1p-normalised
    expression and stores the difference relative to the control group as the
    effect size.

    Parameters
    ----------
    data
        Path to an h5ad file, or a backed/in-memory AnnData object.
    perturbation_column
        Column in ``adata.obs`` that identifies perturbation groups.
    control_label
        Label of the control group.  If ``None``, inferred from common
        patterns (``'non-targeting'``, ``'control'``, etc.).
    gene_name_column
        Column in ``adata.var`` with gene symbols.  If ``None``, uses
        ``adata.var_names``.
    perturbations
        Subset of perturbation labels to include.  If ``None``, all
        non-control groups are processed.
    batch_column
        Column in ``adata.obs`` identifying the batch of each cell.  When
        provided, effects are computed within each batch and combined with
        harmonic-count weights (``w_b = n_pert_b * n_ctrl_b / (n_pert_b +
        n_ctrl_b)``), removing batch-driven confounding.  The per-``(perturbation,
        batch)`` accumulator is spilled to a disk-backed ``np.memmap`` so peak
        memory stays bounded regardless of the number of batches.  When
        ``None`` (default), a single pooled effect is computed.
    chunk_size
        Number of cells to process per chunk.  If ``None`` (default),
        auto-determined from the dataset shape and the available memory budget
        (see ``memory_limit_gb``).
    memory_limit_gb
        Soft memory budget in gigabytes used to size the streaming cell chunk.
        When ``None`` (default), the available system memory is auto-detected
        via ``psutil``.  Passing a value (e.g. ``memory_limit_gb=128``) caps the
        budget for SLURM / cgroup-constrained environments.  Ignored when an
        explicit ``chunk_size`` is given.  Only the chunk size is affected;
        computed values are identical regardless of the budget.
    data_name
        Custom stem for the output filename.  If ``None``, the input file
        stem is used with a ``_cx_avg_log_effects`` suffix.
    output_path
        Exact path for the output h5ad file.  When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for the output file.  Defaults to the input file's
        directory.  *Deprecated* – use ``output_path`` instead.  Will be
        removed in the next major version.
    verbose
        Verbosity level.  ``0`` / ``False`` is silent; ``1`` / ``True``
        prints a summary line.

    Returns
    -------
    AnnData
        On-disk AnnData where ``X`` contains the effect-size matrix
        (perturbation mean minus control mean in log-normalised space),
        ``layers['perturbation_mean']`` contains per-perturbation means,
        and ``uns['control_mean']`` contains the control mean vector.
        When ``batch_column`` is set, ``X`` holds the batch-corrected effect
        (harmonic-count weighted average of within-batch differences),
        ``layers['perturbation_mean']`` holds the **batch-corrected**
        per-perturbation mean, ``layers['control_mean_matched']`` holds the
        per-perturbation weight-matched control reference (so
        ``X = perturbation_mean - control_mean_matched``),
        ``uns['control_mean']`` retains the pooled control mean, and
        ``uns['batch_column']`` / ``uns['batch_ids']`` record the batch column
        name and the batch labels encountered.
    """

    path = resolve_data_path(data)
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_average_log_expression: Reading {path}")
    backed = read_backed(path)
    use_batch = batch_column is not None
    effect_matrix_np = np.empty((0, 0), dtype=np.float64)
    pert_mean_corrected = np.empty((0, 0), dtype=np.float64)
    ctrl_mean_matched = np.empty((0, 0), dtype=np.float64)
    pooled_ctrl_sum = np.empty(0, dtype=np.float64)
    pooled_ctrl_count = 0
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    try:
        # Calculate adaptive chunk_size if not provided
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(
                backed.n_obs, backed.n_vars, available_memory_gb=memory_limit_gb,
            )
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        use_batch = batch_column is not None
        batch_ids: list[str] = []
        if use_batch:
            if batch_column not in backed.obs.columns:
                raise KeyError(
                    f"Batch column '{batch_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
                )
            batch_labels = backed.obs[batch_column].astype(str).to_numpy()

            def _block_fn(block) -> np.ndarray:
                normalised, _ = normalize_total_block(block)
                return np.log1p(_densify_block(normalised))

            def _mean_transform(agg: np.ndarray, n: np.ndarray) -> np.ndarray:
                return agg / n

            (
                effect_matrix_np,
                pert_mean_corrected,
                ctrl_mean_matched,
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
                transform=_mean_transform,
            )
        else:
            groups = [control_label] + candidates
            sums = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
            counts = {label: 0 for label in groups}
            for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
                slice_labels = labels[slc]
                normalised_block, _ = normalize_total_block(block)
                log_block = np.log1p(normalised_block)
                for label in groups:
                    mask = slice_labels == label
                    if not np.any(mask):
                        continue
                    sums[label] += log_block[mask].sum(axis=0)
                    counts[label] += int(mask.sum())
    finally:
        backed.file.close()

    control_mean_matched = None
    if use_batch:
        if pooled_ctrl_count == 0:
            raise ValueError("Control group contains no cells")
        control_mean = pooled_ctrl_sum / pooled_ctrl_count
        pert_means = list(pert_mean_corrected)
        control_mean_matched = ctrl_mean_matched
        effect_matrix = list(effect_matrix_np)
    else:
        if counts[control_label] == 0:
            raise ValueError("Control group contains no cells")
        control_mean = sums[control_label] / counts[control_label]
        effect_matrix = []
        pert_means = []
        for label in candidates:
            if counts[label] == 0:
                raise ValueError(f"Perturbation '{label}' contains no cells")
            mean = sums[label] / counts[label]
            pert_means.append(mean)
            effect_matrix.append(mean - control_mean)

    if not effect_matrix:
        obs_index = pd.Index([], name="perturbation")
        adata = ad.AnnData(
            np.zeros((0, gene_symbols.shape[0])),
            obs=pd.DataFrame(index=obs_index),
            var=pd.DataFrame(index=gene_symbols),
        )
        output_path = resolve_output_path(
            path, suffix="avg_log_effects", output_dir=output_dir, data_name=data_name,
            output_path=output_path,
        )
        if int(verbose) >= 1:
            print(f"[cx] pb.compute_average_log_expression: 0 perturbations × {gene_symbols.shape[0]} genes")
            print(f"[cx] pb.compute_average_log_expression: Saving → {output_path}")
        adata.write(output_path)
        return AnnData(output_path)

    effect_matrix_np = np.vstack(effect_matrix)
    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix_np, obs=obs, var=var)
    adata.layers["perturbation_mean"] = np.vstack(pert_means)
    adata.uns["control_mean"] = control_mean
    if use_batch:
        adata.layers["control_mean_matched"] = np.asarray(control_mean_matched)
        adata.uns["batch_column"] = str(batch_column)
        adata.uns["batch_ids"] = np.asarray(batch_ids, dtype=object)
    output_path = resolve_output_path(
        path, suffix="avg_log_effects", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_average_log_expression: {len(candidates)} perturbations × {len(gene_symbols)} genes")
        print(f"[cx] pb.compute_average_log_expression: Saving → {output_path}")
    adata.write(output_path)
    return AnnData(output_path)


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
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    verbose: int | bool = False,
) -> AnnData:
    """Compute pseudo-bulk log-fold changes relative to control.

    Aggregates normalised counts per perturbation group into a pseudo-bulk
    profile (sum divided by cell count), applies log1p scaling with a
    ``baseline_count`` offset, and stores the difference relative to the
    control group as the log-fold change effect size.

    Parameters
    ----------
    data
        Path to an h5ad file, or a backed/in-memory AnnData object.
    perturbation_column
        Column in ``adata.obs`` that identifies perturbation groups.
    control_label
        Label of the control group.  If ``None``, inferred from common
        patterns (``'non-targeting'``, ``'control'``, etc.).
    gene_name_column
        Column in ``adata.var`` with gene symbols.  If ``None``, uses
        ``adata.var_names``.
    perturbations
        Subset of perturbation labels to include.  If ``None``, all
        non-control groups are processed.
    batch_column
        Column in ``adata.obs`` identifying the batch of each cell.  When
        provided, log-fold changes are computed within each batch and combined
        with harmonic-count weights (``w_b = n_pert_b * n_ctrl_b / (n_pert_b +
        n_ctrl_b)``), removing batch-driven confounding.  The per-``(perturbation,
        batch)`` accumulator is spilled to a disk-backed ``np.memmap`` so peak
        memory stays bounded regardless of the number of batches.  When
        ``None`` (default), a single pooled log-fold change is computed.
    baseline_count
        Pseudo-count added before log transformation
        (``log1p(baseline_count * mean_counts)``).  Default ``1.0``.
    chunk_size
        Number of cells to process per chunk.  If ``None`` (default),
        auto-determined from the dataset shape and the available memory budget
        (see ``memory_limit_gb``).
    memory_limit_gb
        Soft memory budget in gigabytes used to size the streaming cell chunk.
        When ``None`` (default), the available system memory is auto-detected
        via ``psutil``.  Passing a value (e.g. ``memory_limit_gb=128``) caps the
        budget for SLURM / cgroup-constrained environments.  Ignored when an
        explicit ``chunk_size`` is given.  Only the chunk size is affected;
        computed values are identical regardless of the budget.
    data_name
        Custom stem for the output filename.  If ``None``, the input file
        stem is used with a ``_cx_pseudobulk_effects`` suffix.
    output_path
        Exact path for the output h5ad file.  When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for the output file.  Defaults to the input file's
        directory.  *Deprecated* – use ``output_path`` instead.  Will be
        removed in the next major version.
    verbose
        Verbosity level.  ``0`` / ``False`` is silent; ``1`` / ``True``
        prints a summary line.

    Returns
    -------
    AnnData
        On-disk AnnData where ``X`` contains the pseudo-bulk log-fold change
        matrix (perturbation pseudo-bulk minus control pseudo-bulk),
        ``layers['perturbation_bulk']`` contains per-perturbation pseudo-bulk
        vectors, ``uns['control_bulk']`` the control pseudo-bulk vector, and
        ``uns['baseline_count']`` the scaling offset used.
        When ``batch_column`` is set, ``X`` holds the batch-corrected log-fold
        change (harmonic-count weighted average of within-batch differences),
        ``layers['perturbation_bulk']`` holds the **batch-corrected**
        per-perturbation pseudo-bulk, ``layers['control_bulk_matched']`` holds
        the per-perturbation weight-matched control reference (so
        ``X = perturbation_bulk - control_bulk_matched``),
        ``uns['control_bulk']`` retains the pooled control pseudo-bulk, and
        ``uns['batch_column']`` / ``uns['batch_ids']`` record the batch
        column name and the batch labels encountered.
    """

    if baseline_count <= 0:
        raise ValueError("baseline_count must be positive")

    path = resolve_data_path(data)
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_pseudobulk_expression: Reading {path}")
    backed = read_backed(path)
    use_batch = batch_column is not None
    effect_matrix_np = np.empty((0, 0), dtype=np.float64)
    pert_mean_corrected = np.empty((0, 0), dtype=np.float64)
    ctrl_mean_matched = np.empty((0, 0), dtype=np.float64)
    pooled_ctrl_sum = np.empty(0, dtype=np.float64)
    pooled_ctrl_count = 0
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    try:
        # Calculate adaptive chunk_size if not provided
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(
                backed.n_obs, backed.n_vars, available_memory_gb=memory_limit_gb,
            )
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        use_batch = batch_column is not None
        batch_ids: list[str] = []
        if use_batch:
            if batch_column not in backed.obs.columns:
                raise KeyError(
                    f"Batch column '{batch_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
                )
            batch_labels = backed.obs[batch_column].astype(str).to_numpy()

            def _block_fn(block) -> np.ndarray:
                normalised, _ = normalize_total_block(block)
                return _densify_block(normalised)

            def _bulk_transform(agg: np.ndarray, n: np.ndarray) -> np.ndarray:
                return np.log1p(baseline_count * agg / n)

            (
                effect_matrix_np,
                pert_mean_corrected,
                ctrl_mean_matched,
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
                transform=_bulk_transform,
            )
        else:
            groups = [control_label] + candidates
            sums = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
            counts = {label: 0 for label in groups}
            for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
                slice_labels = labels[slc]
                normalised_block, _ = normalize_total_block(block)
                for label in groups:
                    mask = slice_labels == label
                    if not np.any(mask):
                        continue
                    sums[label] += normalised_block[mask].sum(axis=0)
                    counts[label] += int(mask.sum())
    finally:
        backed.file.close()

    control_bulk_matched = None
    if use_batch:
        if pooled_ctrl_count == 0:
            raise ValueError("Control group contains no cells")
        control_bulk = np.log1p(baseline_count * pooled_ctrl_sum / pooled_ctrl_count)
        pert_bulks = list(pert_mean_corrected)
        control_bulk_matched = ctrl_mean_matched
        effect_matrix = list(effect_matrix_np)
    else:
        if counts[control_label] == 0:
            raise ValueError("Control group contains no cells")
        control_bulk = np.log1p(baseline_count * sums[control_label] / counts[control_label])
        effect_matrix = []
        pert_bulks = []
        for label in candidates:
            if counts[label] == 0:
                raise ValueError(f"Perturbation '{label}' contains no cells")
            bulk = np.log1p(baseline_count * sums[label] / counts[label])
            pert_bulks.append(bulk)
            effect_matrix.append(bulk - control_bulk)

    if not effect_matrix:
        obs_index = pd.Index([], name="perturbation")
        adata = ad.AnnData(
            np.zeros((0, gene_symbols.shape[0])),
            obs=pd.DataFrame(index=obs_index),
            var=pd.DataFrame(index=gene_symbols),
        )
        adata.uns["control_bulk"] = control_bulk
        adata.uns["baseline_count"] = float(baseline_count)
        output_path = resolve_output_path(
            path, suffix="pseudobulk_effects", output_dir=output_dir, data_name=data_name,
            output_path=output_path,
        )
        if int(verbose) >= 1:
            print(f"[cx] pb.compute_pseudobulk_expression: 0 perturbations × {gene_symbols.shape[0]} genes")
            print(f"[cx] pb.compute_pseudobulk_expression: Saving → {output_path}")
        adata.write(output_path)
        return AnnData(output_path)

    effect_matrix_np = np.vstack(effect_matrix)
    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix_np, obs=obs, var=var)
    adata.layers["perturbation_bulk"] = np.vstack(pert_bulks)
    adata.uns["control_bulk"] = control_bulk
    adata.uns["baseline_count"] = float(baseline_count)
    if use_batch:
        adata.layers["control_bulk_matched"] = np.asarray(control_bulk_matched)
        adata.uns["batch_column"] = str(batch_column)
        adata.uns["batch_ids"] = np.asarray(batch_ids, dtype=object)
    output_path = resolve_output_path(
        path, suffix="pseudobulk_effects", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_pseudobulk_expression: {len(candidates)} perturbations × {len(gene_symbols)} genes")
        print(f"[cx] pb.compute_pseudobulk_expression: Saving → {output_path}")
    adata.write(output_path)
    return AnnData(output_path)

