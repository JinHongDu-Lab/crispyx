"""Pseudo-bulk effect size estimators operating directly on ``.h5ad`` files."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Iterable, Literal, NamedTuple, Sequence, get_args

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from . import _messages
from ._checkpoint import _create_progress_context
from ._provenance import KEY as _PROVENANCE_KEY, call_fingerprint, reusable, stamp
from ._disk import estimate_bytes, warn_if_disk_space_low
from ._memory import _resolve_memory_limit_bytes
from ._grouping import _group_seed, canonical_call_args, resolve_group_reference_aliases
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
    _replace_on_success,
)


# What aggregate_pseudobulk (and compute_pseudobulk_effects, which aggregates
# through it) computes per profile, and what compute_normalized_effects
# contrasts. Each function validates against its own Literal, so the type
# hint and the accepted values cannot drift apart.
PseudobulkMethod = Literal["mean_log1p", "sum"]
NormalizedEffectMethod = Literal["mean_log1p", "log_mean"]


def _check_method(method: str, allowed, fn_name: str, hint: str = "") -> None:
    if method not in get_args(allowed):
        options = " or ".join(repr(m) for m in get_args(allowed))
        raise ValueError(f"{fn_name}: method must be {options}; received {method!r}.{hint}")


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


class _StratifiedEffects(NamedTuple):
    """What :func:`_streaming_batch_corrected` returns; rows follow ``candidates``."""

    effect: np.ndarray
    perturbation_profile: np.ndarray
    control_profile_matched: np.ndarray
    pooled_control_sum: np.ndarray
    pooled_control_count: int
    batch_ids: list[str]
    standard_error: np.ndarray | None
    standard_error_robust: np.ndarray | None


def _pair_accumulator(n_rows: int, n_genes: int, prefix: str, *, on_disk: bool) -> tuple[np.ndarray, str | None]:
    """A zeroed ``(n_rows, n_genes)`` float64 accumulator: in memory, or in a
    file on disk (whose name is returned, for removal)."""
    if not on_disk:
        return np.zeros((max(n_rows, 1), n_genes), dtype=np.float64), None
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".dat", delete=False)
    tmp.close()
    warn_if_disk_space_low(
        estimate_bytes(max(n_rows, 1), n_genes), tmp.name,
        context="pb batch-corrected accumulator",
    )
    return np.memmap(tmp.name, dtype=np.float64, mode="w+", shape=(max(n_rows, 1), n_genes)), tmp.name


def _group_sums(values: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct ``codes``, the row sums of ``values`` for each, and their counts."""
    unique, inverse = np.unique(codes, return_inverse=True)
    indicator = sparse.csr_matrix(
        (np.ones(codes.shape[0], dtype=np.float64), (inverse, np.arange(codes.shape[0]))),
        shape=(unique.shape[0], codes.shape[0]),
    )
    return unique, indicator @ values, np.bincount(inverse, minlength=unique.shape[0])


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
    standard_errors: bool,
    memory_limit_gb: float | None = None,
) -> _StratifiedEffects:
    """Batch-corrected effects, and optionally their standard errors, in one
    bounded-memory pass over the matrix.

    The per-``(perturbation, batch)`` accumulators -- the only quantities that
    scale with the number of batches, ``pairs x genes`` each -- are kept in
    memory when together they fit in half the memory budget, and are
    otherwise spilled to disk-backed ``np.memmap`` files, so that peak RAM
    stays ``O(chunk_size x n_genes + n_batches x n_genes + n_candidates x
    n_genes)`` however many batches there are. Control per-batch accumulators
    are small and kept in RAM.

    For every ``(perturbation, batch)`` pair with cells on both sides the
    within-batch effect is ``d_b = transform(pert_sum, n_p) -
    transform(ctrl_sum, n_c)``; batches without control cells carry no
    contrast and are skipped. The effect is ``E = sum_b w_b d_b / W`` with
    harmonic-count weights ``w_b = n_p n_c / (n_p + n_c)`` and ``W = sum_b
    w_b``: the estimate of a regression with one intercept per batch and one
    perturbation coefficient. Data without batches is one batch.

    With ``standard_errors`` (``transform`` must then be the mean, ``S / n``),
    two standard errors of ``E`` follow Liu et al. (2026,
    doi:10.64898/2026.10.05.756839):

    * model-based: cells independent within a batch and one within-batch
      variance per arm, the ``w_b``-weighted mean of the batches' population
      variances over batches with at least two cells on that side
      (``s2_p``, ``s2_c``);
      ``SE^2 = (s2_p sum_b w_b^2 / n_p + s2_c sum_b w_b^2 / n_c) / W^2``.
      NaN when no batch has two cells on a side.
    * robust: from the spread of the per-batch effects, with no shared
      variance; ``SE^2 = B / (B - 1) sum_b w_b^2 (d_b - E)^2 / W^2`` over the
      ``B`` contributing batches. NaN when ``B < 2``.

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
    ctrl_sqsums = np.zeros((n_batches, n_genes), dtype=np.float64) if standard_errors else None
    ctrl_counts = np.zeros(n_batches, dtype=np.int64)
    pair_counts = np.zeros(n_pairs, dtype=np.int64)

    n_accumulators = 2 if standard_errors else 1
    on_disk = (
        n_accumulators * max(n_pairs, 1) * n_genes * 8 > 0.5 * _resolve_memory_limit_bytes(memory_limit_gb)
    )
    pair_sums, sums_file = _pair_accumulator(n_pairs, n_genes, "cx_pb_pairsums_", on_disk=on_disk)
    pair_sqsums, sqsums_file = (
        _pair_accumulator(n_pairs, n_genes, "cx_pb_pairsqsums_", on_disk=on_disk)
        if standard_errors else (None, None)
    )
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            acc = block_fn(block)
            bc = batch_code[slc]
            pc = pert_code[slc]
            cm = ctrl_mask_full[slc]

            # Control cells -> per-batch sums (RAM).
            if cm.any():
                ub, sums, n = _group_sums(acc[cm], bc[cm])
                ctrl_sums[ub] += sums
                ctrl_counts[ub] += n
                if standard_errors:
                    ctrl_sqsums[ub] += _group_sums(np.square(acc[cm]), bc[cm])[1]

            # Perturbation cells -> per-(pert, batch) sums (memmap).
            pm = pc >= 0
            if pm.any():
                keys = pc[pm].astype(np.int64) * n_batches + bc[pm]
                slots = np.searchsorted(uniq_pairs, keys)
                us, sums, n = _group_sums(acc[pm], slots)
                pair_sums[us] = pair_sums[us] + sums
                pair_counts[us] += n
                if standard_errors:
                    pair_sqsums[us] = pair_sqsums[us] + _group_sums(np.square(acc[pm]), slots)[1]

        pooled_ctrl_sum = ctrl_sums.sum(axis=0)
        pooled_ctrl_count = int(ctrl_counts.sum())

        # Per-perturbation pooled counts (for the "no cells" guard).
        pooled_pert_counts = np.zeros(n_candidates, dtype=np.int64)
        np.add.at(pooled_pert_counts, pert_of_pair, pair_counts)

        # Weighted numerators for the batch-corrected perturbation expression and
        # the weight-matched control reference; with standard errors, those of
        # the pooled within-batch variances and the sums of w^2/n per arm.
        pert_expr_num = np.zeros((n_candidates, n_genes), dtype=np.float64)
        ctrl_expr_num = np.zeros((n_candidates, n_genes), dtype=np.float64)
        weight_tot = np.zeros(n_candidates, dtype=np.float64)
        if standard_errors:
            pert_var_num = np.zeros((n_candidates, n_genes), dtype=np.float64)
            ctrl_var_num = np.zeros((n_candidates, n_genes), dtype=np.float64)
            pert_var_w = np.zeros(n_candidates, dtype=np.float64)
            ctrl_var_w = np.zeros(n_candidates, dtype=np.float64)
            pert_w2n = np.zeros(n_candidates, dtype=np.float64)
            ctrl_w2n = np.zeros(n_candidates, dtype=np.float64)
            n_strata = np.zeros(n_candidates, dtype=np.int64)

        n_c_pair = ctrl_counts[batch_of_pair]

        def contributing_pairs():
            """Blocks of the pairs with cells on both sides: their perturbation,
            weight, counts and per-batch expressions (and variances)."""
            for start in range(0, n_pairs, 4096):
                end = min(start + 4096, n_pairs)
                n_p = pair_counts[start:end]
                n_c = n_c_pair[start:end]
                valid = (n_p > 0) & (n_c > 0)
                if not valid.any():
                    continue
                n_pv = n_p[valid].astype(np.float64)
                n_cv = n_c[valid].astype(np.float64)
                b_idx = batch_of_pair[start:end][valid]
                Tp = transform(np.asarray(pair_sums[start:end])[valid], n_pv[:, None])
                Tc = transform(ctrl_sums[b_idx], n_cv[:, None])
                if standard_errors:
                    Vp = np.maximum(np.asarray(pair_sqsums[start:end])[valid] / n_pv[:, None] - Tp * Tp, 0.0)
                    Vc = np.maximum(ctrl_sqsums[b_idx] / n_cv[:, None] - Tc * Tc, 0.0)
                else:
                    Vp = Vc = None
                yield (pert_of_pair[start:end][valid], (n_pv * n_cv) / (n_pv + n_cv),
                       n_pv, n_cv, Tp, Tc, Vp, Vc)

        for p_idx, w, n_p, n_c, Tp, Tc, Vp, Vc in contributing_pairs():
            np.add.at(pert_expr_num, p_idx, w[:, None] * Tp)
            np.add.at(ctrl_expr_num, p_idx, w[:, None] * Tc)
            np.add.at(weight_tot, p_idx, w)
            if standard_errors:
                # Batches with one cell on a side estimate no variance there,
                # but still count, with n = 1, in that side's sum of w^2/n.
                wp = np.where(n_p >= 2, w, 0.0)
                wc = np.where(n_c >= 2, w, 0.0)
                np.add.at(pert_var_num, p_idx, wp[:, None] * Vp)
                np.add.at(ctrl_var_num, p_idx, wc[:, None] * Vc)
                np.add.at(pert_var_w, p_idx, wp)
                np.add.at(ctrl_var_w, p_idx, wc)
                np.add.at(pert_w2n, p_idx, w * w / n_p)
                np.add.at(ctrl_w2n, p_idx, w * w / n_c)
                np.add.at(n_strata, p_idx, 1)

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
        del pert_expr_num, ctrl_expr_num

        se = se_robust = None
        if standard_errors:
            with np.errstate(invalid="ignore", divide="ignore"):
                pooled_p = pert_var_num / pert_var_w[:, None]
                pooled_c = ctrl_var_num / ctrl_var_w[:, None]
            del pert_var_num, ctrl_var_num
            se = np.sqrt((pooled_p * pert_w2n[:, None] + pooled_c * ctrl_w2n[:, None]) * inv_w * inv_w)
            del pooled_p, pooled_c
            # The robust SE needs E first: a second pass over the pairs, which
            # avoids the cancellation of sum w^2 d^2 - 2 E sum w^2 d + E^2 sum w^2.
            spread = np.zeros((n_candidates, n_genes), dtype=np.float64)
            for p_idx, w, _, _, Tp, Tc, _, _ in contributing_pairs():
                np.add.at(spread, p_idx, (w * w)[:, None] * np.square(Tp - Tc - effect_matrix[p_idx]))
            with np.errstate(invalid="ignore", divide="ignore"):
                correction = np.where(n_strata >= 2, n_strata / (n_strata - 1.0), np.nan)
            se_robust = np.sqrt(correction[:, None] * spread * inv_w * inv_w)
    finally:
        for accumulator, name in ((pair_sums, sums_file), (pair_sqsums, sqsums_file)):
            if name is not None:
                accumulator._mmap.close()  # type: ignore[attr-defined]
                os.unlink(name)

    return _StratifiedEffects(
        effect_matrix, pert_mean_corrected, ctrl_mean_matched,
        pooled_ctrl_sum, pooled_ctrl_count, batch_ids, se, se_robust,
    )


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
    verbose: int | bool = True,
) -> AnnData:
    """Compute library-size-normalised target-minus-reference effects from cells.

    This is the one-command path from a cell-level ``.h5ad`` to an effect size: it
    normalises each cell's counts to 10,000, aggregates per perturbation, and
    subtracts the reference, in a single streaming pass over the matrix.

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
        cell-count weights, ``w_b = n_tb * n_rb / (n_tb + n_rb)``: the estimate of a
        regression with one intercept per batch and one perturbation coefficient,
        which removes batch-driven confounding. Batches without reference cells
        support no contrast and are skipped. When omitted, a single pooled effect
        is computed. There is no flag to disable the correction: supplying the
        column is the request for it.
    chunk_size
        Cells streamed per chunk. Auto-selected when omitted.
    memory_limit_gb
        Soft memory budget. It sizes the chunk, and decides whether the
        per-(perturbation, batch) accumulators (``pairs x genes`` float64 each;
        two with standard errors) stay in memory, which they do when they fit
        in half of it, or go to temporary files on disk. Results are identical
        regardless.
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

        ``method="mean_log1p"`` also writes two standard errors of the effect
        (Liu et al., 2026, doi:10.64898/2026.10.05.756839), treating data without
        ``batch_column`` as one batch:

        * ``layers['standard_error']``: cells independent within a batch, with one
          within-batch variance per arm pooled over the batches with at least two
          cells on that side (``w_b``-weighted population variances);
          ``SE^2 = (s2_t sum_b w_b^2 / n_tb + s2_r sum_b w_b^2 / n_rb) / W^2``,
          ``W = sum_b w_b``. Without batches this is ``sqrt(v_t / n_t + v_r / n_r)``.
          NaN when no batch has two cells on a side.
        * ``layers['standard_error_robust']``: from the spread of the per-batch
          effects ``d_b`` around the effect ``E``, with no shared variance;
          ``SE^2 = B / (B - 1) sum_b w_b^2 (d_b - E)^2 / W^2`` over the ``B``
          contributing batches. NaN with fewer than two.

    See Also
    --------
    compute_pseudobulk_effects : contrast on the input's own scale; also accepts a
        saved :func:`aggregate_pseudobulk` artifact.
    aggregate_pseudobulk : absolute profiles rather than a contrast.
    """
    log_name = "pb.normalized_effects"
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="compute_normalized_effects",
    )
    _check_method(
        method, NormalizedEffectMethod, "compute_normalized_effects",
        " Use aggregate_pseudobulk for method='sum'.",
    )
    if baseline_count <= 0:
        raise ValueError("baseline_count must be positive")

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

    def _block_fn(block) -> np.ndarray:
        normalised, _ = normalize_total_block(block)
        return _per_cell(_densify_block(normalised))

    def _transform(agg: np.ndarray, n: np.ndarray) -> np.ndarray:
        return _finalise(agg / n)

    path = resolve_data_path(data)
    _messages.print_reading(verbose, log_name, path)
    backed = read_backed(path)
    use_batch = batch_column is not None
    try:
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(
                backed.n_obs, backed.n_vars, available_memory_gb=memory_limit_gb,
            )
            _messages.vprint(verbose, log_name, f"chunk_size={chunk_size} (auto)")
        gene_symbols = pd.Index(ensure_gene_symbol_column(backed, gene_name_column)).astype(str)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in "
                f"adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        if use_batch:
            if batch_column not in backed.obs.columns:
                raise KeyError(
                    f"Batch column '{batch_column}' was not found in adata.obs. "
                    f"Available columns: {list(backed.obs.columns)}"
                )
            batch_labels = backed.obs[batch_column].astype(str).to_numpy()
        else:
            batch_labels = np.zeros(backed.n_obs, dtype=np.int8)  # one batch
        if not (labels == control_label).any():
            raise ValueError("Control group contains no cells")
        result = _streaming_batch_corrected(
            backed,
            labels=labels,
            batch_labels=batch_labels,
            candidates=candidates,
            control_label=control_label,
            n_genes=n_genes,
            chunk_size=chunk_size,
            block_fn=_block_fn,
            transform=_transform,
            standard_errors=method == "mean_log1p",
            memory_limit_gb=memory_limit_gb,
        )
    finally:
        backed.file.close()

    control_profile = _finalise(result.pooled_control_sum / result.pooled_control_count)
    resolved_output = resolve_output_path(
        path,
        suffix="normalized_effects",
        output_dir=output_dir,
        data_name=data_name,
        output_path=output_path,
    )
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = (
        pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
        if candidates else pd.DataFrame(index=obs_index)
    )
    adata = ad.AnnData(
        result.effect.reshape(len(candidates), n_genes), obs=obs, var=pd.DataFrame(index=gene_symbols),
    )
    adata.uns["control_profile"] = control_profile
    adata.uns["method"] = str(method)
    if method == "log_mean":
        adata.uns["baseline_count"] = float(baseline_count)
    if candidates:
        adata.layers["perturbation_profile"] = result.perturbation_profile
        if use_batch:
            adata.layers["control_profile_matched"] = result.control_profile_matched
        if result.standard_error is not None:
            adata.layers["standard_error"] = result.standard_error
            adata.layers["standard_error_robust"] = result.standard_error_robust
    if use_batch:
        adata.uns["batch_column"] = str(batch_column)
        adata.uns["batch_ids"] = np.asarray(result.batch_ids, dtype=object)
    _disk_estimate = warn_if_disk_space_low(
        estimate_bytes(max(len(candidates), 1), n_genes, 1 + len(adata.layers), overhead=1.10),
        resolved_output, context=log_name,
    )
    _messages.print_disk_estimate(verbose, log_name, _disk_estimate)
    _messages.vprint(verbose, log_name, f"{len(candidates)} perturbations × {n_genes} genes")
    _messages.print_saving(verbose, log_name, resolved_output)
    adata.write(resolved_output)
    return AnnData(resolved_output)


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


#: Arguments that decide where and how a pseudo-bulk call runs but not what
#: it writes; a finished output from a call differing only in these is
#: reused. ``data`` is identified by the fingerprint's source path, size and
#: mtime instead.
_OPERATIONAL_ARGS = frozenset({
    "data", "chunk_size", "memory_limit_gb", "bulk_output_path", "data_name",
    "output_path", "output_dir", "verbose", "force",
})


#: obs columns aggregate_pseudobulk writes itself; a source column of the
#: same name is not carried over.
_PROFILE_COLUMNS = ("n_cells", "n_cells_aggregated", "sampled_with_replacement")


def _profile_names(keys: list[tuple[str, ...]]) -> pd.Index:
    """Name each profile by its group key: the value itself for one grouping
    column, the values joined with ``"|"`` for several."""
    names = pd.Index(["|".join(key) for key in keys], name="pseudobulk_id")
    if not names.is_unique:
        clashes = names[names.duplicated()].unique().tolist()
        raise ValueError(
            f"Grouping values containing '|' make profile names ambiguous: {clashes[:5]}. "
            "Rename those values before aggregating."
        )
    return names


def _constant_covariates(
    obs: pd.DataFrame, group_codes: np.ndarray, n_groups: int, *, exclude: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """The obs columns that hold one value within every profile (donor sex,
    age, disease, ...), one row per profile; and the names of the columns
    that vary within some profile and so cannot be carried over."""
    candidates = [column for column in obs.columns if column not in exclude]
    selected = np.flatnonzero(group_codes >= 0)
    if not candidates or selected.size == 0:
        return pd.DataFrame(index=pd.RangeIndex(n_groups)), []
    # Sort the cells by profile once; a column is constant when no cell
    # differs from the one before it within the same profile.
    order = selected[np.argsort(group_codes[selected], kind="stable")]
    sorted_groups = group_codes[order]
    starts = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    keep, varying = [], []
    for column in candidates:
        values = _comparable_values(obs[column])
        if values is None:
            varying.append(column)
            continue
        values = values[order]
        same = values[1:] == values[:-1]
        if values.dtype.kind == "f":
            same |= np.isnan(values[1:]) & np.isnan(values[:-1])
        (keep if np.all(same | starts[1:]) else varying).append(column)
    first = obs.iloc[order[starts]][keep]
    first.index = sorted_groups[starts]
    return first.reindex(pd.RangeIndex(n_groups)), varying


def _comparable_values(series: pd.Series) -> np.ndarray | None:
    """Per-cell values whose ``==`` matches equality of the originals (NaN
    as a value of its own); ``None`` when they cannot be compared."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=np.float64, na_value=np.nan)
    try:
        return pd.factorize(series, use_na_sentinel=False)[0]
    except TypeError:  # unhashable values (lists, dicts) in an object column
        return None


def aggregate_pseudobulk(
    data: str | Path | AnnData | ad.AnnData,
    *,
    groupby: str | Sequence[str],
    method: PseudobulkMethod,
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
    verbose: int | bool = True,
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
        Required; there is no default because each choice is wrong for the
        other use. ``"sum"`` sums raw counts -- what count models need
        (DESeq2, edgeR, :func:`nb_glm_test`, causarray) -- and therefore
        requires finite, non-negative, integer-valued input. ``"mean_log1p"``
        computes the mean of per-cell log1p expression, for contrasts on the
        log scale. Integer count input is transformed with ``log1p`` first;
        continuous input is treated as already log1p-transformed and averaged
        unchanged.
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
        A result matches when it was computed from the same input file (path,
        size, modification time) with the same result-shaping arguments, as
        recorded in its ``uns["crispyx"]``.

    Returns
    -------
    AnnData
        On-disk result with absolute profiles in ``X``, one row per profile
        named by its group key (the value for one ``groupby`` column, the
        values joined with ``"|"`` for several, e.g. ``"donor1|L2/3 IT"``).
        ``obs`` holds the grouping columns, ``n_cells`` and
        ``n_cells_aggregated``, and every other source ``obs`` column that is
        constant within each profile (donor sex, age, disease, ...); columns
        that vary within a profile are left out. ``n_cells`` pivoted over two
        grouping columns is their composition table, e.g. cells per donor and
        cell type. ``uns['crispyx_pseudobulk']`` describes the aggregation and
        ``uns['crispyx']`` is the provenance stamp.
    """
    call_args = dict(locals())  # for the result fingerprint, before any other local
    _check_method(
        method, PseudobulkMethod, "aggregate_pseudobulk",
        " Use compute_normalized_effects for method='log_mean'.",
    )
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
    fingerprint = call_fingerprint(path, call_args, _OPERATIONAL_ARGS)
    if not force and reusable(resolved_output, "pseudobulk", fingerprint, verbose):
        return AnnData(resolved_output)
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
            _messages.warn(
                "pb.aggregate",
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
            _messages.vprint(verbose, "pb.aggregate", f"chunk_size={chunk_size} (auto)")
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
        }

        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        _accumulator_disk_estimate = warn_if_disk_space_low(
            estimate_bytes(max(len(keys), 1), backed.n_vars), tempfile.gettempdir(),
            context="pb.aggregate accumulator",
        )
        _messages.print_disk_estimate(verbose, "pb.aggregate accumulator", _accumulator_disk_estimate)
        _output_disk_estimate = warn_if_disk_space_low(
            estimate_bytes(max(len(keys), 1), backed.n_vars, overhead=1.10), resolved_output,
            context="pb.aggregate",
        )
        _messages.print_disk_estimate(verbose, "pb.aggregate", _output_disk_estimate)
        with tempfile.TemporaryDirectory(prefix="cx_pseudobulk_") as tmpdir:
            sums_path = Path(tmpdir) / "profiles.dat"
            profiles = np.memmap(
                sums_path,
                dtype=np.float64,
                mode="w+",
                shape=(max(len(keys), 1), backed.n_vars),
            )
            profiles.fill(0)
            n_chunks = (backed.n_obs + chunk_size - 1) // chunk_size
            with _create_progress_context(n_chunks, "pb.aggregate", verbose, unit="chunk") as pbar:
                for start in range(0, backed.n_obs, chunk_size):
                    end = min(start + chunk_size, backed.n_obs)
                    local_codes = group_codes[start:end]
                    local_mult = multiplicity[start:end]
                    local = (local_codes >= 0) & (local_mult > 0)
                    if not local.any():
                        pbar.update(1)
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
                    pbar.update(1)
            if method == "mean_log1p" and len(keys):
                profiles[: len(keys)] /= aggregated_counts[:, None]

            obs = pd.DataFrame(
                {column: [key[i] for key in keys] for i, column in enumerate(columns)},
                index=_profile_names(keys),
            )
            covariates, varying = _constant_covariates(
                backed.obs, group_codes, len(keys), exclude=[*columns, *_PROFILE_COLUMNS],
            )
            for column in covariates.columns:
                obs[column] = covariates[column].set_axis(obs.index)  # keeps nullable dtypes
            if varying:
                _messages.vprint(
                    verbose, "pb.aggregate",
                    f"obs columns that vary within a profile are not carried over: {varying}",
                )
            obs["n_cells"] = counts
            obs["n_cells_aggregated"] = aggregated_counts
            obs["sampled_with_replacement"] = bootstrap_size is not None
            var = backed.var.copy()
            var.index = pd.Index(gene_symbols, name=backed.var_names.name)
            result = ad.AnnData(np.asarray(profiles[: len(keys)]), obs=obs, var=var)
            result.uns["crispyx_pseudobulk"] = metadata
            result.uns[_PROVENANCE_KEY] = stamp("pseudobulk", fingerprint)
            with _replace_on_success(resolved_output) as partial:
                result.write(partial)
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
    verbose: int | bool = True,
    force: bool = False,
) -> AnnData:
    """Compute target-minus-control effects from batch-level pseudo-bulk profiles.

    ``data`` may be a file created by :func:`aggregate_pseudobulk` or a
    cell-level h5ad file. Cell-level input is aggregated first using
    ``[perturbation_column, batch_column]``. By default, one effect is retained
    per perturbation and batch. Set ``aggregate_batches=True`` to combine
    within-batch effects using harmonic target/control cell-count weights.
    """
    call_args = dict(locals())  # for the result fingerprint, before any other local
    _check_method(method, PseudobulkMethod, "compute_pseudobulk_effects")
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="compute_pseudobulk_effects",
    )
    source_path = resolve_data_path(data)
    resolved_output = resolve_output_path(
        source_path,
        suffix="pseudobulk_effects_by_batch" if not aggregate_batches else "pseudobulk_effects",
        data_name=data_name,
        output_path=output_path,
        output_dir=output_dir,
    )
    # Identified by the caller's input, so a cell-level input's intermediate
    # pseudo-bulk file is an implementation detail of this call; checked
    # before that aggregation streams the matrix.
    fingerprint = call_fingerprint(
        source_path,
        canonical_call_args(call_args, perturbation_column=perturbation_column, control_label=control_label),
        _OPERATIONAL_ARGS,
    )
    if not force and reusable(resolved_output, "pseudobulk_effects", fingerprint, verbose):
        return AnnData(resolved_output)
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
                _messages.warn(
                    "pb.effects",
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
                "source_is_pseudobulk": bool(input_is_bulk),
                "perturbation_column": str(perturbation_column),
                "batch_column": str(local_batch_column),
                "reference": str(resolved_control),
                "effect": "target_minus_reference",
                "aggregate_batches": bool(aggregate_batches),
                "method": str(marker.get("method", method)),
                "n_missing_control_batches": int(len(missing_batches)),
            }

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
            _accumulator_disk_estimate = warn_if_disk_space_low(
                estimate_bytes(max(n_output, 1), bulk.n_vars, 3), tempfile.gettempdir(),
                context="pb.effects accumulator",
            )
            _messages.print_disk_estimate(verbose, "pb.effects accumulator", _accumulator_disk_estimate)
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
                result.uns[_PROVENANCE_KEY] = stamp("pseudobulk_effects", fingerprint)
                with _replace_on_success(resolved_output) as partial:
                    result.write(partial)
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
    method: NormalizedEffectMethod = "mean_log1p",
    **_ignored,
) -> dict[str, float]:
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="compute_normalized_effects",
    )
    # mean_log1p adds two standard-error layers, and a sum-of-squares
    # accumulator beside the per-(perturbation, batch) sums.
    with_se = method == "mean_log1p"
    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        n_genes = backed.n_vars
        n_outputs = (3 if batch_column is not None else 2) + (2 if with_se else 0)
        if batch_column is None:
            return {
                "tempdir": estimate_bytes(max(len(candidates), 1), n_genes, 2 if with_se else 1),
                "output": estimate_bytes(len(candidates), n_genes, n_outputs, overhead=1.10),
            }
        batch_labels = backed.obs[batch_column].astype(str).to_numpy()
        pert_code = pd.Index(candidates).get_indexer(pd.Index(labels))
        batch_code, batch_uniques = pd.factorize(batch_labels, sort=False)
        n_batches = len(batch_uniques)
        cand_cell = pert_code >= 0
        pair_key_all = pert_code[cand_cell].astype(np.int64) * n_batches + batch_code[cand_cell]
        n_pairs = int(np.unique(pair_key_all).shape[0]) if cand_cell.any() else 0
        return {
            "tempdir": estimate_bytes(max(n_pairs, 1), n_genes, 2 if with_se else 1),
            "output": estimate_bytes(len(candidates), n_genes, n_outputs, overhead=1.10),
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
# ``__all__``.
__all__ = [
    "NormalizedEffectMethod",
    "PseudobulkMethod",
    "aggregate_pseudobulk",
    "compute_normalized_effects",
    "compute_pseudobulk_effects",
]
