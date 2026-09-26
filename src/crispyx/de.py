"""Differential expression testing utilities."""

from __future__ import annotations

import gc
import dataclasses
import logging
import os
import warnings
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Mapping, Tuple

from joblib import Parallel, delayed
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from numpy.typing import ArrayLike

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import norm, rankdata, t as t_dist

from .data import (
    AnnData,
    calculate_optimal_chunk_size,
    calculate_optimal_gene_chunk_size,
    calculate_wilcoxon_chunk_size,
    calculate_nb_glm_chunk_size,
    drop_file_cache,
    ensure_gene_symbol_column,
    get_matrix_storage_format,
    get_perturbation_slice,
    iter_matrix_chunks,
    needs_sorting_for_nbglm,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    scratch_copy_bytes,
    sort_by_perturbation,
    stream_on_fast_axis,
    validate_format_mismatch_policy,
    _read_h5_1d,
    _replace_on_success,
)
from .glm import (
    NBGLMFitter,
    NBGLMBatchFitter,
    ControlStatisticsCache,
    build_design_matrix,
    fit_nb_glm_batch_auto,
    estimate_covariate_effects_streaming,
    estimate_dispersion_map,
    estimate_global_dispersion_streaming,
    fit_dispersion_trend,
    precompute_control_statistics,
    precompute_control_statistics_streaming,
    precompute_global_dispersion,
    precompute_global_dispersion_from_path,
    shrink_dispersions,
    shrink_lfc_apeglm,
    shrink_lfc_apeglm_from_stats,
    _estimate_apeglm_prior_scale,
)
from ._kernels import (
    _rankdata_2d_numba,
    _tie_correction_numba,
    _compute_rank_sums_batch_numba,
    _wilcoxon_sparse_batch_numba,
    _wilcoxon_all_perts_numba,
    _presort_control_nonzeros,
    _compute_ctrl_tie_sums,
    _wilcoxon_presorted_ctrl_numba,
    _wilcoxon_batch_perts_presorted_numba,
    _wilcoxon_stratified_batch_perts_numba,
    _ZERO_PARTITION_THRESHOLD,
)
from ._checkpoint import (
    ResumableRun,
    run_fingerprint,
    _get_checkpoint_interval,
    _create_progress_context,
    _DummyProgress,
)
from . import _messages
from ._disk import estimate_bytes, warn_if_disk_space_low
from ._grouping import resolve_group_reference_aliases
from ._memory import _detected_available_bytes, _resolve_n_jobs, _should_use_streaming
from ._size_factors import (
    _validate_size_factors,
    _median_of_ratios_size_factors,
    _deseq2_style_size_factors,
    _compute_subset_size_factors,
)
from ._statistics import (
    _tie_correction,
    _adjust_pvalue_matrix,
    _compute_se_batched,
    _compute_mom_dispersion_batched,
    _low_expr_in_both_mask,
    _nonestimable_glm_mask,
)
from ._memory import (
    _estimate_max_workers,
    _get_available_memory_mb,
)

logger = logging.getLogger(__name__)


@dataclass
class DifferentialExpressionResult:
    genes: pd.Index
    effect_size: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    method: str
    perturbation: str
    pvalue_adj: np.ndarray | None = None
    result: AnnData | None = field(default=None, repr=False)

    @property
    def result_path(self) -> Path:
        """Path to the on-disk result file.

        Returns
        -------
        Path
            Absolute path to the ``.h5ad`` file backing the result.

        Raises
        ------
        AttributeError
            If the result AnnData has not been initialised.
        """
        if self.result is None:
            raise AttributeError("Result AnnData has not been initialised.")
        return self.result.path

    def __getstate__(self) -> dict:
        """Exclude the AnnData file handle from the pickle payload."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "result"
        }

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle; ``result`` is set to ``None`` (no open file handle)."""
        for key, value in state.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "result", None)


@dataclass
class RankGenesGroupsResult(Mapping[str, DifferentialExpressionResult]):
    genes: pd.Index
    groups: list[str]
    statistics: np.ndarray
    pvalues: np.ndarray
    pvalues_adj: np.ndarray
    logfoldchanges: np.ndarray
    effect_size: np.ndarray
    u_statistics: np.ndarray
    pts: np.ndarray
    pts_rest: np.ndarray
    order: np.ndarray
    groupby: str
    method: str
    control_label: str
    tie_correct: bool
    pvalue_correction: Literal["benjamini-hochberg", "bonferroni"]
    result: AnnData | None = field(default=None, repr=False)
    _group_cache: Dict[str, DifferentialExpressionResult] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.genes = pd.Index(self.genes).astype(str)
        self.groups = [str(group) for group in self.groups]

    @property
    def result_path(self) -> Path:
        if self.result is None:
            raise AttributeError("Result AnnData has not been initialised.")
        return self.result.path

    def __getstate__(self) -> dict:
        """Exclude the AnnData file handle and group cache from the pickle payload."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("result", "_group_cache")
        }

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle; ``result`` is ``None`` and cache is empty."""
        for key, value in state.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "result", None)
        object.__setattr__(self, "_group_cache", {})

    def _ensure_cache(self) -> None:
        if self._group_cache:
            return
        for idx, group in enumerate(self.groups):
            self._group_cache[group] = DifferentialExpressionResult(
                genes=self.genes,
                effect_size=self.effect_size[idx],
                statistic=self.statistics[idx],
                pvalue=self.pvalues[idx],
                method=self.method,
                perturbation=group,
                pvalue_adj=self.pvalues_adj[idx],
                result=self.result,
            )

    def __getitem__(self, key: str) -> DifferentialExpressionResult:
        """Return per-gene DE results for a single perturbation group.

        Parameters
        ----------
        key : str
            Perturbation group name.

        Returns
        -------
        DifferentialExpressionResult
            Results for the requested group.
        """
        self._ensure_cache()
        return self._group_cache[key]

    def __iter__(self):  # type: ignore[override]
        """Iterate over perturbation group names."""
        return iter(self.groups)

    def __len__(self) -> int:
        """Return the number of perturbation groups."""
        return len(self.groups)

    def items(self):  # type: ignore[override]
        """Return (group_name, DifferentialExpressionResult) pairs."""
        self._ensure_cache()
        return self._group_cache.items()

    def to_rank_genes_groups_dict(self) -> dict:
        """Convert results to Scanpy-compatible ``rank_genes_groups`` dict.

        Returns
        -------
        dict
            Dictionary with keys ``'params'``, ``'names'``, ``'scores'``,
            ``'logfoldchanges'``, ``'pvals'``, ``'pvals_adj'``, ``'pts'``,
            ``'pts_rest'``, ``'auc'``, ``'u_stat'``, and ``'full'``.
            Each value (except ``'params'`` and ``'full'``) is a NumPy
            record array sorted by descending absolute score.
        """
        gene_array = self.genes.to_numpy()
        sorted_names = gene_array[self.order]
        sorted_scores = np.take_along_axis(self.statistics, self.order, axis=1)
        sorted_lfc = np.take_along_axis(self.logfoldchanges, self.order, axis=1)
        sorted_pvals = np.take_along_axis(self.pvalues, self.order, axis=1)
        sorted_padj = np.take_along_axis(self.pvalues_adj, self.order, axis=1)
        sorted_pts = np.take_along_axis(self.pts, self.order, axis=1)
        sorted_pts_rest = np.take_along_axis(self.pts_rest, self.order, axis=1)
        sorted_effect = np.take_along_axis(self.effect_size, self.order, axis=1)
        sorted_u = np.take_along_axis(self.u_statistics, self.order, axis=1)

        def to_recarray(matrix: np.ndarray) -> np.recarray:
            arrays = [matrix[idx] for idx in range(matrix.shape[0])]
            return np.rec.fromarrays(arrays, names=self.groups)

        rank_genes_groups = {
            "params": {
                "groupby": self.groupby,
                "method": self.method,
                "reference": self.control_label,
                "tie_correct": self.tie_correct,
                "corr_method": self.pvalue_correction,
            },
            "names": to_recarray(sorted_names.astype(object)),
            "scores": to_recarray(sorted_scores),
            "logfoldchanges": to_recarray(sorted_lfc),
            "pvals": to_recarray(sorted_pvals),
            "pvals_adj": to_recarray(sorted_padj),
            "pts": to_recarray(sorted_pts),
            "pts_rest": to_recarray(sorted_pts_rest),
            "auc": to_recarray(sorted_effect),
            "u_stat": to_recarray(sorted_u),
        }
        rank_genes_groups["full"] = self.to_full_order_dict()
        return rank_genes_groups

    def to_full_order_dict(self) -> dict:
        """Return unsorted matrices keyed by statistic name.

        Unlike :meth:`to_rank_genes_groups_dict`, values are in the
        original gene order (not sorted by score).

        Returns
        -------
        dict
            Keys: ``'scores'``, ``'pvals'``, ``'pvals_adj'``,
            ``'logfoldchanges'``, ``'auc'``, ``'u_stat'``, ``'pts'``,
            ``'pts_rest'``.  Each value is a 2-D ``(n_groups, n_genes)``
            NumPy array (copy).
        """
        return {
            "scores": self.statistics.copy(),
            "pvals": self.pvalues.copy(),
            "pvals_adj": self.pvalues_adj.copy(),
            "logfoldchanges": self.logfoldchanges.copy(),
            "auc": self.effect_size.copy(),
            "u_stat": self.u_statistics.copy(),
            "pts": self.pts.copy(),
            "pts_rest": self.pts_rest.copy(),
        }


def _load_layered_de_result(
    output_path: Path,
    candidates: list[str],
    gene_symbols: list[str],
    perturbation_column: str,
    control_label: str,
    corr_method: str,
    method: str,
) -> "RankGenesGroupsResult":
    """Load an existing ``t_test`` or ``nb_glm_test`` result from its h5ad file."""
    adata = ad.read_h5ad(output_path)

    statistic_matrix = np.array(adata.layers["z_score"])
    pvalue_matrix = np.array(adata.layers["pvalue"])
    pvalue_adj_matrix = np.array(adata.layers["pvalue_adj"])
    # X is the effect size. t_test stores a separate scanpy-style fold change;
    # for nb_glm the fold change is the effect size.
    effect_matrix = np.array(adata.X)
    logfc_matrix = np.array(adata.layers["logfoldchanges"]) if "logfoldchanges" in adata.layers else effect_matrix
    pts_matrix = np.array(adata.layers["pts"])
    pts_rest_matrix = np.broadcast_to(adata.var["pts_rest"].to_numpy(dtype=np.float32), pts_matrix.shape)
    
    # Reconstruct order from statistics
    statistic_for_order = np.where(
        np.isfinite(statistic_matrix), np.abs(statistic_matrix), -np.inf
    )
    order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")
    
    return RankGenesGroupsResult(
        genes=pd.Index(gene_symbols),
        groups=candidates,
        statistics=statistic_matrix,
        pvalues=pvalue_matrix,
        pvalues_adj=pvalue_adj_matrix,
        logfoldchanges=logfc_matrix,
        effect_size=effect_matrix,
        u_statistics=np.zeros_like(effect_matrix),
        pts=pts_matrix,
        pts_rest=pts_rest_matrix,
        order=order_matrix,
        groupby=perturbation_column,
        method=method,
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
        result=AnnData(output_path),
    )


def _resolve_candidates(
    labels: np.ndarray,
    control_label: str,
    perturbations: Iterable[str] | None,
) -> list[str]:
    """Determine which perturbation groups to test.

    Parameters
    ----------
    labels : ndarray
        All perturbation labels from the dataset.
    control_label : str
        Control group label to exclude.
    perturbations : iterable of str or None
        Explicit subset to test.  ``None`` means test all non-control groups.

    Returns
    -------
    list of str
        Candidate perturbation group names.

    Raises
    ------
    ValueError
        If no candidate groups remain after filtering.
    """
    if perturbations is None:
        unique = pd.Index(labels).unique().tolist()
    else:
        unique = [str(p) for p in perturbations]
    candidates = [label for label in unique if label != control_label]
    if not candidates:
        raise ValueError("No perturbation groups available for differential expression testing")
    return candidates


def _group_row_indices(labels: np.ndarray, groups: Iterable[str]) -> dict[str, np.ndarray]:
    """Ascending row indices of every label in ``groups``, in one pass.

    Equivalent to ``{g: np.where(labels == g)[0] for g in groups}`` but
    ``O(n_cells log n_cells)`` instead of ``O(n_cells * n_groups)`` string
    comparisons -- minutes versus seconds for ~18k groups over ~2M cells.
    """
    codes, uniques = pd.factorize(labels)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.arange(len(uniques) + 1)
    starts = np.searchsorted(sorted_codes, bounds[:-1], side="left")
    ends = np.searchsorted(sorted_codes, bounds[1:], side="left")
    position = {str(label): i for i, label in enumerate(uniques)}
    empty = np.empty(0, dtype=np.intp)
    out: dict[str, np.ndarray] = {}
    for group in groups:
        i = position.get(str(group))
        out[group] = order[starts[i]:ends[i]] if i is not None else empty
    return out


#: Arguments that change how a DE call runs but not what it computes; a
#: checkpoint from a call differing only in these is still resumable. The
#: memory and chunking knobs belong here because rerunning with a lower
#: ``memory_limit_gb`` is the natural response to an OOM kill; a path that
#: counts its progress in chunks puts the resolved chunk size in the
#: fingerprint itself (see ``wilcoxon_test``).
_OPERATIONAL_ARGS = frozenset({
    "data", "perturbations", "verbose", "resume", "checkpoint_interval", "n_jobs",
    "profiling", "force", "output_path", "output_dir", "data_name", "scanpy_format",
    "format_mismatch_policy", "memory_limit_gb", "max_dense_fraction",
    "cell_chunk_size", "chunk_size", "irls_batch_size",
})


def _de_fingerprint(path: Path, call_args: dict, **extra) -> dict:
    """:func:`run_fingerprint` of a DE call: the source file, every argument
    that shapes the results (``call_args`` is the function's ``locals()`` at
    entry), and ``extra`` (the resolved perturbations, gene count, ...)."""
    params = {k: v for k, v in call_args.items() if k not in _OPERATIONAL_ARGS}
    return run_fingerprint(path, params=params, **extra)


# Disk a DE run needs beside its output: the resumable partial arrays plus
# the final h5ad, which coexist until the run finishes. One function per
# method, shared by the run's own low-disk warning and estimate_disk_usage.

def _t_test_disk_bytes(n_groups: int, n_genes: int) -> float:
    # partial: statistics/pvalues/pvalues_adj/order (8 B) + lfc/effect/pts (4 B)
    # output: z/pvalue/pvalue_adj (8 B) + X/logfoldchanges/pts (4 B)
    return estimate_bytes(n_groups, n_genes, itemsize=8) * 7 + estimate_bytes(n_groups, n_genes, itemsize=4) * 6


def _nb_glm_disk_bytes(n_groups: int, n_genes: int) -> float:
    # partial: 10 f64 matrices + pts/iterations (4 B) + converged (1 B)
    # output (upper bound): X, z, p, padj, intercept, se, 2 ln-scale, a raw
    # LFC and 3 per-comparison dispersion layers (8 B) + pts (4 B) + 2 small
    return (
        estimate_bytes(n_groups, n_genes, itemsize=8) * 22
        + estimate_bytes(n_groups, n_genes, itemsize=4) * 3
        + estimate_bytes(n_groups, n_genes, itemsize=1) * 3
    )


def _wilcoxon_disk_bytes(n_groups: int, n_genes: int) -> float:
    # partial: effect/u/pvalue/pvalue_adj/z/lfc (8 B) + pts (4 B); the output
    # holds the same seven matrices. The streaming path writes only the
    # output-sized file, so this bounds every path.
    return estimate_bytes(n_groups, n_genes, itemsize=8) * 12 + estimate_bytes(n_groups, n_genes, itemsize=4) * 2


def _release_chunk_memory() -> None:
    """Force Python and glibc to return freed memory to the OS.

    After large per-chunk arrays are deleted, glibc may hold freed pages in
    thread-local arenas.  ``gc.collect()`` tears down Python reference cycles,
    then ``malloc_trim(0)`` (Linux-only) asks glibc to return free heap pages
    to the OS, shrinking virtual address space.  This prevents RLIMIT_AS
    violations over many gene chunks.
    """
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Non-Linux or libc without malloc_trim


def _write_wilcoxon_result_h5ad(
    output_path: Path,
    *,
    effect_matrix: np.ndarray,
    z_matrix: np.ndarray,
    pvalue_matrix: np.ndarray,
    pvalue_adj_matrix: np.ndarray,
    lfc_matrix: np.ndarray,
    u_matrix: np.ndarray,
    pts_matrix: np.ndarray,
    pts_rest: np.ndarray,
    candidates: list[str],
    gene_symbols: pd.Index,
    perturbation_column: str,
    control_label: str,
    tie_correct: bool,
    corr_method: str,
    batch_column: str | None = None,
    stratified_diagnostics: dict[str, object] | None = None,
) -> None:
    """Write wilcoxon result arrays to h5ad.

    obs, var and uns go through anndata (so ``adata.uns`` and the plotting
    readers see the metadata, as for t_test and nb_glm); the matrices, which
    may be memmaps, are then written one dataset at a time via h5py, avoiding
    the triple allocation of memmap → np.array copy → AnnData → ``.write()``.

    ``pts_rest`` is the control arm's detection rate, one value per gene, so
    it is stored as ``var["pts_rest"]`` rather than repeated for every
    perturbation.
    """
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame({"pts_rest": np.asarray(pts_rest, dtype=np.float32)}, index=gene_symbols.astype(str))
    uns = _wilcoxon_uns(
        control_label=control_label, perturbation_column=perturbation_column,
        tie_correct=tie_correct, corr_method=corr_method, batch_column=batch_column,
        stratified_diagnostics=stratified_diagnostics,
    )
    # The metadata lands before the matrices, so the file is written under a
    # partial name: a run killed mid-write must not leave a file whose uns
    # matches and passes for a finished result.
    with _replace_on_success(output_path) as partial:
        ad.AnnData(obs=obs, var=var, uns=uns).write(partial)
        with h5py.File(partial, "r+") as hf:
            _create_array(hf, "X", data=effect_matrix)
            layers_grp = hf.require_group("layers")
            _create_array(layers_grp, "z_score", data=z_matrix)
            _create_array(layers_grp, "pvalue", data=pvalue_matrix)
            _create_array(layers_grp, "pvalue_adj", data=pvalue_adj_matrix)
            _create_array(layers_grp, "logfoldchanges", data=lfc_matrix)
            _create_array(layers_grp, "u_statistic", data=u_matrix)
            _create_array(layers_grp, "pts", data=pts_matrix)


def _create_array(group: h5py.Group, name: str, **kwargs) -> h5py.Dataset:
    """``group.create_dataset`` tagged as an anndata dense array."""
    ds = group.create_dataset(name, **kwargs)
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"
    return ds


def _wilcoxon_uns(
    *,
    control_label: str,
    perturbation_column: str,
    tie_correct: bool,
    corr_method: str,
    batch_column: str | None = None,
    stratified_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    """``uns`` metadata shared by every Wilcoxon result path."""
    uns: dict[str, object] = {
        "method": "wilcoxon",
        "control_label": control_label,
        "perturbation_column": perturbation_column,
        "tie_correct": bool(tie_correct),
        "pvalue_correction": corr_method,
        "stratified": batch_column is not None,
    }
    if batch_column is not None:
        uns["batch_column"] = batch_column
    uns.update(stratified_diagnostics or {})
    return uns


def _build_result_from_h5ad(
    output_path: Path,
    candidates: list[str],
    gene_symbols: pd.Index,
    perturbation_column: str,
    control_label: str,
    tie_correct: bool,
    corr_method: str,
    *,
    memory_limit_gb: float | None = None,
) -> "RankGenesGroupsResult":
    """Build a RankGenesGroupsResult by reading back arrays from the h5ad file.

    For very large result files (> 25% of *physical* memory), returns a result
    with empty arrays and only the ``result`` AnnData reference, so callers
    can access data from disk on demand.

    The lazy threshold is based on *actual physical memory* (``None`` →
    auto-detect via psutil), not ``memory_limit_gb``, because the latter
    may be set artificially low to force the streaming dispatch path.
    """
    from ._memory import _resolve_memory_limit_bytes

    n_groups = len(candidates)
    n_genes = len(gene_symbols)
    result_bytes = n_groups * n_genes * (6 * 8 + 4)  # X + 5 f64 layers + pts
    # Use actual physical memory for the lazy-load decision, not the
    # (possibly tiny) memory_limit_gb used for streaming dispatch.
    physical_budget = _resolve_memory_limit_bytes(None)
    # If an explicit limit was given and is *larger* than physical, honour it
    # (e.g., user trusts their system won't OOM at 256 GB).
    if memory_limit_gb is not None:
        explicit_budget = memory_limit_gb * 1e9
        budget = max(physical_budget, explicit_budget)
    else:
        budget = physical_budget
    skip_loading = result_bytes > budget * 0.25

    if skip_loading:
        logger.info(
            "Result arrays too large to fit in memory (%.1f GB > %.1f GB budget × 0.25). "
            "Returning lazy result — access data via result.result (AnnData h5ad).",
            result_bytes / 1e9, budget / 1e9,
        )
        # Construct a minimal result with zero-size arrays
        empty_f64 = np.empty((0, 0), dtype=np.float64)
        empty_f32 = np.empty((0, 0), dtype=np.float32)
        empty_i64 = np.empty((0, 0), dtype=np.int64)
        result = RankGenesGroupsResult(
            genes=gene_symbols,
            groups=candidates,
            statistics=empty_f64,
            pvalues=empty_f64,
            pvalues_adj=empty_f64,
            logfoldchanges=empty_f64,
            effect_size=empty_f64,
            u_statistics=empty_f64,
            pts=empty_f32,
            pts_rest=empty_f32,
            order=empty_i64,
            groupby=perturbation_column,
            method="wilcoxon",
            control_label=control_label,
            tie_correct=tie_correct,
            pvalue_correction=corr_method,
        )
        result.result = AnnData(output_path)
        return result

    with h5py.File(output_path, "r") as hf:
        z_arr = hf["layers/z_score"][:]
        pval_arr = hf["layers/pvalue"][:]
        pval_adj_arr = hf["layers/pvalue_adj"][:]
        lfc_arr = hf["layers/logfoldchanges"][:]
        effect_arr = hf["X"][:]
        u_arr = hf["layers/u_statistic"][:]
        pts_arr = np.array(hf["layers/pts"][:], dtype=np.float32)
        pts_rest_arr = np.broadcast_to(
            np.asarray(hf["var/pts_rest"][:], dtype=np.float32), pts_arr.shape
        )

    order_arr = np.argsort(-np.abs(z_arr), axis=1, kind="mergesort").astype(np.int64)

    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=z_arr,
        pvalues=pval_arr,
        pvalues_adj=pval_adj_arr,
        logfoldchanges=lfc_arr,
        effect_size=effect_arr,
        u_statistics=u_arr,
        pts=pts_arr,
        pts_rest=pts_rest_arr,
        order=order_arr,
        groupby=perturbation_column,
        method="wilcoxon",
        control_label=control_label,
        tie_correct=tie_correct,
        pvalue_correction=corr_method,
    )
    result.result = AnnData(output_path)
    return result


def _write_rank_genes_groups_hdf5(
    output_path: Path,
    result: "RankGenesGroupsResult",
) -> None:
    """Write rank_genes_groups to HDF5 for Scanpy compatibility.
    
    Writes arrays in full matrix order (groups × genes) to uns/rank_genes_groups/full.
    This format is compatible with Scanpy's rank_genes_groups output but avoids
    the recarray format which causes HDF5 header size limits for large group counts.
    
    Parameters
    ----------
    output_path
        Path to the h5ad file to modify.
    result
        RankGenesGroupsResult containing the DE statistics.
        
    Notes
    -----
    For datasets with many groups (>1000), this adds ~2-6 seconds of I/O overhead.
    The recarray format (with group names as dtype fields) is avoided because it
    hits HDF5 header size limits at ~2000+ groups.
    """
    with h5py.File(output_path, "r+") as handle:
        uns_group = handle.require_group("uns")
        if "rank_genes_groups" in uns_group:
            del uns_group["rank_genes_groups"]
        rgg = uns_group.create_group("rank_genes_groups")
        
        # Store full-order matrices (groups × genes)
        full = rgg.create_group("full")
        full.create_dataset("scores", data=result.statistics)
        full.create_dataset("pvals", data=result.pvalues)
        full.create_dataset("pvals_adj", data=result.pvalues_adj)
        full.create_dataset("logfoldchanges", data=result.logfoldchanges)
        full.create_dataset("auc", data=result.effect_size)
        full.create_dataset("u_stat", data=result.u_statistics)
        full.create_dataset("pts", data=result.pts)
        full.create_dataset("pts_rest", data=result.pts_rest)
        
        # Store order and metadata
        rgg.create_dataset("order", data=result.order)
        rgg.create_dataset("names", data=np.array(result.groups, dtype="S"))
        
        # Store params for compatibility
        params = rgg.create_group("params")
        params.attrs["groupby"] = result.groupby
        params.attrs["method"] = result.method
        params.attrs["reference"] = result.control_label
        params.attrs["tie_correct"] = result.tie_correct
        params.attrs["corr_method"] = result.pvalue_correction


def _load_completed_de_result(
    output_path: Path,
    *,
    memory_limit_gb: float | None = None,
) -> "RankGenesGroupsResult":
    """Load a completed DE result directly from an existing h5ad file.

    Reads all necessary metadata (method, control_label, perturbation_column,
    etc.) from the result file itself so the original input data is not required.
    Dispatches to :func:`_build_result_from_h5ad` for wilcoxon results and
    :func:`_load_layered_de_result` for nb_glm / t_test results.
    """

    def _decode(v) -> str:
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode()
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode()
        return str(v)

    with h5py.File(output_path, "r") as hf:
        # obs index → candidate perturbation labels
        obs_grp = hf["obs"]
        idx_key = obs_grp.attrs.get("_index", "_index")
        if isinstance(idx_key, (bytes, np.bytes_)):
            idx_key = idx_key.decode()
        candidates: list[str] = [_decode(s) for s in _read_h5_1d(obs_grp[idx_key])]

        # var index → gene symbols
        var_grp = hf["var"]
        var_idx_key = var_grp.attrs.get("_index", "_index")
        if isinstance(var_idx_key, (bytes, np.bytes_)):
            var_idx_key = var_idx_key.decode()
        gene_symbols = pd.Index([_decode(s) for s in _read_h5_1d(var_grp[var_idx_key])])

        # uns metadata — wilcoxon writes to group attrs; anndata (nb_glm, t_test)
        # writes scalar datasets inside the uns group.
        uns = hf["uns"]

        def _read_uns(key: str, default=None):
            if key in uns.attrs:
                return _decode(uns.attrs[key])
            if key in uns:
                return _decode(uns[key][()])
            return default

        method = _read_uns("method", "wilcoxon")
        control_label = _read_uns("control_label")
        perturbation_column = _read_uns("perturbation_column")
        corr_method = _read_uns("pvalue_correction", "benjamini-hochberg")

        # tie_correct is wilcoxon-specific (stored as a bool)
        tie_correct: bool = True
        if "tie_correct" in uns.attrs:
            tie_correct = bool(uns.attrs["tie_correct"])
        elif "tie_correct" in uns:
            tie_correct = bool(uns["tie_correct"][()])

    if method == "wilcoxon":
        return _build_result_from_h5ad(
            output_path,
            candidates=candidates,
            gene_symbols=gene_symbols,
            perturbation_column=perturbation_column,
            control_label=control_label,
            tie_correct=tie_correct,
            corr_method=corr_method,
            memory_limit_gb=memory_limit_gb,
        )
    return _load_layered_de_result(
        output_path=output_path,
        candidates=candidates,
        gene_symbols=gene_symbols.tolist(),
        perturbation_column=perturbation_column,
        control_label=control_label,
        corr_method=corr_method,
        method=method,
    )


# ---------------------------------------------------------------------------
# Shared helpers for public DE functions
# ---------------------------------------------------------------------------

def _resolve_de_aliases(
    *,
    perturbation_column: str | None,
    groupby: str | None,
    control_label: str | None,
    reference: str | None,
    min_pct_both: float | None,
    min_pct_ctrl: float,
    min_pct_pert: float,
    fn_name: str,
) -> tuple[str, str | None, float, float]:
    """Resolve groupby/reference aliases and handle min_pct_both.

    Returns ``(perturbation_column, control_label, min_pct_ctrl, min_pct_pert)``.
    """
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name=fn_name,
    )
    # min_pct_both silent alias
    if min_pct_both is not None:
        min_pct_ctrl = float(min_pct_both)
        min_pct_pert = float(min_pct_both)
    return perturbation_column, control_label, min_pct_ctrl, min_pct_pert


def _decode_h5_scalar(value):
    """Decode small HDF5 scalar/attribute values for metadata comparison."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return value


def _read_result_uns_metadata(output_path: Path) -> dict[str, object]:
    """Read scalar ``uns`` metadata from an existing h5ad result."""
    metadata: dict[str, object] = {}
    try:
        with h5py.File(output_path, "r") as hf:
            if "uns" not in hf:
                return metadata
            uns = hf["uns"]
            for key, value in uns.attrs.items():
                metadata[str(key)] = _decode_h5_scalar(value)
            for key, obj in uns.items():
                if isinstance(obj, h5py.Dataset) and obj.shape == ():
                    metadata[str(key)] = _decode_h5_scalar(obj[()])
    except OSError:
        return metadata
    return metadata


def _metadata_bool(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _metadata_matches(
    actual: dict[str, object],
    expected: Mapping[str, object | None],
) -> tuple[bool, str | None]:
    """Return whether cached result metadata matches the requested analysis."""
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, bool):
            if _metadata_bool(actual_value) != expected_value:
                return False, key
            continue
        if expected_value is None:
            if actual_value not in (None, "", b""):
                return False, key
            continue
        if str(actual_value) != str(expected_value):
            return False, key
    return True, None


def _try_load_existing_de_result(
    output_path: "Path",
    *,
    force: bool,
    verbose: int | bool,
    method_name: str,
    memory_limit_gb: float | None,
    expected_metadata: Mapping[str, object | None] | None = None,
) -> "RankGenesGroupsResult | None":
    """Return the loaded result if it already exists and ``force=False``, else ``None``."""
    if not (output_path.exists() and not force):
        return None
    if expected_metadata is not None:
        metadata = _read_result_uns_metadata(output_path)
        matches, mismatch_key = _metadata_matches(metadata, expected_metadata)
        if not matches:
            logger.info(
                "Existing %s result at %s does not match requested metadata (%s); rerunning.",
                method_name, output_path, mismatch_key,
            )
            if verbose:
                print(
                    f"[cx] Existing result does not match requested {method_name} "
                    f"metadata ({mismatch_key}); rerunning."
                )
            return None
    logger.info(
        "Found existing %s result at %s. Loading instead of rerunning.",
        method_name, output_path,
    )
    if verbose:
        print(f"[cx] Loading existing result: {output_path}")
        print("[cx] Pass force=True to rerun the analysis.")
    return _load_completed_de_result(output_path, memory_limit_gb=memory_limit_gb)


def _print_de_summary(
    verbose: int | bool,
    method_name: str,
    n_completed: int,
    n_groups: int,
    n_tested_list: "list[int]",
    n_genes: int,
) -> None:
    """Print verbose ≥ 1 summary for a DE run (t-test / NB-GLM)."""
    if int(verbose) >= 1 and n_tested_list:
        _mean = int(sum(n_tested_list) / len(n_tested_list))
        _pct = 100.0 * _mean / n_genes if n_genes else 0
        print(
            f"[cx] {method_name}: {n_completed}/{n_groups} perturbations "
            f"complete, mean {_mean}/{n_genes} genes tested ({_pct:.0f}%)"
        )


def _print_de_perturbation_verbose(
    verbose: int | bool,
    label: str,
    n_tested: int,
    n_genes: int,
) -> None:
    """Print verbose ≥ 2 per-perturbation gene-count line."""
    if int(verbose) >= 2:
        _pct = 100.0 * n_tested / n_genes if n_genes else 0
        print(
            f"[cx] {label}: {n_tested}/{n_genes} genes tested "
            f"({_pct:.0f}%), {n_genes - n_tested} filtered"
        )


def t_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    min_cells_expressed: int = 0,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_pct_both: float | None = None,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
    cell_chunk_size: int | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    n_jobs: int | None = None,
    verbose: int | bool = True,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    scanpy_format: bool = False,
    memory_limit_gb: float | None = None,
    force: bool = False,
) -> RankGenesGroupsResult:
    """Perform a t-test comparing log-expression means for each perturbation.
    
    Returns a RankGenesGroupsResult containing differential expression statistics.
    Results are stored in an h5ad file with layers containing the statistics.
    
    The RankGenesGroupsResult implements the Mapping interface, so it can be used
    like a dict: `result[perturbation_label]` returns a DifferentialExpressionResult
    for that perturbation.

    Input data **should already be normalized and log-transformed** (for example
    using `scanpy.pp.normalize_total` followed by `scanpy.pp.log1p`). To maintain
    backward compatibility with Scanpy-style workflows, count-like inputs are
    automatically normalized and log-transformed in streaming fashion, with a
    warning to encourage explicit preprocessing upstream.
    
    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        log-transformed expression data.
    perturbation_column
        Column in `adata.obs` indicating perturbation labels.
    groupby
        Alias for ``perturbation_column`` (Scanpy-compatible).
        Mutually exclusive with ``perturbation_column``.
    control_label
        Label for the control/reference group. If None, infers from common patterns.
    reference
        Alias for ``control_label`` (Scanpy-compatible).
        Mutually exclusive with ``control_label``.
    gene_name_column
        Column in `adata.var` with gene symbols. If None, uses `adata.var_names`.
    perturbations
        Specific perturbations to test. If None, tests all non-control groups.
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for
        testing. Genes below this threshold are untested: ``NaN`` in
        ``score`` / ``pvalue`` / ``logfoldchanges`` / ``effect_size``, with
        ``pts`` / ``pts_rest`` still populated.
    min_pct_ctrl
        Minimum fraction of expressing cells for the *control* side. A gene is
        excluded only when *both* the control side *and* the perturbed side are
        jointly low. Default ``0.01``.
    min_pct_pert
        Minimum fraction of expressing cells for the *perturbed* side.
        Default ``0.002`` (lower than ctrl; induction from near-zero baseline is
        biologically valid). Set to ``0.0`` to disable the pct check on pert.
    min_pct_both
        If not ``None``, overrides both ``min_pct_ctrl`` and
        ``min_pct_pert`` with the same value.
    min_mean_ctrl
        Minimum mean expression (log1p units) for the *control* side.
        Default ``0.05``. Excluded genes are written as NaN in
        ``score`` / ``pvalue`` / ``logfoldchanges`` / ``effect_size``;
        ``pts`` and ``pts_rest`` remain populated.
    min_mean_pert
        Minimum mean expression for the *perturbed* side. Default ``0.005``.
        Together with ``min_pct_pert`` this forms a dual condition that is more
        robust to doublet / ambient-RNA artefacts than pct alone.
    cell_chunk_size
        Number of cells to process per chunk (memory vs. speed tradeoff). This
        controls streaming along the cell axis and is distinct from any future
        perturbation_chunk_size option that would batch perturbations. Data must
        already be normalized/log-transformed before chunking.
    data_name
        Custom name for output file. If None, uses "t_test" suffix.
    output_path
        Exact path for the output h5ad file. When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
        Deprecated; use ``output_path`` instead. Will be removed in the
        next major version.
    n_jobs
        Number of parallel workers for computing statistics across perturbations.
        If None, uses all available cores. If 1, runs sequentially.
        If -1, uses all available cores.
    verbose
        If True, show a progress bar for perturbation processing. Requires tqdm.
    resume
        If True, continue an interrupted run from its last checkpoint,
        skipping the perturbations it had finished.
        Partial results are kept beside the output, in a hidden
        ``.<output name>.resume`` directory, together with the checkpoint
        ``<output>.progress.json``; both are removed once the output is
        written. A run resumes only from a checkpoint written by a call with
        the same input file, perturbations and result-affecting parameters;
        any other checkpoint is discarded and the run starts over. A
        completed output is loaded as usual (unless ``force=True``).
    checkpoint_interval
        Number of perturbations between checkpoint saves. Auto-determined if None.
    corr_method
        Method for p-value correction: ``"benjamini-hochberg"`` or ``"bonferroni"``.
    scanpy_format
        If True, write Scanpy-compatible ``uns['rank_genes_groups']`` structure
        in addition to the layer-based storage. Adds ~2-6 seconds of I/O overhead
        for large datasets. Default False for performance.
    memory_limit_gb
        Maximum memory budget in GB. Used for automatic chunk size calculation.
        When ``None`` (default), detects available system memory via ``psutil``.
        For HPC environments, set this to your SLURM ``--mem`` value
        (e.g., ``memory_limit_gb=128``).
    force
        If True, rerun the analysis even when the output h5ad file already
        exists. If False (default), load and return the existing result
        instead of rerunning.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].effect_size`, `result[label].pvalue`, etc. The h5ad file
        path is available at `result.result_path`.

        The h5ad file holds one row per perturbation: the effect size in
        ``X`` and the layers ``z_score``, ``pvalue``, ``pvalue_adj``,
        ``logfoldchanges`` and ``pts``. ``pts_rest`` describes the control
        arm, so it is the per-gene column ``var["pts_rest"]``.

    Notes
    -----
    ``logfoldchanges`` follows scanpy's formula,
    ``log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9))``. When a
    gene has no expressing cells in one arm, that arm's term *is* the
    constant, so the magnitude reported is set by the constant -- its ceiling
    is ``log2(1 / 1e-9) = 29.9`` -- and not by the data. The p-values are
    unaffected, but ranking on ``logfoldchanges`` puts such genes above every
    real effect. ``pts`` / ``pts_rest`` tell them apart: a complete knockdown
    has ``pts = 0`` beside a high ``pts_rest``, an undetectable gene has both
    near zero.
    """
    call_args = dict(locals())  # for the resume fingerprint, before any other local

    perturbation_column, control_label, min_pct_ctrl, min_pct_pert = _resolve_de_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        min_pct_both=min_pct_both,
        min_pct_ctrl=min_pct_ctrl,
        min_pct_pert=min_pct_pert,
        fn_name="t_test",
    )
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        raise ValueError("corr_method must be 'benjamini-hochberg' or 'bonferroni'")

    path = resolve_data_path(data)
    output_path = resolve_output_path(
        path, suffix="t_test", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )

    if (r := _try_load_existing_de_result(
        output_path, force=force, verbose=verbose,
        method_name="t_test", memory_limit_gb=memory_limit_gb,
    )):
        return r

    backed = read_backed(path)
    try:
        # Calculate adaptive chunk_size if not provided
        if cell_chunk_size is None:
            cell_chunk_size = calculate_optimal_chunk_size(
                backed.n_obs, backed.n_vars,
                available_memory_gb=memory_limit_gb,
            )
            _messages.vprint(verbose, "t_test", f"cell_chunk_size={cell_chunk_size} (auto)")
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(list(labels), control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        groups = [control_label] + candidates
        group_index = {label: idx for idx, label in enumerate(groups)}
        label_codes = pd.Categorical(labels, categories=groups).codes

        # Only allow sparse matrices; raise error if dense
        for _, chunk in iter_matrix_chunks(backed, axis=0, chunk_size=100, convert_to_dense=False):
            if not sp.issparse(chunk):
                raise ValueError(
                    "t_test only supports sparse input matrices. Please provide a scipy sparse matrix (e.g., CSR/CSC)."
                )
            # Raise if data looks like raw counts (matches scanpy's check_nonnegative_integers)
            if np.issubdtype(chunk.dtype, np.integer):
                raise ValueError(
                    "Detected integer count data in t_test. "
                    "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                )
            elif np.issubdtype(chunk.dtype, np.floating):
                non_zero = chunk.data[chunk.data > 0]
                is_count_like = non_zero.size > 0 and np.all(np.isclose(non_zero, np.round(non_zero)))
                if is_count_like:
                    raise ValueError(
                        "Detected count-like (integer-valued) floating point data in t_test. "
                        "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                    )
            break  # Only check the first chunk

        n_groups_total = len(groups)
        # Use float64 for accumulation to maintain numerical precision
        sums = np.zeros((n_groups_total, n_genes), dtype=np.float64)
        sumsq = np.zeros((n_groups_total, n_genes), dtype=np.float64)
        counts = np.zeros(n_groups_total, dtype=np.int64)
        expr_counts = np.zeros((n_groups_total, n_genes), dtype=np.int32)
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=cell_chunk_size, convert_to_dense=False
        ):
            slice_codes = label_codes[slc]
            csr = sp.csr_matrix(block)
            for code in np.unique(slice_codes):
                # Cells of a perturbation this run does not test are coded -1,
                # which would index the last group's row.
                if code < 0:
                    continue
                row_mask = slice_codes == code
                group_block = csr[row_mask, :]
                # Expression count: number of nonzero per gene
                expr_counts[code] += np.asarray(group_block.getnnz(axis=0), dtype=np.int32)
                # Sum and sumsq using sparse ops
                sums[code] += group_block.sum(axis=0).A1.astype(np.float64)
                sumsq[code] += group_block.power(2).sum(axis=0).A1.astype(np.float64)
                counts[code] += row_mask.sum()
    finally:
        backed.file.close()

    control_idx = group_index[control_label]
    control_n = counts[control_idx]
    if control_n == 0:
        raise ValueError("Control group contains no cells")

    control_mean = sums[control_idx] / control_n
    # Precompute control term for LFC calculation once (avoid recomputing per perturbation)
    control_mean_expm1 = np.expm1(control_mean) + 1e-9
    control_var = np.zeros_like(control_mean)
    if control_n > 1:
        control_var = (sumsq[control_idx] - (sums[control_idx] ** 2) / control_n) / (control_n - 1)
    control_var = np.clip(control_var, a_min=0, a_max=None)

    # Calculate control pts (proportion of cells expressing each gene)
    control_pts = np.divide(
        expr_counts[control_idx],
        control_n,
        out=np.zeros(n_genes, dtype=float),
        where=control_n > 0,
    )

    # Determine worker count for parallelization
    n_groups = len(candidates)
    worker_count = max(1, min(n_groups, _resolve_n_jobs(n_jobs)))

    # Prepare on-disk buffers for results
    shape = (n_groups, n_genes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")

    # Determine checkpoint interval
    eff_checkpoint_interval = _get_checkpoint_interval(n_groups, checkpoint_interval)
    candidate_to_idx = {label: idx for idx, label in enumerate(candidates)}

    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)

    # pts_rest describes the control arm, so it is one per-gene vector, not a
    # per-comparison matrix.
    pts_rest_vector = control_pts.astype(np.float32)

    _t_test_disk_estimate = warn_if_disk_space_low(
        _t_test_disk_bytes(*shape),
        output_path,
        context="t_test",
    )
    _messages.print_disk_estimate(verbose, "t_test", _t_test_disk_estimate)
    # Per-perturbation results live beside the output until the run finishes,
    # so an interrupted run resumes from its last checkpoint.
    run = ResumableRun(
        output_path,
        checkpoint_path,
        fingerprint=_de_fingerprint(path, call_args, method="t_test", candidates=candidates, n_genes=n_genes),
        # NaN, not 0: a perturbation that fails is never written, and a
        # p-value of 0 would report every gene as significant for it.
        arrays={
            "statistics": (shape, np.float64, np.nan),
            "pvalues": (shape, np.float64, np.nan),
            "pvalues_adj": (shape, np.float64, np.nan),
            "logfoldchanges": (shape, np.float32, np.nan),
            "effect_size": (shape, np.float32, np.nan),
            "pts": (shape, np.float32, 0),
            "order": (shape, np.int64, 0),
        },
        resume=resume,
    )
    # Earlier failures are retried, so only completions carry over.
    completed_labels: list[str] = run.progress.get("completed", [])
    candidates_to_run = [c for c in candidates if c not in set(completed_labels)]
    if run.resumed:
        _messages.vprint(verbose, "t_test", f"Resuming: {len(completed_labels)}/{n_groups} perturbations already done")
    stat_memmap = run.arrays["statistics"]
    pval_memmap = run.arrays["pvalues"]
    lfc_memmap = run.arrays["logfoldchanges"]
    effect_memmap = run.arrays["effect_size"]
    pts_memmap = run.arrays["pts"]
    order_memmap = run.arrays["order"]

    batch_size = worker_count
    effect_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
    stat_buffer = np.zeros((batch_size, n_genes), dtype=np.float64)
    pval_buffer = np.ones((batch_size, n_genes), dtype=np.float64)
    lfc_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
    pts_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
    order_buffer = np.zeros((batch_size, n_genes), dtype=np.int64)
    mean_buffer = np.zeros(n_genes, dtype=np.float64)
    var_buffer = np.zeros(n_genes, dtype=np.float64)
    se_buffer = np.zeros(n_genes, dtype=np.float64)
    lfc_work_buffer = np.zeros(n_genes, dtype=np.float64)  # Work buffer for in-place LFC
    n_tested_per_slot = np.zeros(batch_size, dtype=np.int32)

    def compute_perturbation(label: str, slot: int) -> None:
        idx = group_index[label]
        n_cells = counts[idx]
        if n_cells == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")

        np.divide(sums[idx], n_cells, out=mean_buffer)
        np.copyto(var_buffer, sumsq[idx])
        if n_cells > 1:
            np.subtract(var_buffer, np.square(sums[idx]) / n_cells, out=var_buffer)
            np.divide(var_buffer, n_cells - 1, out=var_buffer)
        else:
            var_buffer.fill(0)
        np.clip(var_buffer, a_min=0, a_max=None, out=var_buffer)

        np.subtract(mean_buffer, control_mean, out=effect_buffer[slot])
        effect_f32 = effect_buffer[slot].astype(np.float32, copy=False)

        # Compute variance terms for SE and Welch-Satterthwaite df
        var_term_pert = var_buffer / n_cells  # var_pert / n_pert
        var_term_ctrl = control_var / control_n  # var_ctrl / n_ctrl
            
        # SE = sqrt(var_pert/n_pert + var_ctrl/n_ctrl)
        np.add(var_term_pert, var_term_ctrl, out=se_buffer)
        np.sqrt(se_buffer, out=se_buffer)

        total_expr = expr_counts[idx] + expr_counts[control_idx]
        valid = (se_buffer > 0) & (total_expr >= min_cells_expressed)
        # Per-condition low-expression filter: drop genes that are
        # jointly low in BOTH the perturbation and control groups.
        low_both = _low_expr_in_both_mask(
            pert_expr_counts=expr_counts[idx],
            control_expr_counts=expr_counts[control_idx],
            pert_mean=mean_buffer,
            control_mean=control_mean,
            n_pert_cells=n_cells,
            n_control_cells=control_n,
            min_pct_ctrl=min_pct_ctrl,
            min_pct_pert=min_pct_pert,
            min_mean_ctrl=min_mean_ctrl,
            min_mean_pert=min_mean_pert,
        )
        valid &= ~low_both
        n_tested_per_slot[slot] = int(valid.sum())

        stat_buffer[slot].fill(np.nan)
        pval_buffer[slot].fill(np.nan)
        stat_buffer[slot][valid] = effect_f32[valid] / se_buffer[valid]
            
        # Welch-Satterthwaite degrees of freedom for Welch's t-test
        # df = (var1/n1 + var2/n2)^2 / ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))
        numerator = (var_term_pert + var_term_ctrl) ** 2
        denominator = np.zeros_like(numerator)
        if n_cells > 1:
            denominator += (var_term_pert ** 2) / (n_cells - 1)
        if control_n > 1:
            denominator += (var_term_ctrl ** 2) / (control_n - 1)
        # Avoid division by zero; set df to a large value when denominator is 0
        # Use np.divide with where to prevent NaN from 0/0
        df_welch = np.divide(
            numerator, denominator,
            out=np.full_like(numerator, 1e6),
            where=denominator > 0
        )
        # Clip df to reasonable bounds (minimum 1, no upper limit needed)
        df_welch = np.clip(df_welch, 1.0, None)
            
        # Use t-distribution for p-value calculation (matches scanpy's Welch's t-test)
        pval_buffer[slot][valid] = 2 * t_dist.sf(np.abs(stat_buffer[slot][valid]), df_welch[valid])

        np.divide(
            expr_counts[idx],
            n_cells,
            out=pts_buffer[slot],
            where=n_cells > 0,
            casting="unsafe",
        )

        order_buffer[slot] = np.argsort(-np.abs(stat_buffer[slot]))
        # Scanpy-compatible log2 fold change: log2((expm1(mean_group) + eps) / (expm1(mean_rest) + eps))
        # Use in-place operations to minimize temporary allocations
        np.expm1(mean_buffer, out=lfc_work_buffer)
        np.add(lfc_work_buffer, 1e-9, out=lfc_work_buffer)
        np.divide(lfc_work_buffer, control_mean_expm1, out=lfc_work_buffer)
        np.log2(lfc_work_buffer, out=lfc_work_buffer)
        lfc_buffer[slot] = lfc_work_buffer.astype(np.float32)

        # Untested genes carry NaN in every column derived from the
        # comparison; reporting 0 would assert "no change" about a gene
        # nobody tested.  pts / pts_rest describe the data and stay.
        if not valid.all():
            invalid = ~valid
            effect_buffer[slot][invalid] = np.nan
            lfc_buffer[slot][invalid] = np.nan

    # Track completed labels
    newly_completed = list(completed_labels)
    newly_failed: list[str] = []
    completed_set = set(completed_labels)
    n_processed = 0
        
    # Helper to save checkpoint
    def _save_t_test_checkpoint() -> None:
        checkpoint_data = {
            "total": n_groups,
            "completed": newly_completed,
            "failed": newly_failed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run.save(**checkpoint_data)

    if n_groups > 0:
        n_tested_list: list[int] = []
        with _create_progress_context(len(candidates_to_run), "t-test DE", verbose) as pbar:
            for batch_start in range(0, n_groups, batch_size):
                batch_labels = candidates[batch_start : batch_start + batch_size]
                # Filter to only labels that need processing
                batch_to_run = [l for l in batch_labels if l not in completed_set]
                    
                for local_idx, label in enumerate(batch_labels):
                    if label in completed_set:
                        continue
                    try:
                        compute_perturbation(label, local_idx)
                    except Exception as e:
                        logger.error(f"Failed perturbation {label}: {e}")
                        newly_failed.append(label)
                        continue

                for local_idx, label in enumerate(batch_labels):
                    if label in completed_set or label in newly_failed:
                        continue
                    global_idx = candidate_to_idx[label]
                    effect_memmap[global_idx] = effect_buffer[local_idx]
                    stat_memmap[global_idx] = stat_buffer[local_idx]
                    pval_memmap[global_idx] = pval_buffer[local_idx]
                    lfc_memmap[global_idx] = lfc_buffer[local_idx]
                    pts_memmap[global_idx] = pts_buffer[local_idx]
                    order_memmap[global_idx] = order_buffer[local_idx]

                    newly_completed.append(label)
                    n_tested_list.append(int(n_tested_per_slot[local_idx]))
                    _print_de_perturbation_verbose(verbose, label, int(n_tested_per_slot[local_idx]), n_genes)
                    n_processed += 1
                    pbar.update(1)
                    logger.debug(f"Completed perturbation: {label}")
                    
                # Save checkpoint after each batch
                if len(batch_to_run) > 0 and n_processed % eff_checkpoint_interval == 0:
                    _save_t_test_checkpoint()
            
        # Final checkpoint
        _save_t_test_checkpoint()
        logger.info(f"Completed {len(newly_completed)}/{n_groups} perturbations")
        _print_de_summary(verbose, "t-test DE", len(newly_completed), n_groups, n_tested_list, n_genes)

    pvalue_adj_memmap = run.arrays["pvalues_adj"]
    _adjust_pvalue_matrix(pval_memmap, method=corr_method, out=pvalue_adj_memmap)

    # Copy out of the memmaps: run.discard() deletes their files, and a view
    # would keep the deleted files mapped for as long as the result lives.
    stat_matrix = np.array(stat_memmap)
    pval_matrix = np.array(pval_memmap)
    pval_adj_matrix = np.array(pvalue_adj_memmap)
    lfc_matrix = np.array(lfc_memmap)
    effect_matrix = np.array(effect_memmap)
    pts_matrix = np.array(pts_memmap)
    order_matrix = np.array(order_memmap)
    del stat_memmap, pval_memmap, pvalue_adj_memmap, lfc_memmap, effect_memmap, pts_memmap, order_memmap
        
    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=stat_matrix,
        pvalues=pval_matrix,
        pvalues_adj=pval_adj_matrix,
        logfoldchanges=lfc_matrix,
        effect_size=effect_matrix,
        u_statistics=np.zeros(shape, dtype=np.float32),
        pts=pts_matrix,
        pts_rest=np.broadcast_to(pts_rest_vector, shape),
        order=order_matrix,
        groupby=perturbation_column,
        method="t_test",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
        result=None,
    )

    # Create AnnData with layer-based storage (avoid recarray-based rank_genes_groups
    # which fails with HDF5 header size limits for large group counts)
    var = pd.DataFrame({"pts_rest": pts_rest_vector}, index=gene_symbols)
    adata = ad.AnnData(effect_matrix, obs=obs, var=var)
    adata.layers["z_score"] = stat_matrix  # t-statistic (converges to z for large n)
    adata.layers["pvalue"] = pval_matrix
    adata.layers["pvalue_adj"] = pval_adj_matrix
    adata.layers["logfoldchanges"] = lfc_matrix
    adata.layers["pts"] = pts_matrix
    adata.uns["method"] = "t_test"
    adata.uns["control_label"] = control_label
    adata.uns["perturbation_column"] = perturbation_column
    adata.uns["pvalue_correction"] = corr_method
    _messages.print_saving(verbose, "t_test", output_path)
    adata.write(output_path)
    
    # Optionally write Scanpy-compatible rank_genes_groups structure
    if scanpy_format:
        _write_rank_genes_groups_hdf5(output_path, result)
    
    run.discard()  # the output is complete; the checkpoint and partial arrays are not needed
    result.result = AnnData(output_path)
    return result


def nb_glm_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    # ---- Data parameters ----
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    covariates: Iterable[str] | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    # ---- Size factor parameters ----
    size_factors: ArrayLike | None = None,
    size_factor_method: Literal["sparse", "deseq2"] = "sparse",
    size_factor_scope: Literal["global", "per_comparison"] = "global",
    scale_size_factors: bool = True,
    # ---- Dispersion parameters ----
    dispersion: float | None = None,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    dispersion_scope: Literal["global", "per_comparison"] = "global",
    share_dispersion: bool = False,
    use_map_dispersion: bool = True,
    shrink_dispersion: bool = True,
    # ---- Optimization parameters ----
    optimization_method: Literal["irls", "lbfgsb"] = "lbfgsb",
    max_iter: int = 25,
    tol: float = 1e-6,
    min_mu: float = 0.5,
    poisson_init_iter: int = 5,
    chunk_size: int | None = None,
    irls_batch_size: int | None = 128,
    # ---- Filtering parameters ----
    min_cells_expressed: int = 0,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_cells_ctrl: int = 1,
    min_cells_pert: int = 1,
    min_pct_both: float | None = None,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
    min_total_count: float = 1.0,
    cook_filter: bool = False,
    # ---- Output parameters ----
    lfc_shrinkage_type: Literal["apeglm", "none"] = "none",
    lfc_base: Literal["log2", "ln"] = "log2",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    se_method: Literal["sandwich", "fisher"] = "sandwich",
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    scanpy_format: bool = False,
    verbose: int | bool = True,
    profiling: bool = False,
    # ---- Resume/Memory parameters ----
    resume: bool = False,
    checkpoint_interval: int | None = None,
    memory_limit_gb: float | None = None,
    max_dense_fraction: float = 0.3,
    n_jobs: int | None = None,
    use_control_cache: bool = True,
    freeze_control: bool | None = None,
    force: bool = False,
) -> RankGenesGroupsResult:
    """Perform negative binomial GLM differential expression test.
    
    Returns a RankGenesGroupsResult containing differential expression statistics.
    Uses a negative binomial GLM framework that can incorporate covariates. 
    Results are stored in an h5ad file with layers containing the statistics.
    
    The RankGenesGroupsResult implements the Mapping interface, so it can be used 
    like a dict: `result[perturbation_label]` returns a DifferentialExpressionResult
    for that perturbation.
    
    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        raw count data.
    perturbation_column
        Column in `adata.obs` indicating perturbation labels.
    groupby
        Alias for ``perturbation_column`` (Scanpy-compatible).
        Mutually exclusive with ``perturbation_column``.
    control_label
        Label for the control/reference group. If None, infers from common patterns.
    reference
        Alias for ``control_label`` (Scanpy-compatible).
        Mutually exclusive with ``control_label``.
    covariates
        Additional columns in `adata.obs` to include as covariates in the GLM.
    gene_name_column
        Column in `adata.var` with gene symbols. If None, uses `adata.var_names`.
    perturbations
        Specific perturbations to test. If None, tests all non-control groups.
    size_factors
        Optional array of per-cell size factors. If None, computes size factors
        using the method specified by ``size_factor_method``.
    size_factor_method
        Method for computing size factors when ``size_factors`` is None.

        - ``"sparse"``: Sparse-aware median-of-ratios (default). Computes geometric
          means using only non-zero values, suitable for sparse single-cell data.
        - ``"deseq2"``: Classic DESeq2/PyDESeq2 style. Uses only genes expressed in
          ALL cells (typically ~50-100 genes). Provides better numerical alignment
          with PyDESeq2 results but may be less robust for very sparse data.
    size_factor_scope
        Scope for size factor computation.

        - ``"global"`` (default): Compute size factors once on the full dataset.
          Recommended for CRISPR screens where all cells come from the same
          experiment and share a common sequencing depth distribution. Faster
          when combined with use_control_cache=True. Note: produces different
          results from PyDESeq2 (rho ~ 0.7-0.8) which uses per-comparison normalization.
        - ``"per_comparison"``: Compute size factors separately for each control +
          perturbation comparison. This matches PyDESeq2's behavior exactly,
          leading to near-perfect LFC, statistic, and p-value concordance
          (rho > 0.97 on Tian-crispra, rho > 0.99 on Adamson_subset). Use this when
          PyDESeq2 compatibility is required or for bulk RNA-seq style analysis.
    scale_size_factors
        If True (default), scale size factors so their geometric mean equals 1.
        This is the standard DESeq2/crispyx behavior. If False, use raw
        median-of-ratios size factors without rescaling, which matches
        PyDESeq2's default behavior and can improve numerical alignment.
    dispersion
        Fixed dispersion parameter for negative binomial. If None, estimates per gene.
    dispersion_method
        Method for estimating dispersion when ``dispersion`` is None.

        - ``"moments"``: Method-of-moments (fast but less accurate).
        - ``"cox-reid"``: Cox-Reid adjusted profile likelihood (slower but more
          accurate, similar to DESeq2). This is the default.
    dispersion_scope
        Scope for dispersion estimation.

        - ``"global"`` (default): Precompute dispersion once using all cells (control +
          all perturbations). This is ~10x faster for multi-perturbation datasets
          since MAP dispersion estimation is done once instead of per-comparison.
          Recommended when perturbation effects are expected to be small relative
          to baseline expression (typical for CRISPR screens).
        - ``"per_comparison"``: Estimate dispersion separately for each control +
          perturbation comparison. More accurate when perturbations cause large
          changes in gene expression variance, but significantly slower.
    share_dispersion
        If True, estimate dispersion once using all cells, then use the same
        dispersion values for all Wald tests. If False (default), estimate
        dispersion separately for each perturbation comparison.
    use_map_dispersion
        If True (default), use MAP dispersion estimation with mean-dispersion trend.
        If False, use MLE dispersion without trend-based shrinkage.
    shrink_dispersion
        If True, fit a mean-dispersion trend and shrink gene-wise dispersions
        toward the trend using an empirical Bayes prior.
    optimization_method
        Method for coefficient optimization.

        - ``"lbfgsb"``: L-BFGS-B optimization (PyDESeq2 style, default). Directly
          optimizes the negative binomial log-likelihood.
        - ``"irls"``: Iteratively Reweighted Least Squares (Fisher scoring). The
          classic GLM fitting approach.
    max_iter
        Maximum iterations for GLM fitting.
    tol
        Convergence tolerance for GLM fitting.
    min_mu
        Minimum mean threshold for IRLS numerical stability. Predicted means
        are clamped to max(min_mu, predicted) during fitting to prevent
        numerical instability from very small means. Default: 0.5, matching
        PyDESeq2's default. Set to 0.0 to disable clamping.
    poisson_init_iter
        Initial Poisson iterations before switching to negative binomial.
    chunk_size
        Number of genes to process per chunk (memory vs. speed tradeoff). Smaller
        values stream more, reducing peak memory at the cost of additional I/O.
    irls_batch_size
        Maximum number of genes to densify per IRLS step. Keep this small to
        limit per-iteration memory when working with large sparse matrices. Set
        to ``None`` to process each chunk without additional batching.
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for
        testing. Genes below this threshold are untested: ``NaN`` in
        ``score`` / ``pvalue`` / ``logfoldchanges`` / ``effect_size``, with
        ``pts`` / ``pts_rest`` still populated.
    min_total_count
        Minimum total count across all cells for a gene to be tested.
    min_pct_ctrl
        Minimum fraction of expressing cells for the *control* side. A gene is
        excluded only when *both* sides are jointly low. Default ``0.01``.
    min_pct_pert
        Minimum fraction of expressing cells for the *perturbed* side.
        Default ``0.002``. Combined with ``min_mean_pert`` this forms a dual
        condition that is more robust to doublet / ambient-RNA artefacts.
    min_pct_both
        If not ``None``, overrides both ``min_pct_ctrl`` and
        ``min_pct_pert`` with the same value.
    min_mean_ctrl
        Minimum mean (size-factor-normalised) expression for the *control* side.
        Default ``0.05``. Excluded genes appear as NaN in ``pvalue`` /
        ``effect`` / ``logfc`` / ``se``; ``pts`` and ``mean`` remain populated.
    min_mean_pert
        Minimum mean expression for the *perturbed* side. Default ``0.005``.
    min_cells_ctrl, min_cells_pert
        Minimum number of expressing cells required in the control and
        perturbed arms for a (gene, perturbation) pair to be tested. Both
        default to ``1``: a pair with no counts at all in one arm has no
        finite effect to estimate, so it is reported as untested (``NaN``
        effect, statistic and p-value) and excluded from the multiple-testing
        correction rather than given a bound-determined estimate. See
        :func:`crispyx._statistics._nonestimable_glm_mask` for why, and for
        why the two sides are set separately (CRISPRi and CRISPRa want
        opposite asymmetries). Set either to ``0`` to disable that side.

        The observation itself is not lost: ``pts`` and ``pts_rest`` are
        populated for every gene, so a pair expressed in 0% of perturbed and
        95% of control cells is plainly visible there. ``lfc_shrinkage_type``
        does not change this: an excluded pair is dropped before any fit, so
        it stays ``NaN`` under ``"apeglm"`` as well.
    cook_filter
        Whether to apply Cook's distance outlier filtering when available.
    lfc_shrinkage_type
        Type of log-fold change shrinkage to apply.

        - ``"none"``: No shrinkage (default).
        - ``"apeglm"``: Adaptive shrinkage using Cauchy prior (PyDESeq2-compatible).
          Preserves large effects while shrinking small/uncertain effects toward
          zero. Also updates standard errors to reflect posterior uncertainty.
    lfc_base
        Log base for fold change output.

        - ``"log2"`` (default): Output log2 fold change, matching PyDESeq2/edgeR.
        - ``"ln"``: Output natural log fold change (raw GLM coefficients).

        Standard error is also converted to match the selected log base.
        Wald statistics remain unchanged since both LFC and SE are scaled equally.
    corr_method
        Method for p-value correction: ``"benjamini-hochberg"`` or ``"bonferroni"``.
    se_method
        Method for computing standard errors.

        - ``"sandwich"`` (default): Sandwich estimator SE = sqrt(c' @ H @ M @ H @ c).
          More robust to model misspecification.
        - ``"fisher"``: Standard Fisher information SE = sqrt(diag(inv(X'WX + ridge*I))).
          Matches PyDESeq2's approach for better p-value parity.
    data_name
        Custom name for output file. If None, uses "nb_glm" suffix.
    output_path
        Exact path for the output h5ad file. When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
        Deprecated; use ``output_path`` instead. Will be removed in the
        next major version.
    scanpy_format
        If True, write Scanpy-compatible ``uns['rank_genes_groups']`` structure
        in addition to the layer-based storage. Adds ~2-6 seconds of I/O overhead
        for large datasets. Default False for performance.
    verbose
        If True, show a progress bar for perturbation fitting and log per-perturbation
        completion at DEBUG level. Requires tqdm to be installed for progress bar.
    profiling
        If True, enable timing and memory profiling. When enabled, stores
        profiling data in ``adata.uns["profiling"]`` with fields
        ``fit_seconds``, ``fit_peak_memory_mb``, and ``profiling_enabled``.
        When False (default), ``adata.uns["profiling"]`` is set to ``"NA"`` to
        avoid profiling overhead in production.
    resume
        If True, continue an interrupted run from its last checkpoint,
        skipping the perturbations it had finished.
        Partial results are kept beside the output, in a hidden
        ``.<output name>.resume`` directory, together with the checkpoint
        ``<output>.progress.json``; both are removed once the output is
        written. A run resumes only from a checkpoint written by a call with
        the same input file, perturbations and result-affecting parameters;
        any other checkpoint is discarded and the run starts over. A
        completed output is loaded as usual (unless ``force=True``).
    checkpoint_interval
        Number of perturbations to process between checkpoint saves. If None,
        auto-determined based on dataset size (1 for <100 perturbations, 10 for
        <1000, 50 for larger).
    memory_limit_gb
        Optional memory limit in GB. If provided, this is used together with
        available system memory to determine when to switch to streaming mode
        for global dispersion estimation. The effective limit is
        min(available_memory, memory_limit_gb). Default is None (use system memory only).
    max_dense_fraction
        Maximum fraction of available memory to use for dense matrix operations.
        If the estimated memory for densifying the full cell×gene matrix exceeds
        max_dense_fraction × min(available_memory, memory_limit_gb), the function
        switches to streaming mode. Default is 0.3 (30% of available memory).
    n_jobs
        Number of parallel workers for fitting GLMs across perturbations.
        If None, uses all available cores. If 1, runs sequentially.
        If -1, uses all available cores.
    use_control_cache
        If True (default), precompute control cell statistics (intercept, weights,
        XᵀWX contributions) once and reuse them across all perturbation comparisons.
        This can significantly reduce memory and computation time when there are
        many perturbations and the control group is large. Only applies when
        no covariates are specified and size_factor_scope="global".
    freeze_control
        Whether to use frozen control sufficient statistics instead of raw control
        matrix for parallel fitting. This dramatically reduces per-worker memory
        from ~5GB to ~1MB for large datasets, enabling more parallel workers.
        
        - None (default): Auto-detect based on dataset size. Frozen control is
          enabled when control_matrix serialization would limit workers to <4,
          AND the required settings (dispersion_scope='global', shrink_dispersion=True)
          are met. For most large datasets (>500K cells), this auto-enables.
        - True: Force frozen control mode. Raises ValueError if requirements not met.
        - False: Disable frozen control (use raw control matrix).
        
        Memory efficiency: Per-worker pickle size is reduced from (control_n × n_genes × 8)
        bytes to just ~1MB of sufficient statistics (W_sum, Wz_sum arrays).
        
        Example: For Replogle-GW-k562 (75K control cells × 8K genes):
        - Without freeze_control: ~4.7 GB per worker → 2 workers max @ 128GB
        - With freeze_control: ~1 MB per worker → 32 workers @ 128GB
        - Time reduction: ~300 hours → ~10 hours
        
        Requirements (enforced when True, auto-checked when None):
        - dispersion_scope="global" (per-comparison dispersion needs raw matrix)
        - shrink_dispersion=True (ensures global dispersion is computed)
        - use_control_cache=True (required for caching)
        
        Technical note: The intercept (β₀) is frozen to the control-only estimate.
        This is valid because control cells have perturbation indicator = 0, so
        μ_control = exp(β₀ + offset) is independent of the perturbation effect.
    force
        If True, rerun the analysis even when the output h5ad file already
        exists. If False (default), load and return the existing result
        instead of rerunning.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].effect_size`, `result[label].pvalue`, etc. The h5ad file
        path is available at `result.result_path`.

        The h5ad file holds one row per perturbation. ``X`` is the fold
        change in ``lfc_base`` (shrunk under ``lfc_shrinkage_type="apeglm"``).
        Layers: ``z_score``, ``pvalue``, ``pvalue_adj``, ``standard_error``,
        ``intercept`` (ln scale), ``pts``, ``converged`` (bool) and
        ``iterations`` (integer); ``logfoldchange_raw`` holds the MLE fold
        change only when ``X`` is shrunk, and with a log2 base
        ``logfoldchange_raw_ln`` / ``standard_error_ln`` keep the ln-scale
        values :func:`shrink_lfc` needs. Per-gene quantities are ``var``
        columns: ``pts_rest``, and, when a global dispersion is fitted,
        ``dispersion`` and ``dispersion_trend``; a per-comparison dispersion
        is stored as the layers ``dispersion``, ``dispersion_raw`` and
        ``dispersion_trend`` instead.
    """
    call_args = dict(locals())  # for the resume fingerprint, before any other local
    perturbation_column, control_label, min_pct_ctrl, min_pct_pert = _resolve_de_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        min_pct_both=min_pct_both,
        min_pct_ctrl=min_pct_ctrl,
        min_pct_pert=min_pct_pert,
        fn_name="nb_glm_test",
    )

    # Validate min_mu parameter
    if min_mu < 0:
        raise ValueError(f"min_mu must be >= 0, got {min_mu}")

    covariates = list(covariates or [])

    # Initialize profiler if enabled (timing + memory sampling)
    profiler = None
    if profiling:
        from .profiling import Profiler
        profiler = Profiler(timing=True, memory=True, memory_method="rss", sampling=True)
        profiler.start("total")
        profiler.start("fit")  # Start fit timing (excludes shrinkage which is separate)

    # Check if dataset needs sorting for efficient I/O
    # Large datasets with many perturbations benefit from having cells sorted
    # by perturbation label, enabling contiguous reads instead of random access
    path = resolve_data_path(data)

    # Early-exit: if output already exists and force=False, reload without rerunning.
    # Use the original (pre-sort) path for output_path resolution so the location
    # is predictable for the caller regardless of internal sorting.
    _candidate_output_path = resolve_output_path(
        path, suffix="nb_glm", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )
    if (r := _try_load_existing_de_result(
        _candidate_output_path, force=force, verbose=verbose,
        method_name="nb_glm", memory_limit_gb=memory_limit_gb,
    )):
        return r

    # Warn if the on-disk matrix is stored in CSC format – row-slicing
    # (used by size-factor computation and control-matrix loading) is
    # extremely slow on CSC.  Recommend the CSR standardized file.
    _storage_fmt = get_matrix_storage_format(path)
    if _storage_fmt == "csc":
        warnings.warn(
            f"The input file '{path.name}' stores its matrix in CSC format. "
            "NB-GLM performs row-wise access (size factors, control matrix, "
            "per-perturbation slices) which is very slow on CSC. "
            "Use the CSR-format standardized file for much better performance.",
            UserWarning,
            stacklevel=2,
        )

    if needs_sorting_for_nbglm(path, perturbation_column=perturbation_column):
        sorted_path = path.parent / f"{path.stem}_sorted.h5ad"
        if not sorted_path.exists():
            logger.info(
                f"Large dataset detected with scattered cells. "
                f"Sorting by perturbation for efficient I/O..."
            )
            path = sort_by_perturbation(
                path,
                perturbation_column=perturbation_column,
                control_label=control_label,
                output_path=sorted_path,
            )
        else:
            logger.info(f"Using existing sorted dataset: {sorted_path}")
            path = sorted_path

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        obs_df = backed.obs[[perturbation_column] + covariates].copy()
        labels = obs_df[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(list(labels), control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        if control_n == 0:
            raise ValueError("Control group contains no cells")
        for label in candidates:
            if not np.any(labels == label):
                raise ValueError(f"Perturbation '{label}' contains no cells")
        
        # Check if file is sorted by perturbation for efficient I/O
        perturbation_boundaries = None
        if "sorting_metadata" in backed.uns:
            metadata = backed.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                perturbation_boundaries = metadata.get("perturbation_boundaries", {})
                if perturbation_boundaries:
                    logger.debug(
                        f"Using sorted file with {len(perturbation_boundaries)} contiguous perturbation groups"
                    )
    finally:
        backed.file.close()

    n_cells_total = obs_df.shape[0]
    
    # Calculate adaptive chunk_size if not provided
    # Only reduces chunk_size when memory would be exceeded; successful datasets keep default (256)
    if chunk_size is None:
        chunk_size = calculate_nb_glm_chunk_size(
            n_obs=n_cells_total,
            n_vars=n_genes,
            n_groups=len(candidates),
            memory_limit_gb=memory_limit_gb,
        )
        _messages.vprint(verbose, "nb_glm_test", f"chunk_size={chunk_size} (auto)")

    # For per_comparison size factors, we skip the expensive global computation
    # since size factors will be recomputed per-comparison anyway.
    # We use dummy size factors here (will be overwritten per-comparison).
    if size_factor_scope == "per_comparison" and size_factors is None:
        # Dummy size factors - will be replaced in worker
        size_factors = np.ones(n_cells_total, dtype=np.float64)
        logger.debug("Skipping global size factor computation for per_comparison mode")
    elif size_factors is None:
        if size_factor_method == "deseq2":
            size_factors = _deseq2_style_size_factors(
                path, chunk_size=chunk_size, scale=scale_size_factors
            )
        else:  # "sparse" (default)
            size_factors = _median_of_ratios_size_factors(
                path, chunk_size=chunk_size, scale=scale_size_factors
            )
    else:
        size_factors = _validate_size_factors(
            size_factors, n_cells_total, scale=scale_size_factors
        )

    offset = np.log(np.clip(size_factors, 1e-8, None))

    # =========================================================================
    # Independent fitting mode: per-perturbation approach
    # =========================================================================
    beta_cov = None
    beta_intercept = None
    global_dispersion = None

    # -------------------------------------------------------------------------
    # Worker function for parallel fitting of a single perturbation group
    # -------------------------------------------------------------------------
    def _fit_perturbation_worker(
        group_idx: int,
        label: str,
        path: str | Path,
        labels: np.ndarray,
        control_mask: np.ndarray,
        control_matrix: np.ndarray | sp.csr_matrix,
        control_expr_counts: np.ndarray,
        control_n: int,
        obs_df: pd.DataFrame,
        covariates: list[str],
        size_factors: np.ndarray,
        offset: np.ndarray,
        n_genes: int,
        min_cells_expressed: int,
        min_pct_ctrl: float,
        min_pct_pert: float,
        min_mean_ctrl: float,
        min_mean_pert: float,
        min_cells_ctrl: int,
        min_cells_pert: int,
        min_total_count: float,
        max_iter: int,
        tol: float,
        min_mu: float,
        poisson_init_iter: int,
        dispersion_method: str,
        global_dispersion: np.ndarray | None,
        shrink_dispersion: bool,
        use_map_dispersion: bool,
        lfc_shrinkage_type: str,
        pts_rest_shared: np.ndarray,
        full_X: np.ndarray | sp.csr_matrix | None = None,
        per_comparison_sf: bool = False,
        se_method: str = "sandwich",
        perturbation_boundaries: dict | None = None,
    ) -> dict:
        """Fit NB-GLM for a single perturbation group and return results."""
        group_mask = labels == label
        subset_mask = control_mask | group_mask
        subset_obs = obs_df.iloc[subset_mask]
        indicator = group_mask[subset_mask].astype(np.float64)
        
        # Compute per-comparison size factors if requested (matches PyDESeq2)
        if per_comparison_sf and full_X is not None:
            subset_sf = _compute_subset_size_factors(full_X, subset_mask, scale=True)
            subset_size_factors = subset_sf
            subset_offset = np.log(np.clip(subset_sf, 1e-8, None))
        else:
            subset_size_factors = np.asarray(size_factors)[subset_mask]
            subset_offset = offset[subset_mask] if offset is not None else np.log(np.clip(subset_size_factors, 1e-8, None))
        
        # Build design matrix
        
        # Build design matrix
        design, design_columns = build_design_matrix(
            subset_obs,
            covariate_columns=covariates,
            perturbation_indicator=indicator,
            intercept=True,
        )
        perturbation_column_index = design_columns.index("perturbation")
        
        control_subset_mask = control_mask[subset_mask]
        group_subset_mask = group_mask[subset_mask]
        group_n = int(group_subset_mask.sum())
        subset_n = int(subset_mask.sum())
        n_control = int(control_subset_mask.sum())

        # Load perturbation group cells
        # Use slice-based access if file is sorted, otherwise use mask
        backed = read_backed(path)
        try:
            if perturbation_boundaries is not None and label in perturbation_boundaries:
                # Slice-based access for sorted files (contiguous, fast)
                start, end = perturbation_boundaries[label]
                group_matrix = backed.X[start:end, :]
            else:
                # Mask-based access for unsorted files (random, slower)
                group_matrix = backed.X[group_mask, :]
            if sp.issparse(group_matrix):
                group_matrix = sp.csr_matrix(group_matrix, dtype=np.float64)
            else:
                group_matrix = np.asarray(group_matrix, dtype=np.float64)
        finally:
            backed.file.close()

        # Combine control and group matrices
        if sp.issparse(control_matrix) and sp.issparse(group_matrix):
            stacked = sp.vstack([control_matrix, group_matrix])
            reorder = np.empty(subset_n, dtype=np.int32)
            reorder[np.where(control_subset_mask)[0]] = np.arange(n_control)
            reorder[np.where(group_subset_mask)[0]] = np.arange(n_control, n_control + group_n)
            subset_matrix = sp.csr_matrix(stacked[reorder, :])
        else:
            if sp.issparse(control_matrix):
                ctrl = control_matrix.toarray()
            else:
                ctrl = control_matrix
            if sp.issparse(group_matrix):
                grp = group_matrix.toarray()
            else:
                grp = group_matrix
            stacked = np.vstack([ctrl, grp])
            reorder = np.empty(subset_n, dtype=np.int32)
            reorder[np.where(control_subset_mask)[0]] = np.arange(n_control)
            reorder[np.where(group_subset_mask)[0]] = np.arange(n_control, n_control + group_n)
            subset_matrix = stacked[reorder, :]

        # Compute expression counts
        if sp.issparse(group_matrix):
            group_expr_counts = np.asarray(group_matrix.getnnz(axis=0)).ravel()
        else:
            group_expr_counts = np.sum(group_matrix > 0, axis=0)

        # Per-group means in size-factor-normalised count units. These feed
        # both the per-condition low-expression filter and ``result["mean"]``
        # below; computing them once avoids a duplicate matrix pass.
        if sp.issparse(group_matrix):
            group_mean = (
                np.asarray(group_matrix.sum(axis=0)).ravel() / max(group_n, 1)
            )
        else:
            group_mean = np.asarray(group_matrix.sum(axis=0)).ravel() / max(group_n, 1)

        if sp.issparse(control_matrix):
            control_mean_raw = (
                np.asarray(control_matrix.sum(axis=0)).ravel() / max(control_n, 1)
            )
        else:
            control_mean_raw = np.asarray(control_matrix.sum(axis=0)).ravel() / max(control_n, 1)

        total_expr_counts = control_expr_counts + group_expr_counts
        valid_mask = total_expr_counts >= min_cells_expressed
        # Per-condition low-expression filter (jointly low in BOTH groups)
        low_both = _low_expr_in_both_mask(
            pert_expr_counts=group_expr_counts,
            control_expr_counts=control_expr_counts,
            pert_mean=group_mean,
            control_mean=control_mean_raw,
            n_pert_cells=group_n,
            n_control_cells=control_n,
            min_pct_ctrl=min_pct_ctrl,
            min_pct_pert=min_pct_pert,
            min_mean_ctrl=min_mean_ctrl,
            min_mean_pert=min_mean_pert,
        )
        valid_mask = valid_mask & ~low_both
        # A pair with no counts in one arm has no finite effect to estimate;
        # see _nonestimable_glm_mask.  Excluded here rather than fitted, so it
        # is reported as untested instead of carrying a bound-determined
        # effect and an artefactual p-value into the FDR correction.
        valid_mask = valid_mask & ~_nonestimable_glm_mask(
            pert_expr_counts=group_expr_counts,
            control_expr_counts=control_expr_counts,
            min_cells_ctrl=min_cells_ctrl,
            min_cells_pert=min_cells_pert,
        )
        valid_indices = np.where(valid_mask)[0]

        # Initialize result arrays
        result = {
            "group_idx": group_idx,
            "n_tested": int(valid_mask.sum()),
            "effect": np.full(n_genes, np.nan, dtype=np.float64),
            "statistic": np.full(n_genes, np.nan, dtype=np.float64),
            "pvalue": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "intercept": np.full(n_genes, np.nan, dtype=np.float64),  # MLE intercept for shrink_lfc
            "se": np.full(n_genes, np.nan, dtype=np.float64),
            "pts": np.zeros(n_genes, dtype=np.float32),
            "pts_rest": np.zeros(n_genes, dtype=np.float32),
            "dispersion": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_trend": np.full(n_genes, np.nan, dtype=np.float64),
            "mean": np.zeros(n_genes, dtype=np.float64),
            "iterations": np.zeros(n_genes, dtype=np.int32),
            "converged": np.zeros(n_genes, dtype=bool),
        }

        # Compute pts
        pts = np.divide(
            group_expr_counts,
            group_n,
            out=np.zeros(n_genes, dtype=np.float32),
            where=group_n > 0,
        )
        # Fractions of expressing cells are descriptions of the data, not
        # inferences from a fit, so they are reported for every gene including
        # the ones no effect is estimated for.  A pair expressed in 0% of
        # perturbed and 95% of control cells is a complete knockdown whose
        # effect is not estimable (see _nonestimable_glm_mask); zeroing these
        # would leave nothing to see it by.
        result["pts"] = pts.astype(np.float32)
        result["pts_rest"] = pts_rest_shared.astype(np.float32)

        # Compute mean expression
        if sp.issparse(subset_matrix):
            normalized = subset_matrix.multiply(1.0 / subset_size_factors[:, None])
            mean_expr = np.asarray(normalized.sum(axis=0)).ravel() / subset_n
        else:
            normalized = subset_matrix / subset_size_factors[:, None]
            mean_expr = normalized.sum(axis=0) / subset_n
        result["mean"] = mean_expr

        if not np.any(valid_mask):
            return result

        # Fit valid genes
        fit_matrix = subset_matrix[:, valid_mask]
        
        # A categorical covariate (batch, donor, lane) is one-hot encoded into
        # the design, which makes the per-gene Hessian arrowhead-structured.
        # fit_nb_glm_batch_auto takes the structured path when the number of
        # levels makes it worthwhile and the dense path otherwise; either way
        # the result is in the column order of `design`.  The intercept and the
        # perturbation column are protected from the group block because the
        # perturbation coefficient is the quantity under test and must not be
        # subject to the group-coefficient clip.
        batch_result = fit_nb_glm_batch_auto(
            design,
            fit_matrix,
            offset=subset_offset,
            protected_columns=(0, perturbation_column_index),
            max_iter=max_iter,
            tol=tol,
            poisson_init_iter=poisson_init_iter,
            dispersion_method=dispersion_method,
            min_mu=min_mu,
            min_total_count=min_total_count,
        )

        # Extract results
        result["converged"][valid_indices] = batch_result.converged
        result["iterations"][valid_indices] = batch_result.n_iter
        result["dispersion_raw"][valid_indices] = batch_result.dispersion

        coefs = batch_result.coef[:, perturbation_column_index]
        ses = batch_result.se[:, perturbation_column_index]
        intercepts = batch_result.coef[:, 0]  # Intercept is always first column

        valid_results = (
            batch_result.converged
            & np.isfinite(coefs)
            & np.isfinite(ses)
            & (ses > 0)
        )
        
        for local_idx, gene_idx in enumerate(valid_indices):
            if not valid_results[local_idx]:
                continue
            coef = coefs[local_idx]
            se = ses[local_idx]
            statistic = coef / se
            pvalue = float(2.0 * norm.sf(abs(statistic)))
            result["statistic"][gene_idx] = statistic
            result["pvalue"][gene_idx] = pvalue
            result["logfc"][gene_idx] = coef
            result["se"][gene_idx] = se
            result["intercept"][gene_idx] = intercepts[local_idx]  # Store fitted intercept

        # Handle dispersion
        if global_dispersion is not None:
            result["dispersion_raw"][:] = global_dispersion
            result["dispersion"][:] = global_dispersion
            result["dispersion_trend"][:] = global_dispersion
        elif shrink_dispersion:
            trend = fit_dispersion_trend(result["mean"], result["dispersion_raw"])
            result["dispersion_trend"] = trend
            if use_map_dispersion:
                # Use PyDESeq2-style MAP estimation with proper fitted values
                # First, compute mu from the fitted model coefficients
                if sp.issparse(subset_matrix):
                    Y = np.asarray(subset_matrix.todense(), dtype=np.float64)
                else:
                    Y = np.asarray(subset_matrix, dtype=np.float64)
                
                # Get fitted values from the model: mu = exp(X @ beta + offset)
                # batch_result.coef has shape (n_valid_genes, n_design_cols)
                # design has shape (n_cells, n_design_cols)
                n_subset = subset_n
                mu = np.zeros((n_subset, n_genes), dtype=np.float64)
                
                # For valid genes, compute mu from fitted coefficients
                for local_idx, gene_idx in enumerate(valid_indices):
                    if batch_result.converged[local_idx]:
                        # eta = X @ beta + offset
                        eta = design @ batch_result.coef[local_idx, :] + offset[subset_mask]
                        mu[:, gene_idx] = np.exp(np.clip(eta, -30, 30))
                
                # For invalid genes, use normalized counts as fallback
                invalid_genes = ~np.isin(np.arange(n_genes), valid_indices[batch_result.converged])
                if np.any(invalid_genes):
                    mu[:, invalid_genes] = Y[:, invalid_genes] / subset_size_factors[:, None]
                
                mu = np.maximum(mu, 1e-10)
                # Use PyDESeq2-style bounds: max(10, n_cells)
                max_disp = max(10.0, float(n_subset))
                disp_map = estimate_dispersion_map(
                    Y, mu, trend, max_disp=max_disp
                )
                result["dispersion"] = disp_map
                
                # CRITICAL: Recompute SE using MAP dispersion (PyDESeq2 style)
                # SE from IRLS was computed using MoM dispersion, but Wald test
                # should use MAP dispersion for proper variance estimation
                ridge = 1e-6
                for local_idx, gene_idx in enumerate(valid_indices):
                    if not valid_results[local_idx]:
                        continue
                    # Compute weights with MAP dispersion: W = mu / (1 + mu * disp)
                    mu_gene = mu[:, gene_idx]
                    disp_gene = disp_map[gene_idx]
                    W = mu_gene / (1.0 + mu_gene * disp_gene)
                    # Compute (X'WX + ridge*I)^{-1}
                    XtW = design.T * W[None, :]
                    XtWX = XtW @ design
                    XtWX += ridge * np.eye(design.shape[1])
                    try:
                        inv_XtWX = np.linalg.inv(XtWX)
                        se_new = np.sqrt(np.maximum(inv_XtWX[perturbation_column_index, perturbation_column_index], 1e-10))
                    except np.linalg.LinAlgError:
                        se_new = result["se"][gene_idx]  # Keep original SE on failure
                    # Update SE, statistic, and p-value
                    coef = result["logfc"][gene_idx]
                    result["se"][gene_idx] = se_new
                    statistic = coef / se_new
                    result["statistic"][gene_idx] = statistic
                    result["pvalue"][gene_idx] = float(2.0 * norm.sf(abs(statistic)))
            else:
                result["dispersion"] = shrink_dispersions(result["dispersion_raw"], trend)
        else:
            result["dispersion_trend"] = result["dispersion_raw"].copy()
            result["dispersion"] = result["dispersion_raw"].copy()

        result["logfc_raw"] = result["logfc"].copy()
        if lfc_shrinkage_type == "apeglm":
            # Full NB-GLM re-fitting with Cauchy prior (matches PyDESeq2)
            # Build mle_coef matrix (n_params, n_genes) from batch_result
            n_params = design.shape[1]
            mle_coef = np.full((n_params, n_genes), np.nan, dtype=np.float64)
            for local_idx, gene_idx in enumerate(valid_indices):
                if valid_results[local_idx]:
                    mle_coef[:, gene_idx] = batch_result.coef[local_idx, :]
            
            # Densify the subset matrix if sparse
            if sp.issparse(subset_matrix):
                counts_dense = subset_matrix.toarray()
            else:
                counts_dense = np.asarray(subset_matrix, dtype=np.float64)
            
            # Call full apeGLM shrinkage with L-BFGS-B re-fitting
            # NOTE: PyDESeq2's lfc_shrink does NOT use min_mu during shrinkage
            shrunk_coef, shrunk_se_arr, shrink_converged = shrink_lfc_apeglm(
                counts=counts_dense,
                design_matrix=design,
                size_factors=subset_size_factors,
                dispersion=result["dispersion"],
                mle_coef=mle_coef,
                mle_se=result["se"],
                shrink_index=perturbation_column_index,
                prior_scale=None,  # Estimate globally from MLE LFC distribution
                n_jobs=1,  # Single-threaded within worker (parallelism at perturbation level)
                min_mu=0.0,  # No min_mu - match PyDESeq2's lfc_shrink behavior
            )
            # Extract shrunken LFC (perturbation coefficient)
            result["logfc"] = shrunk_coef[perturbation_column_index, :]
            result["effect"] = result["logfc"].copy()
            result["se"] = shrunk_se_arr  # Posterior SE from inverse Hessian
        else:
            result["effect"] = result["logfc"].copy()

        return result
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Optimized worker using precomputed control cache
    # -------------------------------------------------------------------------
    def _fit_perturbation_worker_cached(
        group_idx: int,
        label: str,
        path: str | Path,
        labels: np.ndarray,
        control_cache: ControlStatisticsCache,
        size_factors: np.ndarray,
        n_genes: int,
        min_cells_expressed: int,
        min_pct_ctrl: float,
        min_pct_pert: float,
        min_mean_ctrl: float,
        min_mean_pert: float,
        min_cells_ctrl: int,
        min_cells_pert: int,
        min_total_count: float,
        max_iter: int,
        tol: float,
        min_mu: float,
        dispersion_method: str,
        shrink_dispersion: bool,
        use_map_dispersion: bool,
        lfc_shrinkage_type: str,
        se_method: str = "sandwich",
        perturbation_boundaries: dict | None = None,
    ) -> dict:
        """Fit NB-GLM for a perturbation group using cached control statistics.
        
        This is an optimized version that reuses precomputed control cell
        statistics (intercept, weights, XᵀWX contributions) to avoid redundant
        computation across perturbation comparisons.
        """
        group_mask = labels == label
        group_n = int(group_mask.sum())
        n_control = control_cache.control_n
        subset_n = n_control + group_n
        
        # Initialize result arrays
        result = {
            "group_idx": group_idx,
            "effect": np.full(n_genes, np.nan, dtype=np.float64),
            "statistic": np.full(n_genes, np.nan, dtype=np.float64),
            "pvalue": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc": np.full(n_genes, np.nan, dtype=np.float64),
            "logfc_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "intercept": np.full(n_genes, np.nan, dtype=np.float64),  # MLE intercept for shrink_lfc
            "se": np.full(n_genes, np.nan, dtype=np.float64),
            "pts": np.zeros(n_genes, dtype=np.float32),
            "pts_rest": control_cache.pts_rest.copy(),
            "dispersion": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_raw": np.full(n_genes, np.nan, dtype=np.float64),
            "dispersion_trend": np.full(n_genes, np.nan, dtype=np.float64),
            "mean": np.zeros(n_genes, dtype=np.float64),
            "iterations": np.zeros(n_genes, dtype=np.int32),
            "converged": np.zeros(n_genes, dtype=bool),
        }
        
        # Load perturbation group cells
        # Use slice-based access if file is sorted, otherwise use mask
        backed = read_backed(path)
        try:
            if perturbation_boundaries is not None and label in perturbation_boundaries:
                # Slice-based access for sorted files (contiguous, fast)
                start, end = perturbation_boundaries[label]
                group_matrix = backed.X[start:end, :]
            else:
                # Mask-based access for unsorted files (random, slower)
                group_matrix = backed.X[group_mask, :]
            if sp.issparse(group_matrix):
                group_matrix = sp.csr_matrix(group_matrix, dtype=np.float64)
            else:
                group_matrix = np.asarray(group_matrix, dtype=np.float64)
        finally:
            backed.file.close()
        
        # Compute expression counts for perturbation cells
        if sp.issparse(group_matrix):
            group_expr_counts = np.asarray(group_matrix.getnnz(axis=0)).ravel()
        else:
            group_expr_counts = np.sum(group_matrix > 0, axis=0)
        
        # Valid genes mask
        total_expr_counts = control_cache.control_expr_counts + group_expr_counts
        valid_mask = total_expr_counts >= min_cells_expressed
        
        # Compute pts
        pts = np.divide(
            group_expr_counts,
            group_n,
            out=np.zeros(n_genes, dtype=np.float32),
            where=group_n > 0,
        )
        # Reported for every gene; see the matching comment in the uncached
        # path.
        result["pts"] = pts.astype(np.float32)
        
        # Compute mean expression
        subset_size_factors_group = np.asarray(size_factors)[group_mask]
        if sp.issparse(group_matrix):
            normalized_group = group_matrix.multiply(1.0 / subset_size_factors_group[:, None])
            mean_group = np.asarray(normalized_group.sum(axis=0)).ravel()
        else:
            normalized_group = group_matrix / subset_size_factors_group[:, None]
            mean_group = normalized_group.sum(axis=0)
        
        # Combined mean
        result["mean"] = (control_cache.control_mean_expr * n_control + mean_group) / subset_n

        # Per-condition low-expression filter (jointly low in BOTH groups).
        # ``mean_group`` is the size-factor-normalised SUM; divide by group_n
        # for the per-cell mean used by the filter.
        group_mean_per_cell = (
            mean_group / group_n if group_n > 0 else np.zeros_like(mean_group)
        )
        low_both = _low_expr_in_both_mask(
            pert_expr_counts=group_expr_counts,
            control_expr_counts=control_cache.control_expr_counts,
            pert_mean=group_mean_per_cell,
            control_mean=control_cache.control_mean_expr,
            n_pert_cells=group_n,
            n_control_cells=n_control,
            min_pct_ctrl=min_pct_ctrl,
            min_pct_pert=min_pct_pert,
            min_mean_ctrl=min_mean_ctrl,
            min_mean_pert=min_mean_pert,
        )
        valid_mask = valid_mask & ~low_both
        # See the matching comment in the uncached path.
        valid_mask = valid_mask & ~_nonestimable_glm_mask(
            pert_expr_counts=group_expr_counts,
            control_expr_counts=control_cache.control_expr_counts,
            min_cells_ctrl=min_cells_ctrl,
            min_cells_pert=min_cells_pert,
        )
        result["n_tested"] = int(valid_mask.sum())

        if not np.any(valid_mask):
            return result
        
        # Get perturbation offset
        perturbation_offset = np.log(np.clip(subset_size_factors_group, 1e-8, None))
        
        # Create perturbation indicator for the combined design (control first, then perturbation)
        perturbation_indicator = np.concatenate([
            np.zeros(n_control, dtype=np.float64),
            np.ones(group_n, dtype=np.float64)
        ])
        
        # Check if using frozen control mode (memory-efficient parallel fitting)
        if control_cache.use_frozen_control:
            # FROZEN CONTROL PATH: Use sufficient statistics instead of raw matrix
            # Per-worker memory: ~5GB → ~1MB (enables 32 workers instead of 2)
            
            # Create batch fitter with minimal design (only perturbation cells have data)
            batch_fitter = NBGLMBatchFitter(
                design=np.ones((group_n, 1)),  # Placeholder, not used in frozen mode
                offset=perturbation_offset,  # Only perturbation offsets needed
                max_iter=max_iter,
                tol=tol,
                dispersion_method=dispersion_method,
                min_mu=min_mu,
                min_total_count=min_total_count,
            )
            
            batch_result = batch_fitter.fit_batch_with_frozen_control(
                perturbation_matrix=group_matrix,
                perturbation_offset=perturbation_offset,
                control_cache=control_cache,
                valid_mask=valid_mask,
            )
        else:
            # STANDARD PATH: Full control_matrix available
            # Fit using the batch fitter with control cache
            batch_fitter = NBGLMBatchFitter(
                design=np.column_stack([np.ones(subset_n), perturbation_indicator]),
                offset=np.concatenate([control_cache.control_offset, perturbation_offset]),
                max_iter=max_iter,
                tol=tol,
                dispersion_method=dispersion_method,
                min_mu=min_mu,
                min_total_count=min_total_count,
            )
            
            batch_result = batch_fitter.fit_batch_with_control_cache(
                perturbation_matrix=group_matrix,
                perturbation_offset=perturbation_offset,
                control_cache=control_cache,
                perturbation_indicator=perturbation_indicator,
                valid_mask=valid_mask,
            )

        
        valid_indices = np.where(valid_mask)[0]
        
        # Extract results
        result["converged"][valid_indices] = batch_result.converged[valid_indices]
        result["iterations"][valid_indices] = batch_result.n_iter[valid_indices]
        
        # Perturbation coefficient is at index 1 - vectorized computation
        coefs = batch_result.coef[:, 1]  # (n_genes,)
        ses = batch_result.se[:, 1]  # (n_genes,)
        
        # Vectorized: compute statistics for all valid & converged genes at once
        # Build mask for valid, converged genes with finite coef/se
        valid_converged_mask = (
            valid_mask & 
            batch_result.converged &
            np.isfinite(coefs) & 
            np.isfinite(ses) & 
            (ses > 0)
        )
        
        # Compute Wald statistic and p-value vectorized
        statistics = np.divide(
            coefs, ses, 
            out=np.full(n_genes, np.nan), 
            where=valid_converged_mask
        )
        # Use norm.sf (survival function) to avoid underflow for large |z|
        # Note: ndtr(x) = CDF, so 1-ndtr(x) underflows for large x; sf(x) = 1-CDF is stable
        pvalues = np.where(
            valid_converged_mask,
            2.0 * norm.sf(np.abs(statistics)),  # 2-sided p-value
            np.nan
        )
        
        result["statistic"] = statistics
        result["pvalue"] = pvalues
        result["logfc"] = np.where(valid_converged_mask, coefs, np.nan)
        result["se"] = np.where(valid_converged_mask, ses, np.nan)
        result["intercept"] = np.where(valid_converged_mask, batch_result.coef[:, 0], np.nan)  # Store fitted intercept
        
        # Handle dispersion shrinkage
        if shrink_dispersion:
            # ================================================================
            # MEMORY-OPTIMIZED: Use batched computation for dispersion and SE
            # ================================================================
            # Instead of building full (n_control+n_group, n_genes) matrices,
            # process genes in batches to reduce peak memory.
            # ================================================================
            
            # Get coefficients (needed for SE recomputation)
            beta0_all = batch_result.coef[:, 0]  # (n_genes,)
            beta1_all = batch_result.coef[:, 1]  # (n_genes,)
            
            # ================================================================
            # OPTIMIZATION: Check for global dispersion FIRST (before MoM/trend)
            # When dispersion_scope='global', skip expensive per-comparison work
            # ================================================================
            if control_cache.global_dispersion is not None:
                # FAST PATH: Use precomputed global dispersion
                # Skip MoM dispersion computation entirely
                # Skip trend fitting entirely
                logger.debug(f"Using precomputed global dispersion for {label}")
                result["dispersion"] = control_cache.global_dispersion.copy()
                if control_cache.global_dispersion_trend is not None:
                    result["dispersion_trend"] = control_cache.global_dispersion_trend.copy()
                else:
                    # Fallback: use global dispersion as trend proxy
                    result["dispersion_trend"] = control_cache.global_dispersion.copy()
                # dispersion_raw not computed in global mode - set to NaN or copy global
                result["dispersion_raw"] = control_cache.global_dispersion.copy()
                
                # SE handling: frozen control vs standard
                if control_cache.use_frozen_control:
                    # FROZEN CONTROL PATH: Use SE from fit_batch_with_frozen_control directly
                    # No SE recomputation needed since dispersion is global (pre-fitted)
                    # The SE already uses the correct dispersion from control_cache
                    pass  # SE already set from batch_result
                else:
                    # STANDARD PATH: Recompute SE with global dispersion
                    n_control = control_cache.control_matrix.shape[0]
                    n_group = group_matrix.shape[0]
                    if sp.issparse(group_matrix):
                        Y_pert_dense = group_matrix.toarray()
                    else:
                        Y_pert_dense = np.asarray(group_matrix, dtype=np.float64)
                    
                    final_disp = result["dispersion"]
                    recomputed_se = _compute_se_batched(
                        Y_control=control_cache.control_matrix,
                        Y_pert=Y_pert_dense,
                        control_offset=control_cache.control_offset,
                        pert_offset=perturbation_offset,
                        beta0=beta0_all,
                        beta1=beta1_all,
                        dispersion=final_disp,
                        gene_batch_size=5000,
                        se_method=se_method,
                    )
                    
                    # Update result with recomputed SE
                    result["se"] = np.where(valid_converged_mask, recomputed_se, np.nan)
                    
                    # Recompute Wald statistic and p-value with new SE
                    coefs = batch_result.coef[:, 1]
                    statistics = np.divide(
                        coefs, recomputed_se, 
                        out=np.full(n_genes, np.nan), 
                        where=valid_converged_mask
                    )
                    pvalues = np.where(
                        valid_converged_mask,
                        2.0 * norm.sf(np.abs(statistics)),
                        np.nan
                    )
                    result["statistic"] = statistics
                    result["pvalue"] = pvalues
            else:
                # ============================================================
                # STANDARD PATH: Per-comparison dispersion (MoM → trend → MAP)
                # Requires control_matrix - not compatible with frozen control
                # ============================================================
                if control_cache.use_frozen_control:
                    raise ValueError(
                        "Frozen control mode requires global dispersion. "
                        "Set dispersion_scope='global' and shrink_dispersion=True when using freeze_control=True."
                    )
                
                # Prepare perturbation matrix as dense (needed for SE recomputation)
                n_control = control_cache.control_matrix.shape[0]
                n_group = group_matrix.shape[0]
                if sp.issparse(group_matrix):
                    Y_pert_dense = group_matrix.toarray()
                else:
                    Y_pert_dense = np.asarray(group_matrix, dtype=np.float64)
                
                # Compute MoM dispersion using batched processing
                mom_disp = _compute_mom_dispersion_batched(
                    Y_control=control_cache.control_matrix,
                    Y_pert=Y_pert_dense,
                    control_offset=control_cache.control_offset,
                    pert_offset=perturbation_offset,
                    beta0=beta0_all,
                    beta1=beta1_all,
                    converged=batch_result.converged,
                    gene_batch_size=5000,
                )
                result["dispersion_raw"][valid_indices] = mom_disp[valid_indices]
                
                # Fit trend using corrected MoM dispersion
                trend = fit_dispersion_trend(result["mean"], result["dispersion_raw"])
                result["dispersion_trend"] = trend
                
                if use_map_dispersion:
                    # For MAP dispersion, we still need full matrices (can optimize later)
                    # Build combined Y and mu for estimate_dispersion_map
                    Y = np.empty((n_control + n_group, n_genes), dtype=np.float64)
                    Y[:n_control, :] = control_cache.control_matrix
                    Y[n_control:, :] = Y_pert_dense
                    
                    # Compute full mu matrix for MAP
                    offset_combined = np.concatenate([control_cache.control_offset, perturbation_offset])
                    eta = (
                        beta0_all[None, :] + 
                        perturbation_indicator[:, None] * beta1_all[None, :] + 
                        offset_combined[:, None]
                    )
                    np.clip(eta, -30, 30, out=eta)
                    mu = np.exp(eta)
                    del eta
                    mu[:, ~batch_result.converged] = 1e-10
                    np.maximum(mu, 1e-10, out=mu)
                    
                    max_disp = max(10.0, float(subset_n))
                    result["dispersion"] = estimate_dispersion_map(Y, mu, trend, max_disp=max_disp)
                    del Y, mu  # Free large matrices
                else:
                    result["dispersion"] = shrink_dispersions(result["dispersion_raw"], trend)
                
                # ================================================================
                # Recompute SE using batched processing (memory-optimized)
                # ================================================================
                final_disp = result["dispersion"]
                recomputed_se = _compute_se_batched(
                    Y_control=control_cache.control_matrix,
                    Y_pert=Y_pert_dense,
                    control_offset=control_cache.control_offset,
                    pert_offset=perturbation_offset,
                    beta0=beta0_all,
                    beta1=beta1_all,
                    dispersion=final_disp,
                    gene_batch_size=5000,
                    se_method=se_method,
                )
                
                # Update result with recomputed SE
                result["se"] = np.where(valid_converged_mask, recomputed_se, np.nan)
                
                # Recompute Wald statistic and p-value with new SE
                coefs = batch_result.coef[:, 1]
                statistics = np.divide(
                    coefs, recomputed_se, 
                    out=np.full(n_genes, np.nan), 
                    where=valid_converged_mask
                )
                pvalues = np.where(
                    valid_converged_mask,
                    2.0 * norm.sf(np.abs(statistics)),  # Use sf() for numerical stability
                    np.nan
                )
                result["statistic"] = statistics
                result["pvalue"] = pvalues
        else:
            result["dispersion_trend"] = result["dispersion_raw"].copy()
            result["dispersion"] = result["dispersion_raw"].copy()
        
        result["logfc_raw"] = result["logfc"].copy()
        if lfc_shrinkage_type == "apeglm":
            # Full NB-GLM re-fitting with Cauchy prior (matches PyDESeq2)
            # Build mle_coef matrix (n_params, n_genes) from batch_result
            n_params = 2  # Intercept + perturbation
            mle_coef = np.full((n_params, n_genes), np.nan, dtype=np.float64)
            # Only the genes that were actually fitted have an MLE to shrink.
            # ``batch_result.coef`` is zero-initialised, so handing it over
            # whole would give apeGLM a finite starting point for a pair the
            # filters excluded and return a finite effect for it; the NaN left
            # here is what the rest of the result already reports for those.
            mle_coef[0, valid_converged_mask] = batch_result.coef[valid_converged_mask, 0]
            mle_coef[1, valid_converged_mask] = batch_result.coef[valid_converged_mask, 1]
            
            # Build combined count matrix (control + perturbation)
            if sp.issparse(group_matrix):
                Y_pert = group_matrix.toarray()
            else:
                Y_pert = np.asarray(group_matrix, dtype=np.float64)
            counts_combined = np.vstack([control_cache.control_matrix, Y_pert])
            
            # Build combined size factors
            sf_combined = np.concatenate([
                np.exp(control_cache.control_offset),
                subset_size_factors_group
            ])
            
            # Build design matrix for combined data
            design_combined = np.zeros((n_control + group_n, 2), dtype=np.float64)
            design_combined[:, 0] = 1.0  # Intercept
            design_combined[n_control:, 1] = 1.0  # Perturbation indicator
            
            # Call full apeGLM shrinkage
            # NOTE: PyDESeq2's lfc_shrink does NOT use min_mu during shrinkage
            shrunk_coef, shrunk_se_arr, shrink_converged = shrink_lfc_apeglm(
                counts=counts_combined,
                design_matrix=design_combined,
                size_factors=sf_combined,
                dispersion=result["dispersion"],
                mle_coef=mle_coef,
                mle_se=result["se"],
                shrink_index=1,  # LFC is at index 1
                prior_scale=None,
                n_jobs=1,
                min_mu=0.0,  # No min_mu - match PyDESeq2's lfc_shrink behavior
            )
            result["logfc"] = shrunk_coef[1, :]  # Shrunken LFC
            result["effect"] = result["logfc"].copy()
            result["se"] = shrunk_se_arr
        else:
            result["effect"] = result["logfc"].copy()
        
        return result
    # -------------------------------------------------------------------------

    n_groups = len(candidates)
    
    # Determine output path first (needed for checkpoint)
    output_path = resolve_output_path(
        path,
        suffix="nb_glm",
        output_dir=output_dir,
        data_name=data_name,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")
    
    # Determine checkpoint interval
    eff_checkpoint_interval = _get_checkpoint_interval(len(candidates), checkpoint_interval)
    
    # Create index mappings
    candidate_to_idx = {label: idx for idx, label in enumerate(candidates)}

    shape = (n_groups, n_genes)
    _nb_glm_disk_estimate = warn_if_disk_space_low(
        _nb_glm_disk_bytes(*shape),
        output_path,
        context="nb_glm_test",
    )
    _messages.print_disk_estimate(verbose, "nb_glm_test", _nb_glm_disk_estimate)
    # Per-perturbation results live beside the output until the run finishes,
    # so an interrupted run resumes from its last checkpoint.
    run = ResumableRun(
        output_path,
        checkpoint_path,
        fingerprint=_de_fingerprint(path, call_args, method="nb_glm", candidates=candidates, n_genes=n_genes),
        arrays={
            "statistic": (shape, np.float64, np.nan),
            "pvalue": (shape, np.float64, np.nan),
            "pvalue_adj": (shape, np.float64, np.nan),
            "logfoldchange": (shape, np.float64, np.nan),
            "logfoldchange_raw": (shape, np.float64, np.nan),
            "intercept": (shape, np.float64, np.nan),  # MLE intercept for shrink_lfc
            "standard_error": (shape, np.float64, np.nan),
            "dispersion": (shape, np.float64, np.nan),
            "dispersion_raw": (shape, np.float64, np.nan),
            "dispersion_trend": (shape, np.float64, np.nan),
            "pts": (shape, np.float32, 0),
            "iterations": (shape, np.int32, 0),
            "converged": (shape, np.bool_, False),
        },
        resume=resume,
    )
    # Earlier failures are retried, so only completions carry over.
    completed_labels: list[str] = run.progress.get("completed", [])
    candidates_to_run = [c for c in candidates if c not in set(completed_labels)]
    if run.resumed:
        _messages.vprint(verbose, "nb_glm_test", f"Resuming: {len(completed_labels)}/{n_groups} perturbations already done")
    statistic_memmap = run.arrays["statistic"]
    pvalue_memmap = run.arrays["pvalue"]
    logfc_memmap = run.arrays["logfoldchange"]
    logfc_raw_memmap = run.arrays["logfoldchange_raw"]
    intercept_memmap = run.arrays["intercept"]
    se_memmap = run.arrays["standard_error"]
    pts_memmap = run.arrays["pts"]
    dispersion_memmap = run.arrays["dispersion"]
    dispersion_raw_memmap = run.arrays["dispersion_raw"]
    dispersion_trend_memmap = run.arrays["dispersion_trend"]
    iter_memmap = run.arrays["iterations"]
    convergence_memmap = run.arrays["converged"]

    # Load control cells matrix once (all genes)
    # For very large control groups, skip loading and use streaming path
    control_matrix_gb = control_n * n_genes * 8 / 1e9  # dense float64
    if memory_limit_gb is not None:
        _ctrl_mem_limit = min(memory_limit_gb, _get_available_memory_mb() / 1000)
    else:
        _ctrl_mem_limit = _get_available_memory_mb() / 1000
    # Streaming threshold: control dense + 3 work arrays > 30% available memory
    _ctrl_streaming_threshold = max_dense_fraction * _ctrl_mem_limit
    use_streaming_control = (control_matrix_gb * 4) > _ctrl_streaming_threshold

    if use_streaming_control:
        logger.info(
            f"Large control group: {control_n:,} cells × {n_genes:,} genes = "
            f"{control_matrix_gb:.1f} GB dense. Using streaming control statistics."
        )
        control_matrix = None  # Not loaded — streaming will read from disk
        # Compute control_expr_counts via streaming
        control_expr_counts = np.zeros(n_genes, dtype=np.int64)
        backed = read_backed(path)
        try:
            ctrl_indices = np.where(control_mask)[0]
            _chunk = 4096
            for _start in range(0, control_n, _chunk):
                _end = min(_start + _chunk, control_n)
                _idx = ctrl_indices[_start:_end]
                _blk = backed.X[_idx, :]
                if sp.issparse(_blk):
                    control_expr_counts += np.asarray(_blk.getnnz(axis=0)).ravel()
                else:
                    control_expr_counts += np.asarray((_blk > 0).sum(axis=0)).ravel()
        finally:
            backed.file.close()
        drop_file_cache(path)  # Free page cache from streaming reads
    else:
        backed = read_backed(path)
        try:
            control_matrix = backed.X[control_mask, :]
            if sp.issparse(control_matrix):
                control_matrix = sp.csr_matrix(control_matrix, dtype=np.float64)
            else:
                control_matrix = np.asarray(control_matrix, dtype=np.float64)
        finally:
            backed.file.close()
        drop_file_cache(path)  # Free page cache from control matrix read
        # Pre-compute control expression counts
        if sp.issparse(control_matrix):
            control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
        else:
            control_expr_counts = np.sum(control_matrix > 0, axis=0)
        
    # Compute pts_rest once (same for all perturbations)
    pts_rest_shared = np.divide(
        control_expr_counts,
        control_n,
        out=np.zeros(n_genes, dtype=np.float32),
        where=control_n > 0,
    )

    # =====================================================================
    # Parallel fitting of perturbation groups
    # =====================================================================
    # Determine number of parallel workers with memory-awareness
    # For small n_groups, run sequentially to avoid joblib overhead
    # (profiling shows joblib.sleep takes 24s for 2 perturbations)
    effective_n_jobs = _resolve_n_jobs(n_jobs)
        
    # Save original requested workers for auto-detection logic
    # (before memory-based reduction)
    requested_n_jobs = effective_n_jobs
        
    # Memory-aware worker limiting: Adaptive estimation based on dataset statistics
    # Compute group size statistics without loading the full matrix
    # Use numpy unique with counts since labels is a numpy array
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    group_sizes = dict(zip(unique_labels, label_counts))
    pert_group_sizes = [group_sizes.get(g, 0) for g in candidates]
    max_group_size = max(pert_group_sizes) if pert_group_sizes else 1
    avg_group_size = max(1, (n_cells_total - control_n) // max(1, n_groups))
        
    # Use p95 group size for realistic estimate (avoids over-conservative from outliers)
    if len(pert_group_sizes) >= 10:
        p95_group_size = int(np.percentile(pert_group_sizes, 95))
        use_group_size = p95_group_size
    else:
        use_group_size = max_group_size
        
    # Decide early if we can use control cache (needed for memory estimation)
    can_use_cache_early = (
        use_control_cache and 
        len(covariates) == 0 and
        size_factor_scope == "global"
    )
        
    # =====================================================================
    # AUTO-DETECTION: Enable freeze_control for large datasets
    # =====================================================================
    # When freeze_control=None (default), auto-enable if:
    # 1. Control matrix serialization would severely limit workers (<4)
    # 2. Required settings are met (dispersion_scope='global', shrink_dispersion=True)
    #
    # This provides optimal parallelization without user intervention.
        
    if freeze_control is None:
        # Check if settings are compatible with frozen control
        settings_compatible = (
            can_use_cache_early and
            dispersion_scope == "global" and
            shrink_dispersion
        )
            
        if settings_compatible:
            # Estimate per-worker memory in standard mode (matching actual formula below)
            control_matrix_mb_est = control_n * n_genes * 8 / 1e6
            control_matrix_gb_est = control_matrix_mb_est / 1000
            labels_mb_est = n_cells_total * 50 / 1e6  # ~50 bytes per string
            size_factors_mb_est = n_cells_total * 8 / 1e6
            work_arrays_mb_est = (control_n + use_group_size) * n_genes * 8 * 4 / 1e6
            serialized_args_mb_est = (control_matrix_mb_est + labels_mb_est + size_factors_mb_est) * 2.5
            per_worker_standard_mb = serialized_args_mb_est + work_arrays_mb_est + 2000
                
            # Get available memory
            if memory_limit_gb is not None:
                available_mb = memory_limit_gb * 1000
            else:
                try:
                    available_mb = _detected_available_bytes() / 1e6
                except ImportError:
                    available_mb = 8000.0
                
            # ================================================================
            # AUTO-ENABLE CONDITION 1: Large control matrix (>10 GB)
            # For datasets like Feng (110K control × 36K genes = 32 GB),
            # freeze_control significantly reduces per-worker memory.
            # ================================================================
            if control_matrix_gb_est > 10.0:
                freeze_control = True
                logger.info(
                    f"Auto-enabling freeze_control: large control matrix "
                    f"({control_n:,} cells × {n_genes:,} genes = {control_matrix_gb_est:.1f} GB > 10 GB threshold)"
                )
            else:
                # ================================================================
                # AUTO-ENABLE CONDITION 2: Worker count limitation (<4 workers)
                # Original logic: enable if parallelization would be severely limited
                # ================================================================
                base_memory_mb_est = control_matrix_mb_est + 1000
                usable_mb = available_mb * 0.8
                remaining_mb = max(usable_mb - base_memory_mb_est, per_worker_standard_mb)
                max_workers_standard = max(1, int(remaining_mb / per_worker_standard_mb))
                    
                if max_workers_standard < 4 and requested_n_jobs >= 4:
                    freeze_control = True
                    logger.info(
                        f"Auto-enabling freeze_control: standard mode would limit to {max_workers_standard} workers "
                        f"(control: {control_n:,} cells × {n_genes:,} genes = {control_matrix_mb_est:.0f} MB, "
                        f"per_worker: {per_worker_standard_mb:.0f} MB). "
                        f"Frozen control enables ~{requested_n_jobs} workers."
                    )
                else:
                    freeze_control = False
        else:
            freeze_control = False
        
    # Check if frozen control mode is valid (after auto-detection)
    can_use_frozen_control = (
        freeze_control and 
        can_use_cache_early and 
        dispersion_scope == "global" and 
        shrink_dispersion
    )
        
    # When frozen control is enabled but streaming was not, retroactively
    # switch to streaming.  The dense IRLS peak is 4× control_matrix_gb;
    # streaming avoids that by processing chunks.  The loaded control
    # matrix is freed because frozen stats replace it.
    if can_use_frozen_control and not use_streaming_control and control_matrix is not None:
        use_streaming_control = True
        del control_matrix
        control_matrix = None
        gc.collect()
        logger.info(
            "Switching to streaming control statistics for frozen control mode "
            f"(avoids {control_matrix_gb * 4:.1f} GB dense IRLS peak)."
        )
        
    # Memory estimation for joblib parallel execution
    # IMPORTANT: joblib's loky backend serializes (pickles) all function arguments
    # for each worker process. This means control_matrix is copied to each worker,
    # not shared via copy-on-write as it would be with fork().
    #
    # What each worker receives via pickle:
    # 1. control_cache.control_matrix: (control_n × n_genes) × 8 bytes
    #    OR with freeze_control=True: frozen stats only (~1MB total)
    # 2. labels array: n_cells_total strings (pickled as object array)
    # 3. size_factors: n_cells_total × 8 bytes
    # 4. Other small arrays (control_offset, etc.)
    #
    # What each worker allocates during execution:
    # 1. group_matrix (loaded from disk, small: ~200 cells × n_genes)
    # 2. Intermediate arrays for SE recomputation, mu, etc.
    # 3. When dispersion_scope='per_comparison': full Y and mu matrices
    #
    # Pickle overhead is typically 1.5-2× the raw array size due to protocol
    # serialization and Python object overhead.
        
    if can_use_frozen_control:
        # FROZEN CONTROL MODE: Optimized memory estimation
        # 
        # What each worker receives:
        # 1. control_cache with frozen stats (~1MB): W_sum, Wz_sum, etc.
        # 2. perturbation_boundaries dict: ~100KB for 10K perturbations
        # 3. size_factors: still full array (~16MB for 2M cells) - could optimize later
        # 4. labels: still needed for fallback, but could use boundaries only
        #
        # What each worker allocates:
        # 1. group_matrix from disk: (group_size × n_genes) × 8 bytes
        # 2. Work arrays: mu, W, z for perturbation cells only
        # 3. Result arrays: ~n_genes × 8 bytes × 10 arrays
            
        # Frozen stats: 6 arrays of shape (n_genes,)
        frozen_stats_mb = n_genes * 8 * 6 / 1e6  # W_sum, Wz_sum, mu_sum, etc.
            
        # Cache metadata (beta_intercept, dispersion, pts_rest, etc.)
        cache_metadata_mb = n_genes * 8 * 5 / 1e6
            
        # Perturbation boundaries: ~50 bytes per perturbation
        boundaries_mb = n_groups * 50 / 1e6
            
        # Labels and size_factors (still passed but could be optimized)
        labels_mb = n_cells_total * 50 / 1e6
        size_factors_mb = n_cells_total * 8 / 1e6
            
        # Serialized args with reduced pickle overhead (simpler objects)
        control_matrix_mb_for_pickle = frozen_stats_mb + cache_metadata_mb + boundaries_mb
        serialized_args_mb = (control_matrix_mb_for_pickle + labels_mb + size_factors_mb) * 2.0
            
        # Work arrays: only perturbation cells (much smaller!)
        # mu_pert, W_pert, z_pert, Y_pert_valid for fitting
        work_arrays_mb = use_group_size * n_genes * 8 * 5 / 1e6
            
        # Result arrays per worker
        result_arrays_mb = n_genes * 8 * 12 / 1e6  # 12 result fields
            
        # Reduced Python overhead for frozen control (simpler computation)
        python_overhead_mb = 500  # 500 MB instead of 2 GB
            
        per_worker_mb = serialized_args_mb + work_arrays_mb + result_arrays_mb + python_overhead_mb
            
        logger.debug(
            f"Frozen control memory estimate: serialized={serialized_args_mb:.1f}MB, "
            f"work_arrays={work_arrays_mb:.1f}MB, per_worker={per_worker_mb:.1f}MB"
        )
    else:
        control_matrix_mb_for_pickle = control_n * n_genes * 8 / 1e6  # float64
            
        # Context-aware work arrays based on dispersion mode
        if can_use_cache_early and dispersion_scope == "global":
            # Global dispersion: skip MoM/trend, but still need SE recomputation arrays
            work_arrays_mb = (control_n + use_group_size) * n_genes * 8 * 4 / 1e6
        else:
            # Per-comparison: need full Y and mu matrices for MAP dispersion
            work_arrays_mb = (control_n + use_group_size) * n_genes * 8 * 6 / 1e6
            
        # Standard mode: full labels and size_factors arrays
        labels_mb = n_cells_total * 50 / 1e6  # ~50 bytes per string (pickled)
        size_factors_mb = n_cells_total * 8 / 1e6
            
        # Pickle overhead is 2-3× for complex objects due to Python object structure
        serialized_args_mb = (control_matrix_mb_for_pickle + labels_mb + size_factors_mb) * 2.5
            
        # Per-worker total: serialized args + work arrays + Python/process overhead
        per_worker_mb = serialized_args_mb + work_arrays_mb + 2000
        
    # For base memory and logging
    control_matrix_mb = control_n * n_genes * 8 / 1e6
        
    # Base memory: parent process + one copy of control matrix + misc arrays
    base_memory_mb = control_matrix_mb + 1000  # Parent process overhead
        
    # Calculate available memory
    if memory_limit_gb is not None:
        available_mb = memory_limit_gb * 1000
    else:
        try:
            available_mb = _detected_available_bytes() / 1e6
        except ImportError:
            available_mb = 8000.0  # 8 GB default
        
    # Reserve 20% headroom for safety
    usable_mb = available_mb * 0.8
    remaining_mb = max(usable_mb - base_memory_mb, per_worker_mb)
    max_workers_by_memory = max(1, int(remaining_mb / per_worker_mb))
        
    # Dataset-size-aware caps
    full_matrix_gb = (control_n + avg_group_size * n_groups) * n_genes * 8 / 1e9
    if full_matrix_gb < 1.0:
        # Tiny dataset: cap workers to reduce parallelization overhead
        max_workers_by_size = max(4, n_groups // 2)
    else:
        max_workers_by_size = n_groups
        
    # Apply all constraints
    # Use candidates_to_run (not n_groups) for worker limit since that's what we're actually running
    n_to_run = len(candidates_to_run)
    memory_limited_workers = min(max_workers_by_memory, max_workers_by_size, n_to_run)
        
    if memory_limited_workers < effective_n_jobs:
        # Apply memory limiting
        # Determine limiting factor for logging
        if memory_limited_workers == n_to_run and n_to_run < max_workers_by_memory and n_to_run < max_workers_by_size:
            limit_reason = "perturbation_count"
        elif memory_limited_workers == max_workers_by_memory:
            limit_reason = "memory"
        elif memory_limited_workers == max_workers_by_size:
            limit_reason = "small_dataset"
        else:
            limit_reason = "perturbation_count"
        logger.info(
            f"Memory-aware limiting: {effective_n_jobs} -> {memory_limited_workers} workers "
            f"(reason: {limit_reason}, base: {base_memory_mb:.0f}MB, per_worker: {per_worker_mb:.0f}MB, "
            f"available: {available_mb:.0f}MB, full_matrix: {full_matrix_gb:.1f}GB)"
        )
        effective_n_jobs = memory_limited_workers
    effective_n_jobs = max(1, effective_n_jobs)
        
    # For small number of perturbations to run, run sequentially to avoid overhead
    use_parallel = n_to_run >= 4 and effective_n_jobs > 1
        
    # Decide whether to use control cache optimization
    # Control cache is used when: no covariates, use_control_cache=True, global SF
    # Per-comparison size factors require fresh computation per comparison
    can_use_cache = (
        use_control_cache and 
        len(covariates) == 0 and
        size_factor_scope == "global"
    )
        
    # =====================================================================
    # Early memory check: determine if we need streaming mode
    # =====================================================================
    # For very large datasets (e.g., Replogle-GW-k562: 2M cells × 8K genes),
    # loading the full matrix would exceed memory. Check this ONCE before
    # any full matrix loads (full_X for per-comparison SF, all_cell_matrix
    # for global dispersion).
    estimated_matrix_gb = n_cells_total * n_genes * 8 / 1e9  # float64
    if memory_limit_gb is not None:
        effective_memory_limit_gb = min(available_mb / 1000, memory_limit_gb)
    else:
        effective_memory_limit_gb = available_mb / 1000
    memory_budget_gb = max_dense_fraction * effective_memory_limit_gb
    use_streaming_mode = estimated_matrix_gb > memory_budget_gb
        
    if use_streaming_mode:
        logger.info(
            f"Large dataset detected: {n_cells_total:,} cells × {n_genes:,} genes = "
            f"{estimated_matrix_gb:.1f} GB > {memory_budget_gb:.1f} GB budget. "
            f"Using streaming mode for memory efficiency."
        )
        
    # For per-comparison size factors, we need the full count matrix
    # Skip if streaming mode - worker will fall back to global SF
    if size_factor_scope == "per_comparison":
        if use_streaming_mode:
            logger.warning(
                f"Dataset too large ({estimated_matrix_gb:.1f} GB) for per-comparison "
                f"size factors (requires loading full matrix). "
                f"Falling back to global size factors for memory efficiency."
            )
            full_X = None  # Worker will use global SF
        else:
            logger.info("Using per-comparison size factors for PyDESeq2 compatibility...")
            backed = read_backed(path)
            try:
                full_X = backed.X[:]
                if sp.issparse(full_X):
                    full_X = sp.csr_matrix(full_X, dtype=np.float64)
                else:
                    full_X = np.asarray(full_X, dtype=np.float64)
            finally:
                backed.file.close()
    else:
        full_X = None
        
    # Precompute control statistics once if using cache
    control_cache = None
    if can_use_cache:
        # Validate freeze_control requirements (for explicit freeze_control=True)
        if freeze_control:
            if dispersion_scope != "global":
                raise ValueError(
                    "freeze_control=True requires dispersion_scope='global'. "
                    "Per-comparison dispersion needs raw control matrix."
                )
            if not shrink_dispersion:
                raise ValueError(
                    "freeze_control=True requires shrink_dispersion=True. "
                    "Global dispersion must be computed for frozen control mode."
                )
            # Note: Auto-detected freeze_control already logged in auto-detection block
            
        logger.info("Precomputing control cell statistics for cache optimization...")
        control_offset_arr = offset[control_mask]
        if use_streaming_control:
            # Streaming path: read control cells from disk in chunks
            # Forces freeze_control=True (raw matrix never materialised)
            if not freeze_control:
                logger.info(
                    "Forcing freeze_control=True for streaming control statistics "
                    "(raw control matrix too large to fit in memory)."
                )
                freeze_control = True
            control_cache = precompute_control_statistics_streaming(
                path=path,
                control_mask=control_mask,
                control_offset=control_offset_arr,
                max_iter=max_iter,
                tol=tol,
                min_mu=min_mu,
                global_size_factors=size_factors,
                freeze_control=True,
            )
            drop_file_cache(path)  # Free page cache from streaming IRLS
        else:
            control_cache = precompute_control_statistics(
                control_matrix=control_matrix,
                control_offset=control_offset_arr,
                max_iter=max_iter,
                tol=tol,
                min_mu=min_mu,
                dispersion_method="moments",  # Fast initial estimate
                global_size_factors=size_factors,  # Store global SF in cache
                freeze_control=freeze_control,  # Enable frozen control mode
            )
            # If frozen control, the cache holds sufficient statistics —
            # the raw control_matrix is no longer needed.
            if freeze_control and control_matrix is not None:
                del control_matrix
                control_matrix = None
                gc.collect()
            
        # Precompute global dispersion if dispersion_scope='global'
        if dispersion_scope == "global" and shrink_dispersion and use_map_dispersion:
            logger.info("Precomputing global dispersion trend (dispersion_scope='global')...")
                
            if use_streaming_mode:
                # Use path-based streaming for large datasets
                # Reads chunks from disk, never loads full matrix
                control_cache = precompute_global_dispersion_from_path(
                    path=path,
                    control_cache=control_cache,
                    all_cell_offset=offset,
                    fit_type="parametric",
                )
                drop_file_cache(path)  # Free page cache from streaming reads
            else:
                # Load all cells for global dispersion estimation
                backed = read_backed(path)
                try:
                    all_cell_matrix = backed.X[:]
                    if sp.issparse(all_cell_matrix):
                        all_cell_matrix = sp.csr_matrix(all_cell_matrix, dtype=np.float64)
                    else:
                        all_cell_matrix = np.asarray(all_cell_matrix, dtype=np.float64)
                finally:
                    backed.file.close()
                    
                # Compute global dispersion using all cells
                # Use fast_mode=True for speed (MoM + trend shrinkage instead of MAP)
                # Memory-adaptive: switches to streaming if matrix too large
                control_cache = precompute_global_dispersion(
                    control_cache=control_cache,
                    all_cell_matrix=all_cell_matrix,
                    all_cell_offset=offset,
                    n_grid=25,
                    fit_type="parametric",
                    fast_mode=True,  # ~50× faster than full MAP
                    max_dense_fraction=max_dense_fraction,
                    memory_limit_gb=memory_limit_gb,
                )
                del all_cell_matrix  # Free memory
                gc.collect()  # Force garbage collection before spawning workers
                drop_file_cache(path)  # Free page cache from full matrix read
                
            logger.info(f"Global dispersion precomputed: prior_var={control_cache.global_disp_prior_var:.4f}")
        
    # Log progress info
    if resume and completed_labels:
        logger.info(f"Fitting {n_to_run}/{n_groups} remaining perturbations with {effective_n_jobs} workers...")
    else:
        logger.info(f"Fitting {n_groups} perturbations with {effective_n_jobs} workers...")
        
    # Track completed labels during this run
    newly_completed = list(completed_labels)  # Start with already completed
    newly_failed: list[str] = []
    n_processed = 0
        
    # Helper function to write result to memmap
    def _write_result_to_memmap(res: dict, label: str) -> None:
        idx = candidate_to_idx[label]
        statistic_memmap[idx, :] = res["statistic"]
        pvalue_memmap[idx, :] = res["pvalue"]
        logfc_memmap[idx, :] = res["logfc"]
        logfc_raw_memmap[idx, :] = res["logfc_raw"]
        intercept_memmap[idx, :] = res["intercept"]  # MLE intercept for shrink_lfc
        se_memmap[idx, :] = res["se"]
        pts_memmap[idx, :] = res["pts"]
        dispersion_memmap[idx, :] = res["dispersion"]
        dispersion_raw_memmap[idx, :] = res["dispersion_raw"]
        dispersion_trend_memmap[idx, :] = res["dispersion_trend"]
        iter_memmap[idx, :] = res["iterations"]
        convergence_memmap[idx, :] = res["converged"]
        
    # Helper to save checkpoint
    def _save_checkpoint() -> None:
        checkpoint_data = {
            "total": n_groups,
            "completed": newly_completed,
            "failed": newly_failed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run.save(**checkpoint_data)
        
    # Run fitting with progress tracking
    n_tested_list: list[int] = []
    with _create_progress_context(n_to_run, "NB-GLM DE", verbose) as pbar:
        if use_parallel:
            # Use joblib.Parallel with loky backend for true process-based parallelism
            # This avoids GIL contention that limits ThreadPoolExecutor performance
            if can_use_cache:
                results = Parallel(
                    n_jobs=effective_n_jobs,
                    backend="loky",
                    prefer="processes",
                    return_as="generator",  # Stream results for progress updates
                )(
                    delayed(_fit_perturbation_worker_cached)(
                        group_idx=candidate_to_idx[label],
                        label=label,
                        path=path,
                        labels=labels,
                        control_cache=control_cache,
                        size_factors=size_factors,
                        n_genes=n_genes,
                        min_cells_expressed=min_cells_expressed,
                        min_pct_ctrl=min_pct_ctrl,
                        min_pct_pert=min_pct_pert,
                        min_mean_ctrl=min_mean_ctrl,
                        min_mean_pert=min_mean_pert,
                        min_cells_ctrl=min_cells_ctrl,
                        min_cells_pert=min_cells_pert,
                        min_total_count=min_total_count,
                        max_iter=max_iter,
                        tol=tol,
                        min_mu=min_mu,
                        dispersion_method=dispersion_method,
                        shrink_dispersion=shrink_dispersion,
                        use_map_dispersion=use_map_dispersion,
                        lfc_shrinkage_type=lfc_shrinkage_type,
                        se_method=se_method,
                        perturbation_boundaries=perturbation_boundaries,
                    )
                    for label in candidates_to_run
                )
            else:
                results = Parallel(
                    n_jobs=effective_n_jobs,
                    backend="loky",
                    prefer="processes",
                    return_as="generator",
                )(
                    delayed(_fit_perturbation_worker)(
                        group_idx=candidate_to_idx[label],
                        label=label,
                        path=path,
                        labels=labels,
                        control_mask=control_mask,
                        control_matrix=control_matrix,
                        control_expr_counts=control_expr_counts,
                        control_n=control_n,
                        obs_df=obs_df,
                        covariates=covariates,
                        size_factors=size_factors,
                        offset=offset,
                        n_genes=n_genes,
                        min_cells_expressed=min_cells_expressed,
                        min_pct_ctrl=min_pct_ctrl,
                        min_pct_pert=min_pct_pert,
                        min_mean_ctrl=min_mean_ctrl,
                        min_mean_pert=min_mean_pert,
                        min_cells_ctrl=min_cells_ctrl,
                        min_cells_pert=min_cells_pert,
                        min_total_count=min_total_count,
                        max_iter=max_iter,
                        tol=tol,
                        poisson_init_iter=poisson_init_iter,
                        dispersion_method=dispersion_method,
                        global_dispersion=global_dispersion,
                        shrink_dispersion=shrink_dispersion,
                        use_map_dispersion=use_map_dispersion,
                        lfc_shrinkage_type=lfc_shrinkage_type,
                        pts_rest_shared=pts_rest_shared,
                        full_X=full_X,
                        per_comparison_sf=(size_factor_scope == "per_comparison"),
                        se_method=se_method,
                        perturbation_boundaries=perturbation_boundaries,
                    )
                    for label in candidates_to_run
                )
                
            # Process results as they stream in
            for idx, res in enumerate(results):
                label = candidates_to_run[idx]
                try:
                    _write_result_to_memmap(res, label)
                    newly_completed.append(label)
                    n_tested_list.append(res.get("n_tested", 0))
                    _print_de_perturbation_verbose(verbose, label, res.get("n_tested", 0), n_genes)
                    logger.debug(f"Completed perturbation: {label}")
                except Exception as e:
                    logger.error(f"Failed perturbation {label}: {e}")
                    newly_failed.append(label)
                    
                n_processed += 1
                pbar.update(1)
                    
                # Save checkpoint periodically
                if n_processed % eff_checkpoint_interval == 0:
                    _save_checkpoint()
        else:
            # Run sequentially
            for label in candidates_to_run:
                group_idx = candidate_to_idx[label]
                try:
                    if can_use_cache:
                        res = _fit_perturbation_worker_cached(
                            group_idx=group_idx,
                            label=label,
                            path=path,
                            labels=labels,
                            control_cache=control_cache,
                            size_factors=size_factors,
                            n_genes=n_genes,
                            min_cells_expressed=min_cells_expressed,
                            min_pct_ctrl=min_pct_ctrl,
                            min_pct_pert=min_pct_pert,
                            min_mean_ctrl=min_mean_ctrl,
                            min_mean_pert=min_mean_pert,
                            min_cells_ctrl=min_cells_ctrl,
                            min_cells_pert=min_cells_pert,
                            min_total_count=min_total_count,
                            max_iter=max_iter,
                            tol=tol,
                            min_mu=min_mu,
                            dispersion_method=dispersion_method,
                            shrink_dispersion=shrink_dispersion,
                            use_map_dispersion=use_map_dispersion,
                            lfc_shrinkage_type=lfc_shrinkage_type,
                            se_method=se_method,
                            perturbation_boundaries=perturbation_boundaries,
                        )
                    else:
                        res = _fit_perturbation_worker(
                            group_idx=group_idx,
                            label=label,
                            path=path,
                            labels=labels,
                            control_mask=control_mask,
                            control_matrix=control_matrix,
                            control_expr_counts=control_expr_counts,
                            control_n=control_n,
                            obs_df=obs_df,
                            covariates=covariates,
                            size_factors=size_factors,
                            offset=offset,
                            n_genes=n_genes,
                            min_cells_expressed=min_cells_expressed,
                            min_pct_ctrl=min_pct_ctrl,
                            min_pct_pert=min_pct_pert,
                            min_mean_ctrl=min_mean_ctrl,
                            min_mean_pert=min_mean_pert,
                            min_cells_ctrl=min_cells_ctrl,
                            min_cells_pert=min_cells_pert,
                            min_total_count=min_total_count,
                            max_iter=max_iter,
                            tol=tol,
                            min_mu=min_mu,
                            poisson_init_iter=poisson_init_iter,
                            dispersion_method=dispersion_method,
                            global_dispersion=global_dispersion,
                            shrink_dispersion=shrink_dispersion,
                            use_map_dispersion=use_map_dispersion,
                            lfc_shrinkage_type=lfc_shrinkage_type,
                            pts_rest_shared=pts_rest_shared,
                            full_X=full_X,
                            per_comparison_sf=(size_factor_scope == "per_comparison"),
                            se_method=se_method,
                            perturbation_boundaries=perturbation_boundaries,
                        )
                    _write_result_to_memmap(res, label)
                    newly_completed.append(label)
                    n_tested_list.append(res.get("n_tested", 0))
                    _print_de_perturbation_verbose(verbose, label, res.get("n_tested", 0), n_genes)
                    logger.debug(f"Completed perturbation: {label}")
                except Exception as e:
                    logger.error(f"Failed perturbation {label}: {e}")
                    newly_failed.append(label)
                    
                n_processed += 1
                pbar.update(1)
                    
                # Save checkpoint periodically
                if n_processed % eff_checkpoint_interval == 0:
                    _save_checkpoint()
        
    # Final checkpoint save
    _save_checkpoint()
    logger.info(f"Completed {len(newly_completed)}/{n_groups} perturbations")
    _print_de_summary(verbose, "NB-GLM DE", len(newly_completed), n_groups, n_tested_list, n_genes)
    if newly_failed:
        logger.warning(f"Failed {len(newly_failed)} perturbations: {newly_failed[:5]}{'...' if len(newly_failed) > 5 else ''}")

    pvalue_adj_memmap = run.arrays["pvalue_adj"]
    _adjust_pvalue_matrix(pvalue_memmap, corr_method, out=pvalue_adj_memmap)

    gene_symbols = pd.Index(gene_symbols).astype(str)
    statistic_for_order = np.where(
        np.isfinite(statistic_memmap), np.abs(statistic_memmap), -np.inf
    )
    order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")

    # The effect size is the LFC: every fit path sets ``effect`` to a copy
    # of ``logfc`` (shrunk or not), so the LFC is stored once, as X.
    logfc_matrix = np.array(logfc_memmap)
    # The MLE LFC differs from X only when apeGLM shrank it.
    shrunk = lfc_shrinkage_type == "apeglm"
    logfc_raw_matrix = np.array(logfc_raw_memmap) if shrunk else None
    statistic_matrix = np.array(statistic_memmap)
    pvalue_matrix = np.array(pvalue_memmap)
    pvalue_adj_matrix = np.array(pvalue_adj_memmap)
    intercept_matrix = np.array(intercept_memmap)  # MLE intercept for shrink_lfc
    se_matrix = np.array(se_memmap)
    pts_matrix = np.array(pts_memmap, dtype=np.float32)
    iter_matrix = np.array(iter_memmap)
    convergence_matrix = np.array(convergence_memmap)
    # A precomputed global dispersion is one value per gene, copied into
    # every perturbation's row; store it once in var. Otherwise dispersion
    # was estimated per comparison and stays a matrix.
    global_dispersion_fit = control_cache is not None and control_cache.global_dispersion is not None
    if not global_dispersion_fit:
        dispersion_matrix = np.array(dispersion_memmap)
        dispersion_raw_matrix = np.array(dispersion_raw_memmap)
        dispersion_trend_matrix = np.array(dispersion_trend_memmap)
    # pts_rest describes the control arm: the same vector for every row.
    pts_rest_vector = (
        control_cache.pts_rest if control_cache is not None else pts_rest_shared
    ).astype(np.float32)

    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame({"pts_rest": pts_rest_vector}, index=gene_symbols)
    if global_dispersion_fit:
        var["dispersion"] = control_cache.global_dispersion
        var["dispersion_trend"] = (
            control_cache.global_dispersion_trend
            if control_cache.global_dispersion_trend is not None
            else control_cache.global_dispersion
        )

    # Convert from natural log to log2 if requested (PyDESeq2/edgeR convention).
    # shrink_lfc needs the ln-scale MLE values; with a log2 base they are kept
    # as separate *_ln layers, with an ln base the main layers already are.
    ln_layers = {}
    if lfc_base == "log2":
        ln2 = np.log(2)
        ln_layers["logfoldchange_raw_ln"] = logfc_raw_matrix if shrunk else logfc_matrix
        ln_layers["standard_error_ln"] = se_matrix
        logfc_matrix = logfc_matrix / ln2
        if shrunk:
            logfc_raw_matrix = logfc_raw_matrix / ln2
        se_matrix = se_matrix / ln2

    adata = ad.AnnData(logfc_matrix, obs=obs, var=var)
    adata.layers["z_score"] = statistic_matrix
    adata.layers["pvalue"] = pvalue_matrix
    adata.layers["pvalue_adj"] = pvalue_adj_matrix
    if shrunk:
        adata.layers["logfoldchange_raw"] = logfc_raw_matrix
    adata.layers["intercept"] = intercept_matrix  # MLE intercept (ln-scale) for shrink_lfc
    adata.layers["standard_error"] = se_matrix
    for name, matrix in ln_layers.items():
        adata.layers[name] = matrix
    if not global_dispersion_fit:
        adata.layers["dispersion"] = dispersion_matrix
        adata.layers["dispersion_raw"] = dispersion_raw_matrix
        adata.layers["dispersion_trend"] = dispersion_trend_matrix
    adata.layers["converged"] = convergence_matrix
    # Iteration counts are small integers; the narrowest integer type that
    # holds the largest one stores them exactly.
    adata.layers["iterations"] = iter_matrix.astype(
        np.min_scalar_type(int(iter_matrix.max(initial=0)))
    )
    adata.layers["pts"] = pts_matrix
    adata.uns["lfc_base"] = lfc_base  # Store for downstream tools
    adata.uns["method"] = "nb_glm"
    adata.uns["fit_method"] = "independent"
    adata.uns["control_label"] = control_label
    adata.uns["perturbation_column"] = perturbation_column
    adata.uns["covariates"] = covariates
    adata.uns["size_factors"] = size_factors
    adata.uns["original_dataset_path"] = str(path)  # For shrink_lfc to reload data
    adata.uns["size_factor_method"] = size_factor_method
    adata.uns["size_factor_scope"] = size_factor_scope
    adata.uns["dispersion_scope"] = dispersion_scope
    adata.uns["de_filter"] = {
        "min_cells_expressed": int(min_cells_expressed),
        "min_pct_ctrl": float(min_pct_ctrl),
        "min_pct_pert": float(min_pct_pert),
        "min_mean_ctrl": float(min_mean_ctrl),
        "min_mean_pert": float(min_mean_pert),
    }

    # Store profiling results or "NA" for production
    if profiling and profiler is not None:
        profiler.stop("fit")
        profiler.snapshot("fit_end")
        profiler.stop("total")
        profiler.stop_sampling()
        stats = profiler.get_stats()
        adata.uns["profiling"] = {
            "profiling_enabled": True,
            "fit_seconds": stats.get("timing", {}).get("sections", {}).get("fit", {}).get("seconds", 0.0),
            "fit_peak_memory_mb": stats.get("memory", {}).get("peak_mb", 0.0),
            "total_seconds": stats.get("timing", {}).get("total_seconds", 0.0),
        }
    else:
        adata.uns["profiling"] = "NA"

    # output_path already resolved earlier for checkpoint
    if int(verbose) >= 1:
        print(f"[cx] nb_glm_test: Saving \u2192 {output_path}")
    adata.write(output_path)
    
    run.discard()  # the output is complete; the checkpoint and partial arrays are not needed

    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=statistic_matrix,
        pvalues=pvalue_matrix,
        pvalues_adj=pvalue_adj_matrix,
        logfoldchanges=logfc_matrix,
        effect_size=logfc_matrix,
        u_statistics=np.zeros_like(logfc_matrix),
        pts=pts_matrix,
        pts_rest=np.broadcast_to(pts_rest_vector, pts_matrix.shape),
        order=order_matrix,
        groupby=perturbation_column,
        method="nb_glm",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
        result=AnnData(output_path),
    )

    # Optionally write Scanpy-compatible rank_genes_groups structure
    if scanpy_format:
        _write_rank_genes_groups_hdf5(output_path, result)
        # Reload to pick up the new uns structure
        result.result = AnnData(output_path)
    
    return result


def _create_streaming_scaffold(
    path: Path,
    *,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    n_groups: int,
    n_genes: int,
    group_batch_size: int,
    control_label: str,
    perturbation_column: str,
    tie_correct: bool,
    corr_method: str,
) -> None:
    """Write the streaming Wilcoxon result file's obs/var/uns and its empty,
    batch-chunked X and layers, ready to be filled one group batch at a time.

    X is created directly as a chunked dataset, so no ``n_groups x n_genes``
    placeholder is ever materialised.
    """
    uns = _wilcoxon_uns(
        control_label=control_label, perturbation_column=perturbation_column,
        tie_correct=tie_correct, corr_method=corr_method,
    )
    ad.AnnData(obs=obs, var=var, uns=uns).write(path)
    with h5py.File(path, "r+") as hf:
        _create_array(hf, "X", shape=(n_groups, n_genes), dtype="float64",
                      chunks=(min(group_batch_size, n_groups), n_genes))

        layer_names = ["z_score", "pvalue", "pvalue_adj", "logfoldchanges",
                       "u_statistic", "pts"]
        layer_dtypes = {
            "z_score": "float64", "pvalue": "float64", "pvalue_adj": "float64",
            "logfoldchanges": "float64", "u_statistic": "float64",
            "pts": "float32",
        }
        layers_group = hf.require_group("layers")
        for name in layer_names:
            _create_array(
                layers_group, name, shape=(n_groups, n_genes), dtype=layer_dtypes[name],
                chunks=(min(group_batch_size, n_groups), n_genes),
            )



def _wilcoxon_test_streaming(
    path: Path,
    *,
    gene_symbols: pd.Index,
    perturbation_column: str,
    control_label: str,
    candidates: list[str],
    n_genes: int,
    chunk_size: int,
    min_cells_expressed: int,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
    tie_correct: bool,
    corr_method: str,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_interval: int | None,
    scanpy_format: bool,
    verbose: int | bool,
    resume: bool,
    fingerprint: dict,
    group_batch_size: int,
    memory_limit_gb: float | None = None,
) -> "RankGenesGroupsResult":
    """Memory-efficient Wilcoxon test that streams over perturbation group batches.

    Instead of allocating output arrays for ALL groups at once, processes groups
    in batches and writes results incrementally to the output h5ad via h5py.
    This keeps peak memory bounded by ``group_batch_size * n_genes`` rather than
    ``n_groups * n_genes``.

    The h5ad is built inside a :class:`ResumableRun` directory and moved to
    ``output_path`` only once every batch is written, so a file at
    ``output_path`` is always complete and an interrupted run resumes at its
    first unfinished batch.
    """
    n_groups = len(candidates)
    n_gene_chunks = (n_genes + chunk_size - 1) // chunk_size
    n_batches = (n_groups + group_batch_size - 1) // group_batch_size

    gene_symbols = pd.Index(gene_symbols).astype(str)

    # Pre-create h5ad scaffold with obs/var/uns and empty layer datasets
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    # pts_rest (the control arm's detection rate) is per gene; it is filled
    # in place as gene chunks stream past.
    var = pd.DataFrame({"pts_rest": np.zeros(n_genes, dtype=np.float32)}, index=gene_symbols)
    _wilcoxon_disk_estimate = warn_if_disk_space_low(
        _wilcoxon_disk_bytes(n_groups, n_genes),
        output_path,
        context="wilcoxon_test",
    )
    _messages.print_disk_estimate(verbose, "wilcoxon_test", _wilcoxon_disk_estimate)
    run = ResumableRun(
        output_path,
        checkpoint_path,
        fingerprint={**fingerprint, "path": "streaming", "group_batch_size": int(group_batch_size)},
        arrays={},
        resume=resume,
        requires=("result.h5ad",),
    )
    work_path = run.directory / "result.h5ad"
    last_completed_batch = run.progress.get("last_group_batch", -1)
    if run.resumed:
        _messages.vprint(verbose, "wilcoxon_test", f"Resuming at group batch {last_completed_batch + 1}/{n_batches}")
    else:
        _create_streaming_scaffold(
            work_path, obs=obs, var=var, n_groups=n_groups, n_genes=n_genes,
            group_batch_size=group_batch_size, control_label=control_label,
            perturbation_column=perturbation_column, tie_correct=tie_correct,
            corr_method=corr_method,
        )

    eff_checkpoint_interval = _get_checkpoint_interval(n_batches, checkpoint_interval)

    logger.info(
        f"Wilcoxon streaming mode: {n_groups} groups in {n_batches} batches "
        f"(batch_size={group_batch_size}), {n_gene_chunks} gene chunks "
        f"(chunk_size={chunk_size})"
    )

    def _check_not_count_like_streaming(chunk: sp.spmatrix) -> None:
        """Raise ValueError if the first gene chunk looks like raw counts."""
        if np.issubdtype(chunk.dtype, np.integer):
            raise ValueError(
                "Detected integer count data in wilcoxon_test. "
                "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
            )
        if np.issubdtype(chunk.dtype, np.floating):
            non_zero = chunk.data[chunk.data > 0]
            is_count_like = non_zero.size > 0 and np.all(np.isclose(non_zero, np.round(non_zero)))
            if is_count_like:
                raise ValueError(
                    "Detected count-like (integer-valued) floating point data in wilcoxon_test. "
                    "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                )

    def _save_streaming_checkpoint(batch_idx: int) -> None:
        # Each batch's h5py handle is closed (flushed) before this runs.
        run.save(
            total_group_batches=n_batches,
            last_group_batch=batch_idx,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    with _create_progress_context(n_batches, "Wilcoxon DE (group batches)", verbose) as pbar:
        for batch_idx in range(n_batches):
            if batch_idx <= last_completed_batch:
                pbar.update(1)
                continue

            batch_start = batch_idx * group_batch_size
            batch_end = min(batch_start + group_batch_size, n_groups)
            batch_candidates = candidates[batch_start:batch_end]
            bs = len(batch_candidates)

            # Allocate result arrays for this batch only
            batch_effect = np.zeros((bs, n_genes), dtype=np.float64)
            batch_u = np.zeros((bs, n_genes), dtype=np.float64)
            batch_z = np.zeros((bs, n_genes), dtype=np.float64)
            batch_p = np.full((bs, n_genes), np.nan, dtype=np.float64)
            batch_lfc = np.zeros((bs, n_genes), dtype=np.float64)
            batch_pts = np.zeros((bs, n_genes), dtype=np.float32)
            pts_rest_vector = np.zeros(n_genes, dtype=np.float32)

            # Stream gene chunks from backed file
            backed = read_backed(path)
            try:
                labels = backed.obs[perturbation_column].astype(str).to_numpy()
                control_mask = labels == control_label
                control_n = int(control_mask.sum())
                # Precompute integer row indices (faster than boolean indexing
                # on the dense block: O(n_pert) vs O(n_cells) per group)
                control_idx = np.where(control_mask)[0]
                batch_pert_idx = _group_row_indices(labels, batch_candidates)

                dtype_checked_streaming = False
                for slc, block in iter_matrix_chunks(
                    backed, axis=1, chunk_size=chunk_size, convert_to_dense=False,
                    warn_slow_axis=False,
                ):
                    if not sp.issparse(block):
                        raise ValueError(
                            "wilcoxon_test only supports sparse input matrices."
                        )
                    if not dtype_checked_streaming and batch_idx == 0:
                        _check_not_count_like_streaming(block)
                        dtype_checked_streaming = True
                    csr_block = sp.csr_matrix(block)  # Keep native dtype (float32)
                    n_chunk_genes = csr_block.shape[1]

                    # Control stats for this gene chunk (same for all groups)
                    control_values = csr_block[control_mask, :]
                    control_expr = np.asarray(control_values.getnnz(axis=0)).ravel()
                    control_mean = (
                        np.asarray(control_values.mean(axis=0)).ravel()
                        if control_values.nnz
                        else np.zeros(n_chunk_genes, dtype=np.float64)
                    )
                    control_mean_expm1 = np.expm1(control_mean) + 1e-9
                    control_pts_chunk = np.divide(
                        control_expr, control_n,
                        out=np.zeros_like(control_expr, dtype=float),
                        where=control_n > 0,
                    )
                    pts_rest_vector[slc] = control_pts_chunk

                    # Pre-compute perturbation summary stats from sparse
                    pert_expr_counts = []
                    pert_means = []
                    pert_n_cells = []
                    for label in batch_candidates:
                        gv = csr_block[batch_pert_idx[label], :]
                        n_pc = gv.shape[0]
                        pert_n_cells.append(n_pc)
                        ge = np.asarray(gv.getnnz(axis=0)).ravel()
                        pert_expr_counts.append(ge)
                        gm = (
                            np.asarray(gv.mean(axis=0)).ravel()
                            if gv.nnz
                            else np.zeros(n_chunk_genes, dtype=np.float64)
                        )
                        pert_means.append(gm)

                    # Determine valid genes per perturbation using the shared
                    # per-condition low-expression filter (drop genes that are
                    # jointly low in BOTH groups by both pct and mean).
                    valid_masks = []
                    for idx in range(bs):
                        ge = pert_expr_counts[idx]
                        gm = pert_means[idx]
                        total_expr = control_expr + ge
                        valid = total_expr >= min_cells_expressed
                        low_both = _low_expr_in_both_mask(
                            pert_expr_counts=ge,
                            control_expr_counts=control_expr,
                            pert_mean=gm,
                            control_mean=control_mean,
                            n_pert_cells=pert_n_cells[idx],
                            n_control_cells=control_n,
                            min_pct_ctrl=min_pct_ctrl,
                            min_pct_pert=min_pct_pert,
                            min_mean_ctrl=min_mean_ctrl,
                            min_mean_pert=min_mean_pert,
                        )
                        valid_masks.append(valid & ~low_both)

                    any_valid = np.zeros(n_chunk_genes, dtype=bool)
                    for v in valid_masks:
                        any_valid |= v
                    valid_gene_indices = np.where(any_valid)[0]
                    n_valid_genes = len(valid_gene_indices)

                    if n_valid_genes > 0:
                        # Convert valid-gene block to dense ONCE for all cells,
                        # then use integer indexing per group (O(n_pert) vs
                        # O(n_cells) for boolean masks on 400K-row arrays).
                        # Keep native dtype (float32 for typical h5ad) to halve
                        # working-set memory, consistent with Scanpy's wilcoxon.
                        # ctrl_sorted_flat is always float64 inside _presort_control_nonzeros.
                        all_valid_dense = csr_block[:, valid_gene_indices].toarray()
                        control_dense = all_valid_dense[control_idx, :]

                        # Pre-sort control non-zeros once per chunk (~14x
                        # speedup: avoids redundant sort across 4955 groups)
                        ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz, ctrl_n_z = \
                            _presort_control_nonzeros(control_dense)
                        ctrl_tie_sums_s = _compute_ctrl_tie_sums(
                            ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz
                        )

                        # Per-perturbation: pre-stack then call batched kernel
                        # (single prange over n_perts, avoids serial launch overhead)
                        bs_local = len(batch_candidates)
                        total_pert_cells_s = sum(
                            len(batch_pert_idx[label]) for label in batch_candidates
                        )
                        all_pert_stacked_s = np.empty(
                            (total_pert_cells_s, n_valid_genes), dtype=all_valid_dense.dtype
                        )
                        pert_row_offsets_s = np.zeros(bs_local + 1, dtype=np.int64)
                        valid_masks_2d_s = np.empty(
                            (bs_local, n_valid_genes), dtype=np.bool_
                        )
                        valid_u_s = np.zeros((bs_local, n_valid_genes), dtype=np.float64)
                        valid_z_s = np.zeros((bs_local, n_valid_genes), dtype=np.float64)
                        valid_p_s = np.ones((bs_local, n_valid_genes), dtype=np.float64)
                        valid_eff_s = np.zeros((bs_local, n_valid_genes), dtype=np.float64)

                        for idx, label in enumerate(batch_candidates):
                            rows = batch_pert_idx[label]
                            start_s = pert_row_offsets_s[idx]
                            end_s = start_s + len(rows)
                            pert_row_offsets_s[idx + 1] = end_s
                            all_pert_stacked_s[start_s:end_s] = all_valid_dense[rows, :]
                            valid_masks_2d_s[idx] = valid_masks[idx][valid_gene_indices]

                        _wilcoxon_batch_perts_presorted_numba(
                            control_dense,
                            ctrl_sorted_flat,
                            ctrl_offsets,
                            ctrl_n_nz,
                            ctrl_n_z,
                            ctrl_tie_sums_s,
                            all_pert_stacked_s,
                            pert_row_offsets_s,
                            valid_masks_2d_s,
                            tie_correct,
                            _ZERO_PARTITION_THRESHOLD,
                            valid_u_s,
                            valid_z_s,
                            valid_p_s,
                            valid_eff_s,
                        )

                        gene_pos = slc.start + valid_gene_indices
                        # Vectorized 2-D fancy-index write (bs_local calls → 4 calls)
                        batch_u[:, gene_pos] = valid_u_s
                        batch_z[:, gene_pos] = valid_z_s
                        batch_p[:, gene_pos] = valid_p_s
                        batch_effect[:, gene_pos] = valid_eff_s

                    # LFC and pts for this chunk
                    for idx in range(bs):
                        ge = pert_expr_counts[idx]
                        gm = pert_means[idx]
                        np_ = pert_n_cells[idx]
                        valid = valid_masks[idx]

                        # pts describes the data, not the comparison, so it is
                        # reported for untested genes too -- the only way to see
                        # an untested 0%-vs-95% knockdown.
                        pts = np.divide(
                            ge, float(np_),
                            out=np.zeros_like(ge, dtype=float),
                            where=np_ > 0,
                        )
                        lfc = np.log2((np.expm1(gm) + 1e-9) / control_mean_expm1)
                        lfc = np.where(valid, lfc, np.nan)

                        gene_pos = np.arange(slc.start, slc.stop)
                        batch_lfc[idx, gene_pos] = lfc
                        batch_pts[idx, gene_pos] = pts

                        # Applied per row after the block write: the kernel
                        # leaves its defaults in the columns it skips, and a
                        # gene excluded here may be tested for another
                        # perturbation in the same block.
                        if not valid.all():
                            invalid_pos = slc.start + np.where(~valid)[0]
                            batch_u[idx, invalid_pos] = np.nan
                            batch_z[idx, invalid_pos] = np.nan
                            batch_p[idx, invalid_pos] = np.nan
                            batch_effect[idx, invalid_pos] = np.nan
            finally:
                backed.file.close()

            # P-value adjustment and ordering for this batch
            batch_pvalue_adj = np.ones_like(batch_p)
            _adjust_pvalue_matrix(batch_p, corr_method, out=batch_pvalue_adj)

            # Write batch results to h5ad
            sl = slice(batch_start, batch_end)
            with h5py.File(work_path, "r+") as hf:
                hf["X"][sl, :] = batch_effect
                hf["layers/z_score"][sl, :] = batch_z
                hf["layers/pvalue"][sl, :] = batch_p
                hf["layers/pvalue_adj"][sl, :] = batch_pvalue_adj
                hf["layers/logfoldchanges"][sl, :] = batch_lfc
                hf["layers/u_statistic"][sl, :] = batch_u
                hf["layers/pts"][sl, :] = batch_pts
                hf["var/pts_rest"][:] = pts_rest_vector

            del batch_effect, batch_u, batch_z, batch_p, batch_lfc
            del batch_pts, batch_pvalue_adj
            gc.collect()
            _release_chunk_memory()  # return freed batch-array pages to OS before next batch

            # Checkpoint
            if (batch_idx + 1) % eff_checkpoint_interval == 0 or batch_idx == n_batches - 1:
                _save_streaming_checkpoint(batch_idx)
            pbar.update(1)

    logger.info(f"Completed all {n_batches} group batches")
    if int(verbose) >= 1:
        print(f"[cx] Wilcoxon DE: {n_groups} perturbations complete, {n_genes} genes")
    os.replace(work_path, output_path)
    run.discard()

    # Build RankGenesGroupsResult by reading back from h5ad.
    # Uses _build_result_from_h5ad which skips loading for very large results
    # (>25% of memory budget) to avoid OOM on datasets like Huang-HEK293T.
    result = _build_result_from_h5ad(
        output_path,
        candidates=candidates,
        gene_symbols=gene_symbols,
        perturbation_column=perturbation_column,
        control_label=control_label,
        tie_correct=tie_correct,
        corr_method=corr_method,
        memory_limit_gb=memory_limit_gb,
    )

    if scanpy_format and result.statistics.size > 0:
        _write_rank_genes_groups_hdf5(output_path, result)

    return result


def _wilcoxon_test_stratified(
    path: Path,
    *,
    gene_symbols,
    perturbation_column: str,
    control_label: str,
    batch_column: str,
    candidates: list[str],
    n_genes: int,
    chunk_size: int,
    min_cells_expressed: int,
    min_pct_ctrl: float,
    min_pct_pert: float,
    min_mean_ctrl: float,
    min_mean_pert: float,
    tie_correct: bool,
    corr_method: str,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_interval: int | None,
    scanpy_format: bool,
    verbose: int | bool,
    resume: bool,
    fingerprint: dict,
    memory_limit_gb: float | None,
) -> RankGenesGroupsResult:
    """Batch-stratified (van Elteren) Wilcoxon rank-sum test.

    Cells are ranked against the control **within each batch** and the
    per-stratum rank statistics are combined with unit weights.  The
    low-expression filter, log-fold changes and ``pts`` are computed on the
    pooled populations (batch-agnostic); only the rank test is stratified.

    Mirrors the standard single-pass memmap path but partitions the control and
    perturbation cells by ``batch_column`` and calls the stratified kernel.
    """
    n_groups = len(candidates)

    n_gene_chunks = (n_genes + chunk_size - 1) // chunk_size
    eff_checkpoint_interval = _get_checkpoint_interval(n_gene_chunks, checkpoint_interval)

    _wilcoxon_disk_estimate = warn_if_disk_space_low(
        _wilcoxon_disk_bytes(n_groups, n_genes),
        output_path,
        context="wilcoxon_test",
    )
    _messages.print_disk_estimate(verbose, "wilcoxon_test", _wilcoxon_disk_estimate)
    shape = (n_groups, n_genes)
    run = ResumableRun(
        output_path,
        checkpoint_path,
        fingerprint={**fingerprint, "path": "stratified"},
        arrays={
            "effect": (shape, np.float64, 0),
            "u_stat": (shape, np.float64, 0),
            "pvalue": (shape, np.float64, 1.0),
            "pvalue_adj": (shape, np.float64, 0),
            "z_score": (shape, np.float64, 0),
            "logfoldchange": (shape, np.float64, 0),
            "pts": (shape, np.float32, 0),
            "pts_rest": ((n_genes,), np.float32, 0),  # per gene, filled chunk by chunk
        },
        resume=resume,
    )
    last_completed_chunk = run.progress.get("last_gene_chunk", -1)
    if run.resumed:
        _messages.vprint(verbose, "wilcoxon_test", f"Resuming at gene chunk {last_completed_chunk + 1}/{n_gene_chunks}")
    effect_matrix = run.arrays["effect"]
    u_matrix = run.arrays["u_stat"]
    pvalue_matrix = run.arrays["pvalue"]
    z_matrix = run.arrays["z_score"]
    lfc_matrix = run.arrays["logfoldchange"]
    pts_matrix = run.arrays["pts"]
    pts_rest_vector = run.arrays["pts_rest"]

    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_mask = labels == control_label
        control_n = int(control_mask.sum())

        # ----- Batch codes (0..n_batches-1; NaN/unknown -> -1) -----
        batch_values = np.asarray(backed.obs[batch_column].to_numpy())
        batch_codes, _batch_uniques = pd.factorize(batch_values, sort=True)
        batch_codes = batch_codes.astype(np.int64)
        n_batches = len(_batch_uniques)
        n_excluded = int((batch_codes < 0).sum())
        if n_excluded > 0:
            _messages.warn(
                "wilcoxon_test",
                f"{n_excluded} cells have a missing '{batch_column}' value and are "
                "excluded from the batch-stratified test.",
                stacklevel=2,
            )
        if n_batches < 1:
            raise ValueError(
                f"Batch column '{batch_column}' contains no usable batches."
            )

        # ----- Control cells ordered by batch (contiguous per-batch ranges) -----
        control_idx = np.where(control_mask)[0]
        control_batch = batch_codes[control_idx]
        _ckeep = control_batch >= 0
        control_idx = control_idx[_ckeep]
        control_batch = control_batch[_ckeep]
        _corder = np.argsort(control_batch, kind="stable")
        control_idx_sorted = control_idx[_corder]
        control_batch_sorted = control_batch[_corder]
        _b_arange = np.arange(n_batches)
        ctrl_batch_start = np.searchsorted(control_batch_sorted, _b_arange, side="left")
        ctrl_batch_end = np.searchsorted(control_batch_sorted, _b_arange, side="right")
        control_batch_counts = np.bincount(control_batch, minlength=n_batches)
        batches_with_control = control_batch_counts > 0

        # ----- Perturbation cells grouped into (pert, batch) segments -----
        pert_idx = _group_row_indices(labels, candidates)
        seg_offsets = [0]
        seg_batch: list[int] = []
        pert_ptr = [0]
        all_pert_flat_idx_list = []
        pert_total_batch_counts = np.zeros(n_groups, dtype=np.int64)
        pert_shared_batch_counts = np.zeros(n_groups, dtype=np.int64)
        pert_cells_without_control_batch = np.zeros(n_groups, dtype=np.int64)
        running = 0
        for p_i, label in enumerate(candidates):
            p_cells = pert_idx[label]
            p_b = batch_codes[p_cells]
            _pkeep = p_b >= 0
            p_cells = p_cells[_pkeep]
            p_b = p_b[_pkeep]
            if p_b.size:
                p_batch_counts = np.bincount(p_b, minlength=n_batches)
                p_batches = p_batch_counts > 0
                shared_batches = p_batches & batches_with_control
                pert_total_batch_counts[p_i] = int(p_batches.sum())
                pert_shared_batch_counts[p_i] = int(shared_batches.sum())
                pert_cells_without_control_batch[p_i] = int(
                    p_batch_counts[~batches_with_control].sum()
                )
            _porder = np.argsort(p_b, kind="stable")
            p_cells = p_cells[_porder]
            p_b = p_b[_porder]
            all_pert_flat_idx_list.append(p_cells)
            if p_b.size:
                change = np.where(np.diff(p_b) != 0)[0] + 1
                starts = np.concatenate(([0], change))
                ends = np.concatenate((change, [p_b.size]))
                for st, en in zip(starts, ends):
                    running += int(en - st)
                    seg_batch.append(int(p_b[st]))
                    seg_offsets.append(running)
            pert_ptr.append(len(seg_batch))

        all_pert_flat_idx = (
            np.concatenate(all_pert_flat_idx_list)
            if all_pert_flat_idx_list
            else np.empty(0, dtype=np.int64)
        )
        seg_offsets = np.asarray(seg_offsets, dtype=np.int64)
        seg_batch_arr = np.asarray(seg_batch, dtype=np.int64)
        pert_ptr = np.asarray(pert_ptr, dtype=np.int64)
        untestable_pert_mask = pert_shared_batch_counts == 0
        n_untestable_perts = int(untestable_pert_mask.sum())
        if n_untestable_perts:
            example_idx = np.where(untestable_pert_mask)[0][:5]
            examples = [str(candidates[i]) for i in example_idx]
            _messages.warn(
                "wilcoxon_test",
                f"{n_untestable_perts} perturbation(s) have no batches "
                "containing both perturbation and control cells; stratified "
                "rank statistics for these perturbations will be "
                f"set to NaN. Examples: {examples}",
                stacklevel=2,
            )

        # Pooled control cell count per candidate (for LFC/pts, batch-agnostic)
        control_idx_all = np.where(control_mask)[0]

        dtype_checked = False

        def _check_not_count_like(chunk: sp.spmatrix) -> None:
            if np.issubdtype(chunk.dtype, np.integer):
                raise ValueError(
                    "Detected integer count data in wilcoxon_test. "
                    "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                )
            if np.issubdtype(chunk.dtype, np.floating):
                non_zero = chunk.data[chunk.data > 0]
                is_count_like = non_zero.size > 0 and np.all(np.isclose(non_zero, np.round(non_zero)))
                if is_count_like:
                    raise ValueError(
                        "Detected count-like (integer-valued) floating point data in wilcoxon_test. "
                        "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                    )

        # Resume at the first unfinished chunk; iter_matrix_chunks seeks
        # there rather than reading and discarding the completed ones.
        current_chunk = last_completed_chunk + 1
        n_chunks_processed = 0

        def _save_wilcoxon_checkpoint(chunk_idx: int) -> None:
            run.save(
                total_gene_chunks=n_gene_chunks,
                last_gene_chunk=chunk_idx,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        with _create_progress_context(
            n_gene_chunks, "Wilcoxon DE stratified (gene chunks)", verbose,
            initial=current_chunk,
        ) as pbar:
            for slc, block in iter_matrix_chunks(
                backed, axis=1, chunk_size=chunk_size, convert_to_dense=False,
                start_chunk=current_chunk, warn_slow_axis=False,
            ):
                if not dtype_checked:
                    if not sp.issparse(block):
                        raise ValueError(
                            "wilcoxon_test only supports sparse input matrices. Please provide a scipy sparse matrix (e.g., CSR/CSC)."
                        )
                    _check_not_count_like(block)
                    dtype_checked = True

                csr_block = sp.csr_matrix(block)
                n_chunk_genes = csr_block.shape[1]

                # ----- Pooled control stats (LFC / pts) -----
                control_values = csr_block[control_mask, :]
                control_expr = np.asarray(control_values.getnnz(axis=0)).ravel()
                control_mean = (
                    np.asarray(control_values.mean(axis=0)).ravel()
                    if control_values.nnz
                    else np.zeros(n_chunk_genes, dtype=np.float64)
                )
                control_mean_expm1 = np.expm1(control_mean) + 1e-9
                control_pts = np.divide(
                    control_expr,
                    control_n,
                    out=np.zeros_like(control_expr, dtype=float),
                    where=control_n > 0,
                )

                # ----- Pooled perturbation stats -----
                pert_expr_counts = []
                pert_means = []
                pert_n_cells = []
                for label in candidates:
                    group_values = csr_block[pert_idx[label], :]
                    pert_n_cells.append(group_values.shape[0])
                    pert_expr_counts.append(np.asarray(group_values.getnnz(axis=0)).ravel())
                    group_mean = (
                        np.asarray(group_values.mean(axis=0)).ravel()
                        if group_values.nnz
                        else np.zeros(n_chunk_genes, dtype=np.float64)
                    )
                    pert_means.append(group_mean)

                # ----- Per-condition low-expression filter (pooled) -----
                valid_masks = []
                for idx, label in enumerate(candidates):
                    group_expr = pert_expr_counts[idx]
                    group_mean = pert_means[idx]
                    total_expr = control_expr + group_expr
                    valid = total_expr >= min_cells_expressed
                    low_both = _low_expr_in_both_mask(
                        pert_expr_counts=group_expr,
                        control_expr_counts=control_expr,
                        pert_mean=group_mean,
                        control_mean=control_mean,
                        n_pert_cells=pert_n_cells[idx],
                        n_control_cells=control_n,
                        min_pct_ctrl=min_pct_ctrl,
                        min_pct_pert=min_pct_pert,
                        min_mean_ctrl=min_mean_ctrl,
                        min_mean_pert=min_mean_pert,
                    )
                    valid_masks.append(valid & ~low_both)

                rank_valid_masks = [
                    np.zeros_like(valid, dtype=bool)
                    if untestable_pert_mask[idx]
                    else valid
                    for idx, valid in enumerate(valid_masks)
                ]


                any_valid = np.zeros(n_chunk_genes, dtype=bool)
                for valid in rank_valid_masks:
                    any_valid |= valid
                valid_gene_indices = np.where(any_valid)[0]
                n_valid_genes = len(valid_gene_indices)

                chunk_u = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_z = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_p = np.full((n_groups, n_chunk_genes), np.nan, dtype=np.float64)
                chunk_effect = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_lfc = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_pts = np.zeros((n_groups, n_chunk_genes), dtype=np.float32)

                if n_valid_genes > 0:
                    all_valid_dense = csr_block[:, valid_gene_indices].toarray()
                    control_dense = all_valid_dense[control_idx_sorted, :]

                    # Pre-sort control non-zeros per (batch, gene).
                    ctrl_flats = []
                    ctrl_starts = np.zeros((n_batches, n_valid_genes), dtype=np.int64)
                    ctrl_nnz = np.zeros((n_batches, n_valid_genes), dtype=np.int64)
                    ctrl_nz = np.zeros((n_batches, n_valid_genes), dtype=np.int64)
                    ctrl_tie = np.zeros((n_batches, n_valid_genes), dtype=np.float64)
                    base = 0
                    for b in range(n_batches):
                        rows = control_dense[ctrl_batch_start[b]:ctrl_batch_end[b], :]
                        if rows.shape[0] == 0:
                            # No control cells in this batch: leave zeros.
                            continue
                        flat_b, off_b, nnz_b, nz_b = _presort_control_nonzeros(rows)
                        tie_b = _compute_ctrl_tie_sums(flat_b, off_b, nnz_b)
                        ctrl_starts[b] = off_b[:-1] + base
                        ctrl_nnz[b] = nnz_b
                        ctrl_nz[b] = nz_b
                        ctrl_tie[b] = tie_b
                        ctrl_flats.append(flat_b)
                        base += flat_b.shape[0]
                    ctrl_flat = (
                        np.concatenate(ctrl_flats)
                        if ctrl_flats
                        else np.empty(0, dtype=np.float64)
                    )

                    all_pert_stacked = all_valid_dense[all_pert_flat_idx, :]
                    valid_masks_2d = np.array(
                        [vm[valid_gene_indices] for vm in rank_valid_masks]
                    )

                    valid_u = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                    valid_z = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                    valid_p = np.ones((n_groups, n_valid_genes), dtype=np.float64)
                    valid_effect = np.zeros((n_groups, n_valid_genes), dtype=np.float64)

                    _wilcoxon_stratified_batch_perts_numba(
                        ctrl_flat,
                        ctrl_starts,
                        ctrl_nnz,
                        ctrl_nz,
                        ctrl_tie,
                        all_pert_stacked,
                        seg_offsets,
                        seg_batch_arr,
                        pert_ptr,
                        valid_masks_2d,
                        tie_correct,
                        valid_u,
                        valid_z,
                        valid_p,
                        valid_effect,
                    )

                    chunk_u[:, valid_gene_indices] = valid_u
                    chunk_z[:, valid_gene_indices] = valid_z
                    chunk_p[:, valid_gene_indices] = valid_p
                    chunk_effect[:, valid_gene_indices] = valid_effect

                # ----- LFC / pts (pooled, batch-agnostic) -----
                all_expr = np.array(pert_expr_counts)
                all_means = np.array(pert_means)
                all_n = np.array(pert_n_cells, dtype=np.float64)

                n_col = all_n[:, np.newaxis]
                # pts describes the data, not the comparison, so it is
                # reported for untested genes too -- the only way to see an
                # untested 0%-vs-95% knockdown.
                chunk_pts[:] = np.where(n_col > 0, all_expr / n_col, 0.0).astype(np.float32)
                pts_rest_vector[slc] = control_pts

                raw_lfc = np.log2(
                    (np.expm1(all_means) + 1e-9) / control_mean_expm1[np.newaxis, :]
                )

                # Untested genes carry NaN in every column derived from the
                # comparison.  Two masks, because only the rank test is
                # stratified: the test columns follow rank_valid_masks,
                # which also zeroes perturbations sharing no batch with the
                # control, while the pooled logfoldchanges follows the
                # pooled filter alone.
                valid_arr = np.array(valid_masks)
                rank_valid_arr = np.array(rank_valid_masks)
                invalid_arr = ~rank_valid_arr
                chunk_lfc[:] = np.where(valid_arr, raw_lfc, np.nan)
                if invalid_arr.any():
                    chunk_u[invalid_arr] = np.nan
                    chunk_z[invalid_arr] = np.nan
                    chunk_p[invalid_arr] = np.nan
                    chunk_effect[invalid_arr] = np.nan

                u_matrix[:, slc] = chunk_u
                pvalue_matrix[:, slc] = chunk_p
                effect_matrix[:, slc] = chunk_effect
                z_matrix[:, slc] = chunk_z
                lfc_matrix[:, slc] = chunk_lfc
                pts_matrix[:, slc] = chunk_pts

                del csr_block, chunk_u, chunk_z, chunk_p, chunk_effect, chunk_lfc
                del chunk_pts, pert_expr_counts, pert_means
                del pert_n_cells, valid_masks, rank_valid_masks
                del valid_arr, rank_valid_arr, invalid_arr
                if n_valid_genes > 0:
                    del all_valid_dense, control_dense, ctrl_flats, ctrl_flat
                    del ctrl_starts, ctrl_nnz, ctrl_nz, ctrl_tie
                    del all_pert_stacked, valid_u, valid_z, valid_p, valid_effect
                    del valid_masks_2d
                _release_chunk_memory()

                n_chunks_processed += 1
                pbar.update(1)
                if n_chunks_processed % eff_checkpoint_interval == 0:
                    _save_wilcoxon_checkpoint(current_chunk)
                current_chunk += 1

            _save_wilcoxon_checkpoint(current_chunk - 1)
            logger.info(f"Completed {n_chunks_processed} gene chunks (stratified)")
            if int(verbose) >= 1:
                # Untested pairs are NaN in every test column; counting
                # them at the end (not per chunk) stays right across a
                # resume.
                _valid_gene_counts = np.count_nonzero(~np.isnan(u_matrix), axis=1)
                if int(verbose) >= 2:
                    for _gi, _label in enumerate(candidates):
                        _print_de_perturbation_verbose(
                            verbose, _label, int(_valid_gene_counts[_gi]), n_genes
                        )
                _mean = int(_valid_gene_counts.mean()) if n_groups > 0 else 0
                _pct = 100.0 * _mean / n_genes if n_genes else 0
                print(
                    f"[cx] Wilcoxon DE (batch-stratified by '{batch_column}', "
                    f"{n_batches} batches): {n_groups} perturbations complete, "
                    f"mean {_mean}/{n_genes} genes tested ({_pct:.0f}%)"
                )
    finally:
        backed.file.close()

    gene_symbols = pd.Index(gene_symbols).astype(str)
    pvalue_adj_matrix = run.arrays["pvalue_adj"]
    _adjust_pvalue_matrix(pvalue_matrix, corr_method, out=pvalue_adj_matrix)

    if int(verbose) >= 1:
        print(f"[cx] wilcoxon_test: Saving \u2192 {output_path}")
    stratified_diagnostics = {
        "stratified_n_batches": int(n_batches),
        "stratified_n_control_batches": int(batches_with_control.sum()),
        "stratified_n_untestable_perturbations": int(n_untestable_perts),
        "stratified_min_shared_batches_per_perturbation": (
            int(pert_shared_batch_counts.min()) if n_groups else 0
        ),
        "stratified_median_shared_batches_per_perturbation": (
            float(np.median(pert_shared_batch_counts)) if n_groups else 0.0
        ),
        "stratified_perturbation_cells_without_control_batch": int(
            pert_cells_without_control_batch.sum()
        ),
    }
    _write_wilcoxon_result_h5ad(
        output_path,
        effect_matrix=effect_matrix,
        z_matrix=z_matrix,
        pvalue_matrix=pvalue_matrix,
        pvalue_adj_matrix=pvalue_adj_matrix,
        lfc_matrix=lfc_matrix,
        u_matrix=u_matrix,
        pts_matrix=pts_matrix,
        pts_rest=pts_rest_vector,
        candidates=candidates,
        gene_symbols=gene_symbols,
        perturbation_column=perturbation_column,
        control_label=control_label,
        tie_correct=tie_correct,
        corr_method=corr_method,
        batch_column=batch_column,
        stratified_diagnostics=stratified_diagnostics,
    )
    # Release the memmaps (and their pages) before reading the result back.
    del effect_matrix, u_matrix, pvalue_matrix, pvalue_adj_matrix, z_matrix
    del lfc_matrix, pts_matrix, pts_rest_vector
    run.discard()

    _release_chunk_memory()
    result = _build_result_from_h5ad(
        output_path,
        candidates=candidates,
        gene_symbols=gene_symbols,
        perturbation_column=perturbation_column,
        control_label=control_label,
        tie_correct=tie_correct,
        corr_method=corr_method,
        memory_limit_gb=memory_limit_gb,
    )

    if scanpy_format and result.statistics.size > 0:
        _write_rank_genes_groups_hdf5(output_path, result)

    return result


def wilcoxon_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    min_cells_expressed: int = 0,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_pct_both: float | None = None,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
    chunk_size: int | None = None,
    tie_correct: bool = True,
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    verbose: int | bool = True,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    scanpy_format: bool = False,
    memory_limit_gb: float | None = None,
    force: bool = False,
    format_mismatch_policy: Literal["auto", "warn", "convert", "off"] = "auto",
) -> RankGenesGroupsResult:
    """Perform a Wilcoxon rank-sum (Mann-Whitney U) test for each gene.

    Input data **must already be library-size normalised and log-transformed**.
    The function operates directly on the provided matrix without additional
    preprocessing. As a safeguard, the first sparse chunk is inspected and a
    warning is emitted if the data appear to be raw counts (integer or
    count-like floats), encouraging explicit preprocessing upstream.
    
    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        normalised, log-transformed data.
    perturbation_column
        Column in `adata.obs` indicating perturbation labels.
    groupby
        Alias for ``perturbation_column`` (Scanpy-compatible).
        Mutually exclusive with ``perturbation_column``.
    control_label
        Label for the control/reference group. If None, infers from common patterns.
    reference
        Alias for ``control_label`` (Scanpy-compatible).
        Mutually exclusive with ``control_label``.
    gene_name_column
        Column in `adata.var` with gene symbols. If None, uses `adata.var_names`.
    perturbations
        Specific perturbations to test. If None, tests all non-control groups.
    batch_column
        Column in ``adata.obs`` identifying the batch / stratum of each cell
        (e.g. 10x lane / gem-group). When provided, a **batch-stratified**
        (van Elteren) Wilcoxon test is performed: cells are ranked *within each
        batch* separately and the per-stratum rank statistics are combined with
        unit weights (equivalent to summing the Mann-Whitney ``U`` statistics
        across strata). This removes rank inflation caused by batch effects. The
        low-expression filter, log-fold changes and ``pts`` remain pooled across
        all cells; only the rank test is stratified. When ``None`` (default) the
        standard pooled Wilcoxon test is used.
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for
        testing. Genes below this threshold are untested: ``NaN`` in
        ``score`` / ``pvalue`` / ``logfoldchanges`` / ``effect_size``, with
        ``pts`` / ``pts_rest`` still populated.
    min_pct_ctrl
        Minimum fraction of expressing cells for the *control* side. A gene is
        excluded only when *both* sides are jointly low. Default ``0.01``.
    min_pct_pert
        Minimum fraction of expressing cells for the *perturbed* side.
        Default ``0.002`` (lower than ctrl; induction from near-zero baseline is
        biologically valid). Combined with ``min_mean_pert`` this forms a dual
        condition more robust than pct alone.
    min_pct_both
        If not ``None``, overrides both ``min_pct_ctrl`` and
        ``min_pct_pert`` with the same value.
    min_mean_ctrl
        Minimum mean log1p expression for the *control* side. Default ``0.05``.
        Excluded genes are written as NaN in ``score`` / ``pvalue`` /
        ``logfoldchanges`` / ``effect_size``; ``pts`` / ``pts_rest`` remain
        populated. Set to ``0.0`` together with pct thresholds to disable.
    min_mean_pert
        Minimum mean expression for the *perturbed* side. Default ``0.005``.
    chunk_size
        Number of genes to process per chunk (memory vs. speed tradeoff). Smaller
        values stream more, reducing peak memory at the cost of additional I/O.
    tie_correct
        Whether to apply tie correction to the U statistic. Default True for
        more accurate p-values when ties are present in the data.
    corr_method
        Method for p-value correction: "benjamini-hochberg" or "bonferroni".
    data_name
        Custom name for output file. If None, uses "wilcoxon" suffix.
    output_path
        Exact path for the output h5ad file. When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
        Deprecated; use ``output_path`` instead. Will be removed in the
        next major version.
    verbose
        If True, show a progress bar for gene chunk processing. Requires tqdm.
    resume
        If True, continue an interrupted run from its last checkpoint, at the
        first unfinished gene chunk (or perturbation batch, on the streaming
        path). The resolved ``chunk_size`` is part of the call's identity, so
        pass the same value (or leave both on auto with the same memory).
        Partial results are kept beside the output, in a hidden
        ``.<output name>.resume`` directory, together with the checkpoint
        ``<output>.progress.json``; both are removed once the output is
        written. A run resumes only from a checkpoint written by a call with
        the same input file, perturbations and result-affecting parameters;
        any other checkpoint is discarded and the run starts over. A
        completed output is loaded as usual (unless ``force=True``).
    checkpoint_interval
        Number of gene chunks between checkpoint saves. If None, auto-determined
        based on dataset size.
    scanpy_format
        If True, write Scanpy-compatible ``uns['rank_genes_groups']`` structure
        in addition to the layer-based storage. Adds ~2-6 seconds of I/O overhead
        for large datasets. Default False for performance.
    memory_limit_gb
        Maximum memory budget in GB. Controls whether the streaming path is
        used for very large datasets. When ``None`` (default), detects
        available system memory via ``psutil``. For HPC environments with
        fixed allocations, set this to your SLURM ``--mem`` value
        (e.g., ``memory_limit_gb=128``).
    force
        If True, rerun the analysis even when the output h5ad file already
        exists. If False (default), load and return the existing result
        instead of rerunning.
    format_mismatch_policy
        How to handle a source stored as CSR. The test streams the matrix by
        gene (column) chunks, which on CSR re-reads the *whole* matrix once per
        chunk and holds it in memory -- typically ~100x more I/O than CSC:

        * ``"auto"`` (default): measure the source's read throughput and
          convert only when the re-reads would actually cost more than one
          conversion does -- roughly, a large file on a slow (networked)
          filesystem streamed in many chunks. A file that re-reads in
          milliseconds is streamed as-is, silently. See
          :func:`crispyx.data.resolve_auto_format_mismatch_policy`.
        * ``"convert"``: always convert the source to CSC in a temporary file
          beside ``output_path`` (bounded-memory streaming via
          :func:`crispyx.data.convert_to_csc`, honouring ``memory_limit_gb``)
          and stream from that; the temporary file is removed before
          returning. Needs ~2x the source file's size in free disk space
          there; when that is not available the call falls back to ``"warn"``
          behaviour with a warning naming the shortfall. Run
          ``cx.pp.convert_to_csc`` once instead if several steps will reuse
          the file.
        * ``"warn"``: proceed on the CSR source after one warning that
          quantifies the cost.
        * ``"off"``: proceed silently.

    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].effect_size`, `result[label].pvalue`, etc. The h5ad file
        path is available at `result.result_path`.

        The h5ad file holds one row per perturbation: the effect size in
        ``X`` and the layers ``z_score``, ``pvalue``, ``pvalue_adj``,
        ``logfoldchanges``, ``u_statistic`` and ``pts``. ``pts_rest``
        describes the control arm, so it is the per-gene column
        ``var["pts_rest"]``.

    Notes
    -----
    The rank test itself runs in a numba ``prange`` kernel over perturbations
    and uses every CPU the process is allowed to run on; there is no separate
    worker-count parameter.

    ``logfoldchanges`` follows scanpy's formula,
    ``log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9))``. When a
    gene has no expressing cells in one arm, that arm's term *is* the
    constant, so the magnitude reported is set by the constant -- its ceiling
    is ``log2(1 / 1e-9) = 29.9`` -- and not by the data. The p-values are
    unaffected, but ranking on ``logfoldchanges`` puts such genes above every
    real effect. ``pts`` / ``pts_rest`` tell them apart: a complete knockdown
    has ``pts = 0`` beside a high ``pts_rest``, an undetectable gene has both
    near zero.
    """
    call_args = dict(locals())  # for the resume fingerprint, before any other local

    perturbation_column, control_label, min_pct_ctrl, min_pct_pert = _resolve_de_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        min_pct_both=min_pct_both,
        min_pct_ctrl=min_pct_ctrl,
        min_pct_pert=min_pct_pert,
        fn_name="wilcoxon_test",
    )
    validate_format_mismatch_policy(format_mismatch_policy)

    path = resolve_data_path(data)
    output_suffix = "wilcoxon_stratified" if batch_column is not None else "wilcoxon"
    output_path = resolve_output_path(
        path, suffix=output_suffix, output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        if batch_column is not None and batch_column not in backed.obs.columns:
            raise KeyError(
                f"Batch column '{batch_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        if control_n == 0:
            raise ValueError("Control group contains no cells")
        # Memory-efficient validation: avoid creating per-group bool masks (n_groups × n_cells
        # memory, e.g. 18K groups × 3.4M cells = 62 GB for Huang-HCT116). Instead use
        # np.unique to get the set of observed labels in a single O(n_cells) pass.
        _observed_labels = set(np.unique(labels).tolist())
        _missing_perts = [lbl for lbl in candidates if lbl not in _observed_labels]
        if _missing_perts:
            raise ValueError(
                f"Perturbation(s) {_missing_perts[:3]}"
                f"{'...' if len(_missing_perts) > 3 else ''} contain no cells"
            )
        
        # Calculate adaptive gene chunk_size if not provided.
        # Use the dedicated Wilcoxon calculator: no n_groups cap (output arrays
        # are memmapped to disk so RAM is independent of n_groups).
        if chunk_size is None:
            chunk_size = calculate_wilcoxon_chunk_size(
                backed.n_obs, backed.n_vars,
                available_memory_gb=memory_limit_gb,
            )
            _messages.vprint(verbose, "wilcoxon_test", f"chunk_size={chunk_size} (auto)")
    finally:
        backed.file.close()

    n_groups = len(candidates)

    expected_metadata = {
        "method": "wilcoxon",
        "perturbation_column": perturbation_column,
        "control_label": control_label,
        "tie_correct": bool(tie_correct),
        "pvalue_correction": corr_method,
        "batch_column": batch_column,
        "stratified": batch_column is not None,
    }
    if (r := _try_load_existing_de_result(
        output_path, force=force, verbose=verbose,
        method_name="wilcoxon", memory_limit_gb=memory_limit_gb,
        expected_metadata=expected_metadata,
    )):
        return r
    
    # Determine output path and checkpoint path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")

    # The resolved chunk_size is part of the identity: checkpoints count
    # gene chunks.
    fingerprint = _de_fingerprint(
        path, call_args, method="wilcoxon", candidates=candidates, n_genes=n_genes,
        chunk_size=chunk_size,
    )

    # =========================================================================
    # Dispatch. Every path streams gene chunks (axis=1), which on a CSR source
    # re-reads the whole matrix once per chunk; stream_on_fast_axis resolves
    # that per format_mismatch_policy (temporary CSC copy beside the output
    # by default) before any X access.
    # =========================================================================
    with stream_on_fast_axis(
        path, axis=1, policy=format_mismatch_policy, fn_name="wilcoxon_test",
        scratch_dir=output_path.parent, chunk_size=chunk_size,
        memory_limit_gb=memory_limit_gb, verbose=verbose, resumable=resume,
    ) as stream_path:
        # Batch-stratified (van Elteren): rank statistics are computed
        # within-batch and combined. Dedicated standard (memmap) path; the
        # group-batch streaming optimisation is not applied (result arrays are
        # memmapped to disk, so RAM stays bounded regardless of n_groups).
        if batch_column is not None:
            return _wilcoxon_test_stratified(
                stream_path,
                gene_symbols=gene_symbols,
                perturbation_column=perturbation_column,
                control_label=control_label,
                batch_column=batch_column,
                candidates=candidates,
                n_genes=n_genes,
                chunk_size=chunk_size,
                min_cells_expressed=min_cells_expressed,
                min_pct_ctrl=min_pct_ctrl,
                min_pct_pert=min_pct_pert,
                min_mean_ctrl=min_mean_ctrl,
                min_mean_pert=min_mean_pert,
                tie_correct=tie_correct,
                corr_method=corr_method,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval,
                scanpy_format=scanpy_format,
                verbose=verbose,
                resume=resume,
                fingerprint=fingerprint,
                memory_limit_gb=memory_limit_gb,
            )

        # Adaptive dispatch: use group-batch streaming for large datasets.
        use_streaming, _, _, group_batch_size = _should_use_streaming(
            n_groups, n_genes, memory_limit_gb=memory_limit_gb,
        )

        # Only stream when multiple batches are actually needed.
        # If group_batch_size >= n_groups (one batch = all groups), the standard memmap
        # path is strictly better: memmaps are OS-pageable and glibc returns their pages
        # before readback, keeping peak ~10 GB.  The streaming path keeps all result arrays
        # in Python heap and skips malloc_trim, leading to 20-35 GB peaks for mid-size
        # datasets like Feng-gwsnf (4,955 groups × 32,373 genes).
        if use_streaming and group_batch_size < n_groups:
            _messages.vprint(
                verbose, "wilcoxon_test",
                f"Strategy — streaming (batch_size={group_batch_size})",
            )
            return _wilcoxon_test_streaming(
                stream_path,
                gene_symbols=gene_symbols,
                perturbation_column=perturbation_column,
                control_label=control_label,
                candidates=candidates,
                n_genes=n_genes,
                chunk_size=chunk_size,
                min_cells_expressed=min_cells_expressed,
                min_pct_ctrl=min_pct_ctrl,
                min_pct_pert=min_pct_pert,
                min_mean_ctrl=min_mean_ctrl,
                min_mean_pert=min_mean_pert,
                tie_correct=tie_correct,
                corr_method=corr_method,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval,
                scanpy_format=scanpy_format,
                verbose=verbose,
                resume=resume,
                fingerprint=fingerprint,
                group_batch_size=group_batch_size,
                memory_limit_gb=memory_limit_gb,
            )

        _messages.vprint(verbose, "wilcoxon_test", "Strategy — single-pass")
        return _wilcoxon_test_standard(
            stream_path,
            gene_symbols=gene_symbols,
            perturbation_column=perturbation_column,
            control_label=control_label,
            candidates=candidates,
            n_genes=n_genes,
            chunk_size=chunk_size,
            min_cells_expressed=min_cells_expressed,
            min_pct_ctrl=min_pct_ctrl,
            min_pct_pert=min_pct_pert,
            min_mean_ctrl=min_mean_ctrl,
            min_mean_pert=min_mean_pert,
            tie_correct=tie_correct,
            corr_method=corr_method,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            scanpy_format=scanpy_format,
            verbose=verbose,
            resume=resume,
            fingerprint=fingerprint,
            memory_limit_gb=memory_limit_gb,
        )


def _wilcoxon_test_standard(
    path: Path,
    *,
    gene_symbols,
    perturbation_column: str,
    control_label: str,
    candidates: list[str],
    n_genes: int,
    chunk_size: int,
    min_cells_expressed: int,
    min_pct_ctrl: float,
    min_pct_pert: float,
    min_mean_ctrl: float,
    min_mean_pert: float,
    tie_correct: bool,
    corr_method: str,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_interval: int | None,
    scanpy_format: bool,
    verbose: int | bool,
    resume: bool,
    fingerprint: dict,
    memory_limit_gb: float | None,
) -> RankGenesGroupsResult:
    """Standard single-pass pooled Wilcoxon test with memmapped result arrays.

    Every group's per-chunk results go into ``(n_groups, n_genes)`` memmaps
    beside the output (a :class:`ResumableRun`, so an interrupted run resumes
    at its first unfinished gene chunk) and the h5ad is written from them at
    the end. Used whenever the group-batch streaming path is not needed.
    """
    n_groups = len(candidates)

    # Determine checkpoint interval (number of gene chunks between saves)
    n_gene_chunks = (n_genes + chunk_size - 1) // chunk_size
    eff_checkpoint_interval = _get_checkpoint_interval(n_gene_chunks, checkpoint_interval)

    _wilcoxon_disk_estimate = warn_if_disk_space_low(
        _wilcoxon_disk_bytes(n_groups, n_genes),
        output_path,
        context="wilcoxon_test",
    )
    _messages.print_disk_estimate(verbose, "wilcoxon_test", _wilcoxon_disk_estimate)
    shape = (n_groups, n_genes)
    run = ResumableRun(
        output_path,
        checkpoint_path,
        fingerprint={**fingerprint, "path": "standard"},
        arrays={
            "effect": (shape, np.float64, 0),
            "u_stat": (shape, np.float64, 0),
            "pvalue": (shape, np.float64, 1.0),
            "pvalue_adj": (shape, np.float64, 0),
            "z_score": (shape, np.float64, 0),
            "logfoldchange": (shape, np.float64, 0),
            "pts": (shape, np.float32, 0),
            # Per-gene, filled chunk by chunk.
            "pts_rest": ((n_genes,), np.float32, 0),
        },
        resume=resume,
    )
    last_completed_chunk = run.progress.get("last_gene_chunk", -1)
    if run.resumed:
        _messages.vprint(verbose, "wilcoxon_test", f"Resuming at gene chunk {last_completed_chunk + 1}/{n_gene_chunks}")
    effect_matrix = run.arrays["effect"]
    u_matrix = run.arrays["u_stat"]
    pvalue_matrix = run.arrays["pvalue"]
    z_matrix = run.arrays["z_score"]
    lfc_matrix = run.arrays["logfoldchange"]
    pts_matrix = run.arrays["pts"]
    pts_rest_vector = run.arrays["pts_rest"]

    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        # Precompute integer row indices (faster than boolean indexing
        # on the dense block: O(n_pert) vs O(n_cells) per group)
        control_idx = np.where(control_mask)[0]
        pert_idx = _group_row_indices(labels, candidates)

        # Pre-build flat perturbation indices and row offsets once
        # (avoids rebuilding inside the per-chunk stacking loop).
        all_pert_flat_idx = np.concatenate([pert_idx[label] for label in candidates])
        total_pert_cells = len(all_pert_flat_idx)
        pert_row_offsets = np.zeros(n_groups + 1, dtype=np.int64)
        for idx, label in enumerate(candidates):
            pert_row_offsets[idx + 1] = pert_row_offsets[idx] + len(pert_idx[label])

        dtype_checked = False

        def _check_not_count_like(chunk: sp.spmatrix) -> None:
            """Raise ValueError if the chunk looks like raw counts."""
            if np.issubdtype(chunk.dtype, np.integer):
                raise ValueError(
                    "Detected integer count data in wilcoxon_test. "
                    "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                )
            if np.issubdtype(chunk.dtype, np.floating):
                non_zero = chunk.data[chunk.data > 0]
                is_count_like = non_zero.size > 0 and np.all(np.isclose(non_zero, np.round(non_zero)))
                if is_count_like:
                    raise ValueError(
                        "Detected count-like (integer-valued) floating point data in wilcoxon_test. "
                        "Please log-normalize your data first (e.g. cx.pp.normalize_total_log1p)."
                    )

        # Resume at the first unfinished chunk; iter_matrix_chunks seeks
        # there rather than reading and discarding the completed ones.
        current_chunk = last_completed_chunk + 1
        n_chunks_processed = 0

        def _save_wilcoxon_checkpoint(chunk_idx: int) -> None:
            run.save(
                total_gene_chunks=n_gene_chunks,
                last_gene_chunk=chunk_idx,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        with _create_progress_context(
            n_gene_chunks, "Wilcoxon DE (gene chunks)", verbose,
            initial=current_chunk,
        ) as pbar:
            for slc, block in iter_matrix_chunks(
                backed, axis=1, chunk_size=chunk_size, convert_to_dense=False,
                start_chunk=current_chunk, warn_slow_axis=False,
            ):
                if not dtype_checked:
                    if not sp.issparse(block):
                        raise ValueError(
                            "wilcoxon_test only supports sparse input matrices. Please provide a scipy sparse matrix (e.g., CSR/CSC)."
                        )
                    _check_not_count_like(block)
                    dtype_checked = True

                csr_block = sp.csr_matrix(block)  # Keep native dtype (float32)
                n_chunk_genes = csr_block.shape[1]

                # ===== OPTIMIZED BATCH PROCESSING =====
                # 1. Extract control and perturbation data once
                control_values = csr_block[control_mask, :]
                control_expr = np.asarray(control_values.getnnz(axis=0)).ravel()
                control_mean = (
                    np.asarray(control_values.mean(axis=0)).ravel()
                    if control_values.nnz
                    else np.zeros(n_chunk_genes, dtype=np.float64)
                )
                control_mean_expm1 = np.expm1(control_mean) + 1e-9
                control_pts = np.divide(
                    control_expr,
                    control_n,
                    out=np.zeros_like(control_expr, dtype=float),
                    where=control_n > 0,
                )
                chunk_gene_indices = np.arange(slc.start, slc.stop)

                # 2. Pre-compute perturbation expression counts
                pert_expr_counts = []
                pert_means = []
                pert_n_cells = []
                for label in candidates:
                    group_values = csr_block[pert_idx[label], :]
                    n_pert_cells = group_values.shape[0]
                    pert_n_cells.append(n_pert_cells)
                    group_expr = np.asarray(group_values.getnnz(axis=0)).ravel()
                    pert_expr_counts.append(group_expr)
                    group_mean = (
                        np.asarray(group_values.mean(axis=0)).ravel()
                        if group_values.nnz
                        else np.zeros(n_chunk_genes, dtype=np.float64)
                    )
                    pert_means.append(group_mean)

                # 3. Determine valid genes per perturbation using the
                # shared per-condition low-expression filter (drop genes
                # that are jointly low in BOTH groups by both pct and mean).
                valid_masks = []
                for idx, label in enumerate(candidates):
                    group_expr = pert_expr_counts[idx]
                    group_mean = pert_means[idx]
                    total_expr = control_expr + group_expr
                    valid = total_expr >= min_cells_expressed
                    low_both = _low_expr_in_both_mask(
                        pert_expr_counts=group_expr,
                        control_expr_counts=control_expr,
                        pert_mean=group_mean,
                        control_mean=control_mean,
                        n_pert_cells=pert_n_cells[idx],
                        n_control_cells=control_n,
                        min_pct_ctrl=min_pct_ctrl,
                        min_pct_pert=min_pct_pert,
                        min_mean_ctrl=min_mean_ctrl,
                        min_mean_pert=min_mean_pert,
                    )
                    valid_masks.append(valid & ~low_both)

                # Accumulate per-perturbation valid gene counts for verbose output

                # 4. Find union of all valid genes (to minimize dense conversion)
                any_valid = np.zeros(n_chunk_genes, dtype=bool)
                for valid in valid_masks:
                    any_valid |= valid
                valid_gene_indices = np.where(any_valid)[0]
                n_valid_genes = len(valid_gene_indices)

                # 5. Initialize output arrays for this chunk
                chunk_u = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_z = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_p = np.full((n_groups, n_chunk_genes), np.nan, dtype=np.float64)
                chunk_effect = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_lfc = np.zeros((n_groups, n_chunk_genes), dtype=np.float64)
                chunk_pts = np.zeros((n_groups, n_chunk_genes), dtype=np.float32)

                if n_valid_genes > 0:
                    # 6. Convert valid-gene block to dense ONCE for all cells,
                    # then use integer indexing per group (O(n_pert) vs
                    # O(n_cells) for boolean masks on large arrays).
                    # Keep native dtype (float32 for typical h5ad) to halve
                    # working-set memory, consistent with Scanpy's wilcoxon.
                    # ctrl_sorted_flat is always float64 inside _presort_control_nonzeros.
                    all_valid_dense = csr_block[:, valid_gene_indices].toarray()
                    control_dense = all_valid_dense[control_idx, :]

                    # Pre-sort control non-zeros once per chunk (~14x
                    # speedup: avoids redundant sort across all groups)
                    ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz, ctrl_n_z = \
                        _presort_control_nonzeros(control_dense)
                    ctrl_tie_sums = _compute_ctrl_tie_sums(
                        ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz
                    )
                        
                    # 7. Pre-allocate output arrays for valid genes
                    valid_u = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                    valid_z = np.zeros((n_groups, n_valid_genes), dtype=np.float64)
                    valid_p = np.ones((n_groups, n_valid_genes), dtype=np.float64)
                    valid_effect = np.zeros((n_groups, n_valid_genes), dtype=np.float64)

                    # 8. Stack all pert dense matrices and call batched kernel.
                    # Single prange(n_perts) replaces n_perts serial kernel
                    # launches, eliminating the ~25ms prange thread-pool startup
                    # overhead per call (~50x speedup on kernel time).
                    # Vectorised: single fancy-index replaces n_groups-iteration loop.
                    all_pert_stacked = all_valid_dense[all_pert_flat_idx, :]
                    valid_masks_2d = np.array([vm[valid_gene_indices] for vm in valid_masks])

                    _wilcoxon_batch_perts_presorted_numba(
                        control_dense,
                        ctrl_sorted_flat,
                        ctrl_offsets,
                        ctrl_n_nz,
                        ctrl_n_z,
                        ctrl_tie_sums,
                        all_pert_stacked,
                        pert_row_offsets,
                        valid_masks_2d,
                        tie_correct,
                        _ZERO_PARTITION_THRESHOLD,
                        valid_u,
                        valid_z,
                        valid_p,
                        valid_effect,
                    )
                        
                    # 9. Map results back to full chunk gene indices
                    # Vectorised: single 2-D fancy-index write per array
                    # replaces n_groups Python-loop iterations.
                    chunk_u[:, valid_gene_indices] = valid_u
                    chunk_z[:, valid_gene_indices] = valid_z
                    chunk_p[:, valid_gene_indices] = valid_p
                    chunk_effect[:, valid_gene_indices] = valid_effect

                # 10. Compute LFC and pts — batch vectorised
                # (replaces n_groups Python-loop iterations)
                all_expr = np.array(pert_expr_counts)         # (n_groups, n_chunk_genes)
                all_means = np.array(pert_means)              # (n_groups, n_chunk_genes)
                all_n = np.array(pert_n_cells, dtype=np.float64)  # (n_groups,)
                valid_arr = np.array(valid_masks)             # (n_groups, n_chunk_genes)

                n_col = all_n[:, np.newaxis]                  # (n_groups, 1)
                # pts describes the data, not the comparison, so it is
                # reported for untested genes too -- the only way to see an
                # untested 0%-vs-95% knockdown.
                chunk_pts[:] = np.where(n_col > 0, all_expr / n_col, 0.0).astype(np.float32)
                pts_rest_vector[slc] = control_pts

                raw_lfc = np.log2(
                    (np.expm1(all_means) + 1e-9) / control_mean_expm1[np.newaxis, :]
                )

                # Untested genes carry NaN in every column derived from the
                # comparison.  Masked after the block write: the kernel
                # leaves its defaults in the columns it skips, and a gene
                # excluded here may be tested for another perturbation.
                invalid_arr = ~valid_arr
                chunk_lfc[:] = np.where(valid_arr, raw_lfc, np.nan)
                if invalid_arr.any():
                    chunk_u[invalid_arr] = np.nan
                    chunk_z[invalid_arr] = np.nan
                    chunk_p[invalid_arr] = np.nan
                    chunk_effect[invalid_arr] = np.nan

                # 13. Write results to memmap — vectorized 2-D slice
                # (7 calls instead of 7 × n_groups; better cache locality)
                u_matrix[:, slc] = chunk_u
                pvalue_matrix[:, slc] = chunk_p
                effect_matrix[:, slc] = chunk_effect
                z_matrix[:, slc] = chunk_z
                lfc_matrix[:, slc] = chunk_lfc
                pts_matrix[:, slc] = chunk_pts
                    
                # Release transient chunk arrays and return freed pages to OS
                # (prevents glibc arena fragmentation across many gene chunks)
                del csr_block, chunk_u, chunk_z, chunk_p, chunk_effect, chunk_lfc
                del chunk_pts, pert_expr_counts, pert_means
                del pert_n_cells, valid_masks, valid_arr, invalid_arr
                if n_valid_genes > 0:
                    del all_valid_dense, control_dense
                    del ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz, ctrl_n_z, ctrl_tie_sums
                    del all_pert_stacked, valid_u, valid_z, valid_p, valid_effect
                    del valid_masks_2d
                _release_chunk_memory()

                # Update progress and checkpoint
                n_chunks_processed += 1
                pbar.update(1)
                if n_chunks_processed % eff_checkpoint_interval == 0:
                    _save_wilcoxon_checkpoint(current_chunk)
                current_chunk += 1
                
            # Final checkpoint
            _save_wilcoxon_checkpoint(current_chunk - 1)
            logger.info(f"Completed {n_chunks_processed} gene chunks")
            if int(verbose) >= 1:
                # Untested pairs are NaN in every test column; counting them
                # at the end (not per chunk) stays right across a resume.
                _valid_gene_counts = np.count_nonzero(~np.isnan(u_matrix), axis=1)
                if int(verbose) >= 2:
                    for _gi, _label in enumerate(candidates):
                        _print_de_perturbation_verbose(verbose, _label, int(_valid_gene_counts[_gi]), n_genes)
                _mean = int(_valid_gene_counts.mean()) if n_groups > 0 else 0
                _pct = 100.0 * _mean / n_genes if n_genes else 0
                print(f"[cx] Wilcoxon DE: {n_groups} perturbations complete, mean {_mean}/{n_genes} genes tested ({_pct:.0f}%)")
    finally:
        backed.file.close()

    gene_symbols = pd.Index(gene_symbols).astype(str)
    pvalue_adj_matrix = run.arrays["pvalue_adj"]
    _adjust_pvalue_matrix(pvalue_matrix, corr_method, out=pvalue_adj_matrix)

    # Write h5ad directly from memmaps via h5py (avoids triple allocation:
    # memmap + np.array copy + AnnData that previously caused OOM)
    if int(verbose) >= 1:
        print(f"[cx] wilcoxon_test: Saving \u2192 {output_path}")
    _write_wilcoxon_result_h5ad(
        output_path,
        effect_matrix=effect_matrix,
        z_matrix=z_matrix,
        pvalue_matrix=pvalue_matrix,
        pvalue_adj_matrix=pvalue_adj_matrix,
        lfc_matrix=lfc_matrix,
        u_matrix=u_matrix,
        pts_matrix=pts_matrix,
        pts_rest=pts_rest_vector,
        candidates=candidates,
        gene_symbols=gene_symbols,
        perturbation_column=perturbation_column,
        control_label=control_label,
        tie_correct=tie_correct,
        corr_method=corr_method,
    )
    # Release the memmaps (and their pages) before reading the result back.
    del effect_matrix, u_matrix, pvalue_matrix, pvalue_adj_matrix, z_matrix
    del lfc_matrix, pts_matrix, pts_rest_vector
    run.discard()

    _release_chunk_memory()
    result = _build_result_from_h5ad(
        output_path,
        candidates=candidates,
        gene_symbols=gene_symbols,
        perturbation_column=perturbation_column,
        control_label=control_label,
        tie_correct=tie_correct,
        corr_method=corr_method,
        memory_limit_gb=memory_limit_gb,
    )

    # Optionally write Scanpy-compatible rank_genes_groups structure
    if scanpy_format and result.statistics.size > 0:
        _write_rank_genes_groups_hdf5(output_path, result)

    return result


def shrink_lfc(
    data: str | Path | AnnData | ad.AnnData,
    *,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    method: Literal["stats", "full"] = "stats",
    prior_scale_mode: Literal["global", "per_comparison"] = "global",
    min_mu: float = 0.0,
    n_jobs: int = -1,
    batch_size: int = 128,
    profiling: bool = False,
    memory_limit_gb: float | None = None,
    verbose: int | bool = True,
) -> RankGenesGroupsResult:
    """Apply apeGLM log-fold change shrinkage to existing NB-GLM results.
    
    This function applies apeGLM shrinkage using a Cauchy prior on the LFC
    coefficient. Two methods are available:
    
    - **stats** (default, recommended): Uses pre-computed MLE LFC and SE from
      the h5ad file with vectorized Newton-Raphson optimization. ~35× faster
      than full and maintains consistency with stored MLE coefficients.
    - **full**: Re-loads original count data and runs per-gene L-BFGS-B
      optimization. May produce different results from stored MLE for lowly
      expressed genes due to min_mu clamping differences in the likelihood.
    
    .. note::
        The "stats" method is recommended because CRISPYx NB-GLM fitting uses
        min_mu=0.5 clamping for numerical stability, which affects the stored
        MLE coefficients. The "full" method re-evaluates the likelihood without
        this constraint, potentially finding different optima for lowly expressed
        genes. The "stats" method preserves shrinkage direction (always toward
        zero) by working directly with the stored statistics.
    
    This enables separating the base NB-GLM fitting from shrinkage for:
    - Benchmarking: measure base fitting and shrinkage times separately
    - Flexibility: apply shrinkage to existing results
    - Speed: use stats method for production
    
    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx/anndata AnnData object containing
        NB-GLM results from `nb_glm_test`. Must have required layers and
        metadata in `uns`.
    output_dir
        Directory for output h5ad file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, appends "_shrunk" to input name.
    method
        Shrinkage method to use:
        
        - ``"stats"`` (default): Fast vectorized shrinkage using pre-computed
          MLE statistics. Uses Newton-Raphson optimization across all genes
          simultaneously. ~35× faster than "full" and maintains consistency
          with stored MLE coefficients.
        - ``"full"``: Full model re-fitting with L-BFGS-B per gene. Re-loads
          original count data. Note: May produce different results for lowly
          expressed genes due to min_mu clamping differences. Use "stats" for
          consistent shrinkage behavior.
    prior_scale_mode
        How to estimate the Cauchy prior scale parameter:
        
        - ``"global"`` (default): Estimate prior scale once from all 
          perturbations' MLE LFCs. Faster and often more stable.
        - ``"per_comparison"``: Estimate prior scale separately for each
          perturbation vs control comparison. Matches PyDESeq2's behavior
          exactly. Use for benchmarking to demonstrate parity.
    min_mu
        Minimum mean threshold for shrinkage likelihood evaluation. Default: 0.0
        (no clamping), matching PyDESeq2's lfc_shrink which omits min_mu entirely.
        PyDESeq2 uses min_mu=0.5 for NB-GLM fitting but does NOT pass min_mu to
        the shrinkage optimizer. This is intentional: the MLE coefficients represent
        the best fit, and shrinkage should evaluate the same likelihood surface.
    n_jobs
        Number of parallel jobs for per-gene optimization (only used when
        method="full"). Default -1 uses all available cores.
    batch_size
        Number of genes per joblib batch (only used when method="full").
        Default: 128, matching PyDESeq2.
    profiling
        If True, enable timing and memory profiling. When enabled, stores
        profiling data in `adata.uns["profiling"]` with fields:
        - `shrinkage_seconds`: Time for lfcShrink operation
        - `shrinkage_peak_memory_mb`: Peak memory during shrinkage
        - `profiling_enabled`: True
        When False (default), `adata.uns["profiling"]` is set to "NA".
    memory_limit_gb
        Optional memory budget in gigabytes. When ``method="full"``, this
        limits the number of parallel ``n_jobs`` so that joblib workers stay
        within the budget. When ``None`` (default), detects available system
        memory via ``psutil``.
    verbose
        Print basic progress/completion messages. Defaults to ``True``.

    Returns
    -------
    RankGenesGroupsResult
        Updated differential expression results with shrunken LFCs.
        The result h5ad has:
        - `logfoldchange`: shrunken LFC values
        - `logfoldchange_raw`: original MLE LFC values (preserved)
        - `standard_error`: posterior SE reflecting shrinkage uncertainty
        - `X`: updated to shrunken LFC (effect_size)
    
    Examples
    --------
    >>> # Fast default (recommended for production)
    >>> shrunk = crispyx.de.shrink_lfc("nb_glm_result.h5ad")
    
    >>> # Benchmark-accurate (matches PyDESeq2 exactly)
    >>> shrunk = crispyx.de.shrink_lfc(
    ...     "nb_glm_result.h5ad",
    ...     method="full",
    ...     prior_scale_mode="per_comparison",
    ... )
    
    >>> # First run NB-GLM without shrinkage
    >>> result = crispyx.de.nb_glm_test(
    ...     "data.h5ad",
    ...     perturbation_column="perturbation",
    ...     lfc_shrinkage_type="none",  # No shrinkage during fitting
    ... )
    >>> # Then apply shrinkage as a separate step
    >>> shrunk_result = crispyx.de.shrink_lfc(result.result_path)
    """
    path = resolve_data_path(data)
    
    # Validate min_mu parameter
    if min_mu < 0:
        raise ValueError(f"min_mu must be >= 0, got {min_mu}")
    
    # Initialize profiler if enabled (timing + memory sampling)
    profiler = None
    if profiling:
        from .profiling import Profiler
        profiler = Profiler(timing=True, memory=True, memory_method="rss", sampling=True)
        profiler.start("total")

    # Load the NB-GLM result
    adata = ad.read_h5ad(path)
    
    # Establish memory baseline AFTER loading h5ad for fair benchmarking
    # This way, profiling measures only shrinkage memory, not h5ad loading
    if profiling and profiler is not None:
        profiler.snapshot("after_load")
        profiler.reset_peak()  # Reset peak memory after h5ad load
        profiler.start("shrinkage")
    
    # Validate that this is an NB-GLM result
    if adata.uns.get("method") != "nb_glm" or "standard_error" not in adata.layers:
        raise ValueError(
            f"Input file '{path}' is not an nb_glm_test result "
            "(expected uns['method'] == 'nb_glm' and a 'standard_error' layer)."
        )

    # Get required metadata
    control_label = adata.uns.get("control_label", "control")
    perturbation_column = adata.uns.get("perturbation_column", "perturbation")

    # MLE LFC in the file's base: X, unless X was already shrunk, in which
    # case nb_glm_test / shrink_lfc kept the MLE as logfoldchange_raw.
    lfc_base = adata.uns.get("lfc_base", "log2")
    mle_lfc = adata.layers["logfoldchange_raw"] if "logfoldchange_raw" in adata.layers else adata.X
    # ln-scale MLE LFC and SE (apeGLM works on the ln scale). With a log2
    # base they are stored as *_ln layers; with an ln base the main layers
    # already hold them.
    if lfc_base == "ln":
        raw_lfc = mle_lfc
        se = adata.layers.get("standard_error_ln", adata.layers["standard_error"])
    else:
        raw_lfc = adata.layers["logfoldchange_raw_ln"]
        se = adata.layers["standard_error_ln"]
    raw_lfc = np.asarray(raw_lfc)
    se = np.asarray(se)
    # A global dispersion is stored once per gene in var; a per-comparison
    # one is a layer.
    if "dispersion" in adata.var:
        dispersion = np.broadcast_to(adata.var["dispersion"].to_numpy(dtype=np.float64), raw_lfc.shape)
    else:
        dispersion = adata.layers["dispersion"]
    
    # Get fitted intercept from NB-GLM (ln-scale, critical for accurate shrinkage)
    if "intercept" not in adata.layers:
        raise ValueError(
            f"Input file '{path}' lacks 'intercept' layer. "
            "method='full' requires NB-GLM results from nb_glm_test v0.5.1+. "
            "Use method='stats' or re-run nb_glm_test."
        )
    fitted_intercept = adata.layers["intercept"]  # Already ln-scale
    logger.info("Using fitted intercept from NB-GLM for shrinkage")
    
    n_groups, n_genes = raw_lfc.shape
    candidates = list(adata.obs_names.astype(str))
    gene_symbols = pd.Index(adata.var_names).astype(str)
    
    # Estimate global prior scale from ALL perturbations' MLE LFCs
    all_mle_lfc = raw_lfc.ravel()
    all_mle_se = se.ravel()
    valid_mask = np.isfinite(all_mle_lfc) & np.isfinite(all_mle_se) & (all_mle_se > 0)
    global_prior_scale = _estimate_apeglm_prior_scale(
        all_mle_lfc[valid_mask], 
        all_mle_se[valid_mask]
    )
    logger.info(f"Global prior scale for apeGLM: {global_prior_scale:.4f}")
    
    # Initialize output arrays
    shrunk_lfc = np.zeros_like(raw_lfc)
    shrunk_se = np.zeros_like(se)
    total_converged = 0
    total_genes_processed = 0
    
    if method == "stats":
        # Fast stats-based shrinkage using vectorized Newton-Raphson
        # No need to load original dataset - uses pre-computed MLE stats
        logger.info(f"Using fast stats-based shrinkage (method='stats')")
        
        # Try to derive base_mean from fitted intercept for gene-specific priors
        # intercept is ln(mu_control), so base_mean ≈ mean(exp(intercept)) across perturbations
        # This is a proxy for expression level used in gene-specific prior scaling
        if np.any(np.isfinite(fitted_intercept)):
            # Use mean intercept across perturbations for each gene
            # Suppress "Mean of empty slice" RuntimeWarning when all values are NaN for a gene
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                mean_intercept = np.nanmean(fitted_intercept, axis=0)  # Shape: (n_genes,)
            base_mean_proxy = np.exp(np.clip(mean_intercept, -20, 20))  # Clamp to avoid overflow
            logger.debug("Using intercept-derived base_mean for gene-specific priors")
        else:
            base_mean_proxy = None
            logger.debug("base_mean not available, using uniform prior scale")
        
        # Track genes that need full re-fitting
        all_needs_refit = np.zeros((n_groups, n_genes), dtype=bool)
        
        for group_idx, pert_label in enumerate(candidates):
            logger.debug(f"Shrinking LFC for perturbation {group_idx + 1}/{n_groups}: {pert_label}")
            
            mle_lfc_group = raw_lfc[group_idx]
            se_group = se[group_idx]
            
            # Determine prior scale based on mode
            if prior_scale_mode == "per_comparison":
                pert_prior_scale = _estimate_apeglm_prior_scale(mle_lfc_group, se_group)
            else:
                pert_prior_scale = global_prior_scale
            
            # Vectorized shrinkage using Newton-Raphson
            # NOTE: use_gene_specific_prior=False for PyDESeq2 parity (gene-specific
            # prior scaling is non-standard and causes accuracy regression)
            shrunk_lfc_group, shrunk_se_group, converged, needs_refit = shrink_lfc_apeglm_from_stats(
                mle_lfc=mle_lfc_group,
                mle_se=se_group,
                prior_scale=pert_prior_scale,
                base_mean=base_mean_proxy,
                use_gene_specific_prior=False,
                hybrid_fallback=True,
            )
            
            shrunk_lfc[group_idx] = shrunk_lfc_group
            shrunk_se[group_idx] = shrunk_se_group
            all_needs_refit[group_idx] = needs_refit
            
            n_converged = converged.sum()
            total_converged += n_converged
            total_genes_processed += n_genes
            logger.debug(f"  {n_converged}/{n_genes} genes converged for {pert_label}")
        
        # Log warning if many genes didn't converge
        convergence_rate = total_converged / total_genes_processed if total_genes_processed > 0 else 1.0
        if convergence_rate < 0.95:
            logger.warning(
                f"Low convergence rate: {total_converged}/{total_genes_processed} "
                f"({convergence_rate:.1%}) genes converged within max_iter. "
                "Consider using method='full' for better accuracy."
            )
        
        # Log info about genes that might need full re-fitting
        total_needs_refit = all_needs_refit.sum()
        if total_needs_refit > 0:
            refit_rate = total_needs_refit / (n_groups * n_genes)
            logger.info(
                f"Hybrid fallback: {total_needs_refit} gene-perturbation pairs "
                f"({refit_rate:.1%}) flagged for potential accuracy improvement with method='full'"
            )
    
    else:  # method == "full"
        # Full model re-fitting with L-BFGS-B per gene
        logger.info(f"Using full model re-fitting (method='full')")
        
        # Validate original dataset path for full method
        original_dataset_path = adata.uns.get("original_dataset_path")
        if original_dataset_path is None:
            raise ValueError(
                f"Input file '{path}' does not have 'original_dataset_path' in uns. "
                "method='full' requires NB-GLM results from nb_glm_test v0.4.0+. "
                "Use method='stats' or re-run nb_glm_test."
            )
        
        # Strip /workspace/ prefix from Docker paths if present
        if original_dataset_path.startswith("/workspace/"):
            original_dataset_path = original_dataset_path[len("/workspace/"):]
        
        original_path = Path(original_dataset_path)
        if not original_path.exists():
            # Also try the input file's parent as base directory
            input_parent = path.parent
            relative_name = Path(original_dataset_path).name
            alternative_paths = [
                input_parent.parent / ".cache" / relative_name,  # Try ../..cache/
                input_parent / ".cache" / relative_name,  # Try ../.cache/
                input_parent.parent.parent / ".cache" / relative_name,  # Try ../../../.cache/
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    original_path = alt_path
                    break
            else:
                raise FileNotFoundError(
                    f"Original dataset not found: {original_dataset_path}. "
                    "method='full' requires access to the original count data. "
                    "Use method='stats' to shrink without original data."
                )
        
        # Load original dataset for streaming
        backed = read_backed(original_path)
        try:
            labels = backed.obs[perturbation_column].astype(str).to_numpy()
            
            # Get or compute size factors
            size_factors_global = adata.uns.get("size_factors")
            if size_factors_global is not None:
                size_factors_all = np.asarray(size_factors_global, dtype=np.float64)
            else:
                size_factors_all = _median_of_ratios_size_factors(original_path)
            
            # Control cell indices
            control_mask = labels == control_label
            control_idx = np.where(control_mask)[0]
            
            if len(control_idx) == 0:
                raise ValueError(f"No control cells found with label '{control_label}'")
            
            # Load control cells once (they're reused for all perturbations)
            control_counts = backed.X[control_idx, :].toarray() if sp.issparse(backed.X[control_idx, :]) else np.asarray(backed.X[control_idx, :])
            control_size_factors = size_factors_all[control_idx]
            
            # Process each perturbation
            for group_idx, pert_label in enumerate(candidates):
                logger.debug(f"Shrinking LFC for perturbation {group_idx + 1}/{n_groups}: {pert_label}")
                
                # Get perturbation cell indices
                pert_mask = labels == pert_label
                pert_idx = np.where(pert_mask)[0]
                
                if len(pert_idx) == 0:
                    shrunk_lfc[group_idx] = raw_lfc[group_idx]
                    shrunk_se[group_idx] = se[group_idx]
                    continue
                
                # Load perturbation cells
                pert_counts = backed.X[pert_idx, :].toarray() if sp.issparse(backed.X[pert_idx, :]) else np.asarray(backed.X[pert_idx, :])
                pert_size_factors = size_factors_all[pert_idx]
                
                # Combine control and perturbation
                combined_counts = np.vstack([control_counts, pert_counts])
                combined_size_factors = np.concatenate([control_size_factors, pert_size_factors])
                
                # Build design matrix
                n_control = len(control_idx)
                n_pert = len(pert_idx)
                n_combined = n_control + n_pert
                
                design_matrix = np.zeros((n_combined, 2), dtype=np.float64)
                design_matrix[:, 0] = 1.0
                design_matrix[n_control:, 1] = 1.0
                
                # Get MLE coefficients from NB-GLM
                mle_intercept = fitted_intercept[group_idx]
                mle_lfc_group = raw_lfc[group_idx]
                mle_coef = np.vstack([mle_intercept, mle_lfc_group])
                
                disp_group = dispersion[group_idx]
                se_group = se[group_idx]
                
                # Determine prior scale based on mode
                if prior_scale_mode == "per_comparison":
                    pert_prior_scale = _estimate_apeglm_prior_scale(mle_lfc_group, se_group)
                else:
                    pert_prior_scale = global_prior_scale
                
                # Full apeGLM shrinkage with L-BFGS-B per gene
                # NOTE: By default (min_mu=0.0), no min_mu clamping is applied,
                # matching PyDESeq2's lfc_shrink which omits min_mu entirely.
                # PyDESeq2 uses min_mu=0.5 for NB-GLM fitting but does NOT pass
                # min_mu to the shrinkage optimizer.
                shrunk_coef, shrunk_se_group, converged = shrink_lfc_apeglm(
                    counts=combined_counts,
                    design_matrix=design_matrix,
                    size_factors=combined_size_factors,
                    dispersion=disp_group,
                    mle_coef=mle_coef,
                    mle_se=se_group,
                    shrink_index=1,
                    prior_scale=pert_prior_scale,
                    n_jobs=n_jobs,
                    batch_size=batch_size,
                    min_mu=min_mu,
                )
                
                shrunk_lfc[group_idx] = shrunk_coef[1, :]
                shrunk_se[group_idx] = shrunk_se_group
                
                n_converged = converged.sum()
                total_converged += n_converged
                total_genes_processed += n_genes
                logger.debug(f"  {n_converged}/{n_genes} genes converged for {pert_label}")
            
        finally:
            backed.file.close()
    
    # Convert shrunk results from ln-scale to log2 if original output was log2
    if lfc_base == "log2":
        ln2 = np.log(2)
        shrunk_lfc = shrunk_lfc / ln2
        shrunk_se = shrunk_se / ln2
    
    # X becomes the shrunk LFC (the effect size); keep the MLE LFC and, with
    # an ln base, the MLE SE, which the posterior values below replace.
    adata.layers["logfoldchange_raw"] = np.array(mle_lfc)
    if lfc_base == "ln":
        adata.layers["standard_error_ln"] = se
    adata.layers["standard_error"] = shrunk_se  # Posterior SE
    adata.X = shrunk_lfc
    
    # Update metadata
    adata.uns["lfc_shrinkage_type"] = "apeglm"
    adata.uns["apeglm_prior_scale"] = global_prior_scale
    adata.uns["shrinkage_method"] = method
    adata.uns["prior_scale_mode"] = prior_scale_mode
    
    # Store profiling results or "NA" for production
    if profiling and profiler is not None:
        profiler.stop("shrinkage")
        profiler.snapshot("shrinkage_end")
        profiler.stop("total")
        profiler.stop_sampling()
        stats = profiler.get_stats()
        adata.uns["profiling"] = {
            "profiling_enabled": True,
            "shrinkage_seconds": stats.get("timing", {}).get("sections", {}).get("shrinkage", {}).get("seconds", 0.0),
            "shrinkage_peak_memory_mb": stats.get("memory", {}).get("peak_mb", 0.0),
            "total_seconds": stats.get("timing", {}).get("total_seconds", 0.0),
        }
    else:
        adata.uns["profiling"] = "NA"
    
    # Determine output path
    if data_name is None:
        # Append _shrunk to the stem
        stem = path.stem
        if stem.endswith("_shrunk"):
            # Already shrunk, use as-is
            data_name = stem
        else:
            data_name = f"{stem}_shrunk"

    if output_dir is None:
        output_dir = path.parent
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / f"{data_name}.h5ad"
    _shrink_lfc_disk_estimate = warn_if_disk_space_low(
        estimate_bytes(n_groups, n_genes, len(adata.layers) + 1, overhead=1.10),
        output_path,
        context="shrink_lfc",
    )
    _messages.print_disk_estimate(verbose, "shrink_lfc", _shrink_lfc_disk_estimate)
    _messages.print_saving(verbose, "shrink_lfc", output_path)
    adata.write(output_path)
    _messages.print_done(verbose, "shrink_lfc", f"{n_groups} perturbations × {n_genes} genes (method={method!r})")
    
    # Build result object
    corr_method = adata.uns.get("pvalue_correction", "benjamini-hochberg")
    
    # Get matrices from layers
    statistic_matrix = adata.layers.get("z_score", np.zeros_like(shrunk_lfc))
    pvalue_matrix = adata.layers.get("pvalue", np.ones_like(shrunk_lfc))
    pvalue_adj_matrix = adata.layers.get("pvalue_adj", np.ones_like(shrunk_lfc))
    pts_matrix = adata.layers.get("pts", np.zeros((n_groups, n_genes), dtype=np.float32))
    pts_rest_matrix = np.broadcast_to(adata.var["pts_rest"].to_numpy(dtype=np.float32), (n_groups, n_genes))
    
    # Create order matrix
    statistic_for_order = np.where(
        np.isfinite(statistic_matrix), np.abs(statistic_matrix), -np.inf
    )
    order_matrix = np.argsort(-statistic_for_order, axis=1, kind="mergesort")
    
    result = RankGenesGroupsResult(
        genes=gene_symbols,
        groups=candidates,
        statistics=np.asarray(statistic_matrix),
        pvalues=np.asarray(pvalue_matrix),
        pvalues_adj=np.asarray(pvalue_adj_matrix),
        logfoldchanges=shrunk_lfc,
        effect_size=shrunk_lfc,
        u_statistics=np.zeros_like(shrunk_lfc),
        pts=np.asarray(pts_matrix, dtype=np.float32),
        pts_rest=np.asarray(pts_rest_matrix, dtype=np.float32),
        order=order_matrix,
        groupby=perturbation_column,
        method="nb_glm",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
    )
    result.result = AnnData(output_path)
    
    logger.info(
        f"Applied apeGLM LFC shrinkage (method={method}, prior_scale_mode={prior_scale_mode}) "
        f"to {n_groups} perturbations, {n_genes} genes (prior_scale={global_prior_scale:.4f}). "
        f"Output: {output_path}"
    )

    return result


# ---------------------------------------------------------------------------
# Disk-usage resolvers for crispyx.estimate_disk_usage() (see _preflight.py).
#
# Each resolver reproduces the cheap, obs-only preamble the real function
# already runs before allocating its intermediate arrays, then applies the
# exact same per-array dtype counts used at the real warn_if_disk_space_low
# call sites above, so the estimate and the automatic in-call warning can
# never disagree.
# ---------------------------------------------------------------------------


def _estimate_shape_for_t_test(
    path: Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    perturbations: Iterable[str] | None = None,
    **_ignored,
) -> dict[str, float]:
    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(list(labels), control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        n_genes = backed.n_vars
    finally:
        backed.file.close()
    n_groups = len(candidates)
    return {"output": _t_test_disk_bytes(n_groups, n_genes)}


def _estimate_shape_for_nb_glm_test(
    path: Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    perturbations: Iterable[str] | None = None,
    **_ignored,
) -> dict[str, float]:
    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(list(labels), control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        n_genes = backed.n_vars
    finally:
        backed.file.close()
    n_groups = len(candidates)
    return {"output": _nb_glm_disk_bytes(n_groups, n_genes)}


def _estimate_shape_for_wilcoxon_test(
    path: Path,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    format_mismatch_policy: str = "auto",
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    **_ignored,
) -> dict[str, float]:
    validate_format_mismatch_policy(format_mismatch_policy)
    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        n_genes = backed.n_vars
        if chunk_size is None:
            # Same resolution the real call makes, so the scratch decision
            # below sees the chunk count the run will actually stream.
            chunk_size = calculate_wilcoxon_chunk_size(
                backed.n_obs, backed.n_vars, available_memory_gb=memory_limit_gb,
            )
    finally:
        backed.file.close()
    n_groups = len(candidates)
    estimate = {"output": _wilcoxon_disk_bytes(n_groups, n_genes)}
    # A CSR source may first be converted to a temporary CSC copy beside the
    # output; scratch_copy_bytes makes that call the same way the run does.
    scratch = scratch_copy_bytes(
        path, axis=1, policy=format_mismatch_policy, chunk_size=chunk_size
    )
    if scratch is not None:
        estimate["scratch"] = scratch
    return estimate


def _estimate_shape_for_shrink_lfc(path: Path, **_ignored) -> dict[str, float]:
    backed = read_backed(path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        n_layers = len(backed.layers) + 1
    finally:
        backed.file.close()
    return {"output": estimate_bytes(n_obs, n_vars, n_layers, overhead=1.10)}
