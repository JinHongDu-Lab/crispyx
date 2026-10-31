"""Differential expression testing utilities."""

from __future__ import annotations

import gc
import dataclasses
import logging
import os
import shutil
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
from anndata.io import read_elem, write_elem
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import norm, rankdata, t as t_dist

from .data import (
    accepts_in_memory,
    AnnData,
    calculate_optimal_chunk_size,
    calculate_optimal_gene_chunk_size,
    calculate_wilcoxon_chunk_size,
    ensure_gene_symbol_column,
    get_matrix_storage_format,
    get_perturbation_slice,
    iter_matrix_chunks,
    needs_sorting_for_nbglm,
    read_backed,
    read_rows_csc,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    scratch_copy_bytes,
    sort_by_perturbation,
    perturbation_sort_order,
    stream_on_fast_axis,
    validate_format_mismatch_policy,
    _read_h5_1d,
    write_obs,
    _replace_on_success,
    _validate_counts_chunk,
)
from .glm import (
    build_design_matrix,
    check_full_rank,
    dense_columns,
    fit_glm_deseq2,
    fit_two_group_deseq2,
    summarize_controls,
    shrink_lfc_apeglm,
    shrink_lfc_apeglm_from_stats,
    cauchy_map,
    fit_dispersion_trend,
    _estimate_apeglm_prior_scale,
    _apeglm_prior_scale_from_blocks,
)
from ._kernels import (
    _presort_control_nonzeros,
    _compute_ctrl_tie_sums,
    _wilcoxon_batch_perts_presorted_numba,
    _wilcoxon_stratified_batch_perts_numba,
    _ZERO_PARTITION_THRESHOLD,
)
from ._checkpoint import (
    ResumableRun,
    _chunk_rows,
    _column_block_chunks,
    _create_unfilled_array,
    _get_checkpoint_interval,
    _create_progress_context,
    _DummyProgress,
    _row_blocks,
)
from . import _messages
from ._provenance import KEY as _PROVENANCE_KEY, SCHEMAS, call_fingerprint, reusable, stamp, write_stamp
from ._disk import estimate_bytes, warn_if_disk_space_low
from ._grouping import UNLABELLED, canonical_call_args, cell_labels, label_strings, resolve_group_reference_aliases
from ._memory import _resolve_memory_limit_bytes, _resolve_n_jobs, drop_file_cache
from ._irls import ETA_MAX, ETA_MIN
from ._size_factors import (
    _validate_size_factors,
    _library_size_factors,
    _control_geomeans_size_factors,
    _control_reference_size_factors,
)
from ._statistics import (
    _tie_correction,
    _adjust_pvalue_matrix,
    _low_expr_in_both_mask,
    _nonestimable_glm_mask,
)
from ._memory import (
    _estimate_max_workers,
)

logger = logging.getLogger(__name__)


#: Where each field of a DE result lives in its h5ad, by method. ``X`` holds
#: each method's own effect size; ``None`` marks a field the method does not
#: have.
_RESULT_KEYS: dict[str, dict[str, str | None]] = {
    method: {
        "statistics": "layers/z_score",
        "pvalues": "layers/pvalue",
        "pvalues_adj": "layers/pvalue_adj",
        "logfoldchanges": "X" if method in ("nb_glm", "nb_offset") else "layers/logfoldchanges",
        "pts": "layers/pts",
        "u_statistics": "layers/u_statistic" if method == "wilcoxon" else None,
        "auc": "X" if method == "wilcoxon" else None,
        "mean_difference": "X" if method == "t_test" else None,
    }
    for method in ("t_test", "nb_glm", "nb_offset", "wilcoxon")
}


def _is_row_key(key) -> bool:
    """Whether ``key`` is an index h5py reads directly: integers and slices
    (with a step of 1), alone or as a tuple."""
    parts = key if isinstance(key, tuple) else (key,)
    return all(
        isinstance(part, (int, np.integer))
        or (isinstance(part, slice) and part.step in (None, 1))
        for part in parts
    )


#: A result matrix this small a share of the memory budget is read into
#: memory on first use and indexed there (as fast as the arrays results used
#: to hold); a larger one is read from the file a block of rows at a time.
_IN_MEMORY_LAYER_FRACTION = 0.01


class _ResultLayer(np.lib.mixins.NDArrayOperatorsMixin):
    """One ``(n_groups, n_genes)`` matrix of a DE result file.

    A matrix within :data:`_IN_MEMORY_LAYER_FRACTION` of the memory budget
    -- any result on a personal machine, short of a genome-wide screen --
    is read whole on first use and kept, re-read if the file is replaced.
    A larger one is read on access: indexing by rows or slices reads only
    what it selects, so ``result.pvalues[i]`` costs one row however large
    the result is; any other use (``np.asarray``, arithmetic, fancy
    indexing) reads the whole matrix, as an explicit request for all of it.
    """

    def __init__(self, path: Path, key: str) -> None:
        self.path = Path(path)
        self.key = key
        with h5py.File(self.path, "r") as handle:
            self.shape = tuple(handle[key].shape)
            self.dtype = handle[key].dtype
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self._in_memory = nbytes <= _IN_MEMORY_LAYER_FRACTION * _resolve_memory_limit_bytes(None)
        self._values: np.ndarray | None = None
        self._identity: tuple | None = None

    ndim = property(lambda self: len(self.shape))
    size = property(lambda self: int(np.prod(self.shape)))

    def __len__(self) -> int:
        return self.shape[0]

    def _loaded(self) -> np.ndarray | None:
        """The whole matrix, when it is small enough to keep: read once and
        again only if the file has been replaced or rewritten since."""
        if not self._in_memory:
            return None
        stat = os.stat(self.path)
        identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if self._values is None or identity != self._identity:
            with h5py.File(self.path, "r") as handle:
                self._values = handle[self.key][()]
            self._identity = identity
        return self._values

    def __getitem__(self, key) -> np.ndarray:
        values = self._loaded()
        if values is not None:
            return values[key].copy() if isinstance(values[key], np.ndarray) else values[key]
        with h5py.File(self.path, "r") as handle:
            dataset = handle[self.key]
            if _is_row_key(key):
                return dataset[key]
            return dataset[()][key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._loaded()
        if values is not None:
            values = values.copy()
        else:
            with h5py.File(self.path, "r") as handle:
                values = handle[self.key][()]
        return values if dtype is None else values.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(np.asarray(x) if isinstance(x, _ResultLayer) else x for x in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getattr__(self, name: str):
        # Any other ndarray attribute (``.ravel()``, ``.mean()``, ``.T``)
        # works on the whole matrix, read for it.
        if name.startswith("_") or name in ("path", "key", "shape", "dtype"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.key} {self.shape} {self.dtype} of {self.path}>"


def _rank_order(statistics: np.ndarray) -> np.ndarray:
    """Each row's gene positions by descending ``|statistic|``, untested
    (non-finite) genes last, ties in gene order."""
    magnitude = np.where(np.isfinite(statistics), np.abs(statistics), -np.inf)
    return np.argsort(-magnitude, axis=-1, kind="mergesort")


class _RankOrder(_ResultLayer):
    """Scanpy's ``order``: :func:`_rank_order` of the statistics, computed for
    the rows asked for rather than stored."""

    def __init__(self, statistics: _ResultLayer) -> None:
        self.path, self.key = statistics.path, statistics.key
        self.shape, self.dtype = statistics.shape, np.dtype(np.intp)
        self._statistics = statistics

    def __getitem__(self, key) -> np.ndarray:
        rows, columns = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if not _is_row_key(rows):
            return np.asarray(self)[key]
        order = _rank_order(self._statistics[rows])
        return order[(Ellipsis, *columns)] if columns else order

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        n_rows, n_genes = self.shape
        out = np.empty(self.shape, dtype=np.intp)
        with h5py.File(self.path, "r") as handle:
            chunk_rows = _chunk_rows(handle[self.key])
        for sl in _row_blocks(n_rows, n_genes, chunk_rows):
            out[sl] = self[sl]
        return out if dtype is None else out.astype(dtype, copy=False)


@dataclass
class DifferentialExpressionResult:
    """One perturbation's per-gene results; see :class:`RankGenesGroupsResult`
    for what each field holds."""

    genes: pd.Index
    logfoldchanges: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    method: str
    perturbation: str
    pvalue_adj: np.ndarray | None = None
    auc: np.ndarray | None = None
    mean_difference: np.ndarray | None = None
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


class RankGenesGroupsResult(Mapping[str, DifferentialExpressionResult]):
    """Per-perturbation x per-gene DE results, backed by the result file.

    Rows follow ``groups``. The matrices are read from the file on access
    (see :class:`_ResultLayer`): ``result.pvalues[i]`` and ``result[label]``
    read one perturbation's row, and ``np.asarray(result.pvalues)`` reads
    the whole matrix. ``order`` is Scanpy's ranking by ``|statistics|``,
    computed for the rows read.

    ``logfoldchanges`` is the log2 fold change of mean expression for every
    method: for ``wilcoxon_test`` and ``t_test``,
    ``log2((expm1(mean_pert) + 1e-9) / (expm1(mean_ctrl) + 1e-9))`` on the
    log1p input (Scanpy's convention); for ``nb_glm_test``, the fitted
    perturbation coefficient on the log2 scale (``lfc_base="ln"``: natural
    log). The method-specific effect sizes are ``auc`` (``wilcoxon_test``:
    ``U / (n_pert * n_ctrl)``, the probability that a perturbed cell
    exceeds a control cell, 0.5 for no shift), ``u_statistics``
    (``wilcoxon_test``: the Mann-Whitney U) and ``mean_difference``
    (``t_test``: the difference of means on the input's own scale); each is
    ``None`` for the other methods.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            self.groups = [str(g) for g in _h5_index(handle["obs"])]
            self.genes = pd.Index([str(g) for g in _h5_index(handle["var"])])
            uns = handle["uns"]
            self.method = _uns_value(uns, "method", "wilcoxon")
            self.control_label = _uns_value(uns, "control_label")
            self.groupby = _uns_value(uns, "perturbation_column")
            self.pvalue_correction = _uns_value(uns, "pvalue_correction", "benjamini-hochberg")
            self.tie_correct = bool(_uns_value(uns, "tie_correct", self.method == "wilcoxon"))
            pts_rest = np.asarray(handle["var/pts_rest"][()], dtype=np.float32)
        for name, key in _RESULT_KEYS[self.method].items():
            setattr(self, name, None if key is None else _ResultLayer(self.path, key))
        self.pts_rest = np.broadcast_to(pts_rest, (len(self.groups), len(self.genes)))
        self.order = _RankOrder(self.statistics)
        self.result: AnnData | None = AnnData(self.path)

    @property
    def result_path(self) -> Path:
        return self.path

    def __getstate__(self) -> dict:
        """Exclude the AnnData file handle from the pickle payload."""
        return {k: v for k, v in self.__dict__.items() if k != "result"}

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle; ``result`` is ``None`` (no open file handle)."""
        self.__dict__.update(state)
        self.result = None

    def __getitem__(self, key: str) -> DifferentialExpressionResult:
        """Return per-gene DE results for a single perturbation group, read
        from the file.

        Parameters
        ----------
        key : str
            Perturbation group name.

        Returns
        -------
        DifferentialExpressionResult
            Results for the requested group.
        """
        try:
            row = self.groups.index(str(key))
        except ValueError:
            raise KeyError(key) from None
        return DifferentialExpressionResult(
            genes=self.genes,
            logfoldchanges=self.logfoldchanges[row],
            statistic=self.statistics[row],
            pvalue=self.pvalues[row],
            method=self.method,
            perturbation=self.groups[row],
            pvalue_adj=self.pvalues_adj[row],
            auc=None if self.auc is None else self.auc[row],
            mean_difference=None if self.mean_difference is None else self.mean_difference[row],
            result=self.result,
        )

    def __iter__(self):  # type: ignore[override]
        """Iterate over perturbation group names."""
        return iter(self.groups)

    def __len__(self) -> int:
        """Return the number of perturbation groups."""
        return len(self.groups)

    def to_rank_genes_groups_dict(self) -> dict:
        """Convert results to Scanpy-compatible ``rank_genes_groups`` dict.

        Reads every matrix into memory.

        Returns
        -------
        dict
            Dictionary with keys ``'params'``, ``'names'``, ``'scores'``,
            ``'logfoldchanges'``, ``'pvals'``, ``'pvals_adj'``, ``'pts'``,
            ``'pts_rest'``, ``'full'``, and ``'u_stat'`` and ``'auc'`` when
            the method has them (``wilcoxon_test``).
            Each value (except ``'params'`` and ``'full'``) is a NumPy
            record array sorted by descending absolute score.
        """
        full = self.to_full_order_dict()
        order = np.asarray(self.order)

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
            "names": to_recarray(self.genes.to_numpy()[order].astype(object)),
        }
        for name, values in full.items():
            rank_genes_groups[name] = to_recarray(np.take_along_axis(values, order, axis=1))
        rank_genes_groups["full"] = full
        return rank_genes_groups

    def to_full_order_dict(self) -> dict:
        """Return unsorted matrices keyed by statistic name.

        Unlike :meth:`to_rank_genes_groups_dict`, values are in the
        original gene order (not sorted by score). Reads every matrix into
        memory.

        Returns
        -------
        dict
            Keys: ``'scores'``, ``'pvals'``, ``'pvals_adj'``,
            ``'logfoldchanges'``, ``'pts'``, ``'pts_rest'``, and ``'u_stat'``
            and ``'auc'`` when the method has them.  Each value is a 2-D
            ``(n_groups, n_genes)`` NumPy array.
        """
        full = {
            "scores": np.asarray(self.statistics),
            "pvals": np.asarray(self.pvalues),
            "pvals_adj": np.asarray(self.pvalues_adj),
            "logfoldchanges": np.asarray(self.logfoldchanges),
            "pts": np.asarray(self.pts),
            "pts_rest": np.array(self.pts_rest),
        }
        if self.u_statistics is not None:
            full["u_stat"] = np.asarray(self.u_statistics)
        if self.auc is not None:
            full["auc"] = np.asarray(self.auc)
        return full


def _h5_index(group: h5py.Group) -> np.ndarray:
    """The index of an anndata dataframe group (``obs`` / ``var``)."""
    key = group.attrs.get("_index", "_index")
    if isinstance(key, (bytes, np.bytes_)):
        key = key.decode()
    return np.array([v.decode() if isinstance(v, bytes) else v for v in _read_h5_1d(group[key])])


def _uns_value(uns: h5py.Group, key: str, default=None):
    """A scalar ``uns`` entry, as anndata writes it (a scalar dataset) or as
    an attribute."""
    if key in uns.attrs:
        value = uns.attrs[key]
    elif key in uns:
        value = uns[key][()]
    else:
        return default
    if hasattr(value, "item"):
        value = value.item()
    return value.decode() if isinstance(value, bytes) else value


_SIZE_FACTOR_METHODS = ("libsize", "geomeans", "deseq2")


def _global_size_factors(
    path: Path, method: str, chunk_size: int, labels: np.ndarray, control_label: str
) -> np.ndarray:
    """Every cell's size factor by ``method`` (see ``nb_glm_test``), at geometric mean 1."""
    if method == "libsize":
        return _library_size_factors(path, chunk_size=chunk_size)
    if method == "geomeans":
        return _control_geomeans_size_factors(path, labels, control_label, chunk_size=chunk_size)
    if method == "deseq2":
        return _control_reference_size_factors(path, labels, control_label, chunk_size=chunk_size)
    raise ValueError(f"size_factor_method must be 'libsize', 'geomeans' or 'deseq2', not {method!r}")


def _nb_glm_testable_genes(
    *,
    pert_expr_counts: np.ndarray,
    pert_total: np.ndarray,
    n_pert: int,
    control_expr_counts: np.ndarray,
    control_total: np.ndarray,
    n_control: int,
    min_cells_expressed: int,
    min_pct_ctrl: float,
    min_pct_pert: float,
    min_mean_ctrl: float,
    min_mean_pert: float,
    min_cells_ctrl: int,
    min_cells_pert: int,
    min_total_count: float,
) -> np.ndarray:
    """The genes ``nb_glm_test`` tests for one perturbation against the
    controls, after its expression filters. A pair with no counts in one arm
    has no finite effect to estimate (see _nonestimable_glm_mask): it is
    excluded rather than fitted, so it is reported as untested instead of
    carrying a bound-determined effect and an artefactual p-value into the
    FDR correction."""
    valid = (control_expr_counts + pert_expr_counts) >= min_cells_expressed
    valid &= ~_low_expr_in_both_mask(
        pert_expr_counts=pert_expr_counts,
        control_expr_counts=control_expr_counts,
        pert_mean=pert_total / max(n_pert, 1),
        control_mean=control_total / max(n_control, 1),
        n_pert_cells=n_pert,
        n_control_cells=n_control,
        min_pct_ctrl=min_pct_ctrl,
        min_pct_pert=min_pct_pert,
        min_mean_ctrl=min_mean_ctrl,
        min_mean_pert=min_mean_pert,
    )
    valid &= ~_nonestimable_glm_mask(
        pert_expr_counts=pert_expr_counts,
        control_expr_counts=control_expr_counts,
        min_cells_ctrl=min_cells_ctrl,
        min_cells_pert=min_cells_pert,
    )
    valid &= (control_total + pert_total) >= min_total_count
    return valid


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
        Explicit subset to test.  ``None`` means test all non-control groups;
        an empty selection tests nothing, and the caller writes an empty
        result, so a caller that splits perturbations between estimators
        needs no guard.

    Returns
    -------
    list of str
        Candidate perturbation group names.

    Raises
    ------
    ValueError
        If the file holds no non-control group to test.
    """
    if perturbations is None:
        unique = pd.Index(labels).unique().tolist()
        candidates = [label for label in unique if label not in (control_label, UNLABELLED)]
        if not candidates:
            raise ValueError("No perturbation groups available for differential expression testing")
        return candidates
    return [str(p) for p in perturbations if str(p) not in (control_label, UNLABELLED)]


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


#: Arguments that change how a DE call runs but not what it writes; a
#: finished result or a checkpoint from a call differing only in these is
#: still reused. The memory and chunking knobs belong here because rerunning
#: with a lower ``memory_limit_gb`` is the natural response to an OOM kill.
#: ``data`` is identified by the fingerprint's source path, size and mtime.
_OPERATIONAL_ARGS = frozenset({
    "data", "verbose", "resume", "checkpoint_interval", "n_jobs",
    "profiling", "force", "output_path", "output_dir", "data_name",
    "format_mismatch_policy", "memory_limit_gb",
    "cell_chunk_size", "chunk_size",
})


def _de_fingerprint(
    path: Path, call_args: dict, method: str, *, perturbation_column: str, control_label: str | None,
) -> dict:
    """Identity of a DE call, stamped into its result: see
    :func:`~crispyx._provenance.call_fingerprint`. The grouping column and
    control are recorded as resolved from their aliases."""
    call_args = canonical_call_args(
        call_args, perturbation_column=perturbation_column, control_label=control_label,
    )
    return call_fingerprint(path, call_args, _OPERATIONAL_ARGS, method=method)


def _checkpoint_fingerprint(fingerprint: dict, **extra) -> dict:
    """A checkpoint's identity: the result ``fingerprint`` without
    ``scanpy_format``, which only changes the final write, plus the result
    schema the saved values are in (a run interrupted by an older layout is
    restarted, not finished in the new one) and ``extra``, what the saved
    progress also depends on (e.g. the gene chunk width it is counted in)."""
    params = {k: v for k, v in fingerprint["params"].items() if k != "scanpy_format"}
    return {**fingerprint, "params": params, "schema": SCHEMAS["de_result"], **extra}


# Disk a DE run needs beside its output: the resumable partial arrays plus
# the final h5ad, which coexist until the run finishes. One function per
# method, shared by the run's own low-disk warning and estimate_disk_usage.

def _t_test_disk_bytes(n_groups: int, n_genes: int) -> float:
    # partial: statistics/pvalues/pvalues_adj/order (8 B) + lfc/effect/pts (4 B)
    # output: z/pvalue/pvalue_adj (8 B) + X/logfoldchanges/pts (4 B)
    return estimate_bytes(n_groups, n_genes, itemsize=8) * 7 + estimate_bytes(n_groups, n_genes, itemsize=4) * 6


def _nb_glm_disk_bytes(n_groups: int, n_genes: int) -> float:
    # partial and output alike (upper bound): X, z, p, padj, intercept, se,
    # 2 ln-scale, a raw LFC and 3 dispersion layers (8 B) + pts (4 B) +
    # converged (1 B)
    return (
        estimate_bytes(n_groups, n_genes, itemsize=8) * 24
        + estimate_bytes(n_groups, n_genes, itemsize=4) * 2
        + estimate_bytes(n_groups, n_genes, itemsize=1) * 2
    )


def _wilcoxon_disk_bytes(n_groups: int, n_genes: int) -> float:
    # partial: effect/u/pvalue/pvalue_adj/z/lfc (8 B) + pts (4 B); the output
    # holds the same seven matrices. The streaming path writes only the
    # output-sized file, so this bounds every path.
    return estimate_bytes(n_groups, n_genes, itemsize=8) * 12 + estimate_bytes(n_groups, n_genes, itemsize=4) * 2


class _PhaseTimer:
    """Wall time spent in each phase of a loop, for a one-line summary:
    ``lap(phase)`` charges the time since the previous lap to ``phase``."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, phase: str) -> None:
        now = time.perf_counter()
        self.totals[phase] = self.totals.get(phase, 0.0) + now - self._last
        self._last = now

    def summary(self) -> str:
        return ", ".join(f"{phase} {seconds:.1f} s" for phase, seconds in self.totals.items())


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


def _resumable_result(
    output_path: Path,
    checkpoint_path: Path,
    *,
    fingerprint: dict,
    resume: bool,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict,
    x_dtype,
    layers: dict[str, "np.typing.DTypeLike"],
    chunks: tuple[int, int] | None = None,
) -> ResumableRun:
    """The :class:`ResumableRun` of a DE result: an h5ad with ``obs``,
    ``var`` and ``uns`` and an unfilled ``X`` plus ``layers``, one row per
    ``obs`` row and one column per ``var`` row, filled in place as the run
    computes them; contiguous, or in HDF5 ``chunks``."""
    shape = (len(obs), len(var))
    datasets = {"X": (shape, x_dtype), **{f"layers/{name}": (shape, dtype) for name, dtype in layers.items()}}

    def scaffold(path: Path) -> None:
        ad.AnnData(obs=obs, var=var, uns=uns).write(path)
        with h5py.File(path, "r+") as handle:
            for key, (shape_, dtype) in datasets.items():
                _create_unfilled_array(handle, key, shape_, dtype, chunks)

    return ResumableRun(
        output_path, checkpoint_path, fingerprint=fingerprint, resume=resume,
        scaffold=scaffold, datasets=datasets,
    )


def _adjust_pvalues_in_file(handle: h5py.File, corr_method: str) -> None:
    """Fill ``layers/pvalue_adj`` from ``layers/pvalue``, a block of rows at
    a time (the correction is per row)."""
    pvalues, adjusted = handle["layers/pvalue"], handle["layers/pvalue_adj"]
    for sl in _row_blocks(*pvalues.shape, _chunk_rows(pvalues)):
        adjusted[sl] = _adjust_pvalue_matrix(pvalues[sl], corr_method)


def _finish_result(
    run: ResumableRun,
    output_path: Path,
    *,
    obs: pd.DataFrame,
    method: str,
    corr_method: str,
    scanpy_format: bool,
    fingerprint: dict,
    uns: dict | None = None,
) -> RankGenesGroupsResult:
    """Complete the run's result file and move it to ``output_path``: the
    adjusted p-values, ``obs["n_genes_tested"]``, any late ``uns`` entries,
    Scanpy's ``rank_genes_groups`` when asked for, and the provenance stamp,
    which goes last so the file only reads as reusable once it is whole."""
    _adjust_pvalues_in_file(run.file, corr_method)
    run.close()
    work = run.work_path
    with h5py.File(work, "r") as handle:
        obs = obs.assign(n_genes_tested=_count_tested(handle["layers/pvalue"]))
    write_obs(work, obs)
    if uns:
        with h5py.File(work, "r+") as handle:
            for key, value in uns.items():
                if key in handle["uns"]:
                    del handle["uns"][key]
                write_elem(handle["uns"], key, value)
    if scanpy_format:
        _write_scanpy_rank_genes_groups(work, method=method)
    write_stamp(work, stamp("de_result", fingerprint))
    run.finish(output_path)
    return RankGenesGroupsResult(output_path)


def _count_tested(pvalues) -> np.ndarray:
    """Genes tested per perturbation: the finite p-values of each row of
    ``pvalues`` (an array or h5py dataset), read a block of rows at a time so
    a large result is never held whole."""
    n_rows, n_genes = pvalues.shape
    counts = np.empty(n_rows, dtype=np.int64)
    for sl in _row_blocks(n_rows, n_genes, _chunk_rows(pvalues)):
        counts[sl] = np.isfinite(np.asarray(pvalues[sl])).sum(axis=1)
    return counts


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


def _write_scanpy_rank_genes_groups(path: Path, *, method: str) -> None:
    """Write Scanpy's ``uns/rank_genes_groups`` into the DE result at
    ``path`` from its own layers, a block of groups at a time, so it is
    written however large the result is.

    Arrays are in full matrix order (groups x genes) under
    ``uns/rank_genes_groups/full``, with Scanpy's ``order`` beside them. This
    avoids the recarray format, which hits HDF5 header size limits at ~2000+
    groups.
    """
    keys = _RESULT_KEYS[method]
    fields = {
        "scores": keys["statistics"], "pvals": keys["pvalues"], "pvals_adj": keys["pvalues_adj"],
        "logfoldchanges": keys["logfoldchanges"], "pts": keys["pts"],
        "u_stat": keys["u_statistics"], "auc": keys["auc"],
    }
    with h5py.File(path, "r+") as handle:
        groups = [str(g) for g in _h5_index(handle["obs"])]
        uns = handle["uns"]
        params = {
            "groupby": _uns_value(uns, "perturbation_column"),
            "method": method,
            "reference": _uns_value(uns, "control_label"),
            "tie_correct": bool(_uns_value(uns, "tie_correct", method == "wilcoxon")),
            "corr_method": _uns_value(uns, "pvalue_correction", "benjamini-hochberg"),
        }
        pts_rest = np.asarray(handle["var/pts_rest"][()], dtype=np.float32)
        n_groups, n_genes = handle["X"].shape
        if "rank_genes_groups" in uns:
            del uns["rank_genes_groups"]
        rgg = uns.create_group("rank_genes_groups")
        full = rgg.create_group("full")
        for sl in _row_blocks(n_groups, n_genes, _chunk_rows(handle["X"])):
            block = {name: handle[key][sl] for name, key in fields.items() if key is not None}
            block["pts_rest"] = np.broadcast_to(pts_rest, block["pts"].shape)
            block["order"] = _rank_order(block["scores"])
            for name, values in block.items():
                target = rgg if name == "order" else full
                if name not in target:
                    target.create_dataset(name, shape=(n_groups, n_genes), dtype=values.dtype)
                target[name][sl] = values
        rgg.create_dataset("names", data=np.array(groups, dtype="S"))
        params_group = rgg.create_group("params")
        for key, value in params.items():
            params_group.attrs[key] = value


# ---------------------------------------------------------------------------
# Shared helpers for public DE functions
# ---------------------------------------------------------------------------

def _try_load_existing_de_result(
    output_path: "Path",
    *,
    force: bool,
    verbose: int | bool,
    fingerprint: dict,
) -> "RankGenesGroupsResult | None":
    """Return the existing result at ``output_path`` when ``force=False``
    and it was computed by a call with this ``fingerprint``, else ``None``."""
    if force or not reusable(output_path, "de_result", fingerprint, verbose):
        return None
    return RankGenesGroupsResult(output_path)


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


@accepts_in_memory
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

    A perturbation (or control) with a single cell has no variance estimate,
    so its comparison is left untested (``NaN``), as in scanpy.

    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx AnnData / backed anndata.AnnData containing
        log-transformed expression data.
        An in-memory AnnData is streamed from a temporary copy; ``output_path``
        is then required.
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
        ``score`` / ``pvalue`` / ``logfoldchanges`` and the effect size, with
        ``pts`` / ``pts_rest`` still populated.
    min_pct_ctrl
        Minimum fraction of expressing cells for the *control* side. A gene is
        excluded only when *both* the control side *and* the perturbed side are
        jointly low. Default ``0.01``.
    min_pct_pert
        Minimum fraction of expressing cells for the *perturbed* side.
        Default ``0.002`` (lower than ctrl; induction from near-zero baseline is
        biologically valid). Set to ``0.0`` to disable the pct check on pert.
    min_mean_ctrl
        Minimum mean expression (log1p units) for the *control* side.
        Default ``0.05``. Excluded genes are written as NaN in
        ``score`` / ``pvalue`` / ``logfoldchanges`` / ``mean_difference``;
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
        Only a checkpoint from a call with the same input and
        result-affecting parameters is used; partial results live in a hidden
        ``.<output name>.resume`` directory until the output is written.
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
        exists. If False (default), an existing result is loaded instead of
        rerun when it was computed from the same input file (path, size,
        modification time) with the same result-shaping arguments, as
        recorded in its ``uns["crispyx"]``; otherwise it is recomputed.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].logfoldchanges`, `result[label].mean_difference`,
        `result[label].pvalue`, etc. The h5ad file path is available at
        `result.result_path`.

        The h5ad file holds one row per perturbation: the difference of means
        (``mean_difference``) in ``X`` and the layers ``z_score``, ``pvalue``,
        ``pvalue_adj``, ``logfoldchanges`` and ``pts``. ``pts_rest`` describes the control
        arm, so it is the per-gene column ``var["pts_rest"]``. ``obs["n_genes_tested"]``
        counts the genes each perturbation was tested on, after the filters.

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
    # A one-shot iterable would be fingerprinted by its repr and then used up.
    if perturbations is not None:
        perturbations = [str(p) for p in perturbations]
    call_args = dict(locals())  # for the resume fingerprint, before any other local

    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="t_test",
    )
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        raise ValueError("corr_method must be 'benjamini-hochberg' or 'bonferroni'")

    path = resolve_data_path(data)
    output_path = resolve_output_path(
        path, suffix="t_test", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )

    fingerprint = _de_fingerprint(
        path, call_args, method="t_test",
        perturbation_column=perturbation_column, control_label=control_label,
    )
    if (r := _try_load_existing_de_result(
        output_path, force=force, verbose=verbose,
        fingerprint=fingerprint,
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
        labels = label_strings(backed.obs[perturbation_column])
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

    shape = (n_groups, n_genes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")

    # Determine checkpoint interval
    eff_checkpoint_interval = _get_checkpoint_interval(n_groups, checkpoint_interval)
    candidate_to_idx = {label: idx for idx, label in enumerate(candidates)}

    obs = _result_obs(candidates, perturbation_column)
    # pts_rest describes the control arm, so it is one per-gene vector, not a
    # per-comparison matrix.
    var = pd.DataFrame({"pts_rest": control_pts.astype(np.float32)}, index=gene_symbols)

    _t_test_disk_estimate = warn_if_disk_space_low(
        _t_test_disk_bytes(*shape),
        output_path,
        context="t_test",
    )
    _messages.print_disk_estimate(verbose, "t_test", _t_test_disk_estimate)
    # Each perturbation's row is written into the result file as it is
    # computed, so an interrupted run resumes from its last checkpoint and
    # no step holds the whole result.
    run = _resumable_result(
        output_path,
        checkpoint_path,
        fingerprint=_checkpoint_fingerprint(fingerprint),
        resume=resume,
        obs=obs,
        var=var,
        uns={
            "method": "t_test",
            "control_label": control_label,
            "perturbation_column": perturbation_column,
            "pvalue_correction": corr_method,
        },
        x_dtype=np.float32,  # the difference of means
        layers={
            "z_score": np.float64,  # t-statistic (converges to z for large n)
            "pvalue": np.float64,
            "pvalue_adj": np.float64,
            "logfoldchanges": np.float32,
            "pts": np.float32,
        },
    )
    # Earlier failures are retried, so only completions carry over.
    completed_labels: list[str] = run.progress.get("completed", [])
    candidates_to_run = [c for c in candidates if c not in set(completed_labels)]
    if run.resumed:
        _messages.vprint(verbose, "t_test", f"Resuming: {len(completed_labels)}/{n_groups} perturbations already done")
    out_x = run.file["X"]
    out_z, out_p, out_lfc, out_pts = (run.file[f"layers/{k}"] for k in ("z_score", "pvalue", "logfoldchanges", "pts"))

    batch_size = worker_count
    effect_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
    stat_buffer = np.zeros((batch_size, n_genes), dtype=np.float64)
    pval_buffer = np.ones((batch_size, n_genes), dtype=np.float64)
    lfc_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
    pts_buffer = np.zeros((batch_size, n_genes), dtype=np.float32)
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
        # A one-cell side has no variance estimate; treating it as zero
        # leaves the other side's SE alone and makes every gene look
        # significant.  Leave such comparisons untested, as scanpy does.
        if n_cells < 2 or control_n < 2:
            valid[:] = False
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
    n_at_last_save = 0
        
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
                for local_idx, label in enumerate(batch_labels):
                    if label in completed_set:
                        continue
                    try:
                        compute_perturbation(label, local_idx)
                    except Exception as e:
                        logger.error(f"Failed perturbation {label}: {e}")
                        newly_failed.append(label)
                        # NaN, not the file's zeros: a p-value of 0 would
                        # report every gene as significant for it.
                        row = candidate_to_idx[label]
                        for out in (out_x, out_z, out_p, out_lfc):
                            out[row] = np.nan
                        out_pts[row] = 0
                        continue

                for local_idx, label in enumerate(batch_labels):
                    if label in completed_set or label in newly_failed:
                        continue
                    global_idx = candidate_to_idx[label]
                    out_x[global_idx] = effect_buffer[local_idx]
                    out_z[global_idx] = stat_buffer[local_idx]
                    out_p[global_idx] = pval_buffer[local_idx]
                    out_lfc[global_idx] = lfc_buffer[local_idx]
                    out_pts[global_idx] = pts_buffer[local_idx]

                    newly_completed.append(label)
                    n_tested_list.append(int(n_tested_per_slot[local_idx]))
                    _print_de_perturbation_verbose(verbose, label, int(n_tested_per_slot[local_idx]), n_genes)
                    n_processed += 1
                    pbar.update(1)
                    logger.debug(f"Completed perturbation: {label}")
                    
                # A batch completes several perturbations, so save once the
                # interval is crossed rather than on exact multiples.
                if n_processed - n_at_last_save >= eff_checkpoint_interval:
                    _save_t_test_checkpoint()
                    n_at_last_save = n_processed
            
        # Final checkpoint
        _save_t_test_checkpoint()
        logger.info(f"Completed {len(newly_completed)}/{n_groups} perturbations")
        _print_de_summary(verbose, "t-test DE", len(newly_completed), n_groups, n_tested_list, n_genes)

    _messages.print_saving(verbose, "t_test", output_path)
    return _finish_result(
        run, output_path, obs=obs, method="t_test", corr_method=corr_method,
        scanpy_format=scanpy_format, fingerprint=fingerprint,
    )


@accepts_in_memory
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
    size_factor_method: Literal["libsize", "geomeans", "deseq2"] = "libsize",
    # ---- Dispersion parameters ----
    dispersion: float | None = None,
    shrink_dispersion: bool = True,
    targets_per_fit: int = 1,
    # ---- Optimization parameters ----
    max_iter: int = 100,
    tol: float = 1e-8,
    min_mu: float | None = None,
    poisson_init_iter: int = 5,
    chunk_size: int = 2048,
    # ---- Filtering parameters ----
    min_cells_expressed: int = 0,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_cells_ctrl: int = 1,
    min_cells_pert: int = 1,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
    min_total_count: float = 1.0,
    # ---- Output parameters ----
    lfc_shrinkage_type: Literal["apeglm", "none"] = "none",
    lfc_base: Literal["log2", "ln"] = "log2",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
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
    n_jobs: int | None = None,
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
        Path to an h5ad file, or a crispyx AnnData / backed anndata.AnnData containing
        raw count data.
        An in-memory AnnData is streamed from a temporary copy; ``output_path``
        is then required.
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
        String or categorical columns are one-hot encoded (first level
        dropped); numeric columns enter as one continuous column, so give a
        batch as strings or a categorical, not as integer codes. A design in
        which a covariate is confounded with the perturbation (a batch with
        no control samples) raises ``ValueError``.
    gene_name_column
        Column in `adata.var` with gene symbols. If None, uses `adata.var_names`.
    perturbations
        Specific perturbations to test. If None, tests all non-control groups.
    size_factors
        Per-cell size factors to use as given (rescaled to geometric mean 1),
        e.g. scran's or an external run's, in the input's row order. If None, ``size_factor_method``
        computes them, once for all cells; how the methods differ, and how
        they behaved on single-cell screens, is in the user guide
        (:ref:`size-factors`).
    size_factor_method
        How size factors are computed when ``size_factors`` is None.

        - ``"libsize"`` (default): each cell's total count. Tracks sequencing
          depth exactly; assumes a perturbation does not change a cell's
          total captured RNA.
        - ``"geomeans"``: causarray's median of ratios
          (``comp_size_factor(method="geomeans")``) against the controls: a
          cell's median ratio to the controls' geometric means of their
          non-zero counts, over the genes with a control mean of at least 2.
          Robust to a minority of strongly changed genes.
        - ``"deseq2"``: DESeq2's median of ratios against the controls'
          geometric means, on the genes counted in every control cell (the
          reference ``estimateSizeFactors(geoMeans=...)`` accepts, taken from
          the controls; DESeq2's default takes it from all samples). A
          perturbed cell uses the reference genes counted in every cell of its
          perturbation. Meant for pseudo-bulk and bulk counts; with fewer than
          five genes counted in every control (most single-cell data) it falls
          back to ``"geomeans"``.

        The reference, where there is one, is the controls, so a control's
        factor does not depend on the perturbations in the file or the one
        compared.
    dispersion
        Fixed NB dispersion used for every gene and comparison. If None (the
        default), each comparison estimates its own as DESeq2 does on the
        control and perturbed cells: gene-wise Cox-Reid dispersions, a
        mean-dispersion trend, and maximum a posteriori dispersions under a
        log-normal prior around it. Without covariates the controls enter
        through a one-pass summary, so a comparison costs its own cells.
    shrink_dispersion
        If True (default), test at the MAP dispersions (genes far above the
        trend keep their gene-wise one, as in DESeq2); if False, at the
        gene-wise ones.
    targets_per_fit
        Perturbations fitted together in one model with the controls (1, the
        default: each perturbation on its own, as DESeq2 run per comparison).
        With ``k > 1`` the perturbations are taken ``k`` at a time, in the
        order of ``perturbations`` (else as they appear in the file), and
        fitted together with every control on the design ``~ covariates +
        perturbation``, one indicator column per perturbation: the
        dispersions are estimated once for the ``k`` of them, mostly from the
        controls, and each perturbation is tested on its own column against
        the controls. This is how a pseudo-bulk analysis that fits many
        perturbations as levels of one factor runs (Zhu *et al.* 2026
        fitted 50 at a time), and it costs about ``1/k`` of the dispersion
        estimation. Always fitted on the cells (as with covariates); not with
        ``lfc_shrinkage_type="apeglm"`` (use :func:`shrink_lfc` afterwards).
    max_iter, tol
        IRLS iterations and convergence tolerance for designs with covariates
        (PyDESeq2: 250 and 1e-8). Without covariates the two-group fit
        converges to 1e-10 in at most 250 steps.
    min_mu
        Floor on fitted means: means are held at ``max(min_mu, mean)`` while
        fitting (DESeq2's ``minmu``). DESeq2 uses 0.5 for bulk counts and
        recommends 1e-6 for single cells, where many genes have means below
        one count and a floor of 0.5 binds unevenly between the groups
        compared: tests become conservative, and low-count genes are biased
        when the groups differ in depth (see :ref:`size-factors`). None (the
        default) chooses from the input: 1e-6 when at least half of a sample
        of the counts are zero (single cells), 0.5 otherwise (pseudo-bulk,
        bulk). The value used is in ``uns["min_mu"]``.
    poisson_init_iter
        Initial Poisson iterations before switching to negative binomial.
    chunk_size
        Number of cells read per block when computing size factors (default
        2048). Smaller values stream more, reducing peak memory at the cost
        of additional I/O.
    min_cells_expressed
        Minimum total cells (control + perturbation) expressing a gene for
        testing. Genes below this threshold are untested: ``NaN`` in
        ``score`` / ``pvalue`` / ``logfoldchanges`` and the effect size, with
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
        Only a checkpoint from a call with the same input and
        result-affecting parameters is used; partial results live in a hidden
        ``.<output name>.resume`` directory until the output is written.
    checkpoint_interval
        Number of perturbations to process between checkpoint saves. If None,
        auto-determined based on dataset size (1 for <100 perturbations, 10 for
        <1000, 50 for larger).
    memory_limit_gb
        Memory limit in GB; the effective limit is
        min(available memory, memory_limit_gb). Default None (system memory).
        With covariates, apeGLM shrinkage or ``targets_per_fit > 1`` it sizes
        the parallel workers and the gene blocks each fit holds dense; the
        control cells are held sparse.
    n_jobs
        Number of cores. Without covariates the comparisons run in turn, each
        parallel over genes; with covariates, in parallel worker processes.
        If None or -1, uses all available cores; 1 runs sequentially.
    force
        If True, rerun the analysis even when the output h5ad file already
        exists. If False (default), an existing result is loaded instead of
        rerun when it was computed from the same input file (path, size,
        modification time) with the same result-shaping arguments, as
        recorded in its ``uns["crispyx"]``; otherwise it is recomputed.
    
    Returns
    -------
    RankGenesGroupsResult
        Differential expression results. Access results via dict-like interface:
        `result[label].logfoldchanges`, `result[label].pvalue`, etc. The h5ad
        file path is available at `result.result_path`.

        The h5ad file holds one row per perturbation. ``X`` is the fold
        change in ``lfc_base`` (shrunk under ``lfc_shrinkage_type="apeglm"``).
        Layers: ``z_score``, ``pvalue``, ``pvalue_adj``, ``standard_error``,
        ``intercept`` (ln scale), ``pts``, ``converged`` (bool), and the
        dispersions each comparison tested at (``dispersion``), its gene-wise
        estimates (``dispersion_raw``) and trend (``dispersion_trend``);
        ``logfoldchange_raw`` holds the MLE fold change only when ``X`` is
        shrunk, and with a log2 base ``logfoldchange_raw_ln`` /
        ``standard_error_ln`` keep the ln-scale values :func:`shrink_lfc`
        needs. ``var["pts_rest"]`` is the controls' fraction of expressing
        cells. ``obs["n_genes_tested"]`` counts the genes each perturbation
        was tested on, after the filters.

    Notes
    -----
    To reproduce PyDESeq2 (or DESeq2) run on one perturbation and the
    controls with ``~ condition`` (or ``~ covariates + condition``): give
    both the same size factors (``size_factors=`` here,
    ``dds.obs["size_factors"]`` / ``sizeFactors(dds)`` there; DESeq2's own
    are recomputed on each comparison's cells), the same mean floor
    (``min_mu``; PyDESeq2's default is 0.5), and set the expression filters
    to 0 (``min_pct_ctrl``, ``min_pct_pert``, ``min_mean_ctrl``,
    ``min_mean_pert``, ``min_cells_ctrl``, ``min_cells_pert``,
    ``min_total_count``) so the same genes are tested. Compare p-values
    rather than ``pvalue_adj``: crispyx applies Benjamini-Hochberg per
    perturbation without DESeq2's independent filtering, and it does not
    replace Cook's distance outliers (``refit_cooks=False``,
    ``cooks_filter=False`` there). Gene-wise dispersions can differ on
    sparse genes, most often in pseudo-bulk: where the likelihood has a
    second mode at the lower bound (1e-8), DESeq2's local search can stay
    there, while crispyx reports the higher-likelihood dispersion. The
    trends then differ, but the tests hardly do (Tian CRISPRa pseudo-bulk:
    median z difference 0.001-0.015; hits within a few genes, e.g. crispyx
    found 13 for SPAST, PyDESeq2 16).
    """
    # A one-shot iterable would be fingerprinted by its repr and then used up.
    if perturbations is not None:
        perturbations = [str(p) for p in perturbations]
    covariates = list(covariates or [])
    call_args = dict(locals())  # for the resume fingerprint, before any other local
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="nb_glm_test",
    )

    # Validate min_mu parameter
    if min_mu is not None and min_mu < 0:
        raise ValueError(f"min_mu must be >= 0, got {min_mu}")
    if size_factors is None and size_factor_method not in _SIZE_FACTOR_METHODS:
        # Checked before the file is read or sorted, which can take long.
        raise ValueError(f"size_factor_method must be 'libsize', 'geomeans' or 'deseq2', not {size_factor_method!r}")
    if int(targets_per_fit) != targets_per_fit or targets_per_fit < 1:
        raise ValueError(f"targets_per_fit must be a positive integer, got {targets_per_fit!r}")
    targets_per_fit = int(targets_per_fit)
    if targets_per_fit > 1 and lfc_shrinkage_type == "apeglm":
        raise ValueError(
            "lfc_shrinkage_type='apeglm' fits one perturbation at a time; with targets_per_fit > 1 "
            "shrink afterwards with shrink_lfc."
        )

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
    # Identified by the caller's file: a sorted copy made below is an
    # implementation detail of this run.
    fingerprint = _de_fingerprint(
        path, call_args, method="nb_glm",
        perturbation_column=perturbation_column, control_label=control_label,
    )
    if (r := _try_load_existing_de_result(
        _candidate_output_path, force=force, verbose=verbose,
        fingerprint=fingerprint,
    )):
        return r

    # Warn if the on-disk matrix is stored in CSC format – row-slicing
    # (used by size-factor computation and control-matrix loading) is
    # extremely slow on CSC.  Recommend the CSR standardized file.
    _storage_fmt = get_matrix_storage_format(path)
    if _storage_fmt == "csc":
        _messages.emit(
            f"The input file '{path.name}' stores its matrix in CSC format. "
            "NB-GLM performs row-wise access (size factors, control matrix, "
            "per-perturbation slices) which is very slow on CSC. "
            "Use the CSR-format standardized file for much better performance.",
            UserWarning,
        )

    # A negative binomial fit to normalised or log-transformed values runs and
    # returns plausible-looking nonsense, so refuse it up front. Such data is
    # non-integer throughout; a slice along the fast axis is enough to tell.
    backed = read_backed(path)
    try:
        sample = backed.X[:, :256] if _storage_fmt == "csc" else backed.X[:1024]
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        if size_factors is not None:
            size_factors = _validate_size_factors(size_factors, backed.n_obs)
            input_labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()
    _validate_counts_chunk(sample, context="nb_glm_test")
    if min_mu is None:
        min_mu = _default_min_mu(sample, verbose=verbose, name="nb_glm_test")

    if needs_sorting_for_nbglm(path, perturbation_column=perturbation_column):
        # sort_by_perturbation reuses an existing sorted copy only when it
        # was made from this exact source.
        path = sort_by_perturbation(
            path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            output_path=path.parent / f"{path.stem}_sorted.h5ad",
        )
        if size_factors is not None:  # given in the input's row order
            size_factors = size_factors[perturbation_sort_order(
                input_labels, control_label or resolve_control_label(input_labels, None))]

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        obs_df = backed.obs[[perturbation_column] + covariates].copy()
        labels = label_strings(obs_df[perturbation_column])
        control_label = resolve_control_label(list(labels), control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        if control_n == 0:
            raise ValueError("Control group contains no cells")
        # One pass over the labels, not one per perturbation (18k groups x
        # 3.4M cells of string compares on Huang-HCT116).
        observed = set(pd.unique(labels))
        for label in candidates:
            if label not in observed:
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
    if size_factors is None:
        size_factors = _global_size_factors(path, size_factor_method, chunk_size, labels, control_label)

    offset = np.log(np.clip(size_factors, 1e-8, None))

    # -------------------------------------------------------------------------
    # Perturbations with covariates (or apeGLM shrinkage), fitted on their cells
    # -------------------------------------------------------------------------
    def _fit_perturbation_worker(*, threads: int, **kwargs) -> list[dict]:
        """:func:`_fit_on_cells` with numba's kernels limited to ``threads``,
        this worker's share of the cores (each kernel would otherwise start
        one thread per core in every worker process); the caller's thread
        count is restored, since a sequential run fits in the caller's process."""
        import numba

        previous = numba.get_num_threads()
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        try:
            return _fit_on_cells(**kwargs)
        finally:
            numba.set_num_threads(previous)

    def _fit_on_cells(
        labels_in_fit: list[str],
        group_idxs: list[int],
        path: str | Path,
        rows_in_fit: list,
        control_mask: np.ndarray,
        control_matrix: sp.csc_matrix,
        control_expr_counts: np.ndarray,
        control_total: np.ndarray,
        control_n: int,
        obs_df: pd.DataFrame,
        covariates: list[str],
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
        poisson_init_iter: int,
        dispersion: float | None,
        shrink_dispersion: bool,
        lfc_shrinkage_type: str,
        block_genes: int,
    ) -> list[dict]:
        """The perturbations ``labels_in_fit`` and the controls, fitted together
        on their cells with the design ``~ covariates + perturbation``, one
        indicator column per perturbation (glm.fit_glm_deseq2): the
        dispersions are estimated once for them all, and each perturbation is
        tested on its own column against the controls. One perturbation is the
        per-comparison fit. A result per perturbation, as the two-group
        worker's (natural-log LFC and SE)."""
        masks = []
        for rows in rows_in_fit:
            mask = np.zeros(control_mask.shape[0], dtype=bool)
            mask[rows] = True
            masks.append(mask)
        # A perturbation confounded with a covariate (e.g. all its cells in a
        # batch with no controls) has no estimable effect: it fails with the
        # reason, and the others are fitted without it. (One perturbation alone
        # is checked on the joint design below, which is then its own.)
        failed = []
        if covariates and len(labels_in_fit) > 1:
            kept = []
            for k, (label, mask) in enumerate(zip(labels_in_fit, masks)):
                alone = control_mask | mask
                design_k, columns_k = build_design_matrix(
                    obs_df.iloc[alone], covariate_columns=covariates,
                    perturbation_indicator=mask[alone].astype(np.float64), intercept=True,
                )
                try:
                    check_full_rank(design_k, columns_k, f"nb_glm_test, comparison {label!r} vs control")
                except ValueError as exc:
                    failed.append({"group_idx": group_idxs[k], "error": str(exc)})
                    continue
                kept.append(k)
            if not kept:
                return failed
            labels_in_fit = [labels_in_fit[k] for k in kept]
            group_idxs = [group_idxs[k] for k in kept]
            rows_in_fit = [rows_in_fit[k] for k in kept]
            masks = [masks[k] for k in kept]
        # The fit's rows: the controls, then each perturbation's cells (each
        # group in file order, as it is read below).
        fit_rows = np.concatenate([np.flatnonzero(control_mask), *map(np.flatnonzero, masks)])
        design, design_columns = build_design_matrix(
            obs_df.iloc[fit_rows],
            covariate_columns=covariates,
            perturbation_indicator=masks[0][fit_rows].astype(np.float64),
            intercept=True,
        )
        test_columns = [design_columns.index("perturbation")]
        design_columns[test_columns[0]] = f"perturbation {labels_in_fit[0]!r}"
        for label, mask in zip(labels_in_fit[1:], masks[1:]):
            design = np.column_stack([design, mask[fit_rows].astype(np.float64)])
            test_columns.append(design.shape[1] - 1)
            design_columns.append(f"perturbation {label!r}")
        if covariates:
            context = (
                f"nb_glm_test, comparison {labels_in_fit[0]!r} vs control" if len(labels_in_fit) == 1
                else f"nb_glm_test, perturbations {', '.join(map(repr, labels_in_fit))} vs control"
            )
            try:
                check_full_rank(design, design_columns, context)
            except ValueError as exc:
                return failed + [{"group_idx": i, "error": str(exc)} for i in group_idxs]
        subset_size_factors = np.asarray(size_factors)[fit_rows]

        # The fit's counts are its row blocks, left sparse (the controls are
        # not copied): fit_glm_deseq2 and apeGLM below densify block_genes
        # genes at a time.
        parts = [control_matrix]
        pts_list, valids = [], []
        backed = read_backed(path)
        try:
            for rows, mask in zip(rows_in_fit, masks):
                # A slice on a file sorted by perturbation (one contiguous
                # read), else ascending row indices: file order either way.
                group = sp.csr_matrix(backed.X[rows, :], dtype=np.float64)
                group.eliminate_zeros()
                group_n = group.shape[0]
                group_expr_counts = group.getnnz(axis=0)
                group_total = np.asarray(group.sum(axis=0)).ravel()
                parts.append(group.tocsc())
                del group
                pts_list.append((group_expr_counts / max(group_n, 1)).astype(np.float32))
                valids.append(_nb_glm_testable_genes(
                    pert_expr_counts=group_expr_counts, pert_total=group_total, n_pert=group_n,
                    control_expr_counts=control_expr_counts, control_total=control_total, n_control=control_n,
                    min_cells_expressed=min_cells_expressed, min_pct_ctrl=min_pct_ctrl,
                    min_pct_pert=min_pct_pert, min_mean_ctrl=min_mean_ctrl, min_mean_pert=min_mean_pert,
                    min_cells_ctrl=min_cells_ctrl, min_cells_pert=min_cells_pert,
                    min_total_count=min_total_count,
                ))
        finally:
            backed.file.close()
        # A gene is fitted when any perturbation tests it; each reports only its own.
        fit = fit_glm_deseq2(
            design, parts, subset_size_factors, test_columns[0], np.flatnonzero(np.logical_or.reduce(valids)),
            protected_columns=test_columns[1:], dispersion=dispersion, shrink=shrink_dispersion, min_mu=min_mu,
            max_iter=max_iter, beta_tol=tol, poisson_init_iter=poisson_init_iter, block_genes=block_genes,
        )
        results = []
        for group_idx, column, pts, valid in zip(group_idxs, test_columns, pts_list, valids):
            b, b_se = fit.coef[:, column], fit.coef_se[:, column]
            with np.errstate(divide="ignore", invalid="ignore"):
                statistic = b / b_se
            tested = valid & fit.coef_converged & np.isfinite(statistic) & (b_se > 0)
            lfc = np.where(tested, b, np.nan)
            se = np.where(tested, b_se, np.nan)
            result = {
                "group_idx": group_idx,
                "n_tested": int(valid.sum()),
                "statistic": np.where(tested, statistic, np.nan),
                "pvalue": np.where(tested, 2.0 * norm.sf(np.abs(statistic)), np.nan),
                "logfc": lfc,
                "logfc_raw": lfc.copy(),
                "intercept": np.where(tested, fit.intercept, np.nan),
                "se": se,
                # Fractions of expressing cells describe the data, not a fit,
                # so they are reported for every gene, the untested ones
                # included: a pair expressed in 0% of perturbed and 95% of
                # control cells is a complete knockdown whose effect is not
                # estimable.
                "pts": pts,
                "dispersion": fit.dispersion,
                "dispersion_raw": fit.genewise_dispersion,
                "dispersion_trend": fit.trend,
                "converged": valid & fit.coef_converged,
            }
            if lfc_shrinkage_type == "apeglm":
                # apeGLM refits every coefficient under a Cauchy prior on the
                # perturbation's, starting from the MLE; an untested gene has
                # no start and stays NaN. PyDESeq2's lfc_shrink applies no
                # mean floor. (One perturbation per fit: see nb_glm_test.)
                # The prior's scale comes from every gene's MLE fold change;
                # each gene's refit is its own, so they run a block at a time.
                # With many cells, a gene per task on the fit's threads (which
                # share the block): 4.75x at 536k cells. With few, a refit's
                # NumPy calls are too short to leave the GIL free, and threads
                # cost 25% at 8.6k.
                import numba

                ape_threads = numba.get_num_threads() if design.shape[0] >= 100_000 else 1
                mle_coef = np.where(tested[None, :], fit.coef.T, np.nan)
                prior_scale = _estimate_apeglm_prior_scale(mle_coef[column], se)
                shrunk_lfc, shrunk_se = np.full(n_genes, np.nan), np.full(n_genes, np.nan)
                for start in range(0, n_genes, block_genes):
                    cols = slice(start, start + block_genes)
                    shrunk_coef, shrunk_se[cols], _ = shrink_lfc_apeglm(
                        counts=dense_columns(parts, cols),
                        design_matrix=design,
                        size_factors=subset_size_factors,
                        dispersion=fit.dispersion[cols],
                        mle_coef=mle_coef[:, cols],
                        mle_se=se[cols],
                        shrink_index=column,
                        prior_scale=prior_scale,
                        n_jobs=ape_threads,
                        batch_size=1,
                        backend="threading",
                        min_mu=0.0,
                    )
                    shrunk_lfc[cols] = shrunk_coef[column, :]
                result["logfc"] = shrunk_lfc
                result["se"] = shrunk_se
            results.append(result)
        return failed + results

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

    # Each perturbation is tested against the controls with DESeq2's
    # dispersions estimated on that comparison's cells. Without covariates
    # every control term depends on the controls only, so they are summarised
    # in one pass and a comparison costs its own cells
    # (glm.fit_two_group_deseq2). With covariates, apeGLM shrinkage (which
    # refits on the cells) or several perturbations per fit, each fit runs on
    # its control and perturbed cells (glm.fit_glm_deseq2), with the controls
    # held in memory as a sparse matrix and the genes fitted in blocks.
    two_group = not covariates and lfc_shrinkage_type == "none" and targets_per_fit == 1
    # The memory this call may use: memory_limit_gb, capped by what the
    # process's cgroup will still grant (detected when None). Every size
    # chosen below is a share of it.
    budget_gb = _resolve_memory_limit_bytes(memory_limit_gb) / 1e9
    control_summary = None
    control_matrix = None
    if two_group:
        control_rows = np.flatnonzero(control_mask)
        control_sf = np.asarray(size_factors, dtype=np.float64)[control_rows]
        backed = read_backed(path)
        try:
            def _control_blocks():
                for start in range(0, control_rows.size, 4096):
                    rows = control_rows[start:start + 4096]
                    yield np.arange(start, start + rows.size), backed.X[rows, :]

            control_summary = summarize_controls(_control_blocks(), control_sf, n_genes)
        finally:
            backed.file.close()
        drop_file_cache(path)
        control_expr_counts = control_summary.nnz
        control_total = control_summary.total
    else:
        # CSC, so that a fit takes a block of genes without copying the rest;
        # in the file's dtype (a fit densifies its gene blocks to float64).
        control_matrix = read_rows_csc(path, np.flatnonzero(control_mask))
        drop_file_cache(path)  # Free page cache from control matrix read
        control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
        control_total = np.asarray(control_matrix.sum(axis=0), dtype=np.float64).ravel()

    # pts_rest describes the control arm: the same vector for every row.
    pts_rest = np.divide(
        control_expr_counts,
        control_n,
        out=np.zeros(n_genes, dtype=np.float32),
        where=control_n > 0,
    ).astype(np.float32)

    shape = (n_groups, n_genes)
    _nb_glm_disk_estimate = warn_if_disk_space_low(
        _nb_glm_disk_bytes(*shape),
        output_path,
        context="nb_glm_test",
    )
    _messages.print_disk_estimate(verbose, "nb_glm_test", _nb_glm_disk_estimate)

    # =====================================================================
    # The result file: one row per perturbation, written as each is fitted
    # =====================================================================
    shrunk = lfc_shrinkage_type == "apeglm"
    obs = _result_obs(candidates, perturbation_column)
    var = pd.DataFrame({"pts_rest": pts_rest}, index=pd.Index(gene_symbols).astype(str))
    # X holds the LFC (shrunk or not), in lfc_base. shrink_lfc needs the
    # ln-scale MLE values; with a log2 base they are kept as separate *_ln
    # layers, with an ln base the main layers already are.
    layers = {"z_score": np.float64, "pvalue": np.float64, "pvalue_adj": np.float64}
    if shrunk:
        layers["logfoldchange_raw"] = np.float64  # the MLE LFC
    layers["intercept"] = np.float64  # MLE intercept (ln-scale) for shrink_lfc
    layers["standard_error"] = np.float64
    if lfc_base == "log2":
        layers["logfoldchange_raw_ln"] = np.float64
        layers["standard_error_ln"] = np.float64
    layers.update(dispersion=np.float64, dispersion_raw=np.float64, dispersion_trend=np.float64)
    layers["converged"] = np.bool_
    layers["pts"] = np.float32
    run = _resumable_result(
        output_path,
        checkpoint_path,
        fingerprint=_checkpoint_fingerprint(fingerprint),
        resume=resume,
        obs=obs,
        var=var,
        uns={
            "lfc_base": lfc_base,  # Store for downstream tools
            "method": "nb_glm",
            "fit_method": "independent",
            "control_label": control_label,
            "perturbation_column": perturbation_column,
            "pvalue_correction": corr_method,
            "covariates": covariates,
            "size_factors": size_factors,
            "original_dataset_path": str(path),  # For shrink_lfc to reload data
            "size_factor_method": size_factor_method,
            "min_mu": float(min_mu),
            "targets_per_fit": targets_per_fit,
            "de_filter": {
                "min_cells_expressed": int(min_cells_expressed),
                "min_pct_ctrl": float(min_pct_ctrl),
                "min_pct_pert": float(min_pct_pert),
                "min_mean_ctrl": float(min_mean_ctrl),
                "min_mean_pert": float(min_mean_pert),
            },
        },
        x_dtype=np.float64,
        layers=layers,
    )
    # Earlier failures are retried, so only completions carry over.
    completed_labels: list[str] = run.progress.get("completed", [])
    candidates_to_run = [c for c in candidates if c not in set(completed_labels)]
    if run.resumed:
        _messages.vprint(verbose, "nb_glm_test", f"Resuming: {len(completed_labels)}/{n_groups} perturbations already done")

    # Each comparison's rows: a slice of a file sorted by perturbation, else
    # their indices. A worker gets only its own, not the per-cell labels.
    boundaries = perturbation_boundaries or {}
    unsorted = [label for label in candidates if label not in boundaries]
    row_indices = _group_row_indices(labels, unsorted) if unsorted else {}
    group_rows = {
        label: slice(*boundaries[label]) if label in boundaries else row_indices[label]
        for label in candidates
    }

    # =====================================================================
    # Workers: as many as the budget holds
    # =====================================================================
    # The two-group path runs the comparisons in turn, each parallel over
    # genes in threads that share the one control summary: it uses the cores
    # asked for. Otherwise each comparison is a loky worker process. Its
    # arguments are pickled per task, except numeric arrays over 1 MB (the
    # control matrix, the size factors), which joblib writes once to its
    # temp folder and every worker maps; it holds its comparison's counts and
    # means densely.
    effective_n_jobs = _resolve_n_jobs(n_jobs)
    requested_n_jobs = effective_n_jobs
    n_to_run = len(candidates_to_run)
    # The fits on the cells: the perturbations targets_per_fit at a time, in a
    # fixed order, so a resumed run refits the same groups.
    unfinished = set(candidates_to_run)
    fits = [candidates[i:i + targets_per_fit] for i in range(0, len(candidates), targets_per_fit)]
    fits = [labels_in_fit for labels_in_fit in fits if unfinished.intersection(labels_in_fit)]
    n_fits = len(fits)
    if not two_group:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        with h5py.File(path, "r") as handle:
            x_node = handle["X"]
            nnz_per_cell = (
                x_node["data"].shape[0] / max(n_cells_total, 1) if isinstance(x_node, h5py.Group) else n_genes
            )
        group_sizes = dict(zip(unique_labels, label_counts))
        pert_group_sizes = [group_sizes.get(g, 0) for g in candidates]
        max_group_size = max(pert_group_sizes) if pert_group_sizes else 1
        avg_group_size = max(1, (n_cells_total - control_n) // max(1, n_groups))
        # The p95 group size, so a few outliers do not size every worker.
        use_group_size = (
            int(np.percentile(pert_group_sizes, 95)) if len(pert_group_sizes) >= 10 else max_group_size
        )
        # A fit's genes are dense a block at a time: its counts, means and the
        # dispersion and IRLS work arrays, about 16 arrays of cells x genes.
        # A block takes a tenth of the budget, so that several workers fit,
        # and is sized for the largest fit.
        largest_fit_cells = control_n + max_group_size * targets_per_fit
        block_genes = int(min(n_genes, max(1, budget_gb * 1e9 * 0.1 / (largest_fit_cells * 8 * 16))))
        # Sparse sizes are nnz x 12 bytes (float64 data, int32 indices). A
        # worker reads its perturbations' cells (CSR, then CSC) beside one
        # block; the controls it shares with the parent.
        fit_sparse_mb = use_group_size * targets_per_fit * nnz_per_cell * 12 * 2 / 1e6
        block_mb = (control_n + use_group_size * targets_per_fit) * block_genes * 8 * 16 / 1e6
        per_worker_mb = fit_sparse_mb + block_mb + 2000

        # The parent holds the sparse controls plus its own arrays. The
        # largest perturbation is sized on its own rather than by the p95
        # group above: its worker holds its counts however many cells it has
        # -- a 165k-cell non-targeting group fitted as a perturbation is
        # ~20 GB of them.
        largest_group_mb = max_group_size * nnz_per_cell * 12 * 2 / 1e6
        # IRLS runs its gene batches (~100 MB of work arrays each) on the
        # workers' threads, which number the cores asked for in all.
        irls_threads_mb = requested_n_jobs * 200
        base_memory_mb = control_matrix.nnz * 12 / 1e6 + largest_group_mb + irls_threads_mb + 1000
        available_mb = budget_gb * 1000
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

        memory_limited_workers = min(max_workers_by_memory, max_workers_by_size, n_fits)
        if memory_limited_workers < effective_n_jobs:
            if memory_limited_workers == n_fits and n_fits < max_workers_by_memory and n_fits < max_workers_by_size:
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

    # For small number of perturbations to run, run sequentially to avoid overhead.
    use_parallel = n_fits >= 2 and (n_to_run >= 4 or targets_per_fit > 1) and effective_n_jobs > 1 and not two_group

    def _fit_two_group_worker(group_idx: int, label: str, backed_x, rows) -> dict:
        """One perturbation against the control summary; same result as the
        other workers, natural-log LFC and SE."""
        # Sparse throughout: a large group (a non-targeting guide fitted as a
        # perturbation) would not fit in memory densely.
        y = sp.csr_matrix(backed_x[rows, :], dtype=np.float64)
        y.sum_duplicates()
        y.eliminate_zeros()
        s_t = np.asarray(size_factors, dtype=np.float64)[rows]
        group_n = y.shape[0]
        group_expr_counts = y.getnnz(axis=0)
        group_total = np.asarray(y.sum(axis=0)).ravel()
        valid = _nb_glm_testable_genes(
            pert_expr_counts=group_expr_counts, pert_total=group_total, n_pert=group_n,
            control_expr_counts=control_expr_counts, control_total=control_total, n_control=control_n,
            min_cells_expressed=min_cells_expressed, min_pct_ctrl=min_pct_ctrl,
            min_pct_pert=min_pct_pert, min_mean_ctrl=min_mean_ctrl, min_mean_pert=min_mean_pert,
            min_cells_ctrl=min_cells_ctrl, min_cells_pert=min_cells_pert,
            min_total_count=min_total_count,
        )
        fit = fit_two_group_deseq2(
            control_summary, y, s_t, np.flatnonzero(valid),
            dispersion=dispersion, shrink=shrink_dispersion, min_mu=min_mu,
        )
        tested = fit.converged & np.isfinite(fit.statistic)
        lfc = np.where(tested, fit.lfc, np.nan)
        return {
            "group_idx": group_idx,
            "n_tested": int(valid.sum()),
            "statistic": np.where(tested, fit.statistic, np.nan),
            "pvalue": np.where(tested, fit.pvalue, np.nan),
            "logfc": lfc,
            "logfc_raw": lfc.copy(),
            "intercept": np.where(tested, fit.intercept, np.nan),
            "se": np.where(tested, fit.se, np.nan),
            "pts": (group_expr_counts / max(group_n, 1)).astype(np.float32),
            "dispersion": fit.dispersion,
            "dispersion_raw": fit.genewise_dispersion,
            "dispersion_trend": fit.trend,
            "converged": fit.converged,
        }

    # Log progress info
    if resume and completed_labels:
        logger.info(f"Fitting {n_to_run}/{n_groups} remaining perturbations with {effective_n_jobs} workers...")
    else:
        logger.info(f"Fitting {n_groups} perturbations with {effective_n_jobs} workers...")
        
    # Track completed labels during this run
    newly_completed = list(completed_labels)  # Start with already completed
    newly_failed: list[str] = []
        
    # A fitted row goes straight into the result file, in its final units.
    out = {key: run.file[key] for key in ["X", *(f"layers/{name}" for name in layers)]}
    lfc_scale = np.log(2) if lfc_base == "log2" else 1.0

    def _write_result_row(res: dict, label: str) -> None:
        idx = candidate_to_idx[label]
        out["X"][idx] = res["logfc"] / lfc_scale
        out["layers/z_score"][idx] = res["statistic"]
        out["layers/pvalue"][idx] = res["pvalue"]
        if shrunk:
            out["layers/logfoldchange_raw"][idx] = res["logfc_raw"] / lfc_scale
        out["layers/intercept"][idx] = res["intercept"]
        out["layers/standard_error"][idx] = res["se"] / lfc_scale
        if lfc_base == "log2":
            out["layers/logfoldchange_raw_ln"][idx] = res["logfc_raw"]
            out["layers/standard_error_ln"][idx] = res["se"]
        out["layers/dispersion"][idx] = res["dispersion"]
        out["layers/dispersion_raw"][idx] = res["dispersion_raw"]
        out["layers/dispersion_trend"][idx] = res["dispersion_trend"]
        out["layers/converged"][idx] = res["converged"]
        out["layers/pts"][idx] = res["pts"]

    def _write_failed_row(label: str) -> None:
        # NaN, not the file's zeros: a p-value of 0 would report every gene
        # as significant for it.
        idx = candidate_to_idx[label]
        for key, dataset in out.items():
            dataset[idx] = np.nan if dataset.dtype.kind == "f" else 0

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

        def _record(label: str, compute) -> None:
            try:
                res = compute()
                _write_result_row(res, label)
                newly_completed.append(label)
                n_tested_list.append(res["n_tested"])
                _print_de_perturbation_verbose(verbose, label, res["n_tested"], n_genes)
                logger.debug(f"Completed perturbation: {label}")
            except Exception as e:
                logger.error(f"Failed perturbation {label}: {e}")
                newly_failed.append(label)
                _write_failed_row(label)
            pbar.update(1)
            if (len(newly_completed) - len(completed_labels) + len(newly_failed)) % eff_checkpoint_interval == 0:
                _save_checkpoint()

        def _record_fit(labels_in_fit: list[str], compute) -> None:
            # A fit that fails fails each of its perturbations; a resumed run
            # keeps the rows it already wrote.
            def fitted(label: str) -> dict:
                if error is not None:
                    raise error
                if "error" in by_label[label]:
                    raise ValueError(by_label[label]["error"])
                return by_label[label]

            try:
                by_label, error = {candidates[r["group_idx"]]: r for r in compute()}, None
            except Exception as e:
                by_label, error = {}, e
            for label in labels_in_fit:
                if label in unfinished:
                    _record(label, lambda label=label: fitted(label))

        def _fit_or_fail(**kwargs) -> list[dict]:
            try:
                return _fit_perturbation_worker(**kwargs)
            except Exception as exc:
                return [{"group_idx": i, "error": f"{type(exc).__name__}: {exc}"} for i in kwargs["group_idxs"]]

        def _worker_args(labels_in_fit: list[str]) -> dict:
            return dict(
                labels_in_fit=labels_in_fit, group_idxs=[candidate_to_idx[label] for label in labels_in_fit],
                path=path, rows_in_fit=[group_rows[label] for label in labels_in_fit],
                control_mask=control_mask, control_matrix=control_matrix,
                control_expr_counts=control_expr_counts, control_total=control_total, control_n=control_n,
                obs_df=obs_df, covariates=covariates, size_factors=size_factors, n_genes=n_genes,
                min_cells_expressed=min_cells_expressed, min_pct_ctrl=min_pct_ctrl, min_pct_pert=min_pct_pert,
                min_mean_ctrl=min_mean_ctrl, min_mean_pert=min_mean_pert, min_cells_ctrl=min_cells_ctrl,
                min_cells_pert=min_cells_pert, min_total_count=min_total_count, max_iter=max_iter, tol=tol,
                min_mu=min_mu, poisson_init_iter=poisson_init_iter, dispersion=dispersion,
                shrink_dispersion=shrink_dispersion, lfc_shrinkage_type=lfc_shrinkage_type,
                block_genes=block_genes, threads=max(1, requested_n_jobs // (effective_n_jobs if use_parallel else 1)),
            )

        if two_group:
            import numba

            threads = numba.get_num_threads()
            numba.set_num_threads(min(requested_n_jobs, numba.config.NUMBA_NUM_THREADS))
            backed = read_backed(path)
            try:
                for label in candidates_to_run:
                    _record(label, lambda label=label: _fit_two_group_worker(
                        candidate_to_idx[label], label, backed.X, group_rows[label]))
            finally:
                backed.file.close()
                numba.set_num_threads(threads)
        elif use_parallel:
            # Each fit's rows are written as soon as it is done, rather than
            # held until every earlier one is.
            results = Parallel(
                n_jobs=effective_n_jobs, backend="loky", prefer="processes", return_as="generator_unordered",
            )(delayed(_fit_or_fail)(**_worker_args(labels_in_fit)) for labels_in_fit in fits)
            for fit_results in results:
                labels_in_fit = [candidates[r["group_idx"]] for r in fit_results]
                _record_fit(labels_in_fit, lambda fit_results=fit_results: fit_results)
        else:
            for labels_in_fit in fits:
                _record_fit(labels_in_fit,
                            lambda labels_in_fit=labels_in_fit: _fit_or_fail(**_worker_args(labels_in_fit)))

    # Final checkpoint save
    _save_checkpoint()
    logger.info(f"Completed {len(newly_completed)}/{n_groups} perturbations")
    _print_de_summary(verbose, "NB-GLM DE", len(newly_completed), n_groups, n_tested_list, n_genes)
    if newly_failed:
        _messages.warn(
            "nb_glm_test",
            f"{len(newly_failed)} perturbation(s) failed and are NaN in the result (each "
            f"error is logged): {newly_failed[:5]}{'...' if len(newly_failed) > 5 else ''}",
        )

    # Store profiling results or "NA" for production
    if profiling and profiler is not None:
        profiler.stop("fit")
        profiler.snapshot("fit_end")
        profiler.stop("total")
        profiler.stop_sampling()
        stats = profiler.get_stats()
        profiling_uns = {
            "profiling_enabled": True,
            "fit_seconds": stats.get("timing", {}).get("sections", {}).get("fit", {}).get("seconds", 0.0),
            "fit_peak_memory_mb": stats.get("memory", {}).get("peak_mb", 0.0),
            "total_seconds": stats.get("timing", {}).get("total_seconds", 0.0),
        }
    else:
        profiling_uns = "NA"

    _messages.print_saving(verbose, "nb_glm_test", output_path)
    return _finish_result(
        run, output_path, obs=obs, method="nb_glm", corr_method=corr_method,
        scanpy_format=scanpy_format, fingerprint=fingerprint, uns={"profiling": profiling_uns},
    )


# =============================================================================
# Fixed-offset NB test: each perturbation on its own cells, the controls as
# a per-stratum baseline
# =============================================================================

@dataclass(frozen=True)
class _OffsetBaseline:
    """The control cells of a screen reduced to what :func:`nb_offset_test`
    needs: per stratum and gene the baseline mean ``mean[b, g]`` (NaN for a
    stratum without controls), the per-gene dispersion to test at, and the
    controls' per-gene expressing-cell counts and totals for the filters."""

    mean: np.ndarray            # (strata, genes)
    has_controls: np.ndarray    # (strata,)
    pooled_mean: np.ndarray     # (genes,): sum y / sum s over every control
    dispersion: np.ndarray      # (genes,)
    dispersion_raw: np.ndarray  # (genes,)
    dispersion_trend: np.ndarray  # (genes,)
    nnz: np.ndarray             # (genes,)
    total: np.ndarray           # (genes,)
    n_cells: int


def _offset_baseline(
    blocks, size_factors: np.ndarray, strata: np.ndarray, n_strata: int, n_genes: int,
    *, dispersion: "float | np.ndarray | None", min_mean_for_dispersion: float = 0.5,
) -> _OffsetBaseline:
    """One pass over the control cells.

    ``blocks`` yields ``(rows, csr)``: positions into ``size_factors`` /
    ``strata`` and those cells' counts. The baseline of stratum ``b`` is
    ``(sum y + m_g) / (sum s + 1)`` over its controls, one pseudo-cell at the
    pooled control mean ``m_g``: a stratum whose controls happen to have no
    count of a gene then gives a small baseline instead of zero, which would
    send any count among the perturbed cells to the upper bound of the
    effect (with one stratum the pseudo-cell changes nothing). With
    ``dispersion=None`` the dispersion is the method-of-moments estimate
    pooled over strata, ``sum (y - s mu_b)^2 = sum s mu_b + alpha sum s^2
    mu_b^2``, for genes with a pooled mean of at least
    ``min_mean_for_dispersion`` counts per unit size factor, and the
    parametric mean-dispersion trend through them for the others.
    """
    s_all = np.asarray(size_factors, dtype=np.float64)
    sy = np.zeros((n_strata, n_genes))
    syy = np.zeros((n_strata, n_genes)) if dispersion is None else None
    ssy = np.zeros((n_strata, n_genes)) if dispersion is None else None
    s1 = np.zeros(n_strata)
    s2 = np.zeros(n_strata)
    nnz = np.zeros(n_genes, dtype=np.int64)
    total = np.zeros(n_genes)
    n_cells = 0
    for rows, block in blocks:
        csr = sp.csr_matrix(block, dtype=np.float64)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        rows = np.asarray(rows)
        b, s = strata[rows], s_all[rows]
        onehot = sp.csr_matrix((np.ones(rows.size), (b, np.arange(rows.size))), shape=(n_strata, rows.size))
        sy += (onehot @ csr).toarray()
        if dispersion is None:
            syy += (onehot @ csr.multiply(csr)).toarray()
            ssy += (onehot @ (sp.diags(s) @ csr)).toarray()
        s1 += np.bincount(b, weights=s, minlength=n_strata)
        s2 += np.bincount(b, weights=s * s, minlength=n_strata)
        nnz += np.bincount(csr.indices, minlength=n_genes)
        total += np.asarray(csr.sum(axis=0)).ravel()
        n_cells += rows.size
    has_controls = s1 > 0
    pooled = sy.sum(axis=0) / s1.sum()
    mean = np.full((n_strata, n_genes), np.nan)
    mean[has_controls] = (sy[has_controls] + pooled) / (s1[has_controls] + 1.0)[:, None]
    if dispersion is not None:
        final = np.broadcast_to(np.asarray(dispersion, dtype=np.float64), (n_genes,)).astype(np.float64)
        raw = trend = np.full(n_genes, np.nan)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = sy[has_controls] / s1[has_controls][:, None]
            numerator = (syy[has_controls] - 2 * mu * ssy[has_controls] + mu ** 2 * s2[has_controls][:, None]
                         - mu * s1[has_controls][:, None]).sum(axis=0)
            raw = numerator / (mu ** 2 * s2[has_controls][:, None]).sum(axis=0)
        raw = np.where(np.isfinite(raw), np.clip(raw, 1e-8, None), np.nan)
        genewise = (pooled >= min_mean_for_dispersion) & np.isfinite(raw)
        trend = fit_dispersion_trend(pooled, np.where(genewise, raw, np.nan))
        final = np.where(genewise, raw, trend)
    return _OffsetBaseline(
        mean=mean, has_controls=has_controls, pooled_mean=pooled, dispersion=final, dispersion_raw=raw,
        dispersion_trend=trend, nnz=nnz, total=total, n_cells=n_cells,
    )


def _fit_offset_group(
    y: sp.csr_matrix, s: np.ndarray, strata: np.ndarray, baseline: _OffsetBaseline, genes: np.ndarray,
    *, min_mu: float, max_iter: int, tol: float, block_cells: int,
) -> dict:
    """One perturbation's cells against the baseline: for each gene the
    one-parameter NB fit of ``y_i ~ NB(s_i mean[b(i), g] exp(beta), alpha_g)``
    by Fisher scoring, from the Poisson start ``ln(sum y / sum s mean)``.
    ``beta`` is held above ``ln(min_mu) - ln(mean baseline)`` (the fitted-mean
    floor) and below ``ETA_MAX``; its standard error is one over the root of
    the Fisher information at the optimum. The cells are taken
    ``block_cells`` at a time, so a large perturbation is never dense at once.
    Returns natural-log ``lfc``, ``se``, ``converged`` and ``intercept`` (the
    log of the cells' mean expected count under the baseline), NaN off
    ``genes`` and where the baseline is zero."""
    n_cells, n_genes = y.shape
    alpha = baseline.dispersion
    blocks = [slice(start, min(start + block_cells, n_cells)) for start in range(0, n_cells, block_cells)]

    def base_of(sl: slice) -> np.ndarray:
        return s[sl, None] * baseline.mean[strata[sl]]

    bsum = np.zeros(n_genes)
    for sl in blocks:
        bsum += base_of(sl).sum(axis=0)
    ysum = np.asarray(y.sum(axis=0)).ravel()
    fit = np.zeros(n_genes, dtype=bool)
    fit[genes] = True
    fit &= bsum > 0
    lo = np.full(n_genes, ETA_MIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        if min_mu > 0:
            lo = np.where(fit, np.log(min_mu) - np.log(bsum / n_cells), ETA_MIN)
        beta = np.where(fit, np.clip(np.log(ysum / bsum), lo, ETA_MAX), 0.0)
    converged = np.zeros(n_genes, dtype=bool)
    info = np.zeros(n_genes)
    for _ in range(max_iter):
        score = np.zeros(n_genes)
        info = np.zeros(n_genes)
        for sl in blocks:
            mu = base_of(sl) * np.exp(beta)
            w = 1.0 / (1.0 + alpha * mu)
            score += np.asarray(y[sl].multiply(w).sum(axis=0)).ravel() - (mu * w).sum(axis=0)
            info += (mu * w).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            new = np.where(fit & (info > 0), np.clip(beta + score / info, lo, ETA_MAX), beta)
        step = np.abs(new - beta)
        beta = new
        converged = fit & (step < tol)
        if converged[fit].all():
            break
    info = np.zeros(n_genes)
    for sl in blocks:
        mu = base_of(sl) * np.exp(beta)
        info += (mu / (1.0 + alpha * mu)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(fit & (info > 0), 1.0 / np.sqrt(info), np.nan)
        intercept = np.where(fit, np.log(bsum / n_cells), np.nan)
    return {
        "lfc": np.where(fit, beta, np.nan), "se": se, "converged": converged, "intercept": intercept,
        "n_fit": int(fit.sum()),
    }


def _default_min_mu(sample, *, verbose: int | bool, name: str) -> float:
    """DESeq2's floor on fitted means, chosen from the share of zeros in a
    sample of the counts: 1e-6 when at least half are zero (single cells),
    0.5 otherwise (pseudo-bulk, bulk). DESeq2's 0.5 is a bulk setting; in
    sparse single-cell counts it binds more often in the shallower group,
    which makes tests conservative and biases low-count genes when the
    groups differ in depth (docs/notebooks/size_factor_comparison.ipynb)."""
    n_values = sample.shape[0] * sample.shape[1]
    nonzero = sample.nnz if sp.issparse(sample) else np.count_nonzero(sample)
    zero_share = 1.0 - nonzero / max(n_values, 1)
    min_mu = 1e-6 if zero_share >= 0.5 else 0.5
    _messages.vprint(verbose, name, f"min_mu={min_mu:g} ({zero_share:.0%} of a sample of the counts are zero)")
    return min_mu


@accepts_in_memory
def nb_offset_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str | None = None,
    groupby: str | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    strata: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    size_factors: ArrayLike | None = None,
    size_factor_method: Literal["libsize", "geomeans", "deseq2"] = "libsize",
    dispersion: float | ArrayLike | None = None,
    min_mu: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    chunk_size: int = 2048,
    min_cells_expressed: int = 0,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
    min_cells_ctrl: int = 1,
    min_cells_pert: int = 1,
    min_mean_ctrl: float = 0.05,
    min_mean_pert: float = 0.005,
    min_total_count: float = 1.0,
    lfc_base: Literal["log2", "ln"] = "log2",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    data_name: str | None = None,
    output_path: str | Path | None = None,
    scanpy_format: bool = False,
    verbose: int | bool = True,
    memory_limit_gb: float | None = None,
    n_jobs: int | None = None,
    force: bool = False,
) -> RankGenesGroupsResult:
    """Negative binomial test of each perturbation on its own cells, with the
    controls as a fixed per-stratum baseline.

    For gene *g* in a cell *i* of stratum *b*, the model is
    ``y_i ~ NB(s_i m_bg exp(beta), alpha_g)``: ``m_bg`` is the control cells'
    mean in the stratum (counts per unit size factor), fixed from one pass
    over the controls, and ``beta`` is the one parameter fitted per
    perturbation and gene, on the perturbation's cells only. This is the
    cell-level :func:`nb_glm_test` with ``covariates=[strata]`` whose
    stratum effects are taken from the controls rather than estimated
    jointly, which they dominate anyway when every stratum holds thousands
    of control cells against a perturbation's handful: on such screens the
    two agree to a few percent in ``beta`` and its standard error
    (:doc:`/nb_glm`). The difference is cost. The joint fit holds every
    control cell in each fit and solves a design with one column per stratum
    (hundreds, for lane x donor); this fit reads the controls once and then
    each perturbation's own cells, so a genome-wide screen whose
    perturbations have 20-40 cells spread one per stratum, where no
    pseudo-bulk unit can be formed and the joint fit takes days, runs in
    minutes.

    The baseline of a stratum is ``(sum y + m_g) / (sum s + 1)`` over its
    controls: one pseudo-cell at the pooled control mean ``m_g``, so that a
    stratum whose controls happen to have no count of a gene gives a small
    baseline rather than zero (which would send any count among the
    perturbed cells to the upper bound of ``beta``). Cells of a stratum
    without any control carry no baseline and are left out, with a warning.
    The dispersion is fixed per gene: ``dispersion`` as given (one value, or
    one per gene), or estimated from the controls by the method of moments
    pooled over strata, with the parametric mean-dispersion trend standing in
    for genes below 0.5 counts per unit size factor. ``beta`` is fitted by
    Fisher scoring from its Poisson start, held above the fitted-mean floor
    implied by ``min_mu`` and below ``exp(20)``; its standard error is the
    Wald one at the optimum.

    The result has the layout of :func:`nb_glm_test`'s (``X`` the fold
    change in ``lfc_base``, ``standard_error``, ``z_score``, ``pvalue``,
    ``pvalue_adj``, ``intercept``, ``dispersion``, ``pts``, ``converged``),
    so :func:`shrink_lfc` and every reader of an NB-GLM result accept it;
    ``uns["method"]`` is ``"nb_offset"``. The expression filters are
    :func:`nb_glm_test`'s, with the same defaults. An existing output is
    reused when it was computed from the same input and arguments
    (:ref:`auto-reload`).

    Parameters
    ----------
    data
        Path to an h5ad file of raw counts, or a backed AnnData. An
        in-memory AnnData is streamed from a temporary copy; ``output_path``
        is then required.
    perturbation_column, groupby
        obs column with the perturbation labels (``groupby`` is Scanpy's name
        for it).
    control_label, reference
        The control label (``reference`` is Scanpy's name); inferred from
        common patterns when None.
    strata
        obs column of the stratum each cell belongs to (10x lane, donor, or
        their combination as one string column). None fits one stratum, the
        plain two-group comparison.
    gene_name_column
        var column with gene symbols; None uses ``var_names``.
    perturbations
        The perturbations to test; None tests every non-control label. An
        empty list writes an empty result.
    size_factors, size_factor_method
        As in :func:`nb_glm_test`: per-cell factors as given, or computed
        once for all cells by the method.
    dispersion
        The NB dispersion to test at, one value or one per gene (e.g. a
        control-only estimate). None estimates it from the controls as
        described above.
    min_mu
        Floor on fitted means, as in :func:`nb_glm_test`; None chooses 1e-6
        for single cells and 0.5 for pseudo-bulk from the share of zeros.
    max_iter, tol
        Fisher-scoring iterations and the step below which a gene counts as
        converged.
    chunk_size
        Cells read per block when computing size factors.
    min_cells_expressed, min_pct_ctrl, min_pct_pert, min_cells_ctrl, min_cells_pert, min_mean_ctrl, min_mean_pert, min_total_count
        :func:`nb_glm_test`'s expression filters, with its defaults.
    lfc_base
        ``"log2"`` (default) or ``"ln"`` for ``X`` and ``standard_error``.
    corr_method
        ``"benjamini-hochberg"`` or ``"bonferroni"``, per perturbation.
    data_name, output_path
        The output file: ``output_path`` as given, else beside the input
        with the ``nb_offset`` suffix or ``data_name``.
    scanpy_format
        Also write Scanpy's ``uns["rank_genes_groups"]``.
    verbose
        Print progress.
    memory_limit_gb
        Bounds the cells a fit holds dense at a time (``None``: the memory
        available). The baseline itself is ``strata x genes`` floats.
    n_jobs
        Perturbations fitted concurrently, in threads (None or -1: all
        cores). The controls are read once, each perturbation's cells once.
    force
        Recompute even when a reusable output exists.

    Returns
    -------
    RankGenesGroupsResult
        One row per perturbation; see :func:`nb_glm_test`.
    """
    name = "nb_offset_test"
    if perturbations is not None:
        perturbations = [str(p) for p in perturbations]
    call_args = dict(locals())
    call_args.pop("name")
    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column, groupby=groupby, control_label=control_label,
        reference=reference, fn_name=name,
    )
    if min_mu is not None and min_mu < 0:
        raise ValueError(f"min_mu must be >= 0, got {min_mu}")
    if size_factors is None and size_factor_method not in _SIZE_FACTOR_METHODS:
        raise ValueError(f"size_factor_method must be 'libsize', 'geomeans' or 'deseq2', not {size_factor_method!r}")
    if dispersion is not None:
        dispersion = np.asarray(dispersion, dtype=np.float64)
        if not np.isfinite(dispersion).all() or (dispersion <= 0).any():
            raise ValueError(f"{name}: dispersion must be positive and finite")

    path = resolve_data_path(data)
    output_path = resolve_output_path(path, suffix="nb_offset", data_name=data_name, output_path=output_path)
    fingerprint = _de_fingerprint(
        path, call_args, method="nb_offset", perturbation_column=perturbation_column, control_label=control_label,
    )
    if (existing := _try_load_existing_de_result(output_path, force=force, verbose=verbose, fingerprint=fingerprint)):
        return existing

    storage_format = get_matrix_storage_format(path)
    backed = read_backed(path)
    try:
        sample = backed.X[:, :256] if storage_format == "csc" else backed.X[:1024]
        for column in (perturbation_column, strata):
            if column is not None and column not in backed.obs.columns:
                raise KeyError(f"Column '{column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}")
        if size_factors is not None:
            size_factors = _validate_size_factors(size_factors, backed.n_obs)
            input_labels = backed.obs[perturbation_column].astype(str).to_numpy()
        n_genes = backed.n_vars
        if dispersion is not None and dispersion.ndim and dispersion.shape != (n_genes,):
            raise ValueError(f"{name}: dispersion has shape {dispersion.shape}; expected one value or ({n_genes},)")
    finally:
        backed.file.close()
    _validate_counts_chunk(sample, context=name)
    if min_mu is None:
        min_mu = _default_min_mu(sample, verbose=verbose, name=name)

    if needs_sorting_for_nbglm(path, perturbation_column=perturbation_column):
        path = sort_by_perturbation(
            path, perturbation_column=perturbation_column, control_label=control_label,
            output_path=path.parent / f"{path.stem}_sorted.h5ad",
        )
        if size_factors is not None:
            size_factors = size_factors[perturbation_sort_order(
                input_labels, control_label or resolve_control_label(input_labels, None))]

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        labels = label_strings(backed.obs[perturbation_column])
        control_label = resolve_control_label(list(labels), control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        control_mask = labels == control_label
        control_rows = np.flatnonzero(control_mask)
        if control_rows.size == 0:
            raise ValueError("Control group contains no cells")
        observed = set(pd.unique(labels))
        for label in candidates:
            if label not in observed:
                raise ValueError(f"Perturbation '{label}' contains no cells")
        if strata is None:
            stratum_codes, stratum_names = np.zeros(labels.size, dtype=np.int64), np.array(["all"])
        else:
            stratum_codes, stratum_names = pd.factorize(backed.obs[strata].astype(str).to_numpy())
            stratum_codes = stratum_codes.astype(np.int64)
        boundaries = {}
        if "sorting_metadata" in backed.uns:
            metadata = backed.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                boundaries = metadata.get("perturbation_boundaries", {}) or {}
    finally:
        backed.file.close()
    n_strata = len(stratum_names)
    if size_factors is None:
        size_factors = _global_size_factors(path, size_factor_method, chunk_size, labels, control_label)
    size_factors = np.asarray(size_factors, dtype=np.float64)

    # One pass over the controls: the baseline, the dispersion, the filters' counts.
    backed = read_backed(path)
    try:
        def control_blocks():
            for start in range(0, control_rows.size, 4096):
                rows = control_rows[start:start + 4096]
                yield rows, backed.X[rows, :]

        baseline = _offset_baseline(
            control_blocks(), size_factors, stratum_codes, n_strata, n_genes, dispersion=dispersion,
        )
    finally:
        backed.file.close()
    drop_file_cache(path)
    without_controls = ~baseline.has_controls
    orphan_cells = int((without_controls[stratum_codes] & ~control_mask).sum())
    if orphan_cells:
        _messages.warn(
            name,
            f"{int(without_controls.sum())} of {n_strata} strata have no control cell; their {orphan_cells} "
            "perturbed cells carry no baseline and are left out of the fits.",
        )
    n_control = baseline.n_cells
    pts_rest = (baseline.nnz / max(n_control, 1)).astype(np.float32)

    unsorted = [label for label in candidates if label not in boundaries]
    row_indices = _group_row_indices(labels, unsorted) if unsorted else {}
    group_rows = {
        label: np.arange(*boundaries[label]) if label in boundaries else row_indices[label] for label in candidates
    }
    budget = _resolve_memory_limit_bytes(memory_limit_gb)
    # A fit holds a block of cells dense in about five cells x genes arrays
    # (baseline, mean, weight, their products); a block takes a tenth of
    # the budget, so the threads' blocks fit beside the baseline.
    block_cells = int(max(16, min(65536, budget * 0.1 / (n_genes * 8 * 5))))
    n_threads = max(1, _resolve_n_jobs(n_jobs))

    n_groups = len(candidates)
    shape = (n_groups, n_genes)
    _messages.print_disk_estimate(
        verbose, name, warn_if_disk_space_low(_nb_glm_disk_bytes(*shape), output_path, context=name),
    )
    obs = _result_obs(candidates, perturbation_column)
    var = pd.DataFrame({"pts_rest": pts_rest}, index=pd.Index(gene_symbols).astype(str))
    layers = {"z_score": np.float64, "pvalue": np.float64, "pvalue_adj": np.float64, "intercept": np.float64,
              "standard_error": np.float64}
    if lfc_base == "log2":
        layers["logfoldchange_raw_ln"] = np.float64
        layers["standard_error_ln"] = np.float64
    layers.update(dispersion=np.float64, dispersion_raw=np.float64, dispersion_trend=np.float64)
    layers["converged"] = np.bool_
    layers["pts"] = np.float32
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run = _resumable_result(
        output_path, output_path.with_suffix(".progress.json"),
        fingerprint=_checkpoint_fingerprint(fingerprint), resume=False, obs=obs, var=var,
        uns={
            "lfc_base": lfc_base,
            "method": "nb_offset",
            "control_label": control_label,
            "perturbation_column": perturbation_column,
            "strata": strata if strata is not None else "",
            "n_strata": int(n_strata),
            "pvalue_correction": corr_method,
            "size_factors": size_factors,
            "size_factor_method": size_factor_method,
            "original_dataset_path": str(path),
            "min_mu": float(min_mu),
            "dispersion_source": "given" if dispersion is not None else "controls",
            "de_filter": {
                "min_cells_expressed": int(min_cells_expressed),
                "min_pct_ctrl": float(min_pct_ctrl),
                "min_pct_pert": float(min_pct_pert),
                "min_mean_ctrl": float(min_mean_ctrl),
                "min_mean_pert": float(min_mean_pert),
            },
        },
        x_dtype=np.float64, layers=layers,
    )
    out = {key: run.file[key] for key in ["X", *(f"layers/{layer}" for layer in layers)]}
    lfc_scale = np.log(2) if lfc_base == "log2" else 1.0
    candidate_to_idx = {label: idx for idx, label in enumerate(candidates)}

    def fit(label: str, y: sp.csr_matrix, rows: np.ndarray) -> dict:
        s, codes = size_factors[rows], stratum_codes[rows]
        n_pert = y.shape[0]
        expr_counts = y.getnnz(axis=0)
        total = np.asarray(y.sum(axis=0)).ravel()
        valid = _nb_glm_testable_genes(
            pert_expr_counts=expr_counts, pert_total=total, n_pert=n_pert,
            control_expr_counts=baseline.nnz, control_total=baseline.total, n_control=n_control,
            min_cells_expressed=min_cells_expressed, min_pct_ctrl=min_pct_ctrl, min_pct_pert=min_pct_pert,
            min_mean_ctrl=min_mean_ctrl, min_mean_pert=min_mean_pert, min_cells_ctrl=min_cells_ctrl,
            min_cells_pert=min_cells_pert, min_total_count=min_total_count,
        )
        keep = baseline.has_controls[codes]
        if not keep.any():
            raise ValueError(f"every cell of '{label}' is in a stratum without controls")
        if not keep.all():
            y, s, codes = y[keep], s[keep], codes[keep]
        result = _fit_offset_group(
            y, s, codes, baseline, np.flatnonzero(valid), min_mu=min_mu, max_iter=max_iter, tol=tol,
            block_cells=block_cells,
        )
        tested = np.isfinite(result["lfc"]) & np.isfinite(result["se"]) & (result["se"] > 0)
        lfc = np.where(tested, result["lfc"], np.nan)
        se = np.where(tested, result["se"], np.nan)
        with np.errstate(invalid="ignore"):
            statistic = lfc / se
        return {
            "label": label, "n_tested": int(tested.sum()), "lfc": lfc, "se": se, "statistic": statistic,
            "pvalue": np.where(tested, 2.0 * norm.sf(np.abs(statistic)), np.nan),
            "intercept": np.where(tested, result["intercept"], np.nan),
            "converged": tested & result["converged"], "pts": (expr_counts / max(n_pert, 1)).astype(np.float32),
        }

    def write(res: dict) -> None:
        idx = candidate_to_idx[res["label"]]
        out["X"][idx] = res["lfc"] / lfc_scale
        out["layers/z_score"][idx] = res["statistic"]
        out["layers/pvalue"][idx] = res["pvalue"]
        out["layers/intercept"][idx] = res["intercept"]
        out["layers/standard_error"][idx] = res["se"] / lfc_scale
        if lfc_base == "log2":
            out["layers/logfoldchange_raw_ln"][idx] = res["lfc"]
            out["layers/standard_error_ln"][idx] = res["se"]
        out["layers/dispersion"][idx] = baseline.dispersion
        out["layers/dispersion_raw"][idx] = baseline.dispersion_raw
        out["layers/dispersion_trend"][idx] = baseline.dispersion_trend
        out["layers/converged"][idx] = res["converged"]
        out["layers/pts"][idx] = res["pts"]

    def write_failed(label: str) -> None:
        idx = candidate_to_idx[label]
        for dataset in out.values():
            dataset[idx] = np.nan if dataset.dtype.kind == "f" else 0

    def fit_or_fail(label: str, y: sp.csr_matrix, rows: np.ndarray) -> tuple[str, "dict | Exception"]:
        try:
            return label, fit(label, y, rows)
        except Exception as exc:  # reported per perturbation; the run goes on
            return label, exc

    failed: list[str] = []
    n_tested_list: list[int] = []
    backed = read_backed(path)
    try:
        def tasks():
            # The rows are read here, in the caller's thread, as each task is
            # dispatched (joblib pre-dispatches two per thread), and fitted in
            # the pool: numpy releases the GIL in the fits' array work.
            for label in candidates:
                rows = group_rows[label]
                y = sp.csr_matrix(backed.X[rows, :], dtype=np.float64)
                y.sum_duplicates()
                y.eliminate_zeros()
                yield delayed(fit_or_fail)(label, y, rows)

        with _create_progress_context(n_groups, name, verbose) as pbar:
            results = Parallel(n_jobs=n_threads, backend="threading", return_as="generator_unordered")(tasks())
            for label, res in results:
                if isinstance(res, Exception):
                    logger.error(f"Failed perturbation {label}: {res}")
                    failed.append(label)
                    write_failed(label)
                else:
                    write(res)
                    n_tested_list.append(res["n_tested"])
                    _print_de_perturbation_verbose(verbose, label, res["n_tested"], n_genes)
                pbar.update(1)
    finally:
        backed.file.close()
    _print_de_summary(verbose, name, n_groups - len(failed), n_groups, n_tested_list, n_genes)
    if failed:
        _messages.warn(
            name,
            f"{len(failed)} perturbation(s) failed and are NaN in the result (each error is logged): "
            f"{failed[:5]}{'...' if len(failed) > 5 else ''}",
        )
    _messages.print_saving(verbose, name, output_path)
    return _finish_result(
        run, output_path, obs=obs, method="nb_offset", corr_method=corr_method, scanpy_format=scanpy_format,
        fingerprint=fingerprint,
    )


def _result_obs(candidates: list[str], perturbation_column: str) -> pd.DataFrame:
    """The ``obs`` of a DE result: one row per perturbation, the label as the
    index and as the ``perturbation_column`` column (the same strings, so
    anndata accepts the shared name; a list would type an empty column as
    float)."""
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame(index=obs_index)
    obs[perturbation_column] = obs_index.to_numpy()
    return obs


def _wilcoxon_resumable_result(
    output_path: Path,
    checkpoint_path: Path,
    *,
    candidates: list[str],
    gene_symbols,
    perturbation_column: str,
    control_label: str,
    tie_correct: bool,
    corr_method: str,
    resume: bool,
    fingerprint: dict,
    chunk_size: int,
    batch_column: str | None = None,
) -> ResumableRun:
    """The result file a Wilcoxon run fills one gene chunk at a time: the
    AUC in ``X``, the test's layers, and the control arm's detection rate as
    ``var["pts_rest"]``, laid out in column blocks matching those writes
    (:func:`_column_block_chunks`)."""
    var = pd.DataFrame(
        {"pts_rest": np.zeros(len(gene_symbols), dtype=np.float32)},
        index=pd.Index(gene_symbols).astype(str),
    )
    return _resumable_result(
        output_path, checkpoint_path, fingerprint=fingerprint, resume=resume,
        obs=_result_obs(candidates, perturbation_column), var=var,
        uns=_wilcoxon_uns(
            control_label=control_label, perturbation_column=perturbation_column,
            tie_correct=tie_correct, corr_method=corr_method, batch_column=batch_column,
        ),
        x_dtype=np.float64,
        layers={
            "z_score": np.float64, "pvalue": np.float64, "pvalue_adj": np.float64,
            "logfoldchanges": np.float64, "u_statistic": np.float64, "pts": np.float32,
        },
        chunks=_column_block_chunks(len(candidates), len(gene_symbols), chunk_size),
    )


def _write_wilcoxon_chunk(handle: h5py.File, slc: slice, *, auc, u, z, p, lfc, pts, pts_rest) -> None:
    """Write one gene chunk's columns of every group into a Wilcoxon result."""
    handle["X"][:, slc] = auc
    layers = handle["layers"]
    layers["u_statistic"][:, slc] = u
    layers["z_score"][:, slc] = z
    layers["pvalue"][:, slc] = p
    layers["logfoldchanges"][:, slc] = lfc
    layers["pts"][:, slc] = pts
    handle["var/pts_rest"][slc] = pts_rest


def _wilcoxon_test_stratified(
    path: Path,
    *,
    gene_symbols,
    perturbation_column: str,
    labels: np.ndarray,
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
) -> RankGenesGroupsResult:
    """Batch-stratified (van Elteren) Wilcoxon rank-sum test.

    Cells are ranked against the control **within each batch** and the
    per-stratum rank statistics are combined with unit weights.  The
    low-expression filter, log-fold changes and ``pts`` are computed on the
    pooled populations (batch-agnostic); only the rank test is stratified.

    Mirrors the single-pass path but partitions the control and perturbation
    cells by ``batch_column`` and calls the stratified kernel.
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
    run = _wilcoxon_resumable_result(
        output_path, checkpoint_path, candidates=candidates, gene_symbols=gene_symbols,
        perturbation_column=perturbation_column, control_label=control_label,
        tie_correct=tie_correct, corr_method=corr_method, resume=resume,
        fingerprint=_checkpoint_fingerprint(fingerprint, path="stratified", chunk_size=int(chunk_size)),
        chunk_size=chunk_size,
        batch_column=batch_column,
    )
    last_completed_chunk = run.progress.get("last_gene_chunk", -1)
    if run.resumed:
        _messages.vprint(verbose, "wilcoxon_test", f"Resuming at gene chunk {last_completed_chunk + 1}/{n_gene_chunks}")

    backed = read_backed(path)
    try:
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
            timer = _PhaseTimer()
            for slc, block in iter_matrix_chunks(
                backed, axis=1, chunk_size=chunk_size, convert_to_dense=False,
                start_chunk=current_chunk, warn_slow_axis=False,
            ):
                timer.lap("read")
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
                # A gene whose values are all tied gets a NaN rank test from
                # the kernel; it was not tested either.
                chunk_lfc[np.isnan(chunk_p) & rank_valid_arr] = np.nan
                if invalid_arr.any():
                    chunk_u[invalid_arr] = np.nan
                    chunk_z[invalid_arr] = np.nan
                    chunk_p[invalid_arr] = np.nan
                    chunk_effect[invalid_arr] = np.nan

                timer.lap("compute")
                _write_wilcoxon_chunk(
                    run.file, slc, auc=chunk_effect, u=chunk_u, z=chunk_z, p=chunk_p,
                    lfc=chunk_lfc, pts=chunk_pts, pts_rest=control_pts,
                )

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
                timer.lap("write")

            _save_wilcoxon_checkpoint(current_chunk - 1)
            timer.lap("write")
            _messages.vprint(
                verbose, "wilcoxon_test",
                f"{n_chunks_processed} gene chunks: {timer.summary()}",
            )
            logger.info(f"Completed {n_chunks_processed} gene chunks (stratified)")
            if int(verbose) >= 1:
                # Untested pairs are NaN in every test column; counting
                # them at the end (not per chunk) stays right across a
                # resume.
                _valid_gene_counts = _count_tested(run.file["layers/u_statistic"])
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
    return _finish_result(
        run, output_path, obs=_result_obs(candidates, perturbation_column), method="wilcoxon",
        corr_method=corr_method, scanpy_format=scanpy_format, fingerprint=fingerprint,
        uns=stratified_diagnostics,
    )


@accepts_in_memory
def wilcoxon_test(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str | pd.Series | None = None,
    groupby: str | pd.Series | None = None,
    control_label: str | None = None,
    reference: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    batch_column: str | None = None,
    min_cells_expressed: int = 0,
    min_pct_ctrl: float = 0.01,
    min_pct_pert: float = 0.002,
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
        Path to an h5ad file, or a crispyx AnnData / backed anndata.AnnData containing
        normalised, log-transformed data.
        An in-memory AnnData is streamed from a temporary copy; ``output_path``
        is then required.
    perturbation_column
        Column in `adata.obs` indicating perturbation labels, or the labels
        themselves as a ``pd.Series`` indexed by cell (``obs_names``), e.g.
        shuffled labels for a permutation null, so no relabelled copy of the
        data has to be written. The Series is aligned to ``obs_names`` by
        index, so its order does not matter, but every cell needs a label;
        its ``name`` (default ``"perturbation"``) names the column in the
        result. A different Series is a different call: a result computed
        with one is not reused for another. On a CSR input each call makes a
        temporary CSC copy (see ``format_mismatch_policy``); for many
        shuffles, convert once with :func:`crispyx.pp.convert_to_csc` and pass
        the converted file.
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
        ``score`` / ``pvalue`` / ``logfoldchanges`` and the effect size, with
        ``pts`` / ``pts_rest`` still populated. A gene whose values are all
        tied across both groups (e.g. zero in every cell, with the filters
        off) has no rank test and is untested in the same way; Scanpy
        reports ``score=0``, ``pvalue=1`` for it instead.
    min_pct_ctrl
        Minimum fraction of expressing cells for the *control* side. A gene is
        excluded only when *both* sides are jointly low. Default ``0.01``.
    min_pct_pert
        Minimum fraction of expressing cells for the *perturbed* side.
        Default ``0.002`` (lower than ctrl; induction from near-zero baseline is
        biologically valid). Combined with ``min_mean_pert`` this forms a dual
        condition more robust than pct alone.
    min_mean_ctrl
        Minimum mean log1p expression for the *control* side. Default ``0.05``.
        Excluded genes are written as NaN in ``score`` / ``pvalue`` /
        ``logfoldchanges`` / ``auc``; ``pts`` / ``pts_rest`` remain
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
        Only a checkpoint from a call with the same input and
        result-affecting parameters is used; partial results live in a hidden
        ``.<output name>.resume`` directory until the output is written.
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
        exists. If False (default), an existing result is loaded instead of
        rerun when it was computed from the same input file (path, size,
        modification time) with the same result-shaping arguments, as
        recorded in its ``uns["crispyx"]``; otherwise it is recomputed.
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
        `result[label].logfoldchanges`, `result[label].auc`,
        `result[label].pvalue`, etc. The h5ad file path is available at
        `result.result_path`.

        The h5ad file holds one row per perturbation: the AUC
        (``U / (n_pert * n_ctrl)``, 0.5 for no shift) in ``X`` and the layers
        ``z_score``, ``pvalue``, ``pvalue_adj``, ``logfoldchanges``,
        ``u_statistic`` and ``pts``. ``pts_rest``
        describes the control arm, so it is the per-gene column
        ``var["pts_rest"]``. ``obs["n_genes_tested"]`` counts the genes each
        perturbation was tested on, after the filters.

    Notes
    -----
    **Differences from** ``scanpy.tl.rank_genes_groups(method="wilcoxon")``.
    Scanpy tests every gene; crispyx applies four low-expression filters by
    default (``min_pct_ctrl=0.01``, ``min_pct_pert=0.002``,
    ``min_mean_ctrl=0.05``, ``min_mean_pert=0.005``) and reports the genes
    they exclude as untested (``NaN``), which also keeps them out of the
    multiple-testing correction. ``obs["n_genes_tested"]`` says how many
    genes each comparison kept. Setting all four to ``0`` tests Scanpy's gene
    set; a gene whose values are all tied is still untested here, where
    Scanpy reports ``score=0``, ``pvalue=1``.

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
    # A one-shot iterable would be fingerprinted by its repr and then used up.
    if perturbations is not None:
        perturbations = [str(p) for p in perturbations]
    call_args = dict(locals())  # for the resume fingerprint, before any other local

    perturbation_column, control_label = resolve_group_reference_aliases(
        perturbation_column=perturbation_column,
        groupby=groupby,
        control_label=control_label,
        reference=reference,
        fn_name="wilcoxon_test",
    )
    validate_format_mismatch_policy(format_mismatch_policy)

    path = resolve_data_path(data)
    output_suffix = "wilcoxon_stratified" if batch_column is not None else "wilcoxon"
    output_path = resolve_output_path(
        path, suffix=output_suffix, output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )

    fingerprint = _de_fingerprint(
        path, call_args, method="wilcoxon",
        perturbation_column=perturbation_column, control_label=control_label,
    )
    if (r := _try_load_existing_de_result(
        output_path, force=force, verbose=verbose,
        fingerprint=fingerprint,
    )):
        return r
    given_labels = perturbation_column if isinstance(perturbation_column, pd.Series) else None
    if given_labels is not None:
        perturbation_column = "perturbation" if given_labels.name is None else str(given_labels.name)

    backed = read_backed(path)
    try:
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if given_labels is None and perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        if batch_column is not None and batch_column not in backed.obs.columns:
            raise KeyError(
                f"Batch column '{batch_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = (
            label_strings(backed.obs[perturbation_column]) if given_labels is None
            else label_strings(cell_labels(backed.obs, given_labels)[0])
        )
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
        if chunk_size is None:
            chunk_size = calculate_wilcoxon_chunk_size(
                backed.n_obs, backed.n_vars, n_groups=len(candidates),
                available_memory_gb=memory_limit_gb,
            )
            _messages.vprint(verbose, "wilcoxon_test", f"chunk_size={chunk_size} (auto)")
    finally:
        backed.file.close()

    n_groups = len(candidates)

    # Determine output path and checkpoint path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".progress.json")

    if n_groups == 0:
        # An explicit empty selection: the empty result, with no pass over X
        # (``var["pts_rest"]`` is NaN, since the controls were not read).
        run = _wilcoxon_resumable_result(
            output_path, checkpoint_path, candidates=[], gene_symbols=gene_symbols,
            perturbation_column=perturbation_column, control_label=control_label, tie_correct=tie_correct,
            corr_method=corr_method, resume=False, fingerprint=_checkpoint_fingerprint(fingerprint),
            chunk_size=chunk_size or 1, batch_column=batch_column,
        )
        run.file["var/pts_rest"][...] = np.nan
        return _finish_result(
            run, output_path, obs=_result_obs([], perturbation_column), method="wilcoxon",
            corr_method=corr_method, scanpy_format=scanpy_format, fingerprint=fingerprint,
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
        # within-batch and combined.
        if batch_column is not None:
            return _wilcoxon_test_stratified(
                stream_path,
                gene_symbols=gene_symbols,
                perturbation_column=perturbation_column,
                labels=labels,
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
            )

        return _wilcoxon_test_standard(
            stream_path,
            gene_symbols=gene_symbols,
            perturbation_column=perturbation_column,
            labels=labels,
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
        )


def _wilcoxon_test_standard(
    path: Path,
    *,
    gene_symbols,
    perturbation_column: str,
    labels: np.ndarray,
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
) -> RankGenesGroupsResult:
    """Single-pass pooled Wilcoxon test over gene chunks.

    Each gene chunk's results for every group are written into the result
    file beside the output as the chunk finishes (a :class:`ResumableRun`, so
    an interrupted run resumes at its first unfinished gene chunk); memory
    holds one chunk, never the whole result.
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
    run = _wilcoxon_resumable_result(
        output_path, checkpoint_path, candidates=candidates, gene_symbols=gene_symbols,
        perturbation_column=perturbation_column, control_label=control_label,
        tie_correct=tie_correct, corr_method=corr_method, resume=resume,
        fingerprint=_checkpoint_fingerprint(fingerprint, path="standard", chunk_size=int(chunk_size)),
        chunk_size=chunk_size,
    )
    last_completed_chunk = run.progress.get("last_gene_chunk", -1)
    if run.resumed:
        _messages.vprint(verbose, "wilcoxon_test", f"Resuming at gene chunk {last_completed_chunk + 1}/{n_gene_chunks}")

    backed = read_backed(path)
    try:
        control_mask = labels == control_label
        control_n = int(control_mask.sum())
        # Precompute integer row indices (faster than boolean indexing
        # on the dense block: O(n_pert) vs O(n_cells) per group)
        control_idx = np.where(control_mask)[0]
        pert_idx = _group_row_indices(labels, candidates)

        # Pre-build flat perturbation indices and row offsets once
        # (avoids rebuilding inside the per-chunk stacking loop).
        all_pert_flat_idx = np.concatenate([pert_idx[label] for label in candidates] or [np.empty(0, dtype=np.intp)])
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
            timer = _PhaseTimer()
            for slc, block in iter_matrix_chunks(
                backed, axis=1, chunk_size=chunk_size, convert_to_dense=False,
                start_chunk=current_chunk, warn_slow_axis=False,
            ):
                timer.lap("read")
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

                raw_lfc = np.log2(
                    (np.expm1(all_means) + 1e-9) / control_mean_expm1[np.newaxis, :]
                )

                # Untested genes carry NaN in every column derived from the
                # comparison.  Masked after the block write: the kernel
                # leaves its defaults in the columns it skips, and a gene
                # excluded here may be tested for another perturbation.
                invalid_arr = ~valid_arr
                # A gene whose values are all tied gets a NaN rank test from
                # the kernel; it was not tested either.
                chunk_lfc[:] = np.where(valid_arr & ~np.isnan(chunk_p), raw_lfc, np.nan)
                if invalid_arr.any():
                    chunk_u[invalid_arr] = np.nan
                    chunk_z[invalid_arr] = np.nan
                    chunk_p[invalid_arr] = np.nan
                    chunk_effect[invalid_arr] = np.nan

                # 13. Write the chunk's columns into the result file
                timer.lap("compute")
                _write_wilcoxon_chunk(
                    run.file, slc, auc=chunk_effect, u=chunk_u, z=chunk_z, p=chunk_p,
                    lfc=chunk_lfc, pts=chunk_pts, pts_rest=control_pts,
                )
                    
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
                timer.lap("write")
                
            # Final checkpoint
            _save_wilcoxon_checkpoint(current_chunk - 1)
            timer.lap("write")
            _messages.vprint(
                verbose, "wilcoxon_test",
                f"{n_chunks_processed} gene chunks: {timer.summary()}",
            )
            logger.info(f"Completed {n_chunks_processed} gene chunks")
            if int(verbose) >= 1:
                # Untested pairs are NaN in every test column; counting them
                # at the end (not per chunk) stays right across a resume.
                _valid_gene_counts = _count_tested(run.file["layers/u_statistic"])
                if int(verbose) >= 2:
                    for _gi, _label in enumerate(candidates):
                        _print_de_perturbation_verbose(verbose, _label, int(_valid_gene_counts[_gi]), n_genes)
                _mean = int(_valid_gene_counts.mean()) if n_groups > 0 else 0
                _pct = 100.0 * _mean / n_genes if n_genes else 0
                print(f"[cx] Wilcoxon DE: {n_groups} perturbations complete, mean {_mean}/{n_genes} genes tested ({_pct:.0f}%)")
    finally:
        backed.file.close()

    if int(verbose) >= 1:
        print(f"[cx] wilcoxon_test: Saving \u2192 {output_path}")
    return _finish_result(
        run, output_path, obs=_result_obs(candidates, perturbation_column), method="wilcoxon",
        corr_method=corr_method, scanpy_format=scanpy_format, fingerprint=fingerprint,
    )


# Resident size of one idle loky worker (interpreter, numpy, scipy).
_APEGLM_WORKER_BASE_MB = 200.0


def _apeglm_worker_count(
    n_jobs: int, n_cells: int, batch_size: int, memory_limit_gb: float | None
) -> int:
    """Workers for apeGLM shrinkage: ``n_jobs``, capped so the workers fit in
    ``memory_limit_gb`` (or the memory available, when ``None``).

    Each task carries one gene's counts, the two-column design and the log
    size factors (4 float64 vectors of ``n_cells``); a worker holds a batch
    of ``batch_size`` tasks and works on one gene at a time (about 8 more
    vectors). joblib also keeps up to two more batches per worker queued in
    the parent (its default ``pre_dispatch="2*n_jobs"``), so each worker
    accounts for three batches in all.
    """
    per_worker_mb = _APEGLM_WORKER_BASE_MB + (3 * 4 * batch_size + 8) * n_cells * 8 / 1e6
    return min(
        _resolve_n_jobs(n_jobs),
        _estimate_max_workers(
            n_cells,
            1,
            memory_per_worker_mb=per_worker_mb,
            memory_limit_gb=memory_limit_gb,
        ),
    )


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
        The "stats" method is recommended because the NB-GLM fit floors its
        means at ``min_mu`` (``uns["min_mu"]``: 0.5 for pseudo-bulk and bulk,
        1e-6 for single cells by default), which affects the stored MLE
        coefficients. The "full" method re-evaluates the likelihood without
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
        Path to an h5ad file, or a crispyx AnnData / backed anndata.AnnData containing
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
        Optional memory budget in gigabytes. With ``method="full"``, the
        number of parallel workers is capped so that they fit in the budget
        (``None``, the default, uses the memory currently available).
        ``method="stats"`` holds no per-worker data and ignores it.
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
        - `X`: updated to the shrunken LFC
    
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

    # The NB-GLM result is read a block of rows at a time, never whole.
    with h5py.File(path, "r") as handle:
        uns = handle["uns"]
        layer_names = set(handle["layers"])
        # Validate that this is an NB-GLM result
        result_method = _uns_value(uns, "method")
        if result_method not in ("nb_glm", "nb_offset") or "standard_error" not in layer_names:
            raise ValueError(
                f"Input file '{path}' is not an nb_glm_test or nb_offset_test result "
                "(expected uns['method'] of 'nb_glm' or 'nb_offset' and a 'standard_error' layer)."
            )
        if result_method == "nb_offset" and method == "full":
            raise ValueError(
                "method='full' refits the NB-GLM on its design; an nb_offset_test result has none. "
                "Use method='stats'."
            )
        control_label = _uns_value(uns, "control_label", "control")
        perturbation_column = _uns_value(uns, "perturbation_column", "perturbation")
        lfc_base = _uns_value(uns, "lfc_base", "log2")
        size_factors_global = read_elem(uns["size_factors"]) if "size_factors" in uns else None
        original_dataset_path = _uns_value(uns, "original_dataset_path")
        candidates = [str(g) for g in _h5_index(handle["obs"])]
        gene_symbols = pd.Index([str(g) for g in _h5_index(handle["var"])])
        n_groups, n_genes = handle["X"].shape
    if "dispersion" not in layer_names:
        raise ValueError(
            f"Input file '{path}' lacks the 'dispersion' layer (results from crispyx before 0.2.3); "
            "re-run nb_glm_test."
        )
    if "intercept" not in layer_names:
        raise ValueError(
            f"Input file '{path}' lacks 'intercept' layer. "
            "method='full' requires NB-GLM results from nb_glm_test v0.5.1+. "
            "Use method='stats' or re-run nb_glm_test."
        )

    # MLE LFC in the file's base: X, unless X was already shrunk, in which
    # case nb_glm_test / shrink_lfc kept the MLE as logfoldchange_raw.
    mle_key = "layers/logfoldchange_raw" if "logfoldchange_raw" in layer_names else "X"
    # ln-scale MLE LFC and SE (apeGLM works on the ln scale). With a log2
    # base they are stored as *_ln layers; with an ln base the main layers
    # already hold them.
    if lfc_base == "ln":
        raw_key = mle_key
        se_key = "layers/standard_error_ln" if "standard_error_ln" in layer_names else "layers/standard_error"
    else:
        raw_key, se_key = "layers/logfoldchange_raw_ln", "layers/standard_error_ln"
    logger.info("Using fitted intercept from NB-GLM for shrinkage")

    if profiling and profiler is not None:
        profiler.snapshot("after_load")
        profiler.reset_peak()
        profiler.start("shrinkage")

    budget = _resolve_memory_limit_bytes(memory_limit_gb)

    def read_rows(handle: h5py.File, sl: slice) -> dict[str, np.ndarray]:
        rows = {
            "mle": handle[mle_key][sl], "raw": handle[raw_key][sl], "se": handle[se_key][sl],
            "intercept": handle["layers/intercept"][sl],
        }
        rows["dispersion"] = handle["layers/dispersion"][sl]
        return rows

    # Global prior scale from ALL perturbations' MLE LFCs: read once when the
    # pairs fit a quarter of the budget, else re-read in blocks for each
    # evaluation of the objective.
    if n_groups * n_genes * 16 <= 0.25 * budget:
        with h5py.File(path, "r") as handle:
            all_pairs = [(handle[raw_key][()], handle[se_key][()])]
        prior_blocks = lambda: all_pairs  # noqa: E731
    else:
        def prior_blocks():
            with h5py.File(path, "r") as handle:
                for sl in _row_blocks(n_groups, n_genes):
                    yield handle[raw_key][sl], handle[se_key][sl]
    global_prior_scale = _apeglm_prior_scale_from_blocks(prior_blocks)
    del prior_blocks
    logger.info(f"Global prior scale for apeGLM: {global_prior_scale:.4f}")

    total_converged = 0
    total_genes_processed = 0
    total_needs_refit = 0

    def shrink_stats(rows: dict[str, np.ndarray], base_mean_proxy) -> tuple[np.ndarray, np.ndarray]:
        nonlocal total_converged, total_genes_processed, total_needs_refit
        shrunk_lfc = np.zeros_like(rows["raw"])
        shrunk_se = np.zeros_like(rows["se"])
        for i in range(rows["raw"].shape[0]):
            mle_lfc_group, se_group = rows["raw"][i], rows["se"][i]
            if prior_scale_mode == "per_comparison":
                pert_prior_scale = _estimate_apeglm_prior_scale(mle_lfc_group, se_group)
            else:
                pert_prior_scale = global_prior_scale
            # NOTE: use_gene_specific_prior=False for PyDESeq2 parity (gene-specific
            # prior scaling is non-standard and causes accuracy regression)
            shrunk_lfc[i], shrunk_se[i], converged, needs_refit = shrink_lfc_apeglm_from_stats(
                mle_lfc=mle_lfc_group,
                mle_se=se_group,
                prior_scale=pert_prior_scale,
                base_mean=base_mean_proxy,
                use_gene_specific_prior=False,
                hybrid_fallback=True,
            )
            total_converged += int(converged.sum())
            total_genes_processed += n_genes
            total_needs_refit += int(needs_refit.sum())
        return shrunk_lfc, shrunk_se

    if method == "stats":
        # Fast stats-based shrinkage using vectorized Newton-Raphson
        # No need to load original dataset - uses pre-computed MLE stats
        logger.info("Using fast stats-based shrinkage (method='stats')")
        # base_mean proxy for the hybrid-fallback flag: intercept is
        # ln(mu_control), so exp(mean intercept across perturbations)
        # tracks each gene's expression level.
        intercept_sum = np.zeros(n_genes)
        intercept_count = np.zeros(n_genes)
        with h5py.File(path, "r") as handle:
            for sl in _row_blocks(n_groups, n_genes):
                block = handle["layers/intercept"][sl]
                finite = np.isfinite(block)
                intercept_sum += np.where(finite, block, 0.0).sum(axis=0)
                intercept_count += finite.sum(axis=0)
        if intercept_count.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_intercept = intercept_sum / intercept_count
            base_mean_proxy = np.exp(np.clip(mean_intercept, -20, 20))  # Clamp to avoid overflow
        else:
            base_mean_proxy = None
        shrink_rows = lambda rows: shrink_stats(rows, base_mean_proxy)  # noqa: E731
        backed = None
    else:  # method == "full"
        # Full model re-fitting with L-BFGS-B per gene
        logger.info("Using full model re-fitting (method='full')")
        if original_dataset_path is None:
            raise ValueError(
                f"Input file '{path}' does not have 'original_dataset_path' in uns. "
                "method='full' requires a result written by nb_glm_test. "
                "Use method='stats' or re-run nb_glm_test."
            )
        original_path = Path(original_dataset_path)
        if not original_path.exists():
            raise FileNotFoundError(
                f"Original dataset not found: {original_dataset_path} (recorded "
                f"in {path} by nb_glm_test). method='full' refits on the original "
                "counts; move the file back or use method='stats', which needs "
                "only the stored statistics."
            )
        backed = read_backed(original_path)
        labels = label_strings(backed.obs[perturbation_column])
        if size_factors_global is not None:
            size_factors_all = np.asarray(size_factors_global, dtype=np.float64)
        else:
            size_factors_all = _library_size_factors(original_path)
        control_idx = np.where(labels == control_label)[0]
        if len(control_idx) == 0:
            backed.file.close()
            raise ValueError(f"No control cells found with label '{control_label}'")
        # Load control cells once (they're reused for all perturbations)
        control_counts = backed.X[control_idx, :]
        control_counts = control_counts.toarray() if sp.issparse(control_counts) else np.asarray(control_counts)
        control_size_factors = size_factors_all[control_idx]
        pert_rows = _group_row_indices(labels, candidates)
        # Workers sized once, for the largest comparison: probing free
        # memory per perturbation would cost thousands of probes.
        largest_pert = max((len(rows) for rows in pert_rows.values()), default=0)
        apeglm_jobs = _apeglm_worker_count(
            n_jobs, len(control_idx) + largest_pert, batch_size, memory_limit_gb,
        )
        n_control = len(control_idx)

        def shrink_rows(rows: dict[str, np.ndarray], first: int) -> tuple[np.ndarray, np.ndarray]:
            shrunk_lfc = np.zeros_like(rows["raw"])
            shrunk_se = np.zeros_like(rows["se"])
            for i in range(rows["raw"].shape[0]):
                pert_label = candidates[first + i]
                pert_idx = pert_rows[pert_label]
                if len(pert_idx) == 0:
                    shrunk_lfc[i], shrunk_se[i] = rows["raw"][i], rows["se"][i]
                    continue
                pert_counts = backed.X[pert_idx, :]
                pert_counts = pert_counts.toarray() if sp.issparse(pert_counts) else np.asarray(pert_counts)
                combined_counts = np.vstack([control_counts, pert_counts])
                combined_size_factors = np.concatenate([control_size_factors, size_factors_all[pert_idx]])
                design_matrix = np.zeros((n_control + len(pert_idx), 2), dtype=np.float64)
                design_matrix[:, 0] = 1.0
                design_matrix[n_control:, 1] = 1.0
                # MLE coefficients from NB-GLM
                mle_coef = np.vstack([rows["intercept"][i], rows["raw"][i]])
                if prior_scale_mode == "per_comparison":
                    pert_prior_scale = _estimate_apeglm_prior_scale(rows["raw"][i], rows["se"][i])
                else:
                    pert_prior_scale = global_prior_scale
                # NOTE: By default (min_mu=0.0), no min_mu clamping is applied,
                # matching PyDESeq2's lfc_shrink which omits min_mu entirely.
                shrunk_coef, shrunk_se[i], converged = shrink_lfc_apeglm(
                    counts=combined_counts,
                    design_matrix=design_matrix,
                    size_factors=combined_size_factors,
                    dispersion=rows["dispersion"][i],
                    mle_coef=mle_coef,
                    mle_se=rows["se"][i],
                    shrink_index=1,
                    prior_scale=pert_prior_scale,
                    n_jobs=apeglm_jobs,
                    batch_size=batch_size,
                    min_mu=min_mu,
                )
                shrunk_lfc[i] = shrunk_coef[1, :]
            return shrunk_lfc, shrunk_se

    # Determine output path
    if data_name is None:
        # Append _shrunk to the stem
        stem = path.stem
        data_name = stem if stem.endswith("_shrunk") else f"{stem}_shrunk"
    output_dir = path.parent if output_dir is None else Path(output_dir)
    output_path = output_dir / f"{data_name}.h5ad"
    _shrink_lfc_disk_estimate = warn_if_disk_space_low(
        path.stat().st_size + estimate_bytes(n_groups, n_genes, 2),
        output_path,
        context="shrink_lfc",
    )
    _messages.print_disk_estimate(verbose, "shrink_lfc", _shrink_lfc_disk_estimate)
    _messages.print_saving(verbose, "shrink_lfc", output_path)

    # The output is the input with X shrunk: copied, then rewritten a block
    # of rows at a time. X becomes the shrunk LFC (the effect size); the MLE
    # LFC and, with an ln base, the MLE SE, which the posterior values
    # replace, are kept.
    lfc_scale = np.log(2) if lfc_base == "log2" else 1.0
    try:
        with _replace_on_success(output_path) as partial:
            shutil.copyfile(path, partial)
            drop_file_cache(path)
            with h5py.File(partial, "r+") as out:
                layers = out["layers"]
                for name in ("logfoldchange_raw", *(("standard_error_ln",) if lfc_base == "ln" else ())):
                    if name not in layers:
                        _create_unfilled_array(layers, name, (n_groups, n_genes), np.float64)
                for sl in _row_blocks(n_groups, n_genes):
                    rows = read_rows(out, sl)
                    if method == "stats":
                        shrunk_lfc, shrunk_se = shrink_rows(rows)
                    else:
                        shrunk_lfc, shrunk_se = shrink_rows(rows, sl.start)
                    layers["logfoldchange_raw"][sl] = rows["mle"]
                    if lfc_base == "ln":
                        layers["standard_error_ln"][sl] = rows["se"]
                    layers["standard_error"][sl] = shrunk_se / lfc_scale  # Posterior SE
                    out["X"][sl] = shrunk_lfc / lfc_scale
                    out.flush()
                    drop_file_cache(partial)
                uns_out = out["uns"]
                # The provenance copied from the NB-GLM input describes that
                # unshrunk result; left in place it would let nb_glm_test
                # return this file as its own.
                if _PROVENANCE_KEY in uns_out:
                    del uns_out[_PROVENANCE_KEY]
                if profiling and profiler is not None:
                    profiler.stop("shrinkage")
                    profiler.snapshot("shrinkage_end")
                    profiler.stop("total")
                    profiler.stop_sampling()
                    stats = profiler.get_stats()
                    profiling_uns = {
                        "profiling_enabled": True,
                        "shrinkage_seconds": stats.get("timing", {}).get("sections", {}).get("shrinkage", {}).get("seconds", 0.0),
                        "shrinkage_peak_memory_mb": stats.get("memory", {}).get("peak_mb", 0.0),
                        "total_seconds": stats.get("timing", {}).get("total_seconds", 0.0),
                    }
                else:
                    profiling_uns = "NA"
                for key, value in {
                    "lfc_shrinkage_type": "apeglm",
                    "apeglm_prior_scale": global_prior_scale,
                    "shrinkage_method": method,
                    "prior_scale_mode": prior_scale_mode,
                    "profiling": profiling_uns,
                }.items():
                    if key in uns_out:
                        del uns_out[key]
                    write_elem(uns_out, key, value)
            drop_file_cache(partial)
    finally:
        if backed is not None:
            backed.file.close()

    # Log warning if many genes didn't converge
    convergence_rate = total_converged / total_genes_processed if total_genes_processed > 0 else 1.0
    if method == "stats" and convergence_rate < 0.95:
        logger.warning(
            f"Low convergence rate: {total_converged}/{total_genes_processed} "
            f"({convergence_rate:.1%}) genes converged within max_iter. "
            "Consider using method='full' for better accuracy."
        )
    if total_needs_refit > 0:
        logger.info(
            f"Hybrid fallback: {total_needs_refit} gene-perturbation pairs "
            f"({total_needs_refit / (n_groups * n_genes):.1%}) flagged for potential accuracy improvement with method='full'"
        )
    _messages.print_done(verbose, "shrink_lfc", f"{n_groups} perturbations × {n_genes} genes (method={method!r})")
    logger.info(
        f"Applied apeGLM LFC shrinkage (method={method}, prior_scale_mode={prior_scale_mode}) "
        f"to {n_groups} perturbations, {n_genes} genes (prior_scale={global_prior_scale:.4f}). "
        f"Output: {output_path}"
    )
    return RankGenesGroupsResult(output_path)


@dataclass(frozen=True)
class ShrunkLFC:
    """:func:`shrink_lfc_arrays`'s result: the shrunk fold changes, their
    posterior standard errors (the MLE's where an entry was excluded) and the
    Cauchy prior scale used."""

    lfc: np.ndarray
    se: np.ndarray
    prior_scale: float


def shrink_lfc_arrays(
    lfc: ArrayLike,
    se: ArrayLike,
    *,
    prior_scale: float | None = None,
    exclude: ArrayLike | None = None,
    floor: ArrayLike | None = None,
) -> ShrunkLFC:
    """apeGLM shrinkage of fold changes given as arrays, exactly.

    The array-level form of :func:`shrink_lfc`, for effects that did not
    come from one ``nb_glm_test`` file: several results combined, a subset,
    or another estimator's output with a standard error. Each entry's
    posterior mode under a Cauchy prior of scale ``prior_scale`` on the
    natural-log fold change is :func:`~crispyx.glm.cauchy_map`, in closed
    form; the prior scale, when not given, is PyDESeq2's estimate from every
    finite entry that is not excluded (``apeglm_prior_scale`` in a
    ``shrink_lfc`` result). The posterior standard error is one over the
    root of the curvature at the mode, as in ``shrink_lfc``.

    Parameters
    ----------
    lfc, se
        Natural-log fold changes and their standard errors, arrays of one
        shape (``nb_glm_test`` with ``lfc_base="ln"``, or the ``*_ln``
        layers of a log2 result). An entry with a non-finite value in
        either is NaN in the result; ``se == 0`` is returned unshrunk.
    prior_scale
        The Cauchy prior's scale; None estimates it from the data.
    exclude
        Boolean mask of entries kept as given (after ``floor``) and left out
        of the prior-scale estimate, such as each perturbation's own target
        gene, where a knockdown is expected and a prior centred on zero is
        wrong.
    floor
        A lower bound applied to ``lfc`` before anything else, broadcast to
        its shape (NaN where none). A fold change at the fitted-mean floor
        (every count zero in the perturbed cells: -7 on pseudo-bulk with
        ``min_mu=0.01``, -12 to -16 on cells with 1e-6) is a bound, not a
        measurement; ``ln(0.5 / expected count)``, the strongest knockdown
        zero counts can show, is a sensible floor for an excluded target.

    Returns
    -------
    ShrunkLFC
        ``lfc``, ``se`` and ``prior_scale``.
    """
    lfc = np.array(lfc, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    if se.shape != lfc.shape:
        raise ValueError(f"shrink_lfc_arrays: se has shape {se.shape}, lfc {lfc.shape}")
    if floor is not None:
        bound = np.broadcast_to(np.asarray(floor, dtype=np.float64), lfc.shape)
        lfc = np.where(np.isfinite(bound), np.maximum(lfc, bound), lfc)
    excluded = (np.zeros(lfc.shape, dtype=bool) if exclude is None
                else np.broadcast_to(np.asarray(exclude, dtype=bool), lfc.shape))
    valid = np.isfinite(lfc) & np.isfinite(se) & (se >= 0)
    if prior_scale is None:
        pick = valid & ~excluded & (se > 0)
        prior_scale = _estimate_apeglm_prior_scale(lfc[pick], se[pick]) if pick.any() else 1.0
    prior_scale = float(prior_scale)
    mode = cauchy_map(lfc, se, prior_scale)
    out_lfc = np.where(valid, np.where(excluded, lfc, mode), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = prior_scale ** 2 + out_lfc ** 2
        curvature = 1.0 / se ** 2 + 2 * (prior_scale ** 2 - out_lfc ** 2) / denominator ** 2
        post_se = 1.0 / np.sqrt(np.maximum(np.abs(curvature), 1e-12))
    out_se = np.where(valid, np.where(excluded | (se == 0), se, post_se), np.nan)
    return ShrunkLFC(lfc=out_lfc, se=out_se, prior_scale=prior_scale)


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
        labels = label_strings(backed.obs[perturbation_column])
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
        labels = label_strings(backed.obs[perturbation_column])
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
    perturbation_column: str | pd.Series,
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
        labels = (
            label_strings(cell_labels(backed.obs, perturbation_column)[0])
            if isinstance(perturbation_column, pd.Series)
            else label_strings(backed.obs[perturbation_column])
        )
        control_label = resolve_control_label(labels, control_label)
        candidates = _resolve_candidates(labels, control_label, perturbations)
        n_genes = backed.n_vars
        if chunk_size is None:
            # Same resolution the real call makes, so the scratch decision
            # below sees the chunk count the run will actually stream.
            chunk_size = calculate_wilcoxon_chunk_size(
                backed.n_obs, backed.n_vars, n_groups=len(candidates),
                available_memory_gb=memory_limit_gb,
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
        # AnnData 0.13 exposes X as layers[None]; count it once, via the + 1.
        n_layers = sum(key is not None for key in backed.layers.keys()) + 1
    finally:
        backed.file.close()
    return {"output": estimate_bytes(n_obs, n_vars, n_layers, overhead=1.10)}
