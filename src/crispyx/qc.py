"""Quality control utilities for large ``.h5ad`` datasets."""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _messages
from ._memory import _EvictEvery, _resolve_memory_limit_bytes, drop_file_cache
from ._checkpoint import _create_progress_context
from ._disk import estimate_bytes, warn_if_disk_space_low
from .data import (
    accepts_in_memory,
    h5ad_data_bytes,
    AnnData,
    _ensure_csr,
    _replace_on_success,
    check_sparse_layout,
    detect_gene_symbol_column,
    calculate_optimal_chunk_size,
    ensure_gene_symbol_column,
    get_matrix_storage_format,
    is_dense_storage,
    iter_matrix_chunks,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
    write_filtered_subset,
    load_obs,
    load_var,
    write_obs,
)

logger = logging.getLogger(__name__)


@dataclass
class _CellFilterResult:
    """Result of cell filtering with both cell and gene statistics."""
    
    cell_mask: np.ndarray
    gene_counts_per_cell: np.ndarray
    gene_cell_counts_all: np.ndarray  # cells per gene for ALL cells (before perturbation filter)


class _ChunkCache:
    """In-memory cache for CSR chunk data during QC.
    
    Stores CSR chunks in memory during the gene filtering pass, then
    streams them to the write phase without re-reading the original matrix.
    
    For very large datasets, consider using use_chunk_cache=False to avoid
    memory overhead.
    
    Parameters
    ----------
    output_path
        Base path for the output file (not used for caching, kept for API compat).
    """
    
    def __init__(self, output_path: Path | str) -> None:
        self.output_path = Path(output_path)
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (data, indices, indptr_diff)
        self._n_cols: int = 0
    
    def write_chunk(
        self,
        chunk_idx: int,
        data: np.ndarray,
        indices: np.ndarray,
        indptr_diff: np.ndarray,
        n_cols: int,
    ) -> None:
        """Store CSR chunk data in memory."""
        # Ensure list is large enough
        while len(self._chunks) <= chunk_idx:
            self._chunks.append(None)
        self._chunks[chunk_idx] = (data, indices, indptr_diff)
        self._n_cols = n_cols
    
    def iter_filtered_chunks(
        self,
        gene_indices: np.ndarray,
        data_dtype: np.dtype,
    ) -> Iterable[tuple[np.ndarray, np.ndarray, int]]:
        """Iterate through cached chunks, yielding filtered CSR data.
        
        Uses vectorized operations for efficiency.
        
        Parameters
        ----------
        gene_indices
            Indices of genes to keep.
        data_dtype
            Target dtype for data array.
            
        Yields
        ------
        tuple
            (filtered_data, filtered_indices, n_cells_in_chunk)
        """
        # Build vectorized index remapping
        gene_set = set(gene_indices.tolist())
        # Create a dense lookup array: old_idx -> new_idx (or -1 if not kept)
        remap = np.full(self._n_cols, -1, dtype=np.int32)
        remap[gene_indices] = np.arange(len(gene_indices), dtype=np.int32)
        
        for chunk in self._chunks:
            if chunk is None:
                continue
            data, indices, indptr_diff = chunk
            n_cells = len(indptr_diff)
            
            # Vectorized filtering: find which entries have kept genes
            new_col_indices = remap[indices]  # -1 for dropped genes
            keep_mask = new_col_indices >= 0
            
            # Filter data and indices
            filtered_data = data[keep_mask].astype(data_dtype, copy=False)
            filtered_indices = new_col_indices[keep_mask]
            
            yield filtered_data, filtered_indices, n_cells
    
    def cleanup(self) -> None:
        """Clear cached data from memory."""
        self._chunks.clear()
    
    @property
    def chunk_count(self) -> int:
        """Number of chunks cached."""
        return len(self._chunks)


class _DiskChunkCache:
    """On-disk cache for CSR chunk data during QC streaming.

    Chunks are appended to two flat files (values and column indices) in a
    directory under ``$TMPDIR`` -- node-local disk on a cluster, where a
    network filesystem read the cache back 1.75x slower end to end -- then
    read back one chunk at a time for the write phase. The files are written
    and read with plain I/O, not mapped, and every GB of traffic their pages
    are evicted (:class:`~crispyx._memory._EvictEvery`), so
    the cache -- as large as the kept cells' matrix -- never accumulates in
    the page cache a cgroup-limited job is charged for.

    Parameters
    ----------
    output_path
        Path of the QC output.
    estimated_nnz
        Estimated total non-zeros, for the low-disk-space warning.
    data_dtype
        Data type for values array.
    """

    def __init__(
        self,
        output_path: Path | str,
        estimated_nnz: int,
        data_dtype: np.dtype = np.float32,
    ) -> None:
        self.output_path = Path(output_path)
        self._data_dtype = np.dtype(data_dtype)
        self._cache_dir = Path(tempfile.mkdtemp(prefix="crispyx_qc_cache_"))
        warn_if_disk_space_low(
            estimate_bytes(estimated_nnz, itemsize=self._data_dtype.itemsize)
            + estimate_bytes(estimated_nnz, itemsize=4),
            self._cache_dir,
            context="qc chunk cache",
        )
        self._data_path = self._cache_dir / "data.bin"
        self._indices_path = self._cache_dir / "indices.bin"
        self._data_file = open(self._data_path, "wb")
        self._indices_file = open(self._indices_path, "wb")
        # Track chunk boundaries: list of (start, end, indptr_diff)
        self._chunk_info: list[tuple[int, int, np.ndarray]] = []
        self._current_offset = 0
        self._n_cols: int = 0
        self._evict = _EvictEvery(self._data_path, self._indices_path)

    def write_chunk(
        self,
        chunk_idx: int,
        data: np.ndarray,
        indices: np.ndarray,
        indptr_diff: np.ndarray,
        n_cols: int,
    ) -> None:
        """Append CSR chunk data to the cache files."""
        start = self._current_offset
        end = start + len(data)
        np.ascontiguousarray(data, dtype=self._data_dtype).tofile(self._data_file)
        np.ascontiguousarray(indices, dtype=np.int32).tofile(self._indices_file)
        # Store chunk metadata (indptr_diff is small, keep in memory)
        self._chunk_info.append((start, end, indptr_diff.copy()))
        self._current_offset = end
        self._n_cols = n_cols
        self._evict(len(data) * (self._data_dtype.itemsize + 4))

    def iter_filtered_chunks(
        self,
        gene_indices: np.ndarray,
        data_dtype: np.dtype,
    ) -> Iterable[tuple[np.ndarray, np.ndarray, int]]:
        """Iterate cached chunks, filtering genes on-the-fly.

        Parameters
        ----------
        gene_indices
            Indices of genes to keep.
        data_dtype
            Target dtype for data array.

        Yields
        ------
        tuple
            (filtered_data, filtered_indices, n_cells_in_chunk)
        """
        for handle in (self._data_file, self._indices_file):
            handle.close()
        self._data_file = self._indices_file = None
        drop_file_cache(self._data_path)
        drop_file_cache(self._indices_path)
        # Build gene index remapping
        remap = np.full(self._n_cols, -1, dtype=np.int32)
        remap[gene_indices] = np.arange(len(gene_indices), dtype=np.int32)

        itemsize = self._data_dtype.itemsize
        with open(self._data_path, "rb") as data_file, open(self._indices_path, "rb") as indices_file:
            for start, end, indptr_diff in self._chunk_info:
                n_cells = len(indptr_diff)
                count = end - start
                if count == 0:
                    yield np.array([], dtype=data_dtype), np.array([], dtype=np.int32), n_cells
                    continue
                data = np.fromfile(data_file, dtype=self._data_dtype, count=count, offset=0)
                indices = np.fromfile(indices_file, dtype=np.int32, count=count, offset=0)
                self._evict(count * (itemsize + 4))

                # Filter genes
                new_indices = remap[indices]
                keep_mask = new_indices >= 0

                yield data[keep_mask].astype(data_dtype, copy=False), new_indices[keep_mask], n_cells

    def cleanup(self) -> None:
        """Delete the cache files and their directory."""
        for handle in (self._data_file, self._indices_file):
            if handle is not None:
                handle.close()
        self._data_file = self._indices_file = None
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    @property
    def chunk_count(self) -> int:
        """Number of chunks cached."""
        return len(self._chunk_info)


@dataclass
class QualityControlResult:
    """Result of quality control filtering."""

    cell_mask: np.ndarray
    gene_mask: np.ndarray
    perturbation_keep: Dict[str, bool]
    filtered: AnnData | None  # None if output_dir was not provided
    cell_gene_counts: np.ndarray
    gene_cell_counts: np.ndarray

    @property
    def filtered_path(self) -> Path | None:
        """Compatibility accessor exposing the on-disk filename.

        Returns None if no output file was written (output_dir was None).
        """
        if self.filtered is None:
            return None
        return self.filtered.path


def _warn_if_heavily_filtered(
    context: str, kept: int, total: int, *, kind: str, warn_below: float = 0.5,
) -> None:
    """Warn when a filtering step keeps less than ``warn_below`` of its input.

    Independent of ``verbose`` -- like every other warning in the package,
    this signals something the caller should know about regardless of
    verbosity, not something tied to how much routine progress output they
    asked for.
    """
    if total > 0 and kept < total * warn_below:
        _messages.warn(
            context,
            f"only {kept}/{total} {kind} ({100 * kept / total:.0f}%) passed "
            f"filtering. Check your thresholds if this is unexpected.",
        )


@accepts_in_memory
def filter_cells_by_gene_count(
    data: str | Path | AnnData | ad.AnnData,
    *,
    min_genes: int = 100,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    return_counts: bool = False,
    return_full_result: bool = False,
    verbose: int | bool = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray] | _CellFilterResult:
    """Return a boolean mask selecting cells with at least ``min_genes`` expressed genes.

    This function can optionally compute both genes-per-cell (row nnz) AND
    cells-per-gene (column nnz) in a single matrix pass, avoiding a separate
    gene counting pass later.

    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx AnnData / backed anndata.AnnData.
        An in-memory AnnData is streamed from a temporary copy.
    min_genes
        Minimum number of expressed genes per cell.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    return_counts
        If True, return (mask, counts) tuple instead of just mask.
        Ignored if return_full_result is True.
    return_full_result
        If True, return a _CellFilterResult containing cell_mask,
        gene_counts_per_cell, and gene_cell_counts_all (cells per gene
        for all cells, before any perturbation filtering).
    verbose
        Print how many cells passed filtering, and warn if fewer than half did.

    Returns
    -------
    mask or (mask, counts) or _CellFilterResult
        Boolean mask, optionally with counts, or full result dataclass.
    """
    path = resolve_data_path(data)
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        
        gene_counts_per_cell = np.zeros(n_obs, dtype=np.int64)
        
        # Only compute cells-per-gene if full result requested
        if return_full_result:
            gene_cell_counts_all = np.zeros(n_vars, dtype=np.int64)
        else:
            gene_cell_counts_all = None
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            if sp.issparse(block):
                gene_counts_per_cell[slc] = np.asarray(block.getnnz(axis=1)).ravel()
                if gene_cell_counts_all is not None:
                    gene_cell_counts_all += np.asarray(block.getnnz(axis=0)).ravel()
            else:
                gene_counts_per_cell[slc] = np.count_nonzero(block, axis=1)
                if gene_cell_counts_all is not None:
                    gene_cell_counts_all += np.count_nonzero(block, axis=0)
    finally:
        backed.file.close()
    
    mask = gene_counts_per_cell >= min_genes
    kept = int(mask.sum())
    if n_obs > 0:
        _messages.print_done(
            verbose, "pp.filter_cells", f"{kept}/{n_obs} cells kept ({100 * kept / n_obs:.0f}%)",
        )
        _warn_if_heavily_filtered("pp.filter_cells", kept, n_obs, kind="cells")

    if return_full_result:
        return _CellFilterResult(
            cell_mask=mask,
            gene_counts_per_cell=gene_counts_per_cell,
            gene_cell_counts_all=gene_cell_counts_all,
        )

    if return_counts:
        return mask, gene_counts_per_cell
    return mask


@accepts_in_memory
def filter_perturbations_by_cell_count(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    min_cells: int = 50,
    base_mask: np.ndarray | None = None,
    return_counts: bool = False,
    verbose: int | bool = True,
) -> np.ndarray | Tuple[np.ndarray, pd.Series]:
    """Return a mask keeping cells whose perturbation has sufficient representation.

    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx AnnData / backed anndata.AnnData.
        An in-memory AnnData is streamed from a temporary copy.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Label identifying control cells. If None, auto-detected.
    min_cells
        Minimum number of cells required per perturbation.
    base_mask
        Optional mask for cells to consider (e.g., from prior filtering).
    return_counts
        If True, return (mask, cell_counts_per_perturbation) tuple.
    verbose
        Print how many perturbations passed filtering, and warn if fewer
        than half did.

    Returns
    -------
    mask or (mask, counts)
        Boolean mask, optionally with cell counts per perturbation label.
    """
    path = resolve_data_path(data)
    backed = read_backed(path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
    finally:
        backed.file.close()

    if base_mask is None:
        base_mask = np.ones_like(labels, dtype=bool)

    # Vectorized implementation: count cells per perturbation among base_mask cells
    label_series = pd.Series(labels)
    counts = label_series[base_mask].value_counts()
    
    # Map counts back to each cell (vectorized lookup)
    count_per_cell = label_series.map(counts).fillna(0).to_numpy()
    
    # Keep cell if: (is control) OR (has enough cells AND passes base_mask)
    is_control = labels == control_label
    has_enough_cells = count_per_cell >= min_cells
    mask = (is_control | has_enough_cells) & base_mask

    total_perturbations = int(counts.size)
    if total_perturbations > 0:
        qualifying = set(counts[counts >= min_cells].index) | {control_label}
        kept_perturbations = len(qualifying & set(counts.index))
        _messages.print_done(
            verbose, "pp.filter_perturbations",
            f"{kept_perturbations}/{total_perturbations} perturbations kept "
            f"({100 * kept_perturbations / total_perturbations:.0f}%)",
        )
        _warn_if_heavily_filtered(
            "pp.filter_perturbations", kept_perturbations, total_perturbations, kind="perturbations",
        )

    if return_counts:
        return mask, counts
    return mask


@accepts_in_memory
def filter_genes_by_cell_count(
    data: str | Path | AnnData | ad.AnnData,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray | None = None,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    return_counts: bool = False,
    verbose: int | bool = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Return a boolean mask selecting genes expressed in at least ``min_cells`` cells.

    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx AnnData / backed anndata.AnnData.
        An in-memory AnnData is streamed from a temporary copy.
    min_cells
        Minimum number of cells expressing each gene.
    cell_mask
        Optional mask for cells to consider.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    return_counts
        If True, return (mask, counts) tuple instead of just mask.
    verbose
        Print how many genes passed filtering, and warn if fewer than half did.

    Returns
    -------
    mask or (mask, counts)
        Boolean mask, optionally with the raw cell counts per gene.
    """
    path = resolve_data_path(data)
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        counts = np.zeros(backed.n_vars, dtype=np.int64)
        if cell_mask is None:
            cell_mask = np.ones(backed.n_obs, dtype=bool)
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_mask = cell_mask[slc]
            if not np.any(local_mask):
                continue
            selected = block[local_mask]
            if sp.issparse(selected):
                counts += np.asarray(selected.getnnz(axis=0)).ravel()
            else:
                counts += np.count_nonzero(selected, axis=0)
    finally:
        backed.file.close()
    
    mask = counts >= min_cells
    n_vars = int(counts.size)
    if n_vars > 0:
        kept = int(mask.sum())
        _messages.print_done(
            verbose, "pp.filter_genes", f"{kept}/{n_vars} genes kept ({100 * kept / n_vars:.0f}%)",
        )
        _warn_if_heavily_filtered("pp.filter_genes", kept, n_vars, kind="genes")

    if return_counts:
        return mask, counts
    return mask


# Type alias for chunk cache (in memory or on disk)
_ChunkCacheType = Union[_ChunkCache, _DiskChunkCache, None]


@dataclass
class _GeneFilterResult:
    """Result of fused gene filtering and nnz counting."""
    
    gene_mask: np.ndarray
    gene_cell_counts: np.ndarray
    row_nnz: np.ndarray
    total_nnz: int
    data_dtype: np.dtype
    chunk_cache: _ChunkCacheType = None  # Optional cache for write phase (memory or disk)


def _compute_gene_count_delta(
    path: str | Path,
    *,
    removed_cell_mask: np.ndarray,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Compute gene counts for removed cells only (for delta adjustment).
    
    When perturbation filtering removes cells that passed the gene-count filter,
    we need to adjust the all-cell gene counts by subtracting counts from
    removed cells. This function iterates only the removed cells.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    removed_cell_mask
        Boolean mask where True indicates cells to count (i.e., cells that
        passed gene filter but were removed by perturbation filter).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
        
    Returns
    -------
    np.ndarray
        Gene counts for the removed cells only. Subtract from all-cell counts
        to get counts for filtered cells.
    """
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_vars = backed.n_vars
        delta_counts = np.zeros(n_vars, dtype=np.int64)
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_mask = removed_cell_mask[slc]
            if not np.any(local_mask):
                continue
            
            selected = block[local_mask]
            if sp.issparse(selected):
                delta_counts += np.asarray(selected.getnnz(axis=0)).ravel()
            else:
                delta_counts += np.count_nonzero(selected, axis=0)
    finally:
        backed.file.close()
    
    return delta_counts


def _filter_genes_with_cache(
    path: str | Path,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray,
    gene_cell_counts: np.ndarray,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    output_path: Path | None = None,
    cache_mode: Literal['memory', 'disk', 'none'] = 'disk',
    verbose: int | bool = True,
) -> _GeneFilterResult:
    """Compute gene mask and cache CSR data in a single matrix pass.
    
    This function does a single matrix pass that:
    1. Caches CSR chunk data (data, indices, indptr_diff) to memory or disk
    2. Uses pre-computed gene_cell_counts to determine gene_mask
    3. Computes row_nnz and total_nnz from cached data
    
    By caching CSR data during this pass, the write phase can read from
    cache instead of re-reading the original matrix, reducing total passes
    from 4 to 2.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_cells
        Minimum number of cells expressing each gene.
    cell_mask
        Boolean mask for cells to include (from prior cell filtering).
    gene_cell_counts
        Pre-computed cells per gene for filtered cells.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    output_path
        Path for output file. Required if cache_mode is not 'none'.
    cache_mode
        Cache strategy: 'memory' (fast, high RAM), 'disk' (low RAM, a file
        beside the output), or 'none' (no caching, requires re-reading source
        during write). Default is 'disk' for better memory efficiency.
        
    Returns
    -------
    _GeneFilterResult
        Dataclass containing gene_mask, gene_cell_counts, row_nnz, total_nnz,
        data_dtype, and optionally chunk_cache for the write phase.
    """
    # Compute gene mask from pre-computed counts
    gene_mask = gene_cell_counts >= min_cells
    gene_indices = np.flatnonzero(gene_mask)
    
    # Estimate nnz for the disk cache's space check (from file size heuristic)
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_vars = backed.n_vars
        n_obs = backed.n_obs
        n_filtered_cells = int(cell_mask.sum())
        
        # Estimate nnz based on filtered cells ratio
        # We cache FULL CSR data (before gene filtering) so estimate from total nnz
        try:
            # Try to get nnz from backed sparse dataset
            if hasattr(backed.X, 'group') and 'data' in backed.X.group:
                # Backed sparse: access nnz from HDF5 data array
                total_file_nnz = len(backed.X.group['data'])
                estimated_nnz = int(total_file_nnz * (n_filtered_cells / n_obs) * 1.2)
            elif sp.issparse(backed.X):
                # In-memory sparse (shouldn't happen for large files)
                total_file_nnz = backed.X.nnz
                estimated_nnz = int(total_file_nnz * (n_filtered_cells / n_obs) * 1.2)
            else:
                # Dense: estimate ~10% non-zero (typical for scRNA-seq)
                estimated_nnz = int(n_filtered_cells * n_vars * 0.1 * 1.2)
        except Exception:
            # Fallback: assume ~10% density
            estimated_nnz = int(n_filtered_cells * n_vars * 0.1 * 1.2)
    finally:
        backed.file.close()
    
    # Create cache based on mode
    chunk_cache: _ChunkCacheType = None
    if output_path is not None and cache_mode != 'none':
        if cache_mode == 'memory':
            chunk_cache = _ChunkCache(output_path)
        elif cache_mode == 'disk':
            # Detect data dtype from file for the cache
            backed = read_backed(path)
            try:
                if sp.issparse(backed.X):
                    data_dtype_hint = backed.X.dtype
                else:
                    data_dtype_hint = backed.X.dtype
            finally:
                backed.file.close()
            chunk_cache = _DiskChunkCache(output_path, estimated_nnz, data_dtype_hint)
    
    backed = read_backed(path)
    try:
        row_nnz = np.zeros(n_filtered_cells, dtype=np.int64)
        total_nnz = 0
        data_dtype: np.dtype | None = None
        row_offset = 0
        chunk_idx = 0
        n_chunks = (n_obs + chunk_size - 1) // chunk_size

        with _create_progress_context(n_chunks, "pp.qc_summary (gene filter)", verbose, unit="chunk") as pbar:
            for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
                local_cell_mask = cell_mask[slc]
                if not np.any(local_cell_mask):
                    pbar.update(1)
                    continue

                selected = block[local_cell_mask]
                csr = _ensure_csr(selected)

                # Cache the full CSR data for this chunk (before gene filtering)
                if chunk_cache is not None:
                    chunk_cache.write_chunk(
                        chunk_idx,
                        data=csr.data.copy(),
                        indices=csr.indices.copy(),
                        indptr_diff=np.diff(csr.indptr),
                        n_cols=n_vars,
                    )

                # Apply gene mask and compute nnz
                if gene_indices.size:
                    filtered = csr[:, gene_indices]
                else:
                    filtered = csr[:, []]

                filtered_csr = _ensure_csr(filtered)
                counts = np.diff(filtered_csr.indptr)
                size = counts.size
                row_nnz[row_offset : row_offset + size] = counts
                total_nnz += int(filtered_csr.nnz)

                if data_dtype is None and csr.nnz:
                    data_dtype = csr.data.dtype

                row_offset += size
                chunk_idx += 1
                pbar.update(1)
    except BaseException:
        # The caller cleans the cache up only once it has been handed back;
        # a failure while filling it would otherwise leak a cache the size of
        # the kept cells' matrix in $TMPDIR.
        if chunk_cache is not None:
            chunk_cache.cleanup()
        raise
    finally:
        backed.file.close()
    
    if data_dtype is None:
        data_dtype = np.float32
    
    return _GeneFilterResult(
        gene_mask=gene_mask,
        gene_cell_counts=gene_cell_counts,
        row_nnz=row_nnz,
        total_nnz=total_nnz,
        data_dtype=data_dtype,
        chunk_cache=chunk_cache,
    )


def _filter_genes_dense_optimized(
    path: str | Path,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray,
    gene_cell_counts: np.ndarray,
    gene_mask: np.ndarray,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    verbose: int | bool = True,
) -> _GeneFilterResult:
    """Optimized gene filtering for dense-stored datasets.
    
    This function avoids expensive CSR conversion by directly computing
    row nnz from dense blocks using vectorized numpy operations. For
    datasets stored as dense arrays (encoding-type='array'), this is
    significantly faster than the cache-based approach.
    
    Unlike _filter_genes_with_cache, this does NOT cache data because:
    1. Dense→CSR conversion is the bottleneck we're avoiding
    2. Re-reading dense data in write phase is faster than conversion
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_cells
        Minimum number of cells expressing each gene.
    cell_mask
        Boolean mask for cells to include (from prior cell filtering).
    gene_cell_counts
        Pre-computed cells per gene for filtered cells.
    gene_mask
        Pre-computed gene mask (genes with >= min_cells).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
        
    Returns
    -------
    _GeneFilterResult
        Dataclass containing gene_mask, gene_cell_counts, row_nnz, total_nnz,
        data_dtype. chunk_cache is always None for dense path.
    """
    gene_indices = np.flatnonzero(gene_mask)
    
    backed = read_backed(path)
    try:
        ensure_gene_symbol_column(backed, gene_name_column)
        n_filtered_cells = int(cell_mask.sum())
        row_nnz = np.zeros(n_filtered_cells, dtype=np.int64)
        total_nnz = 0
        data_dtype: np.dtype | None = None
        row_offset = 0
        n_chunks = (backed.n_obs + chunk_size - 1) // chunk_size

        with _create_progress_context(n_chunks, "pp.qc_summary (gene filter)", verbose, unit="chunk") as pbar:
            for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
                local_cell_mask = cell_mask[slc]
                if not np.any(local_cell_mask):
                    pbar.update(1)
                    continue

                selected = block[local_cell_mask]

                # Apply gene filter
                if gene_indices.size:
                    filtered = selected[:, gene_indices]
                else:
                    filtered = selected[:, []]

                # Vectorized nnz counting - works for both dense and sparse
                if sp.issparse(filtered):
                    counts = np.asarray(filtered.getnnz(axis=1)).ravel()
                    chunk_nnz = int(filtered.nnz)
                    if data_dtype is None and filtered.nnz:
                        data_dtype = filtered.data.dtype
                else:
                    # Dense: count non-zeros per row without CSR conversion
                    counts = np.count_nonzero(filtered, axis=1)
                    chunk_nnz = int(counts.sum())
                    if data_dtype is None:
                        data_dtype = filtered.dtype

                size = counts.size
                row_nnz[row_offset : row_offset + size] = counts
                total_nnz += chunk_nnz
                row_offset += size
                pbar.update(1)
    finally:
        backed.file.close()
    
    if data_dtype is None:
        data_dtype = np.float32
    
    return _GeneFilterResult(
        gene_mask=gene_mask,
        gene_cell_counts=gene_cell_counts,
        row_nnz=row_nnz,
        total_nnz=total_nnz,
        data_dtype=data_dtype,
        chunk_cache=None,  # No caching for dense path
    )


def _qc_in_memory(
    path: str | Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None,
    output_path: Path,
) -> QualityControlResult:
    """Fast in-memory QC for small datasets (Option A).
    
    Loads entire dataset into memory, processes like Scanpy, and saves.
    This is the fastest approach for datasets that fit in RAM.
    
    Memory-optimized: Uses in-place slicing with a single copy at the end,
    matching Scanpy's memory efficiency.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum genes per cell.
    min_cells_per_perturbation
        Minimum cells per perturbation.
    min_cells_per_gene
        Minimum cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Control label (already resolved).
    gene_name_column
        Column in var containing gene names.
    output_path
        Path for output h5ad file.
        
    Returns
    -------
    QualityControlResult
        QC result with masks and filtered AnnData.
    """
    logger.debug("Using in-memory QC path (small dataset)")
    
    # Load entire dataset
    adata = ad.read_h5ad(path)
    original_n_obs = adata.n_obs
    original_n_vars = adata.n_vars
    original_obs_names = adata.obs_names.to_numpy()
    original_var_names = adata.var_names.to_numpy()
    
    # Convert to CSR if needed (handles CSC)
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
    
    # Get gene names before any filtering
    gene_names = ensure_gene_symbol_column(adata, gene_name_column)
    
    # Compute gene counts per cell before filtering
    if sp.issparse(adata.X):
        gene_counts_per_cell = np.asarray(adata.X.getnnz(axis=1)).ravel()
    else:
        gene_counts_per_cell = np.count_nonzero(adata.X, axis=1)
    
    # ===== Build combined cell mask (avoid intermediate copies) =====
    # Step 1: Gene count filter
    cell_mask_genes = gene_counts_per_cell >= min_genes
    
    # Step 2: Perturbation filter - compute on original data with gene mask
    labels_full = adata.obs[perturbation_column].astype(str).to_numpy()
    # Only count cells that pass gene filter
    labels_passing = labels_full.copy()
    labels_passing[~cell_mask_genes] = '__FILTERED__'
    pert_counts = Counter(labels_passing)
    del pert_counts['__FILTERED__']  # Remove placeholder
    
    # Build perturbation mask on original cells
    cell_mask_pert = np.array([
        cell_mask_genes[i] and (
            labels_full[i] == control_label or 
            pert_counts.get(labels_full[i], 0) >= min_cells_per_perturbation
        )
        for i in range(len(labels_full))
    ])
    
    # Combined cell mask
    combined_cell_mask = cell_mask_pert  # Already includes gene filter
    
    # ===== Compute gene stats on filtered cells only =====
    # Create a view (not copy) for gene stats computation
    X_filtered_cells = adata.X[combined_cell_mask]
    if sp.issparse(X_filtered_cells):
        gene_cell_counts = np.asarray(X_filtered_cells.getnnz(axis=0)).ravel()
    else:
        gene_cell_counts = np.count_nonzero(X_filtered_cells, axis=0)
    del X_filtered_cells
    
    # Build gene mask
    gene_mask = gene_cell_counts >= min_cells_per_gene
    
    # ===== Single copy at the end (like Scanpy) =====
    adata_filtered = adata[combined_cell_mask, :][:, gene_mask].copy()
    
    # Free original data immediately
    del adata
    gc.collect()
    
    # Add gene_symbols to var
    adata_filtered.var["gene_symbols"] = gene_names[gene_mask].to_numpy()

    # Drop stale categories so downstream tools (e.g. scanpy) only see
    # groups that have at least one cell in the filtered subset.
    for _col in adata_filtered.obs.columns:
        if isinstance(adata_filtered.obs[_col].dtype, pd.CategoricalDtype):
            adata_filtered.obs[_col] = adata_filtered.obs[_col].cat.remove_unused_categories()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _replace_on_success(output_path) as partial:
        adata_filtered.write(partial)
    
    # Expand gene_cell_counts to original size
    gene_cell_counts_full = np.zeros(original_n_vars, dtype=np.int64)
    gene_cell_counts_full[gene_mask] = gene_cell_counts[gene_mask]
    
    # Build perturbation_keep dict from filtered data
    filtered_labels = adata_filtered.obs[perturbation_column].astype(str)
    pert_counts_final = filtered_labels.value_counts()
    perturbation_keep = {
        label: (label == control_label) or (pert_counts_final.get(label, 0) >= min_cells_per_perturbation)
        for label in filtered_labels.unique()
    }
    
    # Return backed view for consistency
    del adata_filtered
    gc.collect()
    filtered_adata_view = AnnData(output_path)
    
    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered=filtered_adata_view,
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=gene_cell_counts_full,
    )


def _qc_column_oriented(
    path: str | Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None,
    chunk_size: int,
    output_path: Path,
    verbose: int | bool = True,
) -> QualityControlResult:
    """Column-oriented QC for large CSC datasets (Option B).
    
    Iterates by column chunks (fast for CSC) and accumulates per-cell nnz.
    This maintains O(1) memory relative to data size while being efficient
    for CSC-stored files.
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum genes per cell.
    min_cells_per_perturbation
        Minimum cells per perturbation.
    min_cells_per_gene
        Minimum cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Control label (already resolved).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of columns to process per chunk.
    output_path
        Path for output h5ad file.
        
    Returns
    -------
    QualityControlResult
        QC result with masks and filtered AnnData.
    """
    logger.debug("Using column-oriented QC path (large CSC dataset)")
    
    # Read metadata
    backed = read_backed(path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        gene_names = ensure_gene_symbol_column(backed, gene_name_column)
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()
    
    # Pass 1: Iterate by columns to compute both metrics
    # - genes_per_cell: nnz count per row, accumulated across column chunks
    # - cells_per_gene: nnz count per column, computed per chunk
    genes_per_cell = np.zeros(n_obs, dtype=np.int64)
    cells_per_gene_all = np.zeros(n_vars, dtype=np.int64)
    
    n_col_chunks = (n_vars + chunk_size - 1) // chunk_size
    backed = read_backed(path)
    try:
        with _create_progress_context(n_col_chunks, "pp.qc_summary (pass 1)", verbose, unit="chunk") as pbar:
            for col_start in range(0, n_vars, chunk_size):
                col_end = min(col_start + chunk_size, n_vars)
                # Column slice is O(1) for CSC
                block = backed.X[:, col_start:col_end]

                if sp.issparse(block):
                    # Per-row nnz for this column chunk
                    genes_per_cell += np.asarray(block.getnnz(axis=1)).ravel()
                    # Per-column nnz
                    cells_per_gene_all[col_start:col_end] = np.asarray(block.getnnz(axis=0)).ravel()
                else:
                    genes_per_cell += np.count_nonzero(block, axis=1)
                    cells_per_gene_all[col_start:col_end] = np.count_nonzero(block, axis=0)
                pbar.update(1)
    finally:
        backed.file.close()
    
    gene_counts_per_cell = genes_per_cell
    
    # Cell filtering
    cell_mask = genes_per_cell >= min_genes
    
    # Perturbation filtering (metadata only)
    label_series = pd.Series(labels)
    counts = label_series[cell_mask].value_counts()
    count_per_cell = label_series.map(counts).fillna(0).to_numpy()
    is_control = labels == control_label
    has_enough = count_per_cell >= min_cells_per_perturbation
    combined_cell_mask = (is_control | has_enough) & cell_mask
    
    # Recompute cells_per_gene for filtered cells if needed
    # (only if perturbation filtering removed cells)
    removed_cells = cell_mask & ~combined_cell_mask
    if removed_cells.any():
        # Subtract counts from removed cells using column-oriented pass
        backed = read_backed(path)
        try:
            with _create_progress_context(n_col_chunks, "pp.qc_summary (pass 2)", verbose, unit="chunk") as pbar:
                for col_start in range(0, n_vars, chunk_size):
                    col_end = min(col_start + chunk_size, n_vars)
                    block = backed.X[:, col_start:col_end]
                    selected = block[removed_cells]
                    if sp.issparse(selected):
                        cells_per_gene_all[col_start:col_end] -= np.asarray(selected.getnnz(axis=0)).ravel()
                    else:
                        cells_per_gene_all[col_start:col_end] -= np.count_nonzero(selected, axis=0)
                    pbar.update(1)
        finally:
            backed.file.close()
    
    gene_cell_counts = cells_per_gene_all
    gene_mask = gene_cell_counts >= min_cells_per_gene
    
    # Write filtered subset using existing function
    write_filtered_subset(
        path,
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        output_path=output_path,
        chunk_size=chunk_size,
        var_assignments={"gene_symbols": gene_names[gene_mask]},
    )
    
    filtered = AnnData(output_path)
    filtered_labels = filtered.obs[perturbation_column].astype(str)
    filtered_counts = filtered_labels.value_counts()
    perturbation_keep = {
        label: (label == control_label) or (filtered_counts[label] >= min_cells_per_perturbation)
        for label in filtered_labels.unique()
    }
    
    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered=filtered,
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=gene_cell_counts,
    )


def _qc_row_oriented(
    path: str | Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None,
    chunk_size: int,
    output_path: Path,
    cache_mode: Literal['memory', 'disk', 'none'] = 'disk',
    delta_threshold: float = 0.3,
    verbose: int | bool = True,
) -> QualityControlResult:
    """Row-oriented streaming QC for large CSR/dense datasets.
    
    This is the original streaming implementation optimized for row-oriented
    access patterns (CSR format or dense arrays).
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum genes per cell.
    min_cells_per_perturbation
        Minimum cells per perturbation.
    min_cells_per_gene
        Minimum cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Control label (already resolved).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    output_path
        Path for output h5ad file.
    cache_mode
        Cache strategy: 'memory' (fast, high RAM), 'disk' (low RAM, a file
        beside the output), or 'none' (no caching, requires re-reading source
        during write). Default is 'disk' for better memory efficiency.
    delta_threshold
        Threshold for delta adjustment.
        
    Returns
    -------
    QualityControlResult
        QC result with masks and filtered AnnData.
    """
    logger.debug("Using row-oriented streaming QC path (large CSR/dense dataset)")
    
    # Read metadata
    backed = read_backed(path)
    try:
        gene_names = ensure_gene_symbol_column(backed, gene_name_column)
        n_obs, n_vars = backed.n_obs, backed.n_vars
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()

    # Pass 1: Filter cells by gene count AND compute cells-per-gene for all cells
    cell_filter_result = filter_cells_by_gene_count(
        path,
        min_genes=min_genes,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
        return_full_result=True,
    )
    cell_mask = cell_filter_result.cell_mask
    gene_counts_per_cell = cell_filter_result.gene_counts_per_cell
    gene_cell_counts_all = cell_filter_result.gene_cell_counts_all
    
    # No matrix pass: Filter perturbations by cell count (metadata only)
    perturbation_mask = filter_perturbations_by_cell_count(
        path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_cells=min_cells_per_perturbation,
        base_mask=cell_mask,
    )
    combined_cell_mask = cell_mask & perturbation_mask
    
    # Compute gene counts for filtered cells using delta adjustment
    removed_cell_mask = cell_mask & ~perturbation_mask
    n_removed = int(removed_cell_mask.sum())
    n_filtered = int(combined_cell_mask.sum())
    
    if n_removed == 0:
        gene_cell_counts = gene_cell_counts_all
        logger.debug("No cells removed by perturbation filter, using all-cell gene counts")
    elif n_filtered > 0 and n_removed / n_filtered < delta_threshold:
        logger.debug(
            "Using delta adjustment: %d removed cells (%.1f%% of %d filtered)",
            n_removed, 100 * n_removed / n_filtered, n_filtered
        )
        delta_counts = _compute_gene_count_delta(
            path,
            removed_cell_mask=removed_cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
        )
        gene_cell_counts = gene_cell_counts_all - delta_counts
    else:
        logger.debug(
            "Using full recompute: %d removed cells (%.1f%% of %d filtered)",
            n_removed, 100 * n_removed / n_filtered if n_filtered > 0 else 0, n_filtered
        )
        _, gene_cell_counts = filter_genes_by_cell_count(
            path,
            min_cells=0,
            cell_mask=combined_cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            return_counts=True,
        )
    
    # Pass 2: Gene filtering with nnz counting
    is_dense = is_dense_storage(path)
    gene_mask = gene_cell_counts >= min_cells_per_gene
    
    if is_dense:
        logger.debug("Using dense-optimized path (source stored as dense array)")
        gene_filter_result = _filter_genes_dense_optimized(
            path,
            min_cells=min_cells_per_gene,
            cell_mask=combined_cell_mask,
            gene_cell_counts=gene_cell_counts,
            gene_mask=gene_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        chunk_cache = None
    else:
        logger.debug("Using CSR cache path (source stored as sparse, cache_mode=%s)", cache_mode)
        gene_filter_result = _filter_genes_with_cache(
            path,
            min_cells=min_cells_per_gene,
            cell_mask=combined_cell_mask,
            gene_cell_counts=gene_cell_counts,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            output_path=output_path,
            cache_mode=cache_mode,
            verbose=verbose,
        )
        chunk_cache = gene_filter_result.chunk_cache
    
    gene_mask = gene_filter_result.gene_mask
    
    try:
        write_filtered_subset(
            path,
            cell_mask=combined_cell_mask,
            gene_mask=gene_mask,
            output_path=output_path,
            chunk_size=chunk_size,
            var_assignments={"gene_symbols": gene_names[gene_mask]},
            row_nnz=gene_filter_result.row_nnz,
            total_nnz=gene_filter_result.total_nnz,
            data_dtype=gene_filter_result.data_dtype,
            chunk_cache=chunk_cache,
        )
    finally:
        if chunk_cache is not None:
            chunk_cache.cleanup()

    filtered = AnnData(output_path)
    filtered_labels = filtered.obs[perturbation_column].astype(str)
    filtered_counts = filtered_labels.value_counts()
    perturbation_keep = {
        label: (label == control_label) or (filtered_counts[label] >= min_cells_per_perturbation)
        for label in filtered_labels.unique()
    }

    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered=filtered,
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=gene_cell_counts,
    )


def _qc_masks_only_column_oriented(
    path: str | Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    chunk_size: int,
) -> QualityControlResult:
    """Column-oriented masks-only QC for large CSC datasets.

    Mirrors :func:`_qc_column_oriented` but skips writing a filtered h5ad.
    Iterating by column chunks is O(1) memory relative to data size and
    avoids the pathological ``O(total_nnz)``-per-chunk cost of row-slicing a
    backed CSC matrix (which makes the row-oriented masks path ~100x slower
    at genome scale).

    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum genes per cell.
    min_cells_per_perturbation
        Minimum cells per perturbation.
    min_cells_per_gene
        Minimum cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Control label (already resolved).
    chunk_size
        Number of columns to process per chunk.

    Returns
    -------
    QualityControlResult
        QC result with masks but filtered=None (no output file).
    """
    logger.debug("Computing QC masks only via column-oriented path (large CSC dataset)")

    # Read metadata
    backed = read_backed(path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()

    # Pass 1: Iterate by columns (fast for CSC) to compute both metrics.
    genes_per_cell = np.zeros(n_obs, dtype=np.int64)
    cells_per_gene_all = np.zeros(n_vars, dtype=np.int64)

    backed = read_backed(path)
    try:
        for col_start in range(0, n_vars, chunk_size):
            col_end = min(col_start + chunk_size, n_vars)
            # Column slice is O(1) for CSC.
            block = backed.X[:, col_start:col_end]
            if sp.issparse(block):
                genes_per_cell += np.asarray(block.getnnz(axis=1)).ravel()
                cells_per_gene_all[col_start:col_end] = np.asarray(block.getnnz(axis=0)).ravel()
            else:
                genes_per_cell += np.count_nonzero(block, axis=1)
                cells_per_gene_all[col_start:col_end] = np.count_nonzero(block, axis=0)
    finally:
        backed.file.close()

    gene_counts_per_cell = genes_per_cell

    # Cell filtering
    cell_mask = genes_per_cell >= min_genes

    # Perturbation filtering (metadata only)
    label_series = pd.Series(labels)
    counts = label_series[cell_mask].value_counts()
    count_per_cell = label_series.map(counts).fillna(0).to_numpy()
    is_control = labels == control_label
    has_enough = count_per_cell >= min_cells_per_perturbation
    combined_cell_mask = (is_control | has_enough) & cell_mask

    # Recompute cells_per_gene for filtered cells if perturbation filtering
    # removed any cells (column-oriented pass, fast for CSC).
    removed_cells = cell_mask & ~combined_cell_mask
    if removed_cells.any():
        backed = read_backed(path)
        try:
            for col_start in range(0, n_vars, chunk_size):
                col_end = min(col_start + chunk_size, n_vars)
                block = backed.X[:, col_start:col_end]
                selected = block[removed_cells]
                if sp.issparse(selected):
                    cells_per_gene_all[col_start:col_end] -= np.asarray(selected.getnnz(axis=0)).ravel()
                else:
                    cells_per_gene_all[col_start:col_end] -= np.count_nonzero(selected, axis=0)
        finally:
            backed.file.close()

    gene_cell_counts = cells_per_gene_all
    gene_mask = gene_cell_counts >= min_cells_per_gene

    # Build perturbation_keep dict
    filtered_labels = labels[combined_cell_mask]
    unique_labels = np.unique(filtered_labels)
    label_counts = pd.Series(filtered_labels).value_counts()
    perturbation_keep = {
        label: (label == control_label) or (label_counts.get(label, 0) >= min_cells_per_perturbation)
        for label in unique_labels
    }

    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered=None,  # No output file
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=gene_cell_counts,
    )


def _qc_masks_only(
    path: str | Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str | None,
    chunk_size: int,
    delta_threshold: float = 0.3,
    storage_format: str = 'csr',
) -> QualityControlResult:
    """Compute QC masks without writing output file.
    
    This is a lightweight QC path that returns only the masks and statistics
    without writing a filtered h5ad file. Useful for:
    - Memory-constrained environments where users only need the masks
    - Workflows that apply masks downstream in a custom manner
    - Quick QC statistics without I/O overhead
    
    Parameters
    ----------
    path
        Path to h5ad file.
    min_genes
        Minimum genes per cell.
    min_cells_per_perturbation
        Minimum cells per perturbation.
    min_cells_per_gene
        Minimum cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Control label (already resolved).
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk.
    delta_threshold
        Threshold for delta adjustment.
    storage_format
        On-disk storage format of ``X`` ('csr', 'csc', or 'dense'). When
        'csc', a column-oriented counting path is used to avoid the
        pathological cost of row-slicing a backed CSC matrix.
        
    Returns
    -------
    QualityControlResult
        QC result with masks but filtered=None (no output file).
    """
    logger.debug("Computing QC masks only (no output file)")

    # For large CSC inputs, row-slicing a backed matrix is O(total_nnz) per
    # chunk (~100x slower at genome scale). Use the column-oriented path,
    # which is the natural fast axis for CSC.
    if storage_format == 'csc':
        return _qc_masks_only_column_oriented(
            path,
            min_genes=min_genes,
            min_cells_per_perturbation=min_cells_per_perturbation,
            min_cells_per_gene=min_cells_per_gene,
            perturbation_column=perturbation_column,
            control_label=control_label,
            chunk_size=chunk_size,
        )

    # Read metadata
    backed = read_backed(path)
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()

    # Pass 1: Filter cells by gene count AND compute cells-per-gene for all cells
    cell_filter_result = filter_cells_by_gene_count(
        path,
        min_genes=min_genes,
        gene_name_column=gene_name_column,
        chunk_size=chunk_size,
        return_full_result=True,
    )
    cell_mask = cell_filter_result.cell_mask
    gene_counts_per_cell = cell_filter_result.gene_counts_per_cell
    gene_cell_counts_all = cell_filter_result.gene_cell_counts_all
    
    # Filter perturbations by cell count (metadata only)
    perturbation_mask = filter_perturbations_by_cell_count(
        path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_cells=min_cells_per_perturbation,
        base_mask=cell_mask,
    )
    combined_cell_mask = cell_mask & perturbation_mask
    
    # Compute gene counts for filtered cells using delta adjustment
    removed_cell_mask = cell_mask & ~perturbation_mask
    n_removed = int(removed_cell_mask.sum())
    n_filtered = int(combined_cell_mask.sum())
    
    if n_removed == 0:
        gene_cell_counts = gene_cell_counts_all
    elif n_filtered > 0 and n_removed / n_filtered < delta_threshold:
        delta_counts = _compute_gene_count_delta(
            path,
            removed_cell_mask=removed_cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
        )
        gene_cell_counts = gene_cell_counts_all - delta_counts
    else:
        _, gene_cell_counts = filter_genes_by_cell_count(
            path,
            min_cells=0,
            cell_mask=combined_cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            return_counts=True,
        )
    
    # Compute gene mask
    gene_mask = gene_cell_counts >= min_cells_per_gene
    
    # Build perturbation_keep dict
    filtered_labels = labels[combined_cell_mask]
    unique_labels = np.unique(filtered_labels)
    label_counts = pd.Series(filtered_labels).value_counts()
    perturbation_keep = {
        label: (label == control_label) or (label_counts.get(label, 0) >= min_cells_per_perturbation)
        for label in unique_labels
    }

    return QualityControlResult(
        cell_mask=combined_cell_mask,
        gene_mask=gene_mask,
        perturbation_keep=perturbation_keep,
        filtered=None,  # No output file
        cell_gene_counts=gene_counts_per_cell,
        gene_cell_counts=gene_cell_counts,
    )


@accepts_in_memory(output_required=False)
def quality_control_summary(
    data: str | Path | AnnData | ad.AnnData,
    *,
    min_genes: int = 100,
    min_cells_per_perturbation: int = 50,
    min_cells_per_gene: int = 100,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    cache_mode: Literal['memory', 'disk', 'none'] = 'disk',
    delta_threshold: float = 0.3,
    force_streaming: bool = False,
    verbose: int | bool = True,
) -> QualityControlResult:
    """Run QC with automatic strategy selection for optimal performance.
    
    This function automatically selects the best QC strategy based on:
    1. Small data (any format): In-memory processing (fastest)
    2. Large CSC data: Column-oriented streaming (memory efficient for CSC)
    3. Large CSR/dense data: Row-oriented streaming (current behavior)
    
    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx AnnData / backed anndata.AnnData.
        An in-memory AnnData is streamed from a temporary copy; ``output_path``
        is then required.
    min_genes
        Minimum number of expressed genes per cell.
    min_cells_per_perturbation
        Minimum number of cells required per perturbation.
    min_cells_per_gene
        Minimum number of cells expressing each gene.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Label identifying control cells. If None, auto-detected.
    gene_name_column
        Column in var containing gene names.
    chunk_size
        Number of cells to process per chunk. If None, automatically
        calculated based on available memory.
    memory_limit_gb
        Optional memory limit in GB for strategy selection and chunk size.
        If None, auto-detected from system memory.
    data_name
        Base name for output files.
    output_path
        Exact path for the output h5ad file. When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for output files. If None, returns QC masks without writing
        a filtered h5ad file (QualityControlResult.filtered will be None).
        Deprecated; use ``output_path`` instead. Will be removed in the
        next major version.
    cache_mode
        Cache strategy for row-oriented streaming: 'memory' (fast, high RAM),
        'disk' (low RAM, a file beside the output), or 'none' (no caching,
        requires re-reading source during write). Default is 'disk' for
        better memory efficiency.
    delta_threshold
        Threshold for delta adjustment in row-oriented streaming.
        Default 0.3 (30%).
    force_streaming
        If True, always use streaming path regardless of data size.
        Useful for testing or memory-constrained environments.
        
    Returns
    -------
    QualityControlResult
        Dataclass containing masks, filtered AnnData (or None if output_dir
        is None), and QC statistics.
    """
    path = resolve_data_path(data)
    _messages.print_reading(verbose, "pp.qc_summary", path)
    
    # Read metadata and resolve control label
    backed = read_backed(path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label_resolved = resolve_control_label(labels, control_label)
    finally:
        backed.file.close()
    
    # Detect storage format and file size
    storage_format = get_matrix_storage_format(path)
    # The data's size, not the file's: a standardized file links its matrix.
    file_size_gb = h5ad_data_bytes(path) / 1e9
    
    # The memory this call may use: memory_limit_gb, capped by what the
    # process's cgroup will still grant (detected when None).
    budget_gb = _resolve_memory_limit_bytes(memory_limit_gb) / 1e9
    
    # Estimate memory needed for in-memory processing.
    # For dense arrays, the compressed file size can be much smaller than the
    # actual uncompressed footprint, so we use n_obs * n_vars * dtype_size.
    if storage_format == 'dense':
        import h5py as _h5py
        with _h5py.File(path, 'r') as _f:
            _dtype = _f['X'].dtype if isinstance(_f.get('X'), _h5py.Dataset) else None
        _itemsize = _dtype.itemsize if _dtype is not None else 4  # default float32
        # 2x: one copy to load + one working copy
        estimated_memory_gb = n_obs * n_vars * _itemsize * 2 / 1e9
    else:
        # For sparse HDF5 formats, the compressed file size significantly
        # underestimates the in-memory footprint.  HDF5 gzip compression
        # achieves 3-5× for sparse scRNA-seq data, so a 27 GB file expands to
        # 80+ GB when loaded as a scipy CSR matrix.  QC operations then create
        # additional working copies (boolean indexing, tocsr(), etc.), raising
        # peak usage to ~4-6× the compressed file size.
        # Use 4× as a conservative estimate to avoid OOM on large datasets.
        estimated_memory_gb = file_size_gb * 4
    
    # Determine chunk size for streaming paths
    _chunk_size_auto = chunk_size is None
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(
            n_obs, n_vars, available_memory_gb=budget_gb
        )

    # Handle output_dir=None and output_path=None: return masks only without writing output
    if output_dir is None and output_path is None:
        logger.info("output_dir is None, returning QC masks without writing filtered h5ad")
        _messages.vprint(
            verbose, "pp.qc_summary",
            "output_dir=None and output_path=None — returning masks only (no file written)",
        )
        return _qc_masks_only(
            path,
            min_genes=min_genes,
            min_cells_per_perturbation=min_cells_per_perturbation,
            min_cells_per_gene=min_cells_per_gene,
            perturbation_column=perturbation_column,
            control_label=control_label_resolved,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            delta_threshold=delta_threshold,
            storage_format=storage_format,
        )
    
    # Resolve output path
    filtered_path = resolve_output_path(
        path, suffix="filtered", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )
    _messages.print_saving(verbose, "pp.qc_summary", filtered_path)
    
    # Common kwargs for all strategies
    common_kwargs = {
        "min_genes": min_genes,
        "min_cells_per_perturbation": min_cells_per_perturbation,
        "min_cells_per_gene": min_cells_per_gene,
        "perturbation_column": perturbation_column,
        "control_label": control_label_resolved,
        "gene_name_column": gene_name_column,
        "output_path": filtered_path,
    }
    
    # Select strategy
    # Cap the in-memory threshold at 50 GB to avoid OOM on nodes with very
    # high memory_limit_gb (e.g., 500 GB auto-detected on 1 TB HPC nodes).
    # On a 128 GB node this allows in-memory QC for files up to ~12.5 GB
    # (estimated_memory = file_size × 4 < 50 GB → file < 12.5 GB).
    _in_memory_threshold_gb = min(budget_gb * 0.6, 50.0)
    _chunk_note = f", chunk_size={chunk_size} (auto)" if _chunk_size_auto else ""
    if not force_streaming and estimated_memory_gb < _in_memory_threshold_gb:
        # Option A: In-memory for small datasets
        logger.info(
            f"Using in-memory QC (estimated: {estimated_memory_gb:.1f}GB, threshold: {_in_memory_threshold_gb:.1f}GB)"
        )
        _messages.vprint(
            verbose, "pp.qc_summary",
            f"Strategy — in-memory ({file_size_gb:.1f} GB, estimated {estimated_memory_gb:.1f} GB in memory)",
        )
        result = _qc_in_memory(path, **common_kwargs)
    
    elif storage_format == 'csc':
        # Option B: Column-oriented for large CSC
        logger.info(
            f"Using column-oriented streaming QC (CSC format, {file_size_gb:.2f}GB)"
        )
        _messages.vprint(
            verbose, "pp.qc_summary",
            f"Strategy — column-streaming ({file_size_gb:.1f} GB, estimated "
            f"{estimated_memory_gb:.1f} GB in memory{_chunk_note})",
        )
        result = _qc_column_oriented(path, chunk_size=chunk_size, verbose=verbose, **common_kwargs)
    
    else:
        # Row-oriented streaming for large CSR/dense
        logger.info(
            f"Using row-oriented streaming QC ({storage_format} format, {file_size_gb:.2f}GB)"
        )
        _messages.vprint(
            verbose, "pp.qc_summary",
            f"Strategy — row-streaming ({file_size_gb:.1f} GB, estimated "
            f"{estimated_memory_gb:.1f} GB in memory{_chunk_note})",
        )
        result = _qc_row_oriented(
            path,
            chunk_size=chunk_size,
            cache_mode=cache_mode,
            delta_threshold=delta_threshold,
            verbose=verbose,
            **common_kwargs,
        )

    n_obs_in = int(result.cell_mask.sum())
    n_vars_in = int(result.gene_mask.sum())
    n_perts_total = len(result.perturbation_keep)
    n_perts_in = sum(result.perturbation_keep.values())
    pct_cells = 100 * n_obs_in / n_obs if n_obs else 0
    pct_genes = 100 * n_vars_in / n_vars if n_vars else 0
    message = (
        f"{n_obs_in}/{n_obs} cells kept ({pct_cells:.0f}%), "
        f"{n_vars_in}/{n_vars} genes kept ({pct_genes:.0f}%)"
    )
    if n_perts_total:
        pct_perts = 100 * n_perts_in / n_perts_total
        message += f", {n_perts_in}/{n_perts_total} perturbations kept ({pct_perts:.0f}%)"
    _messages.print_done(verbose, "pp.qc_summary", message)

    _warn_if_heavily_filtered("pp.qc_summary", n_obs_in, n_obs, kind="cells")
    _warn_if_heavily_filtered("pp.qc_summary", n_vars_in, n_vars, kind="genes")
    _warn_if_heavily_filtered("pp.qc_summary", n_perts_in, n_perts_total, kind="perturbations")
    return result



def compare_controls(
    data: str | Path | AnnData | ad.AnnData,
    *,
    guide_column: str,
    controls: Iterable[str],
    reference: str | Iterable[str],
    work_dir: str | Path,
    min_cells: int = 20,
    max_reference_cells: int | None = None,
    seed: int = 0,
    verbose: int | bool = True,
    **nb_glm_kwargs,
) -> pd.DataFrame:
    """Do the cells used as controls behave alike? Typically: are cells with no
    detected guide ("unassigned") like cells with a non-targeting guide?

    Screens with a low assignment rate often pool unassigned cells into the
    controls (Feng *et al.* 2026, *Cell Genomics*, used every unassigned cell).
    That is sound only if unassigned cells are expressed like non-targeting
    ones. Each control guide is tested with :func:`crispyx.de.nb_glm_test`
    twice: against the ``reference`` cells (the unassigned ones), and against
    the other control guides. The second comparison is the yardstick: two
    pools of genuine controls also differ, through guide-to-guide
    differences and the test's own miscalibration. If the reference cells are
    like the control guides, a guide gets about as many hits against them as
    against the other guides; many more means they differ. All control-guide
    cells pooled are also tested against the reference.

    Parameters
    ----------
    data
        Raw counts (h5ad path, or AnnData).
    guide_column
        ``obs`` column naming each cell's guide, or the reference label for
        cells without one.
    controls
        The control (non-targeting) guides, values of ``guide_column``.
    reference
        The value(s) of ``guide_column`` of the cells to check, e.g.
        ``"unassigned"``.
    work_dir
        Directory for the working files: the control and reference cells
        (``controls.h5ad``) and the ``nb_glm_test`` results.
    min_cells
        Control guides with fewer cells are left out (they still count as
        "other control guides").
    max_reference_cells
        Keep at most this many reference cells, drawn at random (``seed``).
        With ``covariates`` every fit runs over all the reference cells, so a
        large unassigned pool is slow; a few tens of thousands are plenty.
    seed
        Seed for ``max_reference_cells``.
    verbose
        Print progress.
    **nb_glm_kwargs
        Passed to every :func:`~crispyx.de.nb_glm_test` call (e.g. ``n_jobs``,
        ``memory_limit_gb``, ``covariates``).

    Returns
    -------
    pandas.DataFrame
        One row per control guide tested, and an ``"all control guides"``
        row: ``cells``, and for each comparison (``vs reference``,
        ``vs other controls``) the genes at BH q < 0.1 (``hits``) and the
        percentage of tested genes with \\|z\\| > 3 (0.27 % for a calibrated
        null). The table is also written to ``work_dir/compare_controls.csv``.
    """
    from .de import nb_glm_test

    owned = {"perturbation_column", "groupby", "control_label", "perturbations", "output_path", "force"}
    if clash := sorted(owned & nb_glm_kwargs.keys()):
        raise TypeError(f"compare_controls sets {clash} of nb_glm_test itself")
    path = resolve_data_path(data)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    guides = load_obs(path)[guide_column].astype(str)
    reference = {reference} if isinstance(reference, str) else {str(r) for r in reference}
    controls = sorted({str(c) for c in controls} - reference)
    sizes = guides.value_counts()
    tested = [g for g in controls if sizes.get(g, 0) >= min_cells]
    if len(tested) < 2:
        raise ValueError(
            f"compare_controls needs at least two control guides with >= {min_cells} cells; "
            f"found {len(tested)} of {len(controls)}"
        )
    is_reference = guides.isin(reference).to_numpy()
    if not is_reference.any():
        raise ValueError(f"no cell has {guide_column!r} in {sorted(reference)}")
    if max_reference_cells is not None and is_reference.sum() > max_reference_cells:
        keep = np.random.default_rng(seed).choice(np.flatnonzero(is_reference), max_reference_cells, replace=False)
        is_reference = np.zeros_like(is_reference)
        is_reference[keep] = True

    subset = work_dir / "controls.h5ad"
    write_filtered_subset(
        path, cell_mask=is_reference | guides.isin(controls).to_numpy(),
        gene_mask=np.ones(len(load_var(path)), dtype=bool), output_path=subset,
    )
    obs = load_obs(subset)
    guide = obs[guide_column].astype(str).to_numpy()
    ref = np.isin(guide, list(reference))
    obs["_cx_vs_reference"] = np.where(ref, "reference", guide)
    obs["_cx_pooled"] = np.where(ref, "reference", "controls")
    write_obs(subset, obs)
    kw = dict(verbose=False, force=True, **nb_glm_kwargs)

    def calls(result, label):
        z, q = np.asarray(result[label].statistic), np.asarray(result[label].pvalue_adj)
        finite = np.isfinite(z)
        return int(np.sum(q[np.isfinite(q)] < 0.1)), 100 * float(np.mean(np.abs(z[finite]) > 3))

    _messages.vprint(verbose, "pp.compare_controls",
                     f"{len(tested)} control guides against {int(ref.sum()):,} reference cells")
    vs_reference = nb_glm_test(subset, perturbation_column="_cx_vs_reference", control_label="reference",
                               perturbations=tested, output_path=work_dir / "vs_reference.h5ad", **kw)
    pooled = nb_glm_test(subset, perturbation_column="_cx_pooled", control_label="reference",
                         perturbations=["controls"], output_path=work_dir / "pooled_vs_reference.h5ad", **kw)
    rows = {}
    for k, label in enumerate(tested):
        obs["_cx_vs_controls"] = np.where(guide == label, label, np.where(ref, "_reference", "other"))
        write_obs(subset, obs)
        vs_controls = nb_glm_test(subset, perturbation_column="_cx_vs_controls", control_label="other",
                                  perturbations=[label], output_path=work_dir / "vs_other_controls.h5ad", **kw)
        hits_r, z3_r = calls(vs_reference, label)
        hits_c, z3_c = calls(vs_controls, label)
        rows[label] = {"cells": int((guide == label).sum()), "vs reference: hits": hits_r,
                       "vs reference: |z|>3 %": z3_r, "vs other controls: hits": hits_c,
                       "vs other controls: |z|>3 %": z3_c}
        _messages.vprint(verbose, "pp.compare_controls",
                         f"{k + 1}/{len(tested)} {label}: {hits_r} hits vs reference, {hits_c} vs other controls")
    hits_p, z3_p = calls(pooled, "controls")
    rows["all control guides"] = {"cells": int((~ref).sum()), "vs reference: hits": hits_p,
                                  "vs reference: |z|>3 %": z3_p, "vs other controls: hits": np.nan,
                                  "vs other controls: |z|>3 %": np.nan}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = guide_column
    table.to_csv(work_dir / "compare_controls.csv")
    return table


def control_guide_correlation(
    data: str | Path | AnnData | ad.AnnData,
    *,
    guide_column: str,
    controls: Iterable[str],
    reference: str | Iterable[str] | None = None,
    n_genes: int = 2000,
    min_cells: int = 20,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    """Correlation between the expression profiles of the control guides.

    Each control guide's counts are summed over its cells (one streaming
    pass over their rows), normalised to ``log1p`` counts per 10,000, and
    taken as a deviation from the control guides' average. The Pearson
    correlation of these deviations over the ``n_genes`` most expressed genes
    is near 0 for guides that differ only by noise; a block of correlated
    guides is variation they share (a batch, an off-target effect), which a
    test treats as signal when those guides are the controls. With
    ``reference`` (e.g. ``"unassigned"``), the reference cells are one more
    profile, so their distance from the guides shows too.

    Parameters
    ----------
    data
        Raw counts (h5ad path, or AnnData).
    guide_column
        ``obs`` column naming each cell's guide.
    controls
        The control (non-targeting) guides, values of ``guide_column``.
    reference
        Value(s) of ``guide_column`` pooled into one more profile, or None.
    n_genes
        The most expressed genes (mean over the control guides) correlated.
    min_cells
        Control guides with fewer cells are left out.
    chunk_size
        Rows read at a time.

    Returns
    -------
    pandas.DataFrame
        The correlation matrix (guides x guides); ``attrs["n_cells"]`` holds
        each profile's number of cells. Plot it with
        :func:`crispyx.plotting.plot_control_guide_correlation`.
    """
    path = resolve_data_path(data)
    guides = load_obs(path)[guide_column].astype(str).to_numpy()
    reference = set() if reference is None else ({reference} if isinstance(reference, str) else set(map(str, reference)))
    sizes = pd.Series(guides).value_counts()
    names = [g for g in sorted({str(c) for c in controls} - reference) if sizes.get(g, 0) >= min_cells]
    if len(names) < 2:
        raise ValueError(f"control_guide_correlation needs at least two control guides with >= {min_cells} cells")
    n_controls = len(names)
    code = pd.Series(range(n_controls), index=names).reindex(guides).to_numpy(dtype=np.float64, copy=True)
    if reference:
        code[np.isin(guides, list(reference))] = n_controls
        names.append(f"{'/'.join(sorted(reference))} (reference)")
    code = np.nan_to_num(code, nan=-1).astype(np.int64)
    rows = np.flatnonzero(code >= 0)
    sums = np.zeros((len(names), len(load_var(path))))
    backed = read_backed(path)
    try:
        for start in range(0, rows.size, chunk_size):
            block_rows = rows[start:start + chunk_size]
            block = sp.csr_matrix(backed.X[block_rows, :], dtype=np.float64)
            member = sp.csr_matrix((np.ones(block_rows.size), (code[block_rows], np.arange(block_rows.size))),
                                   shape=(len(names), block_rows.size))
            sums += np.asarray((member @ block).todense())
    finally:
        backed.file.close()
    profile = np.log1p(sums / np.maximum(sums.sum(axis=1, keepdims=True), 1) * 1e4)
    genes = np.argsort(profile[:n_controls].mean(axis=0))[::-1][:n_genes]
    deviation = profile[:, genes] - profile[:n_controls, genes].mean(axis=0)
    corr = pd.DataFrame(np.corrcoef(deviation), index=names, columns=names)
    corr.attrs["n_cells"] = {name: int(np.sum(code == k)) for k, name in enumerate(names)}
    return corr


# =============================================================================
# Guide assignment
# =============================================================================


def _dominant_guide(counts: np.ndarray, *, min_umi: int, min_ratio: float) -> np.ndarray:
    """Column of the dominant guide per row of a dense cells x guides block,
    or -1 where no guide dominates."""
    counts = np.asarray(counts)
    if counts.shape[1] == 1:
        top = counts[:, 0]
        second = np.zeros(len(counts), dtype=counts.dtype)
        idx = np.zeros(len(counts), dtype=np.int64)
    else:
        part = np.partition(counts, counts.shape[1] - 2, axis=1)
        top, second = part[:, -1], part[:, -2]
        idx = counts.argmax(axis=1)
    ok = (top >= min_umi) & (top >= min_ratio * second) & (top > second)
    return np.where(ok, idx, -1)


def assign_guides(
    guide_counts: "str | Path | ad.AnnData | AnnData | np.ndarray | sp.spmatrix",
    guide_names: "Iterable[str] | None" = None,
    *,
    min_umi: int = 3,
    min_ratio: float = 2.0,
    chunk_size: int = 4096,
    verbose: int | bool = True,
) -> pd.Series:
    """Assign each cell the guide that dominates its guide-capture counts.

    A cell is assigned its most abundant guide when that guide has at least
    ``min_umi`` UMIs and at least ``min_ratio`` times the UMIs of the
    runner-up. Any other cell is unassigned (``pd.NA``): no guide detected, two
    guides at similar counts, a multiplet. An unassigned cell is not a control;
    whether such cells may join the controls is
    :func:`compare_controls`'s question.

    Parameters
    ----------
    guide_counts
        The cells x guides UMI matrix: a dense or sparse array, an AnnData
        whose ``X`` is that matrix (its ``var_names`` are the guides), or a
        path to such an h5ad, which is streamed in blocks of ``chunk_size``
        cells. Guide-capture counts from Cell Ranger are the ``CRISPR Guide
        Capture`` features of its filtered matrix.
    guide_names
        The guides, one per column. Required for a bare array; taken from
        ``var_names`` otherwise.
    min_umi
        Fewest UMIs the top guide needs. Default 3.
    min_ratio
        Smallest ratio of the top guide's UMIs to the runner-up's. Default 2.
        A tie never dominates, whatever the ratio.
    chunk_size
        Cells per streamed block when ``guide_counts`` is a path.
    verbose
        Print how many cells were assigned.

    Returns
    -------
    pd.Series
        The assigned guide per cell, ``pd.NA`` where none dominates, indexed
        by the cells (``obs_names`` where known).
    """
    if not isinstance(min_umi, (int, np.integer)) or min_umi < 1:
        raise ValueError("min_umi must be a positive integer")
    if not (min_ratio >= 1):
        raise ValueError("min_ratio must be at least 1")

    if isinstance(guide_counts, (str, Path, AnnData)) or (
        isinstance(guide_counts, ad.AnnData) and guide_counts.isbacked
    ):
        backed = read_backed(resolve_data_path(guide_counts))
        try:
            names = np.asarray(backed.var_names, dtype=object) if guide_names is None else np.asarray(list(guide_names), dtype=object)
            index = backed.obs_names.copy()
            n_guides = backed.n_vars
            calls = np.empty(backed.n_obs, dtype=np.int64)
            row = 0
            for _, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=True):
                calls[row:row + block.shape[0]] = _dominant_guide(block, min_umi=min_umi, min_ratio=min_ratio)
                row += block.shape[0]
        finally:
            backed.file.close()
    else:
        if isinstance(guide_counts, ad.AnnData):
            matrix = guide_counts.X
            index = guide_counts.obs_names.copy()
            names = np.asarray(guide_counts.var_names, dtype=object) if guide_names is None else np.asarray(list(guide_names), dtype=object)
        else:
            matrix = guide_counts
            index = pd.RangeIndex(matrix.shape[0])
            if guide_names is None:
                raise ValueError("guide_names is required when guide_counts is a bare array")
            names = np.asarray(list(guide_names), dtype=object)
        n_guides = matrix.shape[1]
        calls = np.empty(matrix.shape[0], dtype=np.int64)
        for start in range(0, matrix.shape[0], chunk_size):
            block = matrix[start:start + chunk_size]
            block = block.toarray() if sp.issparse(block) else np.asarray(block)
            calls[start:start + block.shape[0]] = _dominant_guide(block, min_umi=min_umi, min_ratio=min_ratio)
    if len(names) != n_guides:
        raise ValueError(f"{len(names)} guide names for {n_guides} columns")

    assigned = calls >= 0
    values = names[np.maximum(calls, 0)].astype(object)
    values[~assigned] = pd.NA
    out = pd.Series(values, index=index, name="guide", dtype=object)
    _messages.print_done(
        verbose, "pp.assign_guides",
        f"{int(assigned.sum())}/{len(calls)} cells assigned a guide "
        f"(top guide >= {min_umi} UMIs and >= {min_ratio:g}x the runner-up); "
        f"{int((~assigned).sum())} unassigned",
    )
    return out


# =============================================================================
# Perturbation-data checks
# =============================================================================


@dataclass(frozen=True)
class Issue:
    """One finding of :func:`check_perturbation_data`: ``level`` is
    ``"error"`` (the file will mislead an analysis) or ``"warning"`` (worth a
    look)."""

    level: Literal["error", "warning"]
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.message}"


_CONTROL_SYNONYMS = (
    r"^(ntc|nt|nc|neg|negative|non[-_ ]?target(ing)?|control|ctrl|scramble[d]?|"
    r"safe[-_ ]?(target(ing)?|harbou?r)|gfp|empty|mock|no[-_ ]?guide|unassigned)"
    r"([-_ .]?(control|ctrl|guide|sg)?[-_ .]?\d*)$"
)
_SPACER = r"[ACGT]{18,25}"
_SEPARATORS = r"[\s|:;,/]"
_ENSEMBL_ID = r"^ENS[A-Z]*G\d{11}"
_MISSING_LABELS = frozenset({"", "nan", "none", "null", "na", "n/a"})


def _examples(values, k: int = 5) -> str:
    values = list(values)
    shown = ", ".join(repr(str(v)) for v in values[:k])
    return shown + (", ..." if len(values) > k else "")


def check_perturbation_data(
    data: "str | Path | ad.AnnData | AnnData",
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    n_counts_column: str | None = None,
    sample_windows: int = 8,
    window_size: int = 256,
    verbose: int | bool = True,
) -> list[Issue]:
    """Check a perturbation screen's file for the mistakes that mislead an
    analysis without making it fail.

    Reads ``obs``, ``var`` and ``sample_windows`` evenly spaced blocks of
    ``window_size`` rows of ``X``, never the whole matrix. Each check catches a
    mistake that has happened in practice when a published screen was
    prepared:

    * a control synonym (``NTC``, ``non-targeting``, ``scramble``, ``GFP``,
      ``safe-targeting`` ...) left as a perturbation beside the control label,
      so its cells are tested instead of serving as controls;
    * labels that are still guide names: a 18-25 nt spacer, a separator or
      whitespace, an underscore;
    * labels that differ only by case, which split one target in two;
    * many labels and none hyphenated, the mark of symbols truncated at the
      first hyphen (``NKX2-5`` to ``NKX2``);
    * labels missing from ``var`` (targets not measured; expected for a
      non-coding library, a sign of a renamed gene otherwise);
    * cells without a label; no cell with the control label;
    * duplicate gene names, gene names that are Ensembl IDs while a symbol
      column exists;
    * ``X`` not holding integer counts, or cells whose row sum exceeds their
      recorded total (``n_counts_column``);
    * inconsistent compressed-sparse pointers (:func:`crispyx.data.check_sparse_layout`).

    Parameters
    ----------
    data
        Path to the h5ad file, or a backed AnnData.
    perturbation_column
        ``obs`` column with the perturbation labels.
    control_label
        The control label; inferred as in :func:`crispyx.data.resolve_control_label`
        when omitted (an ambiguous inference is itself an error).
    gene_name_column
        ``var`` column with gene symbols, if ``var_names`` are not symbols.
    n_counts_column
        ``obs`` column with each cell's total count, if any; ``n_counts`` or
        ``total_counts`` is used when present and this is omitted.
    sample_windows, window_size
        How much of ``X`` to read: ``sample_windows`` blocks of ``window_size``
        rows, evenly spaced.
    verbose
        Print each issue.

    Returns
    -------
    list[Issue]
        Every finding, errors first. Empty when the file passes. Pass the
        list to :func:`raise_on_errors` to stop a pipeline on an error.
    """
    from .data import load_obs, load_var, resolve_control_label

    name = "pp.check_data"
    path = resolve_data_path(data)
    errors: list[str] = []
    warnings_: list[str] = []
    obs = load_obs(path)
    var = load_var(path)

    # --- labels ---------------------------------------------------------------
    if perturbation_column not in obs.columns:
        errors.append(f"obs has no column '{perturbation_column}' (columns: {_examples(obs.columns, 8)})")
        labels = None
    else:
        raw = obs[perturbation_column]
        missing = raw.isna() | raw.astype(str).str.strip().str.lower().isin(_MISSING_LABELS)
        if missing.any():
            warnings_.append(f"{int(missing.sum())} cells have no perturbation label")
        labels = raw[~missing].astype(str)
    if labels is not None:
        unique = pd.Index(pd.unique(labels.to_numpy()))
        try:
            control = resolve_control_label(unique, control_label, verbose=False)
        except ValueError as exc:
            errors.append(str(exc))
            control = None
        if control is not None and not (labels == control).any():
            errors.append(f"no cell carries the control label {control!r}")
        targets = unique[unique != control] if control is not None else unique
        var_names = pd.Index(var.index.astype(str))
        gene_col = gene_name_column if gene_name_column in var.columns else None
        symbols = pd.Index(var[gene_col].astype(str)) if gene_col else var_names
        measured = pd.Index(symbols).union(var_names)

        upper = targets.str.upper()
        if upper.duplicated().any():
            errors.append("labels differing only by case split one target in two: "
                          + _examples(targets[upper.duplicated(keep=False)]))
        synonyms = targets[targets.str.match(_CONTROL_SYNONYMS, case=False)]
        if len(synonyms):
            also_genes = synonyms[synonyms.isin(measured)]
            real = synonyms[~synonyms.isin(measured)]
            if len(real):
                errors.append(f"control synonyms used as perturbations beside {control!r}: "
                              + _examples(real) + " (their cells are tested, not used as controls)")
            if len(also_genes):
                warnings_.append("labels that read like a control but are also measured genes: "
                                 + _examples(also_genes) + " (CTRL is a gene; make sure these are targets)")
        spacers = targets[targets.str.contains(_SPACER, regex=True)]
        if len(spacers):
            errors.append(f"{len(spacers)} labels contain a guide spacer sequence: " + _examples(spacers))
        separators = targets[targets.str.contains(_SEPARATORS, regex=True)]
        if len(separators):
            errors.append(f"{len(separators)} labels contain whitespace or a separator (raw guide names? "
                          "combinations?): " + _examples(separators))
        underscored = targets[targets.str.contains("_", regex=False) & ~targets.isin(measured)]
        if len(underscored):
            warnings_.append(f"{len(underscored)} labels contain an underscore and are not measured genes "
                             "(guide names?): " + _examples(underscored))
        if len(targets) > 500 and not targets.str.contains("-", regex=False).any():
            warnings_.append(f"{len(targets)} labels and none hyphenated: symbols truncated at '-' "
                             "(NKX2-5 to NKX2, HLA-B to HLA)?")
        flagged = synonyms.union(spacers).union(separators).union(underscored)
        absent = targets[~targets.isin(measured) & ~targets.isin(flagged)]
        if len(absent):
            warnings_.append(f"{len(absent)}/{len(targets)} labels name no gene in var "
                             "(non-coding targets, or renamed genes): " + _examples(absent))

    # --- genes ----------------------------------------------------------------
    var_names = pd.Index(var.index.astype(str))
    if var_names.duplicated().any():
        errors.append("duplicate var_names: " + _examples(var_names[var_names.duplicated()]))
    elif var_names.str.upper().duplicated().any():
        warnings_.append("var_names differing only by case: "
                         + _examples(var_names[var_names.str.upper().duplicated(keep=False)]))
    if len(var_names) and var_names.str.match(_ENSEMBL_ID).mean() > 0.5 and gene_name_column is None:
        symbol_column = detect_gene_symbol_column(path, verbose=False)
        hint = f"; pass gene_name_column='{symbol_column}'" if symbol_column else ""
        warnings_.append("var_names are Ensembl IDs, so results are reported by ID" + hint)
    par_y = var_names[var_names.str.endswith("_PAR_Y") & var_names.str.replace(r"_PAR_Y$", "", regex=True).isin(var_names)]
    if len(par_y):
        warnings_.append(f"{len(par_y)} genes have a _PAR_Y row beside their X row, so their counts are split: "
                         "pp.collapse_genes(by=var_names without the suffix) sums them; " + _examples(par_y))
    suffixed = var_names[var_names.str.match(r"^.+-\d+$")]
    made_unique = suffixed[suffixed.str.replace(r"-\d+$", "", regex=True).isin(var_names)]
    if len(made_unique):
        warnings_.append(f"{len(made_unique)} var_names look like var_names_make_unique suffixes of other var_names, "
                         "so one symbol sits on several rows: pp.collapse_genes(by=...) sums them; "
                         + _examples(made_unique))
    if gene_name_column is not None and gene_name_column in var.columns:
        symbols_all = pd.Index(var[gene_name_column].astype(str))
        repeated = symbols_all[symbols_all.duplicated()].unique()
        if len(repeated):
            warnings_.append(f"{len(repeated)} symbols in '{gene_name_column}' sit on more than one row, so results "
                             f"by symbol split their counts: pp.collapse_genes(by='{gene_name_column}') sums them; "
                             + _examples(repeated))

    # --- X ----------------------------------------------------------------------
    try:
        check_sparse_layout(path)
        layout_ok = True
    except ValueError as exc:
        errors.append(str(exc))
        layout_ok = False  # reading rows of a broken matrix would fail on its pointers
    if n_counts_column is None:
        n_counts_column = next((c for c in ("n_counts", "total_counts") if c in obs.columns), None)
    elif n_counts_column not in obs.columns:
        errors.append(f"obs has no column '{n_counts_column}'")
        n_counts_column = None
    totals = pd.to_numeric(obs[n_counts_column], errors="coerce").to_numpy(dtype=np.float64) if n_counts_column else None
    backed = read_backed(path)
    try:
        n_obs = backed.n_obs
        starts = np.unique(np.linspace(0, max(n_obs - window_size, 0), num=max(sample_windows, 1)).astype(np.int64))
        non_integer = 0
        over_total = 0
        for start in starts if layout_ok else []:
            stop = min(start + window_size, n_obs)
            if stop <= start:
                continue
            block = backed.X[start:stop]
            csr = _ensure_csr(block)
            values = csr.data
            if values.size and not (np.isfinite(values).all() and (values >= 0).all()
                                    and np.abs(values - np.rint(values)).max() <= 1e-6):
                non_integer += 1
            if totals is not None:
                sums = np.asarray(csr.sum(axis=1)).ravel()
                known = np.isfinite(totals[start:stop])
                over_total += int((sums[known] > totals[start:stop][known] * (1 + 1e-6) + 0.5).sum())
    finally:
        backed.file.close()
    if non_integer:
        warnings_.append(f"X does not hold integer counts in {non_integer}/{len(starts)} sampled blocks; "
                         "nb_glm_test, pseudo-bulk sums and downsample_counts need counts "
                         "(pp.recover_counts undoes log1p(normalize_total))")
    if over_total:
        errors.append(f"{over_total} sampled cells have a row sum above their '{n_counts_column}' "
                      "(wrong total, or X is not the matrix the totals describe)")

    issues = [Issue("error", m) for m in errors] + [Issue("warning", m) for m in warnings_]
    for issue in issues:
        _messages.vprint(verbose, name, str(issue))
    _messages.print_done(verbose, name, f"{len(errors)} errors, {len(warnings_)} warnings")
    return issues


def raise_on_errors(issues: "Iterable[Issue]", *, context: str = "check_perturbation_data") -> None:
    """Raise ``ValueError`` listing every ``error``-level issue, if any."""
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise ValueError(f"{context}: {len(errors)} error(s):\n" + "\n".join(f"- {e.message}" for e in errors))

