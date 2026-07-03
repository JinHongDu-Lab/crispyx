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

from .data import (
    AnnData,
    _ensure_csr,
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


class _MemmapChunkCache:
    """Memory-mapped chunk cache for QC streaming.
    
    Uses numpy memmap to store CSR chunks on disk with memory-mapped access,
    reducing RAM usage while maintaining fast access through OS page caching.
    Similar to the approach used in crispyx.glm for large streaming operations.
    
    Parameters
    ----------
    output_path
        Base path for output file. Cache files created in a temp directory.
    estimated_nnz
        Estimated total non-zeros for pre-allocation.
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
        self._estimated_nnz = estimated_nnz
        self._data_dtype = data_dtype
        
        # Create temp directory for memmap files
        self._cache_dir = Path(tempfile.mkdtemp(prefix="crispyx_qc_cache_"))
        
        # Pre-allocate memory-mapped arrays
        self._data_mmap = np.memmap(
            self._cache_dir / "data.mmap",
            dtype=data_dtype,
            mode='w+',
            shape=(estimated_nnz,),
        )
        self._indices_mmap = np.memmap(
            self._cache_dir / "indices.mmap",
            dtype=np.int32,
            mode='w+',
            shape=(estimated_nnz,),
        )
        
        # Track chunk boundaries: list of (start, end, indptr_diff)
        self._chunk_info: list[tuple[int, int, np.ndarray]] = []
        self._current_offset = 0
        self._n_cols: int = 0
    
    def write_chunk(
        self,
        chunk_idx: int,
        data: np.ndarray,
        indices: np.ndarray,
        indptr_diff: np.ndarray,
        n_cols: int,
    ) -> None:
        """Write CSR chunk data to memory-mapped files."""
        nnz = len(data)
        if nnz == 0:
            # Store empty chunk info
            self._chunk_info.append((self._current_offset, self._current_offset, indptr_diff.copy()))
            self._n_cols = n_cols
            return
            
        start = self._current_offset
        end = start + nnz
        
        # Check if we need to expand the memmap (shouldn't happen with good estimate)
        if end > self._estimated_nnz:
            logger.warning(
                "Memmap cache overflow: estimated %d nnz but need %d. "
                "Expanding cache (may be slower).",
                self._estimated_nnz, end
            )
            self._expand_memmaps(end)
        
        # Write to memmap (data stays on disk, not in RAM)
        self._data_mmap[start:end] = data
        self._indices_mmap[start:end] = indices
        
        # Store chunk metadata (indptr_diff is small, keep in memory)
        self._chunk_info.append((start, end, indptr_diff.copy()))
        self._current_offset = end
        self._n_cols = n_cols
    
    def _expand_memmaps(self, new_size: int) -> None:
        """Expand memory-mapped arrays to accommodate more data."""
        # Add 20% buffer to avoid repeated expansions
        new_size = int(new_size * 1.2)
        
        # Create new larger memmaps
        new_data_path = self._cache_dir / "data_expanded.mmap"
        new_indices_path = self._cache_dir / "indices_expanded.mmap"
        
        new_data = np.memmap(new_data_path, dtype=self._data_dtype, mode='w+', shape=(new_size,))
        new_indices = np.memmap(new_indices_path, dtype=np.int32, mode='w+', shape=(new_size,))
        
        # Copy existing data
        new_data[:self._current_offset] = self._data_mmap[:self._current_offset]
        new_indices[:self._current_offset] = self._indices_mmap[:self._current_offset]
        
        # Close and delete old memmaps
        del self._data_mmap
        del self._indices_mmap
        (self._cache_dir / "data.mmap").unlink(missing_ok=True)
        (self._cache_dir / "indices.mmap").unlink(missing_ok=True)
        
        # Rename new files
        new_data_path.rename(self._cache_dir / "data.mmap")
        new_indices_path.rename(self._cache_dir / "indices.mmap")
        
        # Update references
        self._data_mmap = new_data
        self._indices_mmap = new_indices
        self._estimated_nnz = new_size
    
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
        # Build gene index remapping
        remap = np.full(self._n_cols, -1, dtype=np.int32)
        remap[gene_indices] = np.arange(len(gene_indices), dtype=np.int32)
        
        for start, end, indptr_diff in self._chunk_info:
            n_cells = len(indptr_diff)
            
            if start == end:
                # Empty chunk
                yield np.array([], dtype=data_dtype), np.array([], dtype=np.int32), n_cells
                continue
            
            # Read from memmap (OS handles paging efficiently)
            data = np.array(self._data_mmap[start:end])  # Copy to regular array
            indices = np.array(self._indices_mmap[start:end])
            
            # Filter genes
            new_indices = remap[indices]
            keep_mask = new_indices >= 0
            
            yield data[keep_mask].astype(data_dtype, copy=False), new_indices[keep_mask], n_cells
    
    def cleanup(self) -> None:
        """Delete memory-mapped files and cache directory."""
        # Close memmap references
        del self._data_mmap
        del self._indices_mmap
        
        # Remove cache directory
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


def filter_cells_by_gene_count(
    data: str | Path | AnnData | ad.AnnData,
    *,
    min_genes: int = 100,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    return_counts: bool = False,
    return_full_result: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray] | _CellFilterResult:
    """Return a boolean mask selecting cells with at least ``min_genes`` expressed genes.
    
    This function can optionally compute both genes-per-cell (row nnz) AND
    cells-per-gene (column nnz) in a single matrix pass, avoiding a separate
    gene counting pass later.
    
    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx/anndata AnnData object.
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
    
    if return_full_result:
        return _CellFilterResult(
            cell_mask=mask,
            gene_counts_per_cell=gene_counts_per_cell,
            gene_cell_counts_all=gene_cell_counts_all,
        )
    
    if return_counts:
        return mask, gene_counts_per_cell
    return mask


def filter_perturbations_by_cell_count(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    min_cells: int = 50,
    base_mask: np.ndarray | None = None,
    return_counts: bool = False,
) -> np.ndarray | Tuple[np.ndarray, pd.Series]:
    """Return a mask keeping cells whose perturbation has sufficient representation.
    
    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx/anndata AnnData object.
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
    
    if return_counts:
        return mask, counts
    return mask


def filter_genes_by_cell_count(
    data: str | Path | AnnData | ad.AnnData,
    *,
    min_cells: int = 100,
    cell_mask: np.ndarray | None = None,
    gene_name_column: str | None = None,
    chunk_size: int = 2048,
    return_counts: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Return a boolean mask selecting genes expressed in at least ``min_cells`` cells.
    
    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx/anndata AnnData object.
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
    if return_counts:
        return mask, counts
    return mask


# Type alias for chunk cache (either in-memory or memmap)
_ChunkCacheType = Union[_ChunkCache, _MemmapChunkCache, None]


@dataclass
class _GeneFilterResult:
    """Result of fused gene filtering and nnz counting."""
    
    gene_mask: np.ndarray
    gene_cell_counts: np.ndarray
    row_nnz: np.ndarray
    total_nnz: int
    data_dtype: np.dtype
    chunk_cache: _ChunkCacheType = None  # Optional cache for write phase (memory or memmap)


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
    cache_mode: Literal['memory', 'memmap', 'none'] = 'memmap',
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
        Cache strategy: 'memory' (fast, high RAM), 'memmap' (low RAM, disk-based),
        or 'none' (no caching, requires re-reading source during write).
        Default is 'memmap' for better memory efficiency.
        
    Returns
    -------
    _GeneFilterResult
        Dataclass containing gene_mask, gene_cell_counts, row_nnz, total_nnz,
        data_dtype, and optionally chunk_cache for the write phase.
    """
    # Compute gene mask from pre-computed counts
    gene_mask = gene_cell_counts >= min_cells
    gene_indices = np.flatnonzero(gene_mask)
    
    # Estimate nnz for memmap pre-allocation (from file size heuristic)
    # This is a rough estimate; memmap will expand if needed
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
        elif cache_mode == 'memmap':
            # Detect data dtype from file for memmap pre-allocation
            backed = read_backed(path)
            try:
                if sp.issparse(backed.X):
                    data_dtype_hint = backed.X.dtype
                else:
                    data_dtype_hint = backed.X.dtype
            finally:
                backed.file.close()
            chunk_cache = _MemmapChunkCache(output_path, estimated_nnz, data_dtype_hint)
    
    backed = read_backed(path)
    try:
        row_nnz = np.zeros(n_filtered_cells, dtype=np.int64)
        total_nnz = 0
        data_dtype: np.dtype | None = None
        row_offset = 0
        chunk_idx = 0
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_cell_mask = cell_mask[slc]
            if not np.any(local_cell_mask):
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
        
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_cell_mask = cell_mask[slc]
            if not np.any(local_cell_mask):
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
    adata_filtered.write(output_path)
    
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
    
    backed = read_backed(path)
    try:
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
    perturbation_keep = {
        label: (label == control_label) or (filtered_labels[filtered_labels == label].shape[0] >= min_cells_per_perturbation)
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
    cache_mode: Literal['memory', 'memmap', 'none'] = 'memmap',
    delta_threshold: float = 0.3,
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
        Cache strategy: 'memory' (fast, high RAM), 'memmap' (low RAM, disk-based),
        or 'none' (no caching, requires re-reading source during write).
        Default is 'memmap' for better memory efficiency.
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
    perturbation_keep = {
        label: (label == control_label) or (filtered_labels[filtered_labels == label].shape[0] >= min_cells_per_perturbation)
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
    cache_mode: Literal['memory', 'memmap', 'none'] = 'memmap',
    delta_threshold: float = 0.3,
    force_streaming: bool = False,
    verbose: int | bool = False,
) -> QualityControlResult:
    """Run QC with automatic strategy selection for optimal performance.
    
    This function automatically selects the best QC strategy based on:
    1. Small data (any format): In-memory processing (fastest)
    2. Large CSC data: Column-oriented streaming (memory efficient for CSC)
    3. Large CSR/dense data: Row-oriented streaming (current behavior)
    
    Parameters
    ----------
    data
        Path to h5ad file, or a crispyx/anndata AnnData object.
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
        'memmap' (low RAM, disk-based), or 'none' (no caching, requires
        re-reading source during write). Default is 'memmap' for better
        memory efficiency.
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
    if int(verbose) >= 1:
        print(f"[cx] qc.quality_control: Reading {path}")
    
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
    file_size_gb = path.stat().st_size / 1e9
    
    # Determine available memory
    if memory_limit_gb is None:
        try:
            import psutil
            memory_limit_gb = psutil.virtual_memory().available / 1e9 * 0.5  # Use 50% of available
        except ImportError:
            memory_limit_gb = 8.0
    
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
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(
            n_obs, n_vars, available_memory_gb=memory_limit_gb
        )
    
    # Handle output_dir=None and output_path=None: return masks only without writing output
    if output_dir is None and output_path is None:
        logger.info("output_dir is None, returning QC masks without writing filtered h5ad")
        if int(verbose) >= 1:
            print("[cx] qc.quality_control: output_dir=None and output_path=None — returning masks only (no file written)")
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
    if int(verbose) >= 1:
        print(f"[cx] qc.quality_control: Saving → {filtered_path}")
    
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
    _in_memory_threshold_gb = min(memory_limit_gb * 0.6, 50.0)
    if not force_streaming and estimated_memory_gb < _in_memory_threshold_gb:
        # Option A: In-memory for small datasets
        logger.info(
            f"Using in-memory QC (estimated: {estimated_memory_gb:.1f}GB, threshold: {_in_memory_threshold_gb:.1f}GB)"
        )
        if int(verbose) >= 1:
            print(f"[cx] qc.quality_control: Strategy — in-memory ({file_size_gb:.1f} GB, estimated {estimated_memory_gb:.1f} GB in memory)")
        result = _qc_in_memory(path, **common_kwargs)
    
    elif storage_format == 'csc':
        # Option B: Column-oriented for large CSC
        logger.info(
            f"Using column-oriented streaming QC (CSC format, {file_size_gb:.2f}GB)"
        )
        if int(verbose) >= 1:
            print(f"[cx] qc.quality_control: Strategy — column-streaming ({file_size_gb:.1f} GB, estimated {estimated_memory_gb:.1f} GB in memory)")
        result = _qc_column_oriented(path, chunk_size=chunk_size, **common_kwargs)
    
    else:
        # Row-oriented streaming for large CSR/dense
        logger.info(
            f"Using row-oriented streaming QC ({storage_format} format, {file_size_gb:.2f}GB)"
        )
        if int(verbose) >= 1:
            print(f"[cx] qc.quality_control: Strategy — row-streaming ({file_size_gb:.1f} GB, estimated {estimated_memory_gb:.1f} GB in memory)")
        result = _qc_row_oriented(
            path,
            chunk_size=chunk_size,
            cache_mode=cache_mode,
            delta_threshold=delta_threshold,
            **common_kwargs,
        )

    if int(verbose) >= 1:
        n_obs_in = int(result.cell_mask.sum())
        n_vars_in = int(result.gene_mask.sum())
        pct_cells = 100 * n_obs_in / n_obs if n_obs else 0
        pct_genes = 100 * n_vars_in / n_vars if n_vars else 0
        print(
            f"[cx] qc.quality_control: Done  "
            f"{n_obs_in}/{n_obs} cells kept ({pct_cells:.0f}%), "
            f"{n_vars_in}/{n_vars} genes kept ({pct_genes:.0f}%)"
        )
    return result

