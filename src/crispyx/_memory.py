"""Memory management utilities for adaptive batch processing.

This module provides functions for estimating memory usage and
determining optimal batch sizes for parallel processing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _get_available_memory_mb() -> float:
    """Get available system memory in MB, with fallback."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e6
    except ImportError:
        return 8000.0  # 8 GB default fallback


def _estimate_dense_memory_gb(n_cells: int, n_genes: int, n_copies: int = 3) -> float:
    """Estimate memory required to densify a matrix with work arrays.
    
    Parameters
    ----------
    n_cells
        Number of cells (rows).
    n_genes
        Number of genes (columns).
    n_copies
        Number of dense matrix copies needed (default 3: Y, mu, work arrays).
        
    Returns
    -------
    float
        Estimated memory in GB.
    """
    bytes_per_element = 8  # float64
    return n_cells * n_genes * bytes_per_element * n_copies / 1e9


def _estimate_gene_batch_size_fitter(
    n_samples: int,
    n_genes: int,
    n_work_arrays: int = 4,
    target_mb: float = 100.0,
) -> int:
    """Estimate optimal gene batch size based on memory constraints.
    
    Calculates batch size to keep work array memory usage under target_mb.
    Work arrays are typically (n_samples, batch_size) shaped.
    
    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    n_work_arrays
        Number of work arrays allocated per batch (default 4 after optimization).
    target_mb
        Target memory usage in MB for work arrays (default 100 MB).
        
    Returns
    -------
    int
        Recommended gene batch size, clamped between 256 and n_genes.
    """
    bytes_per_gene = n_samples * 8 * n_work_arrays  # float64 = 8 bytes
    target_bytes = target_mb * 1e6
    batch_size = int(target_bytes / bytes_per_gene)
    # Clamp between 256 (minimum for efficiency) and n_genes (maximum)
    return max(256, min(batch_size, n_genes))


def _estimate_max_workers(
    n_samples: int,
    n_genes: int,
    memory_per_worker_mb: float | None = None,
    available_mb: float | None = None,
    memory_limit_mb: float | None = None,
) -> int:
    """Estimate maximum number of parallel workers based on memory constraints.
    
    Limits worker count to prevent OOM from multiple workers each allocating
    large work arrays.
    
    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    memory_per_worker_mb
        Estimated memory per worker in MB. If None, calculated from data size.
    available_mb
        Available memory in MB. If None, uses 80% of system memory.
    memory_limit_mb
        Optional explicit memory limit in MB (e.g., from config). If provided,
        the effective memory budget is min(available_mb, memory_limit_mb).
        
    Returns
    -------
    int
        Recommended maximum number of workers.
    """
    if available_mb is None:
        # Try to get system memory, default to 8 GB if unavailable
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / 1e6 * 0.8
        except ImportError:
            available_mb = 8000.0  # 8 GB default
    
    # Apply explicit memory limit if provided
    if memory_limit_mb is not None:
        effective_mb = min(available_mb, memory_limit_mb * 0.8)  # 80% of limit
    else:
        effective_mb = available_mb
    
    if memory_per_worker_mb is None:
        # Estimate: 4 work arrays + Y subset + overhead
        n_work_arrays = 5
        memory_per_worker_mb = n_samples * n_genes * 8 * n_work_arrays / 1e6
    
    max_workers = max(1, int(effective_mb / memory_per_worker_mb))
    cpu_count = os.cpu_count() or 4
    
    return min(max_workers, cpu_count)


def _resolve_memory_limit_bytes(memory_limit_gb: float | None) -> float:
    """Resolve the effective memory limit in bytes.

    If *memory_limit_gb* is provided, convert it to bytes.
    Otherwise, query the system with ``psutil`` and fall back to 64 GB.

    Parameters
    ----------
    memory_limit_gb
        Explicit memory limit in gigabytes, or ``None`` for auto-detect.

    Returns
    -------
    float
        Memory budget in bytes.
    """
    if memory_limit_gb is not None:
        return memory_limit_gb * 1e9

    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except ImportError:
        return 64 * 1e9  # conservative default


def _should_use_streaming(
    n_groups: int,
    n_genes: int,
    *,
    memory_limit_gb: float | None = None,
    n_float64_arrays: int = 7,
    n_float32_arrays: int = 2,
    peak_multiplier: float = 5.0,
    threshold_fraction: float = 0.30,
) -> tuple[bool, float, float, int]:
    """Decide whether to use group-batch streaming for output arrays.

    Estimates the peak memory of the single-pass (memmap) approach, where
    ``n_groups × n_genes`` result arrays are allocated for all groups at
    once, then copied at the end.  If the estimated peak exceeds
    ``threshold_fraction`` of the available memory budget, streaming is
    recommended.

    The default ``peak_multiplier=5.0`` accounts for five additive
    contributions that are proportional to the output-array footprint:

    1. Memmap pages resident in the page cache after all gene chunks
       have been processed (~1×).
    2. ``numpy.array()`` copies created from the memmaps before the
       temporary directory is deleted (~1×).
    3. Backed h5ad file pages that accumulate in RSS as gene chunks
       are read from the sparse backing store (~1.5×).
    4. AnnData / h5py write buffers and Python working memory (~0.5×).
    5. glibc arena overhead and freed-but-unreturned pages (~1×).

    Parameters
    ----------
    n_groups
        Number of perturbation groups (excluding control).
    n_genes
        Number of genes.
    memory_limit_gb
        Explicit memory cap in GB.  ``None`` → auto-detect via psutil.
    n_float64_arrays
        Number of ``float64`` output arrays per group (default 7).
    n_float32_arrays
        Number of ``float32`` output arrays per group (default 2).
    peak_multiplier
        Factor applied to the memmap footprint to account for the
        end-of-run copy, backing-file RSS, and write overhead
        (default 4×).
    threshold_fraction
        Fraction of available memory above which streaming is triggered
        (default 0.30).

    Returns
    -------
    use_streaming : bool
        ``True`` when the streaming path should be used.
    estimated_peak_bytes : float
        Estimated peak memory for the standard path.
    memory_budget_bytes : float
        Effective memory budget in bytes.
    group_batch_size : int
        Recommended group batch size (meaningful only when
        ``use_streaming`` is ``True``).
    """
    bytes_per_group = n_genes * (n_float64_arrays * 8 + n_float32_arrays * 4)
    memmap_total_bytes = n_groups * bytes_per_group
    estimated_peak_bytes = memmap_total_bytes * peak_multiplier

    memory_budget_bytes = _resolve_memory_limit_bytes(memory_limit_gb)
    streaming_threshold = memory_budget_bytes * threshold_fraction
    use_streaming = estimated_peak_bytes > streaming_threshold

    # Calculate adaptive group batch size
    group_batch_size = n_groups  # default: all at once
    if use_streaming:
        batch_budget_bytes = memory_budget_bytes * (threshold_fraction / 2)
        group_batch_size = max(100, int(batch_budget_bytes / bytes_per_group))
        group_batch_size = min(group_batch_size, n_groups)

        logger.info(
            "Large result matrix detected: %d groups × %d genes = "
            "%.1f GB estimated peak (budget: %.1f GB). "
            "Switching to streaming mode with batch_size=%d.",
            n_groups, n_genes, estimated_peak_bytes / 1e9,
            memory_budget_bytes / 1e9, group_batch_size,
        )

    return use_streaming, estimated_peak_bytes, memory_budget_bytes, group_batch_size
