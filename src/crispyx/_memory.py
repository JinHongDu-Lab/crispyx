"""Memory management utilities for adaptive batch processing.

This module provides functions for estimating memory usage and
determining optimal batch sizes for parallel processing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Worker count for a joblib-style ``n_jobs``, capped at the CPUs this
    process may use (affinity mask and cgroup quota, via joblib).

    ``None`` means every available CPU, as the DE functions document;
    negative values count back from it (``-1`` = all); ``0`` is an error.
    """
    cpus = joblib.cpu_count()
    return min(joblib.effective_n_jobs(-1 if n_jobs is None else n_jobs), cpus)


def _get_available_memory_mb() -> float:
    """Get available system memory in MB, with fallback."""
    try:
        return _detected_available_bytes() / 1e6
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
            available_mb = _detected_available_bytes() / 1e6 * 0.8
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


#: cgroup v1 files report an "unlimited" ceiling as a sentinel near 2**63
#: rather than as an absent file, so anything at or above this is no limit.
_CGROUP_UNLIMITED = 1 << 62

#: Where the cgroup hierarchy is mounted, and where this process's place in it
#: is published. Module constants so a test can point them at a tree it built.
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_V2_MOUNT = "/sys/fs/cgroup"
_CGROUP_V1_MEMORY_MOUNT = "/sys/fs/cgroup/memory"


def _read_cgroup_int(path: Path) -> int | None:
    """One integer from a cgroup file, or ``None`` when it cannot be used.

    Absent, unreadable, and cgroup v2's ``max`` (its word for "no limit") all
    read as ``None``; v1's ~2**63 sentinel for the same thing parses fine and
    is rejected by the caller against :data:`_CGROUP_UNLIMITED`.
    """
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cgroup_stat(path: Path, key: str) -> int | None:
    """The value of one ``key value`` line of a cgroup ``memory.stat``."""
    try:
        with open(path) as handle:
            for line in handle:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        return None
    return None


def _cgroup_memory_dirs() -> list[Path]:
    """Every cgroup directory whose memory ceiling binds this process.

    Reading the hierarchy *root* alone would find a container's limit and
    nothing else: Docker and Kubernetes put the container in a cgroup
    namespace, so its own cgroup *is* what ``/sys/fs/cgroup`` shows. Slurm
    does not -- the job's ceiling sits several levels down, at
    ``/sys/fs/cgroup/system.slice/slurmstepd.scope/job_<id>/.../memory.max``
    (v2) or ``/sys/fs/cgroup/memory/slurm/uid_*/job_*/memory.limit_in_bytes``
    (v1), while the root publishes no limit at all. So the process's own path
    is resolved from ``/proc/self/cgroup``, and the whole chain up to the
    mount is returned: a limit anywhere on it binds everything below, so the
    effective allowance is the tightest of them, not the leaf's.

    Returns leaf-first, and empty off Linux or off a cgroup.
    """
    dirs: list[Path] = []
    try:
        with open(_PROC_SELF_CGROUP) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return dirs
    for line in lines:
        # "<hierarchy-id>:<controllers>:<path>"; v2 is the line with no
        # controller list, v1 publishes one line per controller group.
        _, _, rest = line.partition(":")
        controllers, _, relative = rest.partition(":")
        if not relative.startswith("/"):
            continue
        if not controllers:
            mount = Path(_CGROUP_V2_MOUNT)
        elif "memory" in controllers.split(","):
            mount = Path(_CGROUP_V1_MEMORY_MOUNT)
        else:
            continue
        if not mount.is_dir():
            continue
        # Descend as far as the mount actually goes. In a cgroup namespace
        # /proc/self/cgroup still names the host-side path, none of which
        # exists under the namespaced mount, so this stops at the mount --
        # which is exactly the container's own cgroup. Off a namespace it
        # walks the whole way down to the leaf.
        leaf = mount
        for part in relative.split("/"):
            if not part:
                continue
            candidate = leaf / part
            if not candidate.is_dir():
                break
            leaf = candidate
        for directory in (leaf, *leaf.parents):
            dirs.append(directory)
            if directory == mount:
                break
    return dirs


def _cgroup_available_bytes() -> float | None:
    """Memory this process's cgroup will still grant it, or ``None``.

    Slurm, Docker and Kubernetes all cap a job's memory with a cgroup, but
    ``psutil.virtual_memory()`` reports the *host's* memory. On a shared
    compute node a job allocated 200 GB can see a machine with far more than
    that free, and sizing buffers or chunks from what it sees is how an
    auto-sized run gets OOM-killed while every crispyx budget still looks
    satisfied.

    What is returned is the *headroom* -- ceiling minus what the cgroup
    already holds -- not the ceiling, because the ceiling is what the job was
    granted in total, not what is left to allocate: a job holding 150 GB of
    its 200 GB allocation has 50 GB for the next buffer, and budgeting 200
    would OOM exactly as budgeting the host's free memory does. Usage counts
    only the working set (``memory.current`` less ``inactive_file``), since
    page cache charged to the cgroup by streaming an h5ad is reclaimed under
    pressure rather than causing it -- subtracting it whole would collapse
    every chunk size as soon as a large file had been read once.
    """
    headroom: float | None = None
    for directory in _cgroup_memory_dirs():
        limit = _read_cgroup_int(directory / "memory.max")
        if limit is not None:  # cgroup v2
            usage = _read_cgroup_int(directory / "memory.current")
            reclaimable = _read_cgroup_stat(directory / "memory.stat", "inactive_file")
        else:  # cgroup v1
            limit = _read_cgroup_int(directory / "memory.limit_in_bytes")
            usage = _read_cgroup_int(directory / "memory.usage_in_bytes")
            reclaimable = _read_cgroup_stat(
                directory / "memory.stat", "total_inactive_file"
            )
        if limit is None or not 0 < limit < _CGROUP_UNLIMITED:
            continue
        working_set = max(0.0, float(usage or 0) - float(reclaimable or 0))
        free = max(0.0, float(limit) - working_set)
        headroom = free if headroom is None else min(headroom, free)
    return headroom


def _detected_available_bytes() -> float:
    """Memory this process may actually use, in bytes.

    ``psutil.virtual_memory().available`` capped by what the cgroup will still
    grant. Every crispyx auto-sizing path routes its host-memory reading
    through here so that none of them can budget from memory the job's cgroup
    will not hand out.

    Raises
    ------
    ImportError
        When ``psutil`` is not installed, so callers keep their own default
        and their own warning.
    """
    import psutil
    available = float(psutil.virtual_memory().available)
    cgroup_available = _cgroup_available_bytes()
    if cgroup_available is not None:
        available = min(available, cgroup_available)
    return available


def _resolve_memory_limit_bytes(memory_limit_gb: float | None) -> float:
    """Resolve the effective memory limit in bytes.

    If *memory_limit_gb* is provided, convert it to bytes. Otherwise query
    the system with ``psutil`` and fall back to 64 GB. Either way the result
    is capped by what the process's cgroup will still grant when there is
    one, so a budget can never exceed what the job may actually allocate --
    see :func:`_cgroup_available_bytes`.

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
        budget = memory_limit_gb * 1e9
    else:
        try:
            return _detected_available_bytes()
        except ImportError:
            return 64 * 1e9  # conservative default

    cgroup_available = _cgroup_available_bytes()
    if cgroup_available is not None:
        budget = min(budget, cgroup_available)
    return budget


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
