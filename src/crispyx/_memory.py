"""Memory management utilities for adaptive batch processing.

This module provides functions for estimating memory usage and
determining optimal batch sizes for parallel processing.
"""

from __future__ import annotations

import logging
import os
import warnings
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


def drop_file_cache(path: str | Path) -> None:
    """Write back and evict *path*'s pages from the kernel page cache (Linux only).

    On cgroup-limited systems (SLURM), page-cache pages count toward the
    memory limit, and dirty ones -- written but not yet on disk -- cannot be
    reclaimed until written back, which a network filesystem does slowly.
    Calling this after a streaming read, or after a block of writes, keeps
    the file's pages out of the cgroup budget. ``posix_fadvise`` drops only
    clean pages, so the file is synced first; for a file that was only read
    that costs nothing.

    The call is a no-op on non-Linux platforms or when the file cannot be
    opened.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


class _EvictEvery:
    """Count the bytes a streaming pass moves through some files and, every
    ``nbytes`` of them, :func:`drop_file_cache` each file, so a pass over
    hundreds of GB keeps about ``nbytes`` of their pages resident rather
    than all of them.

    Call it with the size of each block read or written.
    """

    def __init__(self, *paths: str | Path, nbytes: int = 1 << 30) -> None:
        self.paths = paths
        self.nbytes = nbytes
        self._pending = 0

    def __call__(self, moved: int) -> None:
        self._pending += int(moved)
        if self._pending >= self.nbytes:
            for path in self.paths:
                drop_file_cache(path)
            self._pending = 0


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
    memory_limit_gb: float | None = None,
) -> int:
    """Estimate maximum number of parallel workers based on memory constraints.

    Limits worker count to prevent OOM from multiple workers each allocating
    large work arrays: as many workers as fit 80% of the budget
    (:func:`_resolve_memory_limit_bytes`), and no more than the CPUs this
    process may use.

    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    memory_per_worker_mb
        Estimated memory per worker in MB. If None, calculated from data size.
    memory_limit_gb
        Explicit memory budget in GB; ``None`` uses the memory available.

    Returns
    -------
    int
        Recommended maximum number of workers.
    """
    effective_mb = _resolve_memory_limit_bytes(memory_limit_gb) / 1e6 * 0.8
    if memory_per_worker_mb is None:
        # Estimate: 4 work arrays + Y subset + overhead
        n_work_arrays = 5
        memory_per_worker_mb = n_samples * n_genes * 8 * n_work_arrays / 1e6
    max_workers = max(1, int(effective_mb / memory_per_worker_mb))
    return min(max_workers, _resolve_n_jobs(None))


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


def _slurm_allocation_bytes() -> float | None:
    """The memory Slurm allocated this job on this node, from the variables
    it exports to the job (``--mem`` as ``SLURM_MEM_PER_NODE``, ``--mem-per-cpu``
    as ``SLURM_MEM_PER_CPU`` times ``SLURM_CPUS_ON_NODE``, both in MB), or
    ``None`` outside a Slurm job or when neither is set."""
    env = os.environ
    if "SLURM_JOB_ID" not in env:
        return None
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            return float(env["SLURM_MEM_PER_NODE"]) * 2**20
        if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_ON_NODE"):
            return float(env["SLURM_MEM_PER_CPU"]) * float(env["SLURM_CPUS_ON_NODE"]) * 2**20
    except ValueError:
        pass
    return None


_warned_unknown_slurm_limit = False


def _granted_bytes() -> float | None:
    """The most memory this process's job will grant it, when that is known.

    The cgroup's headroom when its limit is visible
    (:func:`_cgroup_available_bytes`). Some clusters give a job a cgroup
    namespace whose limit the job cannot see from inside; Slurm's own
    record of the allocation (:func:`_slurm_allocation_bytes`) stands in
    then. In a Slurm job where neither is available the allocation is
    unknown: crispyx warns once and asks for ``memory_limit_gb``, since
    sizing from the host's memory is how a run outgrows its job.
    """
    global _warned_unknown_slurm_limit
    cgroup_available = _cgroup_available_bytes()
    if cgroup_available is not None:
        return cgroup_available
    allocation = _slurm_allocation_bytes()
    if allocation is None and "SLURM_JOB_ID" in os.environ and not _warned_unknown_slurm_limit:
        _warned_unknown_slurm_limit = True
        warnings.warn(
            "crispyx: running in Slurm job "
            f"{os.environ['SLURM_JOB_ID']} but cannot see its memory limit (no cgroup "
            "limit is visible and SLURM_MEM_PER_NODE / SLURM_MEM_PER_CPU are unset); "
            "pass memory_limit_gb set to the job's --mem so crispyx does not size "
            "itself from the whole node's memory.",
            stacklevel=3,
        )
    return allocation


def _detected_available_bytes() -> float:
    """Memory this process may actually use, in bytes.

    ``psutil.virtual_memory().available`` capped by what the job will grant
    (:func:`_granted_bytes`). Every crispyx auto-sizing path routes its
    host-memory reading through here so that none of them can budget from
    memory the job will not hand out.

    Raises
    ------
    ImportError
        When ``psutil`` is not installed, so callers keep their own default
        and their own warning.
    """
    import psutil
    available = float(psutil.virtual_memory().available)
    granted = _granted_bytes()
    if granted is not None:
        available = min(available, granted)
    return available


def _resolve_memory_limit_bytes(memory_limit_gb: float | None) -> float:
    """Resolve the effective memory limit in bytes.

    If *memory_limit_gb* is provided, convert it to bytes. Otherwise query
    the system with ``psutil`` and fall back to 64 GB. Either way the result
    is capped by what the job will grant when that is known, so a budget can
    never exceed what the job may actually allocate -- see
    :func:`_granted_bytes`.

    Parameters
    ----------
    memory_limit_gb
        Explicit memory limit in gigabytes, or ``None`` for auto-detect.

    Returns
    -------
    float
        Memory budget in bytes.
    """
    if memory_limit_gb is None:
        try:
            return _detected_available_bytes()
        except ImportError:
            return 64 * 1e9  # conservative default
    budget = memory_limit_gb * 1e9
    granted = _granted_bytes()
    if granted is not None:
        budget = min(budget, granted)
    return budget
