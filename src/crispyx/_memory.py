"""Memory management utilities for adaptive batch processing.

This module provides functions for estimating memory usage and
determining optimal batch sizes for parallel processing.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import re
from pathlib import Path

import joblib
import psutil

from . import _messages

logger = logging.getLogger(__name__)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Worker count for a joblib-style ``n_jobs``, capped at the CPUs this
    process may use (affinity mask and cgroup quota, via joblib).

    ``None`` means every available CPU, as the DE functions document;
    negative values count back from it (``-1`` = all); ``0`` is an error.
    """
    cpus = joblib.cpu_count()
    return min(joblib.effective_n_jobs(-1 if n_jobs is None else n_jobs), cpus)


@functools.lru_cache(maxsize=None)
def _page_cache_is_charged() -> bool:
    """Whether this process's page cache counts against a memory limit: a
    cgroup memory limit is visible, or it runs in a Slurm job (whose limit
    a container may hide). Fixed for the life of the process.

    Without such a limit -- a laptop, a workstation, an unconstrained
    container -- the kernel reclaims page cache on its own and keeping it
    is free, so nothing is evicted (a second pass over the same file then
    reads from cache).
    """
    return _cgroup_available_bytes() is not None or "SLURM_JOB_ID" in os.environ


def _is_ram_backed(path: str | Path) -> bool:
    """Whether ``path`` lies on a filesystem held in memory (``tmpfs``,
    ``ramfs``) -- a common ``/tmp`` on cluster nodes and in containers --
    where a file takes RAM, charged to the writer's memory cgroup. Read from
    ``/proc/self/mounts``; ``False`` where that does not exist (macOS,
    Windows), which keeps such files on disk.
    """
    try:
        with open("/proc/self/mounts") as handle:
            mounts = handle.read().splitlines()
        target = os.path.realpath(path)
    except OSError:
        return False
    best, fstype = "", ""
    for line in mounts:
        fields = line.split()
        if len(fields) < 3:
            continue
        # The kernel writes a space in a mount point as "\040".
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[1])
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, fstype = point, fields[2]
    return fstype in ("tmpfs", "ramfs")


def drop_file_cache(path: str | Path) -> None:
    """Evict *path*'s clean pages from the kernel page cache, when page cache
    counts against a memory limit (:func:`_page_cache_is_charged`).

    Under a cgroup limit (SLURM, Kubernetes), page-cache pages count toward
    it. Calling this after a streaming read, or after a block of writes,
    keeps the file's pages out of the job's budget. Pages still dirty are
    left for the kernel to write back, and a later call drops them; the file
    is not synced, which would stall each call on a network filesystem
    (~0.8 s a call on Lustre) for no gain in correctness: a killed process's
    writes are in the page cache either way.

    A no-op without such a limit, on platforms without ``posix_fadvise``
    (macOS, Windows), or when the file cannot be opened.
    """
    if not hasattr(os, "posix_fadvise") or not _page_cache_is_charged():
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


@functools.lru_cache(maxsize=None)
def _allocator_release():
    """The C allocator's call that hands freed memory back to the system:
    glibc's ``malloc_trim(0)``, macOS's ``malloc_zone_pressure_relief(NULL, 0)``,
    or ``None`` where there is none (musl, Windows)."""
    import ctypes

    try:
        libc = ctypes.CDLL(None)
    except (OSError, TypeError):  # Windows has no process-wide namespace
        return None
    for name, argtypes, restype, args in (
        ("malloc_trim", [ctypes.c_size_t], ctypes.c_int, (0,)),
        ("malloc_zone_pressure_relief", [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t, (None, 0)),
    ):
        function = getattr(libc, name, None)
        if function is not None:
            function.argtypes, function.restype = argtypes, restype
            return lambda: function(*args)
    return None


def release_freed_memory() -> None:
    """Collect garbage and return the memory freed so far to the system.

    Both glibc and macOS keep memory a process has freed, and a loop that
    allocates a different set of large arrays each time (one sample after
    another) grows its resident size from that kept memory rather than
    reusing it: twenty Scrublet runs held about twice one run's peak. Call this
    between such steps, once the previous step's arrays are dropped.
    """
    gc.collect()
    release = _allocator_release()
    if release is not None:
        release()


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
        _messages.emit(
            "crispyx: running in Slurm job "
            f"{os.environ['SLURM_JOB_ID']} but cannot see its memory limit (no cgroup "
            "limit is visible and SLURM_MEM_PER_NODE / SLURM_MEM_PER_CPU are unset); "
            "pass memory_limit_gb set to the job's --mem so crispyx does not size "
            "itself from the whole node's memory.",
        )
    return allocation


def _detected_available_bytes() -> float:
    """Memory this process may actually use, in bytes.

    ``psutil.virtual_memory().available`` capped by what the job will grant
    (:func:`_granted_bytes`). Every crispyx auto-sizing path routes its
    host-memory reading through here so that none of them can budget from
    memory the job will not hand out.
    """
    available = float(psutil.virtual_memory().available)
    granted = _granted_bytes()
    if granted is not None:
        available = min(available, granted)
    return available


def _resident_bytes() -> int:
    """The resident memory of this process, in bytes."""
    return psutil.Process().memory_info().rss


#: What importing crispyx added to the process's resident memory (crispyx and
#: the libraries it imports, measured across its import, so not data or
#: libraries the caller loaded first): what an explicit limit leaves out.
#: Recorded once, so a budget does not change with what the session holds
#: later -- data loaded, or memory a call freed that the allocator keeps.
_import_footprint: int = 0


def _record_import_footprint(resident_before: int) -> None:
    global _import_footprint
    _import_footprint = max(0, _resident_bytes() - resident_before)


def _baseline_bytes() -> int:
    """What an explicit limit leaves out: crispyx's import footprint."""
    return _import_footprint


def _given_limit_bytes(memory_limit_gb: float | None) -> float:
    """*memory_limit_gb* in bytes as given, capped by what the job will
    grant (the detected memory when ``None``): for a size that identifies a
    result (PCA's method and chunk, a resumable run's chunks), which must not
    move with crispyx's footprint. Under a cgroup the cap is its headroom,
    so such a size stays put only where the job has the limit free."""
    if memory_limit_gb is None:
        return _detected_available_bytes()
    granted = _granted_bytes()
    return memory_limit_gb * 1e9 if granted is None else min(memory_limit_gb * 1e9, granted)


def _resolve_memory_limit_bytes(memory_limit_gb: float | None) -> float:
    """The bytes a call may allocate, in bytes.

    *memory_limit_gb* is the limit on the whole process, as a job's
    ``--mem`` is: what importing crispyx added (crispyx and the libraries it
    imports, :func:`_baseline_bytes`) is taken off it, as the detected
    memory (``None``) already leaves out what the process holds. Data the
    caller loads, and libraries imported before crispyx, are not counted. Either way the result
    is capped by what the job will grant when that is known, so a budget can
    never exceed what the job may actually allocate -- see
    :func:`_granted_bytes`.

    Parameters
    ----------
    memory_limit_gb
        Memory limit of the process in gigabytes, or ``None`` for auto-detect.

    Returns
    -------
    float
        Memory budget in bytes.

    Raises
    ------
    MemoryError
        When *memory_limit_gb* is no more than crispyx's import footprint: no
        budget is left, and a call sized from none would read its input once
        per row or gene.
    """
    if memory_limit_gb is None:
        return _detected_available_bytes()
    baseline = _baseline_bytes()
    if baseline >= memory_limit_gb * 1e9:
        raise MemoryError(
            f"memory_limit_gb={memory_limit_gb:g} leaves nothing to work with: importing crispyx and its "
            f"libraries took {baseline / 1e9:.2f} GB of it. The limit is the whole process's; raise it."
        )
    return min(memory_limit_gb * 1e9 - baseline, _given_limit_bytes(memory_limit_gb))
