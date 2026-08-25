"""Disk-space estimation and warning utilities for streaming writers.

Unlike ``_memory.py`` (an internal tuning input that silently steers chunk
sizes and streaming decisions), this module has no algorithmic consumer: it
exists purely to tell the user, in plain language, how much disk a call is
about to need and whether that looks tight -- a courtesy message, not a
resource allocator. There is no override parameter analogous to
``memory_limit_gb``; free space is always read from the real filesystem.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

_BYTES_PER_GB = 1e9


def _safe_resolve(path: str | Path) -> Path:
    """Absolute form of *path* for display and lookup; never raises.

    Falls back to the unresolved path if resolution itself fails (e.g. a
    broken symlink or an inaccessible network mount) -- this is only used
    for a human-readable label and a starting point for the free-space
    walk-up below, never for correctness.
    """
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _free_bytes_at(path: str | Path) -> float | None:
    """Free space in bytes on the filesystem containing *path*, or ``None``
    if it could not be determined.

    Walks up to the nearest existing ancestor directory, since *path* is
    often an output file that does not exist yet. Never raises: this must
    work the same way on macOS, Linux, and Windows, where the ways a path
    lookup can fail differ (an unreachable network mount, a permission
    error walking a Windows junction, a drive that was ejected mid-check,
    ...). Any such failure degrades to "unknown" rather than propagating
    and breaking the caller's real computation, since this check is a
    diagnostic, not a precondition.
    """
    p = _safe_resolve(path)
    try:
        while not p.exists():
            parent = p.parent
            if parent == p:  # reached a filesystem root (POSIX "/" or a
                break         # Windows drive/UNC root); nothing further up.
            p = parent
        return float(shutil.disk_usage(p).free)
    except OSError:
        return None


def estimate_bytes(*dims: int, itemsize: int = 8, overhead: float = 1.0) -> float:
    """Size in bytes of a dense array with the given dimensions.

    One primitive covers both use cases in this module, since they're the
    same formula:

    - An ``np.memmap`` accumulator has a fixed, exact layout -- call with
      the default ``overhead=1.0``, e.g. ``estimate_bytes(n_pairs, n_genes)``.
    - A dense h5ad output (``X`` plus layers) additionally carries HDF5
      chunk/metadata bookkeeping -- fold the layer count in as another
      dimension and pass ``overhead~1.10``, e.g.
      ``estimate_bytes(n_obs, n_vars, n_layers, overhead=1.10)``.
    """
    n = 1
    for d in dims:
        n *= d
    return float(n * itemsize * overhead)


def estimate_sparse_output_bytes(
    nnz: int, *, value_itemsize: int = 4, index_itemsize: int = 4, overhead: float = 1.10,
) -> float:
    """Estimate an h5ad file size for a sparse (CSR/CSC) result from nnz."""
    return nnz * (value_itemsize + index_itemsize) * overhead


def estimate_conversion_bytes(source_path: str | Path) -> float:
    """Peak extra disk usage for a whole-file CSR<->CSC conversion.

    The destination is roughly the same size as the source (same nnz,
    swapped index array) and coexists with the source until the caller
    deletes it, so peak usage is ~2x the source file's current size.
    """
    return 2.0 * float(Path(source_path).stat().st_size)


@dataclass(frozen=True)
class DiskEstimate:
    """Required vs. free space at one filesystem location.

    ``free_bytes`` is ``None`` when free space could not be determined --
    an unreachable network mount, a permission error, or some other
    platform-specific quirk. This is a real, expected state on some
    platforms/filesystems, not an error case.
    """

    required_bytes: float
    free_bytes: float | None
    path: Path

    @property
    def required_gb(self) -> float:
        return self.required_bytes / _BYTES_PER_GB

    @property
    def free_gb(self) -> float | None:
        return None if self.free_bytes is None else self.free_bytes / _BYTES_PER_GB

    @property
    def sufficient(self) -> bool:
        # Fail open: an unknown free-space reading should never claim a
        # shortfall it has no evidence for.
        return self.free_bytes is None or self.required_bytes <= self.free_bytes * 0.90

    def __str__(self) -> str:
        if self.free_bytes is None:
            return f"{self.required_gb:.1f} GB required, free space unknown at {self.path}"
        verdict = "OK" if self.sufficient else "MAY NOT FIT"
        return f"{self.required_gb:.1f} GB required, {self.free_gb:.1f} GB free at {self.path} [{verdict}]"


def assess_bytes(required_bytes: float, path: str | Path) -> DiskEstimate:
    """Compare *required_bytes* against real free space at *path*."""
    return DiskEstimate(
        required_bytes=required_bytes,
        free_bytes=_free_bytes_at(path),
        path=_safe_resolve(path),
    )


def warn_if_disk_space_low(
    required_bytes: float,
    output_path: str | Path,
    *,
    min_free_fraction: float = 0.10,
    large_file_gb: float = 20.0,
    context: str = "",
) -> DiskEstimate:
    """Warn (never raises) if *output_path*'s filesystem looks too tight.

    Free space is always auto-detected from the real filesystem via
    ``shutil.disk_usage`` -- there is no override parameter, since this is a
    feasibility heads-up, not a configurable resource budget. Two
    independent triggers:

    1. Insufficient headroom: projected usage would leave less than
       ``min_free_fraction`` of free space unused on the target filesystem.
    2. Large output: ``required_bytes`` alone exceeds ``large_file_gb``,
       regardless of how much free space exists, so users on a shared or
       quota'd volume get a heads-up before a multi-hour write starts.

    Behaves identically on macOS, Linux, and Windows (``shutil.disk_usage``
    is cross-platform). If free space cannot be determined at all -- an
    unreachable network mount, a permission error -- trigger 1 is skipped
    rather than guessed at; trigger 2 still fires since it doesn't depend
    on free space.

    Returns the computed :class:`DiskEstimate` so callers can print a
    ``verbose``-gated confirmation alongside this unconditional warning
    without re-running the free-space lookup.
    """
    estimate = assess_bytes(required_bytes, output_path)
    label = f"{context}: " if context else ""

    if estimate.free_bytes is not None and not estimate.sufficient:
        warnings.warn(
            f"{label}estimated disk usage ({estimate.required_gb:.1f} GB) leaves less than "
            f"{min_free_fraction:.0%} free space at {estimate.path} "
            f"({estimate.free_gb:.1f} GB currently free). The operation may fail "
            "with 'No space left on device'. Free up space or point output_path/TMPDIR at a "
            "volume with more room before rerunning.",
            stacklevel=3,
        )
    elif estimate.required_gb > large_file_gb:
        warnings.warn(
            f"{label}this operation will write approximately {estimate.required_gb:.1f} GB to disk. "
            "Streaming keeps memory bounded, but the on-disk footprint (and the time to write "
            "it) is not -- make sure the output volume has room before running large batches.",
            stacklevel=3,
        )

    return estimate


__all__ = [
    "DiskEstimate",
    "assess_bytes",
    "estimate_bytes",
    "estimate_conversion_bytes",
    "estimate_sparse_output_bytes",
    "warn_if_disk_space_low",
]
