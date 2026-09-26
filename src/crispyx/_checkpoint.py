"""Checkpoint and progress utilities for streaming, resumable computations.

This module provides atomic checkpointing and progress tracking shared by
streaming differential expression tests and by ``batch_process``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

# Check for tqdm availability
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _write_checkpoint_atomic(
    checkpoint_path: Path,
    data: dict,
) -> None:
    """Write checkpoint data atomically using temp file + rename.
    
    This ensures checkpoint file is never corrupted on crash.
    """
    # Ensure parent directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temporary file in the same directory
    tmp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, checkpoint_path)  # atomic, and overwrites on Windows too
    except Exception:
        # Clean up temp file on error
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def _read_checkpoint(
    checkpoint_path: Path,
    required_keys: tuple[str, ...],
) -> dict | None:
    """Read checkpoint file, returning None if missing or corrupted.

    ``required_keys`` distinguishes a valid checkpoint from a corrupted or
    schema-mismatched one. The DE functions (through :class:`ResumableRun`)
    require ``("fingerprint",)``; ``batch_process`` uses
    its own gene-chunk schema and passes ``required_keys=("last_gene_chunk",
    "total_gene_chunks", "batches_used")``. Presence is all that is checked:
    a checkpoint written before 0.1.4 carries ``batches_used`` as a
    coordinate list rather than a packed bitmap, so it is accepted here and
    resumes on the right gene chunk; only :func:`_unpack_bool_matrix` rejects
    the payload, leaving the caller to warn that ``obs['n_batches_used']``
    will undercount.

    Returns
    -------
    dict or None
        Checkpoint data if valid, None if file is missing or corrupted.
    """
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        # Validate required fields
        if not isinstance(data, dict):
            return None
        if any(key not in data for key in required_keys):
            return None
        return data
    except (json.JSONDecodeError, IOError, OSError):
        return None


def _pack_bool_matrix(matrix: np.ndarray) -> dict:
    """Encode a boolean matrix compactly for a JSON checkpoint.

    A coordinate list (``np.argwhere(...).tolist()``) costs one nested JSON
    list per set element, which for ``batch_process``'s ``(n_groups,
    n_batches)`` grid is tens of thousands of them -- megabytes of
    pretty-printed JSON and ~1 s of encoding, rewritten after every gene
    chunk. Packed bits are the same information in ``n_groups * n_batches /
    8`` bytes.
    """
    matrix = np.ascontiguousarray(matrix, dtype=bool)
    return {
        "shape": list(matrix.shape),
        "bits": base64.b64encode(np.packbits(matrix).tobytes()).decode("ascii"),
    }


def _unpack_bool_matrix(payload: object, shape: tuple[int, int]) -> np.ndarray | None:
    """Decode :func:`_pack_bool_matrix`, or ``None`` if it does not fit ``shape``.

    Returning ``None`` rather than raising lets the caller fall back to its
    scan-based recovery, which is what any other unusable checkpoint does.
    """
    if not isinstance(payload, dict):
        return None
    try:
        stored_shape = tuple(int(x) for x in payload["shape"])
        raw = base64.b64decode(payload["bits"])
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None
    if stored_shape != shape:
        return None
    count = shape[0] * shape[1]
    flat = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    if flat.size < count:
        return None
    return flat[:count].astype(bool).reshape(shape)


def _find_last_completed_gene_chunk(
    h5ad_path: Path,
    n_gene_chunks: int,
    chunk_size: int,
    n_genes: int,
    weight_dataset: str = "layers/weight_sum",
) -> int:
    """Scan an output file's weight layer for the last fully-written gene
    chunk, as a fallback when the checkpoint JSON is missing or corrupted.

    Used by ``batch_process``, whose gene chunks are written strictly in
    order (unlike DE's per-candidate resume, where any candidate can
    complete independently): a chunk is "done" once *every* group's row in
    its column slice of ``weight_dataset`` holds a finite, positive weight.
    ``batch_process`` writes ``weight_dataset`` last for each chunk, after
    ``X`` and every other layer, so a chunk this scan reports complete has
    all of its datasets written; a crash before that final write leaves the
    slice at its zero fill value and the chunk is redone. Scanning stops at
    the first not-fully-written chunk. Returns -1 if none are complete.

    Known limitation: a group with zero cells for every gene in a chunk has
    a legitimately-zero weight there and looks indistinguishable from
    "not yet written".
    """
    if not h5ad_path.exists():
        return -1
    try:
        with h5py.File(h5ad_path, "r") as f:
            if weight_dataset not in f:
                return -1
            ds = f[weight_dataset]
            last = -1
            for i in range(n_gene_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, n_genes)
                column = np.asarray(ds[:, start:end])
                row_has_value = np.any(np.isfinite(column) & (column > 0), axis=1)
                if np.all(row_has_value):
                    last = i
                else:
                    break
            return last
    except Exception as e:
        logger.warning(f"Failed to scan h5ad for completed gene chunks: {e}")
        return -1


def _get_checkpoint_interval(n_perturbations: int, checkpoint_interval: int | None) -> int:
    """Determine checkpoint interval based on dataset size.
    
    Parameters
    ----------
    n_perturbations
        Total number of perturbations.
    checkpoint_interval
        User-specified interval, or None for auto.
        
    Returns
    -------
    int
        Number of perturbations to process between checkpoints.
    """
    if checkpoint_interval is not None:
        return max(1, checkpoint_interval)
    # Auto: every 1 for small datasets, every 10 for larger ones
    if n_perturbations < 100:
        return 1
    elif n_perturbations < 1000:
        return 10
    else:
        return 50


def run_fingerprint(source_path: Path, **items) -> dict:
    """Identity of a call, for deciding whether a checkpoint belongs to it.

    Covers the source file (path, size, modification time) and every item
    passed -- the parameters that shape the results -- normalised through
    JSON so a fingerprint read back from a checkpoint compares equal.
    """
    stat = Path(source_path).stat()
    fingerprint = {
        "source": str(source_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        **items,
    }
    return json.loads(json.dumps(fingerprint, default=_jsonable, sort_keys=True))


def _jsonable(value):
    """JSON form of the non-JSON types a DE call's parameters can hold.

    Arrays and pandas objects (e.g. ``size_factors``) become a digest of
    their contents: a ``repr`` is truncated and rounded, so two different
    vectors could share one, and ``tolist()`` of a per-cell vector is tens of
    megabytes rewritten into the checkpoint on every save.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
        hashes = pd.util.hash_pandas_object(value, index=not isinstance(value, pd.Index))
        columns = [str(c) for c in value.columns] if isinstance(value, pd.DataFrame) else None
        return {"type": type(value).__name__, "shape": list(value.shape), "columns": columns,
                "sha256": _sha256(hashes.to_numpy())}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return value.tolist()
        return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _sha256(value)}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class ResumableRun:
    """Result arrays and a progress checkpoint that outlive an interruption.

    The arrays are memmaps in a hidden ``.{output name}.resume`` directory
    beside the output (a temporary directory would vanish with the process,
    taking the finished work with it). The checkpoint JSON records the call's
    :func:`run_fingerprint` plus caller-defined progress, and :meth:`save`
    flushes the arrays before writing it, so everything a checkpoint lists
    as done is on disk.

    A caller that writes its partial output itself (rather than into
    arrays) keeps it in :attr:`directory` too and names it in ``requires``.

    A run resumes only when ``resume`` is set, the checkpoint's fingerprint
    matches this call, and every array exists with the expected shape and
    dtype (plus every file named in ``requires``). Otherwise the stale
    checkpoint and directory are discarded and the run starts fresh, so a
    checkpoint can never be paired with another run's data.
    """

    def __init__(
        self,
        output_path: Path,
        checkpoint_path: Path,
        *,
        fingerprint: dict,
        arrays: dict[str, tuple[tuple[int, ...], "np.typing.DTypeLike", float | int | bool]],
        resume: bool,
        requires: tuple[str, ...] = (),
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.fingerprint = fingerprint
        self.directory = output_path.with_name(f".{output_path.name}.resume")
        checkpoint = _read_checkpoint(checkpoint_path, required_keys=("fingerprint",)) if resume else None
        self.resumed = (
            checkpoint is not None
            and checkpoint["fingerprint"] == fingerprint
            and all((self.directory / name).exists() for name in requires)
            and all(
                (self.directory / f"{name}.dat").exists()
                and (self.directory / f"{name}.dat").stat().st_size
                == int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                for name, (shape, dtype, _fill) in arrays.items()
            )
        )
        self.progress = {k: v for k, v in checkpoint.items() if k != "fingerprint"} if self.resumed else {}
        self.arrays: dict[str, np.memmap] = {}
        if not self.resumed:
            self.discard()
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, (shape, dtype, fill) in arrays.items():
            path = self.directory / f"{name}.dat"
            if self.resumed:
                self.arrays[name] = np.memmap(path, mode="r+", dtype=dtype, shape=shape)
            else:
                array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
                if fill != 0:
                    array.fill(fill)
                self.arrays[name] = array

    def save(self, **progress) -> None:
        """Flush the arrays, then record ``progress`` in the checkpoint."""
        for array in self.arrays.values():
            array.flush()
        _write_checkpoint_atomic(self.checkpoint_path, {"fingerprint": self.fingerprint, **progress})

    def discard(self) -> None:
        """Remove the checkpoint and the arrays (after success, or when stale).

        Drops this object's references to the arrays; once the caller drops
        its own, the mapped pages are released.
        """
        self.arrays = {}
        self.checkpoint_path.unlink(missing_ok=True)
        # ignore_errors: on Windows a file still mapped by a live result view
        # cannot be deleted; the next run on this output removes it.
        shutil.rmtree(self.directory, ignore_errors=True)


class _DummyProgress:
    """Dummy progress bar that does nothing (for when verbose=False)."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n: int = 1):
        pass
    
    def set_postfix(self, **kwargs):
        pass


def _create_progress_context(
    total: int,
    desc: str,
    verbose: int | bool,
    *,
    unit: str = "perturbation",
    initial: int = 0,
) -> "_tqdm | _DummyProgress":
    """Create a progress bar context manager.

    Returns tqdm progress bar if verbose>=1 and tqdm is available,
    otherwise returns a dummy context manager.

    ``initial`` is the work a resumed run has already completed. Passing it
    here rather than calling ``update(initial)`` on a fresh bar matters in a
    log file: the latter writes a ``0/total`` line and then jumps, which
    reads as a run that restarted from nothing and then skipped ahead, and
    has already cost one investigation an afternoon. It also lets tqdm rate
    the remaining work instead of counting the skipped chunks as instant.
    """
    if int(verbose) >= 1 and HAS_TQDM and total > 0:
        return _tqdm(total=total, desc=desc, unit=unit, initial=initial)
    return _DummyProgress()
