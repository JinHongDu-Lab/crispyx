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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import numpy as np
import pandas as pd

from ._memory import drop_file_cache

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

    Covers the source file (resolved path, so any spelling of it matches;
    size; modification time) and every item passed -- the parameters that
    shape the results -- normalised through JSON so a fingerprint read back
    from a checkpoint compares equal.
    """
    source_path = Path(source_path).resolve()
    stat = source_path.stat()
    fingerprint = {
        "source": str(source_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        **items,
    }
    return json.loads(json.dumps(fingerprint, default=_jsonable, sort_keys=True))


# Fingerprint keys that identify the input file (see ``run_fingerprint``).
_SOURCE_KEYS = ("source", "source_size", "source_mtime_ns")


def fingerprint_difference(previous: dict, current: dict) -> str | None:
    """Why the call fingerprinted ``previous`` is not the call ``current``,
    or ``None`` when it is.

    ``None``-valued params are ignored: an argument left at ``None`` and one
    that does not exist mean the same call, so adding or removing a
    ``None``-default parameter does not orphan every existing result and
    checkpoint.
    """
    previous, current = _without_none_params(previous), _without_none_params(current)
    if previous == current:
        return None
    if previous.get("source") != current.get("source"):
        return "it was computed from another input file"
    if any(previous.get(key) != current.get(key) for key in _SOURCE_KEYS):
        return "the input file has changed since"
    if previous.get("inputs") != current.get("inputs"):
        return "an input file has changed since"
    old_params = previous.get("params", {})
    new_params = current.get("params", {})
    for name in sorted(set(old_params) | set(new_params)):
        if old_params.get(name) != new_params.get(name):
            return f"argument '{name}' differs"
    for name in sorted(set(previous) | set(current)):
        if previous.get(name) != current.get(name):
            return f"'{name}' differs"
    return "its recorded inputs differ"


def _without_none_params(fingerprint: dict) -> dict:
    params = fingerprint.get("params")
    if not isinstance(params, dict):
        return fingerprint
    return {**fingerprint, "params": {k: v for k, v in params.items() if v is not None}}


def _jsonable(value):
    """JSON form of the non-JSON types a DE call's parameters can hold.

    Arrays and pandas objects (e.g. ``size_factors``) become a digest of
    their contents: a ``repr`` is truncated and rounded, so two different
    vectors could share one, and ``tolist()`` of a per-cell vector is tens of
    megabytes rewritten into the checkpoint on every save. A Series' name is
    kept beside the digest, which ignores it: per-cell labels given as a
    Series name the result's grouping column.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
        hashes = pd.util.hash_pandas_object(value, index=not isinstance(value, pd.Index))
        columns = [str(c) for c in value.columns] if isinstance(value, pd.DataFrame) else None
        name = None if not isinstance(value, pd.Series) or value.name is None else str(value.name)
        return {"type": type(value).__name__, "shape": list(value.shape), "columns": columns,
                "name": name, "sha256": _sha256(hashes.to_numpy())}
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


#: Elements per block when a result matrix is read or rewritten a block of
#: rows at a time (32 MB of float64), so no step holds a whole result.
_BLOCK_ELEMENTS = 1 << 22


def _row_blocks(n_rows: int, n_genes: int, chunk_rows: int | None = None):
    """Slices covering ``n_rows`` rows of ``n_genes`` columns, about
    :data:`_BLOCK_ELEMENTS` elements each. With ``chunk_rows`` (the rows of
    an HDF5 chunk), each slice is whole bands of chunks, so no chunk is read
    twice."""
    step = max(1, _BLOCK_ELEMENTS // max(n_genes, 1))
    if chunk_rows:
        step = max(1, round(step / chunk_rows)) * chunk_rows
    for start in range(0, n_rows, step):
        yield slice(start, min(start + step, n_rows))


def _chunk_rows(dataset) -> int | None:
    """Rows in one HDF5 chunk of ``dataset`` (``None`` when contiguous or
    not an HDF5 dataset)."""
    chunks = getattr(dataset, "chunks", None)
    return chunks[0] if chunks else None


def _create_array(group: h5py.Group, name: str, **kwargs) -> h5py.Dataset:
    """``group.create_dataset`` tagged as an anndata dense array."""
    ds = group.create_dataset(name, **kwargs)
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"
    return ds


def _create_unfilled_array(
    group: h5py.Group, name: str, shape: tuple[int, ...], dtype, chunks: tuple[int, ...] | None = None,
) -> h5py.Dataset:
    """An anndata dense array whose storage is allocated now but never
    filled: creating it writes nothing (on disk it is a sparse hole until
    written), and the file's layout is final from the start, so filling it
    later only writes data -- a kill mid-write leaves a readable file.
    Unwritten elements read as zero. Contiguous, or in HDF5 ``chunks``
    (every chunk allocated up front, so the layout is still final)."""
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    if chunks is not None:
        dcpl.set_chunk(tuple(int(c) for c in chunks))
    return _create_array(group, name, shape=shape, dtype=dtype, dcpl=dcpl)


#: Bytes in one HDF5 chunk of a result written in column blocks.
_COLUMN_BLOCK_CHUNK_BYTES = 1 << 20


def _column_block_chunks(n_groups: int, n_genes: int, chunk_size: int) -> tuple[int, int]:
    """HDF5 chunk shape for a result written one gene chunk (a block of
    ``chunk_size`` columns) at a time: ``(rows, chunk_size)``, ~1 MiB.

    In a row-major contiguous dataset a gene chunk is one small write per
    group row (10,922 x 4.6 KB per layer on a genome-wide screen), each
    straddling partial pages that the filesystem must read before writing.
    On a network filesystem that read is a round trip: on HPC2021 Lustre a
    gene chunk took 219 s to write contiguously even with its pages cached,
    and 918 s when they were evicted between chunks, against 22 s in these
    chunks. On a local disk the two layouts write alike (the chunked one
    slightly faster), so it is used everywhere; reading one row of a large
    result reads its band of chunks (~2 ms locally), and small results are
    read into memory whole anyway.
    """
    width = max(1, min(int(chunk_size), n_genes))
    rows = max(1, min(n_groups, _COLUMN_BLOCK_CHUNK_BYTES // (width * 8)))
    return rows, width


class ResumableRun:
    """A result file filled in place, and a progress checkpoint, that outlive
    an interruption.

    The work file is :attr:`work_path`, ``result.h5ad`` in a hidden
    ``.{output name}.resume`` directory beside the output (a temporary
    directory would vanish with the process, taking the finished work with
    it). ``scaffold(path)`` creates it with every dataset the run fills,
    pre-sized; the run writes its results into :attr:`file` as they are
    computed, and :meth:`finish` moves the completed file to the output.
    Nothing is held in memory or mapped, so the run's footprint does not
    grow with the result.

    The checkpoint JSON records the call's :func:`run_fingerprint` plus
    caller-defined progress. :meth:`save` hands the file's pending data to the
    operating system and evicts its pages before writing the checkpoint, so
    everything a checkpoint lists as done survives the process being killed
    and little of it stays charged to the process's memory.

    A run resumes only when ``resume`` is set, the checkpoint's fingerprint
    matches this call (:func:`fingerprint_difference`), and the work file
    opens with every dataset in ``datasets`` at its expected shape and dtype.
    Otherwise the stale checkpoint and directory are discarded and the run
    starts fresh, so a checkpoint can never be paired with another run's
    data; a resume that throws away a checkpoint warns and says why.
    """

    def __init__(
        self,
        output_path: Path,
        checkpoint_path: Path,
        *,
        fingerprint: dict,
        resume: bool,
        scaffold: "Callable[[Path], None]",
        datasets: "dict[str, tuple[tuple[int, ...], np.typing.DTypeLike]]",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.fingerprint = fingerprint
        self.directory = output_path.with_name(f".{output_path.name}.resume")
        self.work_path = self.directory / "result.h5ad"
        self.file: h5py.File | None = None
        checkpoint = _read_checkpoint(checkpoint_path, required_keys=("fingerprint",)) if resume else None
        mismatch = (
            fingerprint_difference(checkpoint["fingerprint"], fingerprint)
            if checkpoint is not None and isinstance(checkpoint["fingerprint"], dict)
            else "it cannot be read"
        )
        if checkpoint is not None and mismatch is not None:
            warnings.warn(
                f"Not resuming from the checkpoint at {checkpoint_path}: {mismatch}. "
                "Discarding it and starting over.",
                stacklevel=3,
            )
        self.resumed = (
            checkpoint is not None and mismatch is None and _has_datasets(self.work_path, datasets)
        )
        self.progress = {k: v for k, v in checkpoint.items() if k != "fingerprint"} if self.resumed else {}
        if not self.resumed:
            self.discard()
            self.directory.mkdir(parents=True, exist_ok=True)
            scaffold(self.work_path)
        self.file = h5py.File(self.work_path, "r+")

    def save(self, **progress) -> None:
        """Put the work file's data on disk, then record ``progress`` in the
        checkpoint."""
        self.file.flush()
        drop_file_cache(self.work_path)
        _write_checkpoint_atomic(self.checkpoint_path, {"fingerprint": self.fingerprint, **progress})

    def close(self) -> None:
        """Close the work file, for the steps that finish it by path."""
        if self.file is not None:
            self.file.close()
            self.file = None
        drop_file_cache(self.work_path)

    def finish(self, output_path: Path) -> None:
        """Move the completed work file to ``output_path`` and remove the
        checkpoint and directory."""
        self.close()
        os.replace(self.work_path, output_path)
        self.discard()

    def discard(self) -> None:
        """Remove the checkpoint and the work file (after success, or when stale)."""
        if self.file is not None:
            self.file.close()
            self.file = None
        self.checkpoint_path.unlink(missing_ok=True)
        shutil.rmtree(self.directory, ignore_errors=True)


def _has_datasets(path: Path, datasets: dict) -> bool:
    """Whether the h5 file at ``path`` opens and holds every dataset in
    ``datasets`` with its ``(shape, dtype)``."""
    try:
        with h5py.File(path, "r") as handle:
            return all(
                name in handle
                and handle[name].shape == tuple(shape)
                and handle[name].dtype == np.dtype(dtype)
                for name, (shape, dtype) in datasets.items()
            )
    except (OSError, KeyError):  # absent, or a kill mid-write left it unreadable
        return False


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
