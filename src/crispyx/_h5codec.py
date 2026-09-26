"""Lossless HDF5 re-encoding with HDF5's standard shuffle + deflate filters.

HDF5's own filter pipeline compresses on one thread (~80-110 MB/s), so a
multi-hundred-GB file takes hours. This module produces the exact same
on-disk encoding -- a dataset declared ``compression="gzip", shuffle=True``
-- but byte-shuffles and deflates each chunk on a thread pool (``zlib``
releases the GIL) and hands the finished bytes to
``Dataset.id.write_direct_chunk``. Any HDF5 reader decodes the result with
its built-in filters: h5py, anndata/scanpy, R (rhdf5, zellkonverter),
``h5ls``. No plugin is involved. The reverse direction,
:func:`read_rows`, inflates deflate-compressed chunks on threads for
crispyx's streaming reads.

The tree walk knows nothing about AnnData: it copies every group, dataset,
link and attribute of an arbitrary HDF5 file, so it applies unchanged to any
``.h5ad`` layout, present or future. Verification compares the source and
the result through h5py itself (not through this module's decoder), so a
codec bug cannot hide behind a self-consistent round trip.
"""

from __future__ import annotations

import os
import threading
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator

import h5py
import joblib
import numpy as np

#: Target size of one HDF5 chunk. 1 MiB matches h5py's default chunk cache,
#: so a partial read (a row slice through anndata's backed mode, crispyx's
#: streaming readers) decompresses each chunk once instead of once per
#: touching read -- 4 MiB chunks measured 1.4x slower for sequential and 3x
#: slower for random row reads at the same compression ratio.
CHUNK_BYTES = 2**20

#: Datasets with fewer elements than this are copied unfiltered: a filter on
#: a tiny dataset costs more than it saves.
MIN_FILTERED_SIZE = 1024

#: Bytes read or compared per slab when a dataset is streamed whole.
_SLAB_BYTES = 16 * 2**20

#: Source chunk cache, so a source that is itself chunked/compressed with
#: chunks larger than h5py's 1 MiB default is not re-decoded per slab.
_SOURCE_RDCC_BYTES = 64 * 2**20

_FIXED_WIDTH_NUMERIC_KINDS = frozenset("biufc")


def _is_parallel_encodable(ds: h5py.Dataset) -> bool:
    """Whether *ds*'s in-memory bytes are exactly its HDF5 file layout.

    True for plain fixed-width numbers (incl. bool, stored as an int8 enum,
    and complex, stored as a packed compound). Strings, compounds, refs and
    vlen types go through h5py's own filter instead, where HDF5 converts the
    layout itself.
    """
    dt = ds.dtype
    return (
        dt.kind in _FIXED_WIDTH_NUMERIC_KINDS
        and dt.fields is None
        and h5py.check_vlen_dtype(dt) is None
        and h5py.check_ref_dtype(dt) is None
        and not dt.subdtype
    )


def _wants_filter(ds: h5py.Dataset) -> bool:
    """Whether *ds* is worth re-encoding with shuffle + gzip."""
    if ds.shape is None or ds.ndim == 0 or ds.size < MIN_FILTERED_SIZE:
        return False
    dt = ds.dtype
    string = h5py.check_string_dtype(dt)
    if string is not None:
        # Fixed-width strings compress well; vlen strings live on the global
        # heap, where a filter would only compress the heap pointers.
        return string.length is not None
    if dt.fields is not None:
        return all(f[0].kind != "O" for f in dt.fields.values())
    return dt.kind != "O"


def chunk_shape(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """~``CHUNK_BYTES`` chunk, whole along trailing axes while they fit.

    For a 1-D array this is ``CHUNK_BYTES / itemsize`` elements; for a dense
    matrix it is whole rows, so a row slice decompresses only the rows it
    needs. Only when a single row exceeds the budget are trailing axes split.
    """
    budget = max(1, CHUNK_BYTES // itemsize)
    chunks = [1] * len(shape)
    inner = 1
    for axis in range(len(shape) - 1, -1, -1):
        chunks[axis] = max(1, min(shape[axis], budget // inner))
        inner *= chunks[axis]
    return tuple(chunks)


def encode_chunk(chunk: np.ndarray, level: int) -> bytes:
    """Byte-shuffle then deflate one full chunk, as HDF5's pipeline does."""
    raw = np.ascontiguousarray(chunk).view(np.uint8).reshape(-1, chunk.dtype.itemsize)
    return zlib.compress(np.ascontiguousarray(raw.T), level)


def decode_chunk(blob: bytes, dtype: np.dtype, shape: tuple[int, ...], *, shuffle: bool = True) -> np.ndarray:
    """Inverse of :func:`encode_chunk` (``shuffle=False`` for a deflate-only
    pipeline)."""
    dtype = np.dtype(dtype)
    raw = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    if shuffle and dtype.itemsize > 1:
        raw = np.ascontiguousarray(raw.reshape(dtype.itemsize, -1).T)
    return raw.view(dtype).reshape(shape)


def _pad(block: np.ndarray, chunks: tuple[int, ...]) -> np.ndarray:
    """Zero-pad an edge block to the full chunk shape (HDF5 stores edge
    chunks full-size; the padding is never read back)."""
    if block.shape == chunks:
        return block
    out = np.zeros(chunks, dtype=block.dtype)
    out[tuple(slice(0, n) for n in block.shape)] = block
    return out


def _iter_chunks(src: h5py.Dataset, chunks: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Yield ``(offset, full_chunk)`` over *src* in chunk order."""
    shape = src.shape
    if tuple(chunks[1:]) == tuple(shape[1:]):
        # Chunks span whole trailing axes: read many chunks per h5py call.
        rows = chunks[0]
        row_bytes = max(1, int(np.prod(shape[1:], dtype=np.int64)) * src.dtype.itemsize)
        slab_rows = max(rows, (_SLAB_BYTES // row_bytes) // rows * rows)
        for start in range(0, shape[0], slab_rows):
            slab = src[start : start + slab_rows]
            for off in range(0, slab.shape[0], rows):
                yield (start + off,) + (0,) * (len(shape) - 1), _pad(slab[off : off + rows], chunks)
        return
    grid = [range(0, n, c) for n, c in zip(shape, chunks)]
    for offset in np.ndindex(*[len(g) for g in grid]):
        start = tuple(g[i] for g, i in zip(grid, offset))
        region = tuple(slice(s, s + c) for s, c in zip(start, chunks))
        yield start, _pad(src[region], chunks)


def _copy_attrs(src: h5py.HLObject, dst: h5py.HLObject) -> None:
    for name in src.attrs:
        aid = src.attrs.get_id(name)
        value = src.attrs[name]
        if isinstance(value, h5py.Empty):
            dst.attrs[name] = value
        else:
            dst.attrs.create(name, value, shape=aid.shape, dtype=aid.dtype)


def _tracks_order(obj: h5py.HLObject) -> bool:
    plist = obj.id.get_create_plist()
    flags = plist.get_link_creation_order() if isinstance(obj, h5py.Group) else plist.get_attr_creation_order()
    return bool(flags & h5py.h5p.CRT_ORDER_TRACKED)


def _copy_filtered(
    src: h5py.Dataset,
    parent: h5py.Group,
    name: str,
    *,
    level: int,
    pool: ThreadPoolExecutor,
    window: int,
    on_bytes: Callable[[int], None],
) -> None:
    chunks = chunk_shape(src.shape, src.dtype.itemsize)
    dst = parent.create_dataset(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=chunks,
        maxshape=src.maxshape,
        fillvalue=src.fillvalue,
        compression="gzip",
        compression_opts=level,
        shuffle=True,
        track_order=_tracks_order(src),
    )
    if not _is_parallel_encodable(src):
        # h5py's own (single-threaded) filter handles the layout conversion.
        row_bytes = max(1, int(np.prod(src.shape[1:], dtype=np.int64)) * src.dtype.itemsize)
        step = max(chunks[0], _SLAB_BYTES // row_bytes)
        for start in range(0, src.shape[0], step):
            block = src[start : start + step]
            dst[start : start + block.shape[0]] = block
            on_bytes(block.nbytes)
    else:
        pending: deque = deque()
        chunk_nbytes = int(np.prod(chunks, dtype=np.int64)) * src.dtype.itemsize

        def drain_one() -> None:
            offset, future = pending.popleft()
            dst.id.write_direct_chunk(offset, future.result(), filter_mask=0)
            on_bytes(chunk_nbytes)

        for offset, chunk in _iter_chunks(src, chunks):
            pending.append((offset, pool.submit(encode_chunk, chunk, level)))
            if len(pending) >= window:
                drain_one()
        while pending:
            drain_one()
    _copy_attrs(src, dst)


def copy_tree(
    src: h5py.Group,
    dst: h5py.Group,
    *,
    level: int,
    n_threads: int,
    on_bytes: Callable[[int], None] = lambda n: None,
) -> None:
    """Copy every link, group, dataset and attribute of *src* into *dst*,
    re-encoding each large fixed-width dataset with shuffle + gzip."""
    window = 2 * n_threads
    with ThreadPoolExecutor(n_threads) as pool:
        _copy_group(src, dst, level=level, pool=pool, window=window, on_bytes=on_bytes)


def _copy_group(src, dst, *, level, pool, window, on_bytes) -> None:
    _copy_attrs(src, dst)
    for name in src:
        link = src.get(name, getlink=True)
        if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
            dst[name] = link
            continue
        obj = src[name]
        if isinstance(obj, h5py.Group):
            child = dst.create_group(name, track_order=_tracks_order(obj))
            _copy_group(obj, child, level=level, pool=pool, window=window, on_bytes=on_bytes)
        elif isinstance(obj, h5py.Dataset) and _wants_filter(obj):
            _copy_filtered(obj, dst, name, level=level, pool=pool, window=window, on_bytes=on_bytes)
        else:
            # Unfiltered datasets and committed (named) datatypes.
            src.copy(obj, dst, name=name)
            if isinstance(obj, h5py.Dataset):
                on_bytes(obj.id.get_storage_size())


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _values_equal(a, b) -> bool:
    """Bit-for-bit equality (NaN payloads and signed zeros included) for
    fixed-width data; element equality for vlen/object data."""
    if isinstance(a, h5py.Empty) or isinstance(b, h5py.Empty):
        return isinstance(a, h5py.Empty) and isinstance(b, h5py.Empty) and a.dtype == b.dtype
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "O":
        return all(x == y for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype.itemsize in (1, 2, 4, 8):
        # Same-width unsigned-integer views compare the exact bits without
        # copying either array.
        as_uint = np.dtype(f"u{a.dtype.itemsize}")
        return np.array_equal(a.view(as_uint), b.view(as_uint))
    return a.tobytes() == b.tobytes()


def _check_attrs(src, dst, path: str) -> None:
    if list(src.attrs) != list(dst.attrs):
        raise ValueError(f"{path}: attribute names differ: {list(src.attrs)} vs {list(dst.attrs)}")
    for name in src.attrs:
        if src.attrs.get_id(name).get_type() != dst.attrs.get_id(name).get_type():
            raise ValueError(f"{path}: attribute {name!r} changed type")
        if not _values_equal(src.attrs[name], dst.attrs[name]):
            raise ValueError(f"{path}: attribute {name!r} changed value")


def verify_tree(src: h5py.Group, dst: h5py.Group, on_bytes: Callable[[int], None] = lambda n: None) -> None:
    """Raise ``ValueError`` naming the first difference between two files.

    Compares the full tree: link names and kinds, attribute names, order,
    HDF5 types and values, and every dataset's shape, HDF5 type and
    contents, block by block.
    """
    _check_attrs(src, dst, src.name)
    if list(src) != list(dst):
        raise ValueError(f"{src.name}: members differ: {list(src)} vs {list(dst)}")
    for name in src:
        path = f"{src.name.rstrip('/')}/{name}"
        s_link, d_link = src.get(name, getlink=True), dst.get(name, getlink=True)
        if type(s_link) is not type(d_link):
            raise ValueError(f"{path}: link kind differs")
        if isinstance(s_link, h5py.SoftLink):
            if s_link.path != d_link.path:
                raise ValueError(f"{path}: soft link target differs")
            continue
        if isinstance(s_link, h5py.ExternalLink):
            if (s_link.filename, s_link.path) != (d_link.filename, d_link.path):
                raise ValueError(f"{path}: external link target differs")
            continue
        s_obj, d_obj = src[name], dst[name]
        if isinstance(s_obj, h5py.Group) != isinstance(d_obj, h5py.Group):
            raise ValueError(f"{path}: group/dataset kind differs")
        if isinstance(s_obj, h5py.Group):
            verify_tree(s_obj, d_obj, on_bytes)
            continue
        if isinstance(s_obj, h5py.Datatype) != isinstance(d_obj, h5py.Datatype):
            raise ValueError(f"{path}: datatype/dataset kind differs")
        if isinstance(s_obj, h5py.Datatype):
            _check_attrs(s_obj, d_obj, path)
            if s_obj.id != d_obj.id:
                raise ValueError(f"{path}: named datatype differs")
            continue
        _check_attrs(s_obj, d_obj, path)
        if s_obj.shape != d_obj.shape or s_obj.maxshape != d_obj.maxshape:
            raise ValueError(f"{path}: shape differs")
        if s_obj.id.get_type() != d_obj.id.get_type():
            raise ValueError(f"{path}: HDF5 type differs")
        if s_obj.shape is None or s_obj.ndim == 0 or s_obj.size == 0:
            if not _values_equal(s_obj[()], d_obj[()]):
                raise ValueError(f"{path}: values differ")
            continue
        row_bytes = max(1, int(np.prod(s_obj.shape[1:], dtype=np.int64)) * s_obj.dtype.itemsize)
        step = max(1, _SLAB_BYTES // row_bytes)
        for start in range(0, s_obj.shape[0], step):
            a, b = s_obj[start : start + step], d_obj[start : start + step]
            if not _values_equal(a, b):
                raise ValueError(f"{path}: values differ in rows [{start}, {start + step})")
            on_bytes(a.nbytes)


def open_source(path) -> h5py.File:
    """Open *path* read-only with a chunk cache large enough that a source
    compressed with big chunks is not re-decoded for every slab."""
    return h5py.File(path, "r", rdcc_nbytes=_SOURCE_RDCC_BYTES, rdcc_nslots=10007)


def file_tracks_order(f: h5py.File) -> bool:
    # A File's own id yields the file-creation plist; the root group's
    # creation order lives on the root group itself.
    return _tracks_order(f["/"])


# ---------------------------------------------------------------------------
# Parallel decoding for streaming reads
# ---------------------------------------------------------------------------

def _decode_thread_count() -> int:
    """Threads for streaming decodes: every available CPU (up to 32), since
    one thread inflates only ~300-400 MB/s and the caller is blocked on the
    read anyway. ``OMP_NUM_THREADS`` caps it; joblib sets that in its worker
    processes, so a pool of workers does not start a full pool each."""
    n = min(32, joblib.cpu_count())
    try:
        return max(1, min(n, int(os.environ["OMP_NUM_THREADS"])))
    except (KeyError, ValueError):
        return n


# One pool shared by every reader in the process, so nested or concurrent
# streams never start more threads than it has.
_decode_pool: ThreadPoolExecutor | None = None
_decode_threads = 0
_decode_pool_lock = threading.Lock()


def _shared_decode_pool() -> tuple[ThreadPoolExecutor, int]:
    global _decode_pool, _decode_threads
    with _decode_pool_lock:
        if _decode_pool is None:
            _decode_threads = _decode_thread_count()
            _decode_pool = ThreadPoolExecutor(_decode_threads, thread_name_prefix="crispyx-inflate")
        return _decode_pool, _decode_threads


def _forget_decode_pool() -> None:
    # A forked child inherits the pool object but none of its threads.
    global _decode_pool, _decode_pool_lock
    _decode_pool = None
    _decode_pool_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_decode_pool)


def deflate_shuffle(ds: h5py.Dataset) -> bool | None:
    """``True``/``False`` if *ds* is chunked with exactly shuffle+deflate or
    deflate alone (the ``shuffle`` flag), ``None`` for any other layout."""
    if ds.chunks is None or not _is_parallel_encodable(ds):
        return None
    plist = ds.id.get_create_plist()
    filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    if filters == [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]:
        return True
    if filters == [h5py.h5z.FILTER_DEFLATE]:
        return False
    return None


def supports_parallel_read(ds: h5py.Dataset) -> bool:
    """Whether :func:`read_rows` can decode row ranges of *ds*: a deflate
    pipeline whose chunks span every axis but the first."""
    return (
        isinstance(ds, h5py.Dataset)
        and ds.ndim >= 1
        and deflate_shuffle(ds) is not None
        and tuple(ds.chunks[1:]) == tuple(ds.shape[1:])
    )


def read_rows(ds: h5py.Dataset, start: int, stop: int) -> np.ndarray:
    """``ds[start:stop]``, inflating the chunks on the shared decode pool.

    h5py inflates every chunk on the calling thread, so streaming a
    compressed file is CPU-bound at a few hundred MB/s. This reads the
    compressed chunks with ``read_direct_chunk`` and inflates them on
    threads. For a dataset :func:`supports_parallel_read` accepts, the result
    is identical to ``ds[start:stop]``, which it falls back to for anything
    it does not handle (a chunk HDF5 left unfiltered or never allocated).
    """
    stop = min(stop, ds.shape[0])
    if stop <= start:
        return ds[start:stop]
    pool, n_threads = _shared_decode_pool()
    # Chunks in flight: enough to keep every thread busy, few enough that a
    # read never holds a second full copy of the block it returns.
    window = 2 * n_threads
    shuffle = deflate_shuffle(ds)
    rows = ds.chunks[0]
    tail = (0,) * (ds.ndim - 1)
    out = np.empty((stop - start,) + ds.shape[1:], dtype=ds.dtype)
    pending: deque = deque()

    def place_one() -> None:
        first, future = pending.popleft()
        chunk = future.result()
        lo, hi = max(first, start), min(first + rows, stop)
        out[lo - start : hi - start] = chunk[lo - first : hi - first]

    for first in range(start - start % rows, stop, rows):
        try:
            mask, blob = ds.id.read_direct_chunk((first,) + tail)
        except (KeyError, ValueError, RuntimeError):  # chunk never written
            mask, blob = 1, None
        if mask:  # a filter was skipped or the chunk is missing: let HDF5 read it
            for _first, future in pending:
                future.cancel()
            return ds[start:stop]
        pending.append((first, pool.submit(decode_chunk, blob, ds.dtype, ds.chunks, shuffle=shuffle)))
        if len(pending) >= window:
            place_one()
    while pending:
        place_one()
    return out

