"""Helpers for working with AnnData ``.h5ad`` files in a streaming friendly way."""

from __future__ import annotations

import logging
import os as _os
import re as _re
import tempfile
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, NamedTuple, Sequence

import h5py
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _messages
from ._checkpoint import _create_progress_context
from ._disk import (
    assess_bytes,
    estimate_bytes,
    estimate_conversion_bytes,
    estimate_sparse_output_bytes,
    format_bytes,
    warn_if_disk_space_low,
)

logger = logging.getLogger(__name__)


def drop_file_cache(path: str | Path) -> None:
    """Advise the kernel to drop page cache for *path* (Linux only).

    On cgroup-limited systems (SLURM), page-cache pages count toward the
    memory limit.  Calling this after a streaming read prevents the cached
    file data from consuming the cgroup budget.

    The call is a no-op on non-Linux platforms or when the file cannot be
    opened.
    """
    try:
        fd = _os.open(str(path), _os.O_RDONLY)
        try:
            _os.posix_fadvise(fd, 0, 0, _os.POSIX_FADV_DONTNEED)
        finally:
            _os.close(fd)
    except (OSError, AttributeError):
        pass


from numba import njit, prange

# Numba-accelerated helpers for dense→CSR conversion (60x faster than scipy)
@njit(parallel=True)
def _numba_count_row_nnz(dense: np.ndarray) -> np.ndarray:
    """Count non-zeros per row using parallel numba."""
    n_rows = dense.shape[0]
    row_nnz = np.zeros(n_rows, dtype=np.int64)
    for i in prange(n_rows):
        count = 0
        for j in range(dense.shape[1]):
            if dense[i, j] != 0:
                count += 1
        row_nnz[i] = count
    return row_nnz

@njit(parallel=True)
def _numba_extract_csr_data(
    dense: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Extract CSR data/indices in parallel from dense array."""
    n_rows = dense.shape[0]
    for i in prange(n_rows):
        pos = indptr[i]
        for j in range(dense.shape[1]):
            val = dense[i, j]
            if val != 0:
                data[pos] = val
                indices[pos] = j
                pos += 1

ENSEMBL_PREFIXES = ("ENS", "FBgn", "YAL", "YBL", "YCL", "YDL", "YEL", "YFL", "YGL", "YHL", "YIL", "YJL", "YKL", "YLL", "YML", "YNL", "YOL", "YPL", "YQL", "YRL", "YSL", "YTL", "YUL", "YVL", "YWL", "YXL")


def is_dense_storage(path: str | Path) -> bool:
    """Check if h5ad file stores X matrix as dense array.
    
    Parameters
    ----------
    path
        Path to h5ad file.
        
    Returns
    -------
    bool
        True if X is stored as dense array, False if sparse (CSR/CSC).
    """
    with h5py.File(path, 'r') as f:
        if 'X' not in f:
            return False
        x_obj = f['X']
        if isinstance(x_obj, h5py.Dataset):
            # Dense array stored directly as dataset
            return True
        elif isinstance(x_obj, h5py.Group):
            # Check encoding-type attribute
            encoding = x_obj.attrs.get('encoding-type', b'')
            if isinstance(encoding, bytes):
                encoding = encoding.decode('utf-8')
            return encoding == 'array'
        return False


def get_matrix_storage_format(path: str | Path) -> str:
    """Detect matrix storage format in h5ad file.
    
    Parameters
    ----------
    path
        Path to h5ad file.
        
    Returns
    -------
    str
        One of: 'csr', 'csc', 'dense'
    """
    with h5py.File(path, 'r') as f:
        if 'X' not in f:
            return 'dense'
        x_obj = f['X']
        if isinstance(x_obj, h5py.Dataset):
            # Dense array stored directly as dataset
            return 'dense'
        elif isinstance(x_obj, h5py.Group):
            # Check encoding-type attribute
            encoding = x_obj.attrs.get('encoding-type', b'')
            if isinstance(encoding, bytes):
                encoding = encoding.decode('utf-8')
            if encoding == 'csc_matrix':
                return 'csc'
            elif encoding == 'csr_matrix':
                return 'csr'
            elif encoding == 'array':
                return 'dense'
        return 'dense'


_MISSING = object()


class _LazyFrameAccessor:
    """Provide a read-friendly view over ``obs``/``var`` tables with ``.load``."""

    def __init__(self, parent: "AnnData", attr: str) -> None:
        self._parent = parent
        self._attr = attr
        self._cache: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._cache is None:
            loaded = getattr(self._parent.to_memory(), self._attr)
            if isinstance(loaded, pd.DataFrame):
                loaded = loaded.copy()
            else:
                loaded = pd.DataFrame(loaded)
            self._cache = loaded
        return self._cache

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.load().head(n)

    def __len__(self) -> int:
        return len(self.load())

    def __iter__(self):  # pragma: no cover - passthrough for convenience
        return iter(self.load())

    def __getitem__(self, item):  # pragma: no cover - passthrough for convenience
        return self.load().__getitem__(item)

    def __getattr__(self, name: str):  # pragma: no cover - passthrough for convenience
        return getattr(self.load(), name)

    def __repr__(self) -> str:  # pragma: no cover - display preview
        frame = self.head()
        if frame.empty:
            return f"<{self._attr}: empty DataFrame>"
        return f"<{self._attr} preview>\n{frame}"


def _preview_uns_value(value: Any, n: int = 5) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.head(n)
    if isinstance(value, np.ndarray):
        return value[:n]
    if isinstance(value, Mapping):
        preview: dict[Any, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= n:
                preview["…"] = "…"
                break
            preview[key] = _preview_uns_value(item, n)
        return preview
    if isinstance(value, (list, tuple)):
        return type(value)(value[:n])
    return value


class _LazyUnsEntry:
    """Deferred loader for a single ``uns`` key."""

    def __init__(self, parent: "AnnData", key: str) -> None:
        self._parent = parent
        self._key = key
        self._cache: Any = _MISSING

    def load(self) -> Any:
        if self._cache is _MISSING:
            self._cache = self._parent.to_memory().uns[self._key]
        return self._cache

    def preview(self, n: int = 5) -> Any:
        return _preview_uns_value(self.load(), n)

    def __getattr__(self, name: str):  # pragma: no cover - passthrough for convenience
        return getattr(self.load(), name)

    def __getitem__(self, item):  # pragma: no cover - passthrough for convenience
        return self.load()[item]

    def __repr__(self) -> str:  # pragma: no cover - display preview
        return repr(self.preview())


class _LazyUnsMapping(Mapping[str, _LazyUnsEntry]):
    """Mapping-style accessor exposing ``uns`` keys with lazy loading."""

    def __init__(self, parent: "AnnData") -> None:
        self._parent = parent
        self._cache: dict[str, _LazyUnsEntry] = {}

    def _keys(self) -> list[str]:
        try:
            return list(self._parent.backed.uns.keys())
        except AttributeError:
            return []

    def __getitem__(self, key: str) -> _LazyUnsEntry:
        keys = self._keys()
        if key not in keys:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = _LazyUnsEntry(self._parent, key)
        return self._cache[key]

    def __iter__(self):  # pragma: no cover - passthrough for convenience
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def keys(self):  # pragma: no cover - convenience mirror
        return self._keys()

    def items(self):  # pragma: no cover - convenience mirror
        return [(key, self[key]) for key in self._keys()]

    def __repr__(self) -> str:  # pragma: no cover - display preview
        keys = self._keys()
        if not keys:
            return "<uns: empty>"
        previews = {key: self[key].preview() for key in keys}
        return repr(previews)


class AnnData:
    """Thin wrapper around a backed :class:`anndata.AnnData` handle."""

    def __init__(self, path: str | Path, *, mode: str = "r") -> None:
        """Open a backed AnnData wrapper.

        Parameters
        ----------
        path : str or Path
            Path to an ``.h5ad`` file.
        mode : str, optional
            HDF5 file access mode (default ``'r'``).
        """
        self.path = Path(path)
        self._mode = mode
        self._backed: ad.AnnData | None = None
        self._obs_view: _LazyFrameAccessor | None = None
        self._var_view: _LazyFrameAccessor | None = None
        self._uns_view: _LazyUnsMapping | None = None

    @property
    def filename(self) -> str:
        """Return the underlying filename for compatibility with Scanpy."""

        return str(self.path)

    def __fspath__(self) -> str:  # pragma: no cover - filesystem protocol
        return str(self.path)

    def __str__(self) -> str:  # pragma: no cover - helpful when printing paths
        return str(self.path)

    @property
    def backed(self) -> ad.AnnData:
        """Return the lazily opened backed AnnData handle."""

        if self._backed is None:
            self._backed = ad.read_h5ad(str(self.path), backed=self._mode)
        return self._backed

    def close(self) -> None:
        """Close the underlying file handle if it is open."""

        if self._backed is not None:
            try:
                self._backed.file.close()
            finally:
                self._backed = None
        self._obs_view = None
        self._var_view = None
        self._uns_view = None

    def to_memory(self) -> ad.AnnData:
        """Materialise the backed AnnData into memory."""

        return ad.read_h5ad(str(self.path))

    @property
    def obs(self) -> _LazyFrameAccessor:
        """Lazy accessor for observation (cell) metadata."""
        if self._obs_view is None:
            self._obs_view = _LazyFrameAccessor(self, "obs")
        return self._obs_view

    @property
    def var(self) -> _LazyFrameAccessor:
        """Lazy accessor for variable (gene) metadata."""
        if self._var_view is None:
            self._var_view = _LazyFrameAccessor(self, "var")
        return self._var_view

    @property
    def uns(self) -> _LazyUnsMapping:
        """Lazy accessor for unstructured annotations."""
        if self._uns_view is None:
            self._uns_view = _LazyUnsMapping(self)
        return self._uns_view

    def __enter__(self) -> "AnnData":
        self.backed  # ensure handle is opened
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __getattr__(self, name: str):
        # Guard: during unpickling __init__ has not run yet so _backed doesn't
        # exist.  Without this guard, accessing self.backed calls self._backed
        # which triggers __getattr__("_backed") → infinite recursion.
        if "_backed" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.backed, name)

    def __getstate__(self) -> dict:
        """Return picklable state (path + mode only; file handle is not serialised)."""
        return {"path": self.path, "_mode": self._mode}

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle; file handle will be reopened lazily on next access."""
        self.path = state["path"]
        self._mode = state["_mode"]
        self._backed = None
        self._obs_view = None
        self._var_view = None
        self._uns_view = None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"AnnData(path={self.path!s}, mode='{self._mode}')"

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


def read_backed(path: str | Path) -> ad.AnnData:
    """Open an ``.h5ad`` file in backed mode for low-memory access."""

    return ad.read_h5ad(str(path), backed="r")


# -----------------------------------------------------------------------------
# H5AD Write Helpers (for close-write-reopen pattern)
# -----------------------------------------------------------------------------


@contextmanager
def _h5ad_write_handle(path_or_file: "str | Path | h5py.File") -> Iterator[h5py.File]:
    """Yield an open-for-write h5py.File, accepting either a path or an
    already-open handle.

    Lets a caller writing many obsm/varm/obsp/varp keys in a loop open the
    destination once and pass the handle through, instead of every
    ``write_*_to_h5ad`` call reopening the same file.
    """
    if isinstance(path_or_file, h5py.File):
        yield path_or_file
    else:
        with h5py.File(path_or_file, "r+") as f:
            yield f


def write_obsm_to_h5ad(path: "str | Path | h5py.File", key: str, data: np.ndarray) -> None:
    """Write a dense array to obsm/{key} in an h5ad file.

    Parameters
    ----------
    path
        Path to the h5ad file, or an already-open ``h5py.File`` handle.
    key
        Key under obsm (e.g., 'X_pca').
    data
        Dense numpy array of shape (n_obs, n_dims).
    """
    with _h5ad_write_handle(path) as f:
        if "obsm" not in f:
            f.create_group("obsm")
        obsm = f["obsm"]
        if key in obsm:
            del obsm[key]
        ds = obsm.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"


def write_varm_to_h5ad(path: "str | Path | h5py.File", key: str, data: np.ndarray) -> None:
    """Write a dense array to varm/{key} in an h5ad file.

    Parameters
    ----------
    path
        Path to the h5ad file, or an already-open ``h5py.File`` handle.
    key
        Key under varm (e.g., 'PCs').
    data
        Dense numpy array of shape (n_vars, n_dims).
    """
    with _h5ad_write_handle(path) as f:
        if "varm" not in f:
            f.create_group("varm")
        varm = f["varm"]
        if key in varm:
            del varm[key]
        ds = varm.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"


def write_uns_dict_to_h5ad(path: str | Path, key: str, data: dict) -> None:
    """Write a dict to uns/{key} in an h5ad file.
    
    Handles scalar values, numpy arrays, and nested dicts.
    Uses AnnData-compatible encoding for proper round-trip compatibility.
    
    Parameters
    ----------
    path
        Path to the h5ad file.
    key
        Key under uns (e.g., 'pca').
    data
        Dictionary with string keys and scalar/array values.
    """
    # Variable-length string type for h5py
    str_dtype = h5py.string_dtype(encoding='utf-8')
    
    def _write_value(grp: h5py.Group, k: str, v):
        if k in grp:
            del grp[k]
        if isinstance(v, dict):
            sub = grp.create_group(k)
            for sub_k, sub_v in v.items():
                _write_value(sub, sub_k, sub_v)
        elif isinstance(v, np.ndarray):
            ds = grp.create_dataset(k, data=v)
            ds.attrs["encoding-type"] = "array"
            ds.attrs["encoding-version"] = "0.2.0"
        elif isinstance(v, (list, tuple)):
            arr = np.array(v)
            ds = grp.create_dataset(k, data=arr)
            ds.attrs["encoding-type"] = "array"
            ds.attrs["encoding-version"] = "0.2.0"
        elif isinstance(v, str):
            ds = grp.create_dataset(k, data=v, dtype=str_dtype)
            ds.attrs["encoding-type"] = "string"
            ds.attrs["encoding-version"] = "0.2.0"
        elif isinstance(v, bool):
            # Store bool as numpy bool_ to avoid confusion with int
            ds = grp.create_dataset(k, data=np.bool_(v))
            ds.attrs["encoding-type"] = "numeric-scalar"
            ds.attrs["encoding-version"] = "0.2.0"
        elif isinstance(v, (int, float, np.integer, np.floating)):
            ds = grp.create_dataset(k, data=v)
            ds.attrs["encoding-type"] = "numeric-scalar"
            ds.attrs["encoding-version"] = "0.2.0"
        else:
            # Fallback: try as string
            ds = grp.create_dataset(k, data=str(v), dtype=str_dtype)
            ds.attrs["encoding-type"] = "string"
            ds.attrs["encoding-version"] = "0.2.0"
    
    with h5py.File(path, "r+") as f:
        if "uns" not in f:
            f.create_group("uns")
        uns = f["uns"]
        if key in uns:
            del uns[key]
        grp = uns.create_group(key)
        for k, v in data.items():
            _write_value(grp, k, v)


def write_obsp_to_h5ad(path: "str | Path | h5py.File", key: str, data: sp.spmatrix) -> None:
    """Write a sparse matrix to obsp/{key} in an h5ad file.

    Stores in CSR format following AnnData conventions.

    Parameters
    ----------
    path
        Path to the h5ad file, or an already-open ``h5py.File`` handle.
    key
        Key under obsp (e.g., 'distances', 'connectivities').
    data
        Sparse matrix of shape (n_obs, n_obs).
    """
    _write_pairwise_matrix_to_h5ad(path, "obsp", key, data)


def write_varp_to_h5ad(path: "str | Path | h5py.File", key: str, data: sp.spmatrix) -> None:
    """Write a sparse matrix to varp/{key} in an h5ad file.

    Stores in CSR format following AnnData conventions.

    Parameters
    ----------
    path
        Path to the h5ad file, or an already-open ``h5py.File`` handle.
    key
        Key under varp (e.g., a gene-gene network/correlation matrix).
    data
        Sparse matrix of shape (n_vars, n_vars).
    """
    _write_pairwise_matrix_to_h5ad(path, "varp", key, data)


def _write_pairwise_matrix_to_h5ad(
    path: "str | Path | h5py.File", group_name: str, key: str, data: sp.spmatrix
) -> None:
    csr = sp.csr_matrix(data)

    with _h5ad_write_handle(path) as f:
        if group_name not in f:
            f.create_group(group_name)
        parent = f[group_name]
        if key in parent:
            del parent[key]

        grp = parent.create_group(key)
        grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")
        grp.attrs["shape"] = np.array(csr.shape, dtype=np.int64)
        grp.create_dataset("data", data=csr.data, compression="gzip", compression_opts=4)
        grp.create_dataset("indices", data=csr.indices)
        grp.create_dataset("indptr", data=csr.indptr)


def _copy_extra_slots(
    output_path: "str | Path",
    *,
    backed: ad.AnnData,
    obsm_keys: Sequence[str],
    varm_keys: Sequence[str],
    obsp_keys: Sequence[str],
    varp_keys: Sequence[str],
    cell_mask: np.ndarray | None,
    gene_mask: np.ndarray | None,
) -> None:
    """Copy obsm/varm/obsp/varp from an open backed source AnnData into
    ``output_path``, subsetting by ``cell_mask``/``gene_mask`` when given
    (``None`` means "copy unchanged", i.e. cell/gene identity doesn't
    change -- the case for :func:`downsample_counts`).

    Shared by :func:`write_filtered_subset` and :func:`downsample_counts` so
    the two don't independently re-implement (and risk drifting out of sync
    on) the same slot-preservation logic. All keys are written through one
    shared file handle rather than one open/close per key.
    """
    if not (obsm_keys or varm_keys or obsp_keys or varp_keys):
        return
    with h5py.File(output_path, "r+") as f:
        for key in obsm_keys:
            arr = np.asarray(backed.obsm[key])
            if cell_mask is not None:
                arr = arr[cell_mask]
            write_obsm_to_h5ad(f, key, arr)
        for key in varm_keys:
            arr = np.asarray(backed.varm[key])
            if gene_mask is not None:
                arr = arr[gene_mask]
            write_varm_to_h5ad(f, key, arr)
        for key in obsp_keys:
            mat = backed.obsp[key]
            mat = mat.tocsr() if sp.issparse(mat) else np.asarray(mat)
            if cell_mask is not None:
                mat = mat[cell_mask][:, cell_mask]
            write_obsp_to_h5ad(f, key, mat)
        for key in varp_keys:
            mat = backed.varp[key]
            mat = mat.tocsr() if sp.issparse(mat) else np.asarray(mat)
            if gene_mask is not None:
                mat = mat[gene_mask][:, gene_mask]
            write_varp_to_h5ad(f, key, mat)


def resolve_data_path(
    data: str | Path | "AnnData" | ad.AnnData,
    *,
    require_exists: bool = True,
) -> Path:
    """Resolve the on-disk path for a backed AnnData object or path-like input.
    
    This utility supports flexible input types for crispyx functions, allowing
    users to pass either a file path or an AnnData object.
    
    Parameters
    ----------
    data
        One of:
        - A string or Path to an h5ad file
        - A crispyx.AnnData wrapper (has .path attribute)
        - A backed anndata.AnnData object (has .filename attribute)
    require_exists
        If True (default), verify the resolved path exists.
        
    Returns
    -------
    Path
        The resolved file path.
        
    Raises
    ------
    TypeError
        If data is an in-memory (non-backed) AnnData or unsupported type.
    FileNotFoundError
        If require_exists is True and the path does not exist.
        
    Examples
    --------
    >>> from crispyx.data import resolve_data_path
    >>> path = resolve_data_path("data/counts.h5ad")
    >>> path = resolve_data_path(adata_wrapper)
    >>> path = resolve_data_path(backed_adata)
    """
    if isinstance(data, (str, Path)):
        result = Path(data)
    elif isinstance(data, AnnData):
        result = data.path
    elif isinstance(data, ad.AnnData):
        filename = getattr(data, "filename", None)
        if filename:
            result = Path(filename)
        else:
            raise TypeError(
                "Operations in crispyx expect a backed AnnData object or file path. "
                "The provided AnnData appears to be in-memory (no .filename attribute)."
            )
    else:
        raise TypeError(
            f"Expected a path-like value or backed AnnData; received {type(data)!r}."
        )
    
    if require_exists and not result.exists():
        raise FileNotFoundError(f"Data file not found: {result}")
    
    return result


def resolve_output_path(
    input_path: str | Path,
    *,
    suffix: str,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
) -> Path:
    """Construct an informative output path for an intermediate ``.h5ad`` file.

    Output names follow the pattern ``{stem}_cx_{suffix}.h5ad`` where *stem*
    is the input file's stem (or ``data_name`` when provided).  The ``_cx_``
    token sits between the input identity and the processing step so that:

    * filenames sort naturally by input first, then by operation;
    * the producing tool is unambiguously marked without cluttering the start
      of the filename.

    When ``output_path`` is provided it is returned directly, bypassing all
    name computation.  This lets callers enforce an exact output filename.

    .. deprecated::
        ``output_dir`` is kept for backward compatibility and will be removed
        in the next major version.  Use ``output_path`` instead.
    """
    if output_path is not None:
        return Path(output_path)

    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent

    if data_name:
        # If the provided name already encodes ``_cx_{suffix}``, use it as-is.
        # If it ends with ``_{suffix}`` (old-style, no cx marker), insert _cx.
        # Otherwise append ``_cx_{suffix}``.
        _cx_suffix = f"_cx_{suffix}"
        if data_name.endswith(_cx_suffix):
            base = data_name
        elif data_name.endswith(f"_{suffix}"):
            base = data_name[: -len(f"_{suffix}")] + _cx_suffix
        else:
            base = f"{data_name}{_cx_suffix}"
        return output_dir / f"{base}.h5ad"

    stem = input_path.stem
    if not stem:
        logger.warning(
            "resolve_output_path: input_path %r has no stem; "
            "output will be '_cx_%s.h5ad'",
            str(input_path),
            suffix,
        )
    return output_dir / f"{stem}_cx_{suffix}.h5ad"


def ensure_gene_symbol_column(
    adata: ad.AnnData | ad._core.anndata.AnnDataMixin,
    gene_name_column: str | None,
) -> pd.Index:
    """Return a vector of gene symbols and verify they look like symbols, not Ensembl IDs."""

    if gene_name_column is None:
        raw_names = adata.var_names
        logger.info(
            "No gene_name_column provided; using adata.var_names for gene identifiers."
        )
    else:
        if gene_name_column not in adata.var.columns:
            if gene_name_column == "gene_symbols":
                raw_names = adata.var_names
                logger.info(
                    "Column 'gene_symbols' not found in adata.var; "
                    "using adata.var_names for gene identifiers."
                )
            else:
                raise KeyError(
                    f"Gene name column '{gene_name_column}' was not found in adata.var. Available columns: {list(adata.var.columns)}"
                )
        else:
            raw_names = adata.var[gene_name_column]
    names = pd.Index(raw_names).astype(str)
    _validate_gene_symbols(names)
    return names


def _validate_gene_symbols(names: Sequence[str]) -> None:
    """Perform a basic sanity check that the provided gene identifiers look like symbols."""

    if len(names) == 0:
        raise ValueError("No gene names were provided.")
    names = pd.Index(names).astype(str)
    prefixes = names.str.upper().str.slice(0, 3)
    ensembl_like = prefixes.isin([p[:3] for p in ENSEMBL_PREFIXES]).sum()
    if ensembl_like > len(names) / 2:
        raise ValueError(
            "The majority of provided gene identifiers appear to be Ensembl-style IDs. "
            "Please supply a column containing gene symbols."
        )


def resolve_control_label(
    labels: Sequence[str],
    control_label: str | None,
    *,
    verbose: int | bool = True,
) -> str:
    """Return an explicit control label, inferring one when necessary."""

    if control_label is not None:
        return str(control_label)

    index = pd.Index(labels).astype(str)
    if index.empty:
        raise ValueError(
            "Cannot infer control label because no perturbation labels were provided."
        )
    lower = index.str.lower()

    exact_terms = {"ctrl", "control", "nontarget", "non-target", "non_target"}
    substring_terms = ("ctrl", "control", "nontarget", "non-target", "non_target")

    def _select(predicate) -> str | None:
        for label, lowered in zip(index, lower):
            if predicate(lowered):
                return str(label)
        return None

    candidate = _select(lambda text: text in exact_terms)
    if candidate is None:
        candidate = _select(lambda text: any(term in text for term in substring_terms))
    if candidate is None:
        candidate = _select(lambda text: ("non" in text) and ("target" in text))

    if candidate is None:
        raise ValueError(
            "Unable to infer control label automatically. Please provide 'control_label' explicitly."
        )

    if verbose:
        logger.info("Inferred control label '%s' from perturbation labels.", candidate)
    _messages.vprint(verbose, "resolve_control_label", f"Inferred control label '{candidate}'")
    return candidate


def read_h5ad_ondisk(
    path: str | Path,
    *,
    n_obs: int = 5,
    n_vars: int = 5,
) -> AnnData:
    """Open an ``.h5ad`` file on disk, print a preview, and return a read-only view."""

    adata_ro = AnnData(path)
    backed = adata_ro.backed
    try:
        print(backed)
        if n_obs > 0 and backed.n_obs > 0:
            print("First obs rows:")
            print(backed.obs.head(n_obs))
        if n_vars > 0 and backed.n_vars > 0:
            print("First var rows:")
            print(backed.var.head(n_vars))
    except Exception:
        adata_ro.close()
        raise
    return adata_ro


_SLOW_AXIS_WARNED: set[tuple[str, int]] = set()


def _detect_backed_sparse_format(X: Any) -> str | None:
    """Return 'csr'/'csc' for a backed sparse matrix, else None.

    Works across anndata versions where backed sparse datasets expose either
    a ``format`` or ``format_str`` attribute.
    """
    for attr in ("format", "format_str"):
        fmt = getattr(X, attr, None)
        if isinstance(fmt, str) and fmt in ("csr", "csc"):
            return fmt
    return None


def _warn_slow_axis(fmt: str, axis: int) -> None:
    """Emit a one-time warning when a backed matrix is accessed off its fast axis."""
    key = (fmt, axis)
    if key in _SLOW_AXIS_WARNED:
        return
    _SLOW_AXIS_WARNED.add(key)
    access = "row (axis=0)" if axis == 0 else "column (axis=1)"
    remedy = "convert_to_csr" if fmt == "csc" else "convert_to_csc"
    logger.warning(
        "Backed X is stored as %s but is being streamed by %s slices, which is "
        "O(total_nnz) per chunk and can be ~100x slower. Consider running "
        "cx.data.%s(path) once to store the matrix on its fast axis.",
        fmt.upper(), access, remedy,
    )


def _matrix_shape_and_nbytes(path: Path) -> tuple[tuple[int, int], int]:
    """Shape of ``X`` and the bytes a slow-axis chunk re-reads (``data`` + ``indices``)."""
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Group):
            shape = tuple(int(v) for v in x.attrs["shape"])
            return shape, int(x["data"].nbytes + x["indices"].nbytes)
        return tuple(int(v) for v in x.shape), int(x.nbytes)


def validate_format_mismatch_policy(policy: str) -> None:
    """Raise ``ValueError`` unless ``policy`` is one of the four accepted values.

    Every public entry point that takes ``format_mismatch_policy`` calls this
    first, so a typo fails immediately -- before a cached result is returned
    or a disk estimate silently drops its ``"scratch"`` entry.
    """
    if policy not in ("auto", "warn", "convert", "off"):
        raise ValueError(
            "format_mismatch_policy must be 'auto', 'warn', 'convert', or 'off', "
            f"got {policy!r}"
        )


# --- "auto" policy: is a temporary fast-axis copy worth its own read+write? ---
#
# Converting costs one full read of the source plus one full write; streaming
# off the fast axis costs one full read *per chunk*. Whether that trade pays
# off is not a property of the data but of the filesystem it sits on: a 500 MB
# file in page cache re-reads in ~0.1 s, so even 47 chunks are cheaper than the
# conversion's write, while a 40 GB screen on a network filesystem costs
# minutes per chunk and the conversion repays itself within a handful.
# Hence both gates below -- a structural one (enough repeated passes for the
# extra write to amortize at all) and a measured one (the saving is large in
# absolute terms, which only a genuinely slow read can produce). The rule errs
# towards streaming: declining costs at most the projected saving, whereas
# converting when reads are cheap costs a full write of the matrix.
_SLOW_AXIS_MIN_CHUNKS = 4
_SLOW_AXIS_MIN_SAVING_SECONDS = 60.0
#: Bounded prefix read used to measure the source's read throughput. Big
#: enough to amortize per-read latency on a network filesystem, small enough
#: to be free on a local one.
_THROUGHPUT_PROBE_BYTES = 64 << 20


def _measure_read_throughput(path: Path) -> float | None:
    """Bytes per second reading ``X``'s value array, or ``None`` if unmeasurable.

    Times a bounded prefix read rather than the whole array, so the probe
    costs the same on a 50 MB file and a 40 TB one.
    """
    try:
        with h5py.File(path, "r") as f:
            x = f["X"]
            data = x["data"] if isinstance(x, h5py.Group) else x
            if data.ndim != 1 or data.size == 0:
                return None
            itemsize = max(int(data.dtype.itemsize), 1)
            n_probe = max(1, min(int(data.size), _THROUGHPUT_PROBE_BYTES // itemsize))
            start = _time.perf_counter()
            _ = data[:n_probe]
            elapsed = _time.perf_counter() - start
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return (n_probe * itemsize) / max(elapsed, 1e-9)


class _AutoPolicy(NamedTuple):
    """What ``"auto"`` resolved to, and the numbers behind it."""

    policy: str  # "convert" | "warn" | "off"
    reason: str


def resolve_auto_format_mismatch_policy(
    path: Path, *, axis: int, chunk_size: int
) -> _AutoPolicy:
    """Resolve ``format_mismatch_policy="auto"`` for a mismatched source.

    Converts only when the projected cost of *not* converting -- one full
    re-read of ``X`` per chunk beyond the first -- is both structurally
    amortizable (:data:`_SLOW_AXIS_MIN_CHUNKS` chunks) and large in absolute
    terms (:data:`_SLOW_AXIS_MIN_SAVING_SECONDS`). Declining resolves to
    ``"warn"`` when the projected cost is material anyway (so the user hears
    about it and can convert the file once, up front) and to ``"off"`` when it
    is negligible, which keeps small local runs silent.

    Callers must have established that ``path``'s storage format really is
    mismatched with ``axis``; :func:`estimate_disk_usage` calls this too, so
    its ``"scratch"`` entry reflects what the real call will do.
    """
    shape, nbytes = _matrix_shape_and_nbytes(path)
    n_chunks = max(1, (shape[axis] + chunk_size - 1) // chunk_size)
    throughput = _measure_read_throughput(path)
    if throughput is None:
        return _AutoPolicy(
            "warn",
            "could not measure read throughput for the source; streaming from it "
            "rather than converting on a guess",
        )
    per_chunk_seconds = nbytes / throughput
    projected_seconds = (n_chunks - 1) * per_chunk_seconds
    measurement = (
        f"{n_chunks} chunks x {format_bytes(nbytes)} at "
        f"{format_bytes(throughput)}/s = {per_chunk_seconds:.2f} s per re-read, "
        f"{projected_seconds:.1f} s of re-reading beyond the first chunk"
    )
    if n_chunks >= _SLOW_AXIS_MIN_CHUNKS and projected_seconds >= _SLOW_AXIS_MIN_SAVING_SECONDS:
        return _AutoPolicy("convert", f"converting ({measurement})")
    if projected_seconds >= _SLOW_AXIS_MIN_SAVING_SECONDS:
        return _AutoPolicy(
            "warn",
            f"streaming from the source ({measurement}); fewer than "
            f"{_SLOW_AXIS_MIN_CHUNKS} chunks, so one conversion would not repay "
            "its own write",
        )
    return _AutoPolicy("off", f"streaming from the source ({measurement})")


def scratch_copy_bytes(
    path: str | Path, *, axis: int, policy: str, chunk_size: int
) -> float | None:
    """Disk the temporary fast-axis copy will need, or ``None`` for no copy.

    The one place the disk-usage resolvers in ``de.py``, ``batch.py`` and this
    module ask "will this call convert?", so an estimate can never disagree
    with what the run then does -- including under ``"auto"``, where the
    answer depends on measured throughput rather than on the policy alone.
    """
    validate_format_mismatch_policy(policy)
    path = Path(path)
    fmt = get_matrix_storage_format(path)
    if not ((fmt == "csr" and axis == 1) or (fmt == "csc" and axis == 0)):
        return None
    if policy in ("warn", "off"):
        return None
    if policy == "auto":
        resolved = resolve_auto_format_mismatch_policy(
            path, axis=axis, chunk_size=chunk_size
        )
        if resolved.policy != "convert":
            return None
    return estimate_conversion_bytes(path)


def _slow_axis_cost_message(path: Path, *, fmt: str, axis: int, chunk_size: int) -> str:
    """Quantify what streaming ``path`` off its fast axis costs, with the remedy."""
    target = "csc" if axis == 1 else "csr"
    access = "cell (row)" if axis == 0 else "gene (column)"
    shape, nbytes = _matrix_shape_and_nbytes(path)
    n_chunks = max(1, (shape[axis] + chunk_size - 1) // chunk_size)
    return (
        f"X is stored as {fmt.upper()} but is streamed by {access} chunks, so each "
        f"of the {n_chunks} chunks re-reads the whole matrix "
        f"({format_bytes(nbytes)} of data+indices, "
        f"~{format_bytes(nbytes * n_chunks)} in total) and holds it in memory. "
        f"Pass format_mismatch_policy='convert' to stream from a temporary "
        f"{target.upper()} copy instead, or run cx.pp.convert_to_{target}(path) "
        "once if several steps will reuse the file."
    )


#: Prefix of the temporary fast-axis copies :func:`stream_on_fast_axis`
#: writes. Dot-prefixed so it stays out of the way in an output directory,
#: and distinctive enough that the sweep below can only ever match our own.
_SCRATCH_PREFIX = ".cx_"
#: A temporary copy whose owning process is gone is only removed once it is
#: this old, so a recycled PID cannot make a live run's copy look abandoned.
_STALE_SCRATCH_SECONDS = 24 * 60 * 60


def _process_is_alive(pid: int) -> bool:
    """Whether ``pid`` names a running process; ``True`` when unknowable."""
    try:
        _os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (OSError, OverflowError, ValueError):
        return True  # platform can't tell us -- never delete on a guess
    return True


def _sweep_stale_scratch_copies(
    scratch_dir: Path, *, fn_name: str, verbose: int | bool = False
) -> None:
    """Remove temporary fast-axis copies left behind by dead runs.

    The copies are deleted in a ``finally`` block, which a ``SIGKILL`` (an
    out-of-memory kill, a scheduler timeout) skips -- leaving a hidden file
    the size of the source matrix next to the output. Nothing else would ever
    clean those up: ``atexit`` does not run on ``SIGKILL`` either. So each
    conversion first sweeps the copies of runs that are demonstrably gone.
    """
    try:
        entries = list(scratch_dir.glob(f"{_SCRATCH_PREFIX}*"))
    except OSError:
        return
    now = _time.time()
    for entry in entries:
        # ".cx_{fn}_{pid}_{random}.{fmt}.h5ad" and its ".partial" sibling.
        match = _re.match(rf"^{_re.escape(_SCRATCH_PREFIX)}.+?_(\d+)_", entry.name)
        if match is None:
            continue
        pid = int(match.group(1))
        if pid == _os.getpid():
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if _process_is_alive(pid) and age < _STALE_SCRATCH_SECONDS:
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        _messages.vprint(
            verbose, fn_name,
            f"removed a temporary copy left by an interrupted run: {entry.name}",
        )


@contextmanager
def stream_on_fast_axis(
    path: Path,
    *,
    axis: int,
    policy: str,
    fn_name: str,
    scratch_dir: Path,
    chunk_size: int,
    memory_limit_gb: float | None = None,
    verbose: int | bool = False,
) -> Iterator[Path]:
    """Yield the path to stream ``axis`` chunks of ``X`` from.

    A CSR matrix streamed by columns (``axis=1``), or a CSC one streamed by
    rows (``axis=0``), makes anndata read the *whole* ``data``/``indices``
    arrays for every chunk -- one full read of the file per chunk instead of
    one in total, plus the matrix's full size in transient memory. ``policy``
    (the public ``format_mismatch_policy``) decides what to do about that:

    * ``"auto"``: measure what the re-reads would actually cost on this
      filesystem and convert only when that is worth an extra read and write
      (see :func:`resolve_auto_format_mismatch_policy`), then behave as the
      resolved policy below. This is the default everywhere.
    * ``"convert"``: write a temporary fast-axis copy under ``scratch_dir``
      (:func:`convert_to_csc` / :func:`convert_to_csr`, output buffers bounded
      by ``memory_limit_gb``), yield its path, and delete it on exit. When the
      free space under ``scratch_dir`` cannot hold that copy, the call falls
      back to ``"warn"`` (one :class:`UserWarning` naming the shortfall) rather
      than failing with ``ENOSPC`` after minutes of I/O.
    * ``"warn"``: yield ``path`` after one :class:`UserWarning` quantifying the
      cost (and the one-time ``crispyx.data`` logger line).
    * ``"off"``: yield ``path`` silently.

    A source already stored on its fast axis, or dense, is yielded unchanged.
    The yielded path holds ``X`` only (plus ``obs``/``var``): read ``uns``,
    ``layers``, ``obsm`` etc. from the original ``path``. Callers have made the
    format decision here, so they stream the result with
    ``iter_matrix_chunks(..., warn_slow_axis=False)``.
    """
    validate_format_mismatch_policy(policy)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    path = Path(path)
    fmt = get_matrix_storage_format(path)
    if not ((fmt == "csr" and axis == 1) or (fmt == "csc" and axis == 0)):
        yield path
        return

    if policy == "auto":
        resolved = resolve_auto_format_mismatch_policy(
            path, axis=axis, chunk_size=chunk_size
        )
        _messages.vprint(
            verbose, fn_name, f"format_mismatch_policy='auto': {resolved.reason}"
        )
        policy = resolved.policy

    if policy == "off":
        yield path
        return

    target = "csc" if axis == 1 else "csr"
    access = "cell (row)" if axis == 0 else "gene (column)"
    scratch_dir = Path(scratch_dir)
    if policy == "convert":
        scratch_dir.mkdir(parents=True, exist_ok=True)
        _sweep_stale_scratch_copies(scratch_dir, fn_name=fn_name, verbose=verbose)
        disk = assess_bytes(estimate_conversion_bytes(path), scratch_dir)
        if not disk.sufficient:
            _messages.warn(
                fn_name,
                f"the temporary {target.upper()} copy needs "
                f"~{format_bytes(disk.required_bytes)} at {disk.path} but only "
                f"{format_bytes(disk.free_bytes)} are free; streaming from the "
                f"{fmt.upper()} source instead (as format_mismatch_policy='warn' would). "
                f"Point output_path at a volume with more room, or run "
                f"cx.pp.convert_to_{target}(path) there once. "
                + _slow_axis_cost_message(path, fmt=fmt, axis=axis, chunk_size=chunk_size),
                stacklevel=4,
            )
            policy = "warn"
    if policy == "warn":
        _messages.warn(
            fn_name,
            _slow_axis_cost_message(path, fmt=fmt, axis=axis, chunk_size=chunk_size),
            stacklevel=4,
        )
        # The one-time logger guardrail, emitted once per process.
        _warn_slow_axis(fmt, axis)
        yield path
        return

    # The PID in the name is what lets a later run recognise this copy as
    # abandoned; see _sweep_stale_scratch_copies.
    fd, tmp_name = tempfile.mkstemp(
        dir=scratch_dir,
        prefix=f"{_SCRATCH_PREFIX}{fn_name}_{_os.getpid()}_",
        suffix=f".{target}.h5ad",
    )
    _os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _messages.vprint(
            verbose, fn_name,
            f"X is {fmt.upper()}; converting to a temporary {target.upper()} copy for "
            f"{access} streaming → {tmp_path}",
        )
        convert = convert_to_csc if target == "csc" else convert_to_csr
        convert(path, output_path=tmp_path, memory_limit_gb=memory_limit_gb, verbose=verbose)
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def iter_matrix_chunks(
    adata: ad.AnnData | ad._core.anndata.AnnDataMixin,
    *,
    axis: int = 0,
    chunk_size: int = 1024,
    convert_to_dense: bool = True,
    matrix: np.ndarray | sp.spmatrix | None = None,
    start_chunk: int = 0,
    warn_slow_axis: bool = True,
) -> Iterator[tuple[slice, np.ndarray | sp.spmatrix]]:
    """Yield chunks of ``matrix`` (defaults to ``adata.X``).

    Pass e.g. ``matrix=adata.layers[key]`` to stream a layer with the same
    chunking and slow-axis-format dispatch used for ``X``. ``start_chunk``
    skips directly to that 0-indexed chunk (e.g. for resuming a previously
    interrupted run) without reading the skipped chunks from disk.
    ``warn_slow_axis=False`` suppresses the one-time logger warning about
    streaming a compressed matrix off its fast axis; callers that already
    resolved that decision through :func:`stream_on_fast_axis` pass it.
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    if matrix is None:
        matrix = adata.X
    fmt = _detect_backed_sparse_format(matrix)
    if warn_slow_axis and ((fmt == "csc" and axis == 0) or (fmt == "csr" and axis == 1)):
        _warn_slow_axis(fmt, axis)
    n_obs, n_vars = adata.n_obs, adata.n_vars
    length = n_obs if axis == 0 else n_vars
    for start in range(start_chunk * chunk_size, length, chunk_size):
        end = min(start + chunk_size, length)
        if axis == 0:
            block = matrix[start:end]
        else:
            block = matrix[:, start:end]
        if convert_to_dense:
            block = _to_dense(block)
        yield slice(start, end), block


def _to_dense(matrix: np.ndarray) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def normalize_total_block(
    block: np.ndarray | sp.spmatrix,
    *,
    library_size: np.ndarray | None = None,
    target_sum: float = 1e4,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a library-size normalised dense view of ``block``.

    Parameters
    ----------
    block:
        A slice of the expression matrix with shape ``(n_cells, n_genes)``.
    library_size:
        Optional precomputed library sizes for the cells in ``block``. When
        ``None`` the library size is computed from ``block`` directly.
    target_sum:
        Target total counts per cell after normalisation, matching the default
        used by :func:`scanpy.pp.normalize_total`.
    dtype:
        Data type for the output dense array. Defaults to float64.

    Returns
    -------
    tuple
        A tuple ``(normalised, library_size)`` where ``normalised`` is a dense
        array containing the normalised counts and ``library_size``
        contains the per-cell library sizes that were used.
    """

    dense = _to_dense(block).astype(dtype, copy=True)
    if dense.ndim != 2:
        raise ValueError("block must be two-dimensional")

    if library_size is None:
        library_size = dense.sum(axis=1)
    else:
        library_size = np.asarray(library_size, dtype=dtype)
        if library_size.shape[0] != dense.shape[0]:
            raise ValueError(
                "library_size length does not match the number of cells in block"
            )

    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=dtype),
        where=library_size > 0,
    )
    dense *= scale[:, None]
    return dense, library_size


def _ensure_csr(matrix: np.ndarray | sp.spmatrix, *, dtype: np.dtype | None = None) -> sp.csr_matrix:
    """Convert the provided matrix to CSR format."""

    if sp.isspmatrix_csr(matrix):
        csr = matrix
    elif sp.issparse(matrix):
        csr = matrix.tocsr()
    else:
        csr = sp.csr_matrix(np.asarray(matrix))
    if dtype is not None and csr.dtype != dtype:
        csr = csr.astype(dtype)
    return csr


def _extract_csr_components_dense(
    block: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract CSR data, indices, and row_nnz from dense block efficiently.
    
    Uses numba-accelerated parallel extraction when available (60x faster),
    falling back to numpy vectorized operations otherwise.
    
    Parameters
    ----------
    block
        Dense 2D array of shape (n_rows, n_cols).
    dtype
        Target dtype for data array.
        
    Returns
    -------
    tuple
        (data, indices, row_nnz, total_nnz) where data and indices are flattened
        CSR components and row_nnz is counts per row.
    """
    if block.size == 0:
        empty_data = np.array([], dtype=dtype)
        empty_indices = np.array([], dtype=np.int32)
        empty_nnz = np.zeros(block.shape[0], dtype=np.int64)
        return empty_data, empty_indices, empty_nnz, 0
    
    # Ensure C-contiguous for numba
    if not block.flags['C_CONTIGUOUS']:
        block = np.ascontiguousarray(block)

    # Use numba-accelerated parallel extraction (60x faster than scipy)
    row_nnz = _numba_count_row_nnz(block)
    indptr = np.zeros(block.shape[0] + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_nnz)
    total_nnz = int(indptr[-1])

    if total_nnz == 0:
        return np.array([], dtype=dtype), np.array([], dtype=np.int32), row_nnz, 0

    data = np.empty(total_nnz, dtype=dtype)
    indices = np.empty(total_nnz, dtype=np.int32)
    _numba_extract_csr_data(block.astype(dtype, copy=False), indptr, data, indices)

    return data, indices, row_nnz, total_nnz


def _stream_write_matrix_subset(
    source_path: Path,
    *,
    read_matrix,
    dest_group: str,
    cell_mask: np.ndarray,
    gene_indices: np.ndarray,
    n_obs: int,
    n_vars: int,
    output_path: Path,
    chunk_size: int,
    row_nnz: np.ndarray | None = None,
    total_nnz: int | None = None,
    data_dtype: np.dtype | None = None,
    chunk_cache: Any = None,
) -> None:
    """Stream one CSR-shaped matrix (``X`` or a layer) filtered by
    ``cell_mask``/``gene_indices`` into ``dest_group`` of ``output_path``.

    ``read_matrix`` extracts the source matrix from a freshly-opened backed
    AnnData (``lambda b: b.X`` or ``lambda b: b.layers[key]``). Shared by
    ``write_filtered_subset`` for ``X`` (where a ``chunk_cache`` from
    ``qc.py``'s gene-filter pass can skip re-reading the source) and for each
    ``layers`` key (which always re-reads, since that cache is keyed to ``X``
    only).
    """
    need_counting_pass = row_nnz is None or total_nnz is None or data_dtype is None

    if need_counting_pass:
        row_nnz = np.zeros(n_obs, dtype=np.int64)
        total_nnz = 0
        data_dtype_local: np.dtype | None = None

        backed = read_backed(source_path)
        try:
            matrix = read_matrix(backed)
            row_offset = 0
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=False, matrix=matrix
            ):
                local_mask = cell_mask[slc]
                if not np.any(local_mask):
                    continue
                block = block[local_mask]
                if gene_indices.size:
                    block = block[:, gene_indices]
                else:
                    block = block[:, []]
                csr = _ensure_csr(block)
                counts = np.diff(csr.indptr)
                size = counts.size
                row_nnz[row_offset : row_offset + size] = counts
                total_nnz += int(csr.nnz)
                if data_dtype_local is None and csr.nnz:
                    data_dtype_local = csr.data.dtype
                row_offset += size
        finally:
            backed.file.close()

        if data_dtype_local is None:
            data_dtype_local = np.float32
        data_dtype = data_dtype_local

    warn_if_disk_space_low(
        estimate_sparse_output_bytes(total_nnz), output_path,
        context=f"write_filtered_subset ({dest_group})",
    )

    with h5py.File(output_path, "r+", libver='latest') as dest:
        if dest_group in dest:
            del dest[dest_group]
        grp = dest.create_group(dest_group)
        grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")

        if n_obs == 0 or n_vars == 0:
            grp.create_dataset("data", shape=(0,), dtype=data_dtype)
            grp.create_dataset("indices", shape=(0,), dtype=np.int32)
            grp.create_dataset("indptr", data=np.zeros(n_obs + 1, dtype=np.int64))
            grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)
            return

        indptr = np.zeros(n_obs + 1, dtype=np.int64)
        np.cumsum(row_nnz, out=indptr[1:])

        # Phase 2 I/O Optimization: Use larger HDF5 chunks and write buffering
        # Optimal chunk size: balance between I/O overhead and cache efficiency
        # Target ~1MB chunks for data (assuming float32 = 4 bytes -> ~256K elements)
        hdf5_chunk_size = min(262144, max(8192, total_nnz // 16))  # 8K to 256K elements

        # Write buffer size: accumulate data before writing to reduce I/O syscalls
        write_buffer_size = hdf5_chunk_size * 2  # Buffer 2x chunk size before flushing

        # Use explicit chunk sizes for better I/O performance
        data_ds = grp.create_dataset(
            "data",
            shape=(total_nnz,),
            dtype=data_dtype,
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(total_nnz,),
            dtype=np.int32,
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None
        )
        grp.create_dataset("indptr", data=indptr)
        grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)

        # Stream data with write buffering
        if chunk_cache is not None:
            # Read from cached CSR chunks (avoids re-reading the original matrix)
            offset = 0
            data_buffer = []
            indices_buffer = []
            buffer_nnz = 0

            for filtered_data, filtered_indices, n_cells in chunk_cache.iter_filtered_chunks(
                gene_indices, data_dtype
            ):
                nnz = len(filtered_data)
                if nnz:
                    data_buffer.append(filtered_data)
                    indices_buffer.append(filtered_indices)
                    buffer_nnz += nnz

                    # Flush buffer when it exceeds threshold
                    if buffer_nnz >= write_buffer_size:
                        combined_data = np.concatenate(data_buffer)
                        combined_indices = np.concatenate(indices_buffer)
                        data_ds[offset : offset + buffer_nnz] = combined_data
                        indices_ds[offset : offset + buffer_nnz] = combined_indices
                        offset += buffer_nnz
                        data_buffer = []
                        indices_buffer = []
                        buffer_nnz = 0

            # Flush remaining buffer
            if buffer_nnz > 0:
                combined_data = np.concatenate(data_buffer)
                combined_indices = np.concatenate(indices_buffer)
                data_ds[offset : offset + buffer_nnz] = combined_data
                indices_ds[offset : offset + buffer_nnz] = combined_indices
        else:
            # Read from source matrix (fallback when cache not available)
            backed = read_backed(source_path)
            try:
                matrix = read_matrix(backed)
                offset = 0
                data_buffer = []
                indices_buffer = []
                buffer_nnz = 0

                for slc, block in iter_matrix_chunks(
                    backed, axis=0, chunk_size=chunk_size, convert_to_dense=False, matrix=matrix
                ):
                    local_mask = cell_mask[slc]
                    if not np.any(local_mask):
                        continue
                    block = block[local_mask]
                    if gene_indices.size:
                        block = block[:, gene_indices]
                    else:
                        block = block[:, []]

                    # Use optimized extraction for dense, scipy for sparse
                    if sp.issparse(block):
                        csr = _ensure_csr(block, dtype=data_dtype)
                        chunk_data = csr.data
                        chunk_indices = csr.indices.astype(np.int32, copy=False)
                        nnz = int(csr.nnz)
                    else:
                        # Dense: use numba-accelerated direct extraction
                        chunk_data, chunk_indices, _row_nnz, nnz = _extract_csr_components_dense(
                            block, data_dtype
                        )

                    if nnz:
                        data_buffer.append(chunk_data)
                        indices_buffer.append(chunk_indices)
                        buffer_nnz += nnz

                        # Flush buffer when it exceeds threshold
                        if buffer_nnz >= write_buffer_size:
                            combined_data = np.concatenate(data_buffer)
                            combined_indices = np.concatenate(indices_buffer)
                            data_ds[offset : offset + buffer_nnz] = combined_data
                            indices_ds[offset : offset + buffer_nnz] = combined_indices
                            offset += buffer_nnz
                            data_buffer = []
                            indices_buffer = []
                            buffer_nnz = 0

                # Flush remaining buffer
                if buffer_nnz > 0:
                    combined_data = np.concatenate(data_buffer)
                    combined_indices = np.concatenate(indices_buffer)
                    data_ds[offset : offset + buffer_nnz] = combined_data
                    indices_ds[offset : offset + buffer_nnz] = combined_indices
            finally:
                backed.file.close()


def write_filtered_subset(
    source_path: str | Path,
    *,
    cell_mask: np.ndarray,
    gene_mask: np.ndarray,
    output_path: str | Path,
    chunk_size: int = 4096,
    var_assignments: dict[str, Sequence] | None = None,
    row_nnz: np.ndarray | None = None,
    total_nnz: int | None = None,
    data_dtype: np.dtype | None = None,
    chunk_cache: Any = None,
) -> None:
    """Stream a filtered AnnData view to disk without materialising ``X``.

    Every slot is carried through the filter, not just ``X``/``obs``/``var``:
    ``layers`` are streamed the same way as ``X`` (chunked, never fully
    materialised); ``obsm``/``varm``/``obsp``/``varp``/``uns`` are copied in
    memory (they are small relative to ``X`` — dense embeddings, sparse
    cell-cell/gene-gene graphs, or global metadata) and subset on whichever
    axis applies (``obsm``/``obsp`` by ``cell_mask``, ``varm``/``varp`` by
    ``gene_mask``). A source ``.raw`` is not copied — a warning is emitted
    instead, since regenerating it from the source is the caller's
    responsibility.

    Parameters
    ----------
    source_path
        Path to source h5ad file.
    cell_mask
        Boolean mask for cells to include.
    gene_mask
        Boolean mask for genes to include.
    output_path
        Path for output h5ad file.
    chunk_size
        Number of cells to process per chunk.
    var_assignments
        Optional dict of column assignments for var DataFrame.
    row_nnz
        Optional pre-computed non-zero counts per row for ``X``. When
        provided along with total_nnz and data_dtype, skips ``X``'s counting
        pass (layers always run their own counting pass).
    total_nnz
        Optional pre-computed total non-zero count for ``X``.
    data_dtype
        Optional pre-computed data type for ``X``'s sparse matrix.
    chunk_cache
        Optional _ChunkCache object from qc module. When provided, reads
        ``X``'s CSR data from cache instead of re-reading the source matrix.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backed = read_backed(source_path)
    try:
        obs = backed.obs.iloc[cell_mask].copy()
        var = backed.var.iloc[gene_mask].copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        # Drop stale categories so downstream tools (e.g. scanpy) only see
        # groups that have at least one cell in the filtered subset.
        for _col in obs.columns:
            if isinstance(obs[_col].dtype, pd.CategoricalDtype):
                obs[_col] = obs[_col].cat.remove_unused_categories()
        if var_assignments:
            for key, values in var_assignments.items():
                if len(values) != var.shape[0]:
                    raise ValueError(
                        f"Length mismatch for column '{key}': expected {var.shape[0]}, received {len(values)}"
                    )
                var[key] = np.asarray(values)
        uns = dict(backed.uns)
        # anndata >= 0.13 exposes X itself as layers[None]; only real,
        # named layers should be streamed/copied here.
        layer_keys = [key for key in backed.layers.keys() if key is not None]
        obsm_keys = list(backed.obsm.keys())
        varm_keys = list(backed.varm.keys())
        obsp_keys = list(backed.obsp.keys())
        varp_keys = list(backed.varp.keys())
        has_raw = backed.raw is not None
    finally:
        backed.file.close()

    if has_raw:
        _messages.warn(
            "write_filtered_subset",
            "source AnnData has a `.raw` layer; it is not copied to the "
            "filtered/subsampled output. Regenerate it from the source file "
            "separately if you need it.",
        )

    n_obs = int(cell_mask.sum())
    n_vars = int(gene_mask.sum())

    gene_indices = np.flatnonzero(gene_mask)

    # Use pre-computed X values if all three are provided, otherwise compute
    need_counting_pass = row_nnz is None or total_nnz is None or data_dtype is None
    x_data_dtype = data_dtype
    if need_counting_pass:
        # _stream_write_matrix_subset will compute these; a placeholder dtype
        # is still needed for the initial ad.AnnData(...).write() below.
        x_data_dtype = None
    placeholder_dtype = x_data_dtype if x_data_dtype is not None else np.float32

    placeholder = sp.csr_matrix((n_obs, n_vars), dtype=placeholder_dtype)
    adata = ad.AnnData(placeholder, obs=obs, var=var, uns=uns)
    adata.write(output_path)

    _stream_write_matrix_subset(
        source_path,
        read_matrix=lambda b: b.X,
        dest_group="X",
        cell_mask=cell_mask,
        gene_indices=gene_indices,
        n_obs=n_obs,
        n_vars=n_vars,
        output_path=output_path,
        chunk_size=chunk_size,
        row_nnz=row_nnz,
        total_nnz=total_nnz,
        data_dtype=data_dtype,
        chunk_cache=chunk_cache,
    )

    for key in layer_keys:
        _stream_write_matrix_subset(
            source_path,
            read_matrix=(lambda b, _key=key: b.layers[_key]),
            dest_group=f"layers/{key}",
            cell_mask=cell_mask,
            gene_indices=gene_indices,
            n_obs=n_obs,
            n_vars=n_vars,
            output_path=output_path,
            chunk_size=chunk_size,
        )

    if obsm_keys or varm_keys or obsp_keys or varp_keys:
        backed = read_backed(source_path)
        try:
            _copy_extra_slots(
                output_path,
                backed=backed,
                obsm_keys=obsm_keys,
                varm_keys=varm_keys,
                obsp_keys=obsp_keys,
                varp_keys=varp_keys,
                cell_mask=cell_mask,
                gene_mask=gene_mask,
            )
        finally:
            backed.file.close()


def normalize_total_log1p(
    data: str | Path | "AnnData" | ad.AnnData,
    output_path: str | Path | None = None,
    *,
    normalize: bool = True,
    log1p: bool = True,
    target_sum: float = 1e4,
    chunk_size: int = 4096,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    format_mismatch_policy: Literal["auto", "warn", "convert", "off"] = "auto",
    verbose: int | bool = True,
) -> "AnnData":
    """Stream normalize and/or log-transform an h5ad file without loading it fully into memory.
    
    This function processes the source file in chunks, optionally applying:
    1. Total-count normalization (scanpy.pp.normalize_total equivalent)
    2. Log1p transformation (scanpy.pp.log1p equivalent)
    
    The output is written as a sparse CSR matrix. This is the streaming equivalent
    of calling ``scanpy.pp.normalize_total`` followed by ``scanpy.pp.log1p``.

    Only ``X`` is transformed. ``layers``/``obsm``/``varm``/``obsp``/``varp``/
    ``uns`` are carried through unchanged (cell/gene identity, order, and
    count never change here), same as :func:`write_filtered_subset` and
    :func:`downsample_counts`; a source ``.raw`` is dropped with a warning.

    Parameters
    ----------
    data
        Path to source h5ad file, or a backed AnnData object.
    output_path
        Path for output h5ad file. If None, uses output_dir/data_name pattern.
    normalize
        Whether to apply total-count normalization. Default True.
    log1p
        Whether to apply log1p transformation. Default True.
    target_sum
        Target total counts per cell after normalization. Default 1e4.
        Only used if normalize=True.
    chunk_size
        Number of cells to process per chunk. Default 4096.
    output_dir
        Directory for output file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, uses "normalized" suffix.
    format_mismatch_policy
        How to handle a source stored as CSC, whose row-(cell-)streaming here is
        O(total_nnz) per chunk and can be ~100x slower than CSR:

        * ``"auto"`` (default): measure the source's read throughput and
          convert only when the re-reads would actually cost more than one
          conversion does -- roughly, a large file on a slow (networked)
          filesystem streamed in many chunks. See
          :func:`resolve_auto_format_mismatch_policy`.
        * ``"warn"``: proceed but log a single actionable warning.
        * ``"convert"``: always convert the source to CSR in a temporary
          file (bounded-memory two-pass streaming) and stream from that; the
          temporary file is removed before returning.  This calls
          :func:`convert_to_csr` internally, which temporarily needs ~2x the
          source file's size in free disk space and warns automatically if
          that looks tight.
        * ``"off"``: proceed silently with no warning.
    verbose
        Print progress information.
    
    Returns
    -------
    AnnData
        Read-only AnnData wrapper pointing to the output file.
    
    Examples
    --------
    >>> # Full normalization + log1p (default)
    >>> adata_norm = cx.pp.normalize_total_log1p(adata_ro, output_dir=OUTPUT_DIR, data_name="normalized")
    
    >>> # Only log1p (no normalization)
    >>> adata_log = cx.pp.normalize_total_log1p(adata_ro, normalize=False, output_dir=OUTPUT_DIR)
    
    >>> # Only normalization (no log1p)
    >>> adata_norm = cx.pp.normalize_total_log1p(adata_ro, log1p=False, output_dir=OUTPUT_DIR)
    
    >>> # Use explicit output path
    >>> adata_norm = cx.pp.normalize_total_log1p(adata_ro, "results/normalized.h5ad")

    >>> # Auto-convert a CSC source to CSR for fast cell-streaming
    >>> adata_norm = cx.pp.normalize_total_log1p(csc_path, format_mismatch_policy="convert")
    """
    if not normalize and not log1p:
        raise ValueError("At least one of normalize or log1p must be True")
    validate_format_mismatch_policy(format_mismatch_policy)

    # Resolve input path from various input types
    source_path = resolve_data_path(data, require_exists=True)

    # Resolve the output path from the *source* name here, before a temporary
    # fast-axis copy may replace the path that is actually streamed.
    if output_path is None:
        if normalize and log1p:
            suffix = "normalized_log1p"
        elif normalize:
            suffix = "normalized"
        else:
            suffix = "log1p"
        output_path = resolve_output_path(
            source_path, suffix=suffix, output_dir=output_dir, data_name=data_name,
        )
    output_path = Path(output_path)

    # Cell-(row-)streaming below is pathologically slow on CSC; resolve the
    # format mismatch according to policy before touching X. A temporary CSR
    # copy lives beside the output file, not in $TMPDIR. It carries X only,
    # so metadata and the non-X slots are always read from the source.
    with stream_on_fast_axis(
        source_path, axis=0, policy=format_mismatch_policy,
        fn_name="pp.normalize_total_log1p", scratch_dir=output_path.parent,
        chunk_size=chunk_size, verbose=verbose,
    ) as stream_path:
        return _normalize_total_log1p_impl(
            source_path,
            output_path,
            stream_path=stream_path,
            normalize=normalize,
            log1p=log1p,
            target_sum=target_sum,
            chunk_size=chunk_size,
            verbose=verbose,
        )


def _normalize_total_log1p_impl(
    source_path: Path,
    output_path: Path,
    *,
    stream_path: Path,
    normalize: bool,
    log1p: bool,
    target_sum: float,
    chunk_size: int,
    verbose: int | bool,
) -> "AnnData":
    """Core streaming normalize/log1p implementation (paths already resolved).

    ``X`` is streamed from ``stream_path`` (the source itself, or its
    temporary fast-axis copy); ``obs``/``var``/``uns`` and the copied
    ``layers``/``obsm``/``varm``/``obsp``/``varp`` slots come from
    ``source_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ops = []
    if normalize:
        ops.append("normalize")
    if log1p:
        ops.append("log1p")
    _messages.print_saving(verbose, "pp.normalize_total_log1p", output_path)

    # Metadata and slot inventory from the source file.
    backed = read_backed(source_path)
    try:
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        obs = backed.obs.copy()
        var = backed.var.copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        uns = dict(backed.uns)
        # anndata >= 0.13 exposes X itself as layers[None]; only real,
        # named layers should be streamed/copied here.
        layer_keys = [key for key in backed.layers.keys() if key is not None]
        obsm_keys = list(backed.obsm.keys())
        varm_keys = list(backed.varm.keys())
        obsp_keys = list(backed.obsp.keys())
        varp_keys = list(backed.varp.keys())
        has_raw = backed.raw is not None
    finally:
        backed.file.close()

    # First pass: count non-zeros per row (after normalization, same
    # sparsity as input).
    row_nnz = np.zeros(n_obs, dtype=np.int64)
    total_nnz = 0
    backed = read_backed(stream_path)
    try:
        row_offset = 0
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False,
            warn_slow_axis=False,
        ):
            csr = _ensure_csr(block)
            counts = np.diff(csr.indptr)
            row_nnz[row_offset : row_offset + len(counts)] = counts
            total_nnz += int(csr.nnz)
            row_offset += len(counts)
    finally:
        backed.file.close()

    if has_raw:
        _messages.warn(
            "pp.normalize_total_log1p",
            "source AnnData has a `.raw` layer; it is not copied to the "
            "normalized output. Regenerate it from the source file "
            "separately if you need it.",
        )

    def _copy_slots() -> None:
        if layer_keys:
            with h5py.File(source_path, "r") as src, h5py.File(output_path, "r+") as dst:
                dst_layers = dst.require_group("layers")
                for key in layer_keys:
                    if key in dst_layers:
                        del dst_layers[key]
                    src.copy(src[f"layers/{key}"], dst_layers, name=key)
        if obsm_keys or varm_keys or obsp_keys or varp_keys:
            backed_slots = read_backed(source_path)
            try:
                _copy_extra_slots(
                    output_path,
                    backed=backed_slots,
                    obsm_keys=obsm_keys,
                    varm_keys=varm_keys,
                    obsp_keys=obsp_keys,
                    varp_keys=varp_keys,
                    cell_mask=None,
                    gene_mask=None,
                )
            finally:
                backed_slots.file.close()

    if total_nnz == 0:
        # Empty matrix: write placeholder
        placeholder = sp.csr_matrix((n_obs, n_vars), dtype=np.float32)
        adata = ad.AnnData(placeholder, obs=obs, var=var, uns=uns)
        adata.write(output_path)
        _copy_slots()
        return AnnData(output_path)

    # Choose consistent index dtype for both indptr and indices.
    # scipy requires indptr and indices to share the same integer dtype;
    # mixed int32/int64 triggers "Output dtype not compatible" in scipy >= 1.15.
    idx_dtype = np.int32 if total_nnz <= np.iinfo(np.int32).max else np.int64

    # Compute indptr
    indptr = np.zeros(n_obs + 1, dtype=idx_dtype)
    np.cumsum(row_nnz, out=indptr[1:])

    # HDF5 chunk sizing
    hdf5_chunk_size = min(262144, max(8192, total_nnz // 16))

    # Create output file with placeholder
    placeholder = sp.csr_matrix((n_obs, n_vars), dtype=np.float32)
    adata = ad.AnnData(placeholder, obs=obs, var=var, uns=uns)
    adata.write(output_path)

    # Second pass: normalize, log1p, and write
    with h5py.File(output_path, "r+", libver='latest') as dest:
        if "X" in dest:
            del dest["X"]
        grp = dest.create_group("X")
        grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")
        
        data_ds = grp.create_dataset(
            "data",
            shape=(total_nnz,),
            dtype=np.float32,
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None,
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(total_nnz,),
            dtype=idx_dtype,
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None,
        )
        grp.create_dataset("indptr", data=indptr)
        grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)

        backed = read_backed(stream_path)
        try:
            offset = 0
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=False,
                warn_slow_axis=False,
            ):
                csr = _ensure_csr(block)
                
                # Start with original data
                processed_data = csr.data.astype(np.float32)
                
                # Apply normalization if requested
                if normalize:
                    # Library size per cell
                    lib_sizes = np.asarray(csr.sum(axis=1)).ravel()
                    # Avoid division by zero
                    scale = np.divide(
                        target_sum, lib_sizes,
                        out=np.zeros_like(lib_sizes, dtype=np.float64),
                        where=lib_sizes > 0,
                    )
                    
                    # Apply normalization to data (CSR stores data in row-major order)
                    # For each row i, data[indptr[i]:indptr[i+1]] are the values
                    for i in range(csr.shape[0]):
                        start_idx = csr.indptr[i]
                        end_idx = csr.indptr[i + 1]
                        processed_data[start_idx:end_idx] = (
                            processed_data[start_idx:end_idx].astype(np.float64) * scale[i]
                        ).astype(np.float32)
                
                # Apply log1p if requested
                if log1p:
                    processed_data = np.log1p(processed_data)
                
                # Write to HDF5
                nnz = len(processed_data)
                if nnz:
                    data_ds[offset : offset + nnz] = processed_data
                    indices_ds[offset : offset + nnz] = csr.indices.astype(idx_dtype)
                    offset += nnz
        finally:
            backed.file.close()

    _copy_slots()

    _messages.print_done(verbose, "pp.normalize_total_log1p", f"{n_obs} cells × {n_vars} genes")

    return AnnData(output_path)


def _row_seeds(random_state: int, n_obs: int) -> np.ndarray:
    """Deterministic per-row uint64 seeds for :func:`downsample_counts`.

    A vectorized splitmix64-style bit mix rather than one ``hashlib`` call per
    row (as ``_grouping._group_seed`` uses for per-*group* seeds) -- with up
    to millions of rows, a Python-level hash call per row would dominate
    runtime. Being a pure function of ``(random_state, row_index)``, the
    result -- and therefore each row's thinning -- is independent of
    chunk_size and chunk iteration order.

    Kept at the full 64-bit splitmix64 output (not truncated to 32 bits):
    at the "hundreds of thousands to millions of cells" scale this function
    targets, a 32-bit seed space collides between distinct rows often enough
    to give them bit-identical thinning draws (birthday bound), which breaks
    the documented per-cell independence of the sampling.
    """
    z = np.arange(n_obs, dtype=np.uint64)
    z += np.uint64(0x9E3779B97F4A7C15) + (np.uint64(random_state) << np.uint64(1))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z


def _thin_csr_chunk(indptr: np.ndarray, data: np.ndarray, target: int, seeds: np.ndarray) -> None:
    """In-place exact thinning of a CSR chunk, without replacement.

    For each row whose total exceeds ``target``, draws a new per-gene count
    vector summing to ``target`` via ``Generator.multivariate_hypergeometric``
    -- the same distribution scanpy's ``downsample_counts(..., replace=False)``
    draws via its own choice/searchsorted decomposition
    (``scanpy.preprocessing._simple._downsample_array``), but as a single
    library call per row instead of a hand-rolled cumsum/choice/searchsorted/
    bincount sequence.

    ``seeds[r]`` seeds a fresh ``Generator`` for row ``r``, so results depend
    only on each row's own seed, not on how rows are grouped into chunks.
    """
    n_rows = indptr.shape[0] - 1
    for r in range(n_rows):
        start, end = indptr[r], indptr[r + 1]
        row = data[start:end]
        total = int(row.sum())
        if total <= target:
            continue
        rng = np.random.default_rng(int(seeds[r]))
        row[:] = rng.multivariate_hypergeometric(row, target)


def _validate_counts_chunk(chunk: np.ndarray | sp.spmatrix, *, context: str) -> None:
    """Raise if a chunk's values don't look like non-negative integer counts.

    :func:`downsample_counts` casts ``X`` to int64 and thins it; running it
    on already-normalized/log1p data would silently truncate fractional
    values instead of raising, and (since normalized values are typically
    far below any reasonable ``counts_per_cell``) mostly no-op.
    """
    values = chunk.data if sp.issparse(chunk) else np.asarray(chunk)
    if values.size == 0:
        return
    if np.any(values < 0):
        raise ValueError(
            f"{context}: X contains negative values. downsample_counts requires "
            "non-negative integer counts, not normalized/log-transformed data."
        )
    if not np.all(np.isclose(values, np.round(values))):
        raise ValueError(
            f"{context}: X contains non-integer values. downsample_counts requires "
            "raw integer counts -- run it before cx.pp.normalize_total_log1p, not after."
        )


def downsample_counts(
    data: str | Path | "AnnData" | ad.AnnData,
    output_path: str | Path | None = None,
    *,
    counts_per_cell: int,
    chunk_size: int = 4096,
    data_name: str | None = None,
    random_state: int = 0,
    verbose: int | bool = True,
) -> "AnnData":
    """Stream-thin every cell's counts down to at most ``counts_per_cell``.

    This is the streaming, dependency-free equivalent of
    ``scanpy.pp.downsample_counts(adata, counts_per_cell=..., replace=False)``:
    a cell already at or below the target is left unchanged; a cell above it
    has its counts thinned via exact sampling without replacement (see
    :func:`_thin_csr_chunk`), never loading the full matrix into memory.

    Unlike :func:`crispyx.pp.subsample`, this never changes *which* cells or
    genes are present -- it thins the values within every surviving cell. Use
    ``subsample`` first if you also want to reduce cell counts; the two are
    independent, composable streaming passes.

    Only ``X`` is thinned, matching ``scanpy.pp.downsample_counts``.
    ``layers``/``obsm``/``varm``/``obsp``/``varp``/``uns`` are carried
    through unchanged (cell/gene identity, order, and count never change
    here, only the values in ``X``); a source ``.raw`` is dropped with a
    warning, same as :func:`write_filtered_subset`. ``X`` must hold
    non-negative integer counts.

    Parameters
    ----------
    data
        Path to source h5ad file, or a backed AnnData object.
    output_path
        Path for output h5ad file. If None, derived from ``data_name``.
    counts_per_cell
        Target total count per cell. Must be a positive integer.
    chunk_size
        Number of cells to process per chunk. Default 4096.
    data_name
        Custom name for output file.
    random_state
        Seed. Thinning is deterministic for a fixed seed and independent of
        ``chunk_size``.
    verbose
        Print progress information.

    Returns
    -------
    AnnData
        Read-only AnnData wrapper pointing to the output file.
    """
    if not isinstance(counts_per_cell, (int, np.integer)) or counts_per_cell <= 0:
        raise ValueError("counts_per_cell must be a positive integer")

    source_path = resolve_data_path(data)
    output_path = resolve_output_path(
        source_path, suffix="downsample_counts", data_name=data_name, output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _messages.print_saving(verbose, "pp.downsample_counts", output_path)

    backed = read_backed(source_path)
    try:
        n_obs, n_vars = backed.n_obs, backed.n_vars
        obs = backed.obs.copy()
        var = backed.var.copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        uns = dict(backed.uns)
        # anndata >= 0.13 exposes X itself as layers[None]; only real,
        # named layers should be streamed/copied here.
        layer_keys = [key for key in backed.layers.keys() if key is not None]
        obsm_keys = list(backed.obsm.keys())
        varm_keys = list(backed.varm.keys())
        obsp_keys = list(backed.obsp.keys())
        varp_keys = list(backed.varp.keys())
        has_raw = backed.raw is not None
        # Only the first chunk is checked -- same lightweight heuristic
        # de.py's t_test/wilcoxon_test use for the opposite check (that data
        # looks normalized, not raw). A cell above counts_per_cell whose data
        # isn't actually raw counts would otherwise be silently truncated to
        # int64 and thinned instead of raising.
        for _, chunk in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            _validate_counts_chunk(chunk, context="pp.downsample_counts")
            break
    finally:
        backed.file.close()

    if has_raw:
        _messages.warn(
            "pp.downsample_counts",
            "source AnnData has a `.raw` layer; it is not copied to the "
            "downsampled output. Regenerate it from the source file "
            "separately if you need it.",
        )

    storage_format = get_matrix_storage_format(source_path)
    if storage_format == "dense":
        total_nnz_bound = n_obs * n_vars
        with h5py.File(source_path, "r") as f:
            x_obj = f["X"]
            src_dtype = x_obj.dtype if isinstance(x_obj, h5py.Dataset) else x_obj["data"].dtype
    else:
        with h5py.File(source_path, "r") as f:
            total_nnz_bound = f["X"]["data"].shape[0]
            src_dtype = f["X"]["data"].dtype

    placeholder = sp.csr_matrix((n_obs, n_vars), dtype=src_dtype)
    ad.AnnData(placeholder, obs=obs, var=var, uns=uns).write(output_path)

    n_thinned = 0
    if total_nnz_bound != 0:
        warn_if_disk_space_low(
            estimate_sparse_output_bytes(total_nnz_bound), output_path, context="pp.downsample_counts",
        )
        idx_dtype = np.int32 if total_nnz_bound <= np.iinfo(np.int32).max else np.int64
        hdf5_chunk_size = min(262144, max(8192, total_nnz_bound // 16))
        seeds = _row_seeds(random_state, n_obs)
        row_nnz = np.zeros(n_obs, dtype=np.int64)

        with h5py.File(output_path, "r+", libver="latest") as dest:
            if "X" in dest:
                del dest["X"]
            grp = dest.create_group("X")
            grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
            grp.attrs["encoding-version"] = np.bytes_("0.1.0")

            # Upper-bounded, resizable: downsampling only ever removes nonzeros,
            # so the final size (known only after thinning) is <= total_nnz_bound.
            # One streaming pass writes at the running offset; resize() at the
            # end trims to the true final size -- no second pass over the source
            # and no re-drawing the random thinning to "just count" it first.
            data_ds = grp.create_dataset(
                "data", shape=(total_nnz_bound,), maxshape=(total_nnz_bound,), dtype=src_dtype,
                chunks=(hdf5_chunk_size,) if total_nnz_bound >= hdf5_chunk_size else None,
            )
            indices_ds = grp.create_dataset(
                "indices", shape=(total_nnz_bound,), maxshape=(total_nnz_bound,), dtype=idx_dtype,
                chunks=(hdf5_chunk_size,) if total_nnz_bound >= hdf5_chunk_size else None,
            )

            backed = read_backed(source_path)
            try:
                offset = 0
                row_offset = 0
                n_chunks = (n_obs + chunk_size - 1) // chunk_size
                with _create_progress_context(n_chunks, "pp.downsample_counts", verbose, unit="chunk") as pbar:
                    for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
                        csr = _ensure_csr(block)
                        n_rows = csr.shape[0]
                        chunk_seeds = seeds[row_offset : row_offset + n_rows]

                        # The thinning kernel needs exact integer counts; work in
                        # int64 regardless of the source's on-disk dtype (often
                        # float32) and cast back only when writing out.
                        row_totals_before = np.asarray(csr.sum(axis=1)).ravel()
                        thinned = csr.data.astype(np.int64)
                        _thin_csr_chunk(csr.indptr, thinned, int(counts_per_cell), chunk_seeds)
                        n_thinned += int(np.count_nonzero(row_totals_before > counts_per_cell))

                        keep = thinned > 0
                        cum_keep = np.concatenate(([0], np.cumsum(keep)))
                        row_nnz[row_offset : row_offset + n_rows] = np.diff(cum_keep[csr.indptr])

                        kept_data = thinned[keep].astype(src_dtype, copy=False)
                        kept_indices = csr.indices[keep].astype(idx_dtype, copy=False)
                        nnz = kept_data.shape[0]
                        if nnz:
                            data_ds[offset : offset + nnz] = kept_data
                            indices_ds[offset : offset + nnz] = kept_indices
                            offset += nnz
                        row_offset += n_rows
                        pbar.update(1)
            finally:
                backed.file.close()

            data_ds.resize((offset,))
            indices_ds.resize((offset,))
            indptr = np.zeros(n_obs + 1, dtype=idx_dtype)
            np.cumsum(row_nnz, out=indptr[1:])
            grp.create_dataset("indptr", data=indptr)
            grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)

    # layers/obsm/varm/obsp/varp are untouched by this transform (only X's
    # counts are thinned; cell/gene identity, order, and count never change),
    # but still need to be carried through explicitly since the output is a
    # new file, not an in-place edit of the source. Layers are copied with a
    # direct HDF5 object copy rather than _stream_write_matrix_subset: with
    # an identity mask there is no filtering to do, so a native copy avoids
    # paying for a counting pass plus a full re-serialize of every value.
    if layer_keys:
        with h5py.File(source_path, "r") as src, h5py.File(output_path, "r+") as dst:
            dst_layers = dst.require_group("layers")
            for key in layer_keys:
                if key in dst_layers:
                    del dst_layers[key]
                src.copy(src[f"layers/{key}"], dst_layers, name=key)
    if obsm_keys or varm_keys or obsp_keys or varp_keys:
        backed = read_backed(source_path)
        try:
            _copy_extra_slots(
                output_path,
                backed=backed,
                obsm_keys=obsm_keys,
                varm_keys=varm_keys,
                obsp_keys=obsp_keys,
                varp_keys=varp_keys,
                cell_mask=None,
                gene_mask=None,
            )
        finally:
            backed.file.close()

    _messages.print_done(
        verbose, "pp.downsample_counts",
        f"{n_thinned}/{n_obs} cells thinned to <= {counts_per_cell} counts",
    )
    return AnnData(output_path)


def _conversion_buffer_budget_bytes(memory_limit_gb: float | None) -> float:
    """Bytes the CSR<->CSC converters may hold in their output buffers.

    Half of ``memory_limit_gb`` (or of the memory ``psutil`` reports as
    available when it is ``None``); the other half is headroom for the
    per-chunk working set and the caller's own arrays.
    """
    if memory_limit_gb is None:
        try:
            import psutil
            memory_limit_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for the conversion buffer budget. "
                "Install with: pip install psutil"
            )
            memory_limit_gb = 16.0
    return 0.5 * float(memory_limit_gb) * 1e9


def _contiguous_bands(
    counts: np.ndarray, *, per_item_bytes: int, budget_bytes: float,
) -> list[tuple[int, int]]:
    """Split ``range(len(counts))`` into contiguous ``(start, stop)`` bands.

    Each band's ``sum(counts) * per_item_bytes`` stays within ``budget_bytes``
    where possible; a single index whose count alone exceeds the budget gets
    a band of its own (it cannot be split). One band covering everything is
    returned when the total fits.
    """
    n = len(counts)
    max_items = max(1, int(budget_bytes // per_item_bytes))
    cumulative = np.cumsum(counts, dtype=np.int64)
    if n == 0 or cumulative[-1] <= max_items:
        return [(0, n)]
    bands: list[tuple[int, int]] = []
    start = 0
    while start < n:
        before = int(cumulative[start - 1]) if start else 0
        stop = int(np.searchsorted(cumulative, before + max_items, side="right"))
        stop = max(stop, start + 1)
        bands.append((start, stop))
        start = stop
    return bands


def _scatter_by_key(
    keys: np.ndarray,
    vals: np.ndarray,
    secondary: np.ndarray,
    offset: np.ndarray,
    out_vals: np.ndarray,
    out_secondary: np.ndarray,
) -> None:
    """Scatter one chunk's non-zeros into compressed output buffers.

    ``keys`` is the compressed-axis index (column for CSC output, row for CSR
    output) relative to the band start and ``secondary`` the other axis index.
    ``offset[k]`` is the next free position for key ``k`` and is advanced in
    place, so one key's values stay in source order across chunks.
    """
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    unique_keys, counts = np.unique(keys_sorted, return_counts=True)
    starts = np.cumsum(counts) - counts
    within = np.arange(keys_sorted.size) - np.repeat(starts, counts)
    positions = np.repeat(offset[unique_keys], counts) + within
    out_vals[positions] = vals[order].astype(out_vals.dtype, copy=False)
    out_secondary[positions] = secondary[order]
    offset[unique_keys] += counts


@contextmanager
def _replace_on_success(output_path: Path) -> Iterator[Path]:
    """Yield a partial path beside ``output_path``; rename it over
    ``output_path`` only when the block completes.

    The converters pre-size ``X`` and fill it band by band, so a conversion
    killed midway (Ctrl-C, OOM, ENOSPC) would otherwise leave a structurally
    valid file whose unfilled bands read as zeros. Writing to a partial name
    and renaming on success means ``output_path`` either holds a complete
    conversion or does not exist; the partial file is removed on any exit
    that is not a normal completion.
    """
    # Hidden while it is being written, but only one leading dot: the target
    # may itself be a dot-prefixed scratch copy, and "..cx_x.h5ad.partial"
    # would not match the stale-copy sweep's pattern.
    stem = output_path.name if output_path.name.startswith(".") else f".{output_path.name}"
    partial = output_path.with_name(f"{stem}.partial")
    try:
        yield partial
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    partial.replace(output_path)


def _write_sparse_skeleton(
    output_path: Path,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    *,
    encoding: str,
    shape: tuple[int, int],
    indptr: np.ndarray,
    total_nnz: int,
    value_dtype: np.dtype,
    index_dtype: np.dtype,
) -> None:
    """Write ``obs``/``var`` via anndata, then replace ``X`` with an empty
    compressed-sparse group whose ``data``/``indices`` datasets are pre-sized
    to ``total_nnz`` so the converters can fill them band by band."""
    placeholder = sp.csr_matrix(shape, dtype=value_dtype)
    ad.AnnData(placeholder, obs=obs, var=var).write(output_path)
    hdf5_chunk_size = min(262144, max(8192, total_nnz // 16))
    with h5py.File(output_path, "r+", libver="latest") as dest:
        if "X" in dest:
            del dest["X"]
        grp = dest.create_group("X")
        grp.attrs["encoding-type"] = np.bytes_(encoding)
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")
        grp.attrs["shape"] = np.array(shape, dtype=np.int64)
        chunk_arg = (hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None
        grp.create_dataset("data", shape=(total_nnz,), dtype=value_dtype, chunks=chunk_arg)
        grp.create_dataset("indices", shape=(total_nnz,), dtype=index_dtype, chunks=chunk_arg)
        grp.create_dataset("indptr", data=indptr)


def convert_to_csc(
    data: str | Path | "AnnData" | ad.AnnData,
    *,
    output_path: str | Path | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    verbose: int | bool = True,
) -> "AnnData":
    """Convert a backed h5ad file's matrix from CSR (or dense) to CSC format.

    CSC format allows O(nnz_in_chunk) column-slicing instead of O(total_nnz)
    that CSR requires.  This is required for efficient Wilcoxon rank-sum testing,
    which iterates over gene chunks with ``axis=1`` access patterns.

    The conversion streams the source by rows: one counting pass, then the
    non-zeros are scattered into CSC output buffers one *column band* at a
    time and each finished band is written into the pre-sized output file.
    A band is as many contiguous columns as fit half of ``memory_limit_gb``,
    so peak memory is bounded by that budget (plus one row-chunk working
    buffer) regardless of the matrix size. When the whole output fits -- the
    common case -- there is a single band and hence a single second pass; a
    matrix ``K`` times larger than the budget costs ``K`` passes over the
    source instead of running out of memory.

    Peak *disk* usage is not similarly bounded: the source and destination
    files coexist until the caller deletes the source, so this temporarily
    requires roughly 2x the source file's size in free disk space (e.g. ~54 GB
    for a 27 GB screen).  A warning is emitted automatically if the output
    filesystem looks too tight for that.

    If the input file is already CSC, no file is written; the function returns a
    backed AnnData pointing to the original file.

    Parameters
    ----------
    data
        Path to source h5ad file (CSR or dense), or a backed AnnData.
    output_path
        Explicit path for the output file.  If ``None``, a path is derived from
        ``output_dir``/``data_name`` with ``"_csc"`` appended to the stem.
    chunk_size
        Number of rows (cells) to read at a time during both passes.  Default 4096.
    memory_limit_gb
        Memory budget in GB; the output buffers use at most half of it, and
        the matrix is converted in several column bands when it does not fit.
        When ``None`` (default), detected via ``psutil``. On HPC nodes, pass
        the SLURM ``--mem`` value.
    output_dir
        Directory for the output file.  Defaults to the source file's directory.
    data_name
        Custom name used when building the output filename.
    verbose
        Print progress messages.

    Returns
    -------
    AnnData
        Backed (read-only) AnnData pointing to the written CSC h5ad file,
        or to the source file if it was already CSC.

    Examples
    --------
    >>> adata_csc = cx.pp.convert_to_csc(preprocessed_path, output_dir=OUTPUT_DIR)
    """
    source_path = resolve_data_path(data, require_exists=True)

    # Fast path: input is already CSC — return it directly.
    if get_matrix_storage_format(source_path) == "csc":
        _messages.vprint(verbose, "pp.convert_to_csc", "Already CSC, returning source unchanged")
        return AnnData(source_path)

    # Resolve output path.
    if output_path is None:
        output_path = resolve_output_path(
            source_path,
            suffix="csc",
            output_dir=output_dir,
            data_name=data_name,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warn_if_disk_space_low(
        estimate_conversion_bytes(source_path), output_path,
        context="pp.convert_to_csc",
    )

    _messages.vprint(verbose, "pp.convert_to_csc", f"{source_path} → {output_path}")

    # ------------------------------------------------------------------ Pass 1
    # Read all rows in chunks; count non-zeros per *column*; collect metadata.
    backed = read_backed(source_path)
    try:
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(n_obs, n_vars)
            _messages.vprint(verbose, "pp.convert_to_csc", f"chunk_size={chunk_size} (auto)")
        obs = backed.obs.copy()
        var = backed.var.copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        # A storage-format change must not change the values: keep the dtype.
        value_dtype = np.dtype(backed.X.dtype)

        col_nnz = np.zeros(n_vars, dtype=np.int64)
        total_nnz = 0
        n_chunks = (n_obs + chunk_size - 1) // chunk_size
        with _create_progress_context(n_chunks, "pp.convert_to_csc (pass 1)", verbose, unit="chunk") as pbar:
            for _slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
            ):
                csr = _ensure_csr(block)
                np.add.at(col_nnz, csr.indices, 1)
                total_nnz += csr.nnz
                pbar.update(1)
    finally:
        backed.file.close()

    # Empty matrix edge case.
    if total_nnz == 0:
        placeholder = sp.csc_matrix((n_obs, n_vars), dtype=value_dtype)
        adata = ad.AnnData(placeholder, obs=obs, var=var)
        adata.write(output_path)
        return AnnData(output_path)

    # CSC indptr: length n_vars + 1.  Use int64 when NNZ exceeds INT32_MAX.
    idx_dtype = np.int32 if total_nnz <= np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(n_vars + 1, dtype=idx_dtype)
    np.cumsum(col_nnz, out=indptr[1:])

    # Row-index dtype: int32 suffices for up to ~2 billion cells.
    row_dtype = np.int32 if n_obs <= np.iinfo(np.int32).max else np.int64

    # ------------------------------------------------------------------ Pass 2
    # Scatter CSR non-zeros into CSC buffers one column band at a time and
    # write each finished band into the pre-sized output datasets. A band is
    # as many contiguous columns as fit the memory budget; when the whole
    # output fits there is a single band, i.e. a single pass as before.
    per_item_bytes = value_dtype.itemsize + np.dtype(row_dtype).itemsize
    bands = _contiguous_bands(
        col_nnz,
        per_item_bytes=per_item_bytes,
        budget_bytes=_conversion_buffer_budget_bytes(memory_limit_gb),
    )
    if len(bands) > 1:
        _messages.vprint(
            verbose, "pp.convert_to_csc",
            f"output buffers need {total_nnz * per_item_bytes / 1e9:.1f} GB; converting in "
            f"{len(bands)} column bands ({len(bands)} passes over the source)",
        )
    with _replace_on_success(output_path) as partial:
        _write_sparse_skeleton(
            partial, obs, var, encoding="csc_matrix", shape=(n_obs, n_vars),
            indptr=indptr, total_nnz=total_nnz, value_dtype=value_dtype, index_dtype=row_dtype,
        )
        with h5py.File(partial, "r+") as dest, _create_progress_context(
            n_chunks * len(bands), "pp.convert_to_csc (pass 2)", verbose, unit="chunk"
        ) as pbar:
            ds_data, ds_indices = dest["X/data"], dest["X/indices"]
            for c0, c1 in bands:
                base = int(indptr[c0])
                band_nnz = int(indptr[c1]) - base
                band_data = np.empty(band_nnz, dtype=value_dtype)
                band_rows = np.empty(band_nnz, dtype=row_dtype)
                # offset[c] = next write position within the band for column c0 + c.
                # int64 so positions can exceed INT32_MAX when total_nnz > 2^31.
                offset = indptr[c0:c1].astype(np.int64) - base
                whole = (c0, c1) == (0, n_vars)
                row_global = 0
                backed = read_backed(source_path)
                try:
                    for _slc, block in iter_matrix_chunks(
                        backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
                    ):
                        csr = _ensure_csr(block)
                        n_chunk = csr.shape[0]
                        if csr.nnz:
                            cols = csr.indices
                            vals = csr.data
                            # Global row index for every non-zero in this chunk.
                            rows = np.repeat(
                                np.arange(n_chunk, dtype=row_dtype), np.diff(csr.indptr)
                            ) + row_dtype(row_global)
                            if not whole:
                                keep = (cols >= c0) & (cols < c1)
                                cols, vals, rows = cols[keep], vals[keep], rows[keep]
                            if cols.size:
                                _scatter_by_key(cols - c0, vals, rows, offset, band_data, band_rows)
                        row_global += n_chunk
                        pbar.update(1)
                finally:
                    backed.file.close()
                ds_data[base:base + band_nnz] = band_data
                ds_indices[base:base + band_nnz] = band_rows
                del band_data, band_rows

    _messages.print_done(verbose, "pp.convert_to_csc", f"{n_obs} cells × {n_vars} genes")

    return AnnData(output_path)


def convert_to_csr(
    data: str | Path | "AnnData" | ad.AnnData,
    *,
    output_path: str | Path | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    verbose: int | bool = True,
) -> "AnnData":
    """Convert a backed h5ad file's matrix from CSC (or dense) to CSR format.

    CSR format allows O(nnz_in_chunk) row-slicing instead of O(total_nnz)
    that CSC requires.  This is needed for efficient NB-GLM, size factor
    computation, and any operation that iterates over cell (row) chunks.

    The conversion mirrors :func:`convert_to_csc`: a counting pass, then the
    non-zeros are scattered into CSR output buffers one *row band* at a time,
    each band sized to fit half of ``memory_limit_gb``, so peak memory is
    bounded by that budget plus one chunk working buffer. A dense source is
    already row-ordered and is copied chunk by chunk with no whole-matrix
    buffer at all.

    Peak *disk* usage is not similarly bounded: the source and destination
    files coexist until the caller deletes the source, so this temporarily
    requires roughly 2x the source file's size in free disk space.  A warning
    is emitted automatically if the output filesystem looks too tight for that.
    This also applies to :func:`normalize_total_log1p`'s
    ``format_mismatch_policy="convert"``, which calls this function internally.

    If the input file is already CSR, no file is written; the function returns a
    backed AnnData pointing to the original file.

    Parameters
    ----------
    data
        Path to source h5ad file (CSC or dense), or a backed AnnData.
    output_path
        Explicit path for the output file.  If ``None``, a path is derived from
        ``output_dir``/``data_name`` with ``"_csr"`` appended to the stem.
    chunk_size
        Number of rows (cells) to read at a time during both passes.
        Default is calculated automatically.
    memory_limit_gb
        Memory budget in GB; the output buffers use at most half of it, and a
        CSC source is converted in several row bands when it does not fit.
        When ``None`` (default), detected via ``psutil``.
    output_dir
        Directory for the output file.  Defaults to the source file's directory.
    data_name
        Custom name used when building the output filename.
    verbose
        Print progress messages.

    Returns
    -------
    AnnData
        Backed (read-only) AnnData pointing to the written CSR h5ad file,
        or to the source file if it was already CSR.
    """
    source_path = resolve_data_path(data, require_exists=True)

    # Fast path: input is already CSR — return it directly.
    fmt = get_matrix_storage_format(source_path)
    if fmt == "csr":
        _messages.vprint(verbose, "pp.convert_to_csr", "Already CSR, returning source unchanged")
        return AnnData(source_path)

    # Resolve output path.
    if output_path is None:
        output_path = resolve_output_path(
            source_path,
            suffix="csr",
            output_dir=output_dir,
            data_name=data_name,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warn_if_disk_space_low(
        estimate_conversion_bytes(source_path), output_path,
        context="pp.convert_to_csr",
    )

    source_is_csc = fmt == "csc"

    _messages.vprint(verbose, "pp.convert_to_csr", f"{source_path} → {output_path}")

    # Choose the optimal reading axis based on source format.
    # CSC: column-chunks (axis=1) are fast, row-chunks are O(total_nnz).
    # Dense / CSR: row-chunks (axis=0) are fast.
    read_axis = 1 if source_is_csc else 0

    # ------------------------------------------------------------------ Pass 1
    # Count non-zeros per row and collect metadata.
    backed = read_backed(source_path)
    try:
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(n_obs, n_vars)
            _messages.vprint(verbose, "pp.convert_to_csr", f"chunk_size={chunk_size} (auto)")
        obs = backed.obs.copy()
        var = backed.var.copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        # A storage-format change must not change the values: keep the dtype.
        value_dtype = np.dtype(backed.X.dtype)

        row_nnz = np.zeros(n_obs, dtype=np.int64)
        total_nnz = 0
        n_row_chunks = (n_obs + chunk_size - 1) // chunk_size
        n_col_chunks = (n_vars + chunk_size - 1) // chunk_size

        if source_is_csc:
            # Column chunks: convert each to CSC, count NNZ per row via indices.
            with _create_progress_context(n_col_chunks, "pp.convert_to_csr (pass 1)", verbose, unit="chunk") as pbar:
                for _slc, block in iter_matrix_chunks(
                    backed, axis=1, chunk_size=chunk_size, convert_to_dense=False
                ):
                    if sp.issparse(block):
                        csc = sp.csc_matrix(block)
                        np.add.at(row_nnz, csc.indices, 1)
                        total_nnz += csc.nnz
                    else:
                        # Dense column block: count non-zeros per row.
                        dense = np.asarray(block)
                        nz_mask = dense != 0
                        row_nnz += nz_mask.sum(axis=1)
                        total_nnz += int(nz_mask.sum())
                    pbar.update(1)
        else:
            # Row chunks: convert to CSR, count NNZ per row via indptr diffs.
            with _create_progress_context(n_row_chunks, "pp.convert_to_csr (pass 1)", verbose, unit="chunk") as pbar:
                for _slc, block in iter_matrix_chunks(
                    backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
                ):
                    csr = _ensure_csr(block)
                    row_counts = np.diff(csr.indptr)
                    row_nnz[_slc] += row_counts
                    total_nnz += csr.nnz
                    pbar.update(1)
    finally:
        backed.file.close()

    # Empty matrix edge case.
    if total_nnz == 0:
        placeholder = sp.csr_matrix((n_obs, n_vars), dtype=value_dtype)
        adata = ad.AnnData(placeholder, obs=obs, var=var)
        adata.write(output_path)
        return AnnData(output_path)

    # CSR indptr: length n_obs + 1.
    idx_dtype = np.int32 if total_nnz <= np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(n_obs + 1, dtype=idx_dtype)
    np.cumsum(row_nnz, out=indptr[1:])

    # Column-index dtype: int32 suffices for up to ~2 billion genes.
    col_dtype = np.int32 if n_vars <= np.iinfo(np.int32).max else np.int64

    # ------------------------------------------------------------------ Pass 2
    with _replace_on_success(output_path) as partial:
        _write_sparse_skeleton(
            partial, obs, var, encoding="csr_matrix", shape=(n_obs, n_vars),
            indptr=indptr, total_nnz=total_nnz, value_dtype=value_dtype, index_dtype=col_dtype,
        )
        if source_is_csc:
            # Column-chunk reading (fast on CSC), scattered into CSR buffers one
            # row band at a time -- the transpose of convert_to_csc's pass 2.
            per_item_bytes = value_dtype.itemsize + np.dtype(col_dtype).itemsize
            bands = _contiguous_bands(
                row_nnz,
                per_item_bytes=per_item_bytes,
                budget_bytes=_conversion_buffer_budget_bytes(memory_limit_gb),
            )
            if len(bands) > 1:
                _messages.vprint(
                    verbose, "pp.convert_to_csr",
                    f"output buffers need {total_nnz * per_item_bytes / 1e9:.1f} GB; converting in "
                    f"{len(bands)} row bands ({len(bands)} passes over the source)",
                )
            with h5py.File(partial, "r+") as dest, _create_progress_context(
                n_col_chunks * len(bands), "pp.convert_to_csr (pass 2)", verbose, unit="chunk"
            ) as pbar:
                ds_data, ds_indices = dest["X/data"], dest["X/indices"]
                for r0, r1 in bands:
                    base = int(indptr[r0])
                    band_nnz = int(indptr[r1]) - base
                    band_data = np.empty(band_nnz, dtype=value_dtype)
                    band_cols = np.empty(band_nnz, dtype=col_dtype)
                    # offset[r] = next write position within the band for row r0 + r.
                    offset = indptr[r0:r1].astype(np.int64) - base
                    whole = (r0, r1) == (0, n_obs)
                    col_global = 0
                    backed = read_backed(source_path)
                    try:
                        for _slc, block in iter_matrix_chunks(
                            backed, axis=1, chunk_size=chunk_size, convert_to_dense=False
                        ):
                            csc = sp.csc_matrix(block) if sp.issparse(block) else sp.csc_matrix(np.asarray(block))
                            n_chunk_cols = csc.shape[1]
                            if csc.nnz:
                                rows = csc.indices
                                vals = csc.data
                                # Global column index for every non-zero in this chunk.
                                cols = np.repeat(
                                    np.arange(n_chunk_cols, dtype=col_dtype), np.diff(csc.indptr)
                                ) + col_dtype(col_global)
                                if not whole:
                                    keep = (rows >= r0) & (rows < r1)
                                    rows, vals, cols = rows[keep], vals[keep], cols[keep]
                                if rows.size:
                                    _scatter_by_key(rows - r0, vals, cols, offset, band_data, band_cols)
                            col_global += n_chunk_cols
                            pbar.update(1)
                    finally:
                        backed.file.close()
                    ds_data[base:base + band_nnz] = band_data
                    ds_indices[base:base + band_nnz] = band_cols
                    del band_data, band_cols
        else:
            # Row-chunk reading (dense source): every chunk is already in CSR
            # order, so it is written straight into the output datasets.
            backed = read_backed(source_path)
            try:
                with h5py.File(partial, "r+") as dest, _create_progress_context(
                    n_row_chunks, "pp.convert_to_csr (pass 2)", verbose, unit="chunk"
                ) as pbar:
                    ds_data, ds_indices = dest["X/data"], dest["X/indices"]
                    for _slc, block in iter_matrix_chunks(
                        backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
                    ):
                        csr = _ensure_csr(block)
                        if csr.nnz:
                            dst_start = int(indptr[_slc.start])
                            dst_end = dst_start + csr.nnz
                            ds_data[dst_start:dst_end] = csr.data.astype(value_dtype, copy=False)
                            ds_indices[dst_start:dst_end] = csr.indices.astype(col_dtype, copy=False)
                        pbar.update(1)
            finally:
                backed.file.close()

    _messages.print_done(verbose, "pp.convert_to_csr", f"{n_obs} cells × {n_vars} genes")

    return AnnData(output_path)


# ---------------------------------------------------------------------------
# Disk-usage resolvers for crispyx.estimate_disk_usage() (see _preflight.py).
# ---------------------------------------------------------------------------


def _estimate_shape_for_conversion(path: str | Path, **_ignored) -> dict[str, float]:
    """Shared resolver for :func:`convert_to_csc` and :func:`convert_to_csr`.

    Neither function has any groups to resolve -- the estimate is just the
    same ``estimate_conversion_bytes`` call used by their automatic warning.
    """
    return {"output": estimate_conversion_bytes(path)}


def _estimate_shape_for_normalize_total_log1p(
    path: str | Path,
    *,
    format_mismatch_policy: str = "auto",
    chunk_size: int = 4096,
    **_ignored,
) -> dict[str, float]:
    """Disk-usage resolver for :func:`normalize_total_log1p`.

    Only a conversion touches disk beyond the (nnz-bounded, not separately
    estimated here) streamed output: it calls :func:`convert_to_csr` into a
    temporary file beside the output. Whether one happens is
    :func:`scratch_copy_bytes`'s call, exactly as in the real run.
    """
    validate_format_mismatch_policy(format_mismatch_policy)
    scratch = scratch_copy_bytes(
        path, axis=0, policy=format_mismatch_policy, chunk_size=chunk_size
    )
    return {} if scratch is None else {"scratch": scratch}


def calculate_optimal_chunk_size(
    n_obs: int,
    n_vars: int,
    available_memory_gb: float | None = None,
    safety_factor: float = 8.0,
    min_chunk: int = 512,
    max_chunk: int = 4096,
) -> int:
    """Calculate optimal chunk size based on dataset dimensions and available memory.
    
    Parameters
    ----------
    n_obs
        Number of observations (cells) in the dataset.
    n_vars
        Number of variables (genes) in the dataset.
    available_memory_gb
        Available memory in gigabytes. If None, auto-detects using psutil.
    safety_factor
        Safety multiplier to account for overhead (default 8.0 for backed operations).
    min_chunk
        Minimum chunk size to return (default 512).
    max_chunk
        Maximum chunk size to return (default 4096).
    
    Returns
    -------
    int
        Recommended chunk size, clamped to [min_chunk, max_chunk].
    
    Examples
    --------
    >>> calculate_optimal_chunk_size(100000, 20000, available_memory_gb=32)
    2000
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for chunk size calculation. "
                "Install with: pip install psutil"
            )
            available_memory_gb = 16.0
    
    # Calculate chunk size based on memory
    # Each chunk uses approximately: chunk_size * n_vars * 8 bytes (float64)
    # Multiply by safety_factor for overhead
    bytes_per_chunk = n_vars * 8 * safety_factor
    max_chunk_from_memory = int((available_memory_gb * 1e9) / bytes_per_chunk)
    
    # Clamp to reasonable range
    chunk_size = max(min_chunk, min(max_chunk, max_chunk_from_memory))
    
    logger.info(
        f"Calculated chunk size: {chunk_size} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"available memory: {available_memory_gb:.1f}GB)"
    )
    
    return chunk_size


def calculate_optimal_gene_chunk_size(
    n_obs: int,
    n_vars: int,
    n_groups: int | None = None,
    available_memory_gb: float | None = None,
    safety_factor: float = 8.0,
    memory_fraction: float = 0.5,
    min_chunk: int = 32,
    max_chunk: int = 512,
) -> int:
    """Calculate optimal gene chunk size for column-wise operations.
    
    For operations that iterate over genes (columns), such as Wilcoxon tests,
    each chunk loads all cells for a subset of genes. Memory usage is dominated
    by n_obs × chunk_size rather than chunk_size × n_vars.
    
    Enhanced to account for the number of perturbation groups, which significantly
    impacts memory usage due to output array allocation.
    
    Parameters
    ----------
    n_obs
        Number of observations (cells) in the dataset.
    n_vars
        Number of variables (genes) in the dataset.
    n_groups
        Number of perturbation groups. If provided, used to estimate memory
        for output arrays. Large group counts require smaller chunks.
    available_memory_gb
        Available memory in gigabytes. If None, auto-detects using psutil.
    safety_factor
        Safety multiplier to account for overhead (default 8.0).
    memory_fraction
        Fraction of available memory to use (default 0.5). Leave headroom
        for memory-mapped arrays and system overhead.
    min_chunk
        Minimum chunk size to return (default 32).
    max_chunk
        Maximum chunk size to return (default 512).
    
    Returns
    -------
    int
        Recommended gene chunk size, clamped to [min_chunk, max_chunk].
    
    Examples
    --------
    >>> calculate_optimal_gene_chunk_size(100000, 20000, available_memory_gb=32)
    512
    >>> calculate_optimal_gene_chunk_size(4000000, 38000, n_groups=18000, available_memory_gb=128)
    64
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for chunk size calculation. "
                "Install with: pip install psutil"
            )
            available_memory_gb = 16.0
    
    # Usable memory = fraction of available (default 50%)
    usable_memory_bytes = available_memory_gb * memory_fraction * 1e9
    
    # Base memory: dense conversion of n_obs × chunk_size (float64)
    base_memory_per_gene = n_obs * 8
    
    # Group memory: output arrays of n_groups × chunk_size (float64) × ~8 arrays
    # (effect, u_stat, pvalue, z_score, lfc, pts, pts_rest, order)
    group_memory_per_gene = (n_groups * 8 * 8) if n_groups else 0
    
    # Total memory per gene with safety factor
    total_memory_per_gene = (base_memory_per_gene + group_memory_per_gene) * safety_factor
    
    # Calculate max chunk from memory
    max_chunk_from_memory = int(usable_memory_bytes / total_memory_per_gene) if total_memory_per_gene > 0 else max_chunk
    
    # Dynamic max_chunk based on n_groups (datasets with many groups need smaller chunks)
    effective_max_chunk = max_chunk
    if n_groups is not None:
        if n_groups > 10000:
            effective_max_chunk = min(effective_max_chunk, 128)
        elif n_groups > 5000:
            effective_max_chunk = min(effective_max_chunk, 256)
        elif n_groups > 2000:
            effective_max_chunk = min(effective_max_chunk, 384)
    
    # Cell-count-based caps for very large datasets (Wilcoxon ranking is memory-intensive)
    # On memory-constrained machines (< 32 GB available) these caps act as hard safety guards
    # to avoid OOM. On large-memory machines the memory-formula above (max_chunk_from_memory)
    # already accounts for available RAM, so over-riding it with small fixed values would be
    # unnecessarily conservative — e.g. Feng-gwsf on 128 GB can safely use 384-gene chunks.
    _CELL_COUNT_CAP_MEMORY_THRESHOLD_GB = 32.0
    if available_memory_gb < _CELL_COUNT_CAP_MEMORY_THRESHOLD_GB:
        if n_obs > 1_000_000:
            effective_max_chunk = min(effective_max_chunk, 32)   # Very conservative for >1M cells
        elif n_obs > 500_000:
            effective_max_chunk = min(effective_max_chunk, 64)   # Conservative for >500K cells
        elif n_obs > 300_000:
            effective_max_chunk = min(effective_max_chunk, 128)  # Moderate for >300K cells
    # else: trust max_chunk_from_memory (computed from available_memory_gb above)

    # Additional cell-count cap for ALL memory tiers: the per-chunk transient memory
    # for Wilcoxon (dense block + control presort + pert stacking) scales as
    # ~n_obs × chunk × 12 bytes. On high-cell datasets this can exceed 5% of available
    # memory per chunk, causing glibc arena fragmentation across many chunks.
    _PER_CHUNK_BUDGET_FRACTION = 0.05
    per_chunk_bytes_per_gene = n_obs * 12  # dense(f32) + ctrl(f64) + pert(f32)
    per_chunk_budget = available_memory_gb * _PER_CHUNK_BUDGET_FRACTION * 1e9
    if per_chunk_bytes_per_gene > 0 and per_chunk_bytes_per_gene * effective_max_chunk > per_chunk_budget:
        cell_cap = max(min_chunk, int(per_chunk_budget / per_chunk_bytes_per_gene))
        effective_max_chunk = min(effective_max_chunk, cell_cap)
    
    # Clamp to reasonable range
    chunk_size = max(min_chunk, min(effective_max_chunk, max_chunk_from_memory))
    
    logger.info(
        f"Calculated gene chunk size: {chunk_size} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"groups: {n_groups or 'unknown'}, "
        f"available memory: {available_memory_gb:.1f}GB)"
    )
    
    return chunk_size


def calculate_wilcoxon_chunk_size(
    n_obs: int,
    n_vars: int,
    *,
    available_memory_gb: float | None = None,
    min_chunk: int = 32,
    max_chunk: int = 4096,
) -> int:
    """Calculate optimal gene chunk size for Wilcoxon rank-sum tests.

    Unlike :func:`calculate_optimal_gene_chunk_size`, this function has **no
    n_groups cap**.  Wilcoxon writes all output arrays (effect, pvalue, z-score,
    etc.) to on-disk memory-mapped files immediately, so peak RAM per chunk is
    independent of the number of perturbation groups.  The only effective cap is
    a per-chunk transient-memory budget:

        transient ≈ chunk_size × (n_obs × 4  +  n_ctrl × 8  +  n_pert × 4)
                  ≈ chunk_size × n_obs × 12 bytes

    The budget is set to 15 % of ``available_memory_gb`` so that a single chunk
    never exceeds ~1/7th of the node RAM.

    Parameters
    ----------
    n_obs
        Number of cells in the dataset.
    n_vars
        Number of genes in the dataset (used only for logging).
    available_memory_gb
        Available memory in GB.  When *None*, detected via :mod:`psutil`.
        On HPC nodes, pass the SLURM ``--mem`` value so the cap reflects the
        actual job allocation rather than system-wide free memory.
    min_chunk
        Floor for the returned chunk size (default 32).
    max_chunk
        Ceiling for the returned chunk size (default 4096).  The cell-budget
        cap is applied *before* this ceiling, so ``max_chunk`` is only active
        for small/sparse datasets where the cell cap would be very large.

    Returns
    -------
    int
        Recommended gene chunk size, clamped to ``[min_chunk, max_chunk]``.

    Examples
    --------
    >>> # Feng-gwsnf: 396K cells, 128 GB → 4067
    >>> calculate_wilcoxon_chunk_size(396458, 32373, available_memory_gb=128)
    4067
    >>> # Feng-ts: 1.16M cells, 128 GB → 1378
    >>> calculate_wilcoxon_chunk_size(1161864, 33165, available_memory_gb=128)
    1378
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for Wilcoxon chunk size calculation. "
                "Install with: pip install psutil"
            )
            available_memory_gb = 16.0

    # Per-chunk transient memory budget: 15% of available RAM.
    # transient ≈ chunk_size × n_obs × 12 bytes
    # (dense float32 block + ctrl float64 presort + pert float32 stack).
    _PER_CHUNK_BUDGET_FRACTION = 0.15
    per_chunk_budget = available_memory_gb * _PER_CHUNK_BUDGET_FRACTION * 1e9
    per_chunk_bytes_per_gene = n_obs * 12
    cell_cap = max(min_chunk, int(per_chunk_budget / per_chunk_bytes_per_gene))

    chunk_size = max(min_chunk, min(cell_cap, max_chunk))

    logger.info(
        f"Calculated Wilcoxon chunk size: {chunk_size} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"available memory: {available_memory_gb:.1f}GB)"
    )

    return chunk_size


def calculate_nb_glm_chunk_size(
    n_obs: int,
    n_vars: int,
    n_groups: int | None = None,
    available_memory_gb: float | None = None,
    memory_limit_gb: float | None = None,
    safety_factor: float = 8.0,
    memory_fraction: float = 0.5,
    min_chunk: int = 32,
    max_chunk: int = 256,
) -> int:
    """Calculate optimal gene chunk size for NB-GLM operations.
    
    NB-GLM iterates over genes (columns) and for each chunk:
    - Loads dense count data: n_obs × chunk_size
    - Computes dispersion: requires cell-level statistics
    - Fits GLM: design matrix operations
    
    This function is specifically tuned for NB-GLM memory patterns,
    which differ from Wilcoxon (ranking) and t-test (simple statistics).
    
    Parameters
    ----------
    n_obs
        Number of observations (cells) in the dataset.
    n_vars
        Number of variables (genes) in the dataset.
    n_groups
        Number of perturbation groups. If provided, used to estimate memory
        for output arrays and design matrix overhead.
    available_memory_gb
        Available memory in gigabytes. If None, auto-detects using psutil.
    memory_limit_gb
        Optional hard memory limit in GB. If provided, uses the minimum of
        available memory and this limit.
    safety_factor
        Safety multiplier to account for overhead (default 8.0).
    memory_fraction
        Fraction of available memory to use (default 0.5).
    min_chunk
        Minimum chunk size to return (default 32).
    max_chunk
        Maximum chunk size to return (default 256).
    
    Returns
    -------
    int
        Recommended gene chunk size, clamped to [min_chunk, max_chunk].
        For datasets where memory is sufficient, returns max_chunk (256).
        Only reduces chunk size when memory would be exceeded.
    
    Examples
    --------
    >>> calculate_nb_glm_chunk_size(100000, 20000, n_groups=100, available_memory_gb=128)
    256
    >>> calculate_nb_glm_chunk_size(1200000, 36000, n_groups=500, available_memory_gb=128)
    143
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for NB-GLM chunk size calculation."
            )
            available_memory_gb = 16.0
    
    # Apply memory limit if provided
    if memory_limit_gb is not None:
        available_memory_gb = min(available_memory_gb, memory_limit_gb)
    
    # Usable memory = fraction of available
    usable_memory_bytes = available_memory_gb * memory_fraction * 1e9
    
    # NB-GLM memory per gene (conservative estimate):
    # - Dense counts: n_obs × 8 bytes (float64)
    # - Design matrix contribution: n_obs × 2 × 8 bytes
    # - Working matrices (mu, var, residuals): n_obs × 8 × 3
    # - Output arrays: n_groups × 8 × 8 (lfc, stat, pvalue, se, etc.)
    base_memory_per_gene = n_obs * 8 * 6  # ~48 bytes per cell per gene
    group_memory_per_gene = (n_groups * 8 * 8) if n_groups else 0
    
    total_memory_per_gene = (base_memory_per_gene + group_memory_per_gene) * safety_factor
    
    # Calculate max chunk from memory
    if total_memory_per_gene > 0:
        max_chunk_from_memory = int(usable_memory_bytes / total_memory_per_gene)
    else:
        max_chunk_from_memory = max_chunk
    
    # Clamp to reasonable range
    chunk_size = max(min_chunk, min(max_chunk, max_chunk_from_memory))
    
    logger.info(
        f"Calculated NB-GLM chunk size: {chunk_size} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"groups: {n_groups or 'unknown'}, "
        f"available memory: {available_memory_gb:.1f}GB)"
    )
    
    return chunk_size


def calculate_pca_chunk_size(
    n_obs: int,
    n_vars: int,
    n_comps: int = 50,
    available_memory_gb: float | None = None,
    method: str = "auto",
    memory_fraction: float = 0.5,
    min_chunk: int = 256,
    max_chunk: int = 4096,
) -> tuple[int, str]:
    """Calculate optimal chunk size for streaming PCA.
    
    PCA memory usage depends on the method:
    - sparse_cov: O(genes²) for covariance matrix, fast for small gene counts
    - incremental: O(chunk × genes) for data chunks, better for large gene counts
    
    Parameters
    ----------
    n_obs
        Number of observations (cells) in the dataset.
    n_vars
        Number of variables (genes) in the dataset.
    n_comps
        Number of principal components to compute. Default 50.
    available_memory_gb
        Available memory in gigabytes. If None, auto-detects using psutil.
    method
        PCA method: 'auto', 'sparse_cov', or 'incremental'.
        'auto' selects based on gene count and available memory.
    memory_fraction
        Fraction of available memory to use (default 0.5).
    min_chunk
        Minimum chunk size to return (default 256).
    max_chunk
        Maximum chunk size to return (default 4096).
    
    Returns
    -------
    tuple[int, str]
        (chunk_size, selected_method) where selected_method is 'sparse_cov'
        or 'incremental'.
    
    Examples
    --------
    >>> calculate_pca_chunk_size(100000, 8000, available_memory_gb=32)
    (2048, 'sparse_cov')
    >>> calculate_pca_chunk_size(100000, 50000, available_memory_gb=32)
    (1024, 'incremental')
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for PCA chunk size calculation."
            )
            available_memory_gb = 16.0
    
    usable_memory_gb = available_memory_gb * memory_fraction
    
    # Estimate covariance matrix memory (genes × genes × 8 bytes)
    cov_memory_gb = n_vars * n_vars * 8 / 1e9
    
    # Select method if auto
    if method == "auto":
        # Use sparse_cov if covariance fits in 30% of usable memory
        if cov_memory_gb < usable_memory_gb * 0.3:
            selected_method = "sparse_cov"
        else:
            selected_method = "incremental"
    else:
        selected_method = method
    
    # Calculate chunk size based on method
    if selected_method == "sparse_cov":
        # Need: XTX (genes² × 8), sums (genes × 8), chunk (chunk × genes × 8)
        reserved_gb = cov_memory_gb + n_vars * 8 / 1e9
        remaining_gb = max(0.1, usable_memory_gb - reserved_gb)
    else:
        # Need: mean (genes × 8), IPCA internals (~2 × comps × genes × 8), chunk
        ipca_internal_gb = 2 * n_comps * n_vars * 8 / 1e9
        reserved_gb = n_vars * 8 / 1e9 + ipca_internal_gb
        remaining_gb = max(0.1, usable_memory_gb - reserved_gb)
    
    # Chunk memory: chunk_size × n_vars × 8 bytes (float64 during computation)
    bytes_per_row = n_vars * 8
    max_chunk_from_memory = int(remaining_gb * 0.5 * 1e9 / bytes_per_row)
    
    chunk_size = max(min_chunk, min(max_chunk, max_chunk_from_memory))
    
    logger.info(
        f"PCA chunk size: {chunk_size}, method: {selected_method} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"cov matrix: {cov_memory_gb:.2f} GB, "
        f"available: {available_memory_gb:.1f} GB)"
    )
    
    return chunk_size, selected_method


def calculate_adaptive_qc_thresholds(
    adata: ad.AnnData,
    perturbation_column: str,
    mode: str = "conservative",
    sample_size: int = 10000,
    chunk_size: int | None = None,
) -> dict:
    """Calculate adaptive QC thresholds based on data distribution.
    
    Uses percentile-based approach to determine appropriate QC parameters
    that retain most of the data while filtering outliers.
    
    Parameters
    ----------
    adata
        AnnData object (can be backed).
    perturbation_column
        Column in adata.obs containing perturbation labels.
    mode
        'conservative' (10th percentile, retains ~90%) or 
        'aggressive' (5th percentile, retains ~95%).
    sample_size
        Maximum number of cells to sample for gene expression analysis.
    chunk_size
        Optional fixed chunk size to use. If None, calculated adaptively.
    
    Returns
    -------
    dict
        Dictionary with keys: min_genes, min_cells_per_perturbation,
        min_cells_per_gene, chunk_size.
    
    Examples
    --------
    >>> adata = ad.read_h5ad("data.h5ad", backed='r')
    >>> thresholds = calculate_adaptive_qc_thresholds(adata, "perturbation")
    >>> adata.file.close()
    """
    percentile = 10.0 if mode == "conservative" else 5.0
    
    # Analyze perturbation sizes
    if perturbation_column not in adata.obs.columns:
        raise KeyError(
            f"Perturbation column '{perturbation_column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    pert_counts = adata.obs[perturbation_column].value_counts()
    p_percentile = pert_counts.quantile(percentile / 100.0)
    min_cells_per_pert = int(max(5, min(50, p_percentile)))
    
    # Analyze gene expression (sample if dataset is large)
    n_sample = min(sample_size, adata.n_obs)
    if n_sample < adata.n_obs:
        sample_idx = np.random.choice(adata.n_obs, n_sample, replace=False)
        sample_idx = np.sort(sample_idx)  # Sort for efficient backed access
    else:
        sample_idx = None
    
    # Calculate cells per gene and genes per cell efficiently
    # Use chunked processing to handle backed datasets
    cells_per_gene = np.zeros(adata.n_vars, dtype=np.int64)
    genes_per_cell = np.zeros(n_sample if sample_idx is not None else adata.n_obs, dtype=np.int64)
    
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(adata.n_obs, adata.n_vars)
    cell_idx = 0
    
    for chunk_start in range(0, adata.n_obs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, adata.n_obs)
        
        # Get the chunk indices
        if sample_idx is not None:
            # Get indices that fall in this chunk
            mask = (sample_idx >= chunk_start) & (sample_idx < chunk_end)
            if not mask.any():
                continue
            chunk_indices = sample_idx[mask] - chunk_start
            X_chunk = adata.X[chunk_start:chunk_end][chunk_indices]
        else:
            X_chunk = adata.X[chunk_start:chunk_end]
        
        # Process chunk (works with both sparse and dense)
        if sp.issparse(X_chunk):
            # Count non-zeros per gene (column)
            cells_per_gene += np.asarray(np.diff(X_chunk.tocsc().indptr))
            # Count non-zeros per cell (row)
            chunk_genes_per_cell = np.asarray(np.diff(X_chunk.tocsr().indptr))
        else:
            # Dense matrix
            X_chunk_bool = X_chunk > 0
            cells_per_gene += np.asarray(X_chunk_bool.sum(axis=0)).ravel()
            chunk_genes_per_cell = np.asarray(X_chunk_bool.sum(axis=1)).ravel()
        
        # Store genes per cell for this chunk
        n_cells_in_chunk = chunk_genes_per_cell.shape[0]
        genes_per_cell[cell_idx:cell_idx + n_cells_in_chunk] = chunk_genes_per_cell
        cell_idx += n_cells_in_chunk
    
    # Trim genes_per_cell if we didn't fill it completely
    genes_per_cell = genes_per_cell[:cell_idx]
    
    # Calculate thresholds from the collected statistics
    gene_percentile = np.percentile(cells_per_gene, percentile)
    min_cells_per_gene = int(max(5, min(100, gene_percentile)))
    
    median_genes = int(np.median(genes_per_cell))
    min_genes = max(5, min(50, median_genes // 10))  # 10% of median
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(adata.n_obs, adata.n_vars)
    
    thresholds = {
        "min_genes": min_genes,
        "min_cells_per_perturbation": min_cells_per_pert,
        "min_cells_per_gene": min_cells_per_gene,
        "chunk_size": chunk_size,
    }
    
    logger.info(
        f"Adaptive QC thresholds ({mode} mode):\n"
        f"  - min_genes: {min_genes}\n"
        f"  - min_cells_per_perturbation: {min_cells_per_pert} "
        f"({percentile}th percentile: {p_percentile:.1f})\n"
        f"  - min_cells_per_gene: {min_cells_per_gene} "
        f"({percentile}th percentile: {gene_percentile:.1f})\n"
        f"  - chunk_size: {chunk_size}"
    )
    
    return thresholds


def standardize_dataset(
    dataset_path: Path | str,
    perturbation_column: str,
    control_label: str | None,
    gene_name_column: str | None,
    output_dir: Path | str,
    force: bool = False,
) -> Path:
    """Standardize dataset column names and control labels with caching.
    
    Creates a standardized copy of the dataset with:
    - perturbation_column renamed to 'perturbation'
    - control labels standardized to 'control'
    - gene_name_column set as var.index if specified
    
    This function uses a streaming approach to avoid loading the X matrix
    into memory, making it suitable for very large datasets (>1M cells).
    
    Standardized files are cached in {output_dir}/.cache/ and reused
    unless force=True.
    
    Parameters
    ----------
    dataset_path
        Path to original dataset (.h5ad file).
    perturbation_column
        Name of perturbation column in original dataset.
    control_label
        Control label to standardize. If None, auto-detects.
    gene_name_column
        Gene name column to use as var.index. If None, uses existing var.index.
    output_dir
        Directory for cached standardized files.
    force
        If True, regenerate standardized file even if cache exists.
    
    Returns
    -------
    Path
        Path to standardized dataset (either cached or newly created).
    
    Examples
    --------
    >>> standardized_path = standardize_dataset(
    ...     "data/original.h5ad",
    ...     perturbation_column="gene",
    ...     control_label=None,
    ...     gene_name_column="gene_symbols",
    ...     output_dir="results",
    ...     force=False
    ... )
    """
    import datetime
    import shutil
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    standardized_path = cache_dir / f"standardized_{dataset_path.stem}.h5ad"
    
    # Check if cached version exists
    if standardized_path.exists() and not force:
        logger.info(f"Using cached standardized dataset: {standardized_path}")
        return standardized_path
    
    logger.info(f"Standardizing dataset: {dataset_path.name}")
    logger.info(f"  - Perturbation column: '{perturbation_column}' → 'perturbation'")
    
    # Track standardization metadata
    metadata = {
        "original_path": str(dataset_path),
        "standardization_timestamp": datetime.datetime.now().isoformat(),
        "column_mappings": {},
        "label_mappings": {},
    }
    
    # Read obs/var metadata only (without loading X into memory)
    adata = ad.read_h5ad(dataset_path, backed='r')
    obs_df = adata.obs.copy()
    var_df = adata.var.copy()
    uns_dict = dict(adata.uns)  # shallow copy of uns
    adata.file.close()
    
    # Standardize perturbation column in obs
    if perturbation_column != "perturbation":
        if perturbation_column not in obs_df.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' not found. "
                f"Available: {list(obs_df.columns)}"
            )
        obs_df.rename(columns={perturbation_column: "perturbation"}, inplace=True)
        metadata["column_mappings"]["perturbation"] = perturbation_column
        logger.info(f"  - Renamed '{perturbation_column}' → 'perturbation'")
    
    # Standardize control label
    labels = obs_df["perturbation"].astype(str).to_numpy()
    detected_control = resolve_control_label(labels, control_label, verbose=False)
    
    if detected_control != "control":
        obs_df["perturbation"] = (
            obs_df["perturbation"]
            .astype(str)
            .replace({detected_control: "control"})
        )
        metadata["label_mappings"][detected_control] = "control"
        logger.info(f"  - Standardized control: '{detected_control}' → 'control'")
    else:
        logger.info(f"  - Control label already standardized: 'control'")
    
    # Standardize gene names in var
    if gene_name_column is not None:
        if gene_name_column in var_df.columns:
            if not (var_df.index == var_df[gene_name_column]).all():
                var_df.index = var_df[gene_name_column].values
                metadata["column_mappings"]["var.index"] = gene_name_column
                logger.info(f"  - Set var.index from '{gene_name_column}'")
        else:
            logger.warning(
                f"Gene column '{gene_name_column}' not found in var. "
                f"Using existing var.index."
            )
    
    # Store metadata
    if "standardization_metadata" not in uns_dict:
        uns_dict["standardization_metadata"] = {}
    uns_dict["standardization_metadata"].update(metadata)
    
    # Copy h5ad file at filesystem level (streaming - no memory load)
    logger.info(f"  - Copying dataset (streaming, no X matrix load)...")
    shutil.copy2(dataset_path, standardized_path)
    
    # Modify obs/var/uns in-place using h5py
    logger.info(f"  - Updating metadata in copied file...")
    with h5py.File(standardized_path, 'r+') as f:
        # Update obs - need to rewrite the obs group
        # Read current obs structure and update
        _update_h5ad_dataframe(f, 'obs', obs_df)
        
        # Update var - need to rewrite the var group
        _update_h5ad_dataframe(f, 'var', var_df)
        
        # Update uns - handle the standardization_metadata key
        _update_h5ad_uns(f, 'uns', uns_dict)
    
    logger.info(f"Saved standardized dataset: {standardized_path}")
    
    return standardized_path


def _update_h5ad_dataframe(h5file: h5py.File, group_name: str, df: pd.DataFrame) -> None:
    """Update obs or var DataFrame in an h5ad file in-place.
    
    This function handles the anndata HDF5 format where DataFrames are stored
    as groups with individual columns as datasets.
    """
    if group_name not in h5file:
        return
    
    grp = h5file[group_name]
    
    # Update the index (stored as _index attribute or separate dataset)
    index_key = grp.attrs.get('_index', '_index')
    if isinstance(index_key, bytes):
        index_key = index_key.decode('utf-8')
    
    if index_key in grp:
        del grp[index_key]
    # Store index as variable-length strings
    index_vals = df.index.astype(str).values
    grp.create_dataset(index_key, data=index_vals.astype('O'), dtype=h5py.special_dtype(vlen=str))
    
    # Update column names (stored as 'column-order' attribute)
    col_names = list(df.columns)
    # HDF5 attributes don't support object dtype; encode strings as bytes
    grp.attrs['column-order'] = np.array([c.encode('utf-8') for c in col_names])
    
    # Update each column
    for col in df.columns:
        if col in grp:
            del grp[col]
        
        col_data = df[col]
        
        # Handle categorical columns
        if hasattr(col_data, 'cat'):
            # Store as categorical (anndata format)
            cat_grp = grp.create_group(col)
            cat_grp.attrs['encoding-type'] = 'categorical'
            cat_grp.attrs['encoding-version'] = '0.2.0'
            cat_grp.attrs['ordered'] = col_data.cat.ordered  # Required for anndata
            
            # Store categories
            categories = col_data.cat.categories.astype(str).values
            cat_grp.create_dataset('categories', data=categories.astype('O'), 
                                   dtype=h5py.special_dtype(vlen=str))
            
            # Store codes
            cat_grp.create_dataset('codes', data=col_data.cat.codes.values)
        elif pd.api.types.is_numeric_dtype(col_data):
            # Numeric column (includes bool).
            grp.create_dataset(col, data=col_data.values)
        else:
            # String-like column: plain object, numpy fixed-width str/bytes,
            # or any pandas string dtype (StringDtype, pandas>=3's default
            # "str" dtype) -- store as variable-length strings.
            str_vals = col_data.astype(str).values
            grp.create_dataset(col, data=str_vals.astype('O'),
                               dtype=h5py.special_dtype(vlen=str))


def _update_h5ad_uns(h5file: h5py.File, group_name: str, uns_dict: dict) -> None:
    """Update uns dict in an h5ad file in-place.
    
    Only updates the standardization_metadata key to avoid breaking other uns data.
    """
    if group_name not in h5file:
        h5file.create_group(group_name)
    
    grp = h5file[group_name]
    
    # Only update standardization_metadata to be safe
    key = 'standardization_metadata'
    if key in uns_dict:
        if key in grp:
            del grp[key]
        
        # Store as a group with string datasets for each subkey
        meta_grp = grp.create_group(key)
        meta_grp.attrs['encoding-type'] = 'dict'
        meta_grp.attrs['encoding-version'] = '0.1.0'
        
        for subkey, subval in uns_dict[key].items():
            if isinstance(subval, dict):
                # Nested dict - store as JSON string
                import json
                meta_grp.create_dataset(subkey, data=json.dumps(subval),
                                        dtype=h5py.special_dtype(vlen=str))
            elif isinstance(subval, str):
                meta_grp.create_dataset(subkey, data=subval,
                                        dtype=h5py.special_dtype(vlen=str))
            else:
                meta_grp.create_dataset(subkey, data=str(subval),
                                        dtype=h5py.special_dtype(vlen=str))


def needs_sorting_for_nbglm(
    path: str | Path,
    perturbation_column: str = "perturbation",
    *,
    min_cells: int = 360_000,
    min_perturbations: int = 100,
    contiguity_threshold: float = 0.5,
) -> bool:
    """Check if a dataset would benefit from sorting by perturbation for NB-GLM.
    
    Large datasets with scattered cells benefit from having cells sorted
    by perturbation label, as this enables contiguous I/O reads instead of
    random access. This is especially important when the data is stored on
    HDD (rotational disk).
    
    The default thresholds are based on I/O overhead analysis:
    - At ~100 IOPS (typical HDD), 360K cells = 1 hour of random I/O overhead
    - min_perturbations=100 ensures sufficient parallel workload to benefit
    - contiguity_threshold=0.5 catches scattered datasets
    
    Parameters
    ----------
    path
        Path to h5ad file.
    perturbation_column
        Column in obs containing perturbation labels.
    min_cells
        Minimum number of cells for sorting to be recommended.
        Default: 360,000 (~1 hour of random I/O on HDD at 100 IOPS).
    min_perturbations
        Minimum number of perturbations for sorting to be recommended.
    contiguity_threshold
        If average contiguity is below this threshold, sorting is recommended.
        Contiguity is the fraction of a perturbation's cells that are in a
        contiguous block (1.0 = perfectly contiguous, 0.0 = completely scattered).
    
    Returns
    -------
    bool
        True if sorting is recommended, False otherwise.
    """
    backed = read_backed(path)
    try:
        # First check if file is already sorted
        if "sorting_metadata" in backed.uns:
            metadata = backed.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                logger.debug(f"Dataset is already sorted by '{perturbation_column}'")
                return False
        
        n_cells = backed.n_obs
        
        # Check cell count threshold
        if n_cells < min_cells:
            logger.debug(f"Dataset has {n_cells:,} cells < {min_cells:,}, sorting not needed")
            return False
        
        # Get perturbation labels
        if perturbation_column not in backed.obs.columns:
            logger.warning(f"Perturbation column '{perturbation_column}' not found")
            return False
        
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        unique_labels = np.unique(labels)
        n_perts = len(unique_labels)
        
        # Check perturbation count threshold
        if n_perts < min_perturbations:
            logger.debug(f"Dataset has {n_perts} perturbations < {min_perturbations}, sorting not needed")
            return False
        
        # Sample perturbations to estimate contiguity
        sample_perts = unique_labels[:min(20, n_perts)]
        total_contiguity = 0.0
        
        for pert in sample_perts:
            indices = np.where(labels == pert)[0]
            if len(indices) < 2:
                total_contiguity += 1.0
                continue
            span = indices.max() - indices.min() + 1
            contiguity = len(indices) / span
            total_contiguity += contiguity
        
        avg_contiguity = total_contiguity / len(sample_perts)
        
        if avg_contiguity >= contiguity_threshold:
            logger.debug(f"Dataset contiguity {avg_contiguity:.1%} >= {contiguity_threshold:.1%}, sorting not needed")
            return False
        
        logger.info(
            f"Dataset would benefit from sorting: {n_cells:,} cells, {n_perts} perturbations, "
            f"contiguity {avg_contiguity:.1%} < {contiguity_threshold:.1%}"
        )
        return True
        
    finally:
        backed.file.close()


def sort_by_perturbation(
    path: str | Path,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 4096,
    force: bool = False,
) -> Path:
    """Sort cells by perturbation label for contiguous I/O access.
    
    Creates a new h5ad file with cells reordered so that all cells from each
    perturbation are contiguous. Control cells are placed first, followed by
    each perturbation group in alphabetical order. This enables efficient
    sequential reads when processing perturbations in parallel.
    
    The function works in streaming mode to handle datasets larger than memory.
    Sorting information is stored in uns['sorting_metadata'].
    
    Parameters
    ----------
    path
        Path to input h5ad file.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Label for control cells. If None, auto-detected from common patterns.
    output_path
        Path for output sorted file. If None, appends '_sorted' to input name.
    chunk_size
        Number of cells to process per chunk during streaming write.
    force
        If True, recreate sorted file even if it already exists.
    
    Returns
    -------
    Path
        Path to the sorted h5ad file.
    
    Examples
    --------
    >>> sorted_path = sort_by_perturbation(
    ...     "data/large_dataset.h5ad",
    ...     perturbation_column="perturbation",
    ...     control_label="control",
    ... )
    >>> # sorted_path is now "data/large_dataset_sorted.h5ad"
    
    Notes
    -----
    The sorted file contains additional metadata in uns['sorting_metadata']:
    - original_path: Path to the original unsorted file
    - sort_order: Array mapping new indices to original indices
    - perturbation_boundaries: Dict mapping perturbation labels to (start, end) indices
    - sorted_by: The column used for sorting
    - timestamp: When the sorting was performed

    For sparse inputs the output is always written in CSR format, since sorting
    benefits row-wise (per-perturbation) access patterns used by NB-GLM. CSC
    files used by Wilcoxon do not need perturbation sorting.
    """
    import datetime
    
    path = Path(path)
    
    # Determine output path
    if output_path is None:
        output_path = path.parent / f"{path.stem}_sorted.h5ad"
    else:
        output_path = Path(output_path)
    
    # Check if already sorted
    if output_path.exists() and not force:
        # Verify it's properly sorted
        try:
            backed = read_backed(output_path)
            has_metadata = "sorting_metadata" in backed.uns
            backed.file.close()
            if has_metadata:
                logger.info(f"Using existing sorted file: {output_path}")
                return output_path
        except Exception:
            pass  # File exists but invalid, recreate
    
    logger.info(f"Sorting dataset by perturbation: {path}")
    
    # Read metadata
    backed = read_backed(path)
    try:
        n_cells = backed.n_obs
        n_genes = backed.n_vars
        
        # Get perturbation labels
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' not found. "
                f"Available: {list(backed.obs.columns)}"
            )
        
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        
        # Detect control label if not specified
        if control_label is None:
            control_label = resolve_control_label(labels, None)
        
        # Create sort order: control first, then alphabetical
        unique_labels = sorted(set(labels) - {control_label})
        label_order = [control_label] + unique_labels
        
        # Create mapping from label to sort priority
        label_priority = {label: i for i, label in enumerate(label_order)}
        
        # Get sort indices (stable sort to preserve original order within groups)
        sort_keys = np.array([label_priority[l] for l in labels])
        sort_indices = np.argsort(sort_keys, kind='stable')
        
        # Compute perturbation boundaries
        sorted_labels = labels[sort_indices]
        boundaries = {}
        current_label = None
        start_idx = 0
        
        for i, label in enumerate(sorted_labels):
            if label != current_label:
                if current_label is not None:
                    boundaries[current_label] = [start_idx, i]  # Use list, not tuple
                current_label = label
                start_idx = i
        if current_label is not None:
            boundaries[current_label] = [start_idx, len(sorted_labels)]  # Use list, not tuple
        
        # Read obs and var
        obs_sorted = backed.obs.iloc[sort_indices].copy()
        var = backed.var.copy()
        
        # Get uns (convert to regular dict for modification)
        uns = dict(backed.uns)
        
    finally:
        backed.file.close()
    
    # Add sorting metadata - convert all to h5ad-compatible types
    # Note: boundaries values are [start, end] lists (tuples not serializable)
    sorting_metadata = {
        "original_path": str(path),
        "sorted_by": perturbation_column,
        "control_label": control_label,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_perturbations": len(label_order),
        "perturbation_boundaries": boundaries,  # Dict[str, List[int]]
    }
    # Don't store full sort_order for large datasets (memory waste)
    if len(sort_indices) < 100000:
        sorting_metadata["sort_order"] = sort_indices.tolist()
    uns["sorting_metadata"] = sorting_metadata
    
    # Create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warn_if_disk_space_low(
        1.1 * float(path.stat().st_size), output_path,
        context="sort_by_perturbation",
    )

    logger.info(f"  Reordering {n_cells:,} cells into {len(label_order)} contiguous groups...")

    # Check if source is dense or sparse
    storage_format = get_matrix_storage_format(path)
    is_dense = storage_format == "dense"
    
    try:
        if is_dense:
            # For dense storage: write directly as dense
            _write_sorted_dense(
                source_path=path,
                output_path=output_path,
                sort_indices=sort_indices,
                obs_sorted=obs_sorted,
                var=var,
                uns=uns,
                chunk_size=chunk_size,
            )
        else:
            # For sparse storage: stream as CSR
            _write_sorted_sparse(
                source_path=path,
                output_path=output_path,
                sort_indices=sort_indices,
                obs_sorted=obs_sorted,
                var=var,
                uns=uns,
                chunk_size=chunk_size,
            )
    except Exception:
        # Remove partial output to avoid corrupt file on next run
        if output_path.exists():
            logger.warning(f"  Removing partial sorted file: {output_path}")
            output_path.unlink()
        raise
    
    logger.info(f"Saved sorted dataset: {output_path}")
    logger.info(f"  Perturbation groups: {len(label_order)} (control + {len(unique_labels)} perturbations)")
    
    return output_path


def _write_sorted_dense(
    source_path: Path,
    output_path: Path,
    sort_indices: np.ndarray,
    obs_sorted: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict,
    chunk_size: int,
) -> None:
    """Write sorted file for dense input matrix.
    
    Uses chunked reading to avoid loading full matrix into memory.
    Writes h5ad in two passes: first X matrix via h5py, then metadata via anndata.
    """
    backed = read_backed(source_path)
    try:
        n_cells = backed.n_obs
        n_genes = backed.n_vars
        
        # Get dtype from first chunk
        sample = backed.X[:min(100, n_cells), :]
        dtype = sample.dtype
        
        # Create output with chunked writing for memory efficiency
        with h5py.File(output_path, 'w') as f:
            # Create dataset with chunking for efficient access
            X_out = f.create_dataset(
                'X',
                shape=(n_cells, n_genes),
                dtype=dtype,
                chunks=(min(chunk_size, n_cells), n_genes),
            )
            
            # Write in chunks based on output order
            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                # Get the original indices for this output chunk
                chunk_indices = sort_indices[start:end]
                
                # Read from source (optimize by sorting indices for sequential read)
                read_order = np.argsort(chunk_indices)
                sorted_chunk_indices = chunk_indices[read_order]
                
                # Read data in optimized order
                chunk_data = backed.X[sorted_chunk_indices, :]
                
                # Reorder to match output order
                inverse_order = np.argsort(read_order)
                chunk_data = chunk_data[inverse_order]
                
                X_out[start:end, :] = chunk_data
                
                if (start // chunk_size) % 100 == 0:
                    logger.debug(f"  Written {end:,}/{n_cells:,} cells...")
        
    finally:
        backed.file.close()
    
    # Write metadata using anndata (proper h5ad structure)
    # First create a temp file with correct metadata structure
    temp_path = output_path.with_suffix('.meta.h5ad')
    adata_meta = ad.AnnData(
        X=sp.csr_matrix((len(obs_sorted), len(var)), dtype=np.float32),  # Placeholder
        obs=obs_sorted,
        var=var,
        uns=uns,
    )
    adata_meta.write(temp_path)
    
    # Copy metadata from temp to main file  
    with h5py.File(temp_path, 'r') as src:
        with h5py.File(output_path, 'a') as dst:
            for key in ['obs', 'var', 'uns']:
                if key in src:
                    if key in dst:
                        del dst[key]
                    src.copy(key, dst)
    
    # Cleanup temp file
    temp_path.unlink()



def _write_sorted_sparse(
    source_path: Path,
    output_path: Path,
    sort_indices: np.ndarray,
    obs_sorted: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict,
    chunk_size: int,
) -> None:
    """Write sorted file for sparse input matrix.

    Uses chunked I/O to avoid loading the full matrix into memory.
    Rows are read in output order (chunk_size at a time), converted
    to CSR components, and appended to resizable HDF5 datasets.
    Peak memory is proportional to chunk_size × n_genes, not n_cells × n_genes.
    """
    backed = read_backed(source_path)
    try:
        n_cells = backed.n_obs
        n_genes = backed.n_vars

        # Determine dtype from a small sample
        sample = backed.X[:1]
        if sp.issparse(sample):
            data_dtype = sample.dtype
        else:
            data_dtype = np.float32

        logger.info(
            f"  Streaming sorted sparse write: {n_cells:,} cells, "
            f"chunk_size={chunk_size}"
        )

        # Build CSR structure incrementally via h5py
        with h5py.File(output_path, "w") as f:
            x_grp = f.create_group("X")
            x_grp.attrs["encoding-type"] = "csr_matrix"
            x_grp.attrs["encoding-version"] = "0.1.0"
            x_grp.attrs["shape"] = np.array([n_cells, n_genes], dtype=np.int64)

            # Resizable datasets for data and indices; indptr is pre-allocated
            ds_data = x_grp.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                dtype=data_dtype,
                chunks=(min(262144, max(1, n_cells)),),
            )
            ds_indices = x_grp.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(min(262144, max(1, n_cells)),),
            )
            ds_indptr = x_grp.create_dataset(
                "indptr",
                shape=(n_cells + 1,),
                dtype=np.int64,
            )

            nnz_written = 0
            ds_indptr[0] = 0

            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                # Original row indices for this output chunk
                orig_idx = sort_indices[start:end]

                # Read rows from source (sorted for sequential access)
                read_order = np.argsort(orig_idx)
                sorted_orig_idx = orig_idx[read_order]
                block = backed.X[sorted_orig_idx, :]

                # Undo read_order to restore output order
                inverse_order = np.argsort(read_order)
                if sp.issparse(block):
                    block = sp.csr_matrix(block)[inverse_order, :]
                else:
                    block = sp.csr_matrix(block[inverse_order, :])

                csr = sp.csr_matrix(block, dtype=data_dtype)

                chunk_nnz = csr.nnz
                if chunk_nnz > 0:
                    new_total = nnz_written + chunk_nnz
                    ds_data.resize((new_total,))
                    ds_indices.resize((new_total,))
                    ds_data[nnz_written:new_total] = csr.data
                    ds_indices[nnz_written:new_total] = csr.indices

                # Write indptr for this chunk (int64 to avoid overflow
                # when cumulative nnz exceeds INT32_MAX ≈ 2.1 billion)
                chunk_indptr = csr.indptr[1:].astype(np.int64) + nnz_written
                ds_indptr[start + 1 : end + 1] = chunk_indptr

                nnz_written += chunk_nnz

                if (start // chunk_size) % 50 == 0:
                    logger.debug(
                        f"  Written {end:,}/{n_cells:,} cells "
                        f"({nnz_written:,} nnz)..."
                    )
    finally:
        backed.file.close()

    # Write obs, var, uns metadata using anndata
    temp_path = output_path.with_suffix(".meta.h5ad")
    adata_meta = ad.AnnData(
        X=sp.csr_matrix((len(obs_sorted), len(var)), dtype=np.float32),
        obs=obs_sorted,
        var=var,
        uns=uns,
    )
    adata_meta.write(temp_path)

    with h5py.File(temp_path, "r") as src:
        with h5py.File(output_path, "a") as dst:
            for key in ["obs", "var", "uns"]:
                if key in src:
                    if key in dst:
                        del dst[key]
                    src.copy(key, dst)

    temp_path.unlink()


def get_perturbation_slice(
    adata_or_path: str | Path | ad.AnnData,
    perturbation_label: str,
    perturbation_column: str = "perturbation",
) -> tuple[slice | None, bool]:
    """Get slice for a perturbation's cells, checking if file is sorted.
    
    If the file is sorted by perturbation, returns a contiguous slice.
    Otherwise, returns None for the slice (caller should use boolean mask).
    
    Parameters
    ----------
    adata_or_path
        Path to h5ad file, or an already-opened AnnData object.
    perturbation_label
        Label of the perturbation to get slice for.
    perturbation_column
        Column containing perturbation labels.
    
    Returns
    -------
    tuple[slice | None, bool]
        (slice object or None, is_sorted flag).
        If is_sorted is True, slice is valid for contiguous access.
        If is_sorted is False, slice is None and caller should use mask.
    """
    # Handle both path and AnnData input
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path
        should_close = False
    else:
        adata = read_backed(adata_or_path)
        should_close = True
    
    try:
        # Check for sorting metadata
        if "sorting_metadata" in adata.uns:
            metadata = adata.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                boundaries = metadata.get("perturbation_boundaries", {})
                if perturbation_label in boundaries:
                    start, end = boundaries[perturbation_label]
                    return slice(start, end), True
        
        return None, False
    finally:
        if should_close:
            adata.file.close()


# =============================================================================
# Feature 1 — Backed metadata helpers (load/write obs and var without X)
# =============================================================================

def _decode_h5_str_array(arr: np.ndarray) -> np.ndarray:
    """Decode a byte/object string array to Python ``str`` values."""
    if arr.dtype.kind in ("S", "O"):
        return np.array(
            [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x for x in arr]
        )
    return arr


def _read_h5_1d(item: "h5py.Dataset | h5py.Group") -> np.ndarray:
    """Read a 1-D AnnData element as a decoded numpy array.

    Handles plain datasets as well as the ``nullable-string-array`` /
    ``nullable-integer`` / ``nullable-boolean`` group encodings, where the data
    lives in a ``values`` dataset alongside a boolean ``mask``. anndata >= 0.13
    uses these group encodings by default for pandas >= 3.0 string/nullable
    columns and index, whereas older versions wrote flat datasets.
    """
    if isinstance(item, h5py.Group):
        if "values" not in item:
            raise TypeError(
                "Cannot read HDF5 group with encoding "
                f"{str(item.attrs.get('encoding-type', ''))!r} as a 1-D array."
            )
        values = _decode_h5_str_array(item["values"][()])
        if "mask" in item:
            mask = np.asarray(item["mask"][()], dtype=bool)
            if mask.any():
                values = values.astype(object)
                values[mask] = None
        return values
    return _decode_h5_str_array(item[()])


def _h5_index_length(grp: "h5py.Group") -> int:
    """Return the number of rows in an AnnData obs/var group.

    Resolves the index element (honouring the ``_index`` attribute) and reads
    its length, handling both flat datasets and ``nullable-*`` group encodings
    (where ``len(group)`` would incorrectly count the ``values``/``mask``
    members rather than the rows).
    """
    _idx_raw = grp.attrs.get("_index", "_index")
    index_key = _idx_raw.decode("utf-8") if isinstance(_idx_raw, bytes) else str(_idx_raw)
    item = grp[index_key]
    if isinstance(item, h5py.Group):
        return int(item["values"].shape[0])
    return int(item.shape[0])


def _read_dataframe_from_h5(grp: "h5py.Group") -> pd.DataFrame:
    """Read an AnnData-encoded HDF5 group as a pandas DataFrame."""
    _idx_raw = grp.attrs.get("_index", "_index")
    index_key = _idx_raw.decode("utf-8") if isinstance(_idx_raw, bytes) else str(_idx_raw)
    _col_raw = grp.attrs.get("column-order", [])
    column_order = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in _col_raw]

    index = pd.Index(_read_h5_1d(grp[index_key]))

    all_keys = [k for k in grp.keys() if k != index_key]
    ordered_keys = [k for k in column_order if k in grp] + [
        k for k in all_keys if k not in set(column_order)
    ]

    columns: dict[str, Any] = {}
    for key in ordered_keys:
        item = grp[key]
        if isinstance(item, h5py.Group):
            enc = str(item.attrs.get("encoding-type", ""))
            if enc == "categorical":
                cats = pd.Index(_read_h5_1d(item["categories"]))
                codes = item["codes"][()].astype(np.intp)
                ordered = bool(item.attrs.get("ordered", False))
                columns[key] = pd.Categorical.from_codes(
                    codes, categories=cats, ordered=ordered
                )
            elif "values" in item:
                # nullable-string / nullable-integer / nullable-boolean array
                columns[key] = _read_h5_1d(item)
        else:
            columns[key] = _read_h5_1d(item)

    return pd.DataFrame(columns, index=index)


def _write_dataframe_to_h5(grp: "h5py.Group", df: pd.DataFrame) -> None:
    """Write a pandas DataFrame to an h5py Group in AnnData 0.2.0 encoding."""
    str_dtype = h5py.string_dtype(encoding="utf-8")

    grp.attrs["encoding-type"] = "dataframe"
    grp.attrs["encoding-version"] = "0.2.0"
    grp.attrs["_index"] = "_index"
    grp.attrs["column-order"] = np.array(list(df.columns), dtype=object)

    idx_ds = grp.create_dataset(
        "_index", data=df.index.astype(str).to_numpy(), dtype=str_dtype
    )
    idx_ds.attrs["encoding-type"] = "string-array"
    idx_ds.attrs["encoding-version"] = "0.2.0"

    for col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            cat_grp = grp.create_group(col)
            cat_grp.attrs["encoding-type"] = "categorical"
            cat_grp.attrs["encoding-version"] = "0.2.0"
            cat_grp.attrs["ordered"] = bool(series.cat.ordered)
            cats = series.cat.categories.astype(str).to_numpy()
            cd = cat_grp.create_dataset("categories", data=cats, dtype=str_dtype)
            cd.attrs["encoding-type"] = "string-array"
            cd.attrs["encoding-version"] = "0.2.0"
            codes = series.cat.codes.to_numpy()
            codes_dtype = np.int8 if len(series.cat.categories) < 128 else np.int16
            co = cat_grp.create_dataset("codes", data=codes.astype(codes_dtype))
            co.attrs["encoding-type"] = "array"
            co.attrs["encoding-version"] = "0.2.0"
        elif series.dtype.kind in ("O", "U", "S"):
            vals = series.fillna("").astype(str).to_numpy()
            ds = grp.create_dataset(col, data=vals, dtype=str_dtype)
            ds.attrs["encoding-type"] = "string-array"
            ds.attrs["encoding-version"] = "0.2.0"
        else:
            ds = grp.create_dataset(col, data=series.to_numpy())
            ds.attrs["encoding-type"] = "array"
            ds.attrs["encoding-version"] = "0.2.0"


def load_obs(path: str | Path) -> pd.DataFrame:
    """Load the obs metadata table from an h5ad file without reading X.

    Parameters
    ----------
    path
        Path to the h5ad file.

    Returns
    -------
    pd.DataFrame
        Full obs DataFrame in memory.
    """
    with h5py.File(Path(path), "r") as f:
        if "obs" not in f:
            raise KeyError("h5ad file has no 'obs' group.")
        return _read_dataframe_from_h5(f["obs"])


def load_var(path: str | Path) -> pd.DataFrame:
    """Load the var metadata table from an h5ad file without reading X.

    Parameters
    ----------
    path
        Path to the h5ad file.

    Returns
    -------
    pd.DataFrame
        Full var DataFrame in memory.
    """
    with h5py.File(Path(path), "r") as f:
        if "var" not in f:
            raise KeyError("h5ad file has no 'var' group.")
        return _read_dataframe_from_h5(f["var"])


def write_obs(path: str | Path, df: pd.DataFrame) -> None:
    """Overwrite the obs metadata table in an h5ad file without touching X.

    Parameters
    ----------
    path
        Path to the h5ad file (modified in-place).
    df
        New obs DataFrame. Must have the same number of rows as the existing
        obs table. Index values are written as cell barcodes.

    Raises
    ------
    ValueError
        If the DataFrame length does not match the existing n_obs.
    """
    path = Path(path)
    with h5py.File(path, "r+") as f:
        old_n = _h5_index_length(f["obs"])
        if len(df) != old_n:
            raise ValueError(
                f"DataFrame has {len(df)} rows but the file has {old_n} cells."
            )
        del f["obs"]
        grp = f.create_group("obs")
        _write_dataframe_to_h5(grp, df)


def write_var(path: str | Path, df: pd.DataFrame) -> None:
    """Overwrite the var metadata table in an h5ad file without touching X.

    Parameters
    ----------
    path
        Path to the h5ad file (modified in-place).
    df
        New var DataFrame. Must have the same number of rows as the existing
        var table.

    Raises
    ------
    ValueError
        If the DataFrame length does not match the existing n_vars.
    """
    path = Path(path)
    with h5py.File(path, "r+") as f:
        old_n = _h5_index_length(f["var"])
        if len(df) != old_n:
            raise ValueError(
                f"DataFrame has {len(df)} rows but the file has {old_n} genes."
            )
        del f["var"]
        grp = f.create_group("var")
        _write_dataframe_to_h5(grp, df)


# =============================================================================
# Feature 2 — Gene name standardisation
# =============================================================================

def standardise_gene_names(
    path: str | Path,
    *,
    column: str | None = None,
    strip_version: bool = True,
    normalise_mt_prefix: bool = True,
    lookup_symbols: bool = False,
    species: str = "human",
    unmapped_action: Literal["keep", "error", "warn"] = "warn",
    inplace: bool = True,
) -> "pd.Series | None":
    """Standardise gene identifiers in the var metadata table.

    Applies a deterministic normalisation pipeline:

    1. Strip Ensembl version suffixes (``ENSG00000123.4`` → ``ENSG00000123``).
    2. Normalise ``mt-`` prefix to ``MT-`` (human mitochondrial convention).
    3. Optionally resolve Ensembl IDs to HGNC symbols via ``mygene``
       (requires ``pip install mygene``). A ``tqdm`` progress bar is shown
       during batched lookups.

    Parameters
    ----------
    path
        Path to the h5ad file.
    column
        var column to normalise. ``None`` normalises the index (var_names).
    strip_version
        Strip ``".N"`` Ensembl version suffixes.
    normalise_mt_prefix
        Convert lower-case ``mt-`` prefix to ``MT-``.
    lookup_symbols
        If True, query ``mygene`` to map Ensembl IDs → gene symbols.
    species
        Species string passed to ``mygene`` (default ``"human"``).
    unmapped_action
        What to do for IDs not found by mygene: ``"keep"`` leaves them
        unchanged, ``"warn"`` emits a warning, ``"error"`` raises.
    inplace
        If True, write the result back to the file and return ``None``.
        If False, return a Series without modifying the file.

    Returns
    -------
    pd.Series or None
        Normalised gene names when ``inplace=False``, else ``None``.
    """
    path = Path(path)
    df = load_var(path)

    if column is None:
        names = pd.Series(df.index.astype(str).to_numpy(), name="_index")
    else:
        if column not in df.columns:
            raise KeyError(
                f"Column '{column}' not found in var. "
                f"Available: {list(df.columns)}"
            )
        names = df[column].astype(str).copy()

    # Step 1: strip Ensembl version suffix
    if strip_version:
        names = names.str.replace(r"\.\d+$", "", regex=True)

    # Step 2: normalise mt- prefix
    if normalise_mt_prefix:
        names = names.str.replace(r"^mt-", "MT-", regex=True)

    # Step 3: optional online lookup via mygene
    if lookup_symbols:
        try:
            import mygene  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "The 'mygene' package is required for online symbol lookup. "
                "Install it with: pip install mygene"
            )
        mg = mygene.MyGeneInfo()
        unique_ids = names.unique().tolist()
        symbol_map: dict[str, str] = {}
        batch_size = 1000
        batches = list(range(0, len(unique_ids), batch_size))
        try:
            from tqdm import tqdm as _tqdm  # type: ignore[import]
            it = _tqdm(batches, desc="mygene lookup", unit="batch")
        except ImportError:
            it = iter(batches)
        for start in it:
            batch = unique_ids[start : start + batch_size]
            hits = mg.querymany(
                batch,
                scopes="ensembl.gene,symbol",
                fields="symbol",
                species=species,
                verbose=False,
                as_dataframe=False,
            )
            for hit in hits:
                query_id = hit.get("query", "")
                symbol = hit.get("symbol", "")
                if symbol and not hit.get("notfound", False):
                    symbol_map[query_id] = symbol
        unmapped = [i for i in unique_ids if i not in symbol_map]
        if unmapped:
            msg = f"{len(unmapped)} gene IDs could not be mapped to symbols."
            if unmapped_action == "error":
                raise ValueError(msg + f" First 10: {unmapped[:10]}")
            elif unmapped_action == "warn":
                logger.warning("%s They will be left unchanged.", msg)
        names = names.map(lambda x: symbol_map.get(x, x))

    if not inplace:
        return names

    if column is None:
        df.index = pd.Index(names.to_numpy(), name=df.index.name)
    else:
        df[column] = names.to_numpy()

    write_var(path, df)
    return None


# =============================================================================
# Feature 3 — Perturbation label normalisation
# =============================================================================

_DEFAULT_CONTROL_ALIASES: frozenset[str] = frozenset({
    "ntc", "non-targeting", "non_targeting", "nontarget", "non-target",
    "non_target", "control", "ctrl", "scramble", "scrambled",
    "non-targeting control", "non-targeting-control",
})


def normalise_perturbation_labels(
    path: str | Path,
    column: str,
    *,
    strip_prefixes: list[str] | None = None,
    strip_suffixes: list[str] | None = None,
    strip_suffix_regex: str | None = None,
    control_aliases: list[str] | None = None,
    canonical_control: str = "NTC",
    inplace: bool = True,
) -> "pd.Series | None":
    """Normalise perturbation labels stored in an obs column.

    Applies transformations in order:

    1. Strip specified prefixes via vectorised ``pd.Series.str.replace``.
    2. Strip specified suffixes via vectorised ``pd.Series.str.replace``.
    3. Apply a custom regex substitution (``strip_suffix_regex``).
    4. Map known control aliases to ``canonical_control``.

    Parameters
    ----------
    path
        Path to the h5ad file.
    column
        obs column containing perturbation labels.
    strip_prefixes
        List of prefix strings to remove (e.g. ``["sg-", "sg"]``).
    strip_suffixes
        List of suffix strings to remove (e.g. ``["_KO", "_KD", "_P1P2"]``).
    strip_suffix_regex
        A Python regex applied via ``pd.Series.str.replace`` after
        prefix/suffix stripping.
    control_aliases
        Additional strings (case-insensitive) treated as control labels.
        The built-in aliases (``ntc``, ``ctrl``, ``scramble``, …) are always
        included.
    canonical_control
        Canonical control label substituted for all matched aliases.
    inplace
        If True, write result back and return ``None``.
        If False, return a Series without modifying the file.

    Returns
    -------
    pd.Series or None
        Normalised labels when ``inplace=False``, else ``None``.
    """
    path = Path(path)
    df = load_obs(path)

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in obs. "
            f"Available: {list(df.columns)}"
        )
    labels = df[column].astype(str)

    # Step 1: strip prefixes (vectorised)
    if strip_prefixes:
        for prefix in strip_prefixes:
            labels = labels.str.replace(
                "^" + _re.escape(prefix), "", regex=True
            )

    # Step 2: strip suffixes (vectorised)
    if strip_suffixes:
        for suffix in strip_suffixes:
            labels = labels.str.replace(
                _re.escape(suffix) + "$", "", regex=True
            )

    # Step 3: custom regex substitution (vectorised)
    if strip_suffix_regex:
        labels = labels.str.replace(strip_suffix_regex, "", regex=True)

    # Step 4: unify control labels
    all_aliases = set(_DEFAULT_CONTROL_ALIASES)
    if control_aliases:
        all_aliases.update(a.lower() for a in control_aliases)
    is_control = labels.str.lower().isin(all_aliases)
    labels = labels.where(~is_control, other=canonical_control)

    if not inplace:
        return labels

    if isinstance(df[column].dtype, pd.CategoricalDtype):
        df[column] = pd.Categorical(labels.to_numpy())
    else:
        df[column] = labels.to_numpy()

    write_obs(path, df)
    return None


# =============================================================================
# Feature 4 — Auto-detection of metadata columns
# =============================================================================

_PERTURBATION_COL_ALIASES: frozenset[str] = frozenset({
    "perturbation", "gene", "gene_target", "condition",
    "guide_identity", "target_gene_name", "gene_name", "sgrna",
    "guide", "guide_id", "sgrna_name", "target",
})

_GENE_SYMBOL_COL_ALIASES: frozenset[str] = frozenset({
    "gene_symbols", "gene_name", "gene", "symbol",
    "hgnc_symbol", "gene_symbol", "feature_name",
})

_CTRL_TERMS: frozenset[str] = frozenset({
    "ctrl", "control", "nontarget", "non-target", "non_target",
    "ntc", "scramble", "scrambled",
})


def detect_perturbation_column(
    adata: "str | Path | AnnData | ad.AnnData",
    *,
    control_label: str | None = None,
    min_unique: int = 2,
    verbose: int | bool = True,
) -> str | None:
    """Heuristically identify the obs column containing perturbation labels.

    Scoring:

    * +3 if column name matches known aliases (``perturbation``,
      ``gene_target``, …).
    * +2 if dtype is categorical or object.
    * +1 if unique-value count is in [``min_unique``, 5000].
    * +2 if at least one value matches a known control synonym or
      ``control_label`` when provided.

    Parameters
    ----------
    adata
        Backed AnnData, :class:`~crispyx.data.AnnData`, or path to h5ad file.
    control_label
        Known control label; boosts the score for columns containing it.
    min_unique
        Minimum number of unique values required for the +1 bonus.
    verbose
        Log the detected column name.

    Returns
    -------
    str or None
        Column name with the highest score, or ``None`` if no column scores
        above zero.
    """
    path = resolve_data_path(adata)
    obs = load_obs(path)

    scores: dict[str, int] = {}
    for col in obs.columns:
        score = 0
        if col.lower() in _PERTURBATION_COL_ALIASES:
            score += 3
        dtype = obs[col].dtype
        if isinstance(dtype, pd.CategoricalDtype) or dtype == object:
            score += 2
        try:
            n_unique = int(obs[col].nunique())
        except Exception:
            n_unique = 0
        if min_unique <= n_unique <= 5000:
            score += 1
        lower_vals = obs[col].astype(str).str.lower()
        if control_label is not None:
            if (lower_vals == control_label.lower()).any():
                score += 2
        else:
            if lower_vals.isin(_CTRL_TERMS).any():
                score += 2
        scores[col] = score

    if not scores:
        return None
    best_col, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score <= 0:
        return None
    if verbose:
        logger.info(
            "Detected perturbation column: '%s' (score=%d).", best_col, best_score
        )
    _messages.vprint(
        verbose, "detect_perturbation_column", f"Detected '{best_col}' (score={best_score})",
    )
    return best_col


def detect_gene_symbol_column(
    adata: "str | Path | AnnData | ad.AnnData",
    *,
    verbose: int | bool = True,
) -> str | None:
    """Heuristically identify the var column containing gene symbols.

    Scoring:

    * +3 if column name matches known aliases (``gene_symbols``, ``symbol``, …).
    * +2 if values pass :func:`_validate_gene_symbols` without error.
    * +1 if values do **not** start with Ensembl prefixes.

    Returns ``None`` when no column qualifies, which signals that
    ``var_names`` should be used as a fallback.

    Parameters
    ----------
    adata
        Backed AnnData, :class:`~crispyx.data.AnnData`, or path to h5ad file.
    verbose
        Log the detected column name.

    Returns
    -------
    str or None
    """
    path = resolve_data_path(adata)
    var = load_var(path)

    ensembl_3 = frozenset(p[:3].upper() for p in ENSEMBL_PREFIXES)
    scores: dict[str, int] = {}
    for col in var.columns:
        score = 0
        if col.lower() in _GENE_SYMBOL_COL_ALIASES:
            score += 3
        try:
            _validate_gene_symbols(var[col].astype(str))
            score += 2
        except ValueError:
            pass
        prefixes = var[col].astype(str).str.upper().str.slice(0, 3)
        if not prefixes.isin(ensembl_3).any():
            score += 1
        scores[col] = score

    if not scores:
        return None
    best_col, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score <= 0:
        return None
    if verbose:
        logger.info(
            "Detected gene symbol column: '%s' (score=%d).", best_col, best_score
        )
    _messages.vprint(
        verbose, "detect_gene_symbol_column", f"Detected '{best_col}' (score={best_score})",
    )
    return best_col


def infer_columns(
    adata: "str | Path | AnnData | ad.AnnData",
    *,
    control_label: str | None = None,
    verbose: int | bool = True,
) -> dict[str, str | None]:
    """Detect perturbation and gene-symbol columns in a single call.

    Parameters
    ----------
    adata
        Backed AnnData, :class:`~crispyx.data.AnnData`, or path to h5ad file.
    control_label
        Known control label forwarded to :func:`detect_perturbation_column`.
    verbose
        Log detected column names.

    Returns
    -------
    dict
        ``{"perturbation_column": ..., "gene_name_column": ...}`` where each
        value is the detected column name or ``None``.
    """
    return {
        "perturbation_column": detect_perturbation_column(
            adata, control_label=control_label, verbose=verbose
        ),
        "gene_name_column": detect_gene_symbol_column(adata, verbose=verbose),
    }


# =============================================================================
# Feature 5 — Overlap analysis utilities
# =============================================================================

@dataclass
class OverlapResult:
    """Pairwise overlap statistics between named sets.

    Attributes
    ----------
    count_matrix
        (n_sets × n_sets) DataFrame of pairwise intersection sizes.
    jaccard_matrix
        (n_sets × n_sets) DataFrame of Jaccard similarity coefficients.
    set_sizes
        Series of sizes for each input set.
    """

    count_matrix: pd.DataFrame
    jaccard_matrix: pd.DataFrame
    set_sizes: pd.Series


def compute_overlap(
    sets_dict: dict[str, "set | list"],
    *,
    metric: Literal["count", "jaccard", "both"] = "both",
) -> OverlapResult:
    """Compute pairwise overlap statistics between named sets.

    Parameters
    ----------
    sets_dict
        Mapping of name → set (or list, converted to set).
    metric
        Which matrices to populate: ``"count"``, ``"jaccard"``, or
        ``"both"`` (default).

    Returns
    -------
    OverlapResult
        Object with ``count_matrix``, ``jaccard_matrix``, and ``set_sizes``.

    Examples
    --------
    >>> result = cx.tl.compute_overlap({
    ...     "dataset_A": {"BRCA1", "TP53", "EGFR"},
    ...     "dataset_B": {"TP53", "KRAS"},
    ... })
    >>> result.jaccard_matrix
    """
    names = list(sets_dict.keys())
    sets: dict[str, set] = {k: set(v) for k, v in sets_dict.items()}
    n = len(names)

    count_arr = np.zeros((n, n), dtype=np.int64)
    jaccard_arr = np.zeros((n, n), dtype=np.float64)

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            inter = len(sets[name_i] & sets[name_j])
            if metric in ("count", "both"):
                count_arr[i, j] = inter
            if metric in ("jaccard", "both"):
                union = len(sets[name_i] | sets[name_j])
                jaccard_arr[i, j] = inter / union if union > 0 else 0.0

    sizes = pd.Series({k: len(v) for k, v in sets.items()}, name="set_size")
    return OverlapResult(
        count_matrix=pd.DataFrame(count_arr, index=names, columns=names),
        jaccard_matrix=pd.DataFrame(jaccard_arr, index=names, columns=names),
        set_sizes=sizes,
    )
