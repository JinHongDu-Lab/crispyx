"""Unit tests for crispyx.data.convert_to_csr."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from crispyx.data import convert_to_csr, convert_to_csc, get_matrix_storage_format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csc_h5ad(tmp_path: Path, name: str = "csc.h5ad") -> tuple[Path, np.ndarray]:
    """Write a small CSC h5ad and return (path, dense_matrix)."""
    dense = np.array(
        [
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [4.0, 0.0, 0.0, 1.0],
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "ctrl", "A", "A", "B", "B"]},
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(
        {"gene_symbols": ["G1", "G2", "G3", "G4"]},
        index=[f"g{i}" for i in range(4)],
    )
    adata = ad.AnnData(sp.csc_matrix(dense), obs=obs, var=var)
    path = tmp_path / name
    adata.write(path)
    return path, dense


def _make_csr_h5ad(tmp_path: Path, name: str = "csr.h5ad") -> tuple[Path, np.ndarray]:
    """Write a small CSR h5ad and return (path, dense_matrix)."""
    dense = np.eye(4, dtype=np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(4)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    adata = ad.AnnData(sp.csr_matrix(dense), obs=obs, var=var)
    path = tmp_path / name
    adata.write(path)
    return path, dense


def _make_dense_h5ad(tmp_path: Path, name: str = "dense.h5ad") -> tuple[Path, np.ndarray]:
    """Write a small dense h5ad and return (path, dense_matrix)."""
    dense = np.array(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    adata = ad.AnnData(dense, obs=obs, var=var)
    path = tmp_path / name
    adata.write(path)
    return path, dense


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_convert_to_csr_basic(tmp_path):
    """CSR file reconstructs the same dense matrix as the CSC source."""
    src, dense = _make_csc_h5ad(tmp_path)
    out = tmp_path / "out_csr.h5ad"

    result = convert_to_csr(src, output_path=out, verbose=False)

    # Verify HDF5 encoding metadata.
    with h5py.File(out, "r") as f:
        enc = f["X"].attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode()
        assert enc == "csr_matrix", f"expected csr_matrix, got {enc!r}"
        shape = f["X"].attrs["shape"]
        assert list(shape) == [6, 4]
        # indptr length must be n_obs + 1
        assert f["X/indptr"].shape[0] == 7  # n_obs + 1 = 6 + 1

    # Verify round-trip reconstruction via anndata.
    loaded = ad.read_h5ad(out)
    np.testing.assert_array_almost_equal(
        loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X,
        dense,
    )


def test_convert_to_csr_already_csr(tmp_path):
    """If the source is already CSR, no new file is written."""
    src, dense = _make_csr_h5ad(tmp_path)
    out = tmp_path / "should_not_exist.h5ad"

    result = convert_to_csr(src, output_path=out, verbose=False)

    assert not out.exists(), "convert_to_csr should not write a new file when input is already CSR"
    assert Path(result.filename) == src or Path(result.filename).resolve() == src.resolve()


def test_convert_to_csr_default_output_path(tmp_path):
    """Without an explicit output_path, a '{stem}_cx_csr'-named file is created next to the source."""
    src, _ = _make_csc_h5ad(tmp_path)

    result = convert_to_csr(src, verbose=False)

    expected = tmp_path / "csc_cx_csr.h5ad"
    assert expected.exists(), f"Expected output file not found: {expected}"


def test_convert_to_csr_preserves_obs_var(tmp_path):
    """obs and var DataFrames survive the conversion unchanged."""
    src, _ = _make_csc_h5ad(tmp_path)
    out = tmp_path / "out_meta.h5ad"

    convert_to_csr(src, output_path=out, verbose=False)

    original = ad.read_h5ad(src)
    converted = ad.read_h5ad(out)

    pd.testing.assert_frame_equal(original.obs, converted.obs)
    pd.testing.assert_frame_equal(original.var, converted.var)


def test_convert_to_csr_row_access(tmp_path):
    """CSR file allows efficient row slicing that matches the CSC source."""
    src, dense = _make_csc_h5ad(tmp_path)
    out = tmp_path / "out_row.h5ad"

    convert_to_csr(src, output_path=out, verbose=False)

    loaded = ad.read_h5ad(out)
    X = loaded.X

    for i in range(dense.shape[0]):
        row_csr = X[i, :].toarray().ravel() if sp.issparse(X) else X[i, :]
        np.testing.assert_array_almost_equal(row_csr, dense[i, :])


def test_convert_to_csr_empty_matrix(tmp_path):
    """Empty (all-zero) matrix is handled without error."""
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(2)])
    adata = ad.AnnData(sp.csc_matrix((3, 2), dtype=np.float32), obs=obs, var=var)
    src = tmp_path / "empty.h5ad"
    adata.write(src)
    out = tmp_path / "empty_csr.h5ad"

    result = convert_to_csr(src, output_path=out, verbose=False)

    assert out.exists()
    loaded = ad.read_h5ad(out)
    assert loaded.shape == (3, 2)


def test_convert_to_csr_get_matrix_storage_format(tmp_path):
    """get_matrix_storage_format reports 'csr' after conversion."""
    src, _ = _make_csc_h5ad(tmp_path)
    out = tmp_path / "out_fmt.h5ad"

    convert_to_csr(src, output_path=out, verbose=False)

    assert get_matrix_storage_format(out) == "csr"


def test_convert_to_csr_via_cx_pp(tmp_path):
    """convert_to_csr is accessible as cx.pp.convert_to_csr."""
    import crispyx as cx

    src, dense = _make_csc_h5ad(tmp_path)
    out = tmp_path / "cx_pp_csr.h5ad"

    result = cx.pp.convert_to_csr(src, output_path=out, verbose=False)

    loaded = ad.read_h5ad(out)
    np.testing.assert_array_almost_equal(
        loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X,
        dense,
    )


def test_convert_to_csr_small_chunk(tmp_path):
    """chunk_size=1 (one column per pass on CSC) produces the same result as default."""
    src, dense = _make_csc_h5ad(tmp_path)
    out_default = tmp_path / "default_chunk.h5ad"
    out_small = tmp_path / "small_chunk.h5ad"

    convert_to_csr(src, output_path=out_default, verbose=False)
    convert_to_csr(src, output_path=out_small, chunk_size=1, verbose=False)

    arr_default = ad.read_h5ad(out_default).X
    arr_small = ad.read_h5ad(out_small).X

    a = arr_default.toarray() if sp.issparse(arr_default) else arr_default
    b = arr_small.toarray() if sp.issparse(arr_small) else arr_small
    np.testing.assert_array_almost_equal(a, b)


def test_convert_to_csr_from_dense(tmp_path):
    """Dense on-disk matrix is correctly converted to CSR."""
    src, dense = _make_dense_h5ad(tmp_path)
    out = tmp_path / "dense_to_csr.h5ad"

    assert get_matrix_storage_format(src) == "dense"

    convert_to_csr(src, output_path=out, verbose=False)

    assert get_matrix_storage_format(out) == "csr"
    loaded = ad.read_h5ad(out)
    np.testing.assert_array_almost_equal(
        loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X,
        dense,
    )


def test_convert_to_csr_roundtrip_csc_csr(tmp_path):
    """CSR → CSC → CSR round-trip preserves data exactly."""
    dense = np.array(
        [[5.0, 0.0, 1.0], [0.0, 3.0, 0.0], [2.0, 0.0, 4.0]],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])

    # Start with CSR.
    adata = ad.AnnData(sp.csr_matrix(dense), obs=obs, var=var)
    csr_path = tmp_path / "start_csr.h5ad"
    adata.write(csr_path)

    # CSR → CSC.
    csc_path = tmp_path / "mid_csc.h5ad"
    convert_to_csc(csr_path, output_path=csc_path, verbose=False)
    assert get_matrix_storage_format(csc_path) == "csc"

    # CSC → CSR.
    final_csr_path = tmp_path / "final_csr.h5ad"
    convert_to_csr(csc_path, output_path=final_csr_path, verbose=False)
    assert get_matrix_storage_format(final_csr_path) == "csr"

    loaded = ad.read_h5ad(final_csr_path)
    np.testing.assert_array_almost_equal(
        loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X,
        dense,
    )


def test_convert_to_csr_int64_indptr(tmp_path):
    """Verify dtype selection: int32 for small NNZ."""
    src, _ = _make_csc_h5ad(tmp_path)
    out = tmp_path / "dtype_check.h5ad"

    convert_to_csr(src, output_path=out, verbose=False)

    with h5py.File(out, "r") as f:
        # For 6×4 matrix with 10 non-zeros (< INT32_MAX), indptr should be int32.
        assert f["X/indptr"].dtype in (np.int32, np.int64)
        # indices stores column IDs (0–3), must be integer type.
        assert np.issubdtype(f["X/indices"].dtype, np.integer)


def test_convert_to_csr_warns_when_disk_space_low(tmp_path, monkeypatch):
    """A near-full destination filesystem should warn but not block the conversion."""
    import shutil
    import types

    src, dense = _make_csc_h5ad(tmp_path)
    out = tmp_path / "low_disk.h5ad"

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda p: types.SimpleNamespace(total=1000, used=999, free=1),
    )
    with pytest.warns(UserWarning, match="pp.convert_to_csr"):
        convert_to_csr(src, output_path=out, verbose=False)

    result = ad.read_h5ad(out).X
    arr = result.toarray() if sp.issparse(result) else result
    np.testing.assert_array_almost_equal(arr, dense)
