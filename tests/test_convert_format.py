"""crispyx.data.convert_to_csc / convert_to_csr, tested symmetrically."""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
import crispyx.data as cxd
from crispyx.data import (
    _contiguous_bands,
    _conversion_buffer_budget_bytes,
    convert_to_csc,
    convert_to_csr,
    get_matrix_storage_format,
)

CONVERT = {"csc": convert_to_csc, "csr": convert_to_csr}
OTHER = {"csc": "csr", "csr": "csc"}
targets = pytest.mark.parametrize("target", ["csc", "csr"])


def _dense(dtype=np.float64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.random((50, 30)) < 0.3) * rng.integers(1, 9, (50, 30))).astype(dtype)


def _write(path: Path, dense: np.ndarray, fmt: str) -> Path:
    """Write ``dense`` as an h5ad whose X is stored as ``fmt`` ('csr', 'csc' or 'dense')."""
    n_obs, n_vars = dense.shape
    obs = pd.DataFrame(
        {"perturbation": np.resize(["ctrl", "A", "B"], n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"gene_symbols": [f"G{j}" for j in range(n_vars)]}, index=[f"g{j}" for j in range(n_vars)])
    X = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[fmt](dense)
    ad.AnnData(X, obs=obs, var=var).write(path)
    return path


def _assert_canonical(path: Path, dense: np.ndarray, fmt: str) -> None:
    """The stored arrays equal scipy's canonical (sorted-index) matrix, byte for byte."""
    ref = sp.csc_matrix(dense) if fmt == "csc" else sp.csr_matrix(dense)
    ref.sort_indices()
    with h5py.File(path, "r") as f:
        encoding = f["X"].attrs["encoding-type"]
        assert (encoding.decode() if isinstance(encoding, bytes) else encoding) == f"{fmt}_matrix"
        assert list(f["X"].attrs["shape"]) == list(dense.shape)
        np.testing.assert_array_equal(f["X/indptr"][:], ref.indptr)
        np.testing.assert_array_equal(f["X/indices"][:], ref.indices)
        np.testing.assert_array_equal(f["X/data"][:], ref.data)
        assert f["X/data"].dtype == dense.dtype
    assert get_matrix_storage_format(path) == fmt


@targets
@pytest.mark.parametrize("source_fmt", ["other", "dense"])
def test_conversion_is_exact_and_keeps_metadata(tmp_path, target, source_fmt):
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target] if source_fmt == "other" else "dense")
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)
    original, converted = ad.read_h5ad(src), ad.read_h5ad(out)
    pd.testing.assert_frame_equal(converted.obs, original.obs)
    pd.testing.assert_frame_equal(converted.var, original.var)


@targets
def test_chunking_and_memory_bands_do_not_change_the_file(tmp_path, target):
    """chunk_size=1 and a tiny memory budget (several bands) give the same file."""
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    nnz_per_band_axis = np.count_nonzero(dense, axis=0 if target == "csc" else 1)
    assert len(_contiguous_bands(nnz_per_band_axis, per_item_bytes=12,
                                 budget_bytes=_conversion_buffer_budget_bytes(1e-6))) > 1
    for name, kw in [("one", dict(chunk_size=1)), ("seven", dict(chunk_size=7)),
                     ("banded", dict(chunk_size=7, memory_limit_gb=1e-6))]:
        out = tmp_path / f"{name}.h5ad"
        CONVERT[target](src, output_path=out, verbose=False, **kw).close()
        _assert_canonical(out, dense, target)


@targets
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_conversion_preserves_value_dtype(tmp_path, target, dtype):
    dense = _dense(dtype)
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)


@targets
def test_source_already_in_target_format_is_returned(tmp_path, target):
    src = _write(tmp_path / "src.h5ad", _dense(), target)
    out = tmp_path / "should_not_exist.h5ad"
    result = CONVERT[target](src, output_path=out, verbose=False)
    assert not out.exists()
    assert Path(result.filename).resolve() == src.resolve()
    result.file.close()


@targets
def test_default_output_path_is_next_to_the_source(tmp_path, target):
    src = _write(tmp_path / "screen.h5ad", _dense(), OTHER[target])
    CONVERT[target](src, verbose=False).close()
    assert get_matrix_storage_format(tmp_path / f"screen_cx_{target}.h5ad") == target


@targets
def test_empty_matrix(tmp_path, target):
    dense = np.zeros((3, 2), dtype=np.float32)
    src = _write(tmp_path / "empty.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)


@targets
def test_warns_when_disk_space_low(tmp_path, target, monkeypatch):
    """A near-full destination filesystem warns but does not block the conversion."""
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1000, used=999, free=1))
    with pytest.warns(UserWarning, match=f"pp.convert_to_{target}"):
        CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)


@targets
def test_interrupted_conversion_leaves_no_output_file(tmp_path, target, monkeypatch):
    """A conversion killed while filling the pre-sized output must not leave a
    structurally valid file with zero-filled bands at output_path."""
    src = _write(tmp_path / "src.h5ad", _dense(), OTHER[target])
    out = tmp_path / "out.h5ad"

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cxd, "_scatter_by_key", boom)
    with pytest.raises(KeyboardInterrupt):
        CONVERT[target](src, output_path=out, verbose=False)
    assert not out.exists()
    assert list(tmp_path.glob(".*partial")) == []


@targets
def test_pp_namespace_forwards_to_the_function(tmp_path, target):
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    getattr(cx.pp, f"convert_to_{target}")(src, output_path=out, memory_limit_gb=1e-6, verbose=False).close()
    _assert_canonical(out, dense, target)


def test_contiguous_bands_respects_budget_and_covers_everything():
    counts = np.array([5, 1, 9, 2, 2, 2, 40, 1])
    bands = _contiguous_bands(counts, per_item_bytes=1, budget_bytes=10)
    assert bands[0][0] == 0 and bands[-1][1] == len(counts)
    assert all(b[1] == nb[0] for b, nb in zip(bands, bands[1:]))  # contiguous
    for start, stop in bands:
        # Within budget unless a single index alone exceeds it (index 6: 40).
        assert counts[start:stop].sum() <= 10 or stop - start == 1
    assert (6, 7) in bands
    assert _contiguous_bands(counts, per_item_bytes=1, budget_bytes=1000) == [(0, len(counts))]
    assert _contiguous_bands(np.zeros(0, dtype=np.int64), per_item_bytes=1, budget_bytes=1) == [(0, 0)]
