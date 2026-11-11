"""Tests for cx.pp.downsample_counts / crispyx.data.downsample_counts.

Native, dependency-free streaming equivalent of
``scanpy.pp.downsample_counts(..., replace=False)``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _make_h5ad(
    tmp_path: Path,
    *,
    n_obs: int = 150,
    n_vars: int = 40,
    lam: float = 15.0,
    seed: int = 0,
    with_extra_slots: bool = False,
    with_raw: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    if with_extra_slots:
        adata.obsm["X_pca"] = rng.normal(size=(n_obs, 3)).astype(np.float32)
        adata.varm["PCs"] = rng.normal(size=(n_vars, 3)).astype(np.float32)
        adata.obsp["conn"] = sp.random(n_obs, n_obs, density=0.05, random_state=seed, format="csr")
        adata.varp["gene_net"] = sp.random(n_vars, n_vars, density=0.1, random_state=seed, format="csr")
        adata.layers["counts2"] = (counts * 2).astype(np.float32)
        adata.uns["some_key"] = "hello"
    if with_raw:
        adata.raw = adata

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.h5ad"
    adata.write(path)
    return path


def _lib_sizes(path: Path) -> np.ndarray:
    a = ad.read_h5ad(path)
    return np.asarray(a.X.sum(axis=1)).ravel()


def test_counts_per_cell_must_be_positive(tmp_path):
    path = _make_h5ad(tmp_path)
    with pytest.raises(ValueError):
        cx.pp.downsample_counts(path, counts_per_cell=0, output_path=tmp_path / "out.h5ad")
    with pytest.raises(ValueError):
        cx.pp.downsample_counts(path, counts_per_cell=-5, output_path=tmp_path / "out.h5ad")


def test_cells_above_target_thinned_exactly_and_below_unchanged(tmp_path):
    path = _make_h5ad(tmp_path, seed=1)
    lib_before = _lib_sizes(path)
    target = int(np.median(lib_before))  # roughly half above, half below

    out = cx.pp.downsample_counts(path, counts_per_cell=target, output_path=tmp_path / "out.h5ad", verbose=False)
    lib_after = _lib_sizes(out.path)

    assert (lib_after <= target).all()
    below = lib_before <= target
    np.testing.assert_array_equal(lib_after[below], lib_before[below])
    above = ~below
    if above.any():
        assert np.all(lib_after[above] == target)


def test_output_values_are_nonnegative_integers(tmp_path):
    path = _make_h5ad(tmp_path, seed=2)
    out = cx.pp.downsample_counts(path, counts_per_cell=50, output_path=tmp_path / "out.h5ad", verbose=False)
    X = ad.read_h5ad(out.path).X.toarray()
    assert (X >= 0).all()
    np.testing.assert_allclose(X, np.round(X))


def test_reproducible_with_same_random_state(tmp_path):
    path = _make_h5ad(tmp_path, seed=3)
    out1 = cx.pp.downsample_counts(
        path, counts_per_cell=50, random_state=7, output_path=tmp_path / "out1.h5ad", verbose=False,
    )
    out2 = cx.pp.downsample_counts(
        path, counts_per_cell=50, random_state=7, output_path=tmp_path / "out2.h5ad", verbose=False,
    )
    X1 = ad.read_h5ad(out1.path).X.toarray()
    X2 = ad.read_h5ad(out2.path).X.toarray()
    np.testing.assert_array_equal(X1, X2)


def test_different_random_state_differs(tmp_path):
    path = _make_h5ad(tmp_path, seed=3)
    out1 = cx.pp.downsample_counts(
        path, counts_per_cell=50, random_state=1, output_path=tmp_path / "out1.h5ad", verbose=False,
    )
    out2 = cx.pp.downsample_counts(
        path, counts_per_cell=50, random_state=2, output_path=tmp_path / "out2.h5ad", verbose=False,
    )
    X1 = ad.read_h5ad(out1.path).X.toarray()
    X2 = ad.read_h5ad(out2.path).X.toarray()
    assert not np.array_equal(X1, X2)


def test_result_independent_of_chunk_size(tmp_path):
    path = _make_h5ad(tmp_path, n_obs=120, seed=4)
    out1 = cx.pp.downsample_counts(
        path, counts_per_cell=40, chunk_size=7, output_path=tmp_path / "out1.h5ad", verbose=False,
    )
    out2 = cx.pp.downsample_counts(
        path, counts_per_cell=40, chunk_size=4096, output_path=tmp_path / "out2.h5ad", verbose=False,
    )
    X1 = ad.read_h5ad(out1.path).X.toarray()
    X2 = ad.read_h5ad(out2.path).X.toarray()
    np.testing.assert_array_equal(X1, X2)


def test_preserves_extra_slots(tmp_path):
    path = _make_h5ad(tmp_path, seed=5, with_extra_slots=True)
    src = ad.read_h5ad(path)
    out = cx.pp.downsample_counts(path, counts_per_cell=50, output_path=tmp_path / "out.h5ad", verbose=False)
    om = ad.read_h5ad(out.path)

    np.testing.assert_allclose(om.obsm["X_pca"], src.obsm["X_pca"])
    np.testing.assert_allclose(om.varm["PCs"], src.varm["PCs"])
    np.testing.assert_allclose(om.obsp["conn"].toarray(), src.obsp["conn"].toarray())
    np.testing.assert_allclose(om.varp["gene_net"].toarray(), src.varp["gene_net"].toarray())
    out_counts2 = om.layers["counts2"]
    out_counts2 = out_counts2.toarray() if sp.issparse(out_counts2) else out_counts2
    src_counts2 = src.layers["counts2"]
    src_counts2 = src_counts2.toarray() if sp.issparse(src_counts2) else src_counts2
    np.testing.assert_allclose(out_counts2, src_counts2)
    assert om.uns["some_key"] == "hello"


def test_raw_present_warns_and_is_dropped(tmp_path):
    path = _make_h5ad(tmp_path, seed=6, with_raw=True)
    with pytest.warns(UserWarning, match="raw"):
        out = cx.pp.downsample_counts(path, counts_per_cell=50, output_path=tmp_path / "out.h5ad", verbose=False)
    om = ad.read_h5ad(out.path)
    assert om.raw is None


def test_all_cells_already_below_target_is_a_noop(tmp_path):
    path = _make_h5ad(tmp_path, n_obs=30, n_vars=10, lam=2.0, seed=8)
    lib_before = _lib_sizes(path)
    target = int(lib_before.max()) + 1000
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = cx.pp.downsample_counts(path, counts_per_cell=target, output_path=tmp_path / "out.h5ad", verbose=False)
    lib_after = _lib_sizes(out.path)
    np.testing.assert_array_equal(lib_before, lib_after)


def test_empty_x_still_preserves_extra_slots(tmp_path):
    """An all-zero source X must not short-circuit before the slot copy-through."""
    path = _make_h5ad(tmp_path, n_obs=20, n_vars=8, lam=0.0, seed=9, with_extra_slots=True)
    src = ad.read_h5ad(path)
    out = cx.pp.downsample_counts(path, counts_per_cell=10, output_path=tmp_path / "out.h5ad", verbose=False)
    om = ad.read_h5ad(out.path)

    assert om.X.nnz == 0
    np.testing.assert_allclose(om.obsm["X_pca"], src.obsm["X_pca"])
    np.testing.assert_allclose(om.varm["PCs"], src.varm["PCs"])
    np.testing.assert_allclose(om.obsp["conn"].toarray(), src.obsp["conn"].toarray())
    np.testing.assert_allclose(om.varp["gene_net"].toarray(), src.varp["gene_net"].toarray())
    assert om.uns["some_key"] == "hello"


def test_dense_source_preserves_dtype(tmp_path):
    """A dense-stored int source must not be silently upcast/downcast to float32."""
    rng = np.random.default_rng(10)
    n_obs, n_vars = 40, 12
    counts = rng.integers(0, 30, size=(n_obs, n_vars)).astype(np.int32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    path = tmp_path / "dense.h5ad"
    ad.AnnData(counts, obs=obs, var=var).write(path)

    out = cx.pp.downsample_counts(path, counts_per_cell=15, output_path=tmp_path / "out.h5ad", verbose=False)
    assert ad.read_h5ad(out.path).X.dtype == np.int32


def test_non_integer_data_raises(tmp_path):
    rng = np.random.default_rng(11)
    n_obs, n_vars = 20, 8
    fractional = sp.csr_matrix((rng.poisson(15.0, size=(n_obs, n_vars)) * 1.37).astype(np.float32))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    norm_path = tmp_path / "normalized.h5ad"
    ad.AnnData(fractional, obs=obs, var=var).write(norm_path)

    with pytest.raises(ValueError, match="non-integer"):
        cx.pp.downsample_counts(norm_path, counts_per_cell=10, output_path=tmp_path / "out.h5ad", verbose=False)


def test_negative_data_raises(tmp_path):
    rng = np.random.default_rng(12)
    n_obs, n_vars = 20, 8
    negative = sp.csr_matrix((rng.poisson(15.0, size=(n_obs, n_vars)) - 100).astype(np.float32))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    neg_path = tmp_path / "negative.h5ad"
    ad.AnnData(negative, obs=obs, var=var).write(neg_path)

    with pytest.raises(ValueError, match="negative"):
        cx.pp.downsample_counts(neg_path, counts_per_cell=10, output_path=tmp_path / "out.h5ad", verbose=False)
