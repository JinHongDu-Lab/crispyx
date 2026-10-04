"""PCA, neighbors and UMAP, checked against scikit-learn where it applies.

``use_highly_variable=True`` is covered end to end in ``test_hvg.py``.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import crispyx as cx
from crispyx.data import (
    calculate_pca_chunk_size,
    write_obsm_to_h5ad,
    write_obsp_to_h5ad,
    write_uns_dict_to_h5ad,
    write_varm_to_h5ad,
)
from crispyx.dimred import _streaming_pca_incremental, _streaming_pca_sparse_cov, neighbors, pca, umap


def _counts(n_obs: int = 400, n_vars: int = 60, seed: int = 0) -> np.ndarray:
    """Sparse counts whose first five genes carry most of the variance, so the
    leading principal components are well separated."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n_obs, n_vars)) < 0.3) * rng.poisson(4, (n_obs, n_vars))
    X[:, :5] *= 4
    return X.astype(np.float32)


def _write(path: Path, X: np.ndarray, **slots) -> Path:
    adata = ad.AnnData(sparse.csr_matrix(X), **slots)
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    adata.var_names = [f"gene_{j}" for j in range(X.shape[1])]
    adata.write(path)
    return path


def _assert_same_up_to_sign(actual: np.ndarray, expected: np.ndarray, **tol) -> None:
    """Principal axes are defined up to sign; compare column by column."""
    for j in range(expected.shape[1]):
        sign = np.sign(actual[:, j] @ expected[:, j]) or 1.0
        np.testing.assert_allclose(sign * actual[:, j], expected[:, j], err_msg=f"component {j}", **tol)


@pytest.fixture
def backed(tmp_path):
    X = _counts()
    adata = ad.read_h5ad(_write(tmp_path / "pca.h5ad", X), backed="r")
    yield adata, X.astype(np.float64)
    adata.file.close()


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_obs, n_vars, kwargs, expected_method",
    [
        (100_000, 5_000, dict(available_memory_gb=32, method="auto"), "sparse_cov"),   # ~200 MB covariance
        (100_000, 50_000, dict(available_memory_gb=16, method="auto"), "incremental"),  # ~20 GB covariance
        (100_000, 5_000, dict(method="incremental"), "incremental"),
    ],
    ids=["auto-small", "auto-large", "explicit"],
)
def test_pca_chunk_size_selects_method(n_obs, n_vars, kwargs, expected_method):
    chunk_size, method = calculate_pca_chunk_size(n_obs, n_vars, **kwargs)
    assert isinstance(chunk_size, int) and chunk_size > 0
    assert method == expected_method


def test_pca_chunk_size_respects_bounds():
    chunk_size, _ = calculate_pca_chunk_size(100_000, 1_000, available_memory_gb=1, min_chunk=128, max_chunk=256)
    assert 128 <= chunk_size <= 256


def test_sparse_covariance_pca_matches_sklearn(backed):
    adata, X = backed
    scores, components, variance_ratio, info = _streaming_pca_sparse_cov(
        adata, n_comps=10, show_progress=False, return_info=True,
    )
    ref = PCA(10, svd_solver="full").fit(X)
    np.testing.assert_allclose(variance_ratio, ref.explained_variance_ratio_, atol=1e-12)
    _assert_same_up_to_sign(components.T, ref.components_.T, atol=1e-10)
    _assert_same_up_to_sign(scores, ref.transform(X), atol=1e-8)
    assert info is not None


def test_incremental_pca_matches_sklearn_on_leading_components(backed):
    """Incremental PCA is approximate; the well-separated leading components agree."""
    adata, X = backed
    scores, components, variance_ratio, _ = _streaming_pca_incremental(
        adata, n_comps=10, chunk_size=64, show_progress=False,
    )
    ref = PCA(10, svd_solver="full").fit(X)
    assert components.shape == (10, X.shape[1])
    np.testing.assert_allclose(variance_ratio, ref.explained_variance_ratio_, atol=0.01)
    _assert_same_up_to_sign(scores[:, :5], ref.transform(X)[:, :5], rtol=0.01, atol=0.05)


@pytest.mark.parametrize("use_highly_variable", [False, True])
def test_pca_writes_results_to_the_backed_file(tmp_path, use_highly_variable):
    """cx.pp.pca on an on-disk AnnData writes X_pca / PCs / uns to the file,
    restricted to the highly variable genes only when asked."""
    X = _counts()
    hvg = np.arange(X.shape[1]) < 30
    path = _write(tmp_path / "pca.h5ad", X)
    adata = ad.read_h5ad(path)
    adata.var["highly_variable"] = hvg
    adata.write(path)

    backed = cx.read_h5ad_ondisk(path)
    assert cx.pp.pca(backed, n_comps=8, use_highly_variable=use_highly_variable, show_progress=False) is None

    reopened = cx.read_h5ad_ondisk(path)
    used = X[:, hvg] if use_highly_variable else X
    ref = PCA(8, svd_solver="full").fit(used.astype(np.float64))
    _assert_same_up_to_sign(np.asarray(reopened.obsm["X_pca"]), ref.transform(used), rtol=1e-4, atol=1e-3)
    pcs = np.asarray(reopened.varm["PCs"])
    assert pcs.shape == (X.shape[1], 8)
    np.testing.assert_array_equal(np.any(pcs != 0, axis=1), hvg if use_highly_variable else np.ones_like(hvg))
    info = reopened.uns["pca"].load()
    assert info["n_comps"] == 8
    assert bool(info["use_highly_variable"]) is use_highly_variable
    np.testing.assert_allclose(info["variance_ratio"], ref.explained_variance_ratio_, rtol=1e-4)


def test_pca_copy_from_a_path_returns_the_same_embedding(tmp_path):
    path = _write(tmp_path / "pca.h5ad", _counts())
    copied = cx.pp.pca(path, n_comps=8, copy=True, show_progress=False)
    assert isinstance(copied, ad.AnnData)
    backed = cx.read_h5ad_ondisk(path)
    cx.pp.pca(backed, n_comps=8, show_progress=False)
    np.testing.assert_allclose(copied.obsm["X_pca"], np.asarray(cx.read_h5ad_ondisk(path).obsm["X_pca"]))


# ---------------------------------------------------------------------------
# Neighbors
# ---------------------------------------------------------------------------

def _embedded(n_obs: int = 150, n_dims: int = 8, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    return ad.AnnData(sparse.csr_matrix((n_obs, 5)), obsm={"X_pca": rng.normal(size=(n_obs, n_dims)).astype(np.float32)})


def _neighbor_sets(distances) -> list[set[int]]:
    d = sparse.csr_matrix(distances)
    return [set(d.indices[d.indptr[i]:d.indptr[i + 1]]) for i in range(d.shape[0])]


@pytest.mark.parametrize("n_pcs", [None, 5])
def test_sklearn_neighbors_match_nearest_neighbors(n_pcs):
    adata = _embedded()
    neighbors(adata, n_neighbors=10, n_pcs=n_pcs, method="sklearn", show_progress=False)
    emb = adata.obsm["X_pca"][:, : n_pcs or adata.obsm["X_pca"].shape[1]]
    _, idx = NearestNeighbors(n_neighbors=10).fit(emb).kneighbors(emb)
    assert _neighbor_sets(adata.obsp["distances"]) == [set(row) for row in idx]
    assert adata.obsp["connectivities"].shape == (adata.n_obs, adata.n_obs)
    params = adata.uns["neighbors"]["params"]
    assert (params["n_neighbors"], params["method"], params["metric"]) == (10, "sklearn", "euclidean")
    assert params["n_pcs"] == (n_pcs or emb.shape[1])


def test_umap_method_neighbors_agree_with_exact_neighbors():
    pytest.importorskip("pynndescent")
    adata = _embedded(n_obs=100)
    neighbors(adata, n_neighbors=10, method="umap", show_progress=False)
    emb = adata.obsm["X_pca"]
    _, idx = NearestNeighbors(n_neighbors=10).fit(emb).kneighbors(emb)
    found = _neighbor_sets(adata.obsp["distances"])
    overlap = np.mean([len(f & set(row)) / 10 for f, row in zip(found, idx)])
    assert overlap > 0.95  # approximate search, exact on almost every row at this size


def test_neighbors_copy_leaves_the_input_untouched():
    adata = _embedded()
    result = neighbors(adata, n_neighbors=10, copy=True, method="sklearn", show_progress=False)
    assert "distances" in result.obsp
    assert "distances" not in adata.obsp


def test_neighbors_without_pca_raises():
    with pytest.raises(ValueError, match="not found in adata.obsm"):
        neighbors(ad.AnnData(sparse.random(50, 20, density=0.3, format="csr", random_state=0)), n_neighbors=10)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_components", [2, 3])
def test_umap_embedding(n_components):
    adata = _embedded()
    neighbors(adata, n_neighbors=10, method="sklearn", show_progress=False)
    result = umap(adata, n_components=n_components, min_dist=0.3, spread=1.5, copy=True)
    assert result is not adata
    assert "X_umap" not in adata.obsm
    assert result.obsm["X_umap"].shape == (adata.n_obs, n_components)
    assert np.isfinite(result.obsm["X_umap"]).all()
    assert "umap" in result.uns


def test_umap_without_neighbors_raises():
    with pytest.raises(ValueError, match="not found in adata.uns"):
        umap(_embedded())


def test_pca_neighbors_umap_pipeline_on_a_backed_file(tmp_path):
    path = _write(tmp_path / "pipeline.h5ad", _counts(n_obs=100))
    backed = cx.read_h5ad_ondisk(path)
    cx.pp.pca(backed, n_comps=15, show_progress=False)
    assert cx.pp.neighbors(backed, n_neighbors=5, method="sklearn", show_progress=False) is None
    cx.tl.umap(backed)

    reopened = cx.read_h5ad_ondisk(path)
    assert reopened.obsm["X_pca"].shape == (100, 15)
    assert reopened.obsp["distances"].shape == (100, 100)
    assert reopened.obsp["connectivities"].nnz > 0
    assert reopened.obsm["X_umap"].shape == (100, 2)
    assert reopened.uns["neighbors"].load()["params"]["n_neighbors"] == 5
    assert "umap" in reopened.uns


# ---------------------------------------------------------------------------
# h5ad write helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot", ["obsm", "varm", "obsp", "uns"])
def test_write_helpers_round_trip(tmp_path, slot):
    rng = np.random.default_rng(0)
    path = tmp_path / "slots.h5ad"
    ad.AnnData(sparse.random(50, 20, density=0.3, format="csr", random_state=0)).write(path)
    if slot == "obsm":
        value = rng.normal(size=(50, 10)).astype(np.float32)
        write_obsm_to_h5ad(path, "embedding", value)
        np.testing.assert_array_equal(ad.read_h5ad(path).obsm["embedding"], value)
    elif slot == "varm":
        value = rng.normal(size=(20, 5)).astype(np.float32)
        write_varm_to_h5ad(path, "PCs", value)
        np.testing.assert_array_equal(ad.read_h5ad(path).varm["PCs"], value)
    elif slot == "obsp":
        value = sparse.random(50, 50, density=0.1, format="csr", random_state=1)
        write_obsp_to_h5ad(path, "connectivities", value)
        np.testing.assert_array_equal(ad.read_h5ad(path).obsp["connectivities"].toarray(), value.toarray())
    else:
        value = {"variance_ratio": np.array([0.5, 0.3, 0.1]), "n_comps": 10, "method": "sparse_cov"}
        write_uns_dict_to_h5ad(path, "pca", value)
        stored = ad.read_h5ad(path).uns["pca"]
        np.testing.assert_array_equal(stored["variance_ratio"], value["variance_ratio"])
        assert stored["n_comps"] == 10 and stored["method"] == "sparse_cov"
