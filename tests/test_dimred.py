"""PCA, neighbors and UMAP, checked against scikit-learn where it applies.

``use_highly_variable=True`` is covered end to end in ``test_hvg.py``.
"""

from __future__ import annotations

import os
import tracemalloc
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import crispyx as cx
from crispyx.data import (
    write_obsm_to_h5ad,
    write_obsp_to_h5ad,
    write_uns_dict_to_h5ad,
    write_varm_to_h5ad,
)
from crispyx.dimred import _fit_pca, _pca_chunk_size, _pca_peak_bytes, _pca_scratch_path, neighbors, umap


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
    "n_vars, kwargs, expected_method",
    [
        (5_000, dict(memory_limit_gb=32), "covariance"),     # 0.2 GB matrix
        (50_000, dict(memory_limit_gb=16), "incremental"),   # 20 GB matrix
        (16_000, dict(memory_limit_gb=8), "covariance"),     # 2 GB matrix: fits, so the exact method
        (5_000, dict(memory_limit_gb=32, method="incremental"), "incremental"),
    ],
    ids=["auto-small", "auto-large", "auto-fits", "explicit"],
)
def test_pca_chunk_size_selects_method(n_vars, kwargs, expected_method):
    chunk_size, method = _pca_chunk_size(n_vars, 50, **kwargs)
    assert isinstance(chunk_size, int) and chunk_size > 0
    assert method == expected_method


def test_pca_chunk_size_fits_the_budget():
    chunk_size, method = _pca_chunk_size(20_000, 50, memory_limit_gb=1, method="incremental")
    assert 256 <= chunk_size <= 4096
    assert _pca_peak_bytes(method, 20_000, 50, chunk_size) <= 0.5e9 or chunk_size == 256
    assert _pca_chunk_size(1_000, 50, memory_limit_gb=1, min_chunk=128, max_chunk=256)[0] <= 256


def test_forced_covariance_over_the_budget_raises():
    with pytest.raises(MemoryError, match="method='incremental'"):
        _pca_chunk_size(30_000, 50, memory_limit_gb=4, method="covariance")  # 7.2 GB matrix


@pytest.mark.parametrize("dense", [False, True])
def test_covariance_pca_matches_sklearn(dense):
    X = _counts().astype(np.float64)
    adata = ad.AnnData(X if dense else sparse.csr_matrix(X))
    cx.pp.pca(adata, n_comps=10, method="covariance", chunk_size=64, show_progress=False)
    ref = PCA(10, svd_solver="full").fit(X)
    np.testing.assert_allclose(adata.uns["pca"]["variance_ratio"], ref.explained_variance_ratio_, atol=1e-12)
    np.testing.assert_allclose(adata.uns["pca"]["variance"], ref.explained_variance_, rtol=1e-10)  # n - 1
    _assert_same_up_to_sign(adata.varm["PCs"], ref.components_.T, atol=1e-6)  # stored as float32
    _assert_same_up_to_sign(adata.obsm["X_pca"], ref.transform(X), rtol=1e-5, atol=1e-5)
    assert adata.uns["pca"]["method"] == "covariance"
    assert adata.uns["pca"]["chunk_size"] == 64


@pytest.mark.parametrize("method", ["covariance", "incremental"])
def test_components_have_a_canonical_sign(method):
    """scikit-learn's convention: each component's largest loading is
    positive, whatever sign LAPACK returned, so X_pca is the same on any
    BLAS."""
    adata = ad.AnnData(sparse.csr_matrix(_counts()))
    cx.pp.pca(adata, n_comps=10, method=method, chunk_size=64, use_highly_variable=False, show_progress=False)
    pcs = adata.varm["PCs"]
    assert np.all(pcs[np.argmax(np.abs(pcs), axis=0), np.arange(pcs.shape[1])] > 0)


def test_covariance_pca_is_exact_for_genes_with_large_means():
    """E[x²] - E[x]² cancels for a gene whose mean dwarfs its spread; the
    accumulation shifts by the first chunk's means first."""
    X = _counts().astype(np.float64)
    X[:, :5] += 1e7
    fit = _fit_pca(sparse.csr_matrix(X), "covariance", 10, 64, None, show_progress=False)
    ref = PCA(10, svd_solver="full").fit(X)
    np.testing.assert_allclose(fit["variance_ratio"], ref.explained_variance_ratio_, rtol=1e-8)
    _assert_same_up_to_sign(fit["components"].T, ref.components_.T, atol=1e-7)


def test_covariance_pca_does_not_depend_on_the_chunk():
    matrix = sparse.csr_matrix(_counts())
    a, b = (_fit_pca(matrix, "covariance", 10, chunk, None, show_progress=False) for chunk in (64, 400))
    np.testing.assert_allclose(a["components"], b["components"], atol=1e-12)
    np.testing.assert_allclose(a["variance"], b["variance"], rtol=1e-12)


def test_default_chunk_does_not_depend_on_free_memory():
    """With the memory to spare, the chunk (which IncrementalPCA's result
    depends on) is the same on every machine."""
    chunks = {_pca_chunk_size(2_000, 50, memory_limit_gb=gb, method="incremental")[0] for gb in (4, 16, 256)}
    assert chunks == {4096}


def test_incremental_pca_matches_sklearn_on_leading_components(backed):
    """Incremental PCA is approximate; the well-separated leading components agree."""
    adata, X = backed
    fit = _fit_pca(adata.X, "incremental", 10, 64, None, show_progress=False)
    ref = PCA(10, svd_solver="full").fit(X)
    assert fit["components"].shape == (10, X.shape[1])
    np.testing.assert_allclose(fit["variance_ratio"], ref.explained_variance_ratio_, atol=0.01)
    scores = (X - fit["mean"]) @ fit["components"].T
    _assert_same_up_to_sign(scores[:, :5], ref.transform(X)[:, :5], rtol=0.01, atol=0.05)


def test_incremental_pca_fits_a_short_final_chunk():
    """400 cells in chunks of 98 leave 8 < n_comps: they join the chunk before
    rather than being left out of the fit."""
    X = _counts()
    fit = _fit_pca(sparse.csr_matrix(X), "incremental", 10, 98, None, show_progress=False)
    np.testing.assert_allclose(fit["mean"], X.astype(np.float64).mean(axis=0), rtol=1e-12)


def _traced_peak(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_covariance_fit_holds_one_gene_by_gene_matrix():
    n_vars, chunk = 3_000, 512
    matrix = sparse.random(2_048, n_vars, density=0.05, format="csr", random_state=0, dtype=np.float32)
    peak =_traced_peak(lambda: _fit_pca(matrix, "covariance", 20, chunk, None, show_progress=False))
    assert peak < 1.25 * 8 * n_vars**2 + 16 * chunk * n_vars


def test_backed_pca_memory_does_not_grow_with_the_cells(tmp_path):
    """Scores are written to the file a chunk at a time: 4x the cells adds far
    less than the n x k float32 score matrix."""
    n_comps, peaks = 20, {}
    for n_obs in (20_000, 80_000):
        X = sparse.random(n_obs, 40, density=0.3, format="csr", random_state=0, dtype=np.float32)
        path = tmp_path / f"cells_{n_obs}.h5ad"
        ad.AnnData(X).write(path)
        backed = cx.AnnData(path)
        peaks[n_obs] = _traced_peak(
            lambda: cx.pp.pca(backed, n_comps=n_comps, chunk_size=2_048, show_progress=False)
        )
        with h5py.File(path, "r") as f:
            assert f["obsm/X_pca"].shape == (n_obs, n_comps)
            assert list(f["obsm"]) == ["X_pca"]
        assert not _pca_scratch_path(path).exists()
    assert peaks[80_000] - peaks[20_000] < 0.5 * 60_000 * n_comps * 4


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


@pytest.mark.parametrize("order", ["C", "F"])
def test_pca_leaves_an_in_memory_matrix_untouched(order):
    """Rows of a dense float64 X can reach the projection as views (a one-row
    tail chunk, or a Fortran-ordered X in one chunk); centring must not write
    through them."""
    X = np.asarray(np.random.default_rng(0).random((257, 50)), order=order)
    adata = ad.AnnData(X.copy(order=order))
    cx.pp.pca(adata, n_comps=5, chunk_size=256 if order == "C" else 512, use_highly_variable=False, show_progress=False)
    np.testing.assert_array_equal(adata.X, X)


def test_pca_on_a_linked_file_with_a_read_only_source(tmp_path):
    """A standardized file's X is an external link: PCA writes to the linked
    file and only reads the source, which may be read-only."""
    from crispyx.data import _write_linked_h5ad

    source, linked = tmp_path / "source.h5ad", tmp_path / "linked.h5ad"
    adata = ad.AnnData(sparse.csr_matrix(_counts()))
    adata.write(source)
    _write_linked_h5ad(source, linked, obs=adata.obs, var=adata.var, uns={})
    source.chmod(0o444)
    try:
        if os.access(source, os.W_OK):
            pytest.skip("running as a user who can write a read-only file")
        cx.pp.pca(cx.AnnData(linked), n_comps=5, show_progress=False)
    finally:
        source.chmod(0o644)
    assert ad.read_h5ad(linked).obsm["X_pca"].shape == (400, 5)


def test_interrupted_pca_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    path = _write(tmp_path / "pca.h5ad", _counts())
    cx.pp.pca(cx.AnnData(path), n_comps=5, show_progress=False)
    before = ad.read_h5ad(path).obsm["X_pca"]

    def fail(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("crispyx.dimred._project", fail)
    with pytest.raises(KeyboardInterrupt):
        cx.pp.pca(cx.AnnData(path), n_comps=5, show_progress=False)
    with h5py.File(path, "r") as f:
        assert list(f["obsm"]) == ["X_pca"]
    assert not _pca_scratch_path(path).exists()
    np.testing.assert_array_equal(ad.read_h5ad(path).obsm["X_pca"], before)


def test_pca_on_a_path_writes_to_the_file(tmp_path):
    path = _write(tmp_path / "pca.h5ad", _counts())
    assert cx.pp.pca(str(path), n_comps=8, show_progress=False) is None
    assert ad.read_h5ad(path).obsm["X_pca"].shape == (400, 8)


def test_pca_copy_from_a_path_returns_the_same_embedding(tmp_path):
    path = _write(tmp_path / "pca.h5ad", _counts())
    copied = cx.pp.pca(path, n_comps=8, copy=True, show_progress=False)
    assert isinstance(copied, ad.AnnData)
    assert "X_pca" not in ad.read_h5ad(path).obsm  # copy leaves the file alone
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
    assert umap(adata, n_components=n_components) is None  # in place
    assert adata.obsm["X_umap"].shape == (adata.n_obs, n_components)
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
    copied = cx.pp.pca(backed, n_comps=15, copy=True, show_progress=False)
    assert isinstance(copied, ad.AnnData)
    copied = cx.pp.neighbors(backed, n_neighbors=5, method="sklearn", copy=True, show_progress=False)
    assert isinstance(copied, ad.AnnData) and "distances" in copied.obsp
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
        value = {"variance_ratio": np.array([0.5, 0.3, 0.1]), "n_comps": 10, "method": "covariance"}
        write_uns_dict_to_h5ad(path, "pca", value)
        stored = ad.read_h5ad(path).uns["pca"]
        np.testing.assert_array_equal(stored["variance_ratio"], value["variance_ratio"])
        assert stored["n_comps"] == 10 and stored["method"] == "covariance"


def test_pca_fails_before_the_passes_when_the_file_is_open_elsewhere(tmp_path, monkeypatch):
    """The results could not be written while another handle holds the file:
    say so before computing anything, not after."""
    path = _write(tmp_path / "pca.h5ad", _counts())

    def never(*args, **kwargs):
        raise AssertionError("fit ran although the results could not be written")

    monkeypatch.setattr("crispyx.dimred._fit_pca", never)
    with h5py.File(path, "r"):
        with pytest.raises(OSError, match="open elsewhere"):
            cx.pp.pca(cx.AnnData(path), n_comps=5, show_progress=False)


def test_pca_without_any_highly_variable_gene_raises():
    adata = ad.AnnData(sparse.csr_matrix(_counts()))
    adata.var["highly_variable"] = False
    with pytest.raises(ValueError, match="marks no genes"):
        cx.pp.pca(adata, n_comps=5, show_progress=False)


def test_pca_records_the_chunk_the_fit_used():
    """IncrementalPCA needs at least n_comps rows per chunk."""
    adata = ad.AnnData(sparse.csr_matrix(_counts()))
    cx.pp.pca(adata, n_comps=20, method="incremental", chunk_size=8, use_highly_variable=False, show_progress=False)
    assert adata.uns["pca"]["chunk_size"] == 20
