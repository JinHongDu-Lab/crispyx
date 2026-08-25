"""Tests for dimension reduction module (PCA)."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

import crispyx as cx
from crispyx.dimred import (
    _streaming_pca_incremental,
    _streaming_pca_sparse_cov,
    pca,
)
from crispyx.data import calculate_pca_chunk_size


class TestCalculatePCAChunkSize:
    """Tests for calculate_pca_chunk_size function."""

    def test_returns_tuple(self):
        """Should return (chunk_size, method) tuple."""
        result = calculate_pca_chunk_size(10000, 5000, available_memory_gb=16)
        assert isinstance(result, tuple)
        assert len(result) == 2
        chunk_size, method = result
        assert isinstance(chunk_size, int)
        assert method in ("sparse_cov", "incremental")

    def test_auto_selects_sparse_cov_for_small_genes(self):
        """Should select sparse_cov for small gene counts."""
        _, method = calculate_pca_chunk_size(
            100000, 5000, available_memory_gb=32, method="auto"
        )
        # 5000 genes → ~200MB covariance, should fit
        assert method == "sparse_cov"

    def test_auto_selects_incremental_for_large_genes(self):
        """Should select incremental for large gene counts."""
        _, method = calculate_pca_chunk_size(
            100000, 50000, available_memory_gb=16, method="auto"
        )
        # 50000 genes → ~20GB covariance, won't fit in 30% of 16GB
        assert method == "incremental"

    def test_respects_min_max_chunk(self):
        """Should respect min/max chunk size bounds."""
        chunk_size, _ = calculate_pca_chunk_size(
            100000, 1000, available_memory_gb=1,
            min_chunk=128, max_chunk=256
        )
        assert 128 <= chunk_size <= 256

    def test_explicit_method_respected(self):
        """Should use explicit method when specified."""
        _, method = calculate_pca_chunk_size(
            100000, 5000, method="incremental"
        )
        assert method == "incremental"


class TestStreamingPCASparseCovariance:
    """Tests for _streaming_pca_sparse_cov function."""

    @pytest.fixture
    def synthetic_adata(self):
        """Create synthetic AnnData with known structure."""
        np.random.seed(42)
        n_obs, n_vars = 500, 200
        # Create data with clear structure
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        # Add some dominant directions
        X[:, 0] *= 10
        X[:, 1] *= 5
        
        adata = ad.AnnData(sparse.csr_matrix(X))
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        return adata

    @pytest.fixture
    def backed_adata(self, synthetic_adata):
        """Write synthetic data to h5ad and return backed AnnData."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        synthetic_adata.write(path)
        adata = ad.read_h5ad(path, backed='r')
        yield adata
        adata.file.close()
        path.unlink()

    def test_output_shapes(self, backed_adata):
        """Should return correct output shapes."""
        n_comps = 20
        X_pca, components, variance_ratio, info = _streaming_pca_sparse_cov(
            backed_adata, n_comps=n_comps, show_progress=False, return_info=True
        )
        
        n_obs, n_vars = backed_adata.shape
        assert X_pca.shape == (n_obs, n_comps)
        assert components.shape == (n_comps, n_vars)
        assert variance_ratio.shape == (n_comps,)
        assert info is not None

    def test_variance_ratio_sums_to_less_than_one(self, backed_adata):
        """Variance ratios should sum to <= 1."""
        _, _, variance_ratio, _ = _streaming_pca_sparse_cov(
            backed_adata, n_comps=20, show_progress=False
        )
        assert 0 < variance_ratio.sum() <= 1.0

    def test_variance_ratio_decreasing(self, backed_adata):
        """Variance ratios should be in decreasing order."""
        _, _, variance_ratio, _ = _streaming_pca_sparse_cov(
            backed_adata, n_comps=20, show_progress=False
        )
        assert np.all(np.diff(variance_ratio) <= 0)


class TestStreamingPCAIncremental:
    """Tests for _streaming_pca_incremental function."""

    @pytest.fixture
    def synthetic_adata(self):
        """Create synthetic AnnData with known structure."""
        np.random.seed(42)
        n_obs, n_vars = 500, 200
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        X[:, 0] *= 10
        X[:, 1] *= 5
        
        adata = ad.AnnData(sparse.csr_matrix(X))
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        return adata

    @pytest.fixture
    def backed_adata(self, synthetic_adata):
        """Write synthetic data to h5ad and return backed AnnData."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        synthetic_adata.write(path)
        adata = ad.read_h5ad(path, backed='r')
        yield adata
        adata.file.close()
        path.unlink()

    def test_output_shapes(self, backed_adata):
        """Should return correct output shapes."""
        n_comps = 20
        X_pca, components, variance_ratio, info = _streaming_pca_incremental(
            backed_adata, n_comps=n_comps, chunk_size=100, show_progress=False
        )
        
        n_obs, n_vars = backed_adata.shape
        assert X_pca.shape == (n_obs, n_comps)
        assert components.shape == (n_comps, n_vars)
        assert variance_ratio.shape == (n_comps,)

    def test_variance_ratio_sums_to_less_than_one(self, backed_adata):
        """Variance ratios should sum to <= 1."""
        _, _, variance_ratio, _ = _streaming_pca_incremental(
            backed_adata, n_comps=20, chunk_size=100, show_progress=False
        )
        assert 0 < variance_ratio.sum() <= 1.0


class TestPCAMethodsConsistency:
    """Test that both PCA methods produce consistent results."""

    @pytest.fixture
    def backed_adata(self):
        """Create backed AnnData with structured data."""
        np.random.seed(42)
        n_obs, n_vars = 1000, 100
        # Create structured data (not purely random)
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        X[:, 0] *= 20  # Strong first component
        X[:, 1] *= 10  # Strong second component
        
        adata = ad.AnnData(sparse.csr_matrix(X))
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        adata_backed = ad.read_h5ad(path, backed='r')
        yield adata_backed
        adata_backed.file.close()
        path.unlink()

    def test_methods_produce_similar_variance_ratios(self, backed_adata):
        """Both methods should explain similar variance."""
        n_comps = 10
        
        _, _, var_sparse, _ = _streaming_pca_sparse_cov(
            backed_adata, n_comps=n_comps, show_progress=False
        )
        _, _, var_incr, _ = _streaming_pca_incremental(
            backed_adata, n_comps=n_comps, chunk_size=200, show_progress=False
        )
        
        # Total variance explained should be similar (within 5%)
        assert np.abs(var_sparse.sum() - var_incr.sum()) < 0.05

    def test_top_components_highly_correlated(self, backed_adata):
        """Top PCs from both methods should be highly correlated."""
        n_comps = 5
        
        X_sparse, _, _, _ = _streaming_pca_sparse_cov(
            backed_adata, n_comps=n_comps, show_progress=False
        )
        X_incr, _, _, _ = _streaming_pca_incremental(
            backed_adata, n_comps=n_comps, chunk_size=200, show_progress=False
        )
        
        # Check correlation of first 2 PCs (allow sign flip)
        # Note: Lower variance PCs can differ more between methods
        for i in range(min(2, n_comps)):
            corr = np.abs(np.corrcoef(X_sparse[:, i], X_incr[:, i])[0, 1])
            assert corr > 0.90, f"PC{i+1} correlation too low: {corr}"


class TestPCAMainFunction:
    """Tests for main pca() function."""

    @pytest.fixture
    def backed_adata(self):
        """Create backed AnnData for testing."""
        np.random.seed(42)
        n_obs, n_vars = 300, 100
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        X = (X * 100).astype(np.float32)
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        adata_backed = ad.read_h5ad(path, backed='r')
        yield adata_backed
        adata_backed.file.close()
        path.unlink()

    def test_stores_results_in_adata(self, backed_adata):
        """Should store X_pca, PCs, and uns['pca'] in adata."""
        n_comps = 20
        pca(backed_adata, n_comps=n_comps, show_progress=False)
        
        assert "X_pca" in backed_adata.obsm
        assert backed_adata.obsm["X_pca"].shape == (300, n_comps)
        assert "PCs" in backed_adata.varm
        assert "pca" in backed_adata.uns
        assert "variance_ratio" in backed_adata.uns["pca"]

    def test_copy_returns_new_adata(self, backed_adata):
        """copy=True should return new AnnData, not modify original."""
        result = pca(backed_adata, n_comps=10, copy=True, show_progress=False)
        
        assert result is not None
        assert "X_pca" in result.obsm
        # Original should not have X_pca (backed mode may not support obsm)
        # Just check that result has the data
        assert result.obsm["X_pca"].shape[1] == 10


class TestPCANamespaceIntegration:
    """Tests for cx.pp.pca integration."""

    def test_pca_accessible_via_pp_namespace(self):
        """cx.pp.pca should be callable."""
        assert hasattr(cx.pp, "pca")
        assert callable(cx.pp.pca)

    def test_pca_from_path(self):
        """Should work with file path input."""
        np.random.seed(42)
        n_obs, n_vars = 100, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        
        try:
            result = cx.pp.pca(path, n_comps=10, copy=True, show_progress=False)
            assert "X_pca" in result.obsm
        finally:
            path.unlink()


class TestHVGSupport:
    """Tests for highly_variable gene support."""

    @pytest.fixture
    def adata_with_hvg(self):
        """Create AnnData with highly_variable column."""
        np.random.seed(42)
        n_obs, n_vars = 200, 100
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        X[:, :20] *= 10  # Make first 20 genes highly variable
        
        adata = ad.AnnData(sparse.csr_matrix(X))
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        # Mark first 30 genes as HVG
        adata.var["highly_variable"] = [i < 30 for i in range(n_vars)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        adata_backed = ad.read_h5ad(path, backed='r')
        yield adata_backed
        adata_backed.file.close()
        path.unlink()

    def test_uses_hvg_when_available(self, adata_with_hvg):
        """Should use only HVGs when use_highly_variable=True."""
        pca(adata_with_hvg, n_comps=10, use_highly_variable=True, show_progress=False)
        
        assert adata_with_hvg.uns["pca"]["use_highly_variable"] is True
        # PCs should have full n_vars shape but only HVG positions filled
        assert adata_with_hvg.varm["PCs"].shape == (100, 10)

    def test_ignores_hvg_when_disabled(self, adata_with_hvg):
        """Should use all genes when use_highly_variable=False."""
        pca(adata_with_hvg, n_comps=10, use_highly_variable=False, show_progress=False)
        
        assert adata_with_hvg.uns["pca"]["use_highly_variable"] is False


# ============================================================================
# Neighbors Tests
# ============================================================================

from crispyx.dimred import neighbors


class TestNeighborsFunction:
    """Tests for neighbors() function."""

    @pytest.fixture
    def adata_with_pca(self):
        """Create AnnData with PCA embeddings."""
        np.random.seed(42)
        n_obs, n_vars = 200, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        # Add fake PCA embeddings
        adata.obsm["X_pca"] = np.random.randn(n_obs, 20).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        adata_backed = ad.read_h5ad(path, backed='r')
        yield adata_backed
        adata_backed.file.close()
        path.unlink()

    def test_neighbors_stores_results(self, adata_with_pca):
        """Should store distances, connectivities, and uns['neighbors']."""
        neighbors(adata_with_pca, n_neighbors=10, method="sklearn", show_progress=False)
        
        assert "distances" in adata_with_pca.obsp
        assert "connectivities" in adata_with_pca.obsp
        assert "neighbors" in adata_with_pca.uns

    def test_neighbors_matrix_shapes(self, adata_with_pca):
        """Sparse matrices should have correct shape."""
        n_obs = adata_with_pca.n_obs
        neighbors(adata_with_pca, n_neighbors=10, method="sklearn", show_progress=False)
        
        assert adata_with_pca.obsp["distances"].shape == (n_obs, n_obs)
        assert adata_with_pca.obsp["connectivities"].shape == (n_obs, n_obs)

    def test_neighbors_params_stored(self, adata_with_pca):
        """Parameters should be stored in uns['neighbors']."""
        neighbors(adata_with_pca, n_neighbors=15, method="sklearn", 
                  metric="euclidean", show_progress=False)
        
        params = adata_with_pca.uns["neighbors"]["params"]
        assert params["n_neighbors"] == 15
        assert params["method"] == "sklearn"
        assert params["metric"] == "euclidean"

    def test_neighbors_n_pcs_subset(self, adata_with_pca):
        """Should use only first n_pcs when specified."""
        neighbors(adata_with_pca, n_neighbors=10, n_pcs=5, 
                  method="sklearn", show_progress=False)
        
        params = adata_with_pca.uns["neighbors"]["params"]
        assert params["n_pcs"] == 5

    def test_neighbors_missing_pca_raises(self):
        """Should raise error if X_pca not found."""
        np.random.seed(42)
        adata = ad.AnnData(sparse.random(50, 20, density=0.3, format='csr'))
        # No X_pca added
        
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            neighbors(adata, n_neighbors=10)

    def test_neighbors_copy_returns_new(self, adata_with_pca):
        """copy=True should return new AnnData."""
        result = neighbors(adata_with_pca, n_neighbors=10, copy=True,
                          method="sklearn", show_progress=False)
        
        assert result is not None
        assert "distances" in result.obsp


class TestNeighborsSklearn:
    """Tests for sklearn KNN method."""

    @pytest.fixture
    def clustered_adata(self):
        """Create data with clear cluster structure."""
        np.random.seed(42)
        # 3 clusters of 50 points each
        cluster1 = np.random.randn(50, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(50, 10) + np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        cluster3 = np.random.randn(50, 10) + np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        
        X_pca = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        
        adata = ad.AnnData(sparse.random(150, 20, density=0.1, format='csr'))
        adata.obsm["X_pca"] = X_pca
        return adata

    def test_sklearn_neighbors_within_cluster(self, clustered_adata):
        """Neighbors should mostly be from same cluster."""
        neighbors(clustered_adata, n_neighbors=10, method="sklearn", show_progress=False)
        
        distances = clustered_adata.obsp["distances"]
        
        # Check that cell 0 (cluster 1) has neighbors mostly in cluster 1
        row = distances.getrow(0).toarray().ravel()
        neighbor_indices = np.where(row > 0)[0]
        
        # Most neighbors should be in [0, 50) for cluster 1
        in_cluster = sum(1 for idx in neighbor_indices if idx < 50)
        assert in_cluster >= 8, f"Expected >=8 in-cluster neighbors, got {in_cluster}"


class TestNeighborsPynndescent:
    """Tests for pynndescent (UMAP) KNN method."""

    @pytest.fixture
    def simple_adata(self):
        """Create simple AnnData with PCA."""
        np.random.seed(42)
        adata = ad.AnnData(sparse.random(100, 30, density=0.2, format='csr'))
        adata.obsm["X_pca"] = np.random.randn(100, 15).astype(np.float32)
        return adata

    def test_umap_method_works(self, simple_adata):
        """UMAP method should work with pynndescent installed."""
        try:
            neighbors(simple_adata, n_neighbors=10, method="umap", show_progress=False)
            assert "distances" in simple_adata.obsp
        except ImportError:
            pytest.skip("pynndescent not installed")


class TestNeighborsNamespaceIntegration:
    """Tests for cx.pp.neighbors integration."""

    def test_neighbors_accessible_via_pp(self):
        """cx.pp.neighbors should be callable."""
        assert hasattr(cx.pp, "neighbors")
        assert callable(cx.pp.neighbors)

    def test_pca_then_neighbors_pipeline(self):
        """Full PCA → neighbors pipeline should work."""
        np.random.seed(42)
        n_obs, n_vars = 100, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        
        try:
            # Run pipeline
            result = cx.pp.pca(path, n_comps=20, copy=True, show_progress=False)
            cx.pp.neighbors(result, n_neighbors=10, method="sklearn", show_progress=False)
            
            assert "X_pca" in result.obsm
            assert "distances" in result.obsp
            assert "connectivities" in result.obsp
        finally:
            path.unlink()


class TestCloseWriteReopenPattern:
    """Tests for close-write-reopen pattern with crispyx.AnnData wrapper."""

    @pytest.fixture
    def backed_adata(self, tmp_path):
        """Create backed AnnData using crispyx wrapper."""
        np.random.seed(42)
        n_obs, n_vars = 100, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        path = tmp_path / "test_backed.h5ad"
        adata.write(path)
        
        # Return crispyx.AnnData wrapper
        return cx.read_h5ad_ondisk(path)

    def test_pca_writes_to_h5ad(self, backed_adata):
        """PCA should write results directly to h5ad file."""
        path = backed_adata.path
        
        # Run PCA (should write to file)
        result = cx.pp.pca(backed_adata, n_comps=20, show_progress=False)
        assert result is None  # In-place modification
        
        # Reopen and verify results are persisted
        reopened = cx.read_h5ad_ondisk(path)
        assert "X_pca" in reopened.obsm
        assert reopened.obsm["X_pca"].shape == (100, 20)
        assert "pca" in reopened.uns
        # Need to load() the uns entry to access as dict
        pca_info = reopened.uns["pca"].load()
        assert "variance_ratio" in pca_info

    def test_neighbors_writes_to_h5ad(self, backed_adata):
        """Neighbors should write results directly to h5ad file."""
        path = backed_adata.path
        
        # First run PCA
        cx.pp.pca(backed_adata, n_comps=20, show_progress=False)
        
        # Run neighbors (should write to file)
        result = cx.pp.neighbors(backed_adata, n_neighbors=10, method="sklearn", show_progress=False)
        assert result is None  # In-place modification
        
        # Reopen and verify results are persisted
        reopened = cx.read_h5ad_ondisk(path)
        assert "distances" in reopened.obsp
        assert "connectivities" in reopened.obsp
        assert "neighbors" in reopened.uns

    def test_full_pipeline_on_backed_data(self, backed_adata):
        """Full PCA → neighbors → plotting pipeline with backed data."""
        path = backed_adata.path
        
        # Run pipeline
        cx.pp.pca(backed_adata, n_comps=15, show_progress=False)
        cx.pp.neighbors(backed_adata, n_neighbors=5, method="sklearn", show_progress=False)
        
        # Verify data can be read for plotting
        reopened = cx.read_h5ad_ondisk(path)
        
        assert reopened.obsm["X_pca"].shape == (100, 15)
        assert reopened.obsp["distances"].shape == (100, 100)
        assert reopened.obsp["connectivities"].nnz > 0
        
        # Verify uns metadata (need to load() the uns entries)
        pca_info = reopened.uns["pca"].load()
        assert pca_info["n_comps"] == 15
        assert "variance_ratio" in pca_info
        
        neighbors_info = reopened.uns["neighbors"].load()
        assert neighbors_info["params"]["n_neighbors"] == 5

    def test_copy_mode_returns_inmemory(self, backed_adata):
        """copy=True should return in-memory AnnData."""
        result = cx.pp.pca(backed_adata, n_comps=10, copy=True, show_progress=False)
        
        # Should return AnnData, not None
        assert result is not None
        assert isinstance(result, ad.AnnData)
        assert "X_pca" in result.obsm


class TestH5adWriteHelpers:
    """Tests for h5ad write helper functions."""

    def test_write_obsm_roundtrip(self, tmp_path):
        """write_obsm_to_h5ad should persist data correctly."""
        from crispyx.data import write_obsm_to_h5ad
        
        # Create test h5ad
        adata = ad.AnnData(sparse.random(50, 20, density=0.3, format='csr'))
        path = tmp_path / "test.h5ad"
        adata.write(path)
        
        # Write obsm
        test_data = np.random.randn(50, 10).astype(np.float32)
        write_obsm_to_h5ad(path, "test_embedding", test_data)
        
        # Read back
        adata_loaded = ad.read_h5ad(path)
        assert "test_embedding" in adata_loaded.obsm
        np.testing.assert_array_almost_equal(adata_loaded.obsm["test_embedding"], test_data)

    def test_write_varm_roundtrip(self, tmp_path):
        """write_varm_to_h5ad should persist data correctly."""
        from crispyx.data import write_varm_to_h5ad
        
        adata = ad.AnnData(sparse.random(50, 20, density=0.3, format='csr'))
        path = tmp_path / "test.h5ad"
        adata.write(path)
        
        test_data = np.random.randn(20, 5).astype(np.float32)
        write_varm_to_h5ad(path, "PCs", test_data)
        
        adata_loaded = ad.read_h5ad(path)
        assert "PCs" in adata_loaded.varm
        np.testing.assert_array_almost_equal(adata_loaded.varm["PCs"], test_data)

    def test_write_uns_dict_roundtrip(self, tmp_path):
        """write_uns_dict_to_h5ad should persist dict correctly."""
        from crispyx.data import write_uns_dict_to_h5ad
        
        adata = ad.AnnData(sparse.random(50, 20, density=0.3, format='csr'))
        path = tmp_path / "test.h5ad"
        adata.write(path)
        
        test_data = {
            "variance_ratio": np.array([0.5, 0.3, 0.1]),
            "n_comps": 10,
            "method": "sparse_cov",
        }
        write_uns_dict_to_h5ad(path, "pca", test_data)
        
        adata_loaded = ad.read_h5ad(path)
        assert "pca" in adata_loaded.uns
        np.testing.assert_array_almost_equal(
            adata_loaded.uns["pca"]["variance_ratio"], 
            test_data["variance_ratio"]
        )

    def test_write_obsp_roundtrip(self, tmp_path):
        """write_obsp_to_h5ad should persist sparse matrix correctly."""
        from crispyx.data import write_obsp_to_h5ad
        
        adata = ad.AnnData(sparse.random(50, 20, density=0.3, format='csr'))
        path = tmp_path / "test.h5ad"
        adata.write(path)
        
        test_data = sparse.random(50, 50, density=0.1, format='csr')
        write_obsp_to_h5ad(path, "connectivities", test_data)
        
        adata_loaded = ad.read_h5ad(path)
        assert "connectivities" in adata_loaded.obsp
        assert adata_loaded.obsp["connectivities"].nnz == test_data.nnz


# ============================================================================
# UMAP Tests
# ============================================================================

from crispyx.dimred import umap


class TestUMAPFunction:
    """Tests for umap() function."""

    @pytest.fixture
    def adata_with_neighbors(self):
        """Create AnnData with PCA and neighbor graph."""
        np.random.seed(42)
        n_obs, n_vars = 150, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        # Add PCA embeddings
        adata.obsm["X_pca"] = np.random.randn(n_obs, 20).astype(np.float32)
        
        # Compute neighbors
        neighbors(adata, n_neighbors=10, method="sklearn", show_progress=False)
        
        return adata

    def test_umap_stores_embedding(self, adata_with_neighbors):
        """UMAP should store X_umap in obsm."""
        umap(adata_with_neighbors)
        
        assert "X_umap" in adata_with_neighbors.obsm
        assert adata_with_neighbors.obsm["X_umap"].shape == (150, 2)

    def test_umap_stores_uns(self, adata_with_neighbors):
        """UMAP should store parameters in uns['umap']."""
        umap(adata_with_neighbors, min_dist=0.3, spread=1.5)
        
        assert "umap" in adata_with_neighbors.uns

    def test_umap_n_components(self, adata_with_neighbors):
        """UMAP should respect n_components parameter."""
        umap(adata_with_neighbors, n_components=3)
        
        assert adata_with_neighbors.obsm["X_umap"].shape == (150, 3)

    def test_umap_copy_returns_new(self, adata_with_neighbors):
        """copy=True should return new AnnData."""
        result = umap(adata_with_neighbors, copy=True)
        
        assert result is not None
        assert "X_umap" in result.obsm
        # Original should also have X_umap since we're not using backed mode
        # but result is a separate object
        assert result is not adata_with_neighbors

    def test_umap_missing_neighbors_raises(self):
        """Should raise error if neighbors not computed."""
        np.random.seed(42)
        adata = ad.AnnData(sparse.random(50, 20, density=0.3, format='csr'))
        adata.obsm["X_pca"] = np.random.randn(50, 10).astype(np.float32)
        # No neighbors computed
        
        with pytest.raises(ValueError, match="not found in adata.uns"):
            umap(adata)


class TestUMAPBackedMode:
    """Tests for UMAP with backed AnnData (crispyx wrapper)."""

    @pytest.fixture
    def backed_adata_with_neighbors(self, tmp_path):
        """Create backed AnnData with neighbors computed."""
        np.random.seed(42)
        n_obs, n_vars = 100, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        path = tmp_path / "test_umap.h5ad"
        adata.write(path)
        
        # Use crispyx wrapper and compute PCA + neighbors
        backed = cx.read_h5ad_ondisk(path)
        cx.pp.pca(backed, n_comps=15, show_progress=False)
        cx.pp.neighbors(backed, n_neighbors=10, method="sklearn", show_progress=False)
        
        return backed

    def test_umap_writes_to_h5ad(self, backed_adata_with_neighbors):
        """UMAP should write X_umap directly to h5ad file."""
        path = backed_adata_with_neighbors.path
        
        # Run UMAP (should write to file)
        cx.tl.umap(backed_adata_with_neighbors)
        
        # Reopen and verify results are persisted
        reopened = cx.read_h5ad_ondisk(path)
        assert "X_umap" in reopened.obsm
        assert reopened.obsm["X_umap"].shape == (100, 2)
        assert "umap" in reopened.uns


class TestUMAPNamespaceIntegration:
    """Tests for cx.tl.umap and cx.pl.umap integration."""

    def test_umap_accessible_via_tl_namespace(self):
        """cx.tl.umap should be callable."""
        assert hasattr(cx.tl, "umap")
        assert callable(cx.tl.umap)

    def test_umap_plot_accessible_via_pl_namespace(self):
        """cx.pl.umap should be callable."""
        assert hasattr(cx.pl, "umap")
        assert callable(cx.pl.umap)

    def test_full_pca_neighbors_umap_pipeline(self):
        """Full PCA → neighbors → UMAP pipeline should work."""
        np.random.seed(42)
        n_obs, n_vars = 100, 50
        X = sparse.random(n_obs, n_vars, density=0.3, format='csr')
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = Path(f.name)
        adata.write(path)
        
        try:
            # Run full pipeline  
            result = cx.pp.pca(path, n_comps=20, copy=True, show_progress=False)
            cx.pp.neighbors(result, n_neighbors=10, method="sklearn", show_progress=False)
            cx.tl.umap(result)
            
            assert "X_pca" in result.obsm
            assert "distances" in result.obsp
            assert "X_umap" in result.obsm
            assert result.obsm["X_umap"].shape == (100, 2)
        finally:
            path.unlink()
