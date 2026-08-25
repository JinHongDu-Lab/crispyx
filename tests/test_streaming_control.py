"""Tests for streaming precompute_control_statistics.

Verifies that the streaming (chunk-based) path produces results matching
the original dense path, and that the integration into nb_glm_test works
correctly when the streaming threshold is triggered.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crispyx.glm import (
    precompute_control_statistics,
    precompute_control_statistics_streaming,
    ControlStatisticsCache,
)
from crispyx.de import nb_glm_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_h5ad(tmp_path, n_control=100, n_pert=50, n_genes=30, seed=42):
    """Create a synthetic h5ad file with control + perturbation cells."""
    rng = np.random.default_rng(seed)
    n_cells = n_control + n_pert
    labels = np.array(["control"] * n_control + ["g1"] * n_pert)

    # Generate NB counts with known parameters
    mu_vals = rng.gamma(3, 2, size=n_genes)
    alpha = 0.5
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    for j in range(n_genes):
        r = 1.0 / max(alpha, 0.01)
        p = r / (r + mu_vals[j])
        counts[:, j] = rng.negative_binomial(max(int(r), 1), min(p, 0.999), size=n_cells)

    # Add realistic sparsity (20% zeros)
    mask = rng.random((n_cells, n_genes)) < 0.2
    counts[mask] = 0

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(
        sp.csr_matrix(counts),  # Store as sparse
        obs=obs,
        var=var,
    )
    path = tmp_path / "test_streaming.h5ad"
    adata.write(path)
    return path, counts, labels


# ---------------------------------------------------------------------------
# Test 1: Streaming matches dense — intercept, dispersion, frozen stats
# ---------------------------------------------------------------------------

def test_streaming_matches_dense_control_stats(tmp_path):
    """Streaming and dense precompute_control_statistics produce matching results."""

    path, counts, labels = _make_synthetic_h5ad(tmp_path, n_control=80, n_pert=40, n_genes=25)

    control_mask = labels == "control"
    control_matrix = sp.csr_matrix(counts[control_mask])
    size_factors = counts.sum(axis=1) / np.median(counts.sum(axis=1))
    size_factors = np.maximum(size_factors, 0.1)
    offset = np.log(size_factors)
    control_offset = offset[control_mask]

    # Dense path (original)
    cache_dense = precompute_control_statistics(
        control_matrix=control_matrix,
        control_offset=control_offset,
        max_iter=10,
        tol=1e-6,
        min_mu=0.5,
        global_size_factors=size_factors,
        freeze_control=True,
    )

    # Streaming path (new)
    cache_stream = precompute_control_statistics_streaming(
        path=path,
        control_mask=control_mask,
        control_offset=control_offset,
        max_iter=10,
        tol=1e-6,
        min_mu=0.5,
        global_size_factors=size_factors,
        freeze_control=True,
        chunk_size=16,  # Small chunks to test chunking
    )

    # Intercept should match closely
    np.testing.assert_allclose(
        cache_stream.beta_intercept,
        cache_dense.beta_intercept,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Intercept mismatch between streaming and dense",
    )

    # Dispersion
    np.testing.assert_allclose(
        cache_stream.control_dispersion,
        cache_dense.control_dispersion,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Dispersion mismatch between streaming and dense",
    )

    # Frozen stats
    np.testing.assert_allclose(
        cache_stream.frozen_control_W_sum,
        cache_dense.frozen_control_W_sum,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Frozen W_sum mismatch",
    )
    np.testing.assert_allclose(
        cache_stream.frozen_control_Wz_sum,
        cache_dense.frozen_control_Wz_sum,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Frozen Wz_sum mismatch",
    )
    np.testing.assert_allclose(
        cache_stream.frozen_control_Y_sum,
        cache_dense.frozen_control_Y_sum,
        rtol=1e-10,
        err_msg="Frozen Y_sum mismatch",
    )
    np.testing.assert_allclose(
        cache_stream.frozen_control_mu_sum,
        cache_dense.frozen_control_mu_sum,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Frozen mu_sum mismatch",
    )
    np.testing.assert_allclose(
        cache_stream.frozen_control_resid_sq_sum,
        cache_dense.frozen_control_resid_sq_sum,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Frozen resid_sq_sum mismatch",
    )

    # Expression counts
    np.testing.assert_array_equal(
        cache_stream.control_expr_counts,
        cache_dense.control_expr_counts,
        err_msg="Expression counts mismatch",
    )

    # Mean expression
    np.testing.assert_allclose(
        cache_stream.control_mean_expr,
        cache_dense.control_mean_expr,
        rtol=1e-5,
        err_msg="Mean expression mismatch",
    )

    # pts_rest
    np.testing.assert_allclose(
        cache_stream.pts_rest,
        cache_dense.pts_rest,
        rtol=1e-5,
        err_msg="pts_rest mismatch",
    )

    # Both should report frozen mode
    assert cache_stream.use_frozen_control is True
    assert cache_dense.use_frozen_control is True
    assert cache_stream.control_matrix is None
    assert cache_dense.control_matrix is None


# ---------------------------------------------------------------------------
# Test 2: Streaming requires freeze_control=True
# ---------------------------------------------------------------------------

def test_streaming_requires_freeze_control(tmp_path):
    """Streaming path raises ValueError when freeze_control=False."""
    path, counts, labels = _make_synthetic_h5ad(tmp_path, n_control=30, n_pert=20, n_genes=10)
    control_mask = labels == "control"
    size_factors = np.ones(len(labels))
    control_offset = np.zeros(int(control_mask.sum()))

    with pytest.raises(ValueError, match="freeze_control=True"):
        precompute_control_statistics_streaming(
            path=path,
            control_mask=control_mask,
            control_offset=control_offset,
            freeze_control=False,
        )


# ---------------------------------------------------------------------------
# Test 3: Various chunk sizes produce consistent results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_size", [1, 7, 16, 64, 200])
def test_streaming_chunk_sizes(tmp_path, chunk_size):
    """Different chunk sizes yield the same streaming results."""
    path, counts, labels = _make_synthetic_h5ad(tmp_path, n_control=50, n_pert=30, n_genes=15)
    control_mask = labels == "control"
    control_offset = np.zeros(int(control_mask.sum()))

    cache = precompute_control_statistics_streaming(
        path=path,
        control_mask=control_mask,
        control_offset=control_offset,
        chunk_size=chunk_size,
    )

    # Reference: chunk_size = all control cells at once
    cache_ref = precompute_control_statistics_streaming(
        path=path,
        control_mask=control_mask,
        control_offset=control_offset,
        chunk_size=10000,  # Larger than n_control
    )

    np.testing.assert_allclose(
        cache.beta_intercept, cache_ref.beta_intercept, rtol=1e-10,
    )
    np.testing.assert_allclose(
        cache.frozen_control_W_sum, cache_ref.frozen_control_W_sum, rtol=1e-10,
    )
    np.testing.assert_allclose(
        cache.frozen_control_Y_sum, cache_ref.frozen_control_Y_sum, rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Test 4: Dense storage h5ad files work with streaming
# ---------------------------------------------------------------------------

def test_streaming_dense_h5ad(tmp_path):
    """Streaming works with h5ad files that store X as a dense array."""
    rng = np.random.default_rng(99)
    n_control, n_pert, n_genes = 40, 20, 12
    n_cells = n_control + n_pert
    labels = np.array(["control"] * n_control + ["g1"] * n_pert)
    counts = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float64)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(counts, obs=obs, var=var)  # Dense X
    path = tmp_path / "dense_test.h5ad"
    adata.write(path)

    control_mask = labels == "control"
    control_offset = np.zeros(n_control)

    cache = precompute_control_statistics_streaming(
        path=path,
        control_mask=control_mask,
        control_offset=control_offset,
        chunk_size=10,
    )

    assert cache.use_frozen_control is True
    assert np.all(np.isfinite(cache.beta_intercept))
    assert np.all(np.isfinite(cache.frozen_control_W_sum))
    assert cache.control_n == n_control


# ---------------------------------------------------------------------------
# Test 5: End-to-end nb_glm_test with forced streaming control path
# ---------------------------------------------------------------------------

def test_nb_glm_test_streaming_control_e2e(tmp_path):
    """nb_glm_test produces valid results when streaming control path is forced.

    We use a small dataset but patch the threshold to force the streaming code path.
    """
    rng = np.random.default_rng(77)
    n_control, n_pert, n_genes = 60, 30, 8
    n_cells = n_control + n_pert
    labels = np.array(["control"] * n_control + ["g1"] * n_pert)

    # Synthetic NB counts with known effect on first gene
    alpha = 0.5
    mu_control = np.array([10, 5, 8, 3, 12, 6, 4, 9], dtype=np.float64)
    mu_pert = mu_control.copy()
    mu_pert[0] *= 3.0  # Strong up-regulation of gene0

    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    for j in range(n_genes):
        r = 1.0 / alpha
        p_ctrl = r / (r + mu_control[j])
        p_pert = r / (r + mu_pert[j])
        counts[:n_control, j] = rng.negative_binomial(max(int(r), 1), min(p_ctrl, 0.999), size=n_control)
        counts[n_control:, j] = rng.negative_binomial(max(int(r), 1), min(p_pert, 0.999), size=n_pert)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    path = tmp_path / "e2e_streaming.h5ad"
    adata.write(path)

    # Run nb_glm_test with a normal path first (reference)
    result_ref = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        max_iter=25,
        min_cells_expressed=0,
        n_jobs=1,
        freeze_control=False,
    )

    # Now run with forced streaming by using max_dense_fraction=0
    # This forces use_streaming_control=True (control_matrix_gb * 4 > 0)
    result_stream = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        max_iter=25,
        min_cells_expressed=0,
        n_jobs=1,
        max_dense_fraction=0.0,  # Force streaming path
    )

    # Both should produce valid results
    ref_g1 = result_ref["g1"]
    stream_g1 = result_stream["g1"]

    assert ref_g1.pvalue.shape == stream_g1.pvalue.shape

    # Effect sizes should be correlated (not identical due to different
    # code paths, but same direction and similar magnitude)
    valid = np.isfinite(ref_g1.effect_size) & np.isfinite(stream_g1.effect_size)
    assert valid.sum() >= 4, f"Too few valid genes: {valid.sum()}"

    from scipy.stats import pearsonr
    r, _ = pearsonr(ref_g1.effect_size[valid], stream_g1.effect_size[valid])
    assert r > 0.8, f"Effect size correlation too low: {r:.3f}"

    # Both should detect gene0 as top hit (strongest effect)
    ref_top = np.nanargmin(ref_g1.pvalue)
    stream_top = np.nanargmin(stream_g1.pvalue)
    assert ref_top == 0, f"Reference top hit should be gene0, got gene{ref_top}"
    assert stream_top == 0, f"Streaming top hit should be gene0, got gene{stream_top}"


# ---------------------------------------------------------------------------
# Test 6: Streaming + frozen control end-to-end comparison
# ---------------------------------------------------------------------------

def test_streaming_vs_dense_frozen_e2e(tmp_path):
    """Compare nb_glm_test results: dense+freeze_control vs streaming."""
    rng = np.random.default_rng(111)
    n_control, n_pert, n_genes = 50, 25, 10
    n_cells = n_control + n_pert
    labels = np.array(["control"] * n_control + ["g1"] * n_pert)

    counts = rng.poisson(8, size=(n_cells, n_genes)).astype(np.float64)
    # Add effect to gene 0
    counts[n_control:, 0] = rng.poisson(24, size=n_pert)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    path = tmp_path / "compare_paths.h5ad"
    adata.write(path)

    # Dense freeze_control=True path
    result_dense = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        max_iter=25,
        min_cells_expressed=0,
        n_jobs=1,
        freeze_control=True,
    )

    # Streaming path (forced via max_dense_fraction=0)
    result_stream = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        max_iter=25,
        min_cells_expressed=0,
        n_jobs=1,
        max_dense_fraction=0.0,
    )

    dense_g1 = result_dense["g1"]
    stream_g1 = result_stream["g1"]

    # Since both use freeze_control=True with the same algorithm,
    # they should match very closely
    valid = np.isfinite(dense_g1.effect_size) & np.isfinite(stream_g1.effect_size)
    np.testing.assert_allclose(
        stream_g1.effect_size[valid],
        dense_g1.effect_size[valid],
        rtol=0.05,
        atol=0.05,
        err_msg="Dense vs streaming effect sizes differ too much",
    )
