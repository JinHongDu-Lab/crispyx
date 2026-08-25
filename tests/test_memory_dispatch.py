"""Tests for adaptive memory dispatch and memory_limit_gb parameter."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

from crispyx._memory import (
    _resolve_memory_limit_bytes,
    _should_use_streaming,
)
from crispyx.de import wilcoxon_test, t_test, shrink_lfc


# ---------------------------------------------------------------------------
# _resolve_memory_limit_bytes
# ---------------------------------------------------------------------------

class TestResolveMemoryLimitBytes:
    def test_explicit_value(self):
        result = _resolve_memory_limit_bytes(128)
        assert result == 128 * 1e9

    def test_none_returns_positive(self):
        result = _resolve_memory_limit_bytes(None)
        assert result > 0

    def test_fractional_value(self):
        result = _resolve_memory_limit_bytes(0.5)
        assert result == 0.5 * 1e9


# ---------------------------------------------------------------------------
# _should_use_streaming
# ---------------------------------------------------------------------------

class TestShouldUseStreaming:
    def test_small_dataset_no_streaming(self):
        """Small dataset: 10 groups × 1000 genes should NOT trigger streaming."""
        use, peak, budget, batch = _should_use_streaming(
            10, 1000, memory_limit_gb=128,
        )
        assert use is False
        # batch_size equals n_groups when not streaming
        assert batch == 10

    def test_large_dataset_triggers_streaming(self):
        """~5000 groups × 36000 genes with 4 GB limit should trigger streaming."""
        use, peak, budget, batch = _should_use_streaming(
            5000, 36000, memory_limit_gb=4,
        )
        assert use is True
        assert batch < 5000
        assert batch >= 100

    def test_128gb_large_dataset(self):
        """Real-world scenario: 5000 groups × 36K genes under 128 GB.
        
        This matches the Feng-gwsnf dataset that OOM'd at 47 GB peak
        on HPC with 128 GB allocation.  With peak_multiplier=4.0 the
        dispatch now correctly triggers streaming.
        """
        use, peak, budget, batch = _should_use_streaming(
            5000, 36000, memory_limit_gb=128,
        )
        # bytes_per_group = 36000 * (7*8 + 2*4) = 36000 * 64 = 2,304,000
        # memmap_total = 5000 * 2,304,000 = 11,520,000,000 (~11.5 GB)
        # peak = 46 GB (×4), threshold = 128*1e9*0.30 = 38.4 GB => streaming
        assert use is True
        assert batch <= 5000

    def test_extreme_group_count_triggers_streaming(self):
        """~20000 groups × 36000 genes under 128 GB should trigger streaming."""
        use, peak, budget, batch = _should_use_streaming(
            20000, 36000, memory_limit_gb=128,
        )
        # peak = 20000 * 36000 * 64 * 2 = ~92 GB, threshold = 38.4 GB
        assert use is True
        assert batch < 20000

    def test_custom_threshold_fraction(self):
        """Lower threshold_fraction makes streaming trigger earlier."""
        # Use a medium dataset that does NOT trigger at default 0.30
        # 1000 groups × 20000 genes: memmap=1.28 GB, peak=5.12 GB(×4)
        # threshold@0.30 = 38.4 GB → no stream; threshold@0.03 = 3.84 GB → stream
        use_default, _, _, _ = _should_use_streaming(
            1000, 20000, memory_limit_gb=128, threshold_fraction=0.30,
        )
        use_strict, _, _, _ = _should_use_streaming(
            1000, 20000, memory_limit_gb=128, threshold_fraction=0.03,
        )
        # Strict threshold should trigger while default does not
        assert use_default is False
        assert use_strict is True

    def test_returns_positive_budget(self):
        _, _, budget, _ = _should_use_streaming(10, 100, memory_limit_gb=64)
        assert budget == 64 * 1e9

    def test_batch_size_bounded(self):
        """Batch size should be at least 100 and at most n_groups."""
        use, _, _, batch = _should_use_streaming(
            200, 36000, memory_limit_gb=0.01,  # tiny limit forces streaming
        )
        assert use is True
        assert batch >= 100
        assert batch <= 200


# ---------------------------------------------------------------------------
# wilcoxon_test with memory_limit_gb
# ---------------------------------------------------------------------------

def _make_test_h5ad(tmp_path: Path, n_cells: int = 60, n_genes: int = 20, n_perts: int = 3) -> Path:
    """Create a small normalised sparse h5ad for wilcoxon testing."""
    rng = np.random.default_rng(42)
    counts = rng.poisson(2, size=(n_cells, n_genes)).astype(float)
    labels = (["control"] * (n_cells // 2)
              + [f"pert_{i}" for i in range(n_perts) for _ in range(n_cells // (2 * n_perts))])
    # pad to n_cells
    while len(labels) < n_cells:
        labels.append("control")
    labels = labels[:n_cells]
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    # Normalise
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "test_norm.h5ad"
    adata.write(path)
    return path


class TestWilcoxonMemoryLimit:
    def test_standard_path_with_memory_limit(self, tmp_path):
        """memory_limit_gb=128 on a small dataset should use the standard path."""
        path = _make_test_h5ad(tmp_path)
        result = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
            memory_limit_gb=128,
        )
        assert len(result.groups) == 3
        assert result.pvalues.shape[1] == 20

    def test_streaming_path_with_tiny_limit(self, tmp_path):
        """A very small memory_limit_gb should force the streaming path."""
        path = _make_test_h5ad(tmp_path, n_cells=60, n_genes=20, n_perts=3)
        # 3 groups * 20 genes is tiny, but memory_limit_gb=0.0000001 forces streaming
        result = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
            data_name="stream_test",
            memory_limit_gb=1e-7,
        )
        assert len(result.groups) == 3
        assert result.pvalues.shape[1] == 20

    def test_results_match_across_paths(self, tmp_path):
        """Standard and streaming paths should produce identical results."""
        path = _make_test_h5ad(tmp_path)

        standard = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
            data_name="standard",
            memory_limit_gb=128,
        )
        streaming = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
            data_name="streaming",
            memory_limit_gb=1e-7,  # force streaming
        )

        np.testing.assert_allclose(standard.pvalues, streaming.pvalues, atol=1e-10)
        np.testing.assert_allclose(standard.statistics, streaming.statistics, atol=1e-10)
        np.testing.assert_allclose(standard.logfoldchanges, streaming.logfoldchanges, atol=1e-10)
        np.testing.assert_allclose(standard.effect_size, streaming.effect_size, atol=1e-10)

    def test_default_none_uses_system_memory(self, tmp_path):
        """memory_limit_gb=None (default) should still work and not crash."""
        path = _make_test_h5ad(tmp_path)
        result = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
        )
        assert len(result.groups) == 3


# ---------------------------------------------------------------------------
# t_test with memory_limit_gb
# ---------------------------------------------------------------------------

class TestTTestMemoryLimit:
    def test_with_explicit_memory_limit(self, tmp_path):
        """t_test should accept memory_limit_gb and produce valid results."""
        path = _make_test_h5ad(tmp_path)
        result = t_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
            memory_limit_gb=128,
        )
        assert len(result.groups) == 3
        assert result.pvalues.shape[1] == 20

    def test_default_none(self, tmp_path):
        """memory_limit_gb=None (default) should auto-detect and work."""
        path = _make_test_h5ad(tmp_path)
        result = t_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
        )
        assert len(result.groups) == 3

    def test_small_limit_still_works(self, tmp_path):
        """Even a very small memory_limit_gb should produce valid results."""
        path = _make_test_h5ad(tmp_path)
        result = t_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path,
            data_name="t_test_tiny",
            memory_limit_gb=0.001,
        )
        assert len(result.groups) == 3


# ---------------------------------------------------------------------------
# shrink_lfc with memory_limit_gb
# ---------------------------------------------------------------------------

def _make_nb_glm_h5ad(tmp_path: Path) -> Path:
    """Create a minimal NB-GLM-like result h5ad for shrink_lfc testing."""
    rng = np.random.default_rng(99)
    n_groups, n_genes = 3, 20
    obs = pd.DataFrame(
        {"perturbation": [f"pert_{i}" for i in range(n_groups)]},
        index=[f"g{i}" for i in range(n_groups)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    lfc = rng.standard_normal((n_groups, n_genes)).astype(np.float64)
    se = np.abs(rng.standard_normal((n_groups, n_genes))).astype(np.float64) + 0.1
    disp = np.abs(rng.standard_normal((n_groups, n_genes))).astype(np.float64) + 0.1
    adata = ad.AnnData(
        X=lfc.copy(),
        obs=obs,
        var=var,
        layers={
            "logfoldchanges": lfc.copy(),
            "logfoldchange_raw": lfc.copy(),
            "logfoldchange_raw_ln": lfc * np.log(2),
            "standard_error": se.copy(),
            "standard_error_ln": se * np.log(2),
            "dispersion": disp.copy(),
            "intercept": rng.standard_normal((n_groups, n_genes)),
            "fitted_intercept": rng.standard_normal((n_groups, n_genes)),
            "pvalues": rng.uniform(size=(n_groups, n_genes)),
            "pvalues_adj": rng.uniform(size=(n_groups, n_genes)),
            "statistics": rng.standard_normal((n_groups, n_genes)),
        },
    )
    adata.uns["control_label"] = "control"
    adata.uns["perturbation_column"] = "perturbation"
    adata.uns["lfc_base"] = "log2"
    adata.uns["de_method"] = "nb_glm"
    adata.uns["lfc_shrinkage_type"] = "none"
    path = tmp_path / "nb_glm_result.h5ad"
    adata.write(path)
    return path


class TestShrinkLfcMemoryLimit:
    def test_with_explicit_memory_limit(self, tmp_path):
        """shrink_lfc should accept memory_limit_gb and produce valid results."""
        path = _make_nb_glm_h5ad(tmp_path)
        result = shrink_lfc(
            path,
            output_dir=tmp_path,
            memory_limit_gb=128,
        )
        assert result is not None

    def test_default_none(self, tmp_path):
        """memory_limit_gb=None (default) should work."""
        path = _make_nb_glm_h5ad(tmp_path)
        result = shrink_lfc(
            path,
            output_dir=tmp_path,
            data_name="shrunk_default",
        )
        assert result is not None
