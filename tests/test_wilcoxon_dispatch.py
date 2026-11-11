"""Tests for wilcoxon dispatch, kernel optimizations, and large-dataset correctness.

Covers:
- _presort_control_nonzeros and _wilcoxon_presorted_ctrl_numba correctness
- Parity between presorted and original kernels
- Adaptive dispatch decisions for all benchmark dataset profiles at 128 GB / 32 cores
- Chunk size selection for large datasets
- Streaming vs standard path parity on real data (Adamson_subset)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

from crispyx._kernels import (
    _presort_control_nonzeros,
    _wilcoxon_presorted_ctrl_numba,
    _wilcoxon_sparse_batch_numba,
)
from crispyx._memory import _should_use_streaming
from crispyx.data import calculate_optimal_gene_chunk_size, calculate_wilcoxon_chunk_size
from crispyx.de import wilcoxon_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse_h5ad(
    tmp_path: Path,
    n_ctrl: int,
    n_perts: int,
    cells_per_pert: int,
    n_genes: int,
    sparsity: float = 0.30,
    seed: int = 42,
) -> Path:
    """Create a log-normalised sparse h5ad with given dimensions."""
    rng = np.random.default_rng(seed)
    n_cells = n_ctrl + n_perts * cells_per_pert
    counts = (rng.random((n_cells, n_genes)) < sparsity) * rng.exponential(2, (n_cells, n_genes))
    labels = ["control"] * n_ctrl + [f"pert_{i}" for i in range(n_perts) for _ in range(cells_per_pert)]
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.h5ad"
    adata.write(path)
    return path


# ---------------------------------------------------------------------------
# 1. _presort_control_nonzeros — structure and content
# ---------------------------------------------------------------------------

class TestPresortControlNonzeros:
    """Verify that _presort_control_nonzeros returns sorted, correct content."""

    def _make_ctrl(self, n_ctrl: int, n_genes: int, sparsity: float = 0.3, seed: int = 0):
        rng = np.random.default_rng(seed)
        mat = (rng.random((n_ctrl, n_genes)) < sparsity) * rng.exponential(2, (n_ctrl, n_genes))
        return mat.astype(np.float64)

    def test_output_shape(self):
        ctrl = self._make_ctrl(100, 64)
        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        assert offsets.shape == (65,)          # n_genes + 1
        assert n_nz.shape == (64,)
        assert n_z.shape == (64,)
        assert flat.shape == (int(n_nz.sum()),)

    def test_sorted_per_gene(self):
        ctrl = self._make_ctrl(200, 32, sparsity=0.5)
        flat, offsets, n_nz, _ = _presort_control_nonzeros(ctrl)
        for g in range(32):
            start, end = offsets[g], offsets[g + 1]
            gene_vals = flat[start:end]
            assert np.all(gene_vals[:-1] <= gene_vals[1:]), f"gene {g} not sorted"

    def test_nonzero_count_matches_dense(self):
        ctrl = self._make_ctrl(150, 48)
        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        expected_nz = (ctrl != 0).sum(axis=0)
        np.testing.assert_array_equal(n_nz, expected_nz)
        np.testing.assert_array_equal(n_z, 150 - expected_nz)

    def test_flat_values_match_original(self):
        ctrl = self._make_ctrl(80, 16)
        flat, offsets, n_nz, _ = _presort_control_nonzeros(ctrl)
        for g in range(16):
            start, end = offsets[g], offsets[g + 1]
            extracted = np.sort(ctrl[:, g][ctrl[:, g] != 0])
            np.testing.assert_allclose(flat[start:end], extracted)

    def test_all_zeros_gene(self):
        ctrl = np.zeros((50, 10), dtype=np.float64)
        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        assert flat.size == 0
        np.testing.assert_array_equal(n_nz, 0)
        np.testing.assert_array_equal(n_z, 50)

    def test_all_nonzero_gene(self):
        ctrl = np.ones((30, 8), dtype=np.float64)
        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        assert flat.size == 30 * 8
        np.testing.assert_array_equal(n_nz, 30)
        np.testing.assert_array_equal(n_z, 0)

    @pytest.mark.parametrize("n_ctrl", [100, 1000, 5000])
    def test_large_ctrl_sorted(self, n_ctrl):
        ctrl = self._make_ctrl(n_ctrl, 16, sparsity=0.20)
        flat, offsets, n_nz, _ = _presort_control_nonzeros(ctrl)
        for g in range(16):
            start, end = offsets[g], offsets[g + 1]
            gene_vals = flat[start:end]
            assert np.all(gene_vals[:-1] <= gene_vals[1:])


# ---------------------------------------------------------------------------
# 2. _wilcoxon_presorted_ctrl_numba — parity vs original kernel
# ---------------------------------------------------------------------------

class TestPresortedKernelParity:
    """Verify that presorted kernel produces results identical to original."""

    def _run_both_kernels(self, n_ctrl, n_pert, n_genes, sparsity=0.3, seed=0):
        rng = np.random.default_rng(seed)
        ctrl = (rng.random((n_ctrl, n_genes)) < sparsity) * rng.exponential(3, (n_ctrl, n_genes))
        pert = (rng.random((n_pert, n_genes)) < sparsity) * rng.exponential(3, (n_pert, n_genes))
        ctrl = ctrl.astype(np.float64)
        pert = pert.astype(np.float64)
        valid = np.ones(n_genes, dtype=np.bool_)

        # Original kernel
        u_old = np.zeros(n_genes, dtype=np.float64)
        z_old = np.zeros(n_genes, dtype=np.float64)
        p_old = np.ones(n_genes, dtype=np.float64)
        e_old = np.zeros(n_genes, dtype=np.float64)
        _wilcoxon_sparse_batch_numba(ctrl, pert, valid, True, 0.5, u_old, z_old, p_old, e_old)

        # Presorted kernel
        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        u_new = np.zeros(n_genes, dtype=np.float64)
        z_new = np.zeros(n_genes, dtype=np.float64)
        p_new = np.ones(n_genes, dtype=np.float64)
        e_new = np.zeros(n_genes, dtype=np.float64)
        _wilcoxon_presorted_ctrl_numba(ctrl, flat, offsets, n_nz, n_z, pert, valid, True, 0.5, u_new, z_new, p_new, e_new)

        return (u_old, z_old, p_old, e_old), (u_new, z_new, p_new, e_new)

    @pytest.mark.parametrize("n_ctrl,n_pert,n_genes", [
        (50, 30, 32),
        (500, 200, 64),
        (5000, 100, 128),   # Adamson-scale ctrl
        (50000, 50, 128),   # Feng-scale ctrl (small version)
    ])
    def test_u_stat_parity(self, n_ctrl, n_pert, n_genes):
        old, new = self._run_both_kernels(n_ctrl, n_pert, n_genes)
        np.testing.assert_allclose(old[0], new[0], atol=1e-9, rtol=1e-6,
                                   err_msg=f"U-stat mismatch at n_ctrl={n_ctrl}")

    @pytest.mark.parametrize("n_ctrl,n_pert,n_genes", [
        (50, 30, 32),
        (500, 200, 64),
        (5000, 100, 128),
    ])
    def test_pvalue_parity(self, n_ctrl, n_pert, n_genes):
        old, new = self._run_both_kernels(n_ctrl, n_pert, n_genes)
        np.testing.assert_allclose(old[2], new[2], atol=1e-9, rtol=1e-6,
                                   err_msg=f"P-value mismatch at n_ctrl={n_ctrl}")

    @pytest.mark.parametrize("n_ctrl,n_pert,n_genes", [
        (50, 30, 32),
        (500, 200, 64),
    ])
    def test_zscore_parity(self, n_ctrl, n_pert, n_genes):
        old, new = self._run_both_kernels(n_ctrl, n_pert, n_genes)
        np.testing.assert_allclose(old[1], new[1], atol=1e-9, rtol=1e-6,
                                   err_msg=f"Z-score mismatch at n_ctrl={n_ctrl}")

    def test_no_tie_correct_parity(self):
        old, new = self._run_both_kernels(200, 80, 32)
        # Also check without tie correction
        rng = np.random.default_rng(99)
        ctrl = rng.exponential(3, (200, 32)).astype(np.float64)
        pert = rng.exponential(3, (80, 32)).astype(np.float64)
        valid = np.ones(32, dtype=np.bool_)
        u0 = np.zeros(32); z0 = np.zeros(32); p0 = np.ones(32); e0 = np.zeros(32)
        _wilcoxon_sparse_batch_numba(ctrl, pert, valid, False, 0.5, u0, z0, p0, e0)
        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        u1 = np.zeros(32); z1 = np.zeros(32); p1 = np.ones(32); e1 = np.zeros(32)
        _wilcoxon_presorted_ctrl_numba(ctrl, flat, offsets, n_nz, n_z, pert, valid, False, 0.5, u1, z1, p1, e1)
        np.testing.assert_allclose(u0, u1, atol=1e-9)
        np.testing.assert_allclose(p0, p1, atol=1e-9)

    def test_invalid_genes_masked(self):
        """valid=False genes should produce u=0, z=0, p=1 in both kernels."""
        rng = np.random.default_rng(7)
        ctrl = rng.exponential(3, (100, 16)).astype(np.float64)
        pert = rng.exponential(3, (30, 16)).astype(np.float64)
        valid = np.array([True, False] * 8, dtype=np.bool_)

        u0 = np.zeros(16); z0 = np.zeros(16); p0 = np.ones(16); e0 = np.zeros(16)
        _wilcoxon_sparse_batch_numba(ctrl, pert, valid, True, 0.5, u0, z0, p0, e0)

        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        u1 = np.zeros(16); z1 = np.zeros(16); p1 = np.ones(16); e1 = np.zeros(16)
        _wilcoxon_presorted_ctrl_numba(ctrl, flat, offsets, n_nz, n_z, pert, valid, True, 0.5, u1, z1, p1, e1)

        # Even genes (valid) should match
        np.testing.assert_allclose(u0[valid], u1[valid], atol=1e-9)
        # Odd genes (invalid) should be 0, 0, 1
        np.testing.assert_array_equal(u1[~valid], 0.0)
        np.testing.assert_array_equal(z1[~valid], 0.0)
        np.testing.assert_array_equal(p1[~valid], 1.0)


# ---------------------------------------------------------------------------
# 3. Dispatch decisions at 32 cores / 128 GB for all benchmark datasets
# ---------------------------------------------------------------------------

# Dataset profiles: (n_groups, n_genes, expected_streaming)
# Verified against _should_use_streaming(peak_multiplier=4.0, threshold_fraction=0.30)
DATASET_DISPATCH = [
    # (dataset_name,        n_groups, n_genes,  n_cells,   streaming_expected)
    ("Adamson_subset",         2,     11_630,     1_716,   False),
    ("Adamson",               91,     32_738,    65_337,   False),
    ("Frangieh",             248,     23_712,   218_331,   False),
    ("Tian-crispra",         100,     33_538,    21_193,   False),
    ("Tian-crispri",         184,     33_538,    32_300,   False),
    ("Feng-gwsf",          2_254,     36_518,   322_746,   False),   # 21.1 GB < 38.4 GB
    ("Feng-gwsnf",         4_955,     36_518,   396_458,   True),    # 46.3 GB > 38.4 GB
    ("Feng-ts",              444,     36_518, 1_161_864,   False),   # 4.2 GB < 38.4 GB
    ("Huang-HCT116-est",   7_000,     38_606,   700_000,   True),    # 69.2 GB > 38.4 GB
    ("Huang-HEK293T",     18_311,     38_606, 4_534_299,   True),    # 181 GB > 38.4 GB
]


class TestDispatchDecisions:
    """Dispatch predictions must match expected mode for each benchmark dataset."""

    @pytest.mark.parametrize("name,n_groups,n_genes,n_cells,expected_stream", DATASET_DISPATCH)
    def test_dispatch_mode(self, name, n_groups, n_genes, n_cells, expected_stream):
        use, peak_bytes, budget_bytes, batch_sz = _should_use_streaming(
            n_groups, n_genes, memory_limit_gb=128,
        )
        peak_gb = peak_bytes / 1e9
        budget_gb = budget_bytes / 1e9
        assert use is expected_stream, (
            f"{name}: expected streaming={expected_stream}, got {use}. "
            f"peak={peak_gb:.1f} GB, budget={budget_gb:.1f} GB (threshold={0.30*budget_gb:.1f} GB)"
        )

    def test_dispatch_budget_always_128gb(self):
        """Budget should exactly equal 128 GB when memory_limit_gb=128."""
        for _, n_groups, n_genes, _, _ in DATASET_DISPATCH:
            _, _, budget, _ = _should_use_streaming(n_groups, n_genes, memory_limit_gb=128)
            assert budget == 128e9

    def test_streaming_batch_size_within_budget(self):
        """Batch size for streaming datasets should keep output arrays ≤ 19.2 GB."""
        streaming = [(name, ng, nv) for (name, ng, nv, _, stream) in DATASET_DISPATCH if stream]
        for name, n_groups, n_genes in streaming:
            _, _, budget, batch_sz = _should_use_streaming(n_groups, n_genes, memory_limit_gb=128)
            bytes_per_group = n_genes * (7 * 8 + 2 * 4)
            batch_gb = batch_sz * bytes_per_group / 1e9
            # Batch should use ≤ 15% of budget (= 0.30/2 * budget)
            assert batch_gb <= 0.15 * 128 + 1.0, (
                f"{name}: batch_size={batch_sz} uses {batch_gb:.1f} GB > allowed"
            )

    @pytest.mark.parametrize("name,n_groups,n_genes,n_cells,_", DATASET_DISPATCH)
    def test_chunk_size_bounded(self, name, n_groups, n_genes, n_cells, _):
        """Chunk sizes should be between 32 and 512 for all datasets."""
        chunk = calculate_optimal_gene_chunk_size(
            n_cells, n_genes, n_groups=n_groups, available_memory_gb=128,
        )
        assert 32 <= chunk <= 512, f"{name}: chunk={chunk} out of [32, 512]"

    @pytest.mark.parametrize("name,n_groups,n_genes,n_cells,_", DATASET_DISPATCH)
    def test_chunk_size_respects_cell_caps(self, name, n_groups, n_genes, n_cells, _):
        """Cell-count caps apply on memory-constrained machines (< 32 GB).
        On large-memory machines the memory-formula takes precedence."""
        # Low-memory: hard caps must hold
        chunk_low = calculate_optimal_gene_chunk_size(
            n_cells, n_genes, n_groups=n_groups, available_memory_gb=16,
        )
        if n_cells > 1_000_000:
            assert chunk_low <= 32, f"{name}: chunk={chunk_low} should be ≤32 for >1M cells on 16 GB"
        elif n_cells > 500_000:
            assert chunk_low <= 64, f"{name}: chunk={chunk_low} should be ≤64 for >500K cells on 16 GB"
        elif n_cells > 300_000:
            assert chunk_low <= 128, f"{name}: chunk={chunk_low} should be ≤128 for >300K cells on 16 GB"

        # Large-memory (128 GB): caps are lifted; chunk is driven by the memory formula
        chunk_high = calculate_optimal_gene_chunk_size(
            n_cells, n_genes, n_groups=n_groups, available_memory_gb=128,
        )
        assert 32 <= chunk_high <= 512, f"{name}: chunk={chunk_high} out of [32, 512] on 128 GB"
        # On 128 GB, large-cell datasets should get wider chunks than the 16-GB caps
        if n_cells > 300_000:
            assert chunk_high >= chunk_low, (
                f"{name}: 128 GB chunk ({chunk_high}) should be >= 16 GB chunk ({chunk_low})"
            )

    def test_feng_gwsnf_chunk_and_mode(self):
        """Feng-gwsnf: _should_use_streaming returns True but wilcoxon_test routes to standard path.

        _should_use_streaming correctly identifies the result arrays as large relative to the
        128 GB budget (51 GB estimated peak > 30% threshold). However, the computed
        group_batch_size (9,268) exceeds n_groups (4,955), meaning one batch would cover all
        groups — providing no benefit over the standard memmap path. wilcoxon_test therefore
        skips the streaming dispatch when group_batch_size >= n_groups, keeping peak RSS at
        ~10 GB instead of the 20-35 GB seen with the heap-array streaming path.
        """
        chunk = calculate_optimal_gene_chunk_size(396_458, 36_518, n_groups=4_955, available_memory_gb=128)
        use, _, _, batch_sz = _should_use_streaming(4_955, 36_518, memory_limit_gb=128)
        # Memory analysis still flags this as streaming-worthy (large relative to budget)
        assert use is True, "Feng-gwsnf: _should_use_streaming should still return True"
        # But batch_size >= n_groups means wilcoxon_test will route to standard memmap path
        assert batch_sz >= 4_955, (
            f"Feng-gwsnf: batch_sz={batch_sz} should be >= n_groups=4955 so standard path is used"
        )
        # On 128 GB the memory-formula drives chunk; the old hard cap (128) no longer applies.
        # n_groups=4955 triggers the n_groups > 2000 cap (≤384), so chunk should be in [32, 384].
        assert 32 <= chunk <= 384, f"Feng-gwsnf chunk={chunk} should be in [32, 384] on 128 GB"

    def test_huang_hek293t_streaming(self):
        """Huang-HEK293T: 18311 groups × 38606 genes must trigger streaming."""
        use, peak, budget, batch = _should_use_streaming(18_311, 38_606, memory_limit_gb=128)
        assert use is True
        peak_gb = peak / 1e9
        assert peak_gb > 128, f"Huang-HEK293T peak {peak_gb:.0f} GB should exceed budget"

    def test_feng_ts_standard(self):
        """Feng-ts: 444 groups × 36518 genes should NOT trigger streaming (4.2 GB peak)."""
        use, peak, _, _ = _should_use_streaming(444, 36_518, memory_limit_gb=128)
        assert use is False
        assert peak / 1e9 < 38.4, f"Feng-ts peak {peak/1e9:.1f} GB should be below threshold"


# ---------------------------------------------------------------------------
# 4. Streaming vs standard parity on Adamson_subset (real data)
# ---------------------------------------------------------------------------

ADAMSON_PATH = PROJECT_ROOT / "data" / "Adamson_subset.h5ad"


@pytest.mark.skipif(
    not ADAMSON_PATH.exists(),
    reason="Adamson_subset.h5ad not found — integration test skipped",
)
class TestAdamsonSubsetParity:
    """Standard and streaming paths must produce identical results on real data."""

    @pytest.fixture(autouse=True, scope="class")
    def _normalised_adamson(self, tmp_path_factory):
        """Create a normalised copy of the Adamson subset for all tests."""
        tmp = tmp_path_factory.mktemp("adamson_norm")
        adata = ad.read_h5ad(ADAMSON_PATH)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        norm_path = tmp / "Adamson_subset_norm.h5ad"
        adata.write(norm_path)
        type(self).norm_path = norm_path

    def test_pvalue_parity(self, tmp_path):
        standard = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "std",
            memory_limit_gb=128,
        )
        streaming = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "stream",
            memory_limit_gb=1e-7,
        )
        np.testing.assert_array_equal(sorted(standard.groups), sorted(streaming.groups))
        for g in standard.groups:
            np.testing.assert_allclose(
                standard[g].pvalue, streaming[g].pvalue, atol=1e-10,
                err_msg=f"pvalue mismatch for group {g}",
            )

    def test_effect_size_parity(self, tmp_path):
        standard = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "std2",
            memory_limit_gb=128,
        )
        streaming = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "stream2",
            memory_limit_gb=1e-7,
        )
        for g in standard.groups:
            np.testing.assert_allclose(
                standard[g].effect_size, streaming[g].effect_size, atol=1e-10,
                err_msg=f"effect_size mismatch for group {g}",
            )

    def test_statistic_parity(self, tmp_path):
        standard = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "std3",
            memory_limit_gb=128,
        )
        streaming = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "stream3",
            memory_limit_gb=1e-7,
        )
        for g in standard.groups:
            np.testing.assert_allclose(
                standard[g].statistic, streaming[g].statistic, atol=1e-10,
                err_msg=f"statistic mismatch for group {g}",
            )

    def test_pvalue_adj_parity(self, tmp_path):
        standard = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "std4",
            memory_limit_gb=128,
        )
        streaming = wilcoxon_test(
            self.norm_path,
            perturbation_column="perturbation",
            control_label="control",
            output_dir=tmp_path / "stream4",
            memory_limit_gb=1e-7,
        )
        for g in standard.groups:
            np.testing.assert_allclose(
                standard[g].pvalue_adj, streaming[g].pvalue_adj, atol=1e-10,
                err_msg=f"pvalue_adj mismatch for group {g}",
            )

    def test_dispatch_uses_standard_at_128gb(self):
        """Adamson_subset (2 groups × 11630 genes) should NOT stream at 128 GB."""
        use, peak, _, _ = _should_use_streaming(2, 11_630, memory_limit_gb=128)
        assert use is False
        assert peak / 1e9 < 1.0

    def test_dispatch_uses_streaming_at_tiny_limit(self):
        """Adamson_subset forced to stream by tiny limit."""
        use, _, _, _ = _should_use_streaming(2, 11_630, memory_limit_gb=1e-7)
        assert use is True


# ---------------------------------------------------------------------------
# 5. Wilcoxon on synthetic datasets with large control groups
# ---------------------------------------------------------------------------

class TestLargeControlGroupParity:
    """Parity tests specifically for large n_ctrl (Feng-scale: ≥50K control cells)."""

    @pytest.mark.parametrize("n_ctrl,n_perts", [
        (50, 10),     # tiny — baseline
        (500, 5),     # small control
        (2000, 5),    # medium control
    ])
    def test_streaming_standard_parity_synthetic(self, tmp_path, n_ctrl, n_perts):
        path = _make_sparse_h5ad(tmp_path / f"c{n_ctrl}_p{n_perts}",
                                 n_ctrl=n_ctrl, n_perts=n_perts,
                                 cells_per_pert=10, n_genes=64, seed=n_ctrl)
        std = wilcoxon_test(path, perturbation_column="perturbation",
                            output_dir=tmp_path / f"std_c{n_ctrl}", memory_limit_gb=128)
        stream = wilcoxon_test(path, perturbation_column="perturbation",
                               output_dir=tmp_path / f"st_c{n_ctrl}", data_name="stream",
                               memory_limit_gb=1e-7)
        for g in std.groups:
            np.testing.assert_allclose(std[g].pvalue, stream[g].pvalue, atol=1e-9,
                                       err_msg=f"n_ctrl={n_ctrl}: pvalue mismatch for {g}")
            np.testing.assert_allclose(std[g].effect_size, stream[g].effect_size, atol=1e-9,
                                       err_msg=f"n_ctrl={n_ctrl}: effect_size mismatch for {g}")

    def test_kernel_parity_large_ctrl(self):
        """Direct kernel comparison with n_ctrl=50000 (Feng-scale)."""
        rng = np.random.default_rng(12345)
        n_ctrl, n_pert, n_genes = 50_000, 5, 32
        sparsity = 0.20  # ~20% non-zero typical for Feng datasets

        ctrl = (rng.random((n_ctrl, n_genes)) < sparsity) * rng.exponential(3, (n_ctrl, n_genes))
        pert = (rng.random((n_pert, n_genes)) < sparsity) * rng.exponential(3, (n_pert, n_genes))
        ctrl = ctrl.astype(np.float64)
        pert = pert.astype(np.float64)
        valid = np.ones(n_genes, dtype=np.bool_)

        u_old = np.zeros(n_genes); z_old = np.zeros(n_genes)
        p_old = np.ones(n_genes); e_old = np.zeros(n_genes)
        _wilcoxon_sparse_batch_numba(ctrl, pert, valid, True, 0.5, u_old, z_old, p_old, e_old)

        flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
        u_new = np.zeros(n_genes); z_new = np.zeros(n_genes)
        p_new = np.ones(n_genes); e_new = np.zeros(n_genes)
        _wilcoxon_presorted_ctrl_numba(ctrl, flat, offsets, n_nz, n_z, pert, valid, True, 0.5,
                                       u_new, z_new, p_new, e_new)

        np.testing.assert_allclose(u_old, u_new, atol=1e-9,
                                   err_msg="U-stat mismatch at n_ctrl=50000")
        np.testing.assert_allclose(p_old, p_new, atol=1e-9,
                                   err_msg="P-value mismatch at n_ctrl=50000")
        np.testing.assert_allclose(z_old, z_new, atol=1e-9,
                                   err_msg="Z-score mismatch at n_ctrl=50000")


# ---------------------------------------------------------------------------
# 6. Memory estimation accuracy
# ---------------------------------------------------------------------------

class TestMemoryEstimation:
    """Verify peak memory estimates are consistent with 5× multiplier."""

    def test_peak_equals_5x_memmap(self):
        n_groups, n_genes = 4955, 36518
        bytes_per_group = n_genes * (7 * 8 + 2 * 4)
        expected_memmap = n_groups * bytes_per_group
        expected_peak = expected_memmap * 5.0
        _, peak_bytes, _, _ = _should_use_streaming(n_groups, n_genes, memory_limit_gb=128)
        assert abs(peak_bytes - expected_peak) < 1, (
            f"Peak estimate mismatch: got {peak_bytes}, expected {expected_peak}"
        )

    def test_streaming_triggers_at_30pct_budget(self):
        """Streaming must trigger when peak > 30% of 128 GB = 38.4 GB."""
        # Feng-gwsnf: peak ≈ 46.3 GB > 38.4 GB threshold
        use, peak, budget, _ = _should_use_streaming(4955, 36518, memory_limit_gb=128)
        assert use is True
        assert peak > 0.30 * budget, f"peak {peak/1e9:.1f} GB should exceed threshold {0.30*budget/1e9:.1f} GB"

    def test_no_streaming_below_30pct(self):
        """Standard mode must stay when peak < 30% of budget."""
        # Feng-gwsf: peak ≈ 21.1 GB < 38.4 GB threshold
        use, peak, budget, _ = _should_use_streaming(2254, 36518, memory_limit_gb=128)
        assert use is False
        assert peak <= 0.30 * budget + 1, f"peak {peak/1e9:.1f} GB should be below threshold"

    @pytest.mark.parametrize("mem_gb", [32, 64, 128, 256])
    def test_budget_scales_with_memory_limit(self, mem_gb):
        _, _, budget, _ = _should_use_streaming(100, 1000, memory_limit_gb=mem_gb)
        assert budget == mem_gb * 1e9


# ---------------------------------------------------------------------------
# 7. OOM fix helpers — _write_wilcoxon_result_h5ad / _build_result_from_h5ad
# ---------------------------------------------------------------------------

from crispyx.de import (
    _release_chunk_memory,
    _write_wilcoxon_result_h5ad,
    _build_result_from_h5ad,
)


class TestWriteResultH5ad:
    """Verify _write_wilcoxon_result_h5ad writes correct HDF5 structure."""

    @staticmethod
    def _make_arrays(n_groups: int, n_genes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        return dict(
            effect_matrix=rng.standard_normal((n_groups, n_genes)),
            z_matrix=rng.standard_normal((n_groups, n_genes)),
            pvalue_matrix=rng.random((n_groups, n_genes)),
            pvalue_adj_matrix=rng.random((n_groups, n_genes)),
            lfc_matrix=rng.standard_normal((n_groups, n_genes)),
            u_matrix=rng.random((n_groups, n_genes)) * 1000,
            pts_matrix=rng.random((n_groups, n_genes)).astype(np.float32),
            pts_rest_matrix=rng.random((n_groups, n_genes)).astype(np.float32),
        )

    def test_roundtrip_values(self, tmp_path):
        """Values written via h5py match what AnnData reads back."""
        import h5py
        n_g, n_v = 5, 20
        candidates = [f"pert_{i}" for i in range(n_g)]
        genes = pd.Index([f"gene{i}" for i in range(n_v)])
        arrs = self._make_arrays(n_g, n_v)
        out = tmp_path / "test.h5ad"
        _write_wilcoxon_result_h5ad(
            out, candidates=candidates, gene_symbols=genes,
            perturbation_column="perturbation", control_label="control",
            tie_correct=True, corr_method="benjamini-hochberg", **arrs,
        )
        with h5py.File(out, "r") as hf:
            np.testing.assert_allclose(hf["X"][:], arrs["effect_matrix"])
            np.testing.assert_allclose(hf["layers/z_score"][:], arrs["z_matrix"])
            np.testing.assert_allclose(hf["layers/pvalue"][:], arrs["pvalue_matrix"])
            np.testing.assert_allclose(hf["layers/pvalue_adj"][:], arrs["pvalue_adj_matrix"])
            np.testing.assert_allclose(hf["layers/logfoldchanges"][:], arrs["lfc_matrix"])
            np.testing.assert_allclose(hf["layers/u_statistic"][:], arrs["u_matrix"])
            np.testing.assert_allclose(hf["layers/pts"][:], arrs["pts_matrix"], atol=1e-7)
            np.testing.assert_allclose(hf["layers/pts_rest"][:], arrs["pts_rest_matrix"], atol=1e-7)

    def test_obs_var_metadata(self, tmp_path):
        """obs/var indices are encoded correctly."""
        import h5py
        n_g, n_v = 3, 10
        candidates = [f"pert_{i}" for i in range(n_g)]
        genes = pd.Index([f"gene{i}" for i in range(n_v)])
        arrs = self._make_arrays(n_g, n_v)
        out = tmp_path / "meta.h5ad"
        _write_wilcoxon_result_h5ad(
            out, candidates=candidates, gene_symbols=genes,
            perturbation_column="perturbation", control_label="control",
            tie_correct=False, corr_method="bonferroni", **arrs,
        )
        with h5py.File(out, "r") as hf:
            obs_idx = [x.decode() for x in hf["obs/_index"][:]]
            var_idx = [x.decode() for x in hf["var/_index"][:]]
            assert obs_idx == candidates
            assert var_idx == list(genes.astype(str))
            assert hf["uns"].attrs["method"] == "wilcoxon"
            assert hf["uns"].attrs["tie_correct"] == False
            assert hf["uns"].attrs["pvalue_correction"] == "bonferroni"


class TestBuildResultFromH5ad:
    """Verify _build_result_from_h5ad reads back correctly."""

    @staticmethod
    def _write_and_build(tmp_path, n_groups=4, n_genes=16, memory_limit_gb=128.0):
        rng = np.random.default_rng(7)
        candidates = [f"p{i}" for i in range(n_groups)]
        genes = pd.Index([f"g{i}" for i in range(n_genes)])
        arrs = TestWriteResultH5ad._make_arrays(n_groups, n_genes, seed=7)
        out = tmp_path / "build.h5ad"
        _write_wilcoxon_result_h5ad(
            out, candidates=candidates, gene_symbols=genes,
            perturbation_column="perturbation", control_label="ctrl",
            tie_correct=True, corr_method="benjamini-hochberg", **arrs,
        )
        result = _build_result_from_h5ad(
            out, candidates=candidates, gene_symbols=genes,
            perturbation_column="perturbation", control_label="ctrl",
            tie_correct=True, corr_method="benjamini-hochberg",
            memory_limit_gb=memory_limit_gb,
        )
        return result, arrs

    def test_statistics_match(self, tmp_path):
        result, arrs = self._write_and_build(tmp_path)
        np.testing.assert_allclose(result.statistics, arrs["z_matrix"])
        np.testing.assert_allclose(result.effect_size, arrs["effect_matrix"])
        np.testing.assert_allclose(result.pvalues, arrs["pvalue_matrix"])
        np.testing.assert_allclose(result.logfoldchanges, arrs["lfc_matrix"])

    def test_order_is_z_argsort(self, tmp_path):
        result, arrs = self._write_and_build(tmp_path)
        expected_order = np.argsort(-np.abs(arrs["z_matrix"]), axis=1, kind="mergesort")
        np.testing.assert_array_equal(result.order, expected_order)

    def test_lazy_mode_for_huge_result(self, tmp_path, monkeypatch):
        """With mocked physical memory, result should be lazy (empty arrays)."""
        # Mock psutil to report tiny physical memory so the lazy branch triggers
        import types
        fake_psutil = types.ModuleType("psutil")
        fake_vm = lambda: types.SimpleNamespace(available=100)  # 100 bytes
        fake_psutil.virtual_memory = fake_vm
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        result, _ = self._write_and_build(tmp_path, memory_limit_gb=None)
        assert result.statistics.size == 0
        assert result.result is not None  # AnnData handle should still be set

    def test_groups_and_genes(self, tmp_path):
        result, _ = self._write_and_build(tmp_path, n_groups=6, n_genes=24)
        assert len(result.groups) == 6
        assert len(result.genes) == 24


class TestReleaseChunkMemory:
    """Verify _release_chunk_memory runs without errors on Linux."""

    def test_no_exception(self):
        _release_chunk_memory()

    def test_idempotent(self):
        """Calling twice in a row should not raise."""
        _release_chunk_memory()
        _release_chunk_memory()


# ---------------------------------------------------------------------------
# 8. Adaptive chunk size for high-cell datasets
# ---------------------------------------------------------------------------

class TestAdaptiveChunkSizeHighCells:
    """Verify per-chunk budget cap limits chunk size for high-cell datasets."""

    def test_feng_ts_chunk_capped(self):
        """Feng-ts (1.16M cells, 128 GB): chunk < 512."""
        chunk = calculate_optimal_gene_chunk_size(
            n_obs=1_161_864, n_vars=33_165, n_groups=444,
            available_memory_gb=128.0,
        )
        # 5% of 128 GB = 6.4 GB.  Per gene ≈ 1.16M × 12 = 13.9 MB
        # Budget cap ≈ 460 genes. With other caps it should be < 512.
        assert chunk < 512, f"chunk {chunk} should be < 512 for Feng-ts"
        assert chunk >= 32, f"chunk {chunk} should be >= min_chunk"

    def test_small_dataset_unchanged(self):
        """Small dataset should not be capped by the per-chunk budget."""
        chunk = calculate_optimal_gene_chunk_size(
            n_obs=10_000, n_vars=5_000, n_groups=50,
            available_memory_gb=128.0,
        )
        # 10K cells × 12 bytes = 120 KB per gene; 512 genes = 61 MB << 6.4 GB budget
        assert chunk == 512, f"chunk {chunk} should be 512 for small dataset"


# ---------------------------------------------------------------------------
# 9. Wilcoxon chunk size with 15% budget (Fix 1 verification)
# ---------------------------------------------------------------------------

class TestWilcoxonChunkSize15pct:
    """Verify calculate_wilcoxon_chunk_size with the 15% budget and max_chunk=4096."""

    def test_feng_gwsnf_larger_chunk(self):
        """Feng-gwsnf (393K cells, 128 GB): chunk should increase from ~1355 to ~4067."""
        chunk = calculate_wilcoxon_chunk_size(393465, 32373, available_memory_gb=128)
        assert chunk > 3000, f"chunk {chunk} should be > 3000 for Feng-gwsnf at 128 GB"
        assert chunk <= 4096, f"chunk {chunk} should be <= max_chunk=4096"

    def test_feng_ts_larger_chunk(self):
        """Feng-ts (1.16M cells, 128 GB): chunk should increase from ~459 to ~1378."""
        chunk = calculate_wilcoxon_chunk_size(1161864, 33165, available_memory_gb=128)
        assert chunk > 1000, f"chunk {chunk} should be > 1000 for Feng-ts at 128 GB"

    def test_small_dataset_hits_max_chunk(self):
        """Small dataset (21K cells, 128 GB): budget cap is huge, hits max_chunk=4096."""
        chunk = calculate_wilcoxon_chunk_size(21071, 22040, available_memory_gb=128)
        assert chunk == 4096, f"chunk {chunk} should be 4096 (max_chunk) for small dataset"

    def test_low_memory_still_small(self):
        """Feng-gwsnf at 16 GB: chunk should still be small."""
        chunk = calculate_wilcoxon_chunk_size(393465, 32373, available_memory_gb=16)
        assert chunk < 1000, f"chunk {chunk} should be < 1000 for 16 GB"
        assert chunk >= 32, f"chunk {chunk} should be >= min_chunk=32"

    def test_replogle_gw_larger_chunk(self):
        """Replogle-GW (1.97M cells, 128 GB): chunk should increase from ~271 to ~813."""
        chunk = calculate_wilcoxon_chunk_size(1970000, 8248, available_memory_gb=128)
        assert chunk > 700, f"chunk {chunk} should be > 700 for Replogle-GW at 128 GB"

    def test_fewer_chunks_feng_gwsnf(self):
        """Feng-gwsnf: 8 chunks instead of 24."""
        import math
        chunk = calculate_wilcoxon_chunk_size(393465, 32373, available_memory_gb=128)
        n_chunks = math.ceil(32373 / chunk)
        assert n_chunks <= 10, f"n_chunks {n_chunks} should be <= 10 (was 24)"


# ---------------------------------------------------------------------------
# 8. Dense-gene binary search correctness
# ---------------------------------------------------------------------------

class TestDenseGeneBinarySearch:
    """Verify Wilcoxon results are correct for genes with zero_frac < 0.5.

    These genes previously used the O(n_total*log(n_total)) argsort path.
    After Fix 3 they always use binary search, which must give identical
    statistical results.
    """

    def _make_dense_h5ad(self, tmp_path, n_ctrl=200, n_perts=5,
                         cells_per_pert=10, n_genes=20, seed=99):
        """Create h5ad where ALL genes are dense (zero_frac ≈ 0)."""
        rng = np.random.default_rng(seed)
        n_cells = n_ctrl + n_perts * cells_per_pert
        # Positive expression values only — no zeros
        data = rng.exponential(2.0, (n_cells, n_genes)) + 0.1
        labels = (["control"] * n_ctrl
                  + [f"p{i}" for i in range(n_perts) for _ in range(cells_per_pert)])
        obs = pd.DataFrame({"perturbation": labels},
                           index=[f"c{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        adata = ad.AnnData(sp.csr_matrix(data), obs=obs, var=var)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
        path = tmp_path / "dense_genes.h5ad"
        adata.write(path)
        return path

    def _make_mixed_h5ad(self, tmp_path, n_ctrl=200, n_perts=5,
                         cells_per_pert=10, n_genes=20, seed=99):
        """Create h5ad with a mix of dense and sparse genes.

        First half: dense (zero_frac ≈ 0).
        Second half: sparse (zero_frac ≈ 0.8).
        """
        rng = np.random.default_rng(seed)
        n_cells = n_ctrl + n_perts * cells_per_pert
        n_dense = n_genes // 2
        n_sparse = n_genes - n_dense
        dense = rng.exponential(2.0, (n_cells, n_dense)) + 0.1
        sparse = (rng.random((n_cells, n_sparse)) < 0.2) * rng.exponential(2.0, (n_cells, n_sparse))
        data = np.hstack([dense, sparse])
        labels = (["control"] * n_ctrl
                  + [f"p{i}" for i in range(n_perts) for _ in range(cells_per_pert)])
        obs = pd.DataFrame({"perturbation": labels},
                           index=[f"c{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        adata = ad.AnnData(sp.csr_matrix(data), obs=obs, var=var)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
        path = tmp_path / "mixed_genes.h5ad"
        adata.write(path)
        return path

    def test_dense_genes_vs_scanpy(self, tmp_path):
        """All-dense genes: crispyx matches Scanpy p-values."""
        path = self._make_dense_h5ad(tmp_path)
        result = wilcoxon_test(path, perturbation_column="perturbation",
                               control_label="control", output_dir=tmp_path)
        # Compare with Scanpy
        adata = sc.read_h5ad(path)
        sc.tl.rank_genes_groups(adata, groupby="perturbation",
                                reference="control", method="wilcoxon",
                                tie_correct=True)
        for label in [f"p{i}" for i in range(5)]:
            cx_p = result[label].pvalue
            sc_df = sc.get.rank_genes_groups_df(adata, group=label)
            sc_pvals = sc_df.set_index("names").reindex(result.genes)["pvals"].values
            # Relaxed tolerance — kernel numerics may differ slightly
            np.testing.assert_allclose(cx_p, sc_pvals, rtol=1e-4, atol=1e-10)

    def test_mixed_genes_vs_scanpy(self, tmp_path):
        """Mixed dense+sparse genes: crispyx matches Scanpy p-values."""
        path = self._make_mixed_h5ad(tmp_path)
        result = wilcoxon_test(path, perturbation_column="perturbation",
                               control_label="control", output_dir=tmp_path)
        adata = sc.read_h5ad(path)
        sc.tl.rank_genes_groups(adata, groupby="perturbation",
                                reference="control", method="wilcoxon",
                                tie_correct=True)
        for label in [f"p{i}" for i in range(5)]:
            cx_p = result[label].pvalue
            sc_df = sc.get.rank_genes_groups_df(adata, group=label)
            sc_pvals = sc_df.set_index("names").reindex(result.genes)["pvals"].values
            np.testing.assert_allclose(cx_p, sc_pvals, rtol=1e-4, atol=1e-10)

    def test_all_zero_gene(self, tmp_path):
        """Gene with all zeros across ctrl + pert → p=1, U=0."""
        rng = np.random.default_rng(123)
        n_ctrl, n_perts, cpp, n_genes = 100, 3, 5, 10
        n_cells = n_ctrl + n_perts * cpp
        data = (rng.random((n_cells, n_genes)) < 0.3) * rng.exponential(2, (n_cells, n_genes))
        data[:, 0] = 0.0  # gene 0 is all zeros
        labels = (["control"] * n_ctrl
                  + [f"p{i}" for i in range(n_perts) for _ in range(cpp)])
        obs = pd.DataFrame({"perturbation": labels},
                           index=[f"c{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        adata = ad.AnnData(sp.csr_matrix(data), obs=obs, var=var)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
        path = tmp_path / "allzero.h5ad"
        adata.write(path)
        result = wilcoxon_test(path, perturbation_column="perturbation",
                               control_label="control", output_dir=tmp_path,
                               min_pct_both=0.0, min_mean_ctrl=0.0)
        for label in [f"p{i}" for i in range(n_perts)]:
            assert result[label].pvalue[0] == 1.0, "All-zero gene should have p=1"
