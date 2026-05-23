"""Regression tests to ensure streaming normalisation matches Scanpy."""

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
import h5py
from scipy.stats import norm, rankdata, t as t_dist

from crispyx.data import ensure_gene_symbol_column
from crispyx.de import _tie_correction, t_test, wilcoxon_test
from crispyx.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)


def _to_dense(matrix):
    """Convert sparse or dense matrix to numpy array."""
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


@pytest.fixture
def small_adata(tmp_path):
    counts = np.array(
        [
            [1, 0, 3, 0],
            [0, 2, 1, 0],
            [4, 0, 0, 1],
            [3, 1, 0, 0],
            [0, 0, 2, 2],
            [1, 0, 1, 3],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "ctrl", "A", "A", "B", "B"],
        },
        index=[f"cell_{i}" for i in range(counts.shape[0])],
    )
    var = pd.DataFrame(
        {"gene_symbols": ["G1", "G2", "G3", "G4"]},
        index=[f"g{i}" for i in range(counts.shape[1])],
    )
    # Use sparse matrix for t_test compatibility
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    path = tmp_path / "small.h5ad"
    adata.write(path)
    return path, adata


def _log_normalise_sparse(adata: ad.AnnData, path: Path) -> ad.AnnData:
    """Return a log-normalised sparse copy written to ``path``."""

    normalised = adata.copy()
    sc.pp.normalize_total(normalised, target_sum=1e4)
    sc.pp.log1p(normalised)
    normalised.X = sp.csr_matrix(normalised.X)
    normalised.write(path)
    return normalised


def _assert_t_test_matches_scanpy(path, adata, tmp_path):
    # t_test expects normalized/log-transformed data
    # Create normalized version for testing
    norm_adata = adata.copy()
    sc.pp.normalize_total(norm_adata, target_sum=1e4)
    sc.pp.log1p(norm_adata)
    norm_adata.X = sp.csr_matrix(norm_adata.X) if not sp.issparse(norm_adata.X) else norm_adata.X
    norm_path = tmp_path / "small_norm.h5ad"
    norm_adata.write(norm_path)
    
    results = t_test(
        norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        cell_chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_norm_cx_t_test.h5ad"
    assert output_path.exists()

    # Use the same normalized adata for expected values
    ctrl_mask = norm_adata.obs["perturbation"] == "ctrl"
    ctrl_data = _to_dense(norm_adata[ctrl_mask].X)
    ctrl_mean = ctrl_data.mean(axis=0)
    if ctrl_data.shape[0] > 1:
        ctrl_var = ctrl_data.var(axis=0, ddof=1)
    else:
        ctrl_var = np.zeros_like(ctrl_mean)
    raw_ctrl_X = adata[ctrl_mask].X
    ctrl_expr = np.asarray(_to_dense(raw_ctrl_X) > 0).sum(axis=0).ravel()

    for label, result in results.items():
        mask = norm_adata.obs["perturbation"] == label
        pert_data = _to_dense(norm_adata[mask].X)
        mean = pert_data.mean(axis=0)
        if pert_data.shape[0] > 1:
            var = pert_data.var(axis=0, ddof=1)
        else:
            var = np.zeros_like(mean)
        effect = mean - ctrl_mean
        n_ctrl = ctrl_data.shape[0]
        n_pert = pert_data.shape[0]
        var_term_ctrl = ctrl_var / n_ctrl
        var_term_pert = var / n_pert
        se = np.sqrt(var_term_ctrl + var_term_pert)
        raw_pert_X = adata[mask].X
        pert_expr = np.asarray(_to_dense(raw_pert_X) > 0).sum(axis=0).ravel()
        total_expr = ctrl_expr + pert_expr
        valid = se > 0
        z = np.zeros_like(effect)
        pvalue = np.ones_like(effect)
        z[valid] = effect[valid] / se[valid]
        # Welch-Satterthwaite degrees of freedom for Welch's t-test
        numerator = (var_term_ctrl + var_term_pert) ** 2
        denominator = np.zeros_like(numerator)
        if n_ctrl > 1:
            denominator += (var_term_ctrl ** 2) / (n_ctrl - 1)
        if n_pert > 1:
            denominator += (var_term_pert ** 2) / (n_pert - 1)
        df_welch = np.where(denominator > 0, numerator / denominator, 1e6)
        df_welch = np.clip(df_welch, 1.0, None)
        pvalue[valid] = 2 * t_dist.sf(np.abs(z[valid]), df_welch[valid])

        # Use looser tolerance due to float32 intermediate values in crispyx
        # (matches scanpy's approach for memory efficiency)
        np.testing.assert_allclose(result.effect_size, effect, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(result.statistic, z, rtol=1e-3, atol=0.1)
        # p-values need slightly looser tolerance for Welch df calculations with float32 intermediates
        np.testing.assert_allclose(result.pvalue, pvalue, rtol=1e-3, atol=1e-6)
        assert result.result_path == output_path


def test_average_log_expression_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    result = compute_average_log_expression(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_cx_avg_log_effects.h5ad"
    assert output_path.exists()

    result_mem = result.to_memory()
    result_df = pd.DataFrame(
        result_mem.X, index=result_mem.obs.index, columns=result_mem.var_names
    )

    sc_adata = adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)

    ctrl_mask = sc_adata.obs["perturbation"] == "ctrl"
    ctrl_mean = _to_dense(sc_adata[ctrl_mask].X).mean(axis=0)
    expected = {}
    for label in result_df.index:
        mask = sc_adata.obs["perturbation"] == label
        mean = _to_dense(sc_adata[mask].X).mean(axis=0)
        expected[label] = mean - ctrl_mean
    expected_df = pd.DataFrame(expected).T
    expected_df.columns = sc_adata.var["gene_symbols"].to_list()
    expected_df = expected_df.loc[result_df.index, result_df.columns]

    np.testing.assert_allclose(result_df.to_numpy(), expected_df.to_numpy(), rtol=1e-8, atol=1e-8)
    result.close()


def test_pseudobulk_expression_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    result = compute_pseudobulk_expression(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_cx_pseudobulk_effects.h5ad"
    assert output_path.exists()

    result_mem = result.to_memory()
    result_df = pd.DataFrame(
        result_mem.X, index=result_mem.obs.index, columns=result_mem.var_names
    )

    sc_adata = adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=1e4)

    ctrl_mask = sc_adata.obs["perturbation"] == "ctrl"
    ctrl_mean = _to_dense(sc_adata[ctrl_mask].X).mean(axis=0)
    expected = {}
    for label in result_df.index:
        mask = sc_adata.obs["perturbation"] == label
        mean = _to_dense(sc_adata[mask].X).mean(axis=0)
        expected[label] = np.log1p(mean) - np.log1p(ctrl_mean)
    expected_df = pd.DataFrame(expected).T
    expected_df.columns = sc_adata.var["gene_symbols"].to_list()
    expected_df = expected_df.loc[result_df.index, result_df.columns]

    np.testing.assert_allclose(result_df.to_numpy(), expected_df.to_numpy(), rtol=1e-8, atol=1e-8)
    result.close()


def test_t_test_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    _assert_t_test_matches_scanpy(path, adata, tmp_path)


def test_t_test_accepts_integer_counts(small_adata, tmp_path):
    path, adata = small_adata
    adata_int = adata.copy()
    # Convert to sparse integer matrix
    adata_int.X = sp.csr_matrix(np.asarray(adata_int.X.toarray(), dtype=np.int32))
    int_path = tmp_path / "small_int.h5ad"
    adata_int.write(int_path)

    _assert_t_test_matches_scanpy(int_path, adata_int, tmp_path)


def test_wilcoxon_test_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    norm_path = tmp_path / "small_log_norm.h5ad"
    norm_adata = _log_normalise_sparse(adata, norm_path)
    results = wilcoxon_test(
        norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_log_norm_cx_wilcoxon.h5ad"
    assert output_path.exists()

    log_data = np.asarray(norm_adata.X.toarray())
    ctrl_mask = norm_adata.obs["perturbation"] == "ctrl"
    control_values = log_data[ctrl_mask]
    control_n = float(control_values.shape[0])

    for label, result in results.items():
        mask = norm_adata.obs["perturbation"] == label
        pert_values = log_data[mask]
        pert_n = float(pert_values.shape[0])

        effects = []
        stats = []
        pvals = []
        for gene in range(adata.n_vars):
            combined = np.concatenate([pert_values[:, gene], control_values[:, gene]])[:, None]
            ranks = rankdata(combined, axis=0)
            rank_sum = float(ranks[: pert_values.shape[0]].sum())
            tie = float(_tie_correction(ranks)[0])
            expected = pert_n * (pert_n + control_n + 1.0) / 2.0
            std = float(np.sqrt(tie * pert_n * control_n * (pert_n + control_n + 1.0) / 12.0))
            u_stat = float(rank_sum - pert_n * (pert_n + 1.0) / 2.0)
            effect = u_stat / (control_n * pert_n) - 0.5
            effects.append(effect)
            if std == 0 or np.isnan(std):
                stats.append(0.0)
                pvals.append(1.0)
            else:
                z = (rank_sum - expected) / std
                stats.append(float(z))
                pvals.append(float(2 * norm.sf(abs(z))))

        np.testing.assert_allclose(result.effect_size, np.asarray(effects), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result.statistic, np.asarray(stats), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result.pvalue, np.asarray(pvals), atol=1e-9, rtol=1e-7)
        assert result.result_path == output_path


def test_ensure_gene_symbol_column_fallback_to_index():
    counts = np.zeros((2, 2))
    obs = pd.DataFrame(index=["cell1", "cell2"])
    var = pd.DataFrame(
        {"Ensembl_ID": ["ENS0001", "ENS0002"]},
        index=["GeneA", "GeneB"],
    )
    adata = ad.AnnData(counts, obs=obs, var=var)

    names = ensure_gene_symbol_column(adata, "gene_symbols")

    assert list(names) == ["GeneA", "GeneB"]


def test_t_test_with_n_jobs(small_adata, tmp_path):
    """Test that t_test works with n_jobs parameter."""
    path, adata = small_adata
    norm_path = tmp_path / "small_norm_njobs.h5ad"
    _log_normalise_sparse(adata, norm_path)
    
    # Run with n_jobs=2
    results_parallel = t_test(
        norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        cell_chunk_size=2,
        output_dir=tmp_path,
        data_name="parallel",
        n_jobs=2,
    )
    
    # Run without parallelization
    results_serial = t_test(
        norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        cell_chunk_size=2,
        output_dir=tmp_path,
        data_name="serial",
        n_jobs=1,
    )
    
    # Results should be identical
    assert set(results_parallel.keys()) == set(results_serial.keys())
    for label in results_parallel.keys():
        np.testing.assert_allclose(
            results_parallel[label].effect_size,
            results_serial[label].effect_size,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            results_parallel[label].statistic,
            results_serial[label].statistic,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            results_parallel[label].pvalue,
            results_serial[label].pvalue,
            rtol=1e-10,
            atol=1e-10,
        )


def test_wilcoxon_test_with_n_jobs(small_adata, tmp_path):
    """Test that wilcoxon_test works with n_jobs parameter."""
    path, adata = small_adata
    norm_path = tmp_path / "small_log_norm_wilcoxon.h5ad"
    _log_normalise_sparse(adata, norm_path)

    # Run with n_jobs=2
    results_parallel = wilcoxon_test(
        norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
        data_name="parallel_wilcoxon",
        n_jobs=2,
    )
    
    # Run without parallelization
    results_serial = wilcoxon_test(
        norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
        data_name="serial_wilcoxon",
        n_jobs=1,
    )
    
    # Results should be identical
    assert set(results_parallel.keys()) == set(results_serial.keys())
    for label in results_parallel.keys():
        np.testing.assert_allclose(
            results_parallel[label].effect_size,
            results_serial[label].effect_size,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            results_parallel[label].statistic,
            results_serial[label].statistic,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            results_parallel[label].pvalue,
            results_serial[label].pvalue,
            rtol=1e-10,
            atol=1e-10,
        )


def test_deseq2_size_factors_streaming_parity(tmp_path):
    """Verify streaming DESeq2 size factors match the dense path exactly."""
    from crispyx._size_factors import _deseq2_style_size_factors
    from crispyx.data import iter_matrix_chunks, read_backed

    # Create a small dataset where all genes are expressed in every cell
    np.random.seed(42)
    n_cells, n_genes = 200, 50
    # Poisson counts — ensure every cell-gene has count >= 1
    X = np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32) + 1
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs["pert"] = ["ctrl"] * 100 + ["pert_A"] * 100

    h5ad_path = tmp_path / "test_sf.h5ad"
    adata.write_h5ad(str(h5ad_path))

    # Dense path (small data, threshold not hit)
    sf_dense = _deseq2_style_size_factors(str(h5ad_path), chunk_size=64)

    # Replicate the streaming logic directly to verify correctness
    backed = read_backed(str(h5ad_path))
    n_c = backed.n_obs
    all_expressed = np.ones(n_genes, dtype=bool)
    try:
        for _, block in iter_matrix_chunks(backed, axis=0, chunk_size=64, convert_to_dense=False):
            csr = sp.csr_matrix(block)
            csc = csr.tocsc()
            nnz_per_gene = np.diff(csc.indptr)
            all_expressed[nnz_per_gene < csr.shape[0]] = False
    finally:
        backed.file.close()

    all_expressed_idx = np.where(all_expressed)[0]
    n_all_expressed = len(all_expressed_idx)
    assert n_all_expressed == n_genes  # all genes expressed in all cells

    # Streaming geo_means
    log_sum = np.zeros(n_all_expressed, dtype=np.float64)
    backed = read_backed(str(h5ad_path))
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=64, convert_to_dense=True):
            block_arr = np.asarray(block, dtype=np.float64)
            log_sum += np.log(np.maximum(block_arr[:, all_expressed_idx], 1e-300)).sum(axis=0)
    finally:
        backed.file.close()
    geo_means = np.exp(log_sum / n_c)

    # Streaming median of ratios
    sf_streaming = np.full(n_c, np.nan, dtype=np.float64)
    backed = read_backed(str(h5ad_path))
    try:
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=64, convert_to_dense=True):
            block_arr = np.asarray(block, dtype=np.float64)
            ratios = block_arr[:, all_expressed_idx] / geo_means
            sf_streaming[slc] = np.median(ratios, axis=1)
    finally:
        backed.file.close()

    # Apply same scaling as _deseq2_style_size_factors
    scale_factor = np.exp(np.mean(np.log(np.clip(sf_streaming, 1e-12, None))))
    sf_streaming = sf_streaming / scale_factor

    np.testing.assert_allclose(sf_streaming, sf_dense, rtol=1e-10, atol=1e-12)
