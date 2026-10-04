from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import h5py
from scipy.stats import mannwhitneyu

import crispyx as cx

from crispyx import (
    compute_normalized_effects,
    quality_control_summary,
    t_test,
    wilcoxon_test,
    nb_glm_test,
)


def create_test_dataset(tmp_path):
    x = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 3, 0],
            [0, 1, 0, 0],
            [2, 1, 0, 4],
            [0, 0, 0, 5],
            [1, 0, 0, 6],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "ctrl", "KO1", "KO1", "KO2", "KO2"]},
        index=[f"cell_{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame({"gene_symbol": [f"gene{i}" for i in range(x.shape[1])]})
    var.index = var["gene_symbol"]
    # Use sparse matrix for t_test compatibility
    adata = ad.AnnData(sp.csr_matrix(x), obs=obs, var=var)
    path = tmp_path / "test.h5ad"
    adata.write(path)
    return path, adata


def _log_normalise_sparse(adata: ad.AnnData) -> ad.AnnData:
    """Return a log-normalised sparse copy of ``adata``."""

    normalised = adata.copy()
    sc.pp.normalize_total(normalised)
    sc.pp.log1p(normalised)
    normalised.X = sp.csr_matrix(normalised.X)
    return normalised


def _normalize_total(matrix: np.ndarray, *, target_sum: float = 1e4) -> np.ndarray:
    """Replicate the package's library-size normalisation."""

    matrix = np.asarray(matrix, dtype=float)
    library_size = matrix.sum(axis=1, keepdims=True)
    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=float),
        where=library_size > 0,
    )
    return matrix * scale


def test_quality_control_writes_filtered_dataset(tmp_path):
    path, adata = create_test_dataset(tmp_path)
    result = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="qc_test",
    )
    assert isinstance(result.filtered, cx.AnnData)
    filtered = result.filtered.to_memory()
    expected = adata[result.cell_mask, result.gene_mask]
    np.testing.assert_array_equal(filtered.X.toarray(), expected.X.toarray())
    assert filtered.var["gene_symbols"].tolist() == expected.var["gene_symbol"].tolist()
    with h5py.File(result.filtered_path) as handle:
        encoding = handle["X"].attrs["encoding-type"]
        assert (encoding.decode() if isinstance(encoding, bytes) else encoding) == "csr_matrix"
    result.filtered.close()


def test_gene_symbol_validation(tmp_path):
    x = np.ones((5, 3))
    obs = pd.DataFrame({"perturbation": ["ctrl"] * 5})
    obs.index = [f"cell_{i}" for i in range(obs.shape[0])]
    var = pd.DataFrame(index=["ENSG000001", "ENSG000002", "ENSG000003"])
    adata = ad.AnnData(x, obs=obs, var=var)
    path = tmp_path / "invalid.h5ad"
    adata.write(path)
    with pytest.raises(ValueError, match="Ensembl"):
        quality_control_summary(
            path,
            min_genes=1,
            min_cells_per_perturbation=1,
            min_cells_per_gene=1,
            perturbation_column="perturbation",
            control_label="ctrl",
        )


def test_downstream_effect_outputs(tmp_path):
    path, _ = create_test_dataset(tmp_path)
    qc_result = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="de",
    )
    avg = compute_normalized_effects(
        qc_result.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="avg_effects",
    )
    wilcoxon_input = _log_normalise_sparse(qc_result.filtered.to_memory())
    wilcoxon_input_path = tmp_path / "qc_filtered_log_norm.h5ad"
    wilcoxon_input.write(wilcoxon_input_path)
    pseudo = compute_normalized_effects(
        qc_result.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        method="log_mean",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="pseudo_effects",
    )
    # t_test expects pre-normalized data, use the log-normalized version
    wald = t_test(
        wilcoxon_input_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="t_test",
    )
    wilcoxon = wilcoxon_test(
        wilcoxon_input_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wilcoxon",
    )

    assert isinstance(avg, cx.AnnData)
    assert isinstance(pseudo, cx.AnnData)
    avg_mem = avg.to_memory()
    pseudo_mem = pseudo.to_memory()
    avg_df = pd.DataFrame(avg_mem.X, index=avg_mem.obs.index, columns=avg_mem.var_names)
    pseudo_df = pd.DataFrame(
        pseudo_mem.X, index=pseudo_mem.obs.index, columns=pseudo_mem.var_names
    )
    assert set(avg_df.index) == {"KO1", "KO2"}
    assert set(pseudo_df.index) == {"KO1", "KO2"}
    assert set(wald.keys()) == {"KO1", "KO2"}
    assert set(wilcoxon.keys()) == {"KO1", "KO2"}

    filtered = qc_result.filtered.to_memory()
    ctrl_mask = (filtered.obs["perturbation"] == "ctrl").to_numpy()
    ko1_mask = (filtered.obs["perturbation"] == "KO1").to_numpy()
    # compute_normalized_effects(method="mean_log1p") uses _normalize_total(target_sum=1e4) then log1p
    raw_block = np.asarray(filtered.X.toarray())
    log_block = np.log1p(_normalize_total(raw_block, target_sum=1e4))
    ctrl = log_block[ctrl_mask, 0]
    ko1 = log_block[ko1_mask, 0]
    expected = ko1.mean() - ctrl.mean()
    assert np.isclose(avg_df.loc["KO1", "gene0"], expected)

    assert (tmp_path / "avg_effects_cx_normalized_effects.h5ad").exists()
    assert (tmp_path / "pseudo_effects_cx_normalized_effects.h5ad").exists()
    assert (tmp_path / "t_test_cx_t_test.h5ad").exists()
    assert (tmp_path / "wilcoxon_cx_wilcoxon.h5ad").exists()

    ko1_result = wald["KO1"]
    assert ko1_result.effect_size.shape[0] == 4
    assert ko1_result.method == "t_test"
    ko2_result = wilcoxon["KO2"]
    assert ko2_result.pvalue.shape[0] == 4
    assert ko2_result.method == "wilcoxon"
    avg.close()
    pseudo.close()
    qc_result.filtered.close()


def test_scanpy_style_namespaces_match_direct(tmp_path):
    path, _ = create_test_dataset(tmp_path)
    adata_ro = ad.read_h5ad(path, backed="r")
    try:
        qc_wrapped = cx.pp.qc_summary(
            adata_ro,
            min_genes=1,
            min_cells_per_perturbation=2,
            min_cells_per_gene=1,
            perturbation_column="perturbation",
            control_label="ctrl",
            gene_name_column="gene_symbol",
            output_dir=tmp_path,
            data_name="wrapped",
        )
    finally:
        adata_ro.file.close()

    qc_direct = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="direct",
    )

    assert isinstance(qc_wrapped, cx.AnnData)
    qc_wrapped_mem = qc_wrapped.to_memory()
    filtered_direct = qc_direct.filtered.to_memory()
    pd.testing.assert_index_equal(qc_wrapped_mem.obs.index, filtered_direct.obs.index)
    pd.testing.assert_index_equal(qc_wrapped_mem.var_names, filtered_direct.var_names)

    avg_wrapped = cx.pb.normalized_effects(
        qc_wrapped,
        perturbation_column="perturbation",
        control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_avg",
    )
    avg_direct = compute_normalized_effects(
        qc_direct.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_avg",
    )
    avg_wrapped_mem = avg_wrapped.to_memory()
    avg_direct_mem = avg_direct.to_memory()
    np.testing.assert_allclose(avg_wrapped_mem.X, avg_direct_mem.X)
    pd.testing.assert_index_equal(avg_wrapped_mem.obs.index, avg_direct_mem.obs.index)
    pd.testing.assert_index_equal(avg_wrapped_mem.var_names, avg_direct_mem.var_names)

    pseudo_wrapped = cx.pb.normalized_effects(
        qc_wrapped,
        perturbation_column="perturbation",
        control_label="ctrl",
        method="log_mean",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_pseudo",
    )
    pseudo_direct = compute_normalized_effects(
        qc_direct.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        method="log_mean",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_pseudo",
    )
    pseudo_wrapped_mem = pseudo_wrapped.to_memory()
    pseudo_direct_mem = pseudo_direct.to_memory()
    np.testing.assert_allclose(pseudo_wrapped_mem.X, pseudo_direct_mem.X)
    pd.testing.assert_index_equal(
        pseudo_wrapped_mem.obs.index, pseudo_direct_mem.obs.index
    )
    pd.testing.assert_index_equal(
        pseudo_wrapped_mem.var_names, pseudo_direct_mem.var_names
    )

    # Normalise QC-filtered data before t-test (which rejects count-like data)
    qc_norm = _log_normalise_sparse(qc_wrapped_mem)
    qc_norm_path = tmp_path / "qc_filtered_ttest_norm.h5ad"
    qc_norm.write(qc_norm_path)

    wald_wrapped = cx.tl.rank_genes_groups(
        qc_norm_path,
        perturbation_column="perturbation",
        method="t-test",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_wald",
        min_cells_expressed=0,
    )
    wald_direct = t_test(
        qc_norm_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_wald",
        min_cells_expressed=0,
    )
    # The wrapper returns t_test's own result file: the same layers, values
    # and per-gene pts_rest, not a rebuilt copy.
    wald_mem = wald_wrapped.to_memory()
    wrapped_groups = wald_mem.obs_names.tolist()
    assert wrapped_groups == list(wald_direct.groups)
    direct_effect = np.vstack([wald_direct[group].effect_size for group in wrapped_groups])
    direct_stat = np.vstack([wald_direct[group].statistic for group in wrapped_groups])
    direct_p = np.vstack([wald_direct[group].pvalue for group in wrapped_groups])
    np.testing.assert_array_equal(wald_mem.X, direct_effect)
    np.testing.assert_array_equal(wald_mem.layers["z_score"], direct_stat)
    np.testing.assert_array_equal(wald_mem.layers["pvalue"], direct_p)
    np.testing.assert_array_equal(wald_mem.layers["logfoldchanges"], wald_direct.logfoldchanges)
    np.testing.assert_array_equal(wald_mem.layers["pts"], wald_direct.pts)
    np.testing.assert_array_equal(wald_mem.var["pts_rest"].to_numpy(), wald_direct.pts_rest[0])
    assert (wald_mem.var["pts_rest"] > 0).any()

    wilcoxon_input = _log_normalise_sparse(qc_wrapped_mem)
    wilcoxon_path = tmp_path / "qc_filtered_wilcoxon_norm.h5ad"
    wilcoxon_input.write(wilcoxon_path)

    wilcoxon_wrapped = cx.tl.rank_genes_groups(
        wilcoxon_path,
        perturbation_column="perturbation",
        method="wilcoxon",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_wilcoxon",
        min_cells_expressed=0,
    )
    wilcoxon_direct = wilcoxon_test(
        wilcoxon_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_wilcoxon",
        min_cells_expressed=0,
    )
    # Wilcoxon uses layer-based storage for scalability with large group counts
    # (avoids recarray-based rank_genes_groups which hits HDF5 header limits)
    wrapped_groups = wilcoxon_wrapped.obs.index.tolist()
    direct_order = wilcoxon_direct.to_full_order_dict()
    # Access results from layers instead of uns["rank_genes_groups"]
    wilcoxon_mem = wilcoxon_wrapped.to_memory()
    np.testing.assert_allclose(wilcoxon_mem.layers["z_score"], direct_order["scores"])
    np.testing.assert_allclose(wilcoxon_mem.layers["pvalue"], direct_order["pvals"])
    assert set(wrapped_groups) == set(wilcoxon_direct.groups)
    avg_wrapped.close()
    avg_direct.close()
    pseudo_wrapped.close()
    pseudo_direct.close()
    wald_wrapped.close()
    wilcoxon_wrapped.close()
    qc_wrapped.close()
    qc_direct.filtered.close()


def test_empty_perturbation_group_error(tmp_path):
    """Test that empty perturbation groups raise helpful errors."""
    x = np.array([[1, 2], [3, 4]], dtype=float)
    obs = pd.DataFrame({"perturbation": ["ctrl", "ctrl"]})
    obs.index = ["cell_0", "cell_1"]
    var = pd.DataFrame(index=["gene0", "gene1"])
    adata = ad.AnnData(x, obs=obs, var=var)
    path = tmp_path / "empty_pert.h5ad"
    adata.write(path)

    with pytest.raises(ValueError, match="nonexistent"):
        wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            perturbations=["nonexistent"],
        )


def test_single_cell_perturbation(tmp_path):
    """A one-cell perturbation is still a valid rank-sum test."""
    rng = np.random.default_rng(456)
    x = rng.normal(5, 1, size=(11, 5))
    obs = pd.DataFrame({"perturbation": ["ctrl"] * 10 + ["KO1"]}, index=[f"cell_{i}" for i in range(11)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(5)])
    path = tmp_path / "single_cell.h5ad"
    ad.AnnData(sp.csr_matrix(x), obs=obs, var=var).write(path)

    result = wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        output_path=tmp_path / "single_result.h5ad",
        verbose=False,
    )
    assert result.groups == ["KO1"]
    expected = mannwhitneyu(x[10:], x[:10], method="asymptotic", use_continuity=False, axis=0).pvalue
    np.testing.assert_allclose(result.pvalues[0], expected, rtol=1e-10)


def test_all_zero_gene(tmp_path):
    """Test handling of genes with all-zero counts."""
    rng = np.random.default_rng(789)
    n_cells = 30
    perturbations = np.array(["ctrl"] * 15 + ["KO1"] * 15)
    counts = rng.poisson(5, size=(n_cells, 4))
    # Make one gene all zeros
    counts[:, 2] = 0
    obs = pd.DataFrame({"perturbation": perturbations})
    obs.index = [f"cell_{i}" for i in range(n_cells)]
    var = pd.DataFrame(index=["gene0", "gene1", "zero_gene", "gene3"])
    adata = ad.AnnData(counts, obs=obs, var=var)
    path = tmp_path / "zero_gene.h5ad"
    adata.write(path)

    result = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_cells_expressed=0,
        output_dir=tmp_path,
        data_name="zero",
    )
    # The all-zero gene cannot be fitted: untested, not "no change".
    assert np.isnan(result.pvalues[0, 2]) and np.isnan(result.logfoldchanges[0, 2])
    assert np.isfinite(result.pvalues[0, [0, 1, 3]]).all()
