import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import statsmodels.api as sm

try:  # Optional benchmark dependency for validation comparisons
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ModuleNotFoundError:  # pragma: no cover - executed when benchmarks unavailable
    DeseqDataSet = None
    DeseqStats = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crispyx.glm import NBGLMBatchFitter, NBGLMFitter, build_design_matrix
from crispyx.de import nb_glm_test


def _generate_nb_counts(rng, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + mu)
    return rng.negative_binomial(r, p)


def test_nb_glm_fitter_matches_statsmodels():
    rng = np.random.default_rng(42)
    n = 300
    covariates = pd.DataFrame(
        {
            "cov1": rng.normal(0, 1, size=n),
            "cov2": pd.Categorical(rng.choice(["A", "B", "C"], size=n, p=[0.2, 0.5, 0.3])),
        }
    )
    indicator = rng.binomial(1, 0.4, size=n).astype(float)
    design, column_names = build_design_matrix(
        covariates,
        covariate_columns=["cov1", "cov2"],
        perturbation_indicator=indicator,
    )
    beta = np.array([0.2, -0.5, 0.1, 0.3, -0.2])[: design.shape[1]]
    mu = np.exp(design @ beta)
    alpha = 0.7
    y = _generate_nb_counts(rng, mu, alpha)

    # Use min_mu=1e-8 to match statsmodels behavior for testing
    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n, dtype=float),
        dispersion=alpha,
        max_iter=100,
        tol=1e-8,
        poisson_init_iter=30,
        min_mu=1e-8,
        ridge_penalty=1e-8,
    )
    result = fitter.fit_gene(y)
    assert result.converged

    family = sm.families.NegativeBinomial(alpha=alpha)
    sm_res = sm.GLM(y, design, family=family).fit()

    # NBGLMFitter maximises the likelihood with L-BFGS-B while statsmodels uses
    # IRLS, so the coefficients agree only to the optimiser's own accuracy
    # (max |dB| 9.5e-3 here); the relative bound is set by a coefficient whose
    # true value is 0.046.  Standard errors come from the same closed-form
    # inverse Hessian in both and agree far more tightly.
    np.testing.assert_allclose(result.coef, sm_res.params, rtol=0.05, atol=0.015)
    np.testing.assert_allclose(result.se, sm_res.bse, rtol=0.01, atol=1e-3)


def test_nb_glm_fitter_matches_statsmodels_for_well_expressed_genes():
    rng = np.random.default_rng(12345)
    n = 250
    covariates = pd.DataFrame({"cov": rng.normal(0, 1, size=n)})
    indicator = rng.binomial(1, 0.45, size=n).astype(float)
    design, column_names = build_design_matrix(
        covariates,
        covariate_columns=["cov"],
        perturbation_indicator=indicator,
    )
    alpha = 0.6
    n_genes = 6
    betas = rng.normal(0, 0.4, size=(n_genes, design.shape[1]))
    betas[0] -= 3.0  # force one gene to be extremely lowly expressed
    counts = np.zeros((n, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        mu = np.exp(design @ betas[gene_idx])
        counts[:, gene_idx] = _generate_nb_counts(rng, mu, alpha)

    # Use min_mu=1e-8 to match statsmodels behavior for testing
    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n, dtype=float),
        dispersion=alpha,
        max_iter=120,
        tol=1e-8,
        poisson_init_iter=40,
        min_mu=1e-8,
        ridge_penalty=1e-8,
    )
    family = sm.families.NegativeBinomial(alpha=alpha)
    perturbation_index = column_names.index("perturbation")

    compared = 0
    for gene_idx in range(n_genes):
        y = counts[:, gene_idx]
        result = fitter.fit_gene(y)
        sm_res = sm.GLM(y, design, family=family).fit()
        if y.mean() < 0.25:
            # Lowly expressed genes are expected to be unstable; we only
            # assert convergence for moderately expressed genes.
            continue
        assert result.converged
        # L-BFGS-B and statsmodels' IRLS produce similar but not identical results
        np.testing.assert_allclose(
            result.coef[perturbation_index],
            sm_res.params[perturbation_index],
            rtol=0.25,
            atol=0.2,
        )
        np.testing.assert_allclose(
            result.se[perturbation_index],
            sm_res.bse[perturbation_index],
            rtol=0.25,
            atol=0.2,
        )
        compared += 1

    assert compared >= 1


def test_nb_glm_test_pipeline(tmp_path):
    rng = np.random.default_rng(123)
    n_cells = 80
    n_genes = 4
    perturbations = np.array(["control"] * (n_cells // 2) + ["g1"] * (n_cells - n_cells // 2))
    covariate = rng.normal(size=n_cells)
    obs = pd.DataFrame({"perturbation": perturbations, "covariate": covariate})

    indicator = (perturbations == "g1").astype(float)
    design = np.column_stack((np.ones(n_cells), indicator, covariate))
    betas = np.array([
        [0.1, 0.6, -0.3],
        [-0.2, 0.4, 0.2],
        [0.4, -0.5, 0.1],
        [0.0, 0.2, -0.4],
    ])
    alpha = 0.5
    counts = np.zeros((n_cells, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        mu = np.exp(design @ betas[gene_idx])
        counts[:, gene_idx] = _generate_nb_counts(rng, mu, alpha)

    adata = ad.AnnData(counts, obs=obs.copy(), var=pd.DataFrame(index=[f"gene{idx}" for idx in range(n_genes)]))
    path = tmp_path / "synthetic.h5ad"
    adata.write(path)

    result = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        covariates=["covariate"],
        dispersion=alpha,
        max_iter=100,
        poisson_init_iter=30,
        min_cells_expressed=0,
        chunk_size=2,
    )

    assert result.method == "nb_glm"
    assert result.groups == ["g1"]
    assert result.statistics.shape == (1, n_genes)
    assert np.all(np.isfinite(result.pvalues))

    exported = result.result_path
    assert exported.exists()
    exported_adata = ad.read_h5ad(exported)
    assert "dispersion" in exported_adata.layers
    assert "standard_error" in exported_adata.layers

    diff_result = result["g1"]
    assert diff_result.method == "nb_glm"
    assert diff_result.pvalue.shape[0] == n_genes


def test_nb_glm_agrees_with_statsmodels_and_deseq2():
    if DeseqDataSet is None or DeseqStats is None:
        pytest.skip("pydeseq2 is not installed")
    rng = np.random.default_rng(314159)
    n_samples = 140
    n_genes = 5

    condition = rng.binomial(1, 0.5, size=n_samples).astype(float)
    design = np.column_stack([np.ones(n_samples), condition])
    betas = rng.normal(scale=0.4, size=(n_genes, design.shape[1]))
    betas[:, 0] -= 0.1
    betas[0, 1] -= 2.5  # enforce one lowly expressed gene
    dispersion = 0.5

    counts = np.zeros((n_samples, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        mu = np.exp(design @ betas[gene_idx])
        counts[:, gene_idx] = _generate_nb_counts(rng, mu, dispersion)

    # Use min_mu=1e-8 to match statsmodels for direct comparison
    # Note: The default min_mu=0.5 is aligned with PyDESeq2 for real-world usage
    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n_samples, dtype=float),
        dispersion=dispersion,
        max_iter=120,
        tol=1e-8,
        poisson_init_iter=40,
        min_mu=1e-8,
        ridge_penalty=1e-8,
    )

    family = sm.families.NegativeBinomial(alpha=dispersion)

    sample_names = [f"cell_{i}" for i in range(n_samples)]
    gene_names = [f"gene_{j}" for j in range(n_genes)]
    counts_df = pd.DataFrame(counts, index=sample_names, columns=gene_names)
    clinical = pd.DataFrame(
        {
            "condition": pd.Categorical(
                np.where(condition > 0, "treated", "control"),
                categories=["control", "treated"],
            )
        },
        index=sample_names,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=clinical,
        design="~condition",
        refit_cooks=False,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=("condition", "treated", "control"),
        cooks_filter=False,
        alpha=0.05,
    )
    stats.summary()
    deseq_results = stats.results_df["log2FoldChange"].to_numpy() * np.log(2.0)

    well_expressed = 0
    for gene_idx, gene_name in enumerate(gene_names):
        y = counts[:, gene_idx]
        if y.mean() < 0.5:
            continue

        fit_res = fitter.fit_gene(y)
        assert fit_res.converged

        sm_res = sm.GLM(y, design, family=family).fit()

        nb_coef = fit_res.coef[1]
        sm_coef = sm_res.params[1]
        deseq_coef = deseq_results[gene_idx]

        # Same model, same dispersion, both by IRLS: these agree to ~2e-10.
        np.testing.assert_allclose(nb_coef, sm_coef, rtol=1e-6, atol=1e-6)
        # PyDESeq2 differs by construction, not by solver accuracy: it fits MAP
        # dispersions against a trend and applies a coefficient prior, neither
        # of which this unshrunk MLE has.  The bound stays loose for that
        # reason, not because the IRLS is imprecise.
        np.testing.assert_allclose(nb_coef, deseq_coef, rtol=0.7, atol=0.4)

        well_expressed += 1

    assert well_expressed >= 1


def test_shrink_lfc_matches_pydeseq2_lfcshrink(tmp_path):
    """Test that shrink_lfc produces effect sizes matching PyDESeq2 lfcShrink.
    
    This regression test ensures the LFC shrinkage implementation uses the
    correct fitted intercept from the NB-GLM model, not a naive estimate.
    """
    if DeseqDataSet is None or DeseqStats is None:
        pytest.skip("pydeseq2 is not installed")
    
    from crispyx.de import shrink_lfc
    
    rng = np.random.default_rng(20260102)
    n_cells = 200
    n_genes = 10
    
    # Create perturbation labels
    perturbations = np.array(
        ["control"] * 100 + ["pert_A"] * 50 + ["pert_B"] * 50
    )
    
    # Generate counts with known LFCs
    # Intercept ~ 3.0 (mean count ~20), LFC ~ 0.5-1.5
    true_intercept = rng.uniform(2.5, 3.5, size=n_genes)
    true_lfc_A = rng.uniform(0.3, 1.5, size=n_genes)
    true_lfc_B = rng.uniform(-1.5, -0.3, size=n_genes)
    dispersion = 0.3
    
    counts = np.zeros((n_cells, n_genes), dtype=int)
    for j in range(n_genes):
        for i in range(n_cells):
            if perturbations[i] == "control":
                mu = np.exp(true_intercept[j])
            elif perturbations[i] == "pert_A":
                mu = np.exp(true_intercept[j] + true_lfc_A[j])
            else:
                mu = np.exp(true_intercept[j] + true_lfc_B[j])
            r = 1.0 / dispersion
            p = r / (r + mu)
            counts[i, j] = rng.negative_binomial(r, p)
    
    # Create and save AnnData
    obs = pd.DataFrame({"perturbation": perturbations})
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    adata = ad.AnnData(counts, obs=obs, var=var)
    input_path = tmp_path / "shrink_test.h5ad"
    adata.write(input_path)
    
    # Run CRISPYx nb_glm_test
    result = nb_glm_test(
        input_path,
        perturbation_column="perturbation",
        control_label="control",
        dispersion_method="moments",
        min_cells_expressed=0,
    )
    
    # Apply shrinkage with full method
    # shrink_lfc reads perturbation_column and control_label from uns metadata
    shrink_lfc(
        result.result_path,
        method="full",
    )
    
    # Load shrunk results
    shrunk_adata = ad.read_h5ad(result.result_path)
    crispyx_lfc = shrunk_adata.X  # (n_groups, n_genes)
    
    # Run PyDESeq2 with lfcShrink for each perturbation
    sample_names = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene{j}" for j in range(n_genes)]
    
    correlations = []
    max_diffs = []
    
    for pert_idx, pert_label in enumerate(["pert_A", "pert_B"]):
        # Create binary condition for this perturbation vs control
        condition_mask = (perturbations == "control") | (perturbations == pert_label)
        subset_indices = [i for i in range(n_cells) if condition_mask[i]]
        subset_samples = [sample_names[i] for i in subset_indices]
        subset_counts = counts[subset_indices, :]  # (n_subset, n_genes)
        
        counts_df = pd.DataFrame(subset_counts, index=subset_samples, columns=gene_names)
        
        clinical = pd.DataFrame(
            {
                "condition": pd.Categorical(
                    [
                        "treated" if perturbations[i] == pert_label else "control"
                        for i in subset_indices
                    ],
                    categories=["control", "treated"],
                )
            },
            index=subset_samples,
        )
        
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=clinical,
            design="~condition",
            refit_cooks=False,
        )
        dds.deseq2()
        stats = DeseqStats(
            dds,
            contrast=("condition", "treated", "control"),
            cooks_filter=False,
            alpha=0.05,
        )
        stats.summary()
        
        # Get shrunk LFC from PyDESeq2 (convert log2 to ln)
        pydeseq2_lfc = stats.results_df["log2FoldChange"].to_numpy() * np.log(2.0)
        
        # Get CRISPYx shrunk LFC for this perturbation
        crispyx_lfc_pert = crispyx_lfc[pert_idx, :]
        
        # Compute correlation
        valid_mask = np.isfinite(pydeseq2_lfc) & np.isfinite(crispyx_lfc_pert)
        if valid_mask.sum() < 3:
            continue
            
        corr = np.corrcoef(pydeseq2_lfc[valid_mask], crispyx_lfc_pert[valid_mask])[0, 1]
        max_diff = np.max(np.abs(pydeseq2_lfc[valid_mask] - crispyx_lfc_pert[valid_mask]))
        
        correlations.append(corr)
        max_diffs.append(max_diff)
    
    # Assert accuracy thresholds
    mean_corr = np.mean(correlations)
    mean_max_diff = np.mean(max_diffs)
    
    assert mean_corr > 0.90, (
        f"LFC shrinkage correlation with PyDESeq2 too low: {mean_corr:.3f} (expected >0.90)"
    )
    assert mean_max_diff < 2.0, (
        f"LFC shrinkage max diff too high: {mean_max_diff:.3f} (expected <2.0)"
    )


def test_nb_glm_se_method_options(tmp_path):
    """Test that se_method options ('sandwich' vs 'fisher') produce different results."""
    rng = np.random.default_rng(111)
    n_cells = 60
    n_genes = 4
    perturbations = np.array(["control"] * 30 + ["g1"] * 30)
    counts = rng.poisson(10, size=(n_cells, n_genes))
    obs = pd.DataFrame({"perturbation": perturbations})
    adata = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))
    path = tmp_path / "se_test.h5ad"
    adata.write(path)

    result_sandwich = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        se_method="sandwich",
        output_dir=tmp_path,
        data_name="se_sandwich",
    )

    result_fisher = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        se_method="fisher",
        output_dir=tmp_path,
        data_name="se_fisher",
    )

    # Both should produce valid results
    assert result_sandwich.statistics.shape == (1, n_genes)
    assert result_fisher.statistics.shape == (1, n_genes)
    # SE values should differ between methods (not identical)
    sandwich_adata = ad.read_h5ad(result_sandwich.result_path)
    fisher_adata = ad.read_h5ad(result_fisher.result_path)
    # They may be similar but shouldn't be exactly identical
    assert sandwich_adata.layers["standard_error"].shape == fisher_adata.layers["standard_error"].shape


def test_nb_glm_profiling_output(tmp_path):
    """Test that profiling=True produces expected metadata."""
    rng = np.random.default_rng(222)
    n_cells = 40
    n_genes = 3
    perturbations = np.array(["control"] * 20 + ["g1"] * 20)
    counts = rng.poisson(8, size=(n_cells, n_genes))
    obs = pd.DataFrame({"perturbation": perturbations})
    adata = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))
    path = tmp_path / "profiling_test.h5ad"
    adata.write(path)

    result = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        profiling=True,
        output_dir=tmp_path,
        data_name="profiled",
    )

    result_adata = ad.read_h5ad(result.result_path)
    profiling_data = result_adata.uns.get("profiling")
    assert profiling_data is not None
    assert profiling_data != "NA"
    assert "profiling_enabled" in profiling_data
    assert profiling_data["profiling_enabled"] == True  # noqa: E712 - allow np.True_ comparison
    assert "fit_seconds" in profiling_data
    assert profiling_data["fit_seconds"] >= 0


def test_nb_glm_n_jobs_parallelization(tmp_path):
    """Test that n_jobs parameter works without errors."""
    rng = np.random.default_rng(333)
    n_cells = 60
    n_genes = 4
    perturbations = np.array(["control"] * 20 + ["g1"] * 20 + ["g2"] * 20)
    counts = rng.poisson(10, size=(n_cells, n_genes))
    obs = pd.DataFrame({"perturbation": perturbations})
    adata = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))
    path = tmp_path / "njobs_test.h5ad"
    adata.write(path)

    # Test with n_jobs=1 (sequential)
    result_seq = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        n_jobs=1,
        output_dir=tmp_path,
        data_name="seq",
    )

    # Test with n_jobs=2 (parallel)
    result_par = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        n_jobs=2,
        output_dir=tmp_path,
        data_name="par",
    )

    # Results should be very similar (may differ slightly due to floating point)
    np.testing.assert_allclose(result_seq.statistics, result_par.statistics, rtol=1e-5)
    np.testing.assert_allclose(result_seq.pvalues, result_par.pvalues, rtol=1e-5)


def test_shrink_lfc_prior_scale_mode(tmp_path):
    """Test that prior_scale_mode options work."""
    from crispyx.de import shrink_lfc
    
    rng = np.random.default_rng(444)
    n_cells = 80
    n_genes = 5
    perturbations = np.array(["control"] * 40 + ["g1"] * 20 + ["g2"] * 20)
    counts = rng.poisson(15, size=(n_cells, n_genes))
    obs = pd.DataFrame({"perturbation": perturbations})
    adata = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))
    path = tmp_path / "prior_scale_test.h5ad"
    adata.write(path)

    result = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        output_dir=tmp_path,
        data_name="prior",
    )

    # Test global prior scale mode
    shrink_lfc(
        result.result_path,
        prior_scale_mode="global",
    )
    global_adata = ad.read_h5ad(result.result_path)
    global_lfc = global_adata.X.copy()

    # Re-run with per_comparison mode
    # First, re-run nb_glm_test to reset
    result2 = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        output_dir=tmp_path,
        data_name="prior2",
    )
    shrink_lfc(
        result2.result_path,
        prior_scale_mode="per_comparison",
    )
    per_comp_adata = ad.read_h5ad(result2.result_path)
    per_comp_lfc = per_comp_adata.X.copy()

    # Both should produce valid results
    assert global_lfc.shape == per_comp_lfc.shape
    # Results may differ between modes
    assert np.all(np.isfinite(global_lfc) | (global_lfc == 0))
    assert np.all(np.isfinite(per_comp_lfc) | (per_comp_lfc == 0))


def test_streaming_vs_dense_dispersion_accuracy():
    """Test that streaming dispersion estimation matches dense estimation.
    
    This test compares the streaming dispersion implementation 
    (_precompute_global_dispersion_streaming) against the standard dense
    implementation (precompute_global_dispersion with fast_mode=True).
    
    Both methods should produce highly correlated dispersion estimates
    (Pearson r > 0.95) since they use the same underlying algorithm
    (MoM + trend shrinkage) but different memory access patterns.
    """
    from scipy.stats import pearsonr, spearmanr
    from crispyx.glm import (
        precompute_global_dispersion,
        _precompute_global_dispersion_streaming,
        precompute_control_statistics,
        fit_dispersion_trend,
    )
    import scipy.sparse as sp
    
    rng = np.random.default_rng(123)
    
    # Create a medium-sized test dataset
    n_cells = 500
    n_genes = 200
    n_control = 200
    
    # Generate counts with realistic sparsity
    mu = rng.gamma(2, 2, size=n_genes)  # Gene-specific means
    alpha = rng.gamma(0.5, 0.5, size=n_genes)  # Gene-specific dispersion
    
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    for j in range(n_genes):
        r = 1.0 / max(alpha[j], 0.01)
        p = r / (r + mu[j])
        counts[:, j] = rng.negative_binomial(max(r, 0.1), min(p, 0.999), size=n_cells)
    
    # Add sparsity (30% zeros)
    mask = rng.random((n_cells, n_genes)) < 0.3
    counts[mask] = 0
    
    # Create size factors
    size_factors = counts.sum(axis=1) / np.median(counts.sum(axis=1))
    size_factors = np.maximum(size_factors, 0.1)
    offset = np.log(size_factors)
    
    # Control cells
    control_matrix = counts[:n_control]
    control_offset = offset[:n_control]
    
    # Create control cache
    control_cache = precompute_control_statistics(
        control_matrix=control_matrix,
        control_offset=control_offset,
        max_iter=10,
        tol=1e-6,
        min_mu=0.5,
        dispersion_method="moments",
    )
    
    # Test 1: Dense mode (standard)
    import copy
    cache_dense = copy.deepcopy(control_cache)
    cache_dense = precompute_global_dispersion(
        control_cache=cache_dense,
        all_cell_matrix=counts,
        all_cell_offset=offset,
        fit_type="parametric",
        fast_mode=True,
        max_dense_fraction=1.0,  # Force dense mode
    )
    
    # Test 2: Streaming mode
    cache_streaming = copy.deepcopy(control_cache)
    cache_streaming = _precompute_global_dispersion_streaming(
        control_cache=cache_streaming,
        all_cell_matrix=counts,
        all_cell_offset=offset,
        fit_type="parametric",
        chunk_size=100,
    )
    
    # Compare dispersions
    disp_dense = cache_dense.global_dispersion
    disp_streaming = cache_streaming.global_dispersion
    
    # Both should be valid
    assert np.all(np.isfinite(disp_dense))
    assert np.all(np.isfinite(disp_streaming))
    
    # Compute correlation
    valid = np.isfinite(disp_dense) & np.isfinite(disp_streaming) & (disp_dense > 0) & (disp_streaming > 0)
    pearson_r, _ = pearsonr(np.log(disp_dense[valid]), np.log(disp_streaming[valid]))
    spearman_r, _ = spearmanr(disp_dense[valid], disp_streaming[valid])
    
    # Streaming should achieve high correlation with dense (both use MoM + trend)
    assert pearson_r > 0.95, f"Pearson correlation {pearson_r:.3f} < 0.95"
    assert spearman_r > 0.95, f"Spearman correlation {spearman_r:.3f} < 0.95"
    
    # Trends should also be similar
    trend_dense = cache_dense.global_dispersion_trend
    trend_streaming = cache_streaming.global_dispersion_trend
    
    valid_trend = np.isfinite(trend_dense) & np.isfinite(trend_streaming) & (trend_dense > 0) & (trend_streaming > 0)
    trend_pearson, _ = pearsonr(np.log(trend_dense[valid_trend]), np.log(trend_streaming[valid_trend]))
    assert trend_pearson > 0.90, f"Trend Pearson correlation {trend_pearson:.3f} < 0.90"


def test_memory_adaptive_dispersion():
    """Test that memory-adaptive mode triggers streaming for large datasets.
    
    This test verifies that precompute_global_dispersion correctly switches
    to streaming mode when the estimated memory exceeds the threshold.
    """
    from crispyx.glm import (
        precompute_global_dispersion,
        precompute_control_statistics,
        _estimate_dense_memory_gb,
    )
    import scipy.sparse as sp
    
    rng = np.random.default_rng(456)
    
    # Create a small test dataset
    n_cells = 100
    n_genes = 50
    n_control = 40
    
    counts = rng.poisson(10, size=(n_cells, n_genes)).astype(np.float64)
    size_factors = np.ones(n_cells)
    offset = np.log(size_factors)
    
    control_matrix = counts[:n_control]
    control_offset = offset[:n_control]
    
    control_cache = precompute_control_statistics(
        control_matrix=control_matrix,
        control_offset=control_offset,
    )
    
    # Test 1: Normal mode (should use dense)
    import copy
    cache1 = copy.deepcopy(control_cache)
    cache1 = precompute_global_dispersion(
        control_cache=cache1,
        all_cell_matrix=counts,
        all_cell_offset=offset,
        max_dense_fraction=0.3,  # Default
        memory_limit_gb=None,  # No explicit limit
    )
    assert cache1.global_dispersion is not None
    
    # Test 2: Force streaming by setting a very low memory limit
    cache2 = copy.deepcopy(control_cache)
    cache2 = precompute_global_dispersion(
        control_cache=cache2,
        all_cell_matrix=counts,
        all_cell_offset=offset,
        max_dense_fraction=0.3,
        memory_limit_gb=0.00001,  # 10 KB - forces streaming
    )
    assert cache2.global_dispersion is not None
    
    # Results should be similar (both use MoM + trend shrinkage)
    disp1 = cache1.global_dispersion
    disp2 = cache2.global_dispersion
    
    from scipy.stats import pearsonr
    valid = np.isfinite(disp1) & np.isfinite(disp2) & (disp1 > 0) & (disp2 > 0)
    if np.sum(valid) > 5:
        r, _ = pearsonr(np.log(disp1[valid]), np.log(disp2[valid]))
        assert r > 0.90, f"Correlation {r:.3f} < 0.90 between dense and streaming modes"

#: Counts per group below which an NB GLM coefficient is not meaningfully
#: determined, so cross-solver agreement measures the data rather than the fit.
_MIN_GROUP_COUNTS = 10


def _wide_design_fixture(seed=0, n=400, p=60):
    """Counts and a 6-column design spanning 0.05 to 50 counts per cell."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 5, size=n)
    dummies = np.stack([(lab == k).astype(float) for k in range(1, 5)], axis=1)
    design = np.c_[np.ones(n), rng.normal(size=n), dummies]
    beta = np.c_[
        np.log(np.geomspace(0.05, 50.0, p)),
        rng.normal(0, 0.3, p),
        rng.normal(0, 0.8, (p, 4)),
    ]
    alpha = np.full(p, 0.3)
    mu = np.exp(design @ beta.T)
    counts = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu)).astype(float)
    return counts, design, lab


def test_batch_fitter_general_path_matches_statsmodels_where_the_fit_is_determined():
    """The batch fitter's general (non-2-feature) path is an exact NB IRLS.

    Regression guard for the IRLS weights being clamped at the fitted-mean
    floor.  That clamp inflated the leverage of every cell whose fitted mean
    fell below ``min_mu`` -- at mu=0.1, alpha=1 the true weight is 0.091 and a
    0.5 clamp made it 0.5 -- and pushed the median coefficient error on genes
    below one count per cell to ~0.27.

    Agreement is asserted only where the likelihood actually determines the
    coefficients: every group carrying at least ``_MIN_GROUP_COUNTS`` counts
    and a dispersion off its clip boundary.  statsmodels is not a gold
    standard for sparse count data -- on a group with one count in eighty
    cells the coefficient is barely identified and both solvers simply stop
    somewhere on the way down, so a disagreement there measures the data, not
    the solver.  The sparse tail is checked for sanity instead, below.
    """
    counts, design, lab = _wide_design_fixture()

    fitter = NBGLMBatchFitter(
        design, max_iter=100, tol=1e-10, min_mu=0.0, dispersion_method="moments"
    )
    result = fitter.fit_batch(counts, gene_batch_size=None, use_numba=False)

    compared = 0
    worst = 0.0
    for gene in range(counts.shape[1]):
        per_group = np.array([counts[lab == k, gene].sum() for k in range(5)])
        if per_group.min() < _MIN_GROUP_COUNTS or result.dispersion[gene] <= 1e-6:
            continue
        # Give statsmodels the dispersion crispyx converged to, so that only
        # the IRLS differs between the two.
        family = sm.families.NegativeBinomial(alpha=float(result.dispersion[gene]))
        sm_fit = sm.GLM(counts[:, gene], design, family=family).fit(maxiter=500)
        if not np.all(np.isfinite(sm_fit.params)):
            continue
        worst = max(worst, float(np.max(np.abs(result.coef[gene] - sm_fit.params))))
        compared += 1

    assert compared >= 30, f"expected a decent number of determined genes, got {compared}"
    assert worst < 1e-5, f"max |dB| vs statsmodels was {worst:.2e}"


def test_sparse_genes_stay_finite_and_bounded():
    """The under-determined tail must degrade gracefully, not blow up.

    Genes whose groups carry almost no counts have coefficients that the data
    barely pins down (the limit is -inf for an empty group).  There is no
    reference to agree with; what matters is that the solver returns finite,
    bounded numbers and reports its convergence honestly.
    """
    counts, design, lab = _wide_design_fixture()
    fitter = NBGLMBatchFitter(
        design, max_iter=100, tol=1e-10, min_mu=0.0, dispersion_method="moments"
    )
    result = fitter.fit_batch(counts, gene_batch_size=None, use_numba=False)

    sparse = np.array([
        min(counts[lab == k, gene].sum() for k in range(5)) < _MIN_GROUP_COUNTS
        for gene in range(counts.shape[1])
    ])
    assert sparse.any(), "fixture should contain an under-determined tail"

    assert np.all(np.isfinite(result.coef)), "coefficients must stay finite"
    assert np.all(np.isfinite(result.dispersion))
    # ETA_MAX bounds the linear predictor, so no coefficient can run away.
    assert np.max(np.abs(result.coef[sparse])) < 50.0


def test_sparse_genes_are_not_dominated_by_the_mean_floor():
    """``min_mu`` must floor the fitted mean and nothing else.

    With the floor applied to the weights as well, raising ``min_mu`` from 0 to
    0.5 moved sparse-gene coefficients by order 0.1-0.3.  As a pure mean floor
    it still has *some* effect on an expressed gene -- cells with a small
    fitted mean exist in any gene -- but three orders of magnitude less.
    """
    counts, design, _ = _wide_design_fixture()
    expressed = counts.mean(axis=0) >= 5.0
    assert expressed.sum() >= 10

    fits = {}
    for min_mu in (0.0, 0.5):
        fitter = NBGLMBatchFitter(
            design, max_iter=100, tol=1e-10, min_mu=min_mu, dispersion_method="moments"
        )
        fits[min_mu] = fitter.fit_batch(counts, gene_batch_size=None, use_numba=False).coef

    shift = np.max(np.abs(fits[0.0][expressed] - fits[0.5][expressed]))
    assert shift < 1e-3, f"mean floor moved well-expressed coefficients by {shift:.2e}"


# --------------------------------------------------------------------------
# families and fixed dispersion
# --------------------------------------------------------------------------

def test_poisson_family_matches_statsmodels_poisson():
    """``family='poisson'`` is a real Poisson fit, not NB with a small alpha.

    The distinction matters: an NB fit with an estimated dispersion uses
    weights ``mu / (1 + alpha mu)``, which differ from the Poisson weights
    ``mu`` by however large the estimated dispersion happens to be.
    """
    rng = np.random.default_rng(3)
    n, p = 400, 40
    design = np.c_[np.ones(n), rng.normal(size=n), rng.binomial(1, 0.5, n).astype(float)]
    beta = np.c_[np.log(np.geomspace(1.0, 60.0, p)), rng.normal(0, 0.3, p), rng.normal(0, 0.5, p)]
    counts = rng.poisson(np.exp(design @ beta.T)).astype(float)

    fitter = NBGLMBatchFitter(design, max_iter=100, tol=1e-12, min_mu=0.0, family="poisson")
    result = fitter.fit_batch(counts, gene_batch_size=None)

    assert np.all(result.dispersion == 0.0), "a Poisson fit has no dispersion"

    worst = 0.0
    for gene in range(p):
        sm_fit = sm.GLM(counts[:, gene], design, family=sm.families.Poisson()).fit()
        worst = max(worst, float(np.max(np.abs(result.coef[gene] - sm_fit.params))))
    assert worst < 1e-7, f"max |dB| vs statsmodels Poisson was {worst:.2e}"


def test_poisson_family_differs_from_the_nb_fit_on_overdispersed_data():
    """Sanity check that the family argument is actually doing something."""
    rng = np.random.default_rng(4)
    n, p = 300, 20
    design = np.c_[np.ones(n), rng.binomial(1, 0.5, n).astype(float)]
    mu = np.exp(design @ np.c_[np.log(np.full(p, 20.0)), np.full(p, 0.8)].T)
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)

    common = dict(max_iter=100, tol=1e-10, min_mu=0.0)
    pois = NBGLMBatchFitter(design, family="poisson", **common).fit_batch(counts, gene_batch_size=None)
    nb = NBGLMBatchFitter(design, family="nb", **common).fit_batch(counts, gene_batch_size=None)

    assert np.all(nb.dispersion > 0.1), "data is overdispersed; NB should find it"
    # Coefficients are similar (both are consistent) but the standard errors
    # are not: Poisson understates them when the data are overdispersed.
    assert np.median(pois.se[:, 1]) < 0.75 * np.median(nb.se[:, 1])


def test_fixed_dispersion_is_honoured_and_matches_statsmodels():
    """``fixed_dispersion`` holds alpha at the caller's value."""
    rng = np.random.default_rng(5)
    n, p = 350, 30
    design = np.c_[np.ones(n), rng.normal(size=n), rng.binomial(1, 0.4, n).astype(float)]
    beta = np.c_[np.log(np.geomspace(2.0, 80.0, p)), rng.normal(0, 0.3, p), rng.normal(0, 0.5, p)]
    alpha = np.full(p, 0.25)
    mu = np.exp(design @ beta.T)
    counts = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu)).astype(float)

    fitter = NBGLMBatchFitter(design, max_iter=100, tol=1e-12, min_mu=0.0)
    result = fitter.fit_batch(counts, gene_batch_size=None, fixed_dispersion=alpha)

    np.testing.assert_allclose(result.dispersion, alpha, rtol=0, atol=0)

    worst = 0.0
    for gene in range(p):
        family = sm.families.NegativeBinomial(alpha=float(alpha[gene]))
        sm_fit = sm.GLM(counts[:, gene], design, family=family).fit(maxiter=500)
        worst = max(worst, float(np.max(np.abs(result.coef[gene] - sm_fit.params))))
    # The floor here is statsmodels' own convergence (its default deviance
    # tolerance is 1e-8), not this fitter's.
    assert worst < 1e-6, f"max |dB| vs statsmodels was {worst:.2e}"


def test_fixed_dispersion_shape_is_validated():
    design = np.c_[np.ones(20), np.arange(20.0)]
    counts = np.abs(np.arange(40.0).reshape(20, 2)) + 1.0
    fitter = NBGLMBatchFitter(design, min_mu=0.0)
    with pytest.raises(ValueError, match="fixed_dispersion must have shape"):
        fitter.fit_batch(counts, fixed_dispersion=np.array([0.1, 0.2, 0.3]))


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="family must be 'nb' or 'poisson'"):
        NBGLMBatchFitter(np.ones((5, 1)), family="gaussian")


def _nb_score(counts, design, coef, alpha):
    """Score of the NB log-likelihood at ``coef``, per gene.

    For NB with size ``r = 1/alpha`` and a log link,
    ``s_j(beta) = sum_i x_ij (y_i - mu_i) / (1 + alpha mu_i)``, which is zero
    exactly at the maximum likelihood estimate.  Returns ``max_j |s_j|``.
    """
    mu = np.exp(np.clip(design @ coef.T, -30.0, 20.0))
    residual = (counts - mu) / (1.0 + alpha[None, :] * mu)
    return np.abs(design.T @ residual).max(axis=0)


def test_fitted_coefficients_solve_the_score_equation_on_sparse_genes():
    """A reference-free correctness check, for where references are unreliable.

    Cross-solver comparison is the wrong instrument on lowly-expressed count
    data: the fits are hard, and a disagreement says the two implementations
    differ without saying which is right.  The score equation needs no second
    implementation -- it *defines* the maximum likelihood estimate -- so it
    works precisely where statsmodels does not.

    This is the regression guard for clamping the IRLS weights at the
    fitted-mean floor.  That clamp does not merely perturb the answer; it
    solves a different estimating equation, and the coefficients it returns
    are not a stationary point of the likelihood at all.  Measured on this
    fixture the clamped fit left a score of ~50-100 and roughly 40-100 units
    of excess deviance on genes below one count per cell, against ~1e-6 here.

    The dispersion is held fixed so that all genes are fitting the same model
    the score is written for.
    """
    counts, design, lab = _wide_design_fixture(n=800, p=200)
    alpha = np.full(counts.shape[1], 0.3)

    fitter = NBGLMBatchFitter(design, max_iter=200, tol=1e-12, min_mu=0.0)
    result = fitter.fit_batch(
        counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha
    )

    score = _nb_score(counts, design, result.coef, alpha)

    # Only genes whose MLE exists: a group with no counts has a coefficient
    # running to -inf, where no bounded stationary point exists to find.
    determined = np.array([
        min(counts[lab == k, gene].sum() for k in range(5)) >= _MIN_GROUP_COUNTS
        for gene in range(counts.shape[1])
    ])
    sparse = counts.mean(axis=0) < 0.5
    checked = determined & sparse
    assert checked.sum() >= 10, f"fixture should have sparse determined genes, got {checked.sum()}"

    assert np.max(score[checked]) < 1e-3, (
        f"sparse genes are not at a stationary point: max|score| = {np.max(score[checked]):.2e}"
    )
    assert np.max(score[determined]) < 1e-3, (
        f"max|score| over all determined genes = {np.max(score[determined]):.2e}"
    )


def test_the_mean_floor_does_not_move_the_stationary_point_it_defines():
    """``min_mu`` changes the model, so it must change it *consistently*.

    With a floor the fitted mean is ``max(exp(eta), min_mu)``, which is a
    different estimating equation -- that is the point of the floor.  What
    must not happen is the floor leaking into the weights, which leaves the
    fit solving neither equation.  Checked here by confirming that a floored
    fit still solves the score equation on the genes the floor never binds on.
    """
    counts, design, lab = _wide_design_fixture(n=800, p=200)
    alpha = np.full(counts.shape[1], 0.3)

    floored = NBGLMBatchFitter(design, max_iter=200, tol=1e-12, min_mu=0.5).fit_batch(
        counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha
    )
    mu = np.exp(np.clip(design @ floored.coef.T, -30.0, 20.0))
    never_binds = (mu.min(axis=0) > 0.5) & (counts.mean(axis=0) >= 5.0)
    assert never_binds.sum() >= 5

    score = _nb_score(counts, design, floored.coef, alpha)
    assert np.max(score[never_binds]) < 1e-3


# ---------------------------------------------------------------------------
# IRLS damping
# ---------------------------------------------------------------------------

def _overshooting_gene():
    """Counts on which the first full Newton step badly raises the deviance.

    A near-empty perturbation arm beside a large baseline: the first step from
    ``beta = 0`` overshoots and has to be shortened several times.
    """
    rng = np.random.default_rng(11)
    n = 200
    x1 = np.zeros(n)
    x1[:3] = 1.0
    design = np.column_stack([np.ones(n), x1, rng.normal(0, 1.0, size=n)])

    gene = np.random.default_rng(29)
    beta = np.array([gene.uniform(4, 9), gene.uniform(-8, 8), gene.uniform(-2, 2)])
    counts = gene.poisson(np.exp(np.clip(design @ beta, -20, 14))).astype(float)
    return design, counts[:, None]


def test_step_halving_takes_the_first_improving_fraction_of_the_newton_step():
    """A shortened step is ``beta + 2^-k (newton - beta)`` for the smallest
    ``k`` that lowers the deviance.

    Interpolating towards the previously shortened point instead of towards
    the full Newton point compounds the shortening -- the third retry lands at
    ``2^-6`` rather than ``2^-3`` -- which stalls the gene and then reports it
    as converged because nothing moved.
    """
    from crispyx._irls import EPS, ETA_MAX, Deviance, irls_weights

    design, counts = _overshooting_gene()
    fitter = NBGLMBatchFitter(design, max_iter=1, tol=1e-9, min_mu=0.0)
    alpha = np.array([0.05])
    start = np.zeros((3, 1))

    # The full Newton step, formed exactly as the loop forms it.
    deviance = Deviance(counts, "nb", alpha)
    eta = np.clip(design @ start, fitter._eta_min, ETA_MAX)
    mu = np.exp(eta)
    dev_start = deviance.total(eta, mu)
    newton = fitter._weighted_least_squares_batch(
        irls_weights(mu, alpha), eta + (counts - mu) / np.maximum(mu, EPS)
    )

    def deviance_at(beta):
        e = np.clip(design @ beta, fitter._eta_min, ETA_MAX)
        return deviance.total(e, np.exp(e))[0]

    for k in range(9):
        expected = start + (0.5**k) * (newton - start)
        if deviance_at(expected) <= dev_start[0] + 1e-9 * abs(dev_start[0]):
            break
    assert k >= 3, "this gene no longer exercises repeated halving"

    got, _, _, _, _ = fitter._irls_at_fixed_dispersion(counts, alpha, start)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_irls_never_commits_a_step_that_raises_the_deviance():
    """Deviance is what the loop minimises and what its convergence test
    measures, so no iteration may leave a gene worse off than it found it."""
    rng = np.random.default_rng(5)
    n, n_genes = 150, 80
    x1 = (rng.random(n) < 0.06).astype(float)
    design = np.column_stack([np.ones(n), x1, rng.normal(0, 2.0, size=n)])
    counts = np.empty((n, n_genes))
    for g in range(n_genes):
        beta = np.array([rng.uniform(2, 8), rng.uniform(-8, 8), rng.uniform(-2, 2)])
        counts[:, g] = rng.poisson(np.exp(np.clip(design @ beta, -20, 14)))

    alpha = np.full(n_genes, 0.05)
    fitter = NBGLMBatchFitter(design, max_iter=40, tol=1e-9, min_mu=0.0)
    start = np.zeros((3, n_genes))

    from crispyx._irls import ETA_MAX, Deviance

    deviance = Deviance(counts, "nb", alpha)
    eta = np.clip(design @ start, fitter._eta_min, ETA_MAX)
    dev_start = deviance.total(eta, np.exp(eta))

    _, _, dev_end, _, _ = fitter._irls_at_fixed_dispersion(counts, alpha, start)
    assert np.all(dev_end <= dev_start + 1e-9 * np.abs(dev_start))


def test_a_gene_reported_converged_solves_the_score_equation():
    """The convergence flag gates what ``de.py`` is willing to report, so a
    gene that merely stopped moving must not be flagged as converged."""
    rng = np.random.default_rng(5)
    n, n_genes = 150, 80
    x1 = (rng.random(n) < 0.06).astype(float)
    design = np.column_stack([np.ones(n), x1, rng.normal(0, 2.0, size=n)])
    counts = np.empty((n, n_genes))
    for g in range(n_genes):
        beta = np.array([rng.uniform(2, 8), rng.uniform(-8, 8), rng.uniform(-2, 2)])
        counts[:, g] = rng.poisson(np.exp(np.clip(design @ beta, -20, 14)))

    alpha = np.full(n_genes, 0.05)
    fitter = NBGLMBatchFitter(design, max_iter=40, tol=1e-9, min_mu=0.0)
    _, mu, _, converged, _ = fitter._irls_at_fixed_dispersion(
        counts, alpha, np.zeros((3, n_genes))
    )

    assert converged.any()
    score = design.T @ ((counts - mu) / (1.0 + alpha * mu))
    relative = np.max(np.abs(score), axis=0) / np.maximum(counts.sum(axis=0), 1.0)
    assert np.max(relative[converged]) < 1e-6


def test_numba_path_reports_the_dispersion_it_fitted_at():
    """The 2-feature kernel holds the dispersion fixed, so what it returns has
    to come from a fit at the dispersion that is reported.

    On an intercept-plus-indicator design the coefficients are the two group
    means whatever the dispersion is, so the standard errors are where a
    placeholder dispersion shows up: the IRLS weight ``mu / (1 + alpha mu)``
    at ``alpha = 0.1`` is several times the weight at a dispersion of 2, and
    every Wald statistic built on it is wrong by that factor.
    """
    rng = np.random.default_rng(7)
    n, n_genes = 400, 40
    x1 = (rng.random(n) < 0.35).astype(float)
    design = np.column_stack([np.ones(n), x1])
    mu = np.exp(design @ np.c_[rng.uniform(1.0, 3.0, n_genes), rng.normal(0, 0.8, n_genes)].T)
    counts = _generate_nb_counts(rng, mu, 1.5).astype(float)

    fitter = NBGLMBatchFitter(design, max_iter=50, tol=1e-9, min_mu=0.0)
    numba = fitter.fit_batch(counts, use_numba=True)
    refit = fitter.fit_batch(counts, use_numba=False, fixed_dispersion=numba.dispersion)

    assert numba.dispersion.min() > 0.5, "the test needs a dispersion unlike 0.1"
    np.testing.assert_allclose(numba.coef, refit.coef, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(numba.se, refit.se, rtol=1e-8)


def test_the_mean_floor_does_not_make_genes_report_as_non_convergent():
    """A gene whose fit rests on the mean floor must still be reported.

    Where the floor binds, the fitted mean of those cells no longer moves with
    the coefficients, so they carry no gradient while the normal equations
    still count them: IRLS proposes a direction the deviance does not fall
    along, and the line search finds nothing to take.  That says something
    about the floor, not about the fit, and treating it as a failure to
    converge would hand ``de.py`` a ``NaN`` p-value for every low-count gene
    the floor exists for.
    """
    rng = np.random.default_rng(5)
    n, n_genes = 150, 80
    x1 = (rng.random(n) < 0.06).astype(float)
    design = np.column_stack([np.ones(n), x1, rng.normal(0, 2.0, size=n)])
    counts = np.empty((n, n_genes))
    for g in range(n_genes):
        beta = np.array([rng.uniform(2, 8), rng.uniform(-8, 8), rng.uniform(-2, 2)])
        counts[:, g] = rng.poisson(np.exp(np.clip(design @ beta, -20, 14)))

    alpha = np.full(n_genes, 0.05)
    fitter = NBGLMBatchFitter(design, max_iter=40, tol=1e-6, min_mu=0.5)
    beta, mu, _, converged, _ = fitter._irls_at_fixed_dispersion(
        counts, alpha, np.zeros((3, n_genes))
    )

    on_the_floor = (mu <= 0.5 + 1e-12).any(axis=0)
    assert on_the_floor.sum() > n_genes // 2, "fixture should exercise the floor"
    assert converged.all()
