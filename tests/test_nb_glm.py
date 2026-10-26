from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp
import statsmodels.api as sm

try:  # Optional benchmark dependency for validation comparisons
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ModuleNotFoundError:  # pragma: no cover - executed when benchmarks unavailable
    DeseqDataSet = None
    DeseqStats = None


from crispyx.glm import NBGLMBatchFitter, build_design_matrix
from crispyx.de import nb_glm_test


def _generate_nb_counts(rng, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + mu)
    return rng.negative_binomial(r, p)


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
    np.testing.assert_allclose(exported_adata.layers["dispersion"], alpha)
    assert "standard_error" in exported_adata.layers

    diff_result = result["g1"]
    assert diff_result.method == "nb_glm"
    assert diff_result.pvalue.shape[0] == n_genes


def test_nb_glm_covariates_in_parallel_match_serial(tmp_path):
    """Covariates with enough perturbations (four or more) to fit on worker processes."""
    rng = np.random.default_rng(7)
    n_cells, n_genes = 180, 4
    perturbations = np.repeat(["control", "g1", "g2", "g3", "g4", "g5"], n_cells // 6)
    covariate = rng.normal(size=n_cells)
    mu = np.exp(0.5 + 0.3 * covariate[:, None] + 0.4 * (perturbations == "g1")[:, None]) * np.ones((1, n_genes))
    counts = _generate_nb_counts(rng, mu.ravel(), 0.5).reshape(n_cells, n_genes)
    adata = ad.AnnData(counts, obs=pd.DataFrame({"perturbation": perturbations, "covariate": covariate}),
                       var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))
    path = tmp_path / "synthetic.h5ad"
    adata.write(path)
    kwargs = dict(perturbation_column="perturbation", control_label="control", covariates=["covariate"],
                  min_cells_expressed=0, verbose=False)
    serial = nb_glm_test(path, n_jobs=1, output_path=tmp_path / "serial.h5ad", **kwargs)
    parallel = nb_glm_test(path, n_jobs=2, output_path=tmp_path / "parallel.h5ad", **kwargs)
    assert parallel.groups == serial.groups == ["g1", "g2", "g3", "g4", "g5"]
    np.testing.assert_allclose(parallel.statistics, serial.statistics, rtol=1e-8, equal_nan=True)


def _covariate_screen(tmp_path, seed=11):
    """Five perturbations, a continuous and a categorical covariate, gene-specific dispersion."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 360, 30
    perturbations = np.repeat(["control", "g1", "g2", "g3", "g4", "g5"], n_cells // 6)
    depth = rng.normal(size=n_cells)
    donor = rng.choice(["D1", "D2", "D3"], size=n_cells)
    base = rng.uniform(0.5, 3.0, size=n_genes)
    alpha = rng.uniform(0.1, 0.8, size=n_genes)
    shift = (perturbations == "g1")[:, None] * np.where(np.arange(n_genes) < 5, 0.8, 0.0)
    mu = np.exp(base + 0.3 * depth[:, None] + 0.2 * (donor == "D2")[:, None] + shift)
    counts = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu))
    obs = pd.DataFrame({"perturbation": perturbations, "depth": depth, "donor": donor})
    path = tmp_path / "covariates.h5ad"
    ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])).write(path)
    return path


def test_nb_glm_null_is_calibrated_on_a_replicated_design_with_many_batches(tmp_path):
    """A bulk-like design (two control and two knockdown replicates per batch,
    30 batches as a covariate) leaves ~half the samples as residual degrees
    of freedom. Without the Cox-Reid adjustment in the MAP step, dispersions
    came out at 0.5x the truth and null z-scores had a robust sd of 1.3."""
    rng = np.random.default_rng(0)
    n_batches, n_genes = 30, 150
    base = rng.lognormal(np.log(200), 1.5, n_genes)
    alpha = 0.02 + 1.0 / base
    batch_effect = rng.normal(0, 0.3, (n_batches, n_genes))
    labels = [g for b in range(n_batches) for g in ["control"] * 2 + [f"KD{b}_{k}" for k in range(2) for _ in range(2)]]
    batches = np.repeat([f"b{b}" for b in range(n_batches)], 6)
    mu = rng.uniform(0.5, 2.0, (len(labels), 1)) * base * np.exp(batch_effect[np.repeat(np.arange(n_batches), 6)])
    counts = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu)).astype(np.float32)
    path = tmp_path / "bulk.h5ad"
    ad.AnnData(counts, obs=pd.DataFrame({"perturbation": labels, "batch": batches}),
               var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])).write(path)

    result = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control", covariates=["batch"],
        size_factor_method="deseq2", min_pct_ctrl=0, min_pct_pert=0,
        min_mean_ctrl=0, min_mean_pert=0, output_path=tmp_path / "de.h5ad", n_jobs=1, verbose=False,
    )
    ratio = ad.read_h5ad(result.result_path).layers["dispersion"] / alpha
    assert 0.85 < np.nanmedian(ratio) < 1.2
    z = np.asarray(result.statistics).ravel()
    z = z[np.isfinite(z)]
    assert 0.9 < 1.4826 * np.median(np.abs(z - np.median(z))) < 1.1


def test_controls_too_large_for_memory_raise_with_covariates(tmp_path):
    """With covariates the controls are fitted in memory; when they do not
    fit, the call raises (it used to return all-NaN with the errors only
    logged)."""
    path = _covariate_screen(tmp_path)
    with pytest.raises(MemoryError, match="pseudo-bulk"):
        nb_glm_test(path, perturbation_column="perturbation", control_label="control", covariates=["depth"],
                    memory_limit_gb=1e-6, n_jobs=1, verbose=False, output_path=tmp_path / "out.h5ad")


def test_without_covariates_the_controls_are_streamed(tmp_path):
    """Without covariates, per-comparison dispersions come from the control
    summary, built in one streaming pass: controls that do not fit the memory
    budget are no obstacle."""
    path = _covariate_screen(tmp_path)
    result = nb_glm_test(path, perturbation_column="perturbation", control_label="control",
                         memory_limit_gb=1e-6, n_jobs=1, verbose=False, output_path=tmp_path / "out.h5ad")
    assert np.isfinite(result.statistics).any()


@pytest.mark.parametrize("covariates", [["depth", "donor"], []], ids=["covariates", "two_group"])
@pytest.mark.parametrize("shrink_dispersion", [True, False])
def test_nb_glm_fixed_dispersion_is_used(tmp_path, covariates, shrink_dispersion):
    path = _covariate_screen(tmp_path)
    kwargs = dict(perturbation_column="perturbation", control_label="control", covariates=covariates,
                  min_cells_expressed=0, n_jobs=1, verbose=False, shrink_dispersion=shrink_dispersion)
    low = nb_glm_test(path, dispersion=0.05, output_path=tmp_path / "low.h5ad", **kwargs)
    high = nb_glm_test(path, dispersion=2.0, output_path=tmp_path / "high.h5ad", **kwargs)
    low_ad, high_ad = ad.read_h5ad(low.result_path), ad.read_h5ad(high.result_path)
    np.testing.assert_allclose(high_ad.layers["dispersion"][np.isfinite(high_ad.layers["dispersion"])], 2.0)
    low_se, high_se = low_ad.layers["standard_error"], high_ad.layers["standard_error"]
    ok = np.isfinite(low_se) & np.isfinite(high_se)
    assert ok.any() and np.all(high_se[ok] > low_se[ok])


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


def _floored_fixture(seed=0, n=300, n_genes=120, alpha=0.3):
    """A screen-shaped fixture whose low-count genes rest on the ``min_mu``
    floor: a geometric expression ladder from 0.02 to 50 counts per cell."""
    rng = np.random.default_rng(seed)
    design = np.column_stack([
        np.ones(n), (rng.random(n) < 0.3).astype(float), rng.normal(0, 1, n)
    ])
    size_factors = np.exp(rng.normal(0, 0.3, n))
    beta = np.column_stack([
        np.log(np.geomspace(0.02, 50, n_genes)),
        rng.normal(0, 0.5, n_genes),
        rng.normal(0, 0.3, n_genes),
    ])
    mu = size_factors[:, None] * np.exp(design @ beta.T)
    counts = _generate_nb_counts(rng, mu, alpha).astype(float)
    return counts, design, size_factors, np.full(n_genes, alpha)


def _floor_binds(design, offset, coef, min_mu):
    return (np.exp(design @ coef.T + offset[:, None]) <= min_mu).any(axis=0)


def test_the_mean_floor_does_not_make_genes_report_as_non_convergent():
    """A gene whose fit rests on the mean floor must still be reported.

    Where the floor binds, the fitted mean of those cells no longer moves with
    the coefficients, so they carry no gradient while the normal equations
    still count them: the Newton step is not a descent direction and the line
    search finds nothing to take.  That says something about the floor, not
    about the fit.  Treating it as a failure to converge would hand ``de.py``
    a ``NaN`` p-value for every low-count gene the floor exists for; freezing
    the gene where it stands instead -- which is what judging it by how far it
    moved amounts to -- would report a warm start as a fit.  Such genes take
    the full Newton step and are judged by DESeq2's deviance ratio.
    """
    counts, design, size_factors, alpha = _floored_fixture()
    offset = np.log(size_factors)
    fitter = NBGLMBatchFitter(design, offset=offset, max_iter=200, tol=1e-10, min_mu=0.5)
    result = fitter.fit_batch(
        counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha
    )

    binds = _floor_binds(design, offset, result.coef, 0.5)
    assert binds.sum() > counts.shape[1] // 3, "fixture should exercise the floor"
    assert result.converged.all()
    assert result.n_iter.max() < 200, "no gene should be grinding to the cap"

    # Converged means a fixed point, not merely a gene that stopped moving:
    # one more iteration from here must not move it.
    again = fitter.fit_batch(
        counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha
    )
    np.testing.assert_allclose(again.coef, result.coef, rtol=1e-10, atol=1e-12)


def test_floored_genes_match_pydeseq2_irls():
    """The floor exists for DESeq2 compatibility, so that is what it is worth
    measuring: on the genes it binds on, the coefficients must be PyDESeq2's.

    Judging a floored gene by a descent test it cannot satisfy left it frozen
    a couple of steps from the answer -- 2.8e-02 away in coefficient, against
    1.4e-06 on the genes the floor never touches.
    """
    pytest.importorskip("pydeseq2")
    from pydeseq2.utils import irls_solver

    counts, design, size_factors, alpha = _floored_fixture()
    offset = np.log(size_factors)
    result = NBGLMBatchFitter(
        design, offset=offset, max_iter=200, tol=1e-10, min_mu=0.5
    ).fit_batch(counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha)

    reference = np.array([
        irls_solver(counts[:, g], size_factors, design, alpha[g],
                    min_mu=0.5, beta_tol=1e-12)[0]
        for g in range(counts.shape[1])
    ])
    binds = _floor_binds(design, offset, result.coef, 0.5)
    assert binds.sum() > counts.shape[1] // 3
    assert np.max(np.abs(result.coef[binds] - reference[binds])) < 1e-4


def test_standard_errors_match_pydeseq2_wald_test():
    """``min_mu`` steadies the iteration and has no place in the covariance.

    DESeq2 keeps it out: ``irls_solver`` returns an unthresholded ``mu`` and
    ``wald_test`` rebuilds the weights from it.  Building them from the
    floored mean instead made the standard errors a median of 24% too small on
    floor-binding genes, and the Wald statistics correspondingly too large.
    """
    pytest.importorskip("pydeseq2")
    from pydeseq2.utils import wald_test

    counts, design, size_factors, alpha = _floored_fixture()
    offset = np.log(size_factors)
    result = NBGLMBatchFitter(
        design, offset=offset, max_iter=200, tol=1e-10, min_mu=0.5
    ).fit_batch(counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha)

    ridge = np.diag(np.repeat(1e-6, design.shape[1]))
    contrast = np.array([0.0, 1.0, 0.0])
    reference = np.array([
        wald_test(design, alpha[g], result.coef[g],
                  size_factors * np.exp(design @ result.coef[g]),
                  ridge, contrast, 0.0, None)[2]
        for g in range(counts.shape[1])
    ])
    binds = _floor_binds(design, offset, result.coef, 0.5)
    assert binds.sum() > counts.shape[1] // 3
    np.testing.assert_allclose(result.se[:, 1], reference, rtol=1e-5)


def test_reported_standard_errors_are_invariant_to_the_mean_floor():
    """Stated directly: the floor may move the coefficients, never the
    uncertainty attached to them."""
    counts, design, size_factors, alpha = _floored_fixture()
    offset = np.log(size_factors)
    coef = NBGLMBatchFitter(
        design, offset=offset, max_iter=200, tol=1e-10, min_mu=0.5
    ).fit_batch(
        counts, gene_batch_size=None, use_numba=False, fixed_dispersion=alpha
    ).coef.T

    floored = NBGLMBatchFitter(design, offset=offset, min_mu=0.5)
    free = NBGLMBatchFitter(design, offset=offset, min_mu=0.0)
    np.testing.assert_array_equal(
        floored._compute_se_batch(floored._wald_weights(coef, alpha)),
        free._compute_se_batch(free._wald_weights(coef, alpha)),
    )


# ---------------------------------------------------------------------------
# Per-comparison size factors and dispersion
# ---------------------------------------------------------------------------

def _dense_screen(path: Path, groups: list[str], seed: int = 0) -> tuple[Path, np.ndarray]:
    """Well-expressed NB counts (every gene non-zero in every cell, so DESeq2
    size factors have reference genes) with 8 of 40 genes up 3x off control."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(groups)
    mu = rng.gamma(5, 10, 40)
    effect = np.where(labels[:, None] == "ctrl", 1.0, np.where(np.arange(40) < 8, 3.0, 1.0))
    counts = _generate_nb_counts(rng, mu[None, :] * effect * rng.uniform(0.5, 2, (labels.size, 1)), 0.1)
    counts = np.maximum(counts, 1).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(labels.size)])
    ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(40)])).write(path)
    return path, counts


_NB_KW = dict(perturbation_column="perturbation", control_label="ctrl", verbose=False, n_jobs=1,
              size_factor_method="deseq2")


@pytest.mark.parametrize("lam, given, expected", [
    (0.3, None, 1e-6),   # mostly zeros: single cells
    (20.0, None, 0.5),   # mostly non-zero: pseudo-bulk / bulk
    (0.3, 0.25, 0.25),   # an explicit floor is used as given
])
def test_nb_glm_mean_floor_follows_the_sparsity_of_the_counts(tmp_path, lam, given, expected):
    """min_mu=None takes DESeq2's single-cell floor (1e-6) when at least half
    the counts are zero and its bulk floor (0.5) otherwise; uns records it."""
    rng = np.random.default_rng(0)
    labels = np.array(["ctrl"] * 60 + ["KO1"] * 30)
    counts = rng.poisson(lam, size=(labels.size, 40)).astype(np.float32)
    counts[:, :5] += 1  # a few genes counted everywhere
    path = tmp_path / "counts.h5ad"
    ad.AnnData(counts, obs=pd.DataFrame({"perturbation": labels})).write(path)
    result = nb_glm_test(path, output_path=tmp_path / "de.h5ad", min_mu=given, **_NB_KW)
    assert ad.read_h5ad(result.result_path, backed="r").uns["min_mu"] == expected


@pytest.mark.filterwarnings("ignore:The input file .* CSC format")
@pytest.mark.parametrize("storage", ["dense", "csc"])
def test_nb_glm_refuses_log_transformed_input(tmp_path, storage):
    """Fitted to log1p values the model runs and returns plausible-looking
    nonsense (LFC r 0.93 against the fit on the counts, half the spread of z),
    so it must refuse them before doing any work."""
    path, counts = _dense_screen(tmp_path / "counts.h5ad", ["ctrl"] * 40 + ["KO1"] * 40)
    logged = np.log1p(counts)
    matrix = sp.csc_matrix(logged) if storage == "csc" else logged
    adata = ad.read_h5ad(path)
    ad.AnnData(matrix, obs=adata.obs, var=adata.var).write(tmp_path / "log1p.h5ad")
    with pytest.raises(ValueError, match="nb_glm_test needs raw counts"):
        nb_glm_test(tmp_path / "log1p.h5ad", output_path=tmp_path / "out.h5ad", **_NB_KW)
    assert not (tmp_path / "out.h5ad").exists()


def test_nb_glm_finds_the_planted_effects(tmp_path):
    """Per-comparison dispersions give valid Wald tests that find the planted
    effects."""
    path, _ = _dense_screen(tmp_path / "two.h5ad", ["ctrl"] * 150 + ["KO1"] * 75 + ["KO2"] * 75)
    result = nb_glm_test(path, output_path=tmp_path / "per.h5ad", **_NB_KW)
    assert np.isfinite(result.pvalues).all()
    assert (result.pvalues[:, :8] < 1e-6).all()


def test_apeglm_worker_count_fits_the_memory_budget(monkeypatch):
    import crispyx._memory as memory
    from crispyx._memory import _resolve_n_jobs
    from crispyx.de import _apeglm_worker_count

    monkeypatch.setattr(memory, "_detected_available_bytes", lambda: 1e15)  # the budget decides

    assert _apeglm_worker_count(-1, 10_000, 128, memory_limit_gb=1e-3) == 1
    assert _apeglm_worker_count(2, 10_000, 128, memory_limit_gb=1e4) == min(2, _resolve_n_jobs(2))
    # A worker costs three batches of gene columns (its own and two queued by
    # joblib), not the whole matrix: ~12 GB for 1M cells and batch_size=128,
    # so a 64 GB budget allows a few workers and 256 GB many.
    assert 2 <= _apeglm_worker_count(-1, 1_000_000, 128, memory_limit_gb=64) <= 5 or _resolve_n_jobs(-1) < 2
    assert _apeglm_worker_count(-1, 1_000_000, 128, memory_limit_gb=256) >= min(8, _resolve_n_jobs(-1))


def test_shrink_lfc_full_caps_workers_by_memory_limit(tmp_path, monkeypatch):
    """memory_limit_gb reaches the apeGLM workers and does not change the result."""
    import shutil

    import crispyx.de as de_mod

    rng = np.random.default_rng(444)
    labels = np.array(["control"] * 40 + ["g1"] * 20 + ["g2"] * 20)
    counts = rng.poisson(15, size=(labels.size, 5))
    path = tmp_path / "counts.h5ad"
    ad.AnnData(counts, obs=pd.DataFrame({"perturbation": labels}),
               var=pd.DataFrame(index=[f"gene{i}" for i in range(5)])).write(path)
    result = nb_glm_test(path, perturbation_column="perturbation", control_label="control",
                         output_path=tmp_path / "mle.h5ad", verbose=False)

    seen = []
    real = de_mod.shrink_lfc_apeglm

    def spy(*args, **kwargs):
        seen.append(kwargs["n_jobs"])
        return real(*args, **kwargs)

    monkeypatch.setattr(de_mod, "shrink_lfc_apeglm", spy)
    shrunk = {}
    for name, limit in [("tight", 1e-6), ("auto", None)]:
        copy = tmp_path / f"{name}.h5ad"
        shutil.copy(result.result_path, copy)
        de_mod.shrink_lfc(copy, method="full", n_jobs=2, memory_limit_gb=limit, verbose=False)
        shrunk[name] = ad.read_h5ad(copy).X
    assert seen[:2] == [1, 1]  # one call per perturbation under the tight budget
    assert all(n >= 1 for n in seen[2:])
    np.testing.assert_allclose(shrunk["tight"], shrunk["auto"], rtol=1e-10)


def _batched_bulk(tmp_path, batches, categorical=False):
    rng = np.random.default_rng(1)
    labels = np.array(["control", "control", "KD1", "KD1", "KD2", "KD2"] * (len(batches) // 6))
    counts = rng.negative_binomial(5, 5 / 105, (len(labels), 40)).astype(np.float32)
    batch = pd.Categorical(batches) if categorical else np.asarray(batches)
    obs = pd.DataFrame({"perturbation": labels, "batch": batch})
    path = tmp_path / "bulk.h5ad"
    ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(40)])).write(path)
    return path


@pytest.mark.parametrize("targets_per_fit", [1, 2])
def test_nb_glm_names_a_batch_confounded_with_a_perturbation(tmp_path, caplog, targets_per_fit):
    """Every KD2 sample is in batch b1, which has no controls, so the batch
    and perturbation columns of KD2's comparison are the same column: KD2
    fails with the reason logged, and the rest of the run goes on -- also
    KD1 when the two are fitted in one model."""
    batches = ["b0", "b0", "b0", "b0", "b1", "b1"] * 2
    path = _batched_bulk(tmp_path, batches)
    with pytest.warns(UserWarning, match=r"failed and are NaN.*'KD2'"):
        result = nb_glm_test(
            path, perturbation_column="perturbation", control_label="control", covariates=["batch"],
            min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0, targets_per_fit=targets_per_fit,
            output_path=tmp_path / "de.h5ad", n_jobs=1, verbose=False,
        )
    assert "rank-deficient; 'batch_b1'" in caplog.text
    assert np.all(np.isnan(result["KD2"].statistic))
    assert np.all(np.isfinite(result["KD1"].statistic))


def test_a_confounded_perturbation_does_not_stop_a_parallel_run(tmp_path, caplog):
    """In worker processes too, a perturbation confounded with a batch (KD5:
    all its samples in a batch without controls) fails on its own; the
    others are tested."""
    rng = np.random.default_rng(2)
    labels = np.array(["control"] * 6 + [f"KD{k}" for k in range(1, 6) for _ in range(3)])
    batch = np.array(["b0", "b1"] * 3 + ["b0", "b1", "b0"] * 4 + ["b2"] * 3)
    counts = rng.negative_binomial(5, 5 / 105, (len(labels), 30)).astype(np.float32)
    path = tmp_path / "bulk.h5ad"
    ad.AnnData(counts, obs=pd.DataFrame({"perturbation": labels, "batch": batch}),
               var=pd.DataFrame(index=[f"gene{i}" for i in range(30)])).write(path)
    with pytest.warns(UserWarning, match=r"failed and are NaN.*'KD5'"):
        result = nb_glm_test(
            path, perturbation_column="perturbation", control_label="control", covariates=["batch"],
            min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0,
            output_path=tmp_path / "de.h5ad", n_jobs=2, verbose=False,
        )
    assert "rank-deficient; 'batch_b2'" in caplog.text
    assert np.all(np.isnan(result["KD5"].statistic))
    for k in range(1, 5):
        assert np.isfinite(result[f"KD{k}"].statistic).any()


def test_build_design_matrix_drops_levels_absent_from_the_subset():
    obs = pd.DataFrame({"batch": pd.Categorical(["b0", "b1", "b0", "b1"], categories=["b0", "b1", "b2"])})
    design, names = build_design_matrix(obs, covariate_columns=["batch"], perturbation_indicator=np.array([0, 0, 1, 1]))
    assert names == ["intercept", "perturbation", "batch_b1"]
    assert np.all(design.any(axis=0))


def test_check_full_rank_names_each_column_spanned_by_earlier_ones():
    from crispyx.glm import check_full_rank

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 50))
    design = np.column_stack([np.ones(50), a, b, a + 2 * b, 3 * a])
    with pytest.raises(ValueError, match=r"'a\+2b', '3a' is a linear combination"):
        check_full_rank(design, ["1", "a", "b", "a+2b", "3a"], "test")
    check_full_rank(design[:, :3], ["1", "a", "b"], "test")
    check_full_rank(np.empty((5, 0)), [], "test")
