"""DESeq2 on one perturbation against the controls: against a control summary
(glm.fit_two_group_deseq2), and with covariates on the cells (glm.fit_glm_deseq2)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from crispyx.glm import build_design_matrix, fit_glm_deseq2, fit_two_group_deseq2, summarize_controls

pydeseq2 = pytest.importorskip("pydeseq2")


def _screen(seed=0, n_ctrl=400, n_pert=150, n_genes=300, batch_sd=0.0):
    """Controls and one perturbation; 0.3 to 30 counts per cell, dispersions
    scattered around a DESeq2-shaped trend, 30 genes moved. Each cell is in
    one of three batches, with gene effects of SD ``batch_sd`` (log scale)."""
    rng = np.random.default_rng(seed)
    labels = np.array(["ctrl"] * n_ctrl + ["KO1"] * n_pert)
    mu = rng.lognormal(np.log(3), 1.2, n_genes)
    alpha = (0.05 + 0.6 / mu) * rng.lognormal(0, 0.5, n_genes)
    lfc = np.zeros(n_genes)
    lfc[rng.choice(n_genes, 30, replace=False)] = rng.normal(0, 1, 30)
    sf = rng.lognormal(0, 0.3, labels.size)
    batch = rng.integers(0, 3, labels.size)
    mean = mu[None, :] * sf[:, None] * np.exp(rng.normal(0, batch_sd, (3, n_genes)))[batch]
    mean[labels == "KO1"] *= np.exp(lfc)
    counts = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mean)).astype(np.float64)
    if batch_sd:
        return counts, labels, sf, np.array([f"b{b}" for b in batch])
    return counts, labels, sf


def _pydeseq2(counts, labels, sf, batch=None):
    """PyDESeq2's steps on control + KO1 with the given size factors (no
    Cook's refit or filtering, which crispyx does not do), on ``~ condition``
    or, given batches, ``~ batch + condition``."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    df = pd.DataFrame(counts.astype(int), columns=[f"g{i}" for i in range(counts.shape[1])])
    meta = pd.DataFrame({"condition": labels}, index=df.index)
    if batch is not None:
        meta["batch"] = batch
    dds = DeseqDataSet(counts=df, metadata=meta, design="~condition" if batch is None else "~batch + condition",
                       refit_cooks=False, quiet=True)
    dds.obs["size_factors"] = sf
    dds.layers["normed_counts"] = dds.X / sf[:, None]
    dds.var["_normed_means"] = dds.layers["normed_counts"].mean(0)
    for step in ("fit_genewise_dispersions", "fit_dispersion_trend", "fit_dispersion_prior",
                 "fit_MAP_dispersions", "fit_LFC", "calculate_cooks"):
        getattr(dds, step)()
    stats = DeseqStats(dds, contrast=["condition", "KO1", "ctrl"], cooks_filter=False,
                       independent_filter=False, quiet=True)
    stats.summary()
    return dds, stats.results_df


def _fit(counts, labels, sf, bin_width=0.02, blocks=1):
    ctrl = labels == "ctrl"
    rows = np.array_split(np.arange(ctrl.sum()), blocks)
    summary = summarize_controls(
        [(r, sp.csr_matrix(counts[ctrl][r])) for r in rows], sf[ctrl], counts.shape[1], bin_width=bin_width
    )
    return summary, fit_two_group_deseq2(summary, counts[~ctrl], sf[~ctrl], bin_width=bin_width)


@pytest.fixture(scope="module")
def screen_and_reference():
    counts, labels, sf = _screen()
    return counts, labels, sf, _pydeseq2(counts, labels, sf)


def test_matches_pydeseq2_given_the_same_size_factors(screen_and_reference):
    """Every step agrees with PyDESeq2 run on the same cells and size
    factors. Where PyDESeq2's gene-wise optimiser stops at min_disp, its own
    loss is higher there than at our estimate (low-count genes whose
    moments start is clipped to the bound)."""
    from pydeseq2.utils import fit_lin_mu, nb_nll

    counts, labels, sf, (dds, ref) = screen_and_reference
    _, res = _fit(counts, labels, sf)
    theirs = dds.var["genewise_dispersions"].to_numpy()
    at_bound = theirs <= 1e-8 * 1.0001
    assert at_bound.sum() < 0.1 * at_bound.size
    np.testing.assert_allclose(res.genewise_dispersion[~at_bound], theirs[~at_bound], rtol=1e-3)
    X = np.column_stack([np.ones(labels.size), (labels == "KO1").astype(float)])
    for g in np.flatnonzero(at_bound):
        y = counts[:, g]
        mu = fit_lin_mu(y, sf, X, 0.5)

        def loss(a):
            w = mu / (1 + mu * a)
            return nb_nll(y, mu, a) + 0.5 * np.linalg.slogdet((X.T * w) @ X)[1]

        assert loss(res.genewise_dispersion[g]) <= loss(theirs[g])

    np.testing.assert_allclose(res.map_dispersion, dds.var["MAP_dispersions"], rtol=2e-2)
    np.testing.assert_allclose(res.lfc, ref["log2FoldChange"] * np.log(2), atol=1e-3)
    dz = np.abs(res.statistic - ref["stat"].to_numpy())
    assert np.median(dz) < 1e-3 and dz.max() < 0.05
    assert res.converged.all()


def _bound_losses_are_lower(res, dds, counts, design, sf):
    """Where PyDESeq2's gene-wise optimiser stops at min_disp, our estimate has
    the lower Cox-Reid loss at PyDESeq2's own means."""
    from pydeseq2.utils import nb_nll

    theirs = dds.var["genewise_dispersions"].to_numpy()
    at_bound = theirs <= 1e-8 * 1.0001
    assert at_bound.sum() < 0.1 * at_bound.size
    for g in np.flatnonzero(at_bound):
        y, mu = counts[:, g], dds.layers["_mu_hat"][:, g]

        def loss(a):
            w = mu / (1 + mu * a)
            return nb_nll(y, mu, a) + 0.5 * np.linalg.slogdet((design.T * w) @ design)[1]

        assert loss(res.genewise_dispersion[g]) <= loss(theirs[g])
    return at_bound


def test_glm_with_a_covariate_matches_pydeseq2_given_the_same_size_factors():
    """fit_glm_deseq2 on ~ batch + condition agrees with PyDESeq2 step by step:
    gene-wise dispersions to 1e-3 (where PyDESeq2's optimiser does not stop at
    min_disp), the trend fitted to the same gene-wise values, and the Wald
    statistics. The MAP and final dispersions differ by the trend those
    min_disp genes move (about 5 % here), which moves z by at most 0.05."""
    from crispyx.glm import fit_dispersion_trend

    counts, labels, sf, batch = _screen(seed=2, batch_sd=0.3)
    dds, ref = _pydeseq2(counts, labels, sf, batch)
    design, columns = build_design_matrix(
        pd.DataFrame({"batch": batch}), covariate_columns=["batch"],
        perturbation_indicator=(labels == "KO1").astype(float), intercept=True,
    )
    res = fit_glm_deseq2(design, counts, sf, columns.index("perturbation"))
    at_bound = _bound_losses_are_lower(res, dds, counts, design, sf)
    theirs = dds.var["genewise_dispersions"].to_numpy()
    np.testing.assert_allclose(res.genewise_dispersion[~at_bound], theirs[~at_bound], rtol=1e-3)
    np.testing.assert_allclose(
        fit_dispersion_trend(dds.var["_normed_means"].to_numpy(), theirs), dds.var["fitted_dispersions"], rtol=1e-4,
    )
    np.testing.assert_allclose(res.map_dispersion, dds.var["MAP_dispersions"], rtol=0.1)
    np.testing.assert_allclose(res.lfc, ref["log2FoldChange"] * np.log(2), atol=1e-3)
    dz = np.abs(res.statistic - ref["stat"].to_numpy())
    assert np.median(dz) < 1e-3 and dz.max() < 0.05
    assert res.converged.all()
    assert res.coef.shape == (counts.shape[1], design.shape[1])
    np.testing.assert_array_equal(res.coef[:, columns.index("perturbation")], res.lfc)


@pytest.mark.parametrize("fit", ["two_group", "glm"])
def test_a_given_dispersion_is_tested_at_and_genewise_ones_without_shrinkage(screen_and_reference, fit):
    """dispersion= holds every gene at that value; shrink=False tests at the
    gene-wise estimates. Both engines, the same answers."""
    counts, labels, sf, _ = screen_and_reference
    ctrl = labels == "ctrl"
    if fit == "two_group":
        summary = summarize_controls([(np.arange(ctrl.sum()), sp.csr_matrix(counts[ctrl]))], sf[ctrl],
                                     counts.shape[1])

        def run(**kw):
            return fit_two_group_deseq2(summary, counts[~ctrl], sf[~ctrl], **kw)
    else:
        design = np.column_stack([np.ones(labels.size), (~ctrl).astype(float)])

        def run(**kw):
            return fit_glm_deseq2(design, counts, sf, 1, **kw)

    fixed = run(dispersion=0.2)
    np.testing.assert_array_equal(fixed.dispersion, 0.2)
    assert np.isnan(fixed.genewise_dispersion).all() and np.isnan(fixed.prior_var)
    genewise = run(shrink=False)
    np.testing.assert_array_equal(genewise.dispersion, genewise.genewise_dispersion)
    assert np.isnan(genewise.map_dispersion).all()
    shrunk = run()
    np.testing.assert_allclose(genewise.genewise_dispersion, shrunk.genewise_dispersion)
    assert not np.allclose(genewise.statistic, shrunk.statistic)


def test_binned_controls_match_the_exact_summary(screen_and_reference):
    """Size-factor bins of the default width give what one bin per distinct
    size factor (exact) gives. The largest gaps are low-count genes whose mean
    crosses the floor min_mu inside a bin (control mean near 0.5 here)."""
    counts, labels, sf, _ = screen_and_reference
    exact = _fit(counts, labels, sf, bin_width=0)[1]
    binned = _fit(counts, labels, sf, bin_width=0.02)[1]
    for field in ("genewise_dispersion", "map_dispersion", "dispersion"):
        np.testing.assert_allclose(getattr(binned, field), getattr(exact, field), rtol=5e-4)
    np.testing.assert_allclose(binned.se, exact.se, rtol=1e-4)
    np.testing.assert_allclose(binned.lfc, exact.lfc, atol=1e-4)


@pytest.mark.parametrize("size_factor_method, covariates", [
    ("libsize", False),
    ("deseq2", False),
    ("libsize", True),
])
def test_nb_glm_test_matches_pydeseq2_given_the_same_size_factors(tmp_path, size_factor_method, covariates):
    """End to end: nb_glm_test agrees with PyDESeq2 on each perturbation's
    cells, given the size factors nb_glm_test computes; without covariates
    through the control summary, with a batch covariate on the cells."""
    import anndata as ad

    from crispyx.de import _global_size_factors, nb_glm_test

    rng = np.random.default_rng(1)
    counts, labels, _, batch = _screen(seed=1, n_pert=300, batch_sd=0.3 if covariates else 1e-12)
    labels = labels.copy()
    labels[labels == "KO1"] = np.where(rng.random((labels == "KO1").sum()) < 0.5, "KO1", "KO2")
    path = tmp_path / "screen.h5ad"
    obs = pd.DataFrame({"perturbation": labels, "batch": batch})
    ad.AnnData(counts.astype(np.float32), obs=obs).write(path)
    result = nb_glm_test(
        path, perturbation_column="perturbation", control_label="ctrl", size_factor_method=size_factor_method,
        covariates=["batch"] if covariates else None, n_jobs=2, verbose=False, output_path=tmp_path / "de.h5ad",
        min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0, min_cells_ctrl=0,
        min_cells_pert=0, min_total_count=0,
    )
    sf = _global_size_factors(path, size_factor_method, 2048, labels, "ctrl")
    for name in ("KO1", "KO2"):
        keep = np.isin(labels, ["ctrl", name])
        sub = labels[keep].copy()
        sub[sub == name] = "KO1"
        _, ref = _pydeseq2(counts[keep], sub, sf[keep], batch[keep] if covariates else None)
        z = result[name].statistic
        dz = np.abs(z - ref["stat"].to_numpy())
        assert np.median(dz) < 1e-3 and dz.max() < 0.05, (name, np.median(dz), dz.max())


def test_control_summary_does_not_depend_on_the_blocks(screen_and_reference):
    """Streaming the controls in several blocks gives the same summary, and
    its count histogram holds every count (some above the dense table's 64)."""
    counts, labels, sf, _ = screen_and_reference
    one, _ = _fit(counts, labels, sf, blocks=1)
    many, _ = _fit(counts, labels, sf, blocks=7)
    for field in ("s_mom", "y_mom", "norm_mean", "total", "nnz", "hist_ptr", "hist_val", "hist_cnt"):
        np.testing.assert_allclose(getattr(many, field), getattr(one, field), rtol=1e-12)
    ctrl = counts[labels == "ctrl"]
    assert ctrl.max() > 64
    by_gene = np.bincount(np.repeat(np.arange(ctrl.shape[1]), np.diff(one.hist_ptr)),
                          weights=one.hist_val * one.hist_cnt, minlength=ctrl.shape[1])
    np.testing.assert_allclose(by_gene, ctrl.sum(axis=0))


def _three_targets(tmp_path, covariates):
    """Controls and KO1-KO3 (one simulated perturbation split three ways), with batches."""
    import anndata as ad

    rng = np.random.default_rng(3)
    counts, labels, _, batch = _screen(seed=3, n_pert=360, batch_sd=0.3 if covariates else 1e-12)
    labels = labels.copy()
    labels[labels == "KO1"] = rng.choice(["KO1", "KO2", "KO3"], (labels == "KO1").sum())
    path = tmp_path / "screen.h5ad"
    ad.AnnData(counts.astype(np.float32), obs=pd.DataFrame({"perturbation": labels, "batch": batch})).write(path)
    return path, counts, labels, batch


_NO_FILTERS = dict(min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0, min_cells_ctrl=0,
                   min_cells_pert=0, min_total_count=0)


@pytest.mark.parametrize("covariates", [False, True], ids=["condition", "batch"])
def test_perturbations_fitted_together_match_pydeseq2_with_one_factor(tmp_path, covariates):
    """targets_per_fit=3: the controls and KO1-KO3 in one model, the
    perturbation a factor of four levels (as a pseudo-bulk analysis fits many
    targets at once), against PyDESeq2 fitted the same way with a contrast
    per perturbation."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    from crispyx.de import _global_size_factors, nb_glm_test

    path, counts, labels, batch = _three_targets(tmp_path, covariates)
    result = nb_glm_test(path, perturbation_column="perturbation", control_label="ctrl",
                         covariates=["batch"] if covariates else None, targets_per_fit=3, min_mu=0.5,
                         n_jobs=2, verbose=False, output_path=tmp_path / "de.h5ad", **_NO_FILTERS)
    sf = _global_size_factors(path, "libsize", 2048, labels, "ctrl")
    df = pd.DataFrame(counts.astype(int), columns=[f"g{i}" for i in range(counts.shape[1])])
    meta = pd.DataFrame({"condition": labels, "batch": batch}, index=df.index)
    dds = DeseqDataSet(counts=df, metadata=meta, design="~batch + condition" if covariates else "~condition",
                       refit_cooks=False, quiet=True)
    dds.obs["size_factors"] = sf
    dds.layers["normed_counts"] = dds.X / sf[:, None]
    dds.var["_normed_means"] = dds.layers["normed_counts"].mean(0)
    for step in ("fit_genewise_dispersions", "fit_dispersion_trend", "fit_dispersion_prior",
                 "fit_MAP_dispersions", "fit_LFC", "calculate_cooks"):
        getattr(dds, step)()
    for name in ("KO1", "KO2", "KO3"):
        stats = DeseqStats(dds, contrast=["condition", name, "ctrl"], cooks_filter=False,
                           independent_filter=False, quiet=True)
        stats.summary()
        dz = np.abs(result[name].statistic - stats.results_df["stat"].to_numpy())
        assert np.median(dz) < 1e-3 and np.nanmax(dz) < 0.05, (name, np.median(dz), np.nanmax(dz))


def test_perturbations_are_fitted_in_groups_of_targets_per_fit(tmp_path):
    """targets_per_fit=2 on three perturbations, in the order given: KO1 and
    KO2 together, KO3 on its own -- the same fit as targets_per_fit=1 gives
    it."""
    from crispyx.de import nb_glm_test

    path, *_ = _three_targets(tmp_path, covariates=True)
    kw = dict(perturbation_column="perturbation", control_label="ctrl", covariates=["batch"], n_jobs=1,
              perturbations=["KO1", "KO2", "KO3"], verbose=False, **_NO_FILTERS)
    paired = nb_glm_test(path, targets_per_fit=2, output_path=tmp_path / "paired.h5ad", **kw)
    alone = nb_glm_test(path, output_path=tmp_path / "alone.h5ad", **kw)
    np.testing.assert_allclose(paired["KO3"].statistic, alone["KO3"].statistic, rtol=1e-10, equal_nan=True)
    assert not np.allclose(paired["KO1"].statistic, alone["KO1"].statistic, equal_nan=True)


@pytest.mark.parametrize("kwargs, match", [
    (dict(targets_per_fit=0), "positive integer"),
    (dict(targets_per_fit=2, lfc_shrinkage_type="apeglm"), "shrink_lfc"),
])
def test_targets_per_fit_is_validated(tmp_path, kwargs, match):
    from crispyx.de import nb_glm_test

    path, *_ = _three_targets(tmp_path, covariates=False)
    with pytest.raises(ValueError, match=match):
        nb_glm_test(path, perturbation_column="perturbation", control_label="ctrl", verbose=False,
                    output_path=tmp_path / "de.h5ad", **kwargs)


def test_every_perturbation_fitted_together_is_estimated_as_an_effect():
    """With more perturbations than other columns, the ones after the first are
    still effects under test, not nuisance group levels the structured
    solver bounds at clip_group (10): a fold change of e^11.5 is reported."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), [200, 50, 50, 50, 50])  # controls, then four perturbations
    design = np.column_stack([np.ones(groups.size)] + [(groups == k).astype(float) for k in range(1, 5)])
    lfc = np.zeros((5, 20))
    lfc[3, 0] = 11.5
    counts = rng.poisson(5.0 * np.exp(lfc[groups])).astype(np.float64)
    res = fit_glm_deseq2(design, counts, np.ones(groups.size), 1, protected_columns=(2, 3, 4), min_mu=1e-6)
    observed = np.log(counts[groups == 3, 0].mean() / counts[groups == 0, 0].mean())
    assert res.coef[0, 3] == pytest.approx(observed, abs=1e-6)


def test_sparse_and_dense_perturbed_counts_give_the_same_fit(screen_and_reference):
    counts, labels, sf, _ = screen_and_reference
    summary, dense = _fit(counts, labels, sf)
    sparse = fit_two_group_deseq2(summary, sp.csr_matrix(counts[labels != "ctrl"]), sf[labels != "ctrl"])
    for key in ("lfc", "se", "dispersion", "genewise_dispersion"):
        np.testing.assert_allclose(getattr(sparse, key), getattr(dense, key), rtol=1e-10, equal_nan=True)


def test_prior_variance_has_its_floor_when_no_gene_is_above_the_dispersion_floor():
    """Constant counts put every gene-wise dispersion at its lower bound, so no
    gene enters the residual MAD; the prior variance is still 0.25, not NaN."""
    counts = np.full((60, 5), 3.0)
    labels = np.array(["ctrl"] * 40 + ["KO1"] * 20)
    for fit in (_fit(counts, labels, np.ones(60))[1],
                fit_glm_deseq2(np.column_stack([np.ones(60), labels == "KO1"]), counts, np.ones(60), 1)):
        assert fit.prior_var == 0.25
        assert np.isfinite(fit.dispersion).all()


def test_a_sequential_fit_with_covariates_restores_the_numba_thread_count(tmp_path):
    import numba

    from crispyx.de import nb_glm_test

    path, *_ = _three_targets(tmp_path, covariates=True)
    before = numba.get_num_threads()
    nb_glm_test(path, perturbation_column="perturbation", control_label="ctrl", covariates=["batch"], n_jobs=1,
                perturbations=["KO1"], verbose=False, output_path=tmp_path / "de.h5ad")
    assert numba.get_num_threads() == before


def test_given_size_factors_follow_their_cells_into_the_sorted_copy(tmp_path, monkeypatch):
    """Size factors are given in the input's row order; when nb_glm_test sorts
    the input by perturbation first, each cell keeps its own."""
    import crispyx.de as de

    path, counts, labels, _ = _three_targets(tmp_path, covariates=False)
    sf = np.random.default_rng(1).lognormal(0, 0.4, labels.size)
    kw = dict(perturbation_column="perturbation", control_label="ctrl", size_factors=sf, verbose=False)
    unsorted = de.nb_glm_test(path, output_path=tmp_path / "unsorted.h5ad", **kw)
    monkeypatch.setattr(de, "needs_sorting_for_nbglm", lambda *a, **k: True)
    sorted_ = de.nb_glm_test(path, output_path=tmp_path / "sorted.h5ad", **kw)
    assert (tmp_path / "screen_sorted.h5ad").exists()
    for name in ("KO1", "KO2", "KO3"):
        np.testing.assert_allclose(sorted_[name].statistic, unsorted[name].statistic, rtol=1e-8, equal_nan=True)


def test_rank_genes_groups_passes_the_nb_glm_options(tmp_path):
    import crispyx as cx

    path, *_ = _three_targets(tmp_path, covariates=False)
    result = cx.tl.rank_genes_groups(path, method="nb_glm", perturbation_column="perturbation",
                                     control_label="ctrl", output_dir=tmp_path, verbose=False,
                                     targets_per_fit=3, size_factor_method="geomeans", min_mu=0.5,
                                     shrink_dispersion=False)
    assert result.uns["targets_per_fit"].load() == 3
    assert result.uns["size_factor_method"].load() == "geomeans"
    assert result.uns["min_mu"].load() == 0.5
    with pytest.raises(TypeError, match="fit_method"):
        cx.tl.rank_genes_groups(path, method="nb_glm", perturbation_column="perturbation",
                                control_label="ctrl", output_dir=tmp_path, fit_method="joint")


def test_a_gene_held_at_a_bound_is_not_reported_converged():
    """Convergence is judged on the full Newton step, so a gene whose log mean
    the fit clips at 30 (and so stops moving) is not reported converged, while
    an ordinary gene beside it is."""
    rng = np.random.default_rng(0)
    sf_c, sf_t = rng.lognormal(0, 0.3, 300), rng.lognormal(0, 0.3, 3)
    controls = rng.negative_binomial(20, 20 / (20 + 1e3 * sf_c[:, None]), size=(300, 2)).astype(float)
    perturbed = np.column_stack([3e3 * sf_t, 1e14 * sf_t])  # e^30 is ~1.1e13
    summary = summarize_controls([(np.arange(300), sp.csr_matrix(controls))], sf_c, 2)
    fit = fit_two_group_deseq2(summary, perturbed, sf_t, dispersion=0.05)
    np.testing.assert_array_equal(fit.converged, [True, False])
