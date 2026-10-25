"""DESeq2 on one perturbation against a control summary (glm.fit_two_group_deseq2)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from crispyx.glm import fit_two_group_deseq2, summarize_controls

pydeseq2 = pytest.importorskip("pydeseq2")


def _screen(seed=0, n_ctrl=400, n_pert=150, n_genes=300):
    """Controls and one perturbation; 0.3 to 30 counts per cell, dispersions
    scattered around a DESeq2-shaped trend, 30 genes moved."""
    rng = np.random.default_rng(seed)
    labels = np.array(["ctrl"] * n_ctrl + ["KO1"] * n_pert)
    mu = rng.lognormal(np.log(3), 1.2, n_genes)
    alpha = (0.05 + 0.6 / mu) * rng.lognormal(0, 0.5, n_genes)
    lfc = np.zeros(n_genes)
    lfc[rng.choice(n_genes, 30, replace=False)] = rng.normal(0, 1, 30)
    sf = rng.lognormal(0, 0.3, labels.size)
    mean = mu[None, :] * sf[:, None]
    mean[labels == "KO1"] *= np.exp(lfc)
    counts = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mean)).astype(np.float64)
    return counts, labels, sf


def _pydeseq2(counts, labels, sf):
    """PyDESeq2's steps on control + KO1 with the given size factors (no
    Cook's refit or filtering, which fit_two_group_deseq2 does not do)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    df = pd.DataFrame(counts.astype(int), columns=[f"g{i}" for i in range(counts.shape[1])])
    dds = DeseqDataSet(counts=df, metadata=pd.DataFrame({"condition": labels}, index=df.index),
                       design="~condition", refit_cooks=False, quiet=True)
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


@pytest.mark.parametrize("size_factor_method, scope", [
    ("libsize", "per_comparison"),
    ("deseq2", "per_comparison"),
    pytest.param("libsize", "global", marks=pytest.mark.xfail(
        strict=True, reason="global moments dispersion halfway to the trend; the global scope is to be removed")),
])
def test_nb_glm_test_matches_pydeseq2_given_the_same_size_factors(tmp_path, size_factor_method, scope):
    """End to end: nb_glm_test with per-comparison dispersions (the control
    summary path) agrees with PyDESeq2 on each perturbation's cells, given
    the size factors nb_glm_test computes. The global scope does not."""
    import anndata as ad

    from crispyx.de import _global_size_factors, nb_glm_test

    rng = np.random.default_rng(1)
    counts, labels, _ = _screen(seed=1, n_pert=300)
    labels = labels.copy()
    labels[labels == "KO1"] = np.where(rng.random((labels == "KO1").sum()) < 0.5, "KO1", "KO2")
    path = tmp_path / "screen.h5ad"
    ad.AnnData(counts.astype(np.float32), obs=pd.DataFrame({"perturbation": labels})).write(path)
    result = nb_glm_test(
        path, perturbation_column="perturbation", control_label="ctrl", size_factor_method=size_factor_method,
        dispersion_scope=scope, n_jobs=2, verbose=False, output_path=tmp_path / "de.h5ad",
        min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0, min_cells_ctrl=0,
        min_cells_pert=0, min_total_count=0,
    )
    sf = _global_size_factors(path, size_factor_method, 2048, labels, "ctrl")
    for name in ("KO1", "KO2"):
        keep = np.isin(labels, ["ctrl", name])
        sub = labels[keep].copy()
        sub[sub == name] = "KO1"
        _, ref = _pydeseq2(counts[keep], sub, sf[keep])
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
