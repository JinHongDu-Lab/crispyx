"""Tests for the per-condition low-expression filter applied to DE tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import pytest

import crispyx as cx
from crispyx._statistics import _low_expr_in_both_mask, _nonestimable_glm_mask

# ---------------------------------------------------------------------------
# _low_expr_in_both_mask: a gene is dropped only when BOTH arms are low, and an
# arm is low only when BOTH its expressing fraction and its mean are below
# threshold.  Defaults: min_pct_ctrl=0.01, min_pct_pert=0.002,
# min_mean_ctrl=0.05, min_mean_pert=0.005.
# ---------------------------------------------------------------------------

_OFF = dict(min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, min_mean_pert=0.0)
_SYMMETRIC = dict(min_pct_ctrl=0.01, min_pct_pert=0.01, min_mean_ctrl=0.05, min_mean_pert=0.05)


@pytest.mark.parametrize(
    "expr_p, expr_c, mean_p, mean_c, n_cells, thresholds, expected",
    [
        ([0, 0, 5, 5, 0, 5], [0, 0, 0, 5, 5, 5], [0] * 6, [0] * 6, (10, 10), _OFF, [False] * 6),
        ([0, 0, 50, 0], [0, 50, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], (100, 100), _SYMMETRIC,
         [True, False, False, True]),
        # pct passes while the mean fails: not low.
        ([20, 0], [20, 0], [0.001, 0], [0.001, 0], (100, 100), _SYMMETRIC, [False, True]),
        ([0, 5], [0, 5], [0, 1], [0, 1], (0, 10), {}, [False, False]),
        # Defaults keep a gene expressed in 3% of perturbed cells over a silent control.
        ([0, 3, 0, 80], [0, 0, 0, 80], [0, 0.04, 0, 1.5], [0, 0, 0, 1.5], (100, 100), {},
         [True, False, True, False]),
        # ...and so does the symmetric filter, because the perturbed pct passes.
        ([0, 3, 0, 80], [0, 0, 0, 80], [0, 0.04, 0, 1.5], [0, 0, 0, 1.5], (100, 100), _SYMMETRIC,
         [True, False, True, False]),
        ([5, 0], [0, 0], [0.5, 0], [0, 0], (100, 100), {}, [False, True]),
        # Perturbed pct 0.1% < 0.2% and mean 0.001 < 0.005: low on both arms.
        ([1, 5], [0, 0], [0.001, 0.01], [0, 0], (1000, 1000), {}, [True, False]),
    ],
    ids=["thresholds-off", "jointly-low-only", "either-metric-keeps", "empty-pert-group",
         "defaults-keep-induced", "symmetric", "decoupled-arms", "default-pert-mean"],
)
def test_low_expr_mask(expr_p, expr_c, mean_p, mean_c, n_cells, thresholds, expected):
    mask = _low_expr_in_both_mask(
        pert_expr_counts=np.array(expr_p),
        control_expr_counts=np.array(expr_c),
        pert_mean=np.array(mean_p, dtype=float),
        control_mean=np.array(mean_c, dtype=float),
        n_pert_cells=n_cells[0],
        n_control_cells=n_cells[1],
        **thresholds,
    )
    assert mask.dtype == bool
    assert mask.tolist() == expected


def test_low_expr_mask_min_pct_both_deprecation_warning():
    """Passing min_pct_both emits DeprecationWarning and overrides ctrl/pert."""
    with pytest.warns(DeprecationWarning, match="min_pct_both is deprecated"):
        mask = _low_expr_in_both_mask(
            pert_expr_counts=np.array([1]),
            control_expr_counts=np.array([1]),
            pert_mean=np.array([0.0]),
            control_mean=np.array([0.0]),
            n_pert_cells=10,
            n_control_cells=10,
            min_pct_both=0.5,
        )
    assert mask.tolist() == [True]


# ---------------------------------------------------------------------------
# End-to-end on tiny datasets
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path: Path, *, log_normalise: bool):
    """Build a small AnnData where one gene is jointly silent in BOTH groups."""

    rng = np.random.default_rng(0)
    n_ctrl = 80
    n_pert = 80
    n_cells = n_ctrl + n_pert
    n_genes = 5

    # Gene 0: well expressed in both groups
    # Gene 1: differentially expressed (high in pert, low in ctrl)
    # Gene 2: differentially expressed (high in ctrl, low in pert)
    # Gene 3: SILENT in both groups (should be filtered out)
    # Gene 4: low-but-real signal in both groups
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    counts[:n_ctrl, 0] = rng.poisson(20, n_ctrl)
    counts[n_ctrl:, 0] = rng.poisson(20, n_pert)
    counts[:n_ctrl, 1] = rng.poisson(2, n_ctrl)
    counts[n_ctrl:, 1] = rng.poisson(15, n_pert)
    counts[:n_ctrl, 2] = rng.poisson(15, n_ctrl)
    counts[n_ctrl:, 2] = rng.poisson(2, n_pert)
    # gene 3: leave all zeros (silent in both)
    counts[:n_ctrl, 4] = rng.poisson(5, n_ctrl)
    counts[n_ctrl:, 4] = rng.poisson(5, n_pert)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"gene{i}" for i in range(n_genes)]},
        index=[f"gene{i}" for i in range(n_genes)],
    )
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    if log_normalise:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / ("ds_log.h5ad" if log_normalise else "ds_raw.h5ad")
    adata.write(path)
    return path


_METHODS = {
    "t_test": (cx.t_test, True),
    "wilcoxon_test": (cx.wilcoxon_test, True),
    "nb_glm_test": (cx.nb_glm_test, False),
}


@pytest.mark.parametrize("method", list(_METHODS))
def test_jointly_silent_gene_is_excluded(tmp_path, method):
    fn, log_normalise = _METHODS[method]
    path = _make_dataset(tmp_path, log_normalise=log_normalise)
    res = fn(path, perturbation_column="perturbation", control_label="ctrl",
             output_path=tmp_path / "result.h5ad", verbose=False, **_SYMMETRIC)
    pvals = np.asarray(res.pvalues[res.groups.index("KO1")])
    lfc = np.asarray(res.logfoldchanges[res.groups.index("KO1")])
    assert np.isnan(pvals[3]) and np.isnan(lfc[3]), "gene 3 is silent in both groups"
    assert np.isfinite(pvals[[0, 1, 2, 4]]).all()


def test_thresholds_control_what_is_filtered(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=True)
    kw = dict(perturbation_column="perturbation", control_label="ctrl", verbose=False)
    loose = cx.t_test(path, output_path=tmp_path / "loose.h5ad", **kw, **_OFF)
    strict = cx.t_test(path, output_path=tmp_path / "strict.h5ad", **kw,
                       min_pct_ctrl=0.99, min_pct_pert=0.99, min_mean_ctrl=1e6, min_mean_pert=1e6)
    loose_p = np.asarray(loose.pvalues[0])
    # Filter off: only the zero-variance gene is untested.
    assert np.isnan(loose_p).tolist() == [False, False, False, True, False]
    assert np.isnan(strict.pvalues[0]).sum() > np.isnan(loose_p).sum()


def _make_dataset_induced(tmp_path: Path):
    """Dataset where gene 5 is induced from zero baseline (control near-zero,
    perturbed has modest expression in ~3% of cells)."""
    rng = np.random.default_rng(42)
    n_ctrl = 200
    n_pert = 50   # unbalanced: fewer perturbed cells
    n_cells = n_ctrl + n_pert
    n_genes = 6

    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    # Gene 0-3: well expressed in both
    for g in range(4):
        counts[:n_ctrl, g] = rng.poisson(10, n_ctrl)
        counts[n_ctrl:, g] = rng.poisson(10, n_pert)
    # Gene 4: silent in both (artifact)
    # Gene 5: induced from near-zero baseline — zero in control, expressed in ~6% of pert
    n_expressing_pert = max(3, int(0.06 * n_pert))
    expressing_cells = rng.choice(np.arange(n_ctrl, n_cells), n_expressing_pert, replace=False)
    counts[expressing_cells, 5] = rng.poisson(5, n_expressing_pert)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "induced.h5ad"
    adata.write(path)
    return path


@pytest.mark.parametrize("method", ["wilcoxon_test", "t_test"])
@pytest.mark.parametrize("thresholds", [{}, _SYMMETRIC], ids=["defaults", "symmetric"])
def test_gene_induced_from_zero_baseline_is_tested(tmp_path, method, thresholds):
    """Gene 5 is zero in control and expressed in ~6% of perturbed cells: the
    perturbed pct passes, so it is tested; gene 4 is silent everywhere."""
    path = _make_dataset_induced(tmp_path)
    res = getattr(cx, method)(path, perturbation_column="perturbation", control_label="ctrl",
                              output_path=tmp_path / "result.h5ad", verbose=False, **thresholds)
    pvals = np.asarray(res.pvalues[res.groups.index("KO1")])
    assert np.isnan(pvals[4])
    assert np.isfinite(pvals[5])


# ---------------------------------------------------------------------------
# Estimability filter for the NB-GLM effect (_nonestimable_glm_mask)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pert, control, thresholds, expected",
    [
        ([0, 5, 3, 0, 1], [9, 0, 4, 0, 1], {}, [True, True, False, True, False]),
        # CRISPRi wants a demanding control side, CRISPRa a demanding perturbed side.
        ([1, 4, 9], [9, 9, 9], dict(min_cells_ctrl=5, min_cells_pert=1), [False, False, False]),
        ([1, 4, 9], [9, 9, 9], dict(min_cells_ctrl=1, min_cells_pert=5), [True, True, False]),
        ([9, 9], [2, 9], dict(min_cells_ctrl=5, min_cells_pert=0), [True, False]),
        ([0, 0], [0, 9], dict(min_cells_ctrl=0, min_cells_pert=0), [False, False]),
    ],
    ids=["either-arm-empty", "crispri", "crispra", "control-only", "disabled"],
)
def test_nonestimable_mask(pert, control, thresholds, expected):
    flagged = _nonestimable_glm_mask(
        pert_expr_counts=np.array(pert), control_expr_counts=np.array(control), **thresholds,
    )
    np.testing.assert_array_equal(flagged, expected)


def _separation_adata(tmp_path):
    """Genes spanning estimable, non-estimable and empty."""
    rng = np.random.default_rng(0)
    n, n_genes = 200, 6
    perturbed = np.zeros(n, dtype=bool)
    perturbed[: n // 2] = True
    counts = rng.negative_binomial(5, 5 / (5 + 30.0), size=(n, n_genes)).astype(np.float32)
    counts[perturbed, 1] = 0          # silenced by the perturbation
    counts[~perturbed, 2] = 0         # induced by the perturbation
    counts[:, 3] = 0                  # no counts anywhere
    counts[perturbed, 4] = 0
    counts[np.flatnonzero(perturbed)[0], 4] = 1   # one count: MLE exists
    names = ["normal", "silenced", "induced", "empty", "one_count", "sparse_one_sided"]
    counts[:, 5] = 0
    counts[np.flatnonzero(perturbed)[:3], 5] = 1  # counts in one arm only

    obs = pd.DataFrame(
        {"perturbation": np.where(perturbed, "g1", "control")},
        index=[f"c{i}" for i in range(n)],
    )
    path = tmp_path / "separation.h5ad"
    ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=names)).write_h5ad(path)
    return path, names


def _by_gene(result, names):
    genes = np.asarray(result.genes).ravel()
    index = {g: i for i, g in enumerate(genes)}
    return (
        index,
        np.asarray(result.logfoldchanges).ravel(),
        np.asarray(result.statistics).ravel(),
        np.asarray(result.pvalues).ravel(),
        np.asarray(result.pts).ravel(),
        np.asarray(result.pts_rest).ravel(),
    )


def test_nb_glm_does_not_estimate_effects_for_one_sided_pairs(tmp_path):
    """A pair absent from one arm gets no effect, statistic or p-value.

    The log-link GLM effect is a difference of log means, so with one arm at
    zero it has no finite maximiser and whatever the fitter returns is set by
    where it stopped.  Such pairs are reported as untested, exactly as genes
    with no counts anywhere already are.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)

    for gene in ("silenced", "induced", "empty", "sparse_one_sided"):
        i = index.get(gene)
        if i is None:
            continue  # dropped upstream, which is also "not reported"
        assert np.isnan(lfc[i]), f"{gene} should have no effect estimate"
        assert np.isnan(stat[i]), f"{gene} should have no statistic"
        assert np.isnan(pval[i]), f"{gene} should have no p-value"

    # Genes whose effect is identified are still tested.
    for gene in ("normal", "one_count"):
        i = index[gene]
        assert np.isfinite(lfc[i]), f"{gene} should still be tested"
        assert np.isfinite(stat[i])


def test_excluded_pairs_keep_their_expression_evidence(tmp_path):
    """Untested is not the same as unremarkable: pts must still show it."""
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)

    silenced = index["silenced"]
    assert np.isnan(lfc[silenced])
    assert pts[silenced] == pytest.approx(0.0)
    assert pts_rest[silenced] > 0.9, "the control arm's expression must remain visible"

    induced = index["induced"]
    assert pts[induced] > 0.9
    assert pts_rest[induced] == pytest.approx(0.0)


def test_excluded_pairs_keep_their_expression_evidence_without_the_control_cache(tmp_path):
    """The same, on the path that does not use the cached control statistics.

    That path built ``pts`` and ``pts_rest`` behind the validity mask, so the
    evidence the documentation points at for an excluded pair -- 0% of
    perturbed cells against 95% of control cells -- was zeroed away on
    precisely the pairs it was there for.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, use_control_cache=False, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)

    silenced = index["silenced"]
    assert np.isnan(pval[silenced]), "the pair is still untested"
    assert pts[silenced] == pytest.approx(0.0)
    assert pts_rest[silenced] > 0.9, "the control arm's expression must remain visible"

    induced = index["induced"]
    assert pts[induced] > 0.9
    assert pts_rest[induced] == pytest.approx(0.0)


def test_effect_is_never_reported_as_zero_for_an_excluded_pair(tmp_path):
    """Reporting 0 would describe a silenced gene as unchanged."""
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)
    silenced = index["silenced"]
    assert not (lfc[silenced] == 0.0), "an excluded effect must be NaN, not zero"


@pytest.mark.parametrize("use_control_cache", [True, False], ids=["cached", "uncached"])
def test_shrinkage_does_not_resurrect_an_excluded_pair(tmp_path, use_control_cache):
    """apeGLM shrinks the pairs that were fitted; it does not fit the rest.

    An excluded pair is dropped before any fit, so it has no MLE to shrink.
    The cached worker handed the shrinkage its zero-initialised coefficients
    anyway, which came back as a finite effect beside a NaN p-value -- and on
    both workers the standard error of an untested gene was replaced by the
    literal 1.0 the per-gene fallback returns.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, data_name=f"apeglm_{use_control_cache}",
        lfc_shrinkage_type="apeglm", use_control_cache=use_control_cache,
        verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)
    se = np.asarray(ad.read_h5ad(res.result_path).layers["standard_error"]).ravel()

    for gene in ("silenced", "induced", "empty", "sparse_one_sided"):
        i = index.get(gene)
        if i is None:
            continue  # dropped upstream, which is also "not reported"
        assert np.isnan(pval[i]), f"{gene} should have no p-value"
        assert np.isnan(lfc[i]), f"{gene} should have no effect estimate under shrinkage"
        assert np.isnan(se[i]), f"{gene} should have no standard error, not {se[i]}"

    normal = index["normal"]
    assert np.isfinite(lfc[normal]), "shrinkage must still report the tested genes"
    assert np.isfinite(se[normal]) and se[normal] > 0


def test_min_cells_per_arm_zero_restores_the_previous_behaviour(tmp_path):
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, data_name="disabled", verbose=False,
        min_cells_ctrl=0, min_cells_pert=0,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)
    silenced = index["silenced"]
    assert np.isfinite(lfc[silenced]), "the filter should be switchable off"
    assert np.isfinite(stat[silenced])


def test_asymmetric_thresholds_reach_nb_glm_test(tmp_path):
    """A demanding control threshold drops pairs a symmetric one would keep.

    The "one_count" gene has a single expressing perturbed cell and a fully
    expressed control arm: estimable, and tested by default.  Requiring more
    perturbed cells -- what an activation screen would want -- excludes it,
    while requiring more control cells does not.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    def effect(**kwargs):
        res = nb_glm_test(
            path, perturbation_column="perturbation", control_label="control",
            output_dir=tmp_path, verbose=False, **kwargs,
        )
        index, lfc, *_ = _by_gene(res, names)
        return lfc[index["one_count"]]

    assert np.isfinite(effect(data_name="sym")), "estimable by default"
    assert np.isnan(effect(data_name="strict_pert", min_cells_pert=5))
    assert np.isfinite(effect(data_name="strict_ctrl", min_cells_ctrl=5))


# ---------------------------------------------------------------------------
# One meaning for "untested": NaN in every column derived from the comparison
# ---------------------------------------------------------------------------

def _make_dataset_partly_tested(tmp_path: Path):
    """A gene that ``min_cells_expressed`` excludes for KO1 but not for KO2.

    Two of the 80 control cells express it, so the control side is not "low"
    and the joint low-expression filter leaves it alone; only the total
    expressing-cell count separates the two perturbations.
    """
    rng = np.random.default_rng(3)
    n = 80
    counts = np.zeros((3 * n, 3), dtype=np.float64)
    # gene 0: expressed everywhere
    counts[:, 0] = rng.poisson(20, 3 * n)
    # gene 1: 2 control cells, none in KO1, 30 in KO2
    counts[[0, 1], 1] = 8.0
    counts[2 * n : 2 * n + 30, 1] = rng.poisson(10, 30) + 1
    # gene 2: expressed everywhere
    counts[:, 2] = rng.poisson(6, 3 * n)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n + ["KO1"] * n + ["KO2"] * n},
        index=[f"cell_{i}" for i in range(3 * n)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(3)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "partly_tested.h5ad"
    adata.write(path)
    return path


_UNTESTED_KW = dict(
    perturbation_column="perturbation",
    control_label="ctrl",
    min_cells_expressed=10,
)


def _assert_untested_is_uniform(res, row):
    """No column derived from the comparison may disagree about testedness."""
    pvals = np.asarray(res.pvalues[row])
    lfc = np.asarray(res.logfoldchanges[row])
    stat = np.asarray(res.statistics[row])
    eff = np.asarray(res.effect_size[row])
    untested = np.isnan(pvals)
    assert untested.any(), "fixture should exclude at least one gene"
    for name, col in (("logfoldchanges", lfc), ("statistics", stat), ("effect_size", eff)):
        assert np.isnan(col[untested]).all(), f"{name} is finite where the p-value is NaN"
        assert np.isfinite(col[~untested]).all(), f"{name} is NaN where the p-value is not"
    # Nothing is spelled as "tested, no change".
    assert not (pvals[untested] == 1.0).any()
    assert not (lfc[untested] == 0.0).any()
    # Expression fractions describe the data and survive exclusion.
    assert np.isfinite(res.pts[row]).all()
    assert np.isfinite(res.pts_rest[row]).all()
    assert res.pts_rest[row][untested].max() > 0


@pytest.mark.parametrize("fn", ["wilcoxon_test", "t_test"])
def test_untested_genes_are_nan_in_every_derived_column(tmp_path, fn):
    path = _make_dataset_partly_tested(tmp_path)
    res = getattr(cx, fn)(path, output_dir=tmp_path, data_name=fn, **_UNTESTED_KW)
    ko1 = res.groups.index("KO1")
    assert np.isnan(res.pvalues[ko1][1]), "gene1 should be excluded for KO1"
    _assert_untested_is_uniform(res, ko1)
    # The same gene is tested for KO2, which has enough expressing cells.
    ko2 = res.groups.index("KO2")
    assert np.isfinite(res.pvalues[ko2][1])
    assert np.isfinite(res.logfoldchanges[ko2][1])


@pytest.mark.parametrize("fn", ["wilcoxon_test", "t_test"])
def test_a_row_does_not_depend_on_the_other_perturbations_in_the_run(tmp_path, fn):
    """A gene excluded here must not inherit a p-value from another row's block."""
    path = _make_dataset_partly_tested(tmp_path)
    both = getattr(cx, fn)(path, output_dir=tmp_path, data_name=f"{fn}_both", **_UNTESTED_KW)
    alone = getattr(cx, fn)(
        path, output_dir=tmp_path, data_name=f"{fn}_alone",
        perturbations=["KO1"], **_UNTESTED_KW,
    )
    row_both = both.groups.index("KO1")
    row_alone = alone.groups.index("KO1")
    for attr in ("pvalues", "statistics", "logfoldchanges", "effect_size"):
        np.testing.assert_allclose(
            np.asarray(getattr(both, attr)[row_both]),
            np.asarray(getattr(alone, attr)[row_alone]),
            atol=1e-10,
            err_msg=f"{attr} for KO1 changed with the run's composition",
        )


def test_wilcoxon_paths_agree_on_untested_genes(tmp_path):
    path = _make_dataset_partly_tested(tmp_path)
    standard = cx.wilcoxon_test(
        path, output_dir=tmp_path, data_name="std", memory_limit_gb=128, **_UNTESTED_KW
    )
    streaming = cx.wilcoxon_test(
        path, output_dir=tmp_path, data_name="stream", memory_limit_gb=1e-7, **_UNTESTED_KW
    )
    assert np.isnan(standard.pvalues).any()
    for attr in ("pvalues", "statistics", "logfoldchanges", "effect_size", "pts", "pts_rest"):
        np.testing.assert_allclose(
            np.asarray(getattr(standard, attr)),
            np.asarray(getattr(streaming, attr)),
            atol=1e-10,
            err_msg=f"{attr} differs between the standard and streaming paths",
        )
    _assert_untested_is_uniform(standard, standard.groups.index("KO1"))
    _assert_untested_is_uniform(streaming, streaming.groups.index("KO1"))
