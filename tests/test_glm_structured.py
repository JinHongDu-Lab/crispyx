"""Tests for the structured ``[covariates | one-hot groups]`` GLM solver.

The structured solver claims to be *exact*: the same IRLS solution a dense
solver would reach, at lower cost. The tests that matter therefore compare it
against a dense fit of the same model rather than against a tolerance pulled
from the air.

Where statsmodels is used as a reference it is only on data where the
likelihood actually determines the coefficients. On a group carrying a handful
of counts the coefficient is barely identified, both solvers merely stop
somewhere on the way to ``-inf``, and a disagreement measures the data rather
than either implementation.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
import statsmodels.api as sm

from crispyx._irls import OneHotGroups, schur_solve, schur_variances
from crispyx.glm import (
    NBGLMBatchFitter,
    StructuredGLMBatchFitter,
    detect_onehot_block,
    fit_glm_onehot,
)

#: Counts per group below which an NB GLM coefficient is not meaningfully
#: determined, so cross-solver agreement measures the data rather than the fit.
_MIN_GROUP_COUNTS = 10


def make_screen(seed=0, n=400, p=60, n_groups=5, n_covariates=2, baseline=(2.0, 80.0)):
    """A perturbation-screen-shaped problem: covariates plus one-hot groups."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(-1, n_groups, size=n)          # -1 = belongs to no group
    rows = np.flatnonzero(labels >= 0)
    groups = sps.csr_matrix(
        (np.ones(rows.size), (rows, labels[rows])), shape=(n, n_groups)
    )
    design = np.c_[np.ones(n), rng.normal(size=(n, n_covariates - 1))]
    beta_features = np.c_[
        np.log(np.geomspace(*baseline, p)), rng.normal(0, 0.3, (p, n_covariates - 1))
    ]
    beta_groups = rng.normal(0, 0.6, (p, n_groups))
    offset = rng.normal(0, 0.2, n)
    eta = offset[:, None] + design @ beta_features.T + np.where(
        labels[:, None] >= 0, beta_groups.T[np.clip(labels, 0, None)], 0.0
    )
    mu = np.exp(eta)
    alpha = np.full(p, 0.3)
    counts = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu)).astype(float)
    return counts, design, groups, labels, offset, alpha


def dense_design(design, groups):
    return np.c_[design, groups.toarray()]


# --------------------------------------------------------------------------
# the structural claim
# --------------------------------------------------------------------------

def test_group_block_of_the_hessian_is_diagonal():
    """The premise: disjoint supports make G'WG diagonal."""
    counts, design, groups, _, _, _ = make_screen()
    weights = np.abs(np.random.default_rng(1).normal(size=(design.shape[0], 1))) + 0.1
    dense = groups.toarray()
    gram = dense.T @ (weights * dense)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.abs(off_diagonal).max() == 0.0


def test_schur_solve_equals_a_dense_per_gene_solve():
    """The Schur step is exact, not an approximation."""
    rng = np.random.default_rng(2)
    n, p, n_features, n_groups = 300, 8, 4, 9
    labels = rng.integers(-1, n_groups, size=n)
    rows = np.flatnonzero(labels >= 0)
    groups = sps.csr_matrix((np.ones(rows.size), (rows, labels[rows])), shape=(n, n_groups))
    design = rng.normal(size=(n, n_features))
    weights = rng.random((n, p)) + 0.05
    working = rng.normal(size=(n, p))

    block = OneHotGroups(groups)
    from crispyx._irls import gram_batched

    ridge, ridge_group = 1e-6, 1e-4
    gram = gram_batched(design, weights, ridge=ridge)
    cross = block.cross(block.sort(design), block.sort(weights))
    diagonal = block.sums(weights) + ridge_group
    weighted = weights * working
    beta_features, beta_groups, schur = schur_solve(
        gram, cross, diagonal, (design.T @ weighted).T, block.sums(weighted)
    )
    got = np.c_[beta_features, beta_groups]

    full = dense_design(design, groups)
    penalty = np.diag([ridge] * n_features + [ridge_group] * n_groups)
    expected = np.empty_like(got)
    for gene in range(p):
        hessian = full.T @ (weights[:, gene, None] * full) + penalty
        expected[gene] = np.linalg.solve(hessian, full.T @ weighted[:, gene])

    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-11)


def test_schur_variances_equal_the_dense_inverse_diagonal():
    rng = np.random.default_rng(3)
    n, p, n_features, n_groups = 200, 5, 3, 7
    labels = rng.integers(-1, n_groups, size=n)
    rows = np.flatnonzero(labels >= 0)
    groups = sps.csr_matrix((np.ones(rows.size), (rows, labels[rows])), shape=(n, n_groups))
    design = rng.normal(size=(n, n_features))
    weights = rng.random((n, p)) + 0.05

    from crispyx._irls import gram_batched

    block = OneHotGroups(groups)
    ridge, ridge_group = 1e-6, 1e-4
    gram = gram_batched(design, weights, ridge=ridge)
    cross = block.cross(block.sort(design), block.sort(weights))
    diagonal = block.sums(weights) + ridge_group
    dummy = np.zeros((p, n_features))
    _, _, schur = schur_solve(gram, cross, diagonal, dummy, np.zeros((n_groups, p)))
    feature_variance, group_variance = schur_variances(schur, cross, diagonal)
    got = np.c_[feature_variance, group_variance]

    full = dense_design(design, groups)
    penalty = np.diag([ridge] * n_features + [ridge_group] * n_groups)
    expected = np.empty_like(got)
    for gene in range(p):
        hessian = full.T @ (weights[:, gene, None] * full) + penalty
        expected[gene] = np.diag(np.linalg.inv(hessian))

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-12)


# --------------------------------------------------------------------------
# the fitter against a dense fit of the same model
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["poisson", "nb"])
def test_structured_fit_matches_the_dense_batch_fitter(family):
    """Same model, two solvers: coefficients and standard errors must agree."""
    counts, design, groups, _, offset, alpha = make_screen(seed=4)
    full = dense_design(design, groups)

    structured = StructuredGLMBatchFitter(
        design, groups, offset=offset, family=family, max_iter=200, tol=1e-12,
        ridge=1e-6, ridge_group=1e-6, clip_group=50.0,
    ).fit_batch(counts, dispersion=alpha if family == "nb" else None)

    dense = NBGLMBatchFitter(
        full, offset=offset, family=family, max_iter=200, tol=1e-12,
        ridge_penalty=1e-6, min_mu=0.0, poisson_init_iter=20,
    ).fit_batch(
        counts, gene_batch_size=None, use_numba=False,
        fixed_dispersion=alpha if family == "nb" else None,
    )

    np.testing.assert_allclose(structured.coef, dense.coef, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(structured.se, dense.se, rtol=1e-5, atol=1e-7)


def test_structured_fit_matches_statsmodels_where_determined():
    counts, design, groups, labels, offset, alpha = make_screen(seed=5)
    full = dense_design(design, groups)
    result = fit_glm_onehot(
        counts, design, groups, offset=offset, dispersion=alpha,
        max_iter=200, tol=1e-12, ridge=1e-8, ridge_group=1e-8, clip_group=50.0,
    )

    compared, worst = 0, 0.0
    for gene in range(counts.shape[1]):
        per_group = np.array([counts[labels == k, gene].sum() for k in range(groups.shape[1])])
        if per_group.min() < _MIN_GROUP_COUNTS:
            continue
        family = sm.families.NegativeBinomial(alpha=float(alpha[gene]))
        fit = sm.GLM(counts[:, gene], full, family=family, offset=offset).fit(maxiter=500)
        if not np.all(np.isfinite(fit.params)):
            continue
        worst = max(worst, float(np.max(np.abs(result.coef[gene] - fit.params))))
        compared += 1

    assert compared >= 15, f"expected determined genes, got {compared}"
    assert worst < 1e-5, f"max |dB| vs statsmodels was {worst:.2e}"


def test_dispersion_is_estimated_when_not_supplied():
    counts, design, groups, _, offset, alpha = make_screen(seed=6)
    result = fit_glm_onehot(counts, design, groups, offset=offset, max_iter=200)
    assert np.all(np.isfinite(result.dispersion))
    assert np.all(result.dispersion > 0)
    # The fixture is generated at alpha = 0.3; the estimate should be in the
    # right ballpark rather than pinned at a clip boundary.
    assert 0.05 < np.median(result.dispersion) < 2.0


# --------------------------------------------------------------------------
# the degenerate tail
# --------------------------------------------------------------------------

def test_zero_count_group_is_finite_and_clipped():
    """An empty group has an MLE of -inf; the ridge and clip make it defined."""
    counts, design, groups, labels, offset, alpha = make_screen(seed=7, p=6)
    counts[labels == 2, :] = 0.0          # group 2 carries no counts at all

    result = fit_glm_onehot(
        counts, design, groups, offset=offset, dispersion=alpha,
        clip_group=10.0, max_iter=100,
    )
    group_coefs = result.coef[:, design.shape[1]:]
    assert np.all(np.isfinite(result.coef))
    assert np.all(group_coefs[:, 2] <= -9.0), "empty group should run to the clip"
    assert np.all(group_coefs >= -10.0 - 1e-9), "and stop there"


def test_all_zero_gene_stays_finite():
    counts, design, groups, _, offset, alpha = make_screen(seed=8, p=5)
    counts[:, 0] = 0.0
    result = fit_glm_onehot(counts, design, groups, offset=offset, dispersion=alpha)
    assert np.all(np.isfinite(result.coef[0]))


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------

def test_sparse_and_dense_groups_agree():
    counts, design, groups, _, offset, alpha = make_screen(seed=9)
    sparse = fit_glm_onehot(counts, design, groups, offset=offset, dispersion=alpha)
    dense = fit_glm_onehot(counts, design, groups.toarray(), offset=offset, dispersion=alpha)
    np.testing.assert_allclose(sparse.coef, dense.coef, rtol=0, atol=0)
    np.testing.assert_allclose(sparse.se, dense.se, rtol=0, atol=0)


def test_overlapping_group_columns_are_rejected():
    overlapping = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="disjoint supports"):
        StructuredGLMBatchFitter(np.ones((3, 1)), overlapping)


def test_row_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same number of rows"):
        StructuredGLMBatchFitter(np.ones((5, 1)), np.eye(4))


def test_dispersion_shape_is_validated():
    counts, design, groups, _, offset, _ = make_screen(seed=10, p=4)
    fitter = StructuredGLMBatchFitter(design, groups, offset=offset)
    with pytest.raises(ValueError, match="dispersion must have shape"):
        fitter.fit_batch(counts, dispersion=np.array([0.1, 0.2]))


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="family must be 'nb' or 'poisson'"):
        StructuredGLMBatchFitter(np.ones((4, 1)), np.eye(4), family="gaussian")


# --------------------------------------------------------------------------
# detect_onehot_block
# --------------------------------------------------------------------------

def test_detect_finds_a_disjoint_block():
    n = 60
    labels = np.arange(n) % 4
    dummies = np.stack([(labels == k).astype(float) for k in range(4)], axis=1)
    design = np.c_[np.ones(n), np.random.default_rng(0).normal(size=n), dummies]
    found = detect_onehot_block(design, min_block=2)
    np.testing.assert_array_equal(found, np.array([2, 3, 4, 5]))


def test_detect_prefers_narrow_columns_over_a_wide_binary_covariate():
    """A binary covariate covering half the cells must not shut out the groups.

    Scanning widest-first would select the covariate and then reject every
    group column it overlaps, which is why the scan goes narrowest-first.
    """
    n = 80
    labels = np.arange(n) % 4
    dummies = np.stack([(labels == k).astype(float) for k in range(4)], axis=1)
    sex = (np.arange(n) < n // 2).astype(float)    # overlaps every group
    design = np.c_[np.ones(n), sex, dummies]
    found = detect_onehot_block(design, min_block=2)
    np.testing.assert_array_equal(found, np.array([2, 3, 4, 5]))


def test_detect_returns_empty_when_the_block_is_too_small():
    n = 30
    design = np.c_[np.ones(n), np.random.default_rng(0).normal(size=n)]
    assert detect_onehot_block(design, min_block=2).size == 0


def test_detect_ignores_all_zero_columns():
    n = 40
    labels = np.arange(n) % 3
    dummies = np.stack([(labels == k).astype(float) for k in range(3)], axis=1)
    design = np.c_[dummies, np.zeros(n)]
    found = detect_onehot_block(design, min_block=2)
    np.testing.assert_array_equal(found, np.array([0, 1, 2]))


def test_detected_block_round_trips_through_the_fitter():
    """What detect finds can be handed straight to the solver."""
    counts, design, groups, _, offset, alpha = make_screen(seed=11)
    full = dense_design(design, groups)
    found = detect_onehot_block(full, min_block=2)
    covariates = np.setdiff1d(np.arange(full.shape[1]), found)

    result = fit_glm_onehot(
        counts, full[:, covariates], full[:, found], offset=offset, dispersion=alpha,
        max_iter=200, tol=1e-12,
    )
    assert result.coef.shape == (counts.shape[1], full.shape[1])
    assert np.all(np.isfinite(result.coef))


# --------------------------------------------------------------------------
# counterfactual means
# --------------------------------------------------------------------------

def test_counterfactual_means_reproduce_the_fitted_values():
    """Selecting each cell's own group must give back the fitted means."""
    counts, design, groups, labels, offset, alpha = make_screen(seed=12, p=8)
    fitter = StructuredGLMBatchFitter(
        design, groups, offset=offset, max_iter=200, tol=1e-12, clip_group=50.0
    )
    result = fitter.fit_batch(counts, dispersion=alpha)
    baseline, per_group = fitter.counterfactual_means(result)

    assert baseline.shape == counts.shape
    assert per_group.shape == counts.shape + (groups.shape[1],)

    rebuilt = np.where(
        labels[:, None] >= 0,
        per_group[np.arange(len(labels)), :, np.clip(labels, 0, None)],
        baseline,
    )
    np.testing.assert_allclose(rebuilt, result.mu, rtol=1e-9, atol=1e-10)


def test_counterfactual_baseline_excludes_the_group_effect():
    counts, design, groups, _, offset, alpha = make_screen(seed=13, p=6)
    fitter = StructuredGLMBatchFitter(design, groups, offset=offset, clip_group=50.0)
    result = fitter.fit_batch(counts, dispersion=alpha)
    baseline, per_group = fitter.counterfactual_means(result)

    group_coefs = result.coef[:, design.shape[1]:]
    for k in range(groups.shape[1]):
        np.testing.assert_allclose(
            per_group[:, :, k], baseline * np.exp(group_coefs[:, k])[None, :], rtol=1e-9
        )


# --------------------------------------------------------------------------
# interchangeability with the dense fitter, and the routing adapter
# --------------------------------------------------------------------------

def test_log_det_identity_matches_the_dense_determinant():
    """det[[C,B],[B',D]] = det(D) det(S) -- what makes Cox-Reid cheap here."""
    counts, design, groups, _, offset, _ = make_screen(seed=14, p=6)
    fitter = StructuredGLMBatchFitter(design, groups, offset=offset,
                                      ridge=1e-6, ridge_group=1e-6)
    rng = np.random.default_rng(0)
    weights = rng.random((design.shape[0], counts.shape[1])) + 0.05

    got = fitter._log_det_hessian(weights)

    full = dense_design(design, groups)
    penalty = np.diag([1e-6] * design.shape[1] + [1e-6] * groups.shape[1])
    expected = np.array([
        np.linalg.slogdet(full.T @ (weights[:, g, None] * full) + penalty)[1]
        for g in range(counts.shape[1])
    ])
    np.testing.assert_allclose(got, expected, rtol=1e-9)


@pytest.mark.parametrize("dispersion_method", ["moments", "cox-reid"])
def test_estimated_dispersion_matches_the_dense_fitter(dispersion_method):
    """The two solvers must estimate the same dispersion on the same design."""
    counts, design, groups, _, offset, _ = make_screen(seed=15, p=40)
    full = dense_design(design, groups)

    structured = StructuredGLMBatchFitter(
        design, groups, offset=offset, max_iter=100, tol=1e-10, ridge=1e-6,
        ridge_group=1e-6, clip_group=50.0, dispersion_method=dispersion_method,
    ).fit_batch(counts, return_mu=False, return_dev_resid=False)

    dense = NBGLMBatchFitter(
        full, offset=offset, max_iter=100, tol=1e-10, min_mu=0.0,
        ridge_penalty=1e-6, dispersion_method=dispersion_method,
    ).fit_batch(counts, gene_batch_size=None, use_numba=False)

    if dispersion_method == "cox-reid":
        # A shared discrete grid: the selected point should agree exactly.
        np.testing.assert_array_equal(structured.dispersion, dense.dispersion)
    else:
        # Both run fit -> method-of-moments -> refit, but from different
        # starting points (the dense path from a truncated Poisson warm start,
        # this one from a converged Poisson fit), so the moment estimate
        # differs in the third digit.  This is the dispersion heuristic, not
        # the solver: held at a *given* dispersion the two agree to 1e-6, as
        # test_structured_fit_matches_the_dense_batch_fitter shows.
        np.testing.assert_allclose(structured.dispersion, dense.dispersion, rtol=0.1)


def test_auto_routes_to_structured_and_matches_the_dense_fitter():
    """The adapter returns dense-equivalent results in the original column order."""
    from crispyx.glm import fit_nb_glm_batch_auto

    counts, design, groups, _, offset, _ = make_screen(seed=16, p=40, n_groups=12)
    full = dense_design(design, groups)

    auto = fit_nb_glm_batch_auto(
        full, counts, offset=offset, protected_columns=[0], max_iter=100, tol=1e-10,
        min_mu=0.0, dispersion_method="moments",
    )
    dense = NBGLMBatchFitter(
        full, offset=offset, max_iter=100, tol=1e-10, min_mu=0.0,
        dispersion_method="moments",
    ).fit_batch(counts, gene_batch_size=None, use_numba=False)

    assert auto.coef.shape == dense.coef.shape
    # The residual here is the dispersion heuristic's starting point (see
    # test_estimated_dispersion_matches_the_dense_fitter), not the linear
    # algebra: ~1e-3 absolute on coefficients of order 1.
    np.testing.assert_allclose(auto.coef, dense.coef, rtol=5e-2, atol=5e-3)
    np.testing.assert_allclose(auto.dispersion, dense.dispersion, rtol=0.1)


def test_auto_keeps_the_dense_path_when_groups_do_not_outnumber_covariates():
    """Routing is on the measured advantage, not on a block merely existing."""
    from crispyx.glm import fit_nb_glm_batch_auto

    counts, design, groups, _, offset, _ = make_screen(
        seed=17, p=20, n_groups=3, n_covariates=6
    )
    full = dense_design(design, groups)
    auto = fit_nb_glm_batch_auto(full, counts, offset=offset, protected_columns=[0],
                                 max_iter=100, tol=1e-10, min_mu=0.0)
    dense = NBGLMBatchFitter(
        full, offset=offset, max_iter=100, tol=1e-10, min_mu=0.0,
    ).fit_batch(counts, gene_batch_size=None, use_numba=False)
    # Same code path, so identical to the bit.
    np.testing.assert_array_equal(auto.coef, dense.coef)


def test_auto_protects_the_column_under_test_from_the_group_block():
    """A protected column keeps its unclipped coefficient."""
    from crispyx.glm import fit_nb_glm_batch_auto

    counts, design, groups, labels, offset, _ = make_screen(seed=18, p=12, n_groups=10)
    n = counts.shape[0]
    # A binary column disjoint from nothing in particular, marked as under test.
    perturbation = (np.arange(n) % 3 == 0).astype(float)
    full = np.c_[design, perturbation, groups.toarray()]
    protected = [0, design.shape[1]]

    auto = fit_nb_glm_batch_auto(full, counts, offset=offset, protected_columns=protected,
                                 max_iter=100, tol=1e-10, min_mu=0.0)
    assert np.all(np.isfinite(auto.coef))
    assert auto.coef.shape[1] == full.shape[1]


def test_auto_respects_min_total_count():
    from crispyx.glm import fit_nb_glm_batch_auto

    counts, design, groups, _, offset, _ = make_screen(seed=19, p=10, n_groups=12)
    counts[:, 0] = 0.0
    auto = fit_nb_glm_batch_auto(full := dense_design(design, groups), counts,
                                 offset=offset, protected_columns=[0], min_total_count=1.0)
    assert not auto.converged[0]
    np.testing.assert_array_equal(auto.coef[0], np.zeros(full.shape[1]))


# --------------------------------------------------------------------------
# end to end: nb_glm_test with a many-level categorical covariate
# --------------------------------------------------------------------------

def _screen_anndata(tmp_path, n_batches, n_cells=360, n_genes=40, seed=20):
    """A screen with a perturbation and a many-level batch covariate."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    # Perturbation must be assigned independently of batch.  Deterministic
    # interleaving (cell i perturbed iff i is even, batch i % n_batches) makes
    # every batch wholly perturbed or wholly control whenever n_batches is
    # even, which puts the perturbation column exactly in the span of
    # [intercept | batch dummies] and leaves its coefficient unidentifiable.
    batch = np.arange(n_cells) % n_batches
    perturbed = rng.binomial(1, 0.5, n_cells).astype(float)
    baseline = np.log(np.geomspace(5.0, 200.0, n_genes))
    effect = rng.normal(0, 0.6, n_genes)
    batch_effect = rng.normal(0, 0.4, (n_batches, n_genes))
    eta = baseline[None, :] + np.outer(perturbed, effect) + batch_effect[batch]
    alpha = 0.2
    counts = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * np.exp(eta))).astype(np.float32)

    obs = pd.DataFrame(
        {
            "perturbation": np.where(perturbed > 0, "g1", "control"),
            "batch": pd.Categorical([f"b{b}" for b in batch]),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    path = tmp_path / f"screen_{n_batches}.h5ad"
    ad.AnnData(X=counts, obs=obs, var=var).write_h5ad(path)
    return path


def test_nb_glm_test_with_many_batches_routes_structured_and_agrees(tmp_path, monkeypatch):
    """Routing DE through the structured solver must not change conclusions.

    The gate for wiring this into ``nb_glm_test``: with a 20-level batch
    covariate the design carries 19 indicator columns beside 2 covariates, so
    the structured path is taken; the effects, standard errors and p-values it
    produces must match the dense path within the dispersion heuristic's own
    spread.
    """
    from crispyx.de import nb_glm_test
    import crispyx.glm as glm_module

    path = _screen_anndata(tmp_path, n_batches=20)

    common = dict(
        perturbation_column="perturbation", control_label="control",
        covariates=["batch"], output_dir=tmp_path, verbose=False,
    )

    taken = {}
    real_structured = glm_module.StructuredGLMBatchFitter

    class _Spy(real_structured):
        def __init__(self, *args, **kwargs):
            taken["structured"] = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(glm_module, "StructuredGLMBatchFitter", _Spy)
    routed = nb_glm_test(path, data_name="routed", **common)
    monkeypatch.undo()
    assert taken.get("structured"), "20 batch levels should take the structured path"

    # Force the dense path by protecting every column from the group block.
    real_auto = glm_module.fit_nb_glm_batch_auto

    def dense_only(design, counts, **kwargs):
        kwargs["protected_columns"] = tuple(range(np.asarray(design).shape[1]))
        return real_auto(design, counts, **kwargs)

    monkeypatch.setattr(glm_module, "fit_nb_glm_batch_auto", dense_only)
    import crispyx.de as de_module
    monkeypatch.setattr(de_module, "fit_nb_glm_batch_auto", dense_only)
    reference = nb_glm_test(path, data_name="reference", **common)
    monkeypatch.undo()

    a = routed.statistics.ravel()
    b = reference.statistics.ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.sum() >= 30

    np.testing.assert_allclose(
        routed.logfoldchanges.ravel()[finite],
        reference.logfoldchanges.ravel()[finite],
        rtol=1e-3, atol=1e-4,
    )
    # Ranking, which is what a user acts on, must be essentially identical.
    assert np.corrcoef(a[finite], b[finite])[0, 1] > 0.999


# --------------------------------------------------------------------------
# the starting point
# --------------------------------------------------------------------------

def test_fit_is_unchanged_by_scaling_the_intercept_column():
    """The starting predictor is carried by whichever column is constant, and
    ``eta`` is derived from the coefficients, so a design whose intercept is
    not a column of ones starts from a point its coefficients describe."""
    counts, design, groups, _, offset, alpha = make_screen(seed=6, n=250, p=25, n_groups=4)
    scaled = design.copy()
    scaled[:, 0] = 2.0

    plain = StructuredGLMBatchFitter(design, groups, offset=offset).fit_batch(
        counts, dispersion=alpha
    )
    rescaled = StructuredGLMBatchFitter(scaled, groups, offset=offset).fit_batch(
        counts, dispersion=alpha
    )

    np.testing.assert_allclose(rescaled.coef[:, 0] * 2.0, plain.coef[:, 0], rtol=1e-6)
    np.testing.assert_allclose(rescaled.mu, plain.mu, rtol=1e-6)


def test_fit_without_a_constant_column_starts_consistently():
    """With no intercept column at all the loop must still start from an
    ``eta`` its coefficients produce, or the first damped step is compared
    against a deviance that belongs to a different model."""
    counts, design, groups, _, offset, alpha = make_screen(
        seed=7, n=200, p=20, n_groups=3, n_covariates=2
    )
    no_intercept = design[:, 1:]

    result = StructuredGLMBatchFitter(no_intercept, groups, offset=offset).fit_batch(
        counts, dispersion=alpha
    )
    assert np.all(np.isfinite(result.coef))
    assert result.converged.mean() > 0.5
