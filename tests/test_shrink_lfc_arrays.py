"""tl.shrink_lfc_arrays and the closed-form apeGLM posterior mode."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import crispyx as cx
from crispyx.glm import cauchy_map, shrink_lfc_apeglm_from_stats


def _objective(b, m, se, scale):
    return (b - m) ** 2 / (2 * se ** 2) + np.log1p(b ** 2 / scale ** 2)


def test_cauchy_map_is_the_global_minimum():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 3, 3000)
    se = np.exp(rng.normal(-1, 1.5, 3000))
    for scale in (0.02, 0.4, 1.0):
        got = cauchy_map(m, se, scale)
        for i in range(0, 3000, 37):
            bracket = (min(0.0, m[i]) - 1, max(0.0, m[i]) + 1)
            best = minimize_scalar(_objective, args=(m[i], se[i], scale), bounds=bracket, method="bounded",
                                   options={"xatol": 1e-10})
            assert _objective(got[i], m[i], se[i], scale) <= best.fun + 1e-9
        assert (np.abs(got) <= np.abs(m) + 1e-12).all() and (np.sign(got) * np.sign(m) >= 0).all()


def test_cauchy_map_edge_cases_and_shapes():
    got = cauchy_map([[np.nan, 1.0], [1.0, 2.0]], [[1.0, 0.0], [np.inf, 0.5]], 0.5)
    assert got.shape == (2, 2)
    assert np.isnan(got[0, 0]) and got[0, 1] == 1.0 and np.isnan(got[1, 0])
    assert 0 < got[1, 1] < 2.0
    per_entry = cauchy_map(np.full(4, 2.0), np.full(4, 0.5), np.array([0.1, 0.5, 1.0, 10.0]))
    assert (np.diff(per_entry) > 0).all()  # a wider prior shrinks less


def test_from_stats_uses_the_closed_form():
    rng = np.random.default_rng(1)
    m, se = rng.normal(0, 2, 500), np.exp(rng.normal(-1, 1, 500))
    m[3], se[7] = np.nan, 0.0
    lfc, post_se, converged, _ = shrink_lfc_apeglm_from_stats(m, se, prior_scale=0.3, use_gene_specific_prior=False)
    valid = np.isfinite(m) & (se > 0)
    np.testing.assert_allclose(lfc[valid], cauchy_map(m, se, 0.3)[valid])
    assert converged[valid].all() and not converged[~valid].any()
    assert np.isnan(lfc[3]) and lfc[7] == m[7]  # invalid entries keep the MLE
    assert np.isfinite(post_se[valid]).all() and (post_se[valid] > 0).all()


def test_shrink_lfc_arrays_excludes_floors_and_reports_the_prior():
    rng = np.random.default_rng(2)
    lfc = rng.normal(0, 0.5, (6, 50))
    se = np.full((6, 50), 0.3)
    lfc[:, 0] = -14.0      # a floored knockdown: every count zero in the perturbed cells
    se[:, 0] = 200.0
    lfc[1, 5] = np.nan
    plain = cx.tl.shrink_lfc_arrays(lfc, se)
    assert isinstance(plain, cx.ShrunkLFC) and plain.lfc.shape == (6, 50)
    assert np.abs(plain.lfc[:, 0]).max() < 0.01          # shrunk to ~0
    assert np.isnan(plain.lfc[1, 5]) and np.isnan(plain.se[1, 5])
    assert (plain.se[np.isfinite(plain.se)] > 0).all()
    assert (np.abs(plain.lfc[np.isfinite(plain.lfc)]) <= np.abs(lfc[np.isfinite(lfc)]) + 1e-12).all()

    own = np.zeros((6, 50), dtype=bool)
    own[np.arange(6), 0] = True
    floor = np.full((6, 1), np.nan)
    floor[:3, 0] = -3.9
    kept = cx.tl.shrink_lfc_arrays(lfc, se, exclude=own, floor=floor)
    np.testing.assert_array_equal(kept.lfc[:3, 0], -3.9)           # floored and kept as given
    np.testing.assert_array_equal(kept.lfc[3:, 0], -14.0)          # no floor: kept as given
    np.testing.assert_array_equal(kept.se[:, 0], 200.0)            # an excluded entry keeps its SE
    np.testing.assert_allclose(kept.lfc[:, 1:], plain.lfc[:, 1:], atol=1e-6)  # the other entries hardly move
    assert kept.prior_scale != plain.prior_scale                   # the excluded column left the prior
    fixed = cx.tl.shrink_lfc_arrays(lfc, se, prior_scale=0.25)
    assert fixed.prior_scale == 0.25
    with pytest.raises(ValueError, match="shape"):
        cx.tl.shrink_lfc_arrays(lfc, se[:, :10])


def test_the_pooled_centre_is_unbiased_for_low_counts():
    # No effect anywhere, 5 expected reads per perturbation. A perturbation that by chance has more reads
    # has a higher fold change and a smaller SE, so a precision-weighted centre drifts up; the pooled one does not.
    rng = np.random.default_rng(0)
    n, g, expected = 2000, 20, 5.0
    k = rng.poisson(expected, (n, g)).astype(float)
    with np.errstate(divide="ignore"):
        lfc = np.where(k > 0, np.log(k / expected), -10.0)
        se = np.where(k > 0, 1 / np.sqrt(k), 100.0)
    lfc[:3], se[:3] = 9.0, 8.0                           # upward fits where the dispersion is large
    lfc[3:13], se[3:13] = 3.0, 0.2                       # a few perturbations that strongly induce every gene
    precision_weighted = (lfc / se ** 2).sum(0) / (1 / se ** 2).sum(0)
    assert np.median(precision_weighted) > 0.05          # the bias this guards against
    out = cx.tl.shrink_lfc_arrays(lfc, se, centre="pooled", n_cells=np.full(n, 50))
    assert np.abs(out.centre).max() < 0.05               # pulled by neither kind
    assert abs(np.median(np.nanmedian(out.lfc[13:], axis=0))) < 0.03
    assert out.lfc[3, 0] > 2.5                           # a strong effect keeps its size


def test_a_pooled_centre_keeps_the_shared_response():
    rng = np.random.default_rng(3)
    shared = rng.normal(0, 0.5, 40)                       # every perturbation moves these genes alike,
    shared[:3] = [2.0, -2.0, 1.5]                         # some far beyond 3 SE of zero
    lfc = shared + rng.normal(0, 0.05, (300, 40))
    se = np.full(lfc.shape, 0.2)
    lfc[0, 0], se[0, 0] = -12.0, 150.0                    # a floor: every count zero
    zero = cx.tl.shrink_lfc_arrays(lfc, se)
    pooled = cx.tl.shrink_lfc_arrays(lfc, se, centre="pooled")
    assert zero.centre is None and pooled.centre.shape == (40,)
    np.testing.assert_allclose(pooled.centre, shared, atol=0.05)
    assert np.abs(pooled.lfc.mean(0) - shared).max() < 0.05
    assert np.abs(pooled.lfc[0, 0] - pooled.centre[0]) < 0.01      # a floor goes to the centre, not 0
    assert pooled.prior_scale < zero.prior_scale                    # the prior fits the deviations

    sizes = rng.integers(10, 500, 300)                    # every entry is within 3 SE of its gene's median
    weighted = cx.tl.shrink_lfc_arrays(lfc, se, centre="pooled", n_cells=sizes)
    expected = np.log((sizes[:, None] * np.exp(lfc)).sum(0) / sizes.sum())   # the floor counts as ~0
    np.testing.assert_allclose(weighted.centre, expected)

    own = np.zeros(lfc.shape, dtype=bool)
    own[:, 1] = True                                       # an excluded column has no centre and is kept
    kept = cx.tl.shrink_lfc_arrays(lfc, se, centre="pooled", exclude=own)
    assert kept.centre[1] == 0.0
    np.testing.assert_array_equal(kept.lfc[:, 1], lfc[:, 1])

    with pytest.raises(ValueError, match="perturbations x genes"):
        cx.tl.shrink_lfc_arrays(lfc[0], se[0], centre="pooled")
    with pytest.raises(ValueError, match="n_cells"):
        cx.tl.shrink_lfc_arrays(lfc, se, centre="pooled", n_cells=sizes[:5])
    with pytest.raises(ValueError, match="centre"):
        cx.tl.shrink_lfc_arrays(lfc, se, centre="mean")
