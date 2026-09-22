"""Unit tests for the shared batched-IRLS numerics in :mod:`crispyx._irls`."""
from __future__ import annotations

import numpy as np
import pytest

from crispyx._irls import (
    EPS,
    ETA_MAX,
    ETA_MIN,
    Deviance,
    eta_floor,
    gram_batched,
    irls_weights,
    precondition_columns,
    working_response,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------
# gram_batched
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_features", [1, 2, 5, 17, 41])
def test_gram_matches_einsum_reference(rng, n_features):
    """The BLAS form reproduces the three-operand einsum it replaces."""
    n, p = 137, 23
    design = rng.normal(size=(n, n_features))
    weights = rng.random((n, p)) + 0.05

    expected = np.einsum("ki,kg,kj->gij", design, weights, design, optimize=True)
    np.testing.assert_allclose(gram_batched(design, weights), expected, rtol=1e-12, atol=1e-12)


def test_gram_matches_per_gene_loop(rng):
    """And the naive per-gene definition, which is what it actually means."""
    n, p, d = 90, 7, 6
    design = rng.normal(size=(n, d))
    weights = rng.random((n, p)) + 0.05

    got = gram_batched(design, weights)
    for g in range(p):
        np.testing.assert_allclose(got[g], design.T @ (weights[:, g, None] * design), rtol=1e-12)


def test_gram_is_invariant_to_the_chunk_budget(rng):
    """Cell chunking is an implementation detail, not a numerical choice."""
    n, p, d = 500, 11, 9
    design = rng.normal(size=(n, d))
    weights = rng.random((n, p)) + 0.05

    whole = gram_batched(design, weights, target_mb=1e6)
    for target_mb in (1e-6, 1e-4, 1e-2, 1.0):
        chunked = gram_batched(design, weights, target_mb=target_mb)
        np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=1e-13)


def test_gram_survives_a_budget_smaller_than_one_row(rng):
    """The chunk size floors at one row rather than dividing by zero."""
    n, p, d = 400, 3, 8
    design = rng.normal(size=(n, d))
    weights = rng.random((n, p)) + 0.05

    out = gram_batched(design, weights, target_mb=0.0)
    np.testing.assert_allclose(out, gram_batched(design, weights, target_mb=1e6), rtol=1e-12)


def test_gram_adds_ridge_to_the_diagonal_only(rng):
    n, p, d = 50, 4, 5
    design = rng.normal(size=(n, d))
    weights = rng.random((n, p)) + 0.05

    plain = gram_batched(design, weights)
    ridged = gram_batched(design, weights, ridge=0.25)
    np.testing.assert_allclose(ridged - plain, np.tile(0.25 * np.eye(d), (p, 1, 1)), atol=1e-12)


# --------------------------------------------------------------------------
# irls_weights / working_response
# --------------------------------------------------------------------------

def test_poisson_weight_is_mu(rng):
    mu = rng.random((20, 5)) * 10 + 0.01
    np.testing.assert_allclose(irls_weights(mu, None), mu)
    np.testing.assert_allclose(irls_weights(mu, np.zeros(5)), mu)


def test_nb_weight_matches_the_variance_definition(rng):
    """W = mu^2 / V with V = mu + alpha mu^2, i.e. mu / (1 + alpha mu)."""
    mu = rng.random((20, 5)) * 10 + 0.01
    alpha = rng.random(5) + 0.1
    variance = mu + alpha[None, :] * mu**2
    np.testing.assert_allclose(irls_weights(mu, alpha), mu**2 / variance, rtol=1e-12)


def test_weights_are_not_clamped_at_a_mean_floor():
    """The regression this module exists to prevent.

    A cell with a small fitted mean must keep its small weight; clamping it at
    a 0.5 mean floor overstates its leverage by 5.5x here.
    """
    mu = np.full((1, 1), 0.1)
    w = irls_weights(mu, np.array([1.0]))
    assert w[0, 0] == pytest.approx(0.1 / 1.1)
    assert w[0, 0] < 0.5


def test_working_response_round_trips(rng):
    n, p = 30, 4
    eta = rng.normal(size=(n, p))
    mu = np.exp(eta)
    counts = rng.poisson(mu).astype(float)
    offset = rng.normal(size=n)

    z = working_response(eta + offset[:, None], counts, mu, offset)
    np.testing.assert_allclose(z, eta + (counts - mu) / mu, rtol=1e-10)


# --------------------------------------------------------------------------
# Deviance
# --------------------------------------------------------------------------

def _poisson_deviance_reference(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (term - (y - mu)).sum(axis=0)


def _nb_deviance_reference(y, mu, size):
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    yr = y + size
    return 2.0 * (term - yr * np.log(yr / (mu + size))).sum(axis=0)


def test_poisson_deviance_matches_reference(rng):
    n, p = 60, 8
    mu = rng.random((n, p)) * 5 + 0.01
    counts = rng.poisson(mu).astype(float)
    eta = np.log(mu)

    got = Deviance(counts, "poisson").total(eta, mu)
    np.testing.assert_allclose(got, _poisson_deviance_reference(counts, mu), rtol=1e-10)


def test_nb_deviance_matches_reference(rng):
    n, p = 60, 8
    mu = rng.random((n, p)) * 5 + 0.01
    alpha = rng.random(p) + 0.1
    counts = rng.poisson(mu).astype(float)
    eta = np.log(mu)

    got = Deviance(counts, "nb", alpha).total(eta, mu)
    expected = _nb_deviance_reference(counts, mu, (1.0 / alpha)[None, :])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_deviance_is_zero_at_a_saturated_fit(rng):
    """mu == y is the saturated model, whose deviance is exactly zero."""
    counts = rng.poisson(4.0, size=(40, 6)).astype(float)
    counts = np.maximum(counts, 1.0)  # log(0) is not the point of this test
    dev = Deviance(counts, "poisson").total(np.log(counts), counts)
    np.testing.assert_allclose(dev, 0.0, atol=1e-9)


@pytest.mark.parametrize("family", ["poisson", "nb"])
def test_residuals_square_to_the_total_deviance(rng, family):
    n, p = 50, 7
    mu = rng.random((n, p)) * 5 + 0.05
    alpha = rng.random(p) + 0.1
    counts = rng.poisson(mu).astype(float)
    eta = np.log(mu)

    dev = Deviance(counts, family, alpha if family == "nb" else None)
    resid = dev.residuals(eta, mu)
    np.testing.assert_allclose((resid**2).sum(axis=0), dev.total(eta, mu), rtol=1e-8)


@pytest.mark.parametrize("family", ["poisson", "nb"])
def test_subset_agrees_with_a_fresh_instance(rng, family):
    n, p = 40, 9
    mu = rng.random((n, p)) * 5 + 0.05
    alpha = rng.random(p) + 0.1
    counts = rng.poisson(mu).astype(float)
    eta = np.log(mu)
    idx = np.array([0, 3, 4, 8])

    full = Deviance(counts, family, alpha if family == "nb" else None)
    fresh = Deviance(counts[:, idx], family, alpha[idx] if family == "nb" else None)
    np.testing.assert_allclose(
        full.subset(idx).total(eta[:, idx], mu[:, idx]),
        fresh.total(eta[:, idx], mu[:, idx]),
        rtol=1e-12,
    )


def test_nb_requires_alpha():
    with pytest.raises(ValueError, match="requires alpha"):
        Deviance(np.ones((3, 2)), "nb")


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="must be 'nb' or 'poisson'"):
        Deviance(np.ones((3, 2)), "gaussian")


# --------------------------------------------------------------------------
# preconditioning and the eta floor
# --------------------------------------------------------------------------

def test_preconditioning_round_trips(rng):
    design = np.c_[np.ones(50), rng.normal(size=50) * 1e-3, rng.normal(size=50) * 1e4]
    scaled, scale = precondition_columns(design)

    np.testing.assert_allclose(scaled * scale, design, rtol=1e-12)
    np.testing.assert_allclose(np.sqrt((scaled**2).mean(axis=0)), 1.0, rtol=1e-12)


def test_preconditioning_leaves_an_intercept_alone(rng):
    design = np.c_[np.ones(20), rng.normal(size=20)]
    scaled, scale = precondition_columns(design)
    assert scale[0] == pytest.approx(1.0)
    np.testing.assert_allclose(scaled[:, 0], 1.0)


def test_preconditioning_survives_a_zero_column(rng):
    design = np.c_[np.ones(20), np.zeros(20)]
    scaled, scale = precondition_columns(design)
    assert np.all(np.isfinite(scaled))
    assert scale[1] == 1.0


def test_preconditioning_improves_conditioning(rng):
    design = np.c_[np.ones(200), rng.normal(size=200) * 1e-4]
    scaled, _ = precondition_columns(design)
    assert np.linalg.cond(scaled.T @ scaled) < np.linalg.cond(design.T @ design)


def test_eta_floor_tracks_min_mu():
    assert eta_floor(0.5) == pytest.approx(np.log(0.5))
    assert eta_floor(1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("min_mu", [0.0, None, -1.0])
def test_eta_floor_is_finite_without_a_mean_floor(min_mu):
    """log(0) is -inf; an unfloored fit still needs a usable clip."""
    assert eta_floor(min_mu) == ETA_MIN
    assert np.isfinite(eta_floor(min_mu))


def test_eta_floor_never_underflows_exp():
    assert np.exp(eta_floor(1e-300)) > 0.0
    assert eta_floor(1e-300) == ETA_MIN


# --------------------------------------------------------------------------
# NB deviance conditioning
# --------------------------------------------------------------------------

def _exact_nb_deviance(counts, mu, alpha):
    """The NB deviance of one gene, summed in 60 decimal digits.

    The point of the reference is that it never forms the difference of two
    large logarithms, so it is independent of the conditioning question the
    test is about.
    """
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    size = Decimal(1.0) / Decimal(float(alpha))
    total = Decimal(0)
    for y_i, mu_i in zip(counts, mu):
        y = Decimal(float(y_i))
        m = Decimal(float(mu_i))
        if y > 0:
            total += y * (y / m).ln()
        total += (y + size) * ((m + size) / (y + size)).ln()
    return float(2 * total)


@pytest.mark.parametrize("alpha", [1e-8, 1e-6, 1e-3])
def test_nb_deviance_is_accurate_at_a_near_poisson_dispersion(rng, alpha):
    """The NB deviance must not be lost to cancellation as ``alpha`` shrinks.

    At the ``alpha`` clip floor the size ``1 / alpha`` is 1e8, so the two
    terms ``(y + r) log(y + r)`` and ``(y + r) log(mu + r)`` are each eight
    orders of magnitude larger than the deviance they differ by.  Subtracting
    them loses more than the ``1e-6`` relative convergence tolerance, which
    leaves such a gene unable to ever converge.
    """
    n = 600
    counts = rng.poisson(5.0, size=(n, 1)).astype(float)
    mu = 5.0 * np.exp(rng.normal(0, 0.1, size=(n, 1)))

    got = Deviance(counts, "nb", np.array([alpha])).total(np.log(mu), mu)[0]
    expected = _exact_nb_deviance(counts[:, 0], mu[:, 0], alpha)
    assert abs(got - expected) < 1e-6 * max(abs(expected), 1.0)


def test_nb_deviance_approaches_the_poisson_deviance(rng):
    """The Poisson deviance is the ``alpha -> 0`` limit, and is exactly what a
    near-Poisson gene's NB deviance has to reproduce."""
    n = 400
    counts = rng.poisson(3.0, size=(n, 3)).astype(float)
    mu = 3.0 * np.exp(rng.normal(0, 0.1, size=(n, 3)))

    nb = Deviance(counts, "nb", np.full(3, 1e-10)).total(np.log(mu), mu)
    poisson = Deviance(counts, "poisson").total(np.log(mu), mu)
    np.testing.assert_allclose(nb, poisson, rtol=1e-7)


def test_residuals_stay_consistent_with_the_total_at_a_tiny_dispersion(rng):
    n = 300
    counts = rng.poisson(4.0, size=(n, 2)).astype(float)
    mu = 4.0 * np.exp(rng.normal(0, 0.1, size=(n, 2)))
    deviance = Deviance(counts, "nb", np.full(2, 1e-8))

    residuals = deviance.residuals(np.log(mu), mu)
    np.testing.assert_allclose(
        (residuals**2).sum(axis=0), deviance.total(np.log(mu), mu), rtol=1e-6
    )


# --------------------------------------------------------------------------
# per-column ridge and pre-sliced subsets
# --------------------------------------------------------------------------

def test_gram_ridge_accepts_a_per_column_penalty(rng):
    """A preconditioned design needs a different penalty per column, so that
    the penalty means the same thing in the caller's parameterisation."""
    n, p, d = 80, 5, 4
    design = rng.normal(size=(n, d))
    weights = rng.random((n, p)) + 0.05
    ridge = np.array([0.1, 0.5, 2.0, 7.0])

    got = gram_batched(design, weights, ridge=ridge)
    expected = gram_batched(design, weights) + np.diag(ridge)[None, :, :]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
