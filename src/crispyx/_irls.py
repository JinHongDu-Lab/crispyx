"""Shared numerics for batched IRLS GLM fitting.

The GLM fitters in :mod:`crispyx.glm` all solve the same iteratively reweighted
least squares problem for many genes at once, differing only in how the normal
equations are formed and solved.  The pieces that do not differ live here so
that the fitters cannot drift apart:

* :func:`gram_batched` -- the per-gene weighted Gram matrix ``X' W X``, formed
  with BLAS in cell-sized chunks.
* :class:`Deviance` -- per-gene deviance and deviance residuals with the
  data-only terms precomputed once.
* :func:`irls_weights` and :func:`working_response` -- the log-link IRLS
  quantities for the Poisson and negative-binomial families.
* :func:`precondition_columns` -- column scaling so that the solve sees a
  well-conditioned design.

Numerical conventions
---------------------
``EPS`` guards divisions.  It is deliberately *not* the same quantity as the
``min_mu`` floor that the fitters expose: ``min_mu`` is a modelling choice
(DESeq2 floors fitted means to stabilise low-count genes), whereas ``EPS``
only keeps a division finite.  Conflating the two inflates the IRLS weight of
every cell whose fitted mean falls below the floor.

The linear predictor is clipped to ``[ETA_MIN, ETA_MAX]`` independently of
``min_mu`` so that ``min_mu=0`` remains a valid (unfloored) fit.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "EPS",
    "ETA_MIN",
    "ETA_MAX",
    "Deviance",
    "OneHotGroups",
    "eta_floor",
    "gram_batched",
    "irls_weights",
    "precondition_columns",
    "schur_complement",
    "schur_solve",
    "schur_variances",
    "working_response",
]

#: Division guard.  Not a modelling floor -- see the module docstring.
EPS = 1e-10

#: Linear-predictor clip.  ``exp(20)`` is far above any count a UMI matrix can
#: hold and ``exp(-30)`` is far below one count per cell, so the clip only fires
#: on genes that are diverging.  The upper bound is the one the fitters have
#: always used; the lower bound replaces ``log(min_mu)``, which is ``-inf`` for
#: an unfloored fit.
ETA_MIN = -30.0
ETA_MAX = 20.0

Family = Literal["nb", "poisson"]


def eta_floor(min_mu: float | None) -> float:
    """Lower clip for the linear predictor implied by a fitted-mean floor.

    Clipping ``eta`` at ``log(min_mu)`` rather than flooring ``mu`` afterwards
    keeps ``eta == log(mu)``, which the working response relies on.  When
    ``min_mu`` is zero -- an unfloored fit -- ``log(min_mu)`` is ``-inf``, so
    the generic :data:`ETA_MIN` is used instead.
    """
    if not min_mu or min_mu <= 0.0:
        return ETA_MIN
    return max(ETA_MIN, float(np.log(min_mu)))


def precondition_columns(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale design columns to unit root-mean-square.

    ``X' W X`` is formed from the design as given, so columns on very different
    scales (an intercept of ones beside a covariate with standard deviation
    0.01) make it ill-conditioned and inflate the coefficients the solve
    returns.  Fitting in the scaled space and dividing the result by the scale
    recovers the original-space coefficients exactly.

    An intercept column of ones has root-mean-square 1 and is left untouched.

    Returns
    -------
    scaled
        The design with each column divided by its scale.
    scale
        Per-column scale, shape ``(n_features,)``.  Columns that are all zero
        (or otherwise degenerate) get a scale of 1.
    """
    design = np.asarray(design, dtype=np.float64)
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return design / scale, scale


def gram_batched(
    design: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float | np.ndarray = 0.0,
    target_mb: float = 64.0,
) -> np.ndarray:
    """Per-gene weighted Gram matrices ``X' W_g X``.

    Parameters
    ----------
    design
        Design matrix, shape ``(n_samples, n_features)``.
    weights
        Per-cell, per-gene IRLS weights, shape ``(n_samples, n_genes)``.
    ridge
        Added to the diagonal of every gene's matrix.  A scalar penalises
        every column equally; an array of length ``n_features`` gives a
        per-column penalty, which is what a preconditioned design needs to
        penalise the caller's parameterisation rather than the scaled one.
    target_mb
        Memory budget for the cell-chunked outer-product buffer.  The buffer is
        ``(chunk, n_features**2)``, so the budget bounds peak memory
        independently of ``n_samples``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_genes, n_features, n_features)``.

    Notes
    -----
    The outer products ``X[k] X[k]'`` are flattened to a ``(n_samples,
    n_features**2)`` matrix so the sum over cells is a single BLAS ``gemm``
    against the weights.  The obvious ``numpy.einsum('ki,kg,kj->gij', X, W, X)``
    is a three-operand contraction that numpy stops routing through BLAS once
    the intermediate exceeds its optimisation budget, at which point it falls
    back to a nested loop -- orders of magnitude slower here once
    ``n_features`` passes ~40, for the same values.  Chunking over cells keeps
    the ``gemm`` form's peak memory bounded without giving up the BLAS call.
    """
    design = np.asarray(design, dtype=np.float64)
    n_samples, n_features = design.shape
    n_genes = weights.shape[1]

    bytes_per_row = 8 * n_features * n_features
    chunk = max(1, int(target_mb * 1e6 // max(bytes_per_row, 1)))

    out = np.zeros((n_genes, n_features * n_features), dtype=np.float64)
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        block = design[start:stop]
        outer = (block[:, :, None] * block[:, None, :]).reshape(stop - start, -1)
        out += weights[start:stop].T @ outer

    out = out.reshape(n_genes, n_features, n_features)
    if np.any(ridge):
        idx = np.arange(n_features)
        out[:, idx, idx] += ridge
    return out


def irls_weights(mu: np.ndarray, alpha: np.ndarray | float | None) -> np.ndarray:
    """Log-link IRLS weights ``mu**2 / V(mu)``.

    For the negative binomial ``V = mu + alpha mu**2`` this is
    ``mu / (1 + alpha mu)``; ``alpha=0`` (or ``None``) gives the Poisson weight
    ``mu``.  The weights are returned unclamped: clamping them at a mean floor
    is what over-weights low-count cells.
    """
    if alpha is None:
        return mu
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    if alpha_arr.ndim == 1:
        alpha_arr = alpha_arr[None, :]
    if np.all(alpha_arr == 0.0):
        return mu
    return mu / (1.0 + alpha_arr * mu)


def working_response(
    eta: np.ndarray,
    counts: np.ndarray,
    mu: np.ndarray,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Log-link working response ``eta - offset + (y - mu) / mu``."""
    z = (counts - mu) / np.maximum(mu, EPS)
    z += eta
    if offset is not None:
        z -= offset if offset.ndim == 2 else offset[:, None]
    return z


class Deviance:
    """Per-gene deviance with the data-only terms precomputed.

    With ``eta = log mu`` the unit deviances are

    * Poisson: ``2 (y log y - y eta - y + mu)``
    * negative binomial, size ``r = 1 / alpha``:
      ``2 (y log y - y eta + (y + r) log((mu + r) / (y + r)))``

    ``sum(y log y)`` does not involve ``mu`` and is taken once at
    construction, so each subsequent evaluation costs one logarithm over the
    matrix rather than three.

    The NB term is evaluated as ``(y + r) log1p((mu - y) / (y + r))`` rather
    than as the difference of ``(y + r) log(y + r)`` and ``(y + r) log(mu +
    r)``.  Those two are each of order ``r n log r`` -- at the ``alpha`` clip
    floor ``r`` is ``1e8`` -- while the deviance they differ by is of order
    the sample size, so subtracting them loses the answer to cancellation: at
    ``n = 1e5`` the error swamps the ``1e-6`` relative-deviance convergence
    test, and a near-Poisson gene can never converge.

    Parameters
    ----------
    counts
        Count matrix, shape ``(n_samples, n_genes)``.
    family
        ``"poisson"`` or ``"nb"``.
    alpha
        NB dispersion per gene, shape ``(n_genes,)``.  Required for ``"nb"``.
    """

    __slots__ = ("family", "counts", "size", "const")

    def __init__(
        self,
        counts: np.ndarray,
        family: Family,
        alpha: np.ndarray | None = None,
        *,
        _const: np.ndarray | None = None,
        _size: np.ndarray | None = None,
    ) -> None:
        if family not in ("nb", "poisson"):
            raise ValueError(f"family must be 'nb' or 'poisson', got {family!r}")
        self.family = family
        self.counts = counts
        if family == "nb":
            if _size is not None:
                self.size = _size
            else:
                if alpha is None:
                    raise ValueError("family='nb' requires alpha")
                alpha_arr = np.asarray(alpha, dtype=np.float64).ravel()
                self.size = (1.0 / np.clip(alpha_arr, EPS, 1e10))[None, :]
        else:
            self.size = None

        if _const is not None:
            self.const = _const
            return

        with np.errstate(divide="ignore", invalid="ignore"):
            ylogy = np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1.0)), 0.0)
        if family == "poisson":
            self.const = (ylogy - counts).sum(axis=0)
        else:
            self.const = ylogy.sum(axis=0)

    def subset(self, idx: np.ndarray) -> "Deviance":
        """This deviance restricted to genes ``idx``, without recomputing."""
        return Deviance(
            self.counts[:, idx],
            self.family,
            _const=self.const[idx],
            _size=None if self.size is None else self.size[:, idx],
        )

    def total(self, eta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Deviance per gene, shape ``(n_genes,)``."""
        cross = (self.counts * eta).sum(axis=0)
        if self.family == "poisson":
            return 2.0 * (self.const - cross + mu.sum(axis=0))
        yr = self.counts + self.size
        return 2.0 * (self.const - cross + (yr * np.log1p((mu - self.counts) / yr)).sum(axis=0))

    def residuals(self, eta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Signed deviance residuals, shape ``(n_samples, n_genes)``."""
        counts = self.counts
        with np.errstate(divide="ignore", invalid="ignore"):
            ylogy = np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1.0)), 0.0)
            if self.family == "poisson":
                unit = 2.0 * (ylogy - counts * eta - (counts - mu))
            else:
                yr = counts + self.size
                unit = 2.0 * (
                    ylogy - counts * eta + yr * np.log1p((mu - counts) / yr)
                )
        return np.sign(counts - mu) * np.sqrt(np.maximum(unit, 0.0))


class OneHotGroups:
    """A block of disjoint one-hot design columns, held as per-cell labels.

    When a design is ``[X | G]`` and the columns of ``G`` are one-hot with
    disjoint supports, no cell belongs to two groups, so ``G' W G`` is
    *diagonal* and the per-gene Hessian is an arrowhead matrix::

            d_X            a
        +---------+-------------------+
        |  X'WX   |      X'WG         | d_X
        +---------+-------------------+
        |  G'WX   |   D  (diagonal)   | a
        +---------+-------------------+

    That structure is what :func:`schur_solve` exploits.  This class holds the
    group layout and forms the three blocks, keeping every aggregation on a
    BLAS or sparse path.

    Cells belonging to no group (the reference level of a categorical
    covariate, or unperturbed controls) carry a label of ``-1`` and simply do
    not appear in any group sum.
    """

    __slots__ = ("n_groups", "n_samples", "labels", "_order", "_starts", "_ends", "_matrix")

    def __init__(self, groups):
        import scipy.sparse as sp

        if sp.issparse(groups):
            matrix = groups.tocsc()
            n, a = matrix.shape
            labels = np.full(n, -1, dtype=np.int64)
            for k in range(a):
                rows = matrix.indices[matrix.indptr[k]:matrix.indptr[k + 1]]
                labels[rows] = k
        else:
            dense = np.asarray(groups)
            if dense.ndim != 2:
                raise ValueError("groups must be a 2D array or sparse matrix")
            n, a = dense.shape
            nonzero = dense != 0
            if np.any(nonzero.sum(axis=1) > 1):
                raise ValueError("group columns must have disjoint supports")
            labels = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), -1).astype(np.int64)

        self.n_groups = int(a)
        self.n_samples = int(n)
        self.labels = labels
        # Cells sorted by group, so each group's cells are a contiguous slice
        # and its aggregation is one BLAS call on a contiguous block.
        self._order = np.argsort(labels, kind="stable")
        sorted_labels = labels[self._order]
        self._starts = np.searchsorted(sorted_labels, np.arange(a), side="left")
        self._ends = np.searchsorted(sorted_labels, np.arange(a), side="right")
        self._matrix = None

    @property
    def matrix(self):
        """The groups as a sparse ``(n_samples, n_groups)`` CSR matrix."""
        if self._matrix is None:
            import scipy.sparse as sp

            rows = np.flatnonzero(self.labels >= 0)
            self._matrix = sp.csr_matrix(
                (np.ones(rows.size), (rows, self.labels[rows])),
                shape=(self.n_samples, self.n_groups),
            )
        return self._matrix

    def sort(self, array: np.ndarray) -> np.ndarray:
        """Reorder an ``(n_samples, ...)`` array into group-contiguous order."""
        return np.ascontiguousarray(array[self._order])

    def sums(self, values: np.ndarray) -> np.ndarray:
        """``G' M`` -- per-group sums of an ``(n_samples, n_genes)`` array.

        Returns shape ``(n_groups, n_genes)``.  This is the diagonal of
        ``G' W G`` when ``values`` are the IRLS weights.
        """
        return np.asarray(self.matrix.T @ values)

    def cross(self, design_sorted: np.ndarray, weights_sorted: np.ndarray) -> np.ndarray:
        """``X' W G`` per gene, shape ``(n_groups, n_features, n_genes)``.

        Both inputs must already be in :meth:`sort` order.  Each group is one
        ``(n_features, n_k) @ (n_k, n_genes)`` matrix product, so the weight
        matrix is read once in total.  Forming this as ``d_X`` separate sparse
        products instead re-reads the weights ``d_X`` times, which is where
        the memory traffic (and the runtime) of this step goes.
        """
        n_features = design_sorted.shape[1]
        n_genes = weights_sorted.shape[1]
        out = np.empty((self.n_groups, n_features, n_genes), dtype=np.float64)
        for k in range(self.n_groups):
            start, stop = self._starts[k], self._ends[k]
            if stop > start:
                out[k] = design_sorted[start:stop].T @ weights_sorted[start:stop]
            else:
                out[k] = 0.0
        return out

    def expand(self, per_group: np.ndarray) -> np.ndarray:
        """Scatter ``(n_genes, n_groups)`` values back to ``(n_samples, n_genes)``."""
        out = np.zeros((self.n_samples, per_group.shape[0]), dtype=np.float64)
        assigned = self.labels >= 0
        out[assigned] = per_group.T[self.labels[assigned]]
        return out


def schur_complement(
    gram: np.ndarray, cross: np.ndarray, diagonal: np.ndarray
) -> np.ndarray:
    """``S = C - B D^-1 B'`` per gene, shape ``(n_genes, n_features, n_features)``.

    Computed as a batched ``gemm`` on a ``(n_genes, n_features, n_groups)``
    layout.  The obvious ``numpy.einsum('kip,kjp,kp->pij', B, B, 1/D)``
    produces bit-identical values several times slower, because it does not
    reach BLAS.
    """
    blocks = np.ascontiguousarray(cross.transpose(2, 1, 0))
    scaled = blocks * (1.0 / diagonal).T[:, None, :]
    return gram - np.matmul(scaled, blocks.transpose(0, 2, 1))


def schur_solve(
    gram: np.ndarray,
    cross: np.ndarray,
    diagonal: np.ndarray,
    rhs_features: np.ndarray,
    rhs_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the arrowhead Newton system by the Schur complement of ``D``.

    With ``H = [[C, B], [B', D]]`` and ``D`` diagonal, the solution of
    ``H [bx; ba] = [rx; ra]`` is::

        S  = C - B D^-1 B'
        bx = S^-1 (rx - B D^-1 ra)
        ba = D^-1 (ra - B' bx)

    which replaces a ``(d_X + a)`` dense factorisation per gene with a
    ``d_X`` one plus a division, and is exact to rounding -- the tests check
    it against a per-gene dense solve.

    Parameters
    ----------
    gram
        ``X' W X`` per gene, shape ``(n_genes, n_features, n_features)``.
    cross
        ``X' W G`` per gene, shape ``(n_groups, n_features, n_genes)``.
    diagonal
        ``diag(G' W G)`` per gene, shape ``(n_groups, n_genes)``.
    rhs_features, rhs_groups
        ``X' W z`` shape ``(n_genes, n_features)`` and ``G' W z`` shape
        ``(n_groups, n_genes)``.

    Returns
    -------
    beta_features, beta_groups, schur
        Shapes ``(n_genes, n_features)``, ``(n_genes, n_groups)`` and the
        Schur complement ``(n_genes, n_features, n_features)``, the last so
        that standard errors can reuse the factorisation.
    """
    inverse_diagonal = 1.0 / diagonal                        # (a, p)
    # (p, d_X, a): the layout the batched matrix products below want.
    blocks = np.ascontiguousarray(cross.transpose(2, 1, 0))
    scaled = blocks * inverse_diagonal.T[:, None, :]
    schur = schur_complement(gram, cross, diagonal)
    rhs = rhs_features - np.matmul(scaled, rhs_groups.T[:, :, None])[:, :, 0]
    beta_features = np.linalg.solve(schur, rhs[:, :, None])[:, :, 0]
    beta_groups = (
        rhs_groups.T - np.matmul(blocks.transpose(0, 2, 1), beta_features[:, :, None])[:, :, 0]
    ) * inverse_diagonal.T
    return beta_features, beta_groups, schur


def schur_variances(
    schur: np.ndarray, cross: np.ndarray, diagonal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal of the inverse arrowhead Hessian, for standard errors.

    For ``H = [[C, B], [B', D]]`` with Schur complement ``S``::

        (H^-1)_XX = S^-1
        (H^-1)_GG = D^-1 + D^-1 B' S^-1 B D^-1

    so the covariate variances come free from the factorisation already
    computed by :func:`schur_solve` and the group variances need only the
    quadratic form ``B_k' S^-1 B_k`` per group.
    """
    inverse_diagonal = 1.0 / diagonal                        # (a, p)
    schur_inverse = np.linalg.inv(schur)                     # (p, dx, dx)
    feature_variance = np.diagonal(schur_inverse, axis1=1, axis2=2)

    blocks = np.ascontiguousarray(cross.transpose(2, 1, 0))  # (p, dx, a)
    quadratic = (blocks * np.matmul(schur_inverse, blocks)).sum(axis=1)   # (p, a)
    group_variance = inverse_diagonal.T + quadratic * inverse_diagonal.T**2
    return feature_variance, group_variance
