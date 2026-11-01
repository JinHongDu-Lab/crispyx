"""wilcoxon_test(distribution="hypergeometric") against permutation p-values."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import rankdata

import crispyx as cx
from crispyx.de import wilcoxon_test

NO_FILTER = dict(min_cells_expressed=0, min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, min_mean_pert=0.0)
N_CTRL, N_PERT = 3000, 60
# (perturbed cells expressing, controls expressing) for near-empty genes, then a dense gene
SPARSE = [(1, 3), (2, 3), (2, 20), (3, 20), (4, 60)]


def _screen(path: Path, n_strata: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = N_CTRL + N_PERT
    x = np.zeros((n, len(SPARSE) + 1))
    for g, (in_pert, in_ctrl) in enumerate(SPARSE):
        x[rng.choice(N_CTRL, in_ctrl, replace=False), g] = rng.integers(1, 4, in_ctrl)
        x[N_CTRL + rng.choice(N_PERT, in_pert, replace=False), g] = rng.integers(1, 4, in_pert)
    x[:, -1] = rng.poisson(3.0, n) + rng.normal(0, 0.1, n)
    labels = np.r_[np.repeat("control", N_CTRL), np.repeat("A", N_PERT)]
    strata = rng.integers(0, n_strata, n)
    obs = pd.DataFrame({"perturbation": labels, "batch": strata.astype(str)}, index=[f"c{i}" for i in range(n)])
    ad.AnnData(sp.csr_matrix(x), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(x.shape[1])])).write(path)
    return x, labels == "A", strata


def _permutation_pvalues(x, in_group, strata, n_perm=20000, seed=1):
    """Two-sided p of sum_b (R_b - E_b), labels permuted within strata. The
    ranks within a stratum do not change under a permutation, so each draw
    only sums the ranks at new positions."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(x.shape[1]):
        ranks = np.empty(len(x))
        for b in np.unique(strata):
            rows = strata == b
            ranks[rows] = rankdata(x[rows, g]) - (rows.sum() + 1) / 2   # centred: R_b - E_b sums them
        observed = abs(ranks[in_group].sum())
        null = np.zeros(n_perm)
        for b in np.unique(strata):
            rows = np.flatnonzero(strata == b)
            n1 = int(in_group[rows].sum())
            if n1:
                pick = rng.random((n_perm, rows.size)).argpartition(n1, axis=1)[:, :n1]
                null += ranks[rows][pick].sum(axis=1)
        out.append(np.mean(np.abs(null) >= observed - 1e-9))
    return np.array(out)


@pytest.mark.parametrize("n_strata", [1, 3], ids=["pooled", "stratified"])
def test_the_hypergeometric_null_matches_permutation_p_values(tmp_path, n_strata):
    x, in_group, strata = _screen(tmp_path / "s.h5ad", n_strata)
    batch = {"batch_column": "batch"} if n_strata > 1 else {}
    run = lambda distribution, name: wilcoxon_test(  # noqa: E731
        tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control",
        distribution=distribution, output_path=tmp_path / name, verbose=False, **batch, **NO_FILTER,
    )
    normal, hyper = run("normal", "n.h5ad"), run("hypergeometric", "h.h5ad")
    order = hyper.genes.get_indexer([f"g{g}" for g in range(x.shape[1])])
    p_normal, p_hyper = normal.pvalues[0][order], hyper.pvalues[0][order]
    p_perm = _permutation_pvalues(x, in_group, strata)
    sparse = slice(0, len(SPARSE))
    assert (p_normal[sparse] < p_perm[sparse] / 10).any()                   # the failure this fixes
    ratio = p_hyper[sparse] / p_perm[sparse]
    assert ((ratio > 0.4) & (ratio < 2.5)).all(), (p_hyper, p_perm)
    np.testing.assert_allclose(p_hyper[-1], p_normal[-1], rtol=1e-9)       # a dense gene: unchanged
    np.testing.assert_allclose(hyper.statistics[0], normal.statistics[0])  # z and AUC are the same
    np.testing.assert_allclose(hyper.auc[0], normal.auc[0])
    assert ad.read_h5ad(tmp_path / "h.h5ad", backed="r").uns["distribution"] == "hypergeometric"


def test_the_distribution_is_checked_and_reaches_rank_genes_groups(tmp_path):
    _screen(tmp_path / "s.h5ad", 1)
    common = dict(perturbation_column="perturbation", control_label="control", verbose=False)
    with pytest.raises(ValueError, match="tie_correct"):
        wilcoxon_test(tmp_path / "s.h5ad", distribution="hypergeometric", tie_correct=False, **common)
    with pytest.raises(ValueError, match="distribution"):
        wilcoxon_test(tmp_path / "s.h5ad", distribution="exact", **common)
    direct = wilcoxon_test(tmp_path / "s.h5ad", distribution="hypergeometric", output_path=tmp_path / "a.h5ad",
                           **common, **NO_FILTER)
    via = cx.tl.rank_genes_groups(tmp_path / "s.h5ad", groupby="perturbation", reference="control",
                                  method="wilcoxon", distribution="hypergeometric",
                                  output_dir=tmp_path / "b", verbose=False, **NO_FILTER)
    (written,) = (tmp_path / "b").glob("*.h5ad")
    np.testing.assert_allclose(ad.read_h5ad(written).layers["pvalue"], direct.pvalues)
