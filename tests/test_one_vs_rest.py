"""One-vs-rest markers: ``reference="rest"`` in ``cx.tl.rank_genes_groups``,
checked against ``sc.tl.rank_genes_groups(reference="rest")``."""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

import crispyx as cx
from crispyx.de import RankGenesGroupsResult

NO_FILTERS = dict(min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0)


def _clusters(n_per: int = 40, n_genes: int = 30, seed: int = 0) -> ad.AnnData:
    """Four clusters, each with its own markers; log-normalised and sparse.
    Gene 0 is detected in cluster ``c1`` only; the last gene in no cell."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(["c0", "c1", "c2", "c3"], n_per)
    means = rng.uniform(0.2, 2.0, n_genes)[None, :].repeat(4, axis=0)
    for k in range(4):
        means[k, 1 + 3 * k : 4 + 3 * k] *= 4
    counts = rng.poisson(means[np.repeat(np.arange(4), n_per)]).astype(np.float32)
    counts[labels != "c1", 0] = 0
    counts[:, -1] = 0
    adata = ad.AnnData(sp.csr_matrix(counts), obs=pd.DataFrame({"cluster": labels}, index=[f"cell{i}" for i in range(len(labels))]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _markers(path, out, **kwargs) -> RankGenesGroupsResult:
    cx.tl.rank_genes_groups(path, groupby="cluster", reference="rest", output_path=out, verbose=False, **kwargs)
    return RankGenesGroupsResult(out)


def _scanpy(adata: ad.AnnData, method: str, groups="all") -> dict[str, pd.DataFrame]:
    adata = adata.copy()
    sc.tl.rank_genes_groups(adata, "cluster", groups=groups, reference="rest", method=method,
                            tie_correct=True, pts=True, corr_method="benjamini-hochberg")
    names = adata.uns["rank_genes_groups"]["names"].dtype.names
    return {g: sc.get.rank_genes_groups_df(adata, g).set_index("names").loc[adata.var_names] for g in names}


@pytest.mark.parametrize("fmt", ["csc", "csr"])
@pytest.mark.parametrize("method", ["wilcoxon", "t-test"])
def test_one_vs_rest_matches_scanpy(tmp_path, method, fmt):
    adata = _clusters()
    stored = adata.copy()
    stored.X = sp.csc_matrix(adata.X) if fmt == "csc" else adata.X
    path = tmp_path / "clusters.h5ad"
    stored.write_h5ad(path)
    result = _markers(path, tmp_path / f"{method}.h5ad", method=method, **NO_FILTERS)
    expected = _scanpy(adata, method)
    assert result.groups == ["c0", "c1", "c2", "c3"]
    for g, i in zip(result.groups, range(4)):
        want = expected[g]
        tested = np.isfinite(np.asarray(result.pvalues[i]))
        # Scanpy tests a gene detected in no cell (score 0, p 1); crispyx does not.
        assert tested.tolist() == [True] * 29 + [False]
        np.testing.assert_allclose(np.asarray(result.statistics[i])[tested], want["scores"][tested], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(result.pvalues[i])[tested], want["pvals"][tested], rtol=1e-5, atol=1e-300)
        np.testing.assert_allclose(np.asarray(result.logfoldchanges[i])[tested], want["logfoldchanges"][tested], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(result.pts[i]), want["pct_nz_group"], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(result.pts_rest[i]), want["pct_nz_reference"], rtol=1e-6)
    # Gene 0 is a marker of c1 only: tested, though the rest never expresses it.
    assert result.pts_rest[1][0] == 0 and result.statistics[1][0] > 3


def test_rest_includes_groups_not_tested(tmp_path):
    adata = _clusters()
    path = tmp_path / "clusters.h5ad"
    adata.write_h5ad(path)
    result = _markers(path, tmp_path / "c2.h5ad", perturbations=["c2"], **NO_FILTERS)
    want = _scanpy(adata, "wilcoxon", groups=["c2"])["c2"]
    assert result.groups == ["c2"]
    tested = np.isfinite(np.asarray(result.statistics[0]))
    np.testing.assert_allclose(np.asarray(result.statistics[0])[tested], want["scores"][tested], rtol=1e-5)


def test_unlabelled_cells_are_in_no_comparison(tmp_path):
    adata = _clusters()
    labelled = adata.obs["cluster"].to_numpy() != "c3"
    adata.obs["cluster"] = adata.obs["cluster"].where(labelled)  # c3 unlabelled
    path = tmp_path / "clusters.h5ad"
    adata.write_h5ad(path)
    result = _markers(path, tmp_path / "r.h5ad", **NO_FILTERS)
    want = _scanpy(adata[labelled].copy(), "wilcoxon")
    assert result.groups == ["c0", "c1", "c2"]
    tested = np.isfinite(np.asarray(result.statistics[0]))
    np.testing.assert_allclose(np.asarray(result.statistics[0])[tested], want["c0"]["scores"][tested], rtol=1e-5)


def test_rest_result_is_read_by_the_plots_and_scanpy_format(tmp_path):
    adata = _clusters()
    path = tmp_path / "clusters.h5ad"
    adata.write_h5ad(path)
    out = tmp_path / "markers.h5ad"
    cx.wilcoxon_test(path, groupby="cluster", reference="rest", output_path=out, scanpy_format=True, verbose=False)
    result = RankGenesGroupsResult(out)
    table = cx.pl.rank_genes_groups_df(out, group="c1")
    assert {"names", "scores", "pvals_adj"} <= set(table.columns)
    stored = ad.read_h5ad(out)
    assert "pts_rest" in stored.layers and "pts_rest" not in stored.var
    with h5py.File(out, "r") as handle:
        assert handle["uns/rank_genes_groups/params"].attrs["reference"] == "rest"
    np.testing.assert_array_equal(stored.uns["rank_genes_groups"]["full"]["pts_rest"], stored.layers["pts_rest"])
    materialised = cx.pl.materialize_rank_genes_groups(out)
    assert "pts_rest" in materialised.uns["rank_genes_groups"]
    assert result.control_label == "rest"


def test_rest_is_refused_where_it_does_not_apply(tmp_path):
    adata = _clusters()
    path = tmp_path / "clusters.h5ad"
    adata.write_h5ad(path)
    with pytest.raises(ValueError, match="NB-GLM needs a control label"):
        cx.tl.rank_genes_groups(path, groupby="cluster", reference="rest", method="nb_glm", verbose=False)
    with pytest.raises(ValueError, match="neither batch_column"):
        cx.wilcoxon_test(path, groupby="cluster", reference="rest", batch_column="cluster", verbose=False)


@pytest.mark.parametrize("test", [cx.wilcoxon_test, cx.t_test])
def test_rest_with_an_empty_selection_writes_an_empty_result(tmp_path, test):
    adata = _clusters()
    path = tmp_path / "clusters.h5ad"
    adata.write_h5ad(path)
    test(path, groupby="cluster", reference="rest", perturbations=[], output_path=tmp_path / "empty.h5ad",
         verbose=False)
    assert RankGenesGroupsResult(tmp_path / "empty.h5ad").groups == []


@pytest.mark.parametrize("test", [cx.wilcoxon_test, cx.t_test])
def test_rest_with_one_group_raises(tmp_path, test):
    adata = _clusters()
    adata.obs["cluster"] = "c0"
    path = tmp_path / "one.h5ad"
    adata.write_h5ad(path)
    with pytest.raises(ValueError, match="at least two labelled groups"):
        test(path, groupby="cluster", reference="rest", output_path=tmp_path / "r.h5ad", verbose=False)
