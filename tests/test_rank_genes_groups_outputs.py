"""cx.tl.rank_genes_groups and the plotting readers on every DE method's file."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx import de


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    root = tmp_path_factory.mktemp("rgg")
    rng = np.random.default_rng(1)
    n_genes = 60
    labels = np.array(["control"] * 150 + ["A"] * 60 + ["B"] * 60)
    mu = rng.gamma(0.8, 2.0, n_genes)
    effect = np.ones((3, n_genes))
    effect[1:, :10] = rng.lognormal(0, 1.0, (2, 10))
    group = np.array([{"control": 0, "A": 1, "B": 2}[x] for x in labels])
    counts = rng.negative_binomial(2, 2 / (2 + mu[None] * effect[group])).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(labels.size)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var).write(root / "counts.h5ad")
    norm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    ad.AnnData(sp.csr_matrix(norm), obs=obs, var=var).write(root / "norm.h5ad")
    return root


METHODS = {
    "t-test": (de.t_test, "norm.h5ad"),
    "wilcoxon": (de.wilcoxon_test, "norm.h5ad"),
    "nb_glm": (de.nb_glm_test, "counts.h5ad"),
}


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("scanpy_format", [False, True])
def test_rank_genes_groups_and_readers_report_the_stored_values(screen, tmp_path, method, scanpy_format):
    fn, source = METHODS[method]
    common = dict(perturbation_column="perturbation", control_label="control", verbose=False)
    direct = fn(screen / source, output_dir=tmp_path, data_name="direct", scanpy_format=scanpy_format, **common)
    wrapped = cx.tl.rank_genes_groups(
        screen / source, method=method, output_dir=tmp_path, data_name="wrapped", **common,
    )
    # The wrapper returns the method's own result file.
    wrapped_mem = wrapped.to_memory()
    assert wrapped_mem.uns["method"] == ("t_test" if method == "t-test" else method)
    np.testing.assert_array_equal(wrapped_mem.X, direct.effect_size)
    np.testing.assert_array_equal(wrapped_mem.var["pts_rest"].to_numpy(), direct.pts_rest[0])
    wrapped.close()

    order_of = {g: i for i, g in enumerate(direct.groups)}
    for path in (direct.result_path, wrapped.path):
        for group in ("A", "B"):
            row = order_of[group]
            df = cx.pl.rank_genes_groups_df(path, group=group)
            genes = list(direct.genes)
            idx = np.array([genes.index(name) for name in df["names"]])
            np.testing.assert_array_equal(df["pvals"].to_numpy(), direct.pvalues[row, idx])
            np.testing.assert_array_equal(df["logfoldchanges"].to_numpy(), direct.logfoldchanges[row, idx])
            np.testing.assert_array_equal(df["pts"].to_numpy(), direct.pts[row, idx])
            np.testing.assert_array_equal(df["pts_rest"].to_numpy(), direct.pts_rest[row, idx])
            assert (df["pts_rest"] > 0).any()

        mat = cx.pl.materialize_rank_genes_groups(path, n_genes=5)
        rgg = mat.uns["rank_genes_groups"]
        assert {"names", "scores", "logfoldchanges", "pvals", "pts", "pts_rest"} <= set(rgg)
        assert set(rgg["pts_rest"].dtype.names) == {"A", "B"}

    if scanpy_format:
        with h5py.File(direct.result_path, "r") as f:
            full = f["uns/rank_genes_groups/full"]
            np.testing.assert_array_equal(full["pts_rest"][()], np.asarray(direct.pts_rest))
            np.testing.assert_array_equal(full["logfoldchanges"][()], direct.logfoldchanges)


def test_reloading_an_existing_result_matches_the_run(screen, tmp_path):
    """force=False reloads each method's file into the same result object."""
    common = dict(perturbation_column="perturbation", control_label="control", verbose=False, output_dir=tmp_path)
    for fn, source in METHODS.values():
        first = fn(screen / source, **common)
        again = fn(screen / source, **common)  # loads the existing file
        for field in ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "effect_size", "pts", "pts_rest"):
            np.testing.assert_array_equal(np.asarray(getattr(again, field)), np.asarray(getattr(first, field)), err_msg=field)
        assert again.method == first.method
