"""Backed handles and DE results survive a pickle round trip.

Reuse of an existing DE result is covered in ``test_result_provenance.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from crispyx.data import AnnData
from crispyx.de import DifferentialExpressionResult, wilcoxon_test


def _make_h5ad(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_ctrl, n_pert, n_genes = 40, 30, 20
    counts = (rng.random((n_ctrl + n_pert, n_genes)) < 0.4) * rng.poisson(5, (n_ctrl + n_pert, n_genes))
    X = np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e4)
    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"c{i}" for i in range(n_ctrl + n_pert)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    path = tmp_path / "data.h5ad"
    ad.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs, var=var).write(path)
    return path


def test_getattr_guard_raises_attribute_error_before_init():
    """__getattr__ must raise AttributeError, not recurse, when _backed is absent."""
    wrapper = object.__new__(AnnData)  # bypasses __init__
    with pytest.raises(AttributeError):
        _ = wrapper.obs


def test_anndata_round_trip_reopens_lazily(tmp_path):
    wrapper = AnnData(_make_h5ad(tmp_path), mode="r")
    _ = wrapper.backed  # open the handle so the round trip has to drop it
    restored = pickle.loads(pickle.dumps(wrapper))
    assert restored.path == wrapper.path
    assert restored._mode == wrapper._mode
    assert restored._backed is None
    assert len(restored.backed.obs) == 70
    restored.close()
    wrapper.close()


def test_rank_genes_groups_result_round_trip(tmp_path):
    result = wilcoxon_test(
        _make_h5ad(tmp_path), perturbation_column="perturbation", control_label="ctrl",
        output_path=tmp_path / "result.h5ad", min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0,
    )
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_array_equal(restored.genes, result.genes)
    assert restored.groups == result.groups
    np.testing.assert_array_equal(restored.pvalues, result.pvalues)
    np.testing.assert_array_equal(restored.logfoldchanges, result.logfoldchanges)
    assert restored.result is None
    assert restored._group_cache == {}
    # Dict-style access still works on the restored object.
    item = restored["KO1"]
    assert item.perturbation == "KO1"
    np.testing.assert_array_equal(item.pvalue, result["KO1"].pvalue)


def test_differential_expression_result_round_trip():
    rng = np.random.default_rng(0)
    der = DifferentialExpressionResult(
        genes=pd.Index([f"gene{i}" for i in range(10)]),
        effect_size=rng.random(10),
        statistic=rng.random(10),
        pvalue=rng.random(10),
        method="wilcoxon",
        perturbation="KO1",
        pvalue_adj=rng.random(10),
        result=None,
    )
    restored = pickle.loads(pickle.dumps(der))
    np.testing.assert_array_equal(restored.genes, der.genes)
    np.testing.assert_array_equal(restored.effect_size, der.effect_size)
    np.testing.assert_array_equal(restored.pvalue_adj, der.pvalue_adj)
    assert restored.result is None
