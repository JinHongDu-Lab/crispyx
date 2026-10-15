"""Differential-expression edge cases common in real screens: one-cell
groups and cells with no perturbation label."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx

_NO_FILTER = dict(min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, min_mean_pert=0.0)


@pytest.fixture
def screen(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array(["ctrl"] * 60 + ["A"] * 20 + ["single"])
    counts = rng.poisson(4, (labels.size, 12)).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(labels.size)])
    path = tmp_path / "data.h5ad"
    ad.AnnData(sp.csr_matrix(np.log1p(counts)), obs=obs).write(path)
    return path


def test_t_test_leaves_one_cell_groups_untested(tmp_path, screen):
    result = cx.t_test(
        screen, perturbation_column="perturbation", control_label="ctrl",
        output_path=tmp_path / "t.h5ad", verbose=False, **_NO_FILTER,
    )
    single = result["single"]
    assert np.isnan(single.pvalue).all()
    assert np.isnan(single.statistic).all()
    assert np.isfinite(result["A"].pvalue).all()
