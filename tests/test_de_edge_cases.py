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


@pytest.mark.parametrize("method", ["t_test", "wilcoxon_test", "nb_glm_test"])
def test_cells_without_a_label_are_in_no_comparison(tmp_path, method):
    rng = np.random.default_rng(1)
    labels = pd.Categorical(["ctrl"] * 40 + ["A"] * 20 + [np.nan] * 10)
    counts = rng.poisson(4, (len(labels), 8)).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(labels))])
    X = counts if method == "nb_glm_test" else np.log1p(counts)
    ad.AnnData(sp.csr_matrix(X), obs=obs).write(tmp_path / "d.h5ad")
    with_nan = getattr(cx, method)(
        tmp_path / "d.h5ad", perturbation_column="perturbation", control_label="ctrl",
        output_path=tmp_path / "out.h5ad", verbose=False, **_NO_FILTER,
    )
    labelled = ad.read_h5ad(tmp_path / "d.h5ad")[:60].copy()
    labelled.write(tmp_path / "labelled.h5ad")
    without = getattr(cx, method)(
        tmp_path / "labelled.h5ad", perturbation_column="perturbation", control_label="ctrl",
        output_path=tmp_path / "out_labelled.h5ad", verbose=False, **_NO_FILTER,
    )
    assert list(with_nan.groups) == ["A"]
    if method != "nb_glm_test":  # its global size factors use every cell in the file
        np.testing.assert_allclose(with_nan["A"].pvalue, without["A"].pvalue, rtol=1e-6)


def test_de_refuses_to_write_over_its_input(screen):
    with pytest.raises(ValueError, match="is the input file"):
        cx.t_test(
            screen, perturbation_column="perturbation", control_label="ctrl",
            output_path=screen, verbose=False,
        )
    assert ad.read_h5ad(screen).n_obs == 81
