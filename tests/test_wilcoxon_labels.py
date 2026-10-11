"""wilcoxon_test with the labels given as a Series instead of an obs column,
the permutation-null use case: no relabelled copy of the data is written."""

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
    labels = np.array(["ctrl"] * 40 + ["A"] * 20 + ["B"] * 20)
    counts = rng.poisson(4, (labels.size, 10)).astype(np.float32)
    counts[labels == "A", :3] *= 3
    obs = pd.DataFrame(
        {"perturbation": labels, "batch": np.where(np.arange(labels.size) % 2, "b1", "b2")},
        index=[f"c{i}" for i in range(labels.size)],
    )
    path = tmp_path / "data.h5ad"
    ad.AnnData(sp.csc_matrix(np.log1p(counts)), obs=obs).write(path)
    return path, obs


@pytest.mark.parametrize("batch_column", [None, "batch"], ids=["pooled", "stratified"])
@pytest.mark.parametrize("memory_limit_gb", [128, 1e-7], ids=["standard", "streaming"])
def test_series_labels_match_an_obs_column(tmp_path, screen, batch_column, memory_limit_gb):
    """Shuffled labels as a Series give the result the same shuffle written
    into obs gives, whatever order the Series is in."""
    path, obs = screen
    shuffled = pd.Series(
        np.random.default_rng(1).permutation(obs["perturbation"].to_numpy()),
        index=obs.index, name="shuffle",
    )
    relabelled = ad.read_h5ad(path)
    relabelled.obs["shuffle"] = shuffled.to_numpy()
    relabelled_path = tmp_path / "relabelled.h5ad"
    relabelled.write(relabelled_path)

    common = dict(control_label="ctrl", batch_column=batch_column,
                  memory_limit_gb=memory_limit_gb, verbose=False, **_NO_FILTER)
    from_series = cx.wilcoxon_test(
        path, perturbation_column=shuffled.iloc[::-1], output_path=tmp_path / "series.h5ad", **common,
    )
    from_obs = cx.wilcoxon_test(
        relabelled_path, perturbation_column="shuffle", output_path=tmp_path / "obs.h5ad", **common,
    )
    assert from_series.groups == from_obs.groups
    for field in ("statistics", "pvalues", "logfoldchanges", "auc"):
        np.testing.assert_array_equal(getattr(from_series, field), getattr(from_obs, field), err_msg=field)
    written = ad.read_h5ad(tmp_path / "series.h5ad")
    assert written.uns["perturbation_column"] == "shuffle"
    assert "shuffle" in written.obs


def test_each_shuffle_is_its_own_result(tmp_path, screen, capsys):
    """A fixed output_path recomputes for a new Series and reuses for the same one."""
    path, obs = screen
    rng = np.random.default_rng(2)
    output = tmp_path / "null.h5ad"
    common = dict(control_label="ctrl", output_path=output, verbose=True, **_NO_FILTER)
    first = pd.Series(rng.permutation(obs["perturbation"].to_numpy()), index=obs.index)
    second = pd.Series(rng.permutation(obs["perturbation"].to_numpy()), index=obs.index)

    a = cx.wilcoxon_test(path, perturbation_column=first, **common)
    capsys.readouterr()
    cx.wilcoxon_test(path, perturbation_column=first.copy(), **common)
    assert "Loading existing result" in capsys.readouterr().out
    b = cx.wilcoxon_test(path, perturbation_column=second, **common)
    assert "argument 'perturbation_column' differs" in capsys.readouterr().out
    assert not np.allclose(a.pvalues, b.pvalues, equal_nan=True)


def test_series_missing_cells_raises(tmp_path, screen):
    path, obs = screen
    partial = obs["perturbation"].iloc[1:]
    with pytest.raises(ValueError, match="no label for 1 of 80 cells"):
        cx.wilcoxon_test(path, perturbation_column=partial, control_label="ctrl",
                         output_path=tmp_path / "x.h5ad", verbose=False)


def test_disk_estimate_accepts_series_labels(screen):
    path, obs = screen
    estimate = cx.estimate_disk_usage(
        cx.wilcoxon_test, path, perturbation_column=obs["perturbation"], control_label="ctrl",
    )
    assert "output" in estimate
