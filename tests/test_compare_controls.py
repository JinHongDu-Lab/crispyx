"""pp.compare_controls: are unassigned cells like cells with a non-targeting guide?"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import crispyx as cx


def _screen(tmp_path, shift):
    """Four non-targeting guides (60 cells each) and 400 unassigned cells; in
    the unassigned cells 20 genes are scaled by ``shift``."""
    rng = np.random.default_rng(0)
    n_genes = 200
    guides = np.array([f"NT{k}" for k in range(4) for _ in range(60)] + ["unassigned"] * 400)
    mu = rng.lognormal(np.log(5), 1.0, n_genes)
    mean = np.tile(mu, (guides.size, 1)) * rng.lognormal(0, 0.2, guides.size)[:, None]
    mean[np.ix_(guides == "unassigned", np.arange(20))] *= shift
    counts = rng.negative_binomial(5, 5 / (5 + mean)).astype(np.float32)
    path = tmp_path / "screen.h5ad"
    ad.AnnData(counts, obs=pd.DataFrame({"guide": guides}, index=[f"c{i}" for i in range(guides.size)]),
               var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])).write(path)
    return path


@pytest.mark.parametrize("shift", [1.0, 3.0], ids=["alike", "shifted"])
def test_compare_controls_flags_reference_cells_that_differ(tmp_path, shift):
    path = _screen(tmp_path, shift)
    table = cx.pp.compare_controls(path, guide_column="guide", controls=[f"NT{k}" for k in range(4)],
                                   reference="unassigned", work_dir=tmp_path / "work", n_jobs=1, verbose=False)
    guides = table.drop(index="all control guides")
    assert list(guides.index) == ["NT0", "NT1", "NT2", "NT3"]
    assert (guides["cells"] == 60).all() and table.loc["all control guides", "cells"] == 240
    assert (tmp_path / "work" / "compare_controls.csv").exists()
    # against other genuine controls: next to nothing either way
    assert guides["vs other controls: hits"].max() <= 2
    if shift == 1.0:
        assert guides["vs reference: hits"].max() <= 2
        assert table.loc["all control guides", "vs reference: hits"] <= 2
    else:  # the shifted genes are found against the reference, not among the guides
        assert (guides["vs reference: hits"] >= 15).all()
        assert table.loc["all control guides", "vs reference: hits"] >= 18


def test_control_guide_correlation_shows_guides_that_share_variation(tmp_path):
    """NT0 and NT1 share a shift in 30 genes (a batch, an off-target effect):
    their deviation profiles correlate; unrelated guides do not."""
    rng = np.random.default_rng(1)
    n_genes = 300
    guides = np.array([f"NT{k}" for k in range(5) for _ in range(150)] + ["unassigned"] * 600)
    mean = np.tile(rng.lognormal(np.log(8), 1.0, n_genes), (guides.size, 1))
    shared = np.isin(guides, ["NT0", "NT1"])
    mean[np.ix_(shared, np.arange(30))] *= 2.5
    mean[np.ix_(guides == "unassigned", np.arange(30, 60))] *= 2.0
    counts = rng.negative_binomial(10, 10 / (10 + mean)).astype(np.float32)
    path = tmp_path / "screen.h5ad"
    ad.AnnData(counts, obs=pd.DataFrame({"guide": guides}, index=[f"c{i}" for i in range(guides.size)]),
               var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])).write(path)
    corr = cx.pp.control_guide_correlation(path, guide_column="guide", controls=[f"NT{k}" for k in range(5)],
                                           reference="unassigned", chunk_size=100)
    assert list(corr.index) == [f"NT{k}" for k in range(5)] + ["unassigned (reference)"]
    assert corr.attrs["n_cells"]["NT0"] == 150 and corr.attrs["n_cells"]["unassigned (reference)"] == 600
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert corr.loc["NT0", "NT1"] > 0.5
    others = corr.loc[["NT2", "NT3", "NT4"], ["NT2", "NT3", "NT4"]].to_numpy()
    assert np.abs(others[np.triu_indices(3, 1)]).max() < corr.loc["NT0", "NT1"]
    ax = cx.pl.control_guide_correlation(corr)
    assert [t.get_text() for t in ax.get_yticklabels()][0].endswith(")")


def test_compare_controls_needs_two_control_guides(tmp_path):
    path = _screen(tmp_path, 1.0)
    with pytest.raises(ValueError, match="at least two control guides"):
        cx.pp.compare_controls(path, guide_column="guide", controls=["NT0"], reference="unassigned",
                               work_dir=tmp_path / "work", verbose=False)
