"""Tests for cx.pp.subsample / crispyx.sample.subsample.

Covers: exact n vs. frac, single- and multi-column stratification, unit="cell"
vs. unit="<column>" cluster sampling, drop_insufficient behavior, reproducibility,
and that write_filtered_subset (the shared streaming writer) now carries
obsm/obsp/varm/layers/uns through a filtered/subsampled output, warning-and-
skipping `.raw`.
"""
from __future__ import annotations

from pathlib import Path

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx.data import write_filtered_subset


def _make_h5ad(
    tmp_path: Path,
    *,
    n_cells_per_group: dict[str, int],
    n_genes: int = 20,
    batches_per_group: dict[str, list[str]] | None = None,
    seed: int = 0,
    with_extra_slots: bool = False,
    with_raw: bool = False,
) -> Path:
    """Build a small synthetic h5ad with a 'pert' obs column (and optionally
    'batch'), returning its path.
    """
    rng = np.random.default_rng(seed)
    perts, batches = [], []
    for pert, n in n_cells_per_group.items():
        perts.extend([pert] * n)
        if batches_per_group is not None:
            choices = batches_per_group[pert]
            batches.extend(rng.choice(choices, size=n).tolist())
    n_cells = len(perts)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)

    obs_dict = {"pert": perts}
    if batches_per_group is not None:
        obs_dict["batch"] = batches
    obs = pd.DataFrame(obs_dict, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])

    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    if with_extra_slots:
        adata.obsm["X_pca"] = rng.normal(size=(n_cells, 3)).astype(np.float32)
        adata.varm["PCs"] = rng.normal(size=(n_genes, 3)).astype(np.float32)
        conn = sp.random(n_cells, n_cells, density=0.05, random_state=seed, format="csr")
        adata.obsp["connectivities"] = conn
        adata.varp["gene_net"] = sp.random(n_genes, n_genes, density=0.1, random_state=seed, format="csr")
        adata.layers["counts2"] = (counts * 2).astype(np.float32)
        adata.uns["some_key"] = "hello"
    if with_raw:
        adata.raw = adata

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.h5ad"
    adata.write(path)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_requires_exactly_one_of_n_or_frac(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 10, "control": 10})
    with pytest.raises(TypeError):
        cx.pp.subsample(path)
    with pytest.raises(TypeError):
        cx.pp.subsample(path, n=5, frac=0.5)


def test_unknown_groupby_column_raises(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 10, "control": 10})
    with pytest.raises(KeyError):
        cx.pp.subsample(path, n=1, groupby="does_not_exist")


def test_unknown_unit_column_raises(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 10, "control": 10})
    with pytest.raises(KeyError):
        cx.pp.subsample(path, n=1, unit="does_not_exist")


# ---------------------------------------------------------------------------
# Exact n / frac, stratification
# ---------------------------------------------------------------------------

def test_exact_n_global_no_groupby(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 30, "control": 30})
    out = cx.pp.subsample(path, n=10, groupby=None, verbose=False)
    assert out.n_obs == 10


def test_frac_per_group(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 40, "B": 20, "control": 40})
    out = cx.pp.subsample(path, frac=0.5, groupby="pert", verbose=False)
    counts = out.obs["pert"].value_counts()
    assert counts.get("A", 0) == 20
    assert counts.get("B", 0) == 10
    assert counts.get("control", 0) == 20


def test_n_per_group_caps_every_stratum(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 40, "B": 40, "control": 40})
    out = cx.pp.subsample(path, n=15, groupby="pert", verbose=False)
    counts = out.obs["pert"].value_counts()
    assert (counts == 15).all()
    assert out.n_obs == 45


def test_multi_column_groupby_strata(tmp_path):
    path = _make_h5ad(
        tmp_path,
        n_cells_per_group={"A": 40, "control": 40},
        batches_per_group={"A": ["b1", "b2"], "control": ["b1", "b2"]},
    )
    out = cx.pp.subsample(path, n=5, groupby=["pert", "batch"], verbose=False)
    combo_counts = out.obs.groupby(["pert", "batch"], observed=True).size()
    assert (combo_counts == 5).all()


# ---------------------------------------------------------------------------
# unit="<column>" cluster sampling
# ---------------------------------------------------------------------------

def test_cluster_sampling_no_partial_unit_leakage(tmp_path):
    path = _make_h5ad(
        tmp_path,
        n_cells_per_group={"A": 60, "control": 60},
        batches_per_group={
            "A": ["b1", "b2", "b3", "b4"],
            "control": ["b1", "b2", "b3", "b4"],
        },
        seed=1,
    )
    out = cx.pp.subsample(path, n=2, groupby="pert", unit="batch", verbose=False)

    src = cx.read_backed(path)
    src_obs = src.obs.copy()
    src.file.close()

    for pert_value, sub in out.obs.groupby("pert", observed=True):
        kept_batches = set(sub["batch"].unique())
        # every kept batch's cells for this stratum must ALL be present
        for b in kept_batches:
            expected = int(((src_obs["pert"] == pert_value) & (src_obs["batch"] == b)).sum())
            actual = int((sub["batch"] == b).sum())
            assert actual == expected
        # no batch outside the kept set leaks in
        stratum_batches = set(src_obs.loc[src_obs["pert"] == pert_value, "batch"].unique())
        assert kept_batches <= stratum_batches
        assert len(kept_batches) == 2


# ---------------------------------------------------------------------------
# drop_insufficient behavior
# ---------------------------------------------------------------------------

def test_drop_insufficient_default_drops_small_group(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 3, "control": 40})
    with pytest.warns(UserWarning, match="dropped"):
        out = cx.pp.subsample(path, n=10, groupby="pert", verbose=False)
    counts = out.obs["pert"].value_counts()
    assert counts.get("A", 0) == 0
    assert counts.get("control", 0) == 10


def test_drop_insufficient_false_keeps_small_group_in_full(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 3, "control": 40})
    with pytest.warns(UserWarning, match="kept in full"):
        out = cx.pp.subsample(
            path, n=10, groupby="pert", drop_insufficient=False, verbose=False,
        )
    counts = out.obs["pert"].value_counts()
    assert counts.get("A", 0) == 3
    assert counts.get("control", 0) == 10


def test_n_zero_is_not_treated_as_insufficient(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 3, "control": 40})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = cx.pp.subsample(path, n=0, groupby="pert", verbose=False)
    subsample_warnings = [w for w in record if "pp.subsample" in str(w.message)]
    assert not subsample_warnings
    assert out.n_obs == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_with_same_random_state(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 50, "control": 50}, seed=3)
    out1 = cx.pp.subsample(
        path, n=10, groupby="pert", random_state=42,
        output_path=tmp_path / "out1.h5ad", verbose=False,
    )
    out2 = cx.pp.subsample(
        path, n=10, groupby="pert", random_state=42,
        output_path=tmp_path / "out2.h5ad", verbose=False,
    )
    assert list(out1.obs_names) == list(out2.obs_names)


def test_different_random_state_differs(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 50, "control": 50}, seed=3)
    out1 = cx.pp.subsample(
        path, n=10, groupby="pert", random_state=1,
        output_path=tmp_path / "out1.h5ad", verbose=False,
    )
    out2 = cx.pp.subsample(
        path, n=10, groupby="pert", random_state=2,
        output_path=tmp_path / "out2.h5ad", verbose=False,
    )
    assert list(out1.obs_names) != list(out2.obs_names)


def test_result_independent_of_chunk_size(tmp_path):
    path = _make_h5ad(tmp_path, n_cells_per_group={"A": 50, "control": 50}, seed=4)
    out1 = cx.pp.subsample(
        path, n=10, groupby="pert", chunk_size=8,
        output_path=tmp_path / "out1.h5ad", verbose=False,
    )
    out2 = cx.pp.subsample(
        path, n=10, groupby="pert", chunk_size=4096,
        output_path=tmp_path / "out2.h5ad", verbose=False,
    )
    assert list(out1.obs_names) == list(out2.obs_names)


# ---------------------------------------------------------------------------
# Slot preservation (write_filtered_subset)
# ---------------------------------------------------------------------------

def test_subsample_preserves_extra_slots(tmp_path):
    path = _make_h5ad(
        tmp_path, n_cells_per_group={"A": 40, "control": 40},
        with_extra_slots=True, seed=5,
    )
    out = cx.pp.subsample(path, n=10, groupby="pert", verbose=False)

    src_mem = ad.read_h5ad(path)
    out_mem = ad.read_h5ad(out.path)

    mask = src_mem.obs_names.isin(out_mem.obs_names)
    assert mask.sum() == out_mem.n_obs

    np.testing.assert_allclose(out_mem.obsm["X_pca"], src_mem.obsm["X_pca"][mask])
    np.testing.assert_allclose(out_mem.varm["PCs"], src_mem.varm["PCs"])
    np.testing.assert_allclose(
        out_mem.obsp["connectivities"].toarray(),
        src_mem.obsp["connectivities"][mask][:, mask].toarray(),
    )
    np.testing.assert_allclose(out_mem.varp["gene_net"].toarray(), src_mem.varp["gene_net"].toarray())
    out_counts2 = out_mem.layers["counts2"]
    out_counts2 = out_counts2.toarray() if sp.issparse(out_counts2) else out_counts2
    np.testing.assert_allclose(out_counts2, src_mem.layers["counts2"][mask])
    assert out_mem.uns["some_key"] == "hello"


def test_write_filtered_subset_preserves_extra_slots_directly(tmp_path):
    """Item 1 regression: the shared writer used by pp.filter_cells/etc. too."""
    path = _make_h5ad(
        tmp_path, n_cells_per_group={"A": 20, "control": 20},
        with_extra_slots=True, seed=6,
    )
    src_mem = ad.read_h5ad(path)
    cell_mask = np.zeros(src_mem.n_obs, dtype=bool)
    cell_mask[::2] = True
    gene_mask = np.ones(src_mem.n_vars, dtype=bool)

    out_path = tmp_path / "filtered.h5ad"
    write_filtered_subset(path, cell_mask=cell_mask, gene_mask=gene_mask, output_path=out_path)
    out_mem = ad.read_h5ad(out_path)

    assert out_mem.n_obs == int(cell_mask.sum())
    np.testing.assert_allclose(out_mem.obsm["X_pca"], src_mem.obsm["X_pca"][cell_mask])
    np.testing.assert_allclose(out_mem.varm["PCs"], src_mem.varm["PCs"])
    np.testing.assert_allclose(
        out_mem.obsp["connectivities"].toarray(),
        src_mem.obsp["connectivities"][cell_mask][:, cell_mask].toarray(),
    )
    np.testing.assert_allclose(out_mem.varp["gene_net"].toarray(), src_mem.varp["gene_net"].toarray())
    out_counts2 = out_mem.layers["counts2"]
    out_counts2 = out_counts2.toarray() if sp.issparse(out_counts2) else out_counts2
    np.testing.assert_allclose(out_counts2, src_mem.layers["counts2"][cell_mask])
    assert out_mem.uns["some_key"] == "hello"


def test_write_filtered_subset_subsets_varp_by_gene_mask(tmp_path):
    """varp is gene-by-gene, so it must be subset along gene_mask, not cell_mask."""
    path = _make_h5ad(
        tmp_path, n_cells_per_group={"A": 20, "control": 20},
        with_extra_slots=True, seed=13,
    )
    src_mem = ad.read_h5ad(path)
    cell_mask = np.ones(src_mem.n_obs, dtype=bool)
    gene_mask = np.zeros(src_mem.n_vars, dtype=bool)
    gene_mask[::2] = True

    out_path = tmp_path / "filtered_genes.h5ad"
    write_filtered_subset(path, cell_mask=cell_mask, gene_mask=gene_mask, output_path=out_path)
    out_mem = ad.read_h5ad(out_path)

    assert out_mem.n_vars == int(gene_mask.sum())
    np.testing.assert_allclose(
        out_mem.varp["gene_net"].toarray(),
        src_mem.varp["gene_net"][gene_mask][:, gene_mask].toarray(),
    )


def test_write_filtered_subset_warns_and_drops_raw(tmp_path):
    path = _make_h5ad(
        tmp_path, n_cells_per_group={"A": 20, "control": 20}, with_raw=True, seed=7,
    )
    src_mem = ad.read_h5ad(path)
    cell_mask = np.ones(src_mem.n_obs, dtype=bool)
    gene_mask = np.ones(src_mem.n_vars, dtype=bool)
    out_path = tmp_path / "filtered.h5ad"

    with pytest.warns(UserWarning, match="raw"):
        write_filtered_subset(path, cell_mask=cell_mask, gene_mask=gene_mask, output_path=out_path)

    out_mem = ad.read_h5ad(out_path)
    assert out_mem.raw is None
