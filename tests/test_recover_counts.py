"""pp.recover_counts: the integer counts behind log1p(normalize_total(counts))."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _counts(n_cells=50, n_genes=20, seed=0, *, min_one=True):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3.0, (n_cells, n_genes)).astype(np.float32)
    counts[rng.random(counts.shape) < 0.4] = 0
    if min_one:
        counts[np.arange(n_cells), np.arange(n_cells) % n_genes] = 1  # every cell has a 1
    counts[0] = 0  # one empty cell
    return counts


def _write_log1p(path: Path, counts, *, target=1e4, n_counts_column=None, fmt="csr"):
    totals = counts.sum(axis=1)
    scaled = np.divide(counts * target, totals[:, None], out=np.zeros_like(counts), where=totals[:, None] > 0)
    x = np.log1p(scaled).astype(np.float32)
    x = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[fmt](x)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(counts.shape[0])])
    if n_counts_column:
        obs[n_counts_column] = totals
    a = ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(counts.shape[1])]))
    a.obsm["X_pca"] = np.zeros((counts.shape[0], 2), dtype=np.float32)
    a.layers["copy"] = sp.csr_matrix(x)
    a.write(path)
    return path


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_row_minimum_scaling_recovers_counts_exactly(tmp_path, fmt):
    counts = _counts()
    src = _write_log1p(tmp_path / "log.h5ad", counts, fmt=fmt)
    out = cx.pp.recover_counts(src, output_path=tmp_path / "counts.h5ad", verbose=False, format_mismatch_policy="off")
    got = ad.read_h5ad(out.filename)
    assert sp.isspmatrix_csr(got.X) and got.X.dtype == np.float32
    np.testing.assert_array_equal(got.X.toarray(), counts)
    assert got.uns["crispyx_recover_counts"]["scale"] == "row_minimum"
    assert "X_pca" in got.obsm and "copy" in got.layers


def test_totals_column_recovers_cells_whose_smallest_count_is_above_one(tmp_path):
    counts = _counts(min_one=False)
    counts[counts == 1] = 2  # no cell has a 1 any more
    src = _write_log1p(tmp_path / "log.h5ad", counts, n_counts_column="n_counts")
    out = cx.pp.recover_counts(src, output_path=tmp_path / "counts.h5ad", n_counts_column="n_counts", verbose=False)
    np.testing.assert_array_equal(ad.read_h5ad(out.filename).X.toarray(), counts)
    with pytest.raises(ValueError, match="not integers"):
        cx.pp.recover_counts(src, output_path=tmp_path / "fail.h5ad", verbose=False)
    assert not (tmp_path / "fail.h5ad").exists()


def test_refuses_a_matrix_that_already_holds_counts(tmp_path):
    counts = _counts()
    ad.AnnData(sp.csr_matrix(counts)).write(tmp_path / "counts.h5ad")
    with pytest.raises(ValueError, match="already holds integer counts"):
        cx.pp.recover_counts(tmp_path / "counts.h5ad", output_path=tmp_path / "out.h5ad", verbose=False)


def test_result_is_independent_of_chunk_size(tmp_path):
    counts = _counts(n_cells=70)
    src = _write_log1p(tmp_path / "log.h5ad", counts)
    a = cx.pp.recover_counts(src, output_path=tmp_path / "a.h5ad", chunk_size=7, verbose=False)
    b = cx.pp.recover_counts(src, output_path=tmp_path / "b.h5ad", chunk_size=1000, verbose=False)
    np.testing.assert_array_equal(ad.read_h5ad(a.filename).X.toarray(), ad.read_h5ad(b.filename).X.toarray())


def test_unknown_totals_column_and_bad_totals_raise(tmp_path):
    counts = _counts()
    src = _write_log1p(tmp_path / "log.h5ad", counts, n_counts_column="n_counts")
    with pytest.raises(KeyError, match="missing"):
        cx.pp.recover_counts(src, output_path=tmp_path / "o.h5ad", n_counts_column="missing", verbose=False)
    obs = cx.load_obs(src)
    obs["n_counts"] = obs["n_counts"].astype(float)
    obs.iloc[3, obs.columns.get_loc("n_counts")] = -1.0
    cx.write_obs(src, obs)
    with pytest.raises(ValueError, match="non-negative, finite totals"):
        cx.pp.recover_counts(src, output_path=tmp_path / "o.h5ad", n_counts_column="n_counts", verbose=False)
