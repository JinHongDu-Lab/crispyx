"""pb.aggregate writes mostly-zero profiles as CSR and others dense."""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

import crispyx as cx


def _cells(path, *, n_cells=600, n_genes=300, n_units=120, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.05, (n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        "perturbation": np.where(np.arange(n_cells) < n_cells // 2, "control", "KO"),
        "unit": [f"u{i % n_units}" for i in range(n_cells)],
    }, index=[f"c{i}" for i in range(n_cells)])
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])).write(path)
    return counts, obs


def test_sparse_profiles_are_written_as_csr_and_dense_ones_dense(tmp_path):
    counts, obs = _cells(tmp_path / "cells.h5ad")
    summed = cx.pb.aggregate(tmp_path / "cells.h5ad", groupby=["perturbation", "unit"], method="sum", min_cells=1,
                             output_path=tmp_path / "pb.h5ad", verbose=False)
    with h5py.File(summed.filename, "r") as f:
        assert isinstance(f["X"], h5py.Group) and f["X"].attrs["encoding-type"] == b"csr_matrix"
    got = ad.read_h5ad(summed.filename)
    expected = pd.DataFrame(counts, index=obs.index).groupby([obs["perturbation"], obs["unit"]]).sum()
    assert sp.isspmatrix_csr(got.X)
    np.testing.assert_allclose(np.sort(got.X.toarray().sum(axis=1)), np.sort(expected.to_numpy().sum(axis=1)))
    cx.data.check_sparse_layout(summed.filename)
    # averages of log values are mostly non-zero: dense
    means = cx.pb.aggregate(tmp_path / "cells.h5ad", groupby=["perturbation"], method="mean_log1p", min_cells=1,
                            output_path=tmp_path / "pb_mean.h5ad", verbose=False)
    with h5py.File(means.filename, "r") as f:
        assert isinstance(f["X"], h5py.Dataset)


def test_nb_glm_runs_on_the_sparse_pseudobulk(tmp_path):
    _cells(tmp_path / "cells.h5ad")
    summed = cx.pb.aggregate(tmp_path / "cells.h5ad", groupby=["perturbation", "unit"], method="sum", min_cells=1,
                             output_path=tmp_path / "pb.h5ad", verbose=False)
    result = cx.nb_glm_test(summed.filename, perturbation_column="perturbation", control_label="control",
                            output_path=tmp_path / "glm.h5ad", verbose=False, min_pct_ctrl=0, min_pct_pert=0,
                            min_mean_ctrl=0, min_mean_pert=0)
    assert np.isfinite(np.asarray(result.pvalues)).sum() > 100


def test_profiles_without_genes_are_written_dense(tmp_path):
    from crispyx.pseudobulk import _write_profiles

    with h5py.File(tmp_path / "pb.h5", "w") as f:
        assert _write_profiles(f, np.zeros((3, 0)), (3, 0)) == "dense"
        assert f["X"].shape == (3, 0)
