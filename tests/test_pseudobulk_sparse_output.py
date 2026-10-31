"""pb.aggregate writes mostly-zero profiles as CSR and others dense."""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
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


def _effects_reference(bulk_path, aggregate_batches):
    bulk = ad.read_h5ad(bulk_path)
    X = bulk.X.toarray() if sp.issparse(bulk.X) else np.asarray(bulk.X)
    labels, units = bulk.obs["perturbation"].astype(str).to_numpy(), bulk.obs["unit"].astype(str).to_numpy()
    control_of = {u: i for i, u in zip(np.flatnonzero(labels == "control"), units[labels == "control"])}
    targets = np.flatnonzero(labels == "KO")
    per_batch = np.stack([X[i] - X[control_of[units[i]]] for i in targets])
    if not aggregate_batches:
        return per_batch
    n = bulk.obs["n_cells"].to_numpy()
    weights = np.array([n[i] * n[control_of[units[i]]] / (n[i] + n[control_of[units[i]]]) for i in targets])
    return (weights[:, None] * per_batch).sum(axis=0, keepdims=True) / weights.sum()


@pytest.mark.parametrize("aggregate_batches", [False, True])
def test_effects_run_on_the_sparse_pseudobulk(tmp_path, aggregate_batches):
    """pb.effects takes gene blocks (column slices) of the profiles; from a
    CSR file that read failed, and would have read the file whole per block."""
    _cells(tmp_path / "cells.h5ad")
    common = dict(perturbation_column="perturbation", batch_column="unit", aggregate_batches=aggregate_batches,
                  verbose=False, force=True)
    from_cells = cx.pb.effects(tmp_path / "cells.h5ad", method="sum", min_cells=1, bulk_output_path=tmp_path / "pb.h5ad",
                               output_path=tmp_path / "effects.h5ad", **common)
    with h5py.File(tmp_path / "pb.h5ad", "r") as f:
        assert f["X"].attrs["encoding-type"] == b"csr_matrix"
    expected = _effects_reference(tmp_path / "pb.h5ad", aggregate_batches)
    np.testing.assert_allclose(ad.read_h5ad(from_cells.filename).X, expected, atol=1e-9)
    # the saved sparse profiles as input, and too big for memory: through a temporary CSC copy
    from_bulk = cx.pb.effects(tmp_path / "pb.h5ad", output_path=tmp_path / "effects2.h5ad", **common)
    np.testing.assert_allclose(ad.read_h5ad(from_bulk.filename).X, expected, atol=1e-9)
    on_disk = cx.pb.effects(tmp_path / "pb.h5ad", output_path=tmp_path / "effects3.h5ad", memory_limit_gb=1e-6,
                            **common)
    np.testing.assert_allclose(ad.read_h5ad(on_disk.filename).X, expected, atol=1e-9)
    assert not list(tmp_path.glob("**/profiles_csc.h5ad"))
