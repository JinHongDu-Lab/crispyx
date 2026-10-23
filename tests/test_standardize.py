"""standardize_dataset relabels a dataset without copying its matrices."""

from __future__ import annotations

import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from crispyx.data import read_backed, standardize_dataset


@pytest.fixture
def raw(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.random(60, 20, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    obs = pd.DataFrame(
        {"gene_target": ["CTRL"] * 10 + ["Non-Targeting"] * 30 + ["A"] * 20},
        index=[f"c{i}" for i in range(60)],
    )
    adata = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(20)]))
    adata.layers["counts"] = X.copy()
    adata.obsm["X_pca"] = rng.normal(size=(60, 3))
    path = tmp_path / "data" / "raw.h5ad"
    path.parent.mkdir()
    adata.write(path)
    return path, adata


def _standardize(path, out):
    return standardize_dataset(path, perturbation_column="gene_target", control_label="Non-Targeting",
                               gene_name_column=None, output_dir=out)


def test_metadata_is_rewritten_and_the_matrices_are_linked(raw, tmp_path):
    path, source = raw
    std = _standardize(path, tmp_path / "results")
    with h5py.File(std, "r") as f:
        assert isinstance(f.get("X", getlink=True), h5py.ExternalLink)
        assert isinstance(f["layers"].get("counts", getlink=True), h5py.ExternalLink)
        assert not isinstance(f.get("obsm", getlink=True), h5py.ExternalLink)
    result = ad.read_h5ad(std)
    assert (result.X != source.X).nnz == 0
    assert (result.layers["counts"] != source.layers["counts"]).nnz == 0
    np.testing.assert_array_equal(result.obsm["X_pca"], source.obsm["X_pca"])
    # CTRL is a gene here, not the control.
    assert result.obs["perturbation"].value_counts().to_dict() == {"control": 30, "A": 20, "CTRL": 10}


def test_the_link_survives_moving_source_and_cache_together(raw, tmp_path):
    path, source = raw
    _standardize(path, tmp_path / "results")
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(str(tmp_path / "data"), moved / "data")
    shutil.move(str(tmp_path / "results"), moved / "results")
    backed = read_backed(moved / "results" / ".cache" / "standardized_raw.h5ad")
    try:
        assert (backed.X[:5] != source.X[:5]).nnz == 0
    finally:
        backed.file.close()


def test_writes_into_the_standardized_file_never_reach_the_source(raw, tmp_path):
    path, _ = raw
    before = path.read_bytes()
    std = _standardize(path, tmp_path / "results")
    with h5py.File(std, "r+") as f:
        f["obsm"].create_dataset("X_umap", data=np.zeros((60, 2)))
        f["layers"].create_dataset("extra", data=np.zeros(3))
    assert path.read_bytes() == before


def test_data_bytes_count_the_linked_matrix(raw, tmp_path):
    from crispyx.data import h5ad_data_bytes

    path, _ = raw
    std = _standardize(path, tmp_path / "results")
    with h5py.File(path, "r") as f:
        x_bytes = sum(f["X"][k].id.get_storage_size() for k in f["X"])
    assert h5ad_data_bytes(std) >= x_bytes


def test_qc_on_a_standardized_file_sizes_from_the_linked_matrix(raw, tmp_path, monkeypatch, capsys):
    """QC's in-memory/streaming choice uses the data's size, not the
    metadata file's (a 4.5M-cell screen was loaded whole and OOM-killed)."""
    from crispyx import qc
    from crispyx import data as cxd

    path, _ = raw
    std = _standardize(path, tmp_path / "results")
    seen = []
    real = cxd.h5ad_data_bytes
    monkeypatch.setattr(qc, "h5ad_data_bytes", lambda p: seen.append(real(p)) or seen[-1])
    qc.quality_control_summary(std, perturbation_column="perturbation", control_label="control",
                               min_genes=1, min_cells_per_perturbation=1, min_cells_per_gene=1,
                               output_path=tmp_path / "qc.h5ad", verbose=False)
    assert seen and seen[0] >= h5py.File(path, "r")["X/data"].id.get_storage_size()
