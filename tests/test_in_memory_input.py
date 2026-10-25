"""Streaming entry points accept an in-memory AnnData through a scratch file."""

from __future__ import annotations

import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _screen(shuffled: bool = False, log: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(0)
    labels = np.repeat(["control", "KD1", "KD2"], 40)
    if shuffled:  # not sorted by perturbation, so nb_glm_test writes a sorted copy
        labels = rng.permutation(labels)
    counts = rng.negative_binomial(5, 5 / 7, (labels.size, 25)).astype(np.float32)
    counts[labels == "KD1", :5] *= 3
    if log:  # t_test and wilcoxon_test refuse raw counts
        counts = np.log1p(counts)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(labels.size)])
    return ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(25)]))


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    """Point tempfile at an empty directory so leftover scratch files show."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _same_result(a, b) -> None:
    with h5py.File(a.result_path, "r") as fa, h5py.File(b.result_path, "r") as fb:
        for key in ["X", *(f"layers/{name}" for name in fa["layers"])]:
            np.testing.assert_array_equal(fa[key][()], fb[key][()], err_msg=key)


@pytest.mark.parametrize("method", ["t_test", "wilcoxon_test", "nb_glm_test"])
def test_de_on_in_memory_data_matches_the_file_and_leaves_no_scratch(tmp_path, scratch_root, method):
    adata = _screen(shuffled=True, log=method != "nb_glm_test")
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)
    kwargs = dict(perturbation_column="perturbation", control_label="control", verbose=False)
    if method == "nb_glm_test":
        kwargs["n_jobs"] = 1
    fn = getattr(cx, method)
    from_file = fn(path, output_path=tmp_path / "from_file.h5ad", **kwargs)
    from_memory = fn(adata, output_path=tmp_path / "from_memory.h5ad", **kwargs)
    _same_result(from_file, from_memory)
    assert list(scratch_root.iterdir()) == []


def test_namespace_and_qc_take_in_memory_data(tmp_path, scratch_root):
    adata = _screen()
    result = cx.tl.rank_genes_groups(_screen(log=True), perturbation_column="perturbation", control_label="control",
                                     method="t_test", output_dir=tmp_path, verbose=False)
    assert result.shape == (2, 25)  # KD1, KD2 x genes
    mask = cx.pp.filter_cells(adata, min_genes=1, verbose=False)
    assert mask.shape == (adata.n_obs,)
    filtered = cx.pp.qc_summary(adata, perturbation_column="perturbation", control_label="control",
                                min_genes=1, min_cells_per_perturbation=1, min_cells_per_gene=1,
                                output_path=tmp_path / "qc.h5ad", verbose=False)
    assert filtered.shape[0] == adata.n_obs
    assert list(scratch_root.iterdir()) == []


def test_in_memory_input_needs_an_output_path_and_cannot_resume(tmp_path, scratch_root):
    adata = _screen()
    with pytest.raises(ValueError, match="in-memory AnnData has no file to name the output after"):
        cx.t_test(_screen(log=True), perturbation_column="perturbation", control_label="control", verbose=False)
    with pytest.raises(ValueError, match="resume identifies a run by its input file"):
        cx.t_test(adata, perturbation_column="perturbation", control_label="control",
                  output_path=tmp_path / "de.h5ad", resume=True, verbose=False)
    assert list(scratch_root.iterdir()) == []


def test_scratch_copy_is_removed_when_the_run_fails(tmp_path, scratch_root):
    with pytest.raises(Exception):
        cx.t_test(_screen(), perturbation_column="no_such_column", control_label="control",
                  output_path=tmp_path / "de.h5ad", verbose=False)
    assert list(scratch_root.iterdir()) == []


def test_output_path_given_by_position_or_through_the_namespace(tmp_path, scratch_root):
    adata = _screen()
    for result in (
        cx.normalize_total_log1p(adata, tmp_path / "positional.h5ad", verbose=False),
        cx.pp.normalize_total_log1p(adata, output_path=tmp_path / "namespace.h5ad", verbose=False),
        cx.pp.downsample_counts(adata, tmp_path / "ds.h5ad", counts_per_cell=5, verbose=False),
    ):
        assert result.shape == adata.shape
        result.close()
    assert list(scratch_root.iterdir()) == []


def test_qc_summary_without_an_output_returns_masks_only(tmp_path, scratch_root):
    result = cx.quality_control_summary(_screen(), perturbation_column="perturbation", control_label="control",
                                        min_genes=1, min_cells_per_perturbation=1, min_cells_per_gene=1,
                                        verbose=False)
    assert result.cell_mask.shape == (120,)
    assert list(scratch_root.iterdir()) == [] and list(tmp_path.glob("*.h5ad")) == []


def test_scratch_copy_goes_beside_the_output_when_tmp_is_in_memory(tmp_path, scratch_root, monkeypatch):
    """A RAM-backed /tmp would hold the copy in memory, charged to the job."""
    import crispyx.data as cxd

    monkeypatch.setattr(cxd, "_is_ram_backed", lambda path: True)
    seen = []
    real_write = ad.AnnData.write_h5ad
    monkeypatch.setattr(ad.AnnData, "write_h5ad", lambda self, path, **kw: (seen.append(path), real_write(self, path, **kw))[1])
    out_dir = tmp_path / "out"
    cx.t_test(_screen(log=True), perturbation_column="perturbation", control_label="control",
              output_path=out_dir / "de.h5ad", verbose=False)
    assert seen[0].parent.parent == out_dir
    assert list(scratch_root.iterdir()) == []
    assert [p.name for p in out_dir.iterdir() if p.name.startswith(".cx_")] == []
