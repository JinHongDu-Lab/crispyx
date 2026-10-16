"""Wilcoxon kernels, gene chunking, and the result file.

Covers:
- _presort_control_nonzeros and the presorted kernel against scipy's ranks
- The result not depending on the gene chunk width (set by the memory budget)
- The result file's layout, and the result read back from it
- Agreement with Scanpy on dense and mixed dense/sparse genes

The memory-based chunk sizing is tested in ``test_memory_dispatch.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from crispyx._kernels import (
    _compute_ctrl_tie_sums,
    _presort_control_nonzeros,
    _wilcoxon_batch_perts_presorted_numba,
)
from crispyx.de import wilcoxon_test

ADAMSON_PATH = Path(__file__).resolve().parents[1] / "data" / "Adamson_subset.h5ad"

_RESULT_FIELDS = ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "auc", "pts", "pts_rest")


def _write_log_normalised(path: Path, data: np.ndarray, labels: list[str]) -> Path:
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=[f"g{i}" for i in range(data.shape[1])])
    adata = ad.AnnData(sp.csr_matrix(data), obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(path)
    return path


def _labels(n_ctrl: int, n_perts: int, cells_per_pert: int) -> list[str]:
    return ["control"] * n_ctrl + [f"p{i}" for i in range(n_perts) for _ in range(cells_per_pert)]


def _assert_results_agree(wide, narrow):
    assert wide.groups == narrow.groups
    for field in _RESULT_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(narrow, field)), np.asarray(getattr(wide, field)),
            atol=1e-10, err_msg=f"{field} depends on the gene chunk width",
        )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _random_ctrl(n_ctrl, n_genes, sparsity, seed=0):
    rng = np.random.default_rng(seed)
    return ((rng.random((n_ctrl, n_genes)) < sparsity) * rng.exponential(2, (n_ctrl, n_genes))).astype(np.float64)


@pytest.mark.parametrize(
    "ctrl",
    [
        _random_ctrl(150, 48, 0.3),
        _random_ctrl(5000, 16, 0.2),
        np.zeros((50, 10)),
        np.ones((30, 8)),
    ],
    ids=["sparse", "large", "all-zero", "all-nonzero"],
)
def test_presort_returns_each_genes_sorted_nonzeros(ctrl):
    flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
    n_ctrl, n_genes = ctrl.shape
    expected_nz = (ctrl != 0).sum(axis=0)
    assert offsets.shape == (n_genes + 1,)
    np.testing.assert_array_equal(n_nz, expected_nz)
    np.testing.assert_array_equal(n_z, n_ctrl - expected_nz)
    assert flat.shape == (int(expected_nz.sum()),)
    for g in range(n_genes):
        np.testing.assert_array_equal(flat[offsets[g]:offsets[g + 1]], np.sort(ctrl[ctrl[:, g] != 0, g]))


def _reference_rank_test(ctrl, pert, valid, tie_correct):
    """U, z, p and AUC of each gene from scipy's ranks: NaN where every
    value is tied, the neutral fill (0, 0, 1, 0.5) where ``valid`` is False."""
    from scipy.stats import rankdata, tiecorrect
    n0, n1 = ctrl.shape[0], pert.shape[0]
    n = n0 + n1
    out = np.zeros((4, ctrl.shape[1]))
    out[2] = 1.0
    out[3] = 0.5
    for g in np.flatnonzero(valid):
        ranks = rankdata(np.concatenate([ctrl[:, g], pert[:, g]]))
        if tiecorrect(ranks) == 0:
            out[:, g] = np.nan
            continue
        rank_sum = ranks[n0:].sum()
        u = rank_sum - n1 * (n1 + 1) / 2
        std = np.sqrt((tiecorrect(ranks) if tie_correct else 1.0) * n1 * n0 * (n + 1) / 12)
        z = (rank_sum - n1 * (n + 1) / 2) / std
        out[:, g] = u, z, math.erfc(abs(z) / math.sqrt(2)), u / (n1 * n0)
    return out


@pytest.mark.parametrize("tie_correct", [True, False], ids=["tie-correct", "no-tie-correct"])
@pytest.mark.parametrize(
    "n_ctrl, n_pert, n_genes, sparsity, every_other_invalid",
    [
        (50, 30, 32, 0.3, False),
        (500, 200, 64, 0.3, False),
        (50_000, 50, 128, 0.2, False),  # Feng-scale control
        (200, 80, 32, 1.0, False),
        (100, 30, 16, 1.0, True),
    ],
    ids=["small", "medium", "large-ctrl", "dense", "invalid-genes"],
)
def test_presorted_kernel_matches_scipy_ranks(n_ctrl, n_pert, n_genes, sparsity, tie_correct, every_other_invalid):
    """The production kernel against scipy's ranks, on rounded values (so
    non-zero ties occur) plus genes tied at zero, at a non-zero value and
    with the perturbation all zero."""
    rng = np.random.default_rng(n_ctrl)

    def draw(n):
        values = (rng.random((n, n_genes)) < sparsity) * rng.exponential(3, (n, n_genes))
        return np.round(values, 1)

    ctrl, pert = draw(n_ctrl), draw(n_pert)
    ctrl[:, 0] = pert[:, 0] = 0.0  # all zero
    ctrl[:, 1] = pert[:, 1] = 2.5  # tied at one non-zero value
    pert[:, 2] = 0.0               # perturbation silent, control not
    valid = np.ones(n_genes, dtype=np.bool_)
    if every_other_invalid:
        valid[3::2] = False

    flat, offsets, n_nz, n_z = _presort_control_nonzeros(ctrl)
    out = np.zeros((4, 1, n_genes))
    out[2] = 1.0
    _wilcoxon_batch_perts_presorted_numba(
        ctrl, flat, offsets, n_nz, n_z, _compute_ctrl_tie_sums(flat, offsets, n_nz),
        pert, np.array([0, n_pert]), valid[None, :], tie_correct, 0.5, *out,
    )
    expected = _reference_rank_test(ctrl, pert, valid, tie_correct)
    for name, a, e in zip(("u", "z", "p", "auc"), out[:, 0], expected):
        np.testing.assert_allclose(a, e, atol=1e-9, rtol=1e-6, err_msg=name)
    assert np.isnan(out[:, 0, :2]).all()


# ---------------------------------------------------------------------------
# Gene chunk width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_ctrl, n_perts", [(50, 10), (500, 5), (2000, 5)])
def test_result_does_not_depend_on_the_chunk_width(tmp_path, n_ctrl, n_perts):
    rng = np.random.default_rng(n_ctrl)
    n_cells = n_ctrl + n_perts * 10
    data = (rng.random((n_cells, 64)) < 0.3) * rng.exponential(2, (n_cells, 64))
    path = _write_log_normalised(tmp_path / "data.h5ad", data, _labels(n_ctrl, n_perts, 10))
    kw = dict(perturbation_column="perturbation", verbose=False)
    wide = wilcoxon_test(path, output_path=tmp_path / "wide.h5ad", **kw)
    narrow = wilcoxon_test(path, output_path=tmp_path / "narrow.h5ad", chunk_size=7, **kw)
    _assert_results_agree(wide, narrow)


@pytest.mark.skipif(not ADAMSON_PATH.exists(), reason="data/Adamson_subset.h5ad not found")
def test_result_does_not_depend_on_the_chunk_width_on_real_data(tmp_path):
    """Real sparsity and tie structure; 2,000 genes keep the narrow run short."""
    adata = ad.read_h5ad(ADAMSON_PATH)[:, :2000].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    path = tmp_path / "adamson_norm.h5ad"
    adata.write(path)
    kw = dict(perturbation_column="perturbation", control_label="control", verbose=False)
    wide = wilcoxon_test(path, output_path=tmp_path / "wide.h5ad", **kw)
    narrow = wilcoxon_test(path, output_path=tmp_path / "narrow.h5ad", chunk_size=64, **kw)
    _assert_results_agree(wide, narrow)


# ---------------------------------------------------------------------------
# Result file
# ---------------------------------------------------------------------------

def _small_result(tmp_path: Path, **kw):
    rng = np.random.default_rng(1)
    data = (rng.random((90, 24)) < 0.4) * rng.exponential(2, (90, 24))
    path = _write_log_normalised(tmp_path / "data.h5ad", data, _labels(30, 6, 10))
    return wilcoxon_test(
        path, perturbation_column="perturbation", control_label="control", verbose=False,
        output_path=tmp_path / "result.h5ad", min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, **kw,
    )


def test_result_layout(tmp_path):
    result = _small_result(tmp_path, tie_correct=False, corr_method="bonferroni")
    with h5py.File(result.result_path, "r") as hf:
        assert sorted(hf["layers"]) == ["logfoldchanges", "pts", "pvalue", "pvalue_adj", "u_statistic", "z_score"]
        assert hf["X"].shape == (6, 24) and hf["X"].dtype == np.float64
        assert hf["layers/pts"].dtype == np.float32
    stored = ad.read_h5ad(result.result_path)
    assert stored.obs_names.tolist() == [f"p{i}" for i in range(6)]
    assert stored.var["pts_rest"].dtype == np.float32 and (stored.var["pts_rest"] > 0).all()
    assert stored.uns["method"] == "wilcoxon"
    assert not stored.uns["tie_correct"]
    assert stored.uns["pvalue_correction"] == "bonferroni"
    assert not stored.uns["stratified"]
    assert (stored.obs["n_genes_tested"] == np.isfinite(stored.layers["pvalue"]).sum(axis=1)).all()


def test_result_reads_the_file_back(tmp_path):
    result = _small_result(tmp_path)
    stored = ad.read_h5ad(result.result_path)
    z = stored.layers["z_score"]
    np.testing.assert_array_equal(np.asarray(result.statistics), z)
    np.testing.assert_array_equal(np.asarray(result.auc), stored.X)
    np.testing.assert_array_equal(np.asarray(result.u_statistics), stored.layers["u_statistic"])
    np.testing.assert_array_equal(result.pvalues[3], stored.layers["pvalue"][3])
    np.testing.assert_array_equal(result.pts_rest, np.broadcast_to(stored.var["pts_rest"], z.shape))
    np.testing.assert_array_equal(np.asarray(result.order), np.argsort(-np.abs(z), axis=1, kind="mergesort"))
    np.testing.assert_array_equal(result["p2"].auc, stored.X[2])


# ---------------------------------------------------------------------------
# Agreement with Scanpy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dense_fraction", [1.0, 0.5], ids=["dense", "mixed"])
def test_matches_scanpy_on_dense_and_mixed_genes(tmp_path, dense_fraction):
    """Genes with few zeros take the binary-search path, which must match Scanpy."""
    rng = np.random.default_rng(99)
    n_cells, n_genes = 250, 20
    n_dense = int(n_genes * dense_fraction)
    dense = rng.exponential(2.0, (n_cells, n_dense)) + 0.1
    sparse = (rng.random((n_cells, n_genes - n_dense)) < 0.2) * rng.exponential(2.0, (n_cells, n_genes - n_dense))
    path = _write_log_normalised(tmp_path / "data.h5ad", np.hstack([dense, sparse]), _labels(200, 5, 10))
    result = wilcoxon_test(path, perturbation_column="perturbation", control_label="control",
                           output_path=tmp_path / "result.h5ad", verbose=False)
    adata = sc.read_h5ad(path)
    sc.tl.rank_genes_groups(adata, groupby="perturbation", reference="control", method="wilcoxon", tie_correct=True)
    for label in result.groups:
        sc_df = sc.get.rank_genes_groups_df(adata, group=label).set_index("names")
        np.testing.assert_allclose(result[label].pvalue, sc_df.reindex(result.genes)["pvals"].to_numpy(),
                                   rtol=1e-4, atol=1e-10, err_msg=label)


@pytest.mark.parametrize(
    "path_kwargs",
    [dict(memory_limit_gb=128), dict(memory_limit_gb=1e-7), dict(batch_column="batch")],
    ids=["standard", "streaming", "stratified"],
)
@pytest.mark.parametrize("tie_correct", [True, False], ids=["tie-correct", "no-tie-correct"])
def test_all_zero_gene_is_untested(tmp_path, path_kwargs, tie_correct):
    """With the filters off, a gene that is zero in every cell has no rank
    test: NaN in every derived column, like t_test and nb_glm_test, not
    "tested, no change" (Scanpy reports p = 1 here) -- on every path, with
    or without the tie correction."""
    rng = np.random.default_rng(123)
    data = (rng.random((115, 10)) < 0.3) * rng.exponential(2, (115, 10))
    data[:, 0] = 0.0
    path = _write_log_normalised(tmp_path / "data.h5ad", data, _labels(100, 3, 5))
    adata = ad.read_h5ad(path)
    adata.obs["batch"] = np.resize(["b1", "b2"], adata.n_obs)
    adata.write(path)
    result = wilcoxon_test(path, perturbation_column="perturbation", control_label="control",
                           output_path=tmp_path / "result.h5ad", verbose=False,
                           min_cells_expressed=0, min_pct_ctrl=0.0, min_pct_pert=0.0,
                           min_mean_ctrl=0.0, min_mean_pert=0.0, tie_correct=tie_correct,
                           **path_kwargs)
    for field in ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "auc"):
        column = np.asarray(getattr(result, field))
        assert np.isnan(column[:, 0]).all(), field
        assert np.isfinite(column[:, 1:]).all(), field
    assert np.isfinite(result.pts[:, 0]).all()  # the data itself is still described
