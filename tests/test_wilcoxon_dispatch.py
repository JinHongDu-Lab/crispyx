"""Wilcoxon kernels, the standard vs streaming paths, and the result file helpers.

Covers:
- _presort_control_nonzeros and parity of the presorted kernel with the reference kernel
- Standard (memmap) and streaming (group-batch) paths giving the same result
- _write_wilcoxon_result_h5ad / _build_result_from_h5ad
- Agreement with Scanpy on dense and mixed dense/sparse genes

The memory-based dispatch heuristics are tested in ``test_memory_dispatch.py``.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from crispyx._kernels import (
    _presort_control_nonzeros,
    _wilcoxon_presorted_ctrl_numba,
    _wilcoxon_sparse_batch_numba,
)
from crispyx.de import _build_result_from_h5ad, _write_wilcoxon_result_h5ad, wilcoxon_test

ADAMSON_PATH = Path(__file__).resolve().parents[1] / "data" / "Adamson_subset.h5ad"

_RESULT_FIELDS = ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "effect_size", "pts", "pts_rest")


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


def _assert_paths_agree(standard, streaming):
    assert standard.groups == streaming.groups
    for field in _RESULT_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(streaming, field)), np.asarray(getattr(standard, field)),
            atol=1e-10, err_msg=f"{field} differs between the standard and streaming paths",
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


def _run_kernel(kernel_args, ctrl, pert, valid, tie_correct):
    n_genes = ctrl.shape[1]
    out = (np.zeros(n_genes), np.zeros(n_genes), np.ones(n_genes), np.zeros(n_genes))
    kernel_args(ctrl, pert, valid, tie_correct, out)
    return out


@pytest.mark.parametrize(
    "n_ctrl, n_pert, n_genes, sparsity, tie_correct, every_other_invalid",
    [
        (50, 30, 32, 0.3, True, False),
        (500, 200, 64, 0.3, True, False),
        (50_000, 50, 128, 0.2, True, False),  # Feng-scale control
        (200, 80, 32, 1.0, False, False),
        (100, 30, 16, 1.0, True, True),
    ],
    ids=["small", "medium", "large-ctrl", "no-tie-correct", "invalid-genes"],
)
def test_presorted_kernel_matches_reference_kernel(n_ctrl, n_pert, n_genes, sparsity, tie_correct, every_other_invalid):
    rng = np.random.default_rng(n_ctrl)
    ctrl = ((rng.random((n_ctrl, n_genes)) < sparsity) * rng.exponential(3, (n_ctrl, n_genes))).astype(np.float64)
    pert = ((rng.random((n_pert, n_genes)) < sparsity) * rng.exponential(3, (n_pert, n_genes))).astype(np.float64)
    valid = np.ones(n_genes, dtype=np.bool_)
    if every_other_invalid:
        valid[1::2] = False

    def reference(c, p, v, t, out):
        _wilcoxon_sparse_batch_numba(c, p, v, t, 0.5, *out)

    def presorted(c, p, v, t, out):
        _wilcoxon_presorted_ctrl_numba(c, *_presort_control_nonzeros(c), p, v, t, 0.5, *out)

    expected = _run_kernel(reference, ctrl, pert, valid, tie_correct)
    actual = _run_kernel(presorted, ctrl, pert, valid, tie_correct)
    for name, a, e in zip(("u", "z", "p", "effect"), actual, expected):
        np.testing.assert_allclose(a, e, atol=1e-9, rtol=1e-6, err_msg=name)
    # Invalid genes keep the neutral fill: u=0, z=0, p=1.
    u, z, p, _ = actual
    np.testing.assert_array_equal(u[~valid], 0.0)
    np.testing.assert_array_equal(z[~valid], 0.0)
    np.testing.assert_array_equal(p[~valid], 1.0)


# ---------------------------------------------------------------------------
# Standard vs streaming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_ctrl, n_perts", [(50, 10), (500, 5), (2000, 5)])
def test_streaming_matches_standard_on_synthetic_data(tmp_path, n_ctrl, n_perts):
    rng = np.random.default_rng(n_ctrl)
    n_cells = n_ctrl + n_perts * 10
    data = (rng.random((n_cells, 64)) < 0.3) * rng.exponential(2, (n_cells, 64))
    path = _write_log_normalised(tmp_path / "data.h5ad", data, _labels(n_ctrl, n_perts, 10))
    kw = dict(perturbation_column="perturbation", verbose=False)
    standard = wilcoxon_test(path, output_path=tmp_path / "std.h5ad", memory_limit_gb=128, **kw)
    streaming = wilcoxon_test(path, output_path=tmp_path / "stream.h5ad", memory_limit_gb=1e-7, **kw)
    _assert_paths_agree(standard, streaming)


@pytest.mark.skipif(not ADAMSON_PATH.exists(), reason="data/Adamson_subset.h5ad not found")
def test_streaming_matches_standard_on_real_data(tmp_path):
    """Real sparsity and tie structure; 2,000 genes keep the streaming run short."""
    adata = ad.read_h5ad(ADAMSON_PATH)[:, :2000].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    path = tmp_path / "adamson_norm.h5ad"
    adata.write(path)
    kw = dict(perturbation_column="perturbation", control_label="control", verbose=False)
    standard = wilcoxon_test(path, output_path=tmp_path / "std.h5ad", memory_limit_gb=128, **kw)
    streaming = wilcoxon_test(path, output_path=tmp_path / "stream.h5ad", memory_limit_gb=1e-7, **kw)
    _assert_paths_agree(standard, streaming)


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------

def _result_arrays(n_groups: int, n_genes: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        effect_matrix=rng.standard_normal((n_groups, n_genes)),
        z_matrix=rng.standard_normal((n_groups, n_genes)),
        pvalue_matrix=rng.random((n_groups, n_genes)),
        pvalue_adj_matrix=rng.random((n_groups, n_genes)),
        lfc_matrix=rng.standard_normal((n_groups, n_genes)),
        u_matrix=rng.random((n_groups, n_genes)) * 1000,
        pts_matrix=rng.random((n_groups, n_genes)).astype(np.float32),
        pts_rest=rng.random(n_genes).astype(np.float32),
    )


def _write_result(path: Path, n_groups: int = 4, n_genes: int = 16, **overrides) -> tuple[dict, dict]:
    meta = dict(
        candidates=[f"p{i}" for i in range(n_groups)],
        gene_symbols=pd.Index([f"g{i}" for i in range(n_genes)]),
        perturbation_column="perturbation",
        control_label="ctrl",
        tie_correct=True,
        corr_method="benjamini-hochberg",
    )
    meta.update(overrides)
    arrays = _result_arrays(n_groups, n_genes)
    _write_wilcoxon_result_h5ad(path, **meta, **arrays)
    return meta, arrays


def test_write_result_layout(tmp_path):
    out = tmp_path / "result.h5ad"
    meta, arrs = _write_result(out, tie_correct=False, corr_method="bonferroni")
    with h5py.File(out, "r") as hf:
        np.testing.assert_allclose(hf["X"][:], arrs["effect_matrix"])
        for layer, key in [
            ("z_score", "z_matrix"), ("pvalue", "pvalue_matrix"), ("pvalue_adj", "pvalue_adj_matrix"),
            ("logfoldchanges", "lfc_matrix"), ("u_statistic", "u_matrix"), ("pts", "pts_matrix"),
        ]:
            np.testing.assert_allclose(hf[f"layers/{layer}"][:], arrs[key], err_msg=layer)
        assert "pts_rest" not in hf["layers"]
    result = ad.read_h5ad(out)
    np.testing.assert_array_equal(result.var["pts_rest"].to_numpy(), arrs["pts_rest"])
    assert result.obs_names.tolist() == meta["candidates"]
    assert result.var_names.tolist() == list(meta["gene_symbols"])
    assert result.uns["method"] == "wilcoxon"
    assert not result.uns["tie_correct"]
    assert result.uns["pvalue_correction"] == "bonferroni"
    assert not result.uns["stratified"]


def test_build_result_reads_the_file_back(tmp_path):
    out = tmp_path / "result.h5ad"
    meta, arrs = _write_result(out, n_groups=6, n_genes=24)
    result = _build_result_from_h5ad(out, **meta, memory_limit_gb=128.0)
    assert list(result.groups) == meta["candidates"]
    assert list(result.genes) == list(meta["gene_symbols"])
    np.testing.assert_allclose(result.statistics, arrs["z_matrix"])
    np.testing.assert_allclose(result.effect_size, arrs["effect_matrix"])
    np.testing.assert_allclose(result.pvalues, arrs["pvalue_matrix"])
    np.testing.assert_allclose(result.logfoldchanges, arrs["lfc_matrix"])
    np.testing.assert_array_equal(result.pts_rest, np.broadcast_to(arrs["pts_rest"], result.pts.shape))
    np.testing.assert_array_equal(result.order, np.argsort(-np.abs(arrs["z_matrix"]), axis=1, kind="mergesort"))


def test_build_result_is_lazy_when_memory_is_short(tmp_path, monkeypatch):
    """With too little physical memory the arrays stay on disk."""
    import sys
    import types

    fake_psutil = types.ModuleType("psutil")
    fake_psutil.virtual_memory = lambda: types.SimpleNamespace(available=100)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    out = tmp_path / "result.h5ad"
    meta, _ = _write_result(out)
    result = _build_result_from_h5ad(out, **meta, memory_limit_gb=None)
    assert result.statistics.size == 0
    assert result.result is not None  # the AnnData handle is still set


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
def test_all_zero_gene_is_untested(tmp_path, path_kwargs):
    """With the filters off, a gene that is zero in every cell has no rank
    test (z = 0/0): NaN in every derived column, like t_test and nb_glm_test,
    not "tested, no change" (Scanpy reports p = 1 here)."""
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
                           min_mean_ctrl=0.0, min_mean_pert=0.0, **path_kwargs)
    for field in ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "effect_size"):
        column = np.asarray(getattr(result, field))
        assert np.isnan(column[:, 0]).all(), field
        assert np.isfinite(column[:, 1:]).all(), field
    assert np.isfinite(result.pts[:, 0]).all()  # the data itself is still described
