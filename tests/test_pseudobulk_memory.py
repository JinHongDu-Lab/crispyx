"""Tests for pseudo-bulk chunk-size auto-selection and ``memory_limit_gb``."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx import pseudobulk as pb_mod
from crispyx.pseudobulk import compute_normalized_effects


def _write(tmp_path, n_cells=40, n_genes=6, name="data.h5ad"):
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    perturbation = (["ctrl"] * (n_cells // 2)) + (["A"] * (n_cells - n_cells // 2))
    obs = pd.DataFrame(
        {"perturbation": perturbation},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_symbols": [f"G{i}" for i in range(n_genes)]},
        index=[f"g{i}" for i in range(n_genes)],
    )
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    path = tmp_path / name
    adata.write(path)
    return path


def test_namespace_chunk_size_defaults_none(tmp_path):
    """``cx.pb.normalized_effects`` auto-sizes by default and matches an
    explicit chunk size bit-for-bit."""
    path = _write(tmp_path)
    auto = cx.pb.normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", output_path=tmp_path / "auto.h5ad",
    ).to_memory()
    explicit = cx.pb.normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", chunk_size=3,
        output_path=tmp_path / "explicit.h5ad",
    ).to_memory()
    np.testing.assert_allclose(auto.X, explicit.X, rtol=1e-12, atol=1e-14)


def test_memory_limit_threaded_into_chunk_sizing(tmp_path, monkeypatch):
    """``memory_limit_gb`` is forwarded to ``calculate_optimal_chunk_size`` as
    ``available_memory_gb``."""
    path = _write(tmp_path)
    seen = {}
    orig = pb_mod.calculate_optimal_chunk_size

    def spy(n_obs, n_vars, available_memory_gb=None, **kw):
        seen["available_memory_gb"] = available_memory_gb
        return orig(n_obs, n_vars, available_memory_gb=available_memory_gb, **kw)

    monkeypatch.setattr(pb_mod, "calculate_optimal_chunk_size", spy)
    compute_normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", memory_limit_gb=4.0,
        output_path=tmp_path / "out.h5ad",
    )
    assert seen["available_memory_gb"] == 4.0


def test_memory_limit_result_invariance(tmp_path):
    """Results are identical regardless of ``memory_limit_gb`` (only the chunk
    size, not the numbers, changes)."""
    path = _write(tmp_path)
    small = compute_normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", memory_limit_gb=0.001,
        output_path=tmp_path / "small.h5ad",
    ).to_memory()
    big = compute_normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", memory_limit_gb=256.0,
        output_path=tmp_path / "big.h5ad",
    ).to_memory()
    np.testing.assert_allclose(small.X, big.X, rtol=1e-12, atol=1e-14)


def test_memory_limit_with_batch_correction(tmp_path):
    """``memory_limit_gb`` works together with batch correction and does not
    change the computed effect."""
    rng = np.random.default_rng(1)
    n_cells, n_genes = 60, 5
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    perturbation = np.where(np.arange(n_cells) % 2 == 0, "ctrl", "A")
    batch = np.where(np.arange(n_cells) < n_cells // 2, "b1", "b2")
    obs = pd.DataFrame(
        {"perturbation": perturbation, "batch": batch},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_symbols": [f"G{i}" for i in range(n_genes)]},
        index=[f"g{i}" for i in range(n_genes)],
    )
    path = tmp_path / "b.h5ad"
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var).write(path)

    ref = compute_normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", batch_column="batch", chunk_size=64,
        output_path=tmp_path / "ref.h5ad",
    ).to_memory()
    limited = compute_normalized_effects(
        path, perturbation_column="perturbation", control_label="ctrl",
        method="mean_log1p",
        gene_name_column="gene_symbols", batch_column="batch", memory_limit_gb=0.001,
        output_path=tmp_path / "lim.h5ad",
    ).to_memory()
    np.testing.assert_allclose(ref.X, limited.X, rtol=1e-10, atol=1e-12)


def test_warns_when_disk_space_low_for_batch_corrected_accumulator(tmp_path, monkeypatch):
    """The batch-corrected accumulator warns on a near-full disk but still completes."""
    import shutil
    import types

    from crispyx.pseudobulk import compute_normalized_effects

    rng = np.random.default_rng(2)
    n_cells, n_genes = 40, 5
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    perturbation = np.where(np.arange(n_cells) % 2 == 0, "ctrl", "A")
    batch = np.where(np.arange(n_cells) < n_cells // 2, "b1", "b2")
    obs = pd.DataFrame(
        {"perturbation": perturbation, "batch": batch},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    path = tmp_path / "low_disk.h5ad"
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var).write(path)

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda p: types.SimpleNamespace(total=1000, used=999, free=1),
    )
    with pytest.warns(UserWarning, match="pb batch-corrected accumulator"):
        result = compute_normalized_effects(
            path, perturbation_column="perturbation", control_label="ctrl",
            batch_column="batch", output_path=tmp_path / "out.h5ad",
        ).to_memory()
    assert result.shape[0] == 1
