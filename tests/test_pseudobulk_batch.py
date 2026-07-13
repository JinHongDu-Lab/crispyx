"""Tests for batch-corrected pseudobulk effect sizes."""

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
from crispyx.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)


def _write(tmp_path, counts, perturbation, batch=None, name="data.h5ad"):
    obs_data = {"perturbation": list(perturbation)}
    if batch is not None:
        obs_data["batch"] = list(batch)
    obs = pd.DataFrame(
        obs_data,
        index=[f"cell_{i}" for i in range(counts.shape[0])],
    )
    var = pd.DataFrame(
        {"gene_symbols": [f"G{i}" for i in range(counts.shape[1])]},
        index=[f"g{i}" for i in range(counts.shape[1])],
    )
    adata = ad.AnnData(sp.csr_matrix(np.asarray(counts, dtype=float)), obs=obs, var=var)
    path = tmp_path / name
    adata.write(path)
    return path


@pytest.fixture
def batched_adata(tmp_path):
    """Two batches, control + one perturbation present in both batches."""
    counts = np.array(
        [
            [1, 0, 3, 0],  # b1 ctrl
            [0, 2, 1, 0],  # b1 ctrl
            [4, 0, 0, 1],  # b1 A
            [3, 1, 0, 0],  # b1 A
            [0, 0, 2, 2],  # b2 ctrl
            [1, 0, 1, 3],  # b2 ctrl
            [2, 1, 0, 4],  # b2 A
            [0, 3, 1, 1],  # b2 A
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "ctrl", "ctrl", "A", "A"]
    batch = ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"]
    path = _write(tmp_path, counts, perturbation, batch)
    return path


# ---------------------------------------------------------------------------
# Single-batch equivalence: batch correction with one batch == pooled
# ---------------------------------------------------------------------------

def test_single_batch_equivalence_avg_log(tmp_path):
    counts = np.array(
        [
            [1, 0, 3, 0],
            [0, 2, 1, 0],
            [4, 0, 0, 1],
            [3, 1, 0, 0],
            [0, 0, 2, 2],
            [1, 0, 1, 3],
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "B", "B"]
    batch = ["only"] * 6
    path = _write(tmp_path, counts, perturbation, batch)

    pooled = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", output_path=tmp_path / "pooled.h5ad",
    ).to_memory()
    batched = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=tmp_path / "batched.h5ad",
    ).to_memory()

    np.testing.assert_allclose(batched.X, pooled.X, rtol=1e-10, atol=1e-12)


def test_single_batch_equivalence_pseudobulk(tmp_path):
    counts = np.array(
        [
            [1, 0, 3, 0],
            [0, 2, 1, 0],
            [4, 0, 0, 1],
            [3, 1, 0, 0],
            [0, 0, 2, 2],
            [1, 0, 1, 3],
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "B", "B"]
    batch = ["only"] * 6
    path = _write(tmp_path, counts, perturbation, batch)

    pooled = compute_pseudobulk_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", output_path=tmp_path / "pooled.h5ad",
    ).to_memory()
    batched = compute_pseudobulk_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=tmp_path / "batched.h5ad",
    ).to_memory()

    np.testing.assert_allclose(batched.X, pooled.X, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Reference: independent per-batch harmonic-weight implementation
# ---------------------------------------------------------------------------

def _reference_batch_effect(counts, perturbation, batch, label, transform):
    perturbation = np.asarray(perturbation)
    batch = np.asarray(batch)
    # normalize_total(target_sum=1e4) per cell
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    normed = counts / lib * 1e4

    total_w = 0.0
    weighted = None
    for b in np.unique(batch):
        p_mask = (perturbation == label) & (batch == b)
        c_mask = (perturbation == "ctrl") & (batch == b)
        n_p = int(p_mask.sum())
        n_c = int(c_mask.sum())
        if n_p == 0 or n_c == 0:
            continue
        w = (n_p * n_c) / (n_p + n_c)
        delta = transform(normed[p_mask]) - transform(normed[c_mask])
        weighted = w * delta if weighted is None else weighted + w * delta
        total_w += w
    return weighted / total_w


def test_batch_corrected_avg_log_matches_reference(batched_adata):
    counts = np.array(
        [
            [1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0],
            [0, 0, 2, 2], [1, 0, 1, 3], [2, 1, 0, 4], [0, 3, 1, 1],
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "ctrl", "ctrl", "A", "A"]
    batch = ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"]

    result = compute_average_log_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "out.h5ad",
    ).to_memory()

    def transform(x):
        return np.log1p(x).mean(axis=0)

    expected = _reference_batch_effect(counts, perturbation, batch, "A", transform)
    np.testing.assert_allclose(result.X[0], expected, rtol=1e-8, atol=1e-10)


def test_batch_corrected_pseudobulk_matches_reference(batched_adata):
    counts = np.array(
        [
            [1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0],
            [0, 0, 2, 2], [1, 0, 1, 3], [2, 1, 0, 4], [0, 3, 1, 1],
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "ctrl", "ctrl", "A", "A"]
    batch = ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"]

    result = compute_pseudobulk_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "out.h5ad",
    ).to_memory()

    def transform(x):
        return np.log1p(x.mean(axis=0))

    expected = _reference_batch_effect(counts, perturbation, batch, "A", transform)
    np.testing.assert_allclose(result.X[0], expected, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Confound removal
# ---------------------------------------------------------------------------

def test_batch_removes_confound(tmp_path):
    """Gene composition (not level) differs by batch, and group representation
    across batches is unbalanced, creating a spurious pooled effect.

    Within each batch the perturbation and control have identical composition
    (non-DE), so the batch-corrected estimate is exactly 0, while the pooled
    estimate is strongly biased because controls come mostly from batch b1
    (gene-0 rich) and perturbation cells mostly from batch b2 (gene-2 rich).
    """
    n_genes = 3
    b1_profile = [90, 5, 5]   # gene-0 dominant composition
    b2_profile = [5, 5, 90]   # gene-2 dominant composition

    rows = []
    perturbation = []
    batch = []

    # Batch b1: 40 ctrl + 5 A, all share the b1 composition
    for _ in range(40):
        rows.append(b1_profile); perturbation.append("ctrl"); batch.append("b1")
    for _ in range(5):
        rows.append(b1_profile); perturbation.append("A"); batch.append("b1")
    # Batch b2: 5 ctrl + 40 A, all share the b2 composition
    for _ in range(5):
        rows.append(b2_profile); perturbation.append("ctrl"); batch.append("b2")
    for _ in range(40):
        rows.append(b2_profile); perturbation.append("A"); batch.append("b2")

    counts = np.vstack(rows).astype(float)
    path = _write(tmp_path, counts, perturbation, batch)

    pooled = compute_pseudobulk_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", output_path=tmp_path / "pooled.h5ad",
    ).to_memory()
    batched = compute_pseudobulk_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=tmp_path / "batched.h5ad",
    ).to_memory()

    # Pooled estimate is strongly biased away from 0 (composition confounded
    # with group representation across batches).
    assert np.abs(pooled.X).max() > 0.2
    # Batch-corrected estimate removes the confound entirely (non-DE within
    # every batch).
    np.testing.assert_allclose(batched.X[0], np.zeros(n_genes), atol=1e-10)
    assert np.abs(batched.X).max() < np.abs(pooled.X).max()


# ---------------------------------------------------------------------------
# Validation and metadata
# ---------------------------------------------------------------------------

def test_missing_batch_column_raises(tmp_path):
    counts = np.array([[1, 0], [0, 2], [3, 1], [1, 1]], dtype=float)
    perturbation = ["ctrl", "ctrl", "A", "A"]
    path = _write(tmp_path, counts, perturbation, batch=["b1", "b1", "b1", "b1"])

    with pytest.raises(KeyError):
        compute_pseudobulk_expression(
            path, perturbation_column="perturbation", control_label="ctrl",
            gene_name_column="gene_symbols", batch_column="does_not_exist",
            output_path=tmp_path / "out.h5ad",
        )


def test_batch_with_empty_pert_batch_skipped(tmp_path):
    """Perturbation A absent in b2; its effect uses b1 only, no crash."""
    counts = np.array(
        [
            [1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0],
            [0, 0, 2, 2], [1, 0, 1, 3],
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "ctrl", "ctrl"]
    batch = ["b1", "b1", "b1", "b1", "b2", "b2"]
    path = _write(tmp_path, counts, perturbation, batch)

    result = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=tmp_path / "out.h5ad",
    ).to_memory()
    assert result.shape[0] == 1
    assert np.isfinite(result.X).all()


def test_no_shared_batch_raises(tmp_path):
    """Perturbation and control never co-occur in any batch -> error."""
    counts = np.array(
        [[1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0]],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A"]
    batch = ["b1", "b1", "b2", "b2"]
    path = _write(tmp_path, counts, perturbation, batch)

    with pytest.raises(ValueError, match="shares no batch"):
        compute_average_log_expression(
            path, perturbation_column="perturbation", control_label="ctrl",
            gene_name_column="gene_symbols", batch_column="batch",
            output_path=tmp_path / "out.h5ad",
        )


def test_uns_fields_written(batched_adata):
    result = compute_pseudobulk_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "out.h5ad",
    ).to_memory()
    assert result.uns["batch_column"] == "batch"
    assert sorted(map(str, result.uns["batch_ids"])) == ["b1", "b2"]


def test_no_batch_column_unchanged(tmp_path):
    """Default (batch_column=None) reproduces the pooled output and adds no
    batch metadata."""
    counts = np.array(
        [[1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0]],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A"]
    path = _write(tmp_path, counts, perturbation)

    result = compute_pseudobulk_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", output_path=tmp_path / "out.h5ad",
    ).to_memory()
    assert "batch_column" not in result.uns
    assert "batch_ids" not in result.uns


# ---------------------------------------------------------------------------
# Namespace wiring
# ---------------------------------------------------------------------------

def test_namespace_passes_batch_column(batched_adata):
    direct = compute_pseudobulk_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "direct.h5ad",
    ).to_memory()
    wrapped = cx.pb.pseudobulk(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "wrapped.h5ad",
    ).to_memory()
    np.testing.assert_allclose(wrapped.X, direct.X, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Batch-corrected per-perturbation mean layer (item 2)
# ---------------------------------------------------------------------------

def test_corrected_mean_consistency_avg_log(batched_adata):
    """X == perturbation_mean - control_mean_matched (batch-corrected layers)."""
    result = compute_average_log_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "out.h5ad",
    ).to_memory()
    assert "control_mean_matched" in result.layers
    recon = result.layers["perturbation_mean"] - result.layers["control_mean_matched"]
    np.testing.assert_allclose(result.X, recon, rtol=1e-10, atol=1e-12)


def test_corrected_mean_consistency_pseudobulk(batched_adata):
    """X == perturbation_bulk - control_bulk_matched (batch-corrected layers)."""
    result = compute_pseudobulk_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "out.h5ad",
    ).to_memory()
    assert "control_bulk_matched" in result.layers
    recon = result.layers["perturbation_bulk"] - result.layers["control_bulk_matched"]
    np.testing.assert_allclose(result.X, recon, rtol=1e-10, atol=1e-12)


def test_corrected_mean_matches_reference(batched_adata):
    """Corrected perturbation_mean / control_mean_matched match an independent
    per-batch harmonic-weight reference."""
    counts = np.array(
        [
            [1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0],
            [0, 0, 2, 2], [1, 0, 1, 3], [2, 1, 0, 4], [0, 3, 1, 1],
        ],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "ctrl", "ctrl", "A", "A"]
    batch = ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"]

    result = compute_average_log_expression(
        batched_adata, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=batched_adata.parent / "out.h5ad",
    ).to_memory()

    tf = lambda x: np.log1p(x).mean(axis=0)

    def weighted(which):
        perturbation_arr = np.asarray(perturbation)
        batch_arr = np.asarray(batch)
        lib = counts.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        normed = counts / lib * 1e4
        num = None
        tot = 0.0
        for b in np.unique(batch_arr):
            pm = (perturbation_arr == "A") & (batch_arr == b)
            cm = (perturbation_arr == "ctrl") & (batch_arr == b)
            n_p, n_c = int(pm.sum()), int(cm.sum())
            if n_p == 0 or n_c == 0:
                continue
            w = (n_p * n_c) / (n_p + n_c)
            expr = tf(normed[pm]) if which == "pert" else tf(normed[cm])
            num = w * expr if num is None else num + w * expr
            tot += w
        return num / tot

    np.testing.assert_allclose(result.layers["perturbation_mean"][0], weighted("pert"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.layers["control_mean_matched"][0], weighted("ctrl"), rtol=1e-8, atol=1e-10)


def test_single_batch_mean_equals_pooled(tmp_path):
    """With one batch, corrected perturbation_mean equals the pooled mean and
    control_mean_matched equals the pooled uns['control_mean']."""
    counts = np.array(
        [[1, 0, 3, 0], [0, 2, 1, 0], [4, 0, 0, 1], [3, 1, 0, 0], [0, 0, 2, 2], [1, 0, 1, 3]],
        dtype=float,
    )
    perturbation = ["ctrl", "ctrl", "A", "A", "B", "B"]
    path = _write(tmp_path, counts, perturbation, batch=["only"] * 6)

    pooled = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", output_path=tmp_path / "pooled.h5ad",
    ).to_memory()
    batched = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=tmp_path / "batched.h5ad",
    ).to_memory()

    np.testing.assert_allclose(
        batched.layers["perturbation_mean"], pooled.layers["perturbation_mean"],
        rtol=1e-10, atol=1e-12,
    )
    # Single batch -> matched control equals the pooled control for every pert.
    for row in batched.layers["control_mean_matched"]:
        np.testing.assert_allclose(row, pooled.uns["control_mean"], rtol=1e-10, atol=1e-12)


def test_matched_layer_only_with_batch(tmp_path):
    """control_mean_matched present iff batch_column is set."""
    counts = np.array([[1, 0], [0, 2], [3, 1], [1, 1]], dtype=float)
    perturbation = ["ctrl", "ctrl", "A", "A"]
    path = _write(tmp_path, counts, perturbation, batch=["b1", "b1", "b1", "b1"])

    pooled = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", output_path=tmp_path / "pooled.h5ad",
    ).to_memory()
    batched = compute_average_log_expression(
        path, perturbation_column="perturbation", control_label="ctrl",
        gene_name_column="gene_symbols", batch_column="batch",
        output_path=tmp_path / "batched.h5ad",
    ).to_memory()
    assert "control_mean_matched" not in pooled.layers
    assert "control_mean_matched" in batched.layers

