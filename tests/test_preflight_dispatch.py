"""Tests for the public on-demand query crispyx.estimate_disk_usage()."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import pytest

import crispyx as cx
from crispyx._disk import DiskEstimate
from crispyx._preflight import estimate_disk_usage


def _make_normalised_h5ad(
    tmp_path: Path, n_cells: int = 60, n_genes: int = 20, n_perts: int = 3,
    with_batch: bool = False,
) -> Path:
    rng = np.random.default_rng(0)
    counts = rng.poisson(2, size=(n_cells, n_genes)).astype(float)
    labels = (
        ["control"] * (n_cells // 2)
        + [f"pert_{i}" for i in range(n_perts) for _ in range(n_cells // (2 * n_perts))]
    )
    while len(labels) < n_cells:
        labels.append("control")
    labels = labels[:n_cells]
    obs_dict = {"perturbation": labels}
    if with_batch:
        obs_dict["batch"] = [f"b{i % 3}" for i in range(n_cells)]
    obs = pd.DataFrame(obs_dict, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "test_norm.h5ad"
    adata.write(path)
    return path


def _make_csr_h5ad(tmp_path: Path, n_cells: int = 40, n_genes: int = 15) -> Path:
    rng = np.random.default_rng(1)
    counts = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    path = tmp_path / "csr_source.h5ad"
    adata.write(path)
    return path


class TestEstimateDiskUsage:
    def test_accepts_function_name_string(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path)
        result = estimate_disk_usage(
            "t_test", path, perturbation_column="perturbation", control_label="control",
        )
        assert "tempdir" in result
        assert isinstance(result["tempdir"], DiskEstimate)
        assert result["tempdir"].required_bytes > 0

    def test_accepts_function_object(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path)
        by_name = estimate_disk_usage(
            "t_test", path, perturbation_column="perturbation", control_label="control",
        )
        by_ref = estimate_disk_usage(
            cx.t_test, path, perturbation_column="perturbation", control_label="control",
        )
        assert by_name.keys() == by_ref.keys()
        assert by_name["tempdir"].required_bytes == by_ref["tempdir"].required_bytes

    def test_batch_column_adds_tempdir_entry(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path, with_batch=True)
        result = estimate_disk_usage(
            "compute_normalized_effects", path,
            perturbation_column="perturbation", batch_column="batch",
        )
        assert {"tempdir", "output"} <= result.keys()

    def test_no_batch_column_omits_tempdir(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path)
        result = estimate_disk_usage(
            "compute_normalized_effects", path, perturbation_column="perturbation",
        )
        assert "tempdir" not in result
        assert "output" in result

    def test_unregistered_function_raises_with_supported_list(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path)
        with pytest.raises(ValueError, match="No disk-usage estimator registered"):
            estimate_disk_usage("not_a_real_function", path)

    def test_conversion_functions_ignore_irrelevant_kwargs(self, tmp_path):
        path = _make_csr_h5ad(tmp_path)
        result = estimate_disk_usage(
            "convert_to_csc", path, chunk_size=4096, verbose=False,
        )
        assert result["output"].required_bytes == pytest.approx(2 * path.stat().st_size)

    def test_wilcoxon_reports_both_possible_sinks_without_batch(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path)
        result = estimate_disk_usage(
            "wilcoxon_test", path, perturbation_column="perturbation", control_label="control",
        )
        assert {"tempdir", "output"} <= result.keys()

    def test_wilcoxon_stratified_only_reports_tempdir(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path, with_batch=True)
        result = estimate_disk_usage(
            "wilcoxon_test", path,
            perturbation_column="perturbation", control_label="control", batch_column="batch",
        )
        assert result.keys() == {"tempdir"}

    def test_aggregate_pseudobulk(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path, with_batch=True)
        result = estimate_disk_usage(
            "aggregate_pseudobulk", path, groupby=["perturbation", "batch"],
        )
        assert {"tempdir", "output"} <= result.keys()

    def test_reachable_via_top_level_crispyx_namespace(self, tmp_path):
        path = _make_normalised_h5ad(tmp_path)
        result = cx.estimate_disk_usage(
            "t_test", path, perturbation_column="perturbation", control_label="control",
        )
        assert "tempdir" in result

    def test_estimate_matches_scale_of_real_run(self, tmp_path):
        """The estimate should be the right order of magnitude, not just nonzero."""
        path = _make_normalised_h5ad(tmp_path, n_cells=200, n_genes=500, n_perts=5)
        result = estimate_disk_usage(
            "t_test", path, perturbation_column="perturbation", control_label="control",
        )
        # 5 groups x 500 genes x (3 x 8-byte + 4 x 4-byte arrays) = 200,000 bytes.
        assert result["tempdir"].required_bytes == pytest.approx(5 * 500 * (3 * 8 + 4 * 4))
