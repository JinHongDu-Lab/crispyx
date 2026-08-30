"""Tests covering benchmark helpers that summarise experiment outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from benchmarking.tools.run_benchmarks import _summarise_dataframe, create_benchmark_suite
from crispyx import compute_normalized_effects


def _create_demo_dataset(tmp_path: Path, control_label: str = "ctrl") -> Path:
    matrix = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 3, 0],
            [0, 1, 0, 0],
            [2, 1, 0, 4],
            [0, 0, 0, 5],
            [1, 0, 0, 6],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"perturbation": [control_label, control_label, "KO1", "KO1", "KO2", "KO2"]},
        index=[f"cell_{idx}" for idx in range(matrix.shape[0])],
    )
    var = pd.DataFrame({"gene_symbol": [f"gene{idx}" for idx in range(matrix.shape[1])]})
    var.index = var["gene_symbol"]
    adata = ad.AnnData(matrix, obs=obs, var=var)
    dataset_path = tmp_path / "benchmark_demo.h5ad"
    adata.write(dataset_path)
    return dataset_path


def test_summarise_dataframe_includes_result_path(tmp_path: Path) -> None:
    dataset_path = _create_demo_dataset(tmp_path)
    result = compute_normalized_effects(
        dataset_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        method="mean_log1p",
        output_dir=tmp_path,
        data_name="summary_test",
    )

    context = {"output_dir": tmp_path}
    summary = _summarise_dataframe(result, context)

    assert summary["rows"] == 2
    assert summary["columns"] == 4
    assert summary["result_path"] == "summary_test_cx_normalized_effects.h5ad"

    result.close()


def test_create_benchmark_suite_infers_control_label(tmp_path: Path) -> None:
    dataset_path = _create_demo_dataset(tmp_path, control_label="non_target_control")

    methods = create_benchmark_suite(dataset_path, tmp_path)

    assert methods  # sanity check
    # Only check methods that don't depend on other methods (lfcShrink methods
    # don't have control_label since they process results from base methods)
    # and that actually accept control_label (format-conversion utilities like
    # crispyx_standardize_csr operate on file paths only, with no perturbation
    # semantics, so they never carry a control_label kwarg).
    primary_methods = {
        name: method
        for name, method in methods.items()
        if method.depends_on is None and "control_label" in method.kwargs
    }
    assert primary_methods  # sanity check
    control_labels = {
        method.kwargs.get("control_label") for method in primary_methods.values()
    }
    assert control_labels == {"non_target_control"}
