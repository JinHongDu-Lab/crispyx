"""Tests covering benchmark helpers that summarise experiment outputs."""

from __future__ import annotations

import resource
import sys
import unittest.mock as mock
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

from benchmarking.tools.run_benchmarks import (
    _summarise_dataframe,
    _uses_mmap_backed_files,
    _worker,
    BenchmarkMethod,
    create_benchmark_suite,
)
from crispyx import compute_average_log_expression


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
    result = compute_average_log_expression(
        dataset_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="summary_test",
    )

    context = {"output_dir": tmp_path}
    summary = _summarise_dataframe(result, context)

    assert summary["rows"] == 2
    assert summary["columns"] == 4
    assert summary["result_path"] == "summary_test_cx_avg_log_effects.h5ad"

    result.close()


def test_create_benchmark_suite_infers_control_label(tmp_path: Path) -> None:
    dataset_path = _create_demo_dataset(tmp_path, control_label="non_target_control")

    methods = create_benchmark_suite(dataset_path, tmp_path)

    assert methods  # sanity check
    # Only check methods that don't depend on other methods (lfcShrink methods
    # don't have control_label since they process results from base methods)
    # Also exclude format-conversion methods (crispyx_standardize_csr) which
    # don't operate on perturbation labels.
    primary_methods = {
        name: method for name, method in methods.items()
        if method.depends_on is None and "control_label" in method.kwargs
    }
    control_labels = {
        method.kwargs.get("control_label") for method in primary_methods.values()
    }
    assert control_labels == {"non_target_control"}


# ---------------------------------------------------------------------------
# Regression test: RLIMIT_CPU must NOT be set inside the worker for any method.
# RLIMIT_CPU accumulates CPU time across all pthreads (Numba, BLAS, rpy2).
# With 32 threads, a 21600 s CPU limit fires at ~675 s wall time — far below
# the intended 6-hour per-method budget.  Wall-clock enforcement is handled
# by the parent via process.join(timeout=time_limit+5).
# ---------------------------------------------------------------------------

def _noop(**kwargs):
    return None


def _noop_summary(result, context):
    return {}


@mock.patch("os._exit")
@mock.patch("resource.setrlimit")
def test_worker_does_not_set_rlimit_cpu_for_crispyx_method(mock_setrlimit, mock_os_exit):
    """RLIMIT_CPU must never be set for crispyx DE methods."""
    method = BenchmarkMethod(
        name="crispyx_de_wilcoxon",
        description="test",
        function=_noop,
        kwargs={},
        summary=_noop_summary,
    )
    q = mock.MagicMock()
    _worker(q, method, context={}, memory_limit=None, time_limit=21600,
            n_threads=32, use_cgroups_memory=False)

    cpu_calls = [
        call for call in mock_setrlimit.call_args_list
        if call.args and call.args[0] == resource.RLIMIT_CPU
    ]
    assert cpu_calls == [], (
        "RLIMIT_CPU was set inside the worker for a crispyx method — "
        "this causes SIGKILL when Numba threads accumulate CPU time faster "
        "than wall time (n_threads × wall_time > RLIMIT_CPU limit)."
    )


@mock.patch("os._exit")
@mock.patch("resource.setrlimit")
def test_worker_does_not_set_rlimit_cpu_for_scanpy_method(mock_setrlimit, mock_os_exit):
    """RLIMIT_CPU must never be set even for non-crispyx methods (scanpy, pertpy)."""
    method = BenchmarkMethod(
        name="scanpy_de_wilcoxon",
        description="test",
        function=_noop,
        kwargs={},
        summary=_noop_summary,
    )
    q = mock.MagicMock()
    _worker(q, method, context={}, memory_limit=None, time_limit=21600,
            n_threads=8, use_cgroups_memory=False)

    cpu_calls = [
        call for call in mock_setrlimit.call_args_list
        if call.args and call.args[0] == resource.RLIMIT_CPU
    ]
    assert cpu_calls == [], (
        "RLIMIT_CPU was set inside the worker for a scanpy method — "
        "OpenBLAS threads would accumulate CPU time and fire SIGXCPU "
        "well before the intended wall-clock budget."
    )


@mock.patch("os._exit")
@mock.patch("resource.setrlimit")
def test_worker_does_not_set_rlimit_as_for_crispyx_method(mock_setrlimit, mock_os_exit):
    """RLIMIT_AS must not be set for crispyx methods (mmap files inflate VSZ)."""
    method = BenchmarkMethod(
        name="crispyx_de_t_test",
        description="test",
        function=_noop,
        kwargs={},
        summary=_noop_summary,
    )
    assert _uses_mmap_backed_files(method), (
        "crispyx_de_t_test should be identified as an mmap-backed method"
    )
    q = mock.MagicMock()
    _worker(q, method, context={}, memory_limit=128 * 1024**3, time_limit=21600,
            n_threads=32, use_cgroups_memory=False)

    as_calls = [
        call for call in mock_setrlimit.call_args_list
        if call.args and call.args[0] == resource.RLIMIT_AS
    ]
    assert as_calls == [], (
        "RLIMIT_AS was set for a crispyx method — this causes spurious SIGKILL "
        "because mmap'd h5ad files inflate VSZ to ~42 GB while RSS stays at ~6 GB."
    )

