"""Tests for DE result auto-reload (force param) and pickle round-trip fixes.

Covers:
- AnnData.__getattr__ recursion guard (Tasks 1)
- AnnData.__getstate__ / __setstate__ (Task 2)
- RankGenesGroupsResult pickle round-trip (Task 3)
- DifferentialExpressionResult pickle round-trip (Task 3)
- wilcoxon_test / t_test / nb_glm_test auto-reload when output exists (Task 4)
- force=True always reruns and overwrites (Task 4)
- verbose output printed on reload (Task 4)
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import crispyx as cx
from crispyx.data import AnnData
from crispyx.de import (
    DifferentialExpressionResult,
    RankGenesGroupsResult,
    t_test,
    wilcoxon_test,
    nb_glm_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h5ad(
    tmp_path: Path,
    *,
    n_ctrl: int = 40,
    n_pert: int = 30,
    n_genes: int = 20,
    log_normalise: bool = True,
    seed: int = 0,
    fname: str = "data.h5ad",
) -> Path:
    """Build a minimal log-normalised h5ad for DE tests."""
    rng = np.random.default_rng(seed)
    n_cells = n_ctrl + n_pert
    counts = (rng.random((n_cells, n_genes)) < 0.4) * rng.poisson(5, (n_cells, n_genes))
    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)
    if log_normalise:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / fname
    adata.write(path)
    return path


# ---------------------------------------------------------------------------
# Task 1+2: AnnData pickle fixes
# ---------------------------------------------------------------------------

class TestAnnDataPickle:
    def test_getattr_guard_raises_attribute_error_before_init(self, tmp_path):
        """__getattr__ must raise AttributeError (not recurse) if _backed is absent."""
        path = _make_h5ad(tmp_path)
        wrapper = object.__new__(AnnData)  # bypasses __init__
        with pytest.raises(AttributeError):
            _ = wrapper.obs  # triggers __getattr__ without _backed set

    def test_roundtrip_preserves_path_and_mode(self, tmp_path):
        """pickle.loads(pickle.dumps(wrapper)) must restore path and mode."""
        path = _make_h5ad(tmp_path)
        wrapper = AnnData(path, mode="r")
        restored = pickle.loads(pickle.dumps(wrapper))
        assert restored.path == wrapper.path
        assert restored._mode == wrapper._mode

    def test_roundtrip_does_not_open_file_eagerly(self, tmp_path):
        """After unpickling the backed handle should be None (lazy-open)."""
        path = _make_h5ad(tmp_path)
        wrapper = AnnData(path, mode="r")
        # Open the handle before pickling to confirm it is cleared on restore.
        _ = wrapper.backed
        restored = pickle.loads(pickle.dumps(wrapper))
        assert restored._backed is None

    def test_roundtrip_reopens_file_on_access(self, tmp_path):
        """After unpickling, accessing .backed should reopen the file."""
        path = _make_h5ad(tmp_path)
        wrapper = AnnData(path, mode="r")
        restored = pickle.loads(pickle.dumps(wrapper))
        # Access should not raise and should return a valid AnnData.
        assert restored.backed is not None
        assert len(restored.backed.obs) > 0
        restored.close()

    def test_no_recursion_error_on_pickle(self, tmp_path):
        """Regression: AnnData pickle must not raise RecursionError."""
        path = _make_h5ad(tmp_path)
        wrapper = AnnData(path)
        try:
            data = pickle.dumps(wrapper)
            pickle.loads(data)
        except RecursionError:
            pytest.fail("RecursionError raised when pickling AnnData")


# ---------------------------------------------------------------------------
# Task 3: RankGenesGroupsResult / DifferentialExpressionResult pickle
# ---------------------------------------------------------------------------

class TestResultPickle:
    def _make_rank_result(self, tmp_path: Path) -> RankGenesGroupsResult:
        """Run a minimal wilcoxon to produce a real RankGenesGroupsResult."""
        path = _make_h5ad(tmp_path)
        return wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )

    def test_rank_genes_groups_result_pickle_roundtrip(self, tmp_path):
        """RankGenesGroupsResult must survive pickle.dumps / pickle.loads."""
        result = self._make_rank_result(tmp_path)
        try:
            data = pickle.dumps(result)
            restored = pickle.loads(data)
        except RecursionError:
            pytest.fail("RecursionError raised when pickling RankGenesGroupsResult")
        except Exception as exc:
            pytest.fail(f"Unexpected error during pickle: {exc}")

        # Core arrays intact
        np.testing.assert_array_equal(restored.genes, result.genes)
        np.testing.assert_array_equal(restored.groups, result.groups)
        np.testing.assert_allclose(restored.pvalues, result.pvalues, equal_nan=True)
        np.testing.assert_allclose(restored.logfoldchanges, result.logfoldchanges, equal_nan=True)
        # result (AnnData handle) should be None after unpickling
        assert restored.result is None
        # _group_cache must be empty dict (not missing)
        assert restored._group_cache == {}

    def test_differential_expression_result_pickle_roundtrip(self, tmp_path):
        """DifferentialExpressionResult must survive pickle round-trip."""
        n_genes = 10
        genes = pd.Index([f"gene{i}" for i in range(n_genes)])
        der = DifferentialExpressionResult(
            genes=genes,
            effect_size=np.random.rand(n_genes),
            statistic=np.random.rand(n_genes),
            pvalue=np.random.rand(n_genes),
            method="wilcoxon",
            perturbation="KO1",
            pvalue_adj=np.random.rand(n_genes),
            result=None,
        )
        restored = pickle.loads(pickle.dumps(der))
        np.testing.assert_array_equal(restored.genes, der.genes)
        np.testing.assert_allclose(restored.effect_size, der.effect_size)
        assert restored.result is None

    def test_rank_result_dict_access_after_pickle(self, tmp_path):
        """Dict-style access on a restored RankGenesGroupsResult must work."""
        result = self._make_rank_result(tmp_path)
        restored = pickle.loads(pickle.dumps(result))
        assert "KO1" in restored
        item = restored["KO1"]
        assert item.perturbation == "KO1"
        assert len(item.pvalue) == len(result.genes)


# ---------------------------------------------------------------------------
# Task 4: Auto-reload (force param)
# ---------------------------------------------------------------------------

class TestWilcoxonAutoReload:
    def test_second_call_reloads_without_rerunning(self, tmp_path):
        """Second call without force=True should load from disk, not rerun."""
        path = _make_h5ad(tmp_path)
        result1 = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        output_path = tmp_path / "data_cx_wilcoxon.h5ad"
        assert output_path.exists()

        mtime_before = output_path.stat().st_mtime
        time.sleep(0.01)  # ensure mtime would differ if file were rewritten

        result2 = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        mtime_after = output_path.stat().st_mtime

        # File must NOT have been rewritten
        assert mtime_after == mtime_before, "Output file was overwritten on second call without force=True"
        # Results should match
        np.testing.assert_allclose(result2.pvalues, result1.pvalues, equal_nan=True)
        np.testing.assert_array_equal(result2.groups, result1.groups)

    def test_verbose_prints_reload_notice(self, tmp_path, capsys):
        """verbose=True should print a reload notice on second call."""
        path = _make_h5ad(tmp_path)
        _run_kwargs = dict(
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        wilcoxon_test(path, **_run_kwargs)
        wilcoxon_test(path, verbose=True, **_run_kwargs)
        captured = capsys.readouterr()
        assert "Loading existing result" in captured.out
        assert "force=True" in captured.out

    def test_force_true_reruns_and_overwrites(self, tmp_path):
        """force=True must overwrite the existing output file."""
        path = _make_h5ad(tmp_path)
        _run_kwargs = dict(
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        wilcoxon_test(path, **_run_kwargs)
        output_path = tmp_path / "data_cx_wilcoxon.h5ad"
        mtime_before = output_path.stat().st_mtime
        time.sleep(0.05)

        wilcoxon_test(path, force=True, **_run_kwargs)
        mtime_after = output_path.stat().st_mtime
        assert mtime_after > mtime_before, "Output file was not overwritten when force=True"


class TestTTestAutoReload:
    def test_second_call_reloads_without_rerunning(self, tmp_path):
        path = _make_h5ad(tmp_path)
        result1 = t_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        output_path = tmp_path / "data_cx_t_test.h5ad"
        assert output_path.exists()

        mtime_before = output_path.stat().st_mtime
        time.sleep(0.01)

        result2 = t_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        mtime_after = output_path.stat().st_mtime

        assert mtime_after == mtime_before, "Output file was overwritten on second call without force=True"
        np.testing.assert_allclose(result2.pvalues, result1.pvalues, equal_nan=True)

    def test_force_true_reruns_and_overwrites(self, tmp_path):
        path = _make_h5ad(tmp_path)
        _run_kwargs = dict(
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        t_test(path, **_run_kwargs)
        output_path = tmp_path / "data_cx_t_test.h5ad"
        mtime_before = output_path.stat().st_mtime
        time.sleep(0.05)

        t_test(path, force=True, **_run_kwargs)
        mtime_after = output_path.stat().st_mtime
        assert mtime_after > mtime_before, "Output file was not overwritten when force=True"

    def test_verbose_prints_reload_notice(self, tmp_path, capsys):
        path = _make_h5ad(tmp_path)
        _run_kwargs = dict(
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
        )
        t_test(path, **_run_kwargs)
        t_test(path, verbose=True, **_run_kwargs)
        captured = capsys.readouterr()
        assert "Loading existing result" in captured.out
        assert "force=True" in captured.out


class TestNbGlmAutoReload:
    def test_second_call_reloads_without_rerunning(self, tmp_path):
        path = _make_h5ad(tmp_path, n_ctrl=60, n_pert=40, n_genes=15)
        result1 = nb_glm_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
            n_jobs=1,
        )
        output_path = tmp_path / "data_cx_nb_glm.h5ad"
        assert output_path.exists()

        mtime_before = output_path.stat().st_mtime
        time.sleep(0.01)

        result2 = nb_glm_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
            n_jobs=1,
        )
        mtime_after = output_path.stat().st_mtime

        assert mtime_after == mtime_before, "Output file was overwritten on second call without force=True"
        np.testing.assert_allclose(result2.pvalues, result1.pvalues, equal_nan=True)

    def test_force_true_reruns_and_overwrites(self, tmp_path):
        path = _make_h5ad(tmp_path, n_ctrl=60, n_pert=40, n_genes=15)
        _run_kwargs = dict(
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
            n_jobs=1,
        )
        nb_glm_test(path, **_run_kwargs)
        output_path = tmp_path / "data_cx_nb_glm.h5ad"
        mtime_before = output_path.stat().st_mtime
        time.sleep(0.05)

        nb_glm_test(path, force=True, **_run_kwargs)
        mtime_after = output_path.stat().st_mtime
        assert mtime_after > mtime_before, "Output file was not overwritten when force=True"

    def test_verbose_prints_reload_notice(self, tmp_path, capsys):
        path = _make_h5ad(tmp_path, n_ctrl=60, n_pert=40, n_genes=15)
        _run_kwargs = dict(
            perturbation_column="perturbation",
            control_label="ctrl",
            output_dir=tmp_path,
            min_pct_both=0.0,
            min_mean_ctrl=0.0,
            n_jobs=1,
        )
        nb_glm_test(path, **_run_kwargs)
        nb_glm_test(path, verbose=True, **_run_kwargs)
        captured = capsys.readouterr()
        assert "Loading existing result" in captured.out
        assert "force=True" in captured.out
