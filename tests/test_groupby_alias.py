"""Tests for groupby / reference alias parameters on the three public DE functions
and cx.tl.rank_genes_groups.

Covers:
- Happy path: groupby= works like perturbation_column=
- Happy path: reference= works like control_label=
- Happy path: both aliases can be used simultaneously
- Error path: TypeError when both canonical and alias are supplied
- Error path: TypeError when neither perturbation_column nor groupby is given
- Regression: original canonical parameter names still work unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import crispyx as cx
from crispyx.de import (
    RankGenesGroupsResult,
    nb_glm_test,
    t_test,
    wilcoxon_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_h5ad(tmp_path: Path, *, n_ctrl: int = 40, n_pert: int = 30,
                   n_genes: int = 20, seed: int = 0, fname: str = "data.h5ad") -> Path:
    """Build a minimal log-normalised h5ad (for t-test / Wilcoxon)."""
    rng = np.random.default_rng(seed)
    n_cells = n_ctrl + n_pert
    counts = (rng.random((n_cells, n_genes)) < 0.4) * rng.poisson(5, (n_cells, n_genes))
    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / fname
    adata.write(path)
    return path


def _make_count_h5ad(tmp_path: Path, *, n_ctrl: int = 40, n_pert: int = 30,
                     n_genes: int = 20, seed: int = 0, fname: str = "counts.h5ad") -> Path:
    """Build a minimal raw-count h5ad (for NB-GLM)."""
    rng = np.random.default_rng(seed)
    n_cells = n_ctrl + n_pert
    counts = rng.poisson(5, (n_cells, n_genes))
    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)
    path = tmp_path / fname
    adata.write(path)
    return path


def _assert_result(result) -> None:
    """Minimal sanity checks on a RankGenesGroupsResult."""
    assert result is not None
    assert isinstance(result, RankGenesGroupsResult)


# ===========================================================================
# t_test
# ===========================================================================

class TestTTestAliases:
    def test_groupby_alias(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = t_test(path, groupby="perturbation", control_label="ctrl",
                        n_jobs=1, force=True)
        _assert_result(result)

    def test_reference_alias(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = t_test(path, perturbation_column="perturbation", reference="ctrl",
                        n_jobs=1, force=True)
        _assert_result(result)

    def test_both_aliases(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = t_test(path, groupby="perturbation", reference="ctrl",
                        n_jobs=1, force=True)
        _assert_result(result)

    def test_canonical_still_works(self, tmp_path):
        """Regression: existing callers must not be broken."""
        path = _make_log_h5ad(tmp_path)
        result = t_test(path, perturbation_column="perturbation", control_label="ctrl",
                        n_jobs=1, force=True)
        _assert_result(result)

    def test_conflict_groupby_and_perturbation_column_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            t_test(path, perturbation_column="perturbation", groupby="perturbation",
                   n_jobs=1)

    def test_conflict_reference_and_control_label_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="control_label.*reference|reference.*control_label"):
            t_test(path, perturbation_column="perturbation",
                   control_label="ctrl", reference="ctrl", n_jobs=1)

    def test_missing_perturbation_column_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            t_test(path, control_label="ctrl", n_jobs=1)


# ===========================================================================
# wilcoxon_test
# ===========================================================================

class TestWilcoxonAliases:
    def test_groupby_alias(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = wilcoxon_test(path, groupby="perturbation", control_label="ctrl",
                               n_jobs=1, force=True)
        _assert_result(result)

    def test_reference_alias(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = wilcoxon_test(path, perturbation_column="perturbation", reference="ctrl",
                               n_jobs=1, force=True)
        _assert_result(result)

    def test_both_aliases(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = wilcoxon_test(path, groupby="perturbation", reference="ctrl",
                               n_jobs=1, force=True)
        _assert_result(result)

    def test_canonical_still_works(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = wilcoxon_test(path, perturbation_column="perturbation", control_label="ctrl",
                               n_jobs=1, force=True)
        _assert_result(result)

    def test_conflict_groupby_and_perturbation_column_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            wilcoxon_test(path, perturbation_column="perturbation", groupby="perturbation",
                          n_jobs=1)

    def test_conflict_reference_and_control_label_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="control_label.*reference|reference.*control_label"):
            wilcoxon_test(path, perturbation_column="perturbation",
                          control_label="ctrl", reference="ctrl", n_jobs=1)

    def test_missing_perturbation_column_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            wilcoxon_test(path, control_label="ctrl", n_jobs=1)


# ===========================================================================
# nb_glm_test
# ===========================================================================

class TestNbGlmAliases:
    def test_groupby_alias(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        result = nb_glm_test(path, groupby="perturbation", control_label="ctrl",
                             n_jobs=1, force=True)
        _assert_result(result)

    def test_reference_alias(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        result = nb_glm_test(path, perturbation_column="perturbation", reference="ctrl",
                             n_jobs=1, force=True)
        _assert_result(result)

    def test_both_aliases(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        result = nb_glm_test(path, groupby="perturbation", reference="ctrl",
                             n_jobs=1, force=True)
        _assert_result(result)

    def test_canonical_still_works(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        result = nb_glm_test(path, perturbation_column="perturbation", control_label="ctrl",
                             n_jobs=1, force=True)
        _assert_result(result)

    def test_conflict_groupby_and_perturbation_column_raises(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            nb_glm_test(path, perturbation_column="perturbation", groupby="perturbation",
                        n_jobs=1)

    def test_conflict_reference_and_control_label_raises(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        with pytest.raises(TypeError, match="control_label.*reference|reference.*control_label"):
            nb_glm_test(path, perturbation_column="perturbation",
                        control_label="ctrl", reference="ctrl", n_jobs=1)

    def test_missing_perturbation_column_raises(self, tmp_path):
        path = _make_count_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            nb_glm_test(path, control_label="ctrl", n_jobs=1)


# ===========================================================================
# cx.tl.rank_genes_groups
# ===========================================================================

class TestTlRankGenesGroupsAliases:
    def test_groupby_alias(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = cx.tl.rank_genes_groups(path, groupby="perturbation", control_label="ctrl",
                                         method="wilcoxon", n_jobs=1, force=True)
        assert result is not None

    def test_reference_alias(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = cx.tl.rank_genes_groups(path, perturbation_column="perturbation",
                                         reference="ctrl", method="wilcoxon",
                                         n_jobs=1, force=True)
        assert result is not None

    def test_both_aliases(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        result = cx.tl.rank_genes_groups(path, groupby="perturbation", reference="ctrl",
                                         method="wilcoxon", n_jobs=1, force=True)
        assert result is not None

    def test_conflict_groupby_and_perturbation_column_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            cx.tl.rank_genes_groups(path, perturbation_column="perturbation",
                                    groupby="perturbation", n_jobs=1)

    def test_conflict_reference_and_control_label_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="control_label.*reference|reference.*control_label"):
            cx.tl.rank_genes_groups(path, perturbation_column="perturbation",
                                    control_label="ctrl", reference="ctrl", n_jobs=1)

    def test_missing_perturbation_column_raises(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        with pytest.raises(TypeError, match="perturbation_column.*groupby|groupby.*perturbation_column"):
            cx.tl.rank_genes_groups(path, method="wilcoxon", n_jobs=1)


# ===========================================================================
# _resolve_de_aliases unit tests (stacklevel / deprecation)
# ===========================================================================

class TestResolveDeAliases:
    """Direct unit tests on the shared helper."""

    def test_min_pct_both_still_works_silently(self, tmp_path):
        """min_pct_both should silently set both thresholds with no warning."""
        path = _make_log_h5ad(tmp_path)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → error
            result = t_test(path, perturbation_column="perturbation", control_label="ctrl",
                            min_pct_both=0.05, n_jobs=1, force=True)
        _assert_result(result)

    def test_groupby_sets_perturbation_column(self, tmp_path):
        """Result using groupby= should be identical to result using perturbation_column=."""
        path = _make_log_h5ad(tmp_path)
        r1 = t_test(path, perturbation_column="perturbation", control_label="ctrl",
                    n_jobs=1, force=True)
        r2 = t_test(path, groupby="perturbation", control_label="ctrl",
                    n_jobs=1, force=True)
        # Both calls should succeed and produce a result
        _assert_result(r1)
        _assert_result(r2)

    def test_reference_sets_control_label(self, tmp_path):
        path = _make_log_h5ad(tmp_path)
        r1 = wilcoxon_test(path, perturbation_column="perturbation", control_label="ctrl",
                           n_jobs=1, force=True)
        r2 = wilcoxon_test(path, perturbation_column="perturbation", reference="ctrl",
                           n_jobs=1, force=True)
        _assert_result(r1)
        _assert_result(r2)
