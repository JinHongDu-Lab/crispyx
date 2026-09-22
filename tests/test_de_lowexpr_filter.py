"""Tests for the per-condition low-expression filter applied to DE tests."""

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
from crispyx._statistics import _low_expr_in_both_mask


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------

def test_low_expr_mask_disabled_by_zero_thresholds():
    n_genes = 6
    expr_p = np.array([0, 0, 5, 5, 0, 5])
    expr_c = np.array([0, 0, 0, 5, 5, 5])
    mean_p = np.zeros(n_genes)
    mean_c = np.zeros(n_genes)
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=10,
        n_control_cells=10,
        min_pct_ctrl=0.0,
        min_pct_pert=0.0,
        min_mean_ctrl=0.0,
        min_mean_pert=0.0,
    )
    assert mask.dtype == bool
    assert not mask.any(), "Filter must be a no-op when both thresholds are 0"


def test_low_expr_mask_drops_only_jointly_low_genes():
    # 4 genes, 100 cells per group.
    expr_p = np.array([0, 0,   50, 0])
    expr_c = np.array([0, 50,  0,  0])
    mean_p = np.array([0.0, 0.0, 1.0, 0.0])
    mean_c = np.array([0.0, 1.0, 0.0, 0.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,
    )
    # Gene 0: zero everywhere -> drop.
    # Gene 1: expressed in control -> keep.
    # Gene 2: expressed in pert    -> keep.
    # Gene 3: zero everywhere -> drop.
    assert mask.tolist() == [True, False, False, True]


def test_low_expr_mask_requires_both_metrics_to_drop():
    # Many cells (passes pct) but mean below threshold:
    # should still pass (only one metric fails) -> NOT dropped.
    expr_p = np.array([20, 0])
    expr_c = np.array([20, 0])
    mean_p = np.array([0.001, 0.0])  # below mean threshold
    mean_c = np.array([0.001, 0.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,
    )
    # Gene 0: pct = 0.2 (passes), mean below -> low_p/low_c is False (pct passes)
    #   -> NOT dropped.
    # Gene 1: zero everywhere -> dropped.
    assert mask.tolist() == [False, True]


def test_low_expr_mask_handles_empty_groups():
    expr = np.array([0, 5])
    mean = np.array([0.0, 1.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr,
        control_expr_counts=expr,
        pert_mean=mean,
        control_mean=mean,
        n_pert_cells=0,
        n_control_cells=10,
    )
    assert not mask.any()


# ---------------------------------------------------------------------------
# End-to-end tests on tiny datasets
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path: Path, *, log_normalise: bool):
    """Build a small AnnData where one gene is jointly silent in BOTH groups."""

    rng = np.random.default_rng(0)
    n_ctrl = 80
    n_pert = 80
    n_cells = n_ctrl + n_pert
    n_genes = 5

    # Gene 0: well expressed in both groups
    # Gene 1: differentially expressed (high in pert, low in ctrl)
    # Gene 2: differentially expressed (high in ctrl, low in pert)
    # Gene 3: SILENT in both groups (should be filtered out)
    # Gene 4: low-but-real signal in both groups
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    counts[:n_ctrl, 0] = rng.poisson(20, n_ctrl)
    counts[n_ctrl:, 0] = rng.poisson(20, n_pert)
    counts[:n_ctrl, 1] = rng.poisson(2, n_ctrl)
    counts[n_ctrl:, 1] = rng.poisson(15, n_pert)
    counts[:n_ctrl, 2] = rng.poisson(15, n_ctrl)
    counts[n_ctrl:, 2] = rng.poisson(2, n_pert)
    # gene 3: leave all zeros (silent in both)
    counts[:n_ctrl, 4] = rng.poisson(5, n_ctrl)
    counts[n_ctrl:, 4] = rng.poisson(5, n_pert)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"gene{i}" for i in range(n_genes)]},
        index=[f"gene{i}" for i in range(n_genes)],
    )
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    if log_normalise:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / ("ds_log.h5ad" if log_normalise else "ds_raw.h5ad")
    adata.write(path)
    return path


def test_t_test_excludes_jointly_silent_gene(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    lfc = np.asarray(res.logfoldchanges[pert_idx])
    # Gene 3 is silent in both groups -> NaN in pvalue and logfc
    assert np.isnan(pvals[3]), f"Expected NaN for gene 3, got {pvals[3]}"
    assert np.isnan(lfc[3]), f"Expected NaN logfc for gene 3, got {lfc[3]}"
    # Other genes should have finite p-values
    finite_other = np.isfinite(pvals[[0, 1, 2, 4]])
    assert finite_other.all(), f"Non-silent genes should have finite p-values: {pvals}"


def test_wilcoxon_excludes_jointly_silent_gene(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    assert np.isnan(pvals[3]), f"Expected NaN for gene 3, got {pvals[3]}"
    finite_other = np.isfinite(pvals[[0, 1, 2, 4]])
    assert finite_other.all()


def test_t_test_disabled_filter_recovers_legacy_behaviour(tmp_path):
    """With both thresholds at 0 the filter is inert."""
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.0,
        min_pct_pert=0.0,
        min_mean_ctrl=0.0,
        min_mean_pert=0.0,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    # With filter off, no NaN should appear from low-expression masking.
    # Gene 3 has zero variance in both groups -> SE=0 -> still NaN by the
    # original valid-mask logic (or 1.0 in legacy code). Allow either.
    assert np.isnan(pvals[3]) or pvals[3] == 1.0


def test_nb_glm_excludes_jointly_silent_gene(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=False)  # raw counts
    res = cx.nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,
        n_jobs=1,
        verbose=False,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    assert np.isnan(pvals[3]), f"Expected NaN for gene 3, got {pvals[3]}"


def test_filter_thresholds_change_filtering(tmp_path):
    """A tight threshold drops more genes than a loose one."""
    path = _make_dataset(tmp_path, log_normalise=True)
    loose = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.0,
        min_pct_pert=0.0,
        min_mean_ctrl=0.0,
        min_mean_pert=0.0,
    )
    # Strict: pct<0.99 AND mean<1e6 in both groups effectively drops every
    # gene whose pct is below 99% in both groups.
    # force=True because the output file from the loose run already exists.
    strict = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.99,
        min_pct_pert=0.99,
        min_mean_ctrl=1e6,
        min_mean_pert=1e6,
        force=True,
    )
    pert_idx = strict.groups.index("KO1")
    n_strict_nan = int(np.isnan(strict.pvalues[pert_idx]).sum())
    n_loose_nan = int(np.isnan(loose.pvalues[pert_idx]).sum())
    assert n_strict_nan > n_loose_nan, (
        f"Strict threshold should NaN more genes (loose={n_loose_nan}, strict={n_strict_nan})"
    )


# ---------------------------------------------------------------------------
# Asymmetric filter tests (v0.0.3)
# ---------------------------------------------------------------------------

def test_low_expr_mask_asymmetric_default_retains_pert_expressed_gene():
    """With v0.0.4 defaults (min_pct_pert=0.002, min_mean_pert=0.005), a gene
    expressed in perturbed (pct >= min_pct_pert) is retained even when ctrl is low."""
    # 4 genes, 100 control / 100 perturbed cells.
    # Gene 0: zero everywhere -> drop
    # Gene 1: ctrl sparse+low mean, pert pct=0.03 >= min_pct_pert=0.002 -> KEEP
    # Gene 2: ctrl sparse+low mean, pert pct=0.0 < min_pct_pert and mean=0 < min_mean_pert -> DROP
    # Gene 3: both sides well-expressed -> KEEP
    expr_p = np.array([0,  3,  0, 80])
    expr_c = np.array([0,  0,  0, 80])
    mean_p = np.array([0.0, 0.04, 0.0, 1.5])
    mean_c = np.array([0.0, 0.0,  0.0, 1.5])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_ctrl=0.01,
        min_pct_pert=0.002,
        min_mean_ctrl=0.05,
        min_mean_pert=0.005,
    )
    # Gene 0: pct_p=0 < 0.002, mean_p=0 < 0.005 -> low_p=True; pct_c=0 < 0.01, mean_c=0 < 0.05 -> low_c=True -> DROP
    # Gene 1: pct_p=0.03 >= 0.002 -> low_p=False -> KEEP
    # Gene 2: pct_p=0 < 0.002, mean_p=0 < 0.005 -> low_p=True; low_c=True -> DROP
    # Gene 3: pct_p=0.8 >= 0.002 -> low_p=False -> KEEP
    assert mask.tolist() == [True, False, True, False], f"Got {mask.tolist()}"


def test_low_expr_mask_pert_mean_reenable_reproduces_v002():
    """Explicitly setting min_mean_pert=0.05 (symmetric) gives expected results."""
    expr_p = np.array([0,  3,  0, 80])
    expr_c = np.array([0,  0,  0, 80])
    mean_p = np.array([0.0, 0.04, 0.0, 1.5])
    mean_c = np.array([0.0, 0.0,  0.0, 1.5])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,  # symmetric: same as ctrl
    )
    # Gene 0: pct_p=0 < 0.01, mean_p=0 < 0.05 -> low_p=True; low_c=True -> DROP
    # Gene 1: pct_p=0.03 >= 0.01 -> low_p=False -> KEEP (pct check passes)
    # Gene 2: pct_p=0 < 0.01, mean_p=0 < 0.05 -> low_p=True; low_c=True -> DROP
    # Gene 3: pct_p=0.8 -> KEEP
    assert mask.tolist() == [True, False, True, False]


def _make_dataset_induced(tmp_path: Path):
    """Dataset where gene 5 is induced from zero baseline (control near-zero,
    perturbed has modest expression in ~3% of cells)."""
    rng = np.random.default_rng(42)
    n_ctrl = 200
    n_pert = 50   # unbalanced: fewer perturbed cells
    n_cells = n_ctrl + n_pert
    n_genes = 6

    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    # Gene 0-3: well expressed in both
    for g in range(4):
        counts[:n_ctrl, g] = rng.poisson(10, n_ctrl)
        counts[n_ctrl:, g] = rng.poisson(10, n_pert)
    # Gene 4: silent in both (artifact)
    # Gene 5: induced from near-zero baseline — zero in control, expressed in ~6% of pert
    n_expressing_pert = max(3, int(0.06 * n_pert))
    expressing_cells = rng.choice(np.arange(n_ctrl, n_cells), n_expressing_pert, replace=False)
    counts[expressing_cells, 5] = rng.poisson(5, n_expressing_pert)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "induced.h5ad"
    adata.write(path)
    return path


def test_wilcoxon_asymmetric_retains_induced_gene(tmp_path):
    """Gene induced from near-zero baseline is retained with v0.0.4 defaults."""
    path = _make_dataset_induced(tmp_path)
    res = cx.wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        # v0.0.4 defaults: min_pct_ctrl=0.01, min_pct_pert=0.002, min_mean_pert=0.005
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    # Gene 4 (silent in both) must be NaN
    assert np.isnan(pvals[4]), f"Gene 4 (silent) should be NaN, got {pvals[4]}"
    # Gene 5 (induced) must NOT be NaN with the new asymmetric default
    assert not np.isnan(pvals[5]), (
        f"Gene 5 (induced from zero baseline) should have finite p-value with "
        f"min_mean_pert=0.0 default, got {pvals[5]}"
    )


def test_t_test_asymmetric_retains_induced_gene(tmp_path):
    """Same retention check for t_test."""
    path = _make_dataset_induced(tmp_path)
    res = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        # v0.0.4 defaults: min_pct_ctrl=0.01, min_pct_pert=0.002, min_mean_pert=0.005
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    assert np.isnan(pvals[4]), f"Gene 4 (silent) should be NaN"
    assert not np.isnan(pvals[5]), (
        f"Gene 5 (induced) should be finite with v0.0.4 defaults, got {pvals[5]}"
    )


def test_wilcoxon_symmetric_compat_retains_expressed_gene(tmp_path):
    """With min_pct_ctrl=min_pct_pert=0.01 (symmetric), gene 5 (pct_p=6%)
    is still retained since pct_p >= min_pct_pert (pct check passes → not
    filtered in either version)."""
    path = _make_dataset_induced(tmp_path)
    res = cx.wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,   # explicit symmetric filter
        force=True,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    # Gene 5 (pct_p ~ 6% > 1%) should be retained (non-NaN) even under
    # the symmetric filter because the pct check passes for the pert side.
    assert not np.isnan(pvals[5]), (
        f"Gene 5 (pct_p=6%) should be retained with symmetric filter "
        f"(min_mean_pert=0.05), got {pvals[5]}"
    )


# ---------------------------------------------------------------------------
# New v0.0.4 tests
# ---------------------------------------------------------------------------

def test_low_expr_mask_decoupled_ctrl_pert_thresholds():
    """min_pct_ctrl and min_pct_pert are independent: a gene with pct_p above
    min_pct_pert but ctrl below min_pct_ctrl is NOT filtered."""
    expr_p = np.array([5,  0])   # gene 0: pert expressed; gene 1: both zero
    expr_c = np.array([0,  0])   # gene 0: ctrl silent
    mean_p = np.array([0.5, 0.0])
    mean_c = np.array([0.0, 0.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_ctrl=0.01,
        min_pct_pert=0.002,
        min_mean_ctrl=0.05,
        min_mean_pert=0.005,
    )
    # Gene 0: pct_p=0.05 >= min_pct_pert=0.002 -> low_p=False -> KEEP
    # Gene 1: pct_p=0 < 0.002, mean_p=0 < 0.005 -> low_p=True; low_c=True -> DROP
    assert mask.tolist() == [False, True], f"Got {mask.tolist()}"


def test_low_expr_mask_min_pct_both_deprecation_warning():
    """Passing min_pct_both emits DeprecationWarning and overrides ctrl/pert."""
    expr_p = np.array([0])
    expr_c = np.array([0])
    mean_p = np.array([0.0])
    mean_c = np.array([0.0])
    with pytest.warns(DeprecationWarning, match="min_pct_both is deprecated"):
        _low_expr_in_both_mask(
            pert_expr_counts=expr_p,
            control_expr_counts=expr_c,
            pert_mean=mean_p,
            control_mean=mean_c,
            n_pert_cells=10,
            n_control_cells=10,
            min_pct_both=0.01,
        )


def test_low_expr_mask_default_min_mean_pert_enabled():
    """With the new default min_mean_pert=0.005, a gene with mean_p < 0.005
    and pct_p < 0.002 is filtered (dual condition both fail)."""
    expr_p = np.array([1, 5])    # gene 0: 1 cell (pct=0.001 < 0.002); gene 1: 5 cells
    expr_c = np.array([0, 0])
    mean_p = np.array([0.001, 0.01])  # gene 0 mean < 0.005; gene 1 mean >= 0.005
    mean_c = np.array([0.0, 0.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=1000,
        n_control_cells=1000,
        # Use new defaults explicitly
        min_pct_ctrl=0.01,
        min_pct_pert=0.002,
        min_mean_ctrl=0.05,
        min_mean_pert=0.005,
    )
    # Gene 0: pct_p=0.001 < 0.002 AND mean_p=0.001 < 0.005 -> low_p=True;
    #         pct_c=0 < 0.01 AND mean_c=0 < 0.05 -> low_c=True -> DROP
    # Gene 1: pct_p=0.005 >= 0.002 -> low_p=False -> KEEP
    assert mask.tolist() == [True, False], f"Got {mask.tolist()}"


def test_wilcoxon_filtered_genes_are_nan_not_one(tmp_path):
    """Genes filtered by low-expression should produce NaN p-values, not 1.0."""
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_ctrl=0.01,
        min_pct_pert=0.01,
        min_mean_ctrl=0.05,
        min_mean_pert=0.05,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    # Gene 3 is jointly silent -> must be NaN, NOT 1.0
    assert np.isnan(pvals[3]), f"Expected NaN for filtered gene 3, got {pvals[3]}"
    assert pvals[3] != 1.0, "Filtered gene p-value must be NaN, not 1.0"



# ---------------------------------------------------------------------------
# Estimability filter for the NB-GLM effect (_nonestimable_glm_mask)
# ---------------------------------------------------------------------------

from crispyx._statistics import _nonestimable_glm_mask  # noqa: E402


def test_nonestimable_mask_flags_either_empty_arm():
    flagged = _nonestimable_glm_mask(
        pert_expr_counts=np.array([0, 5, 3, 0, 1]),
        control_expr_counts=np.array([9, 0, 4, 0, 1]),
    )
    np.testing.assert_array_equal(flagged, [True, True, False, True, False])


def test_nonestimable_mask_thresholds_are_independent():
    """CRISPRi wants a demanding control side, CRISPRa a demanding perturbed
    side, so the two thresholds must be settable separately."""
    pert = np.array([1, 4, 9])
    control = np.array([9, 9, 9])

    crispri = _nonestimable_glm_mask(
        pert_expr_counts=pert, control_expr_counts=control,
        min_cells_ctrl=5, min_cells_pert=1,
    )
    np.testing.assert_array_equal(crispri, [False, False, False])

    crispra = _nonestimable_glm_mask(
        pert_expr_counts=pert, control_expr_counts=control,
        min_cells_ctrl=1, min_cells_pert=5,
    )
    np.testing.assert_array_equal(crispra, [True, True, False])


def test_nonestimable_mask_control_side_alone():
    flagged = _nonestimable_glm_mask(
        pert_expr_counts=np.array([9, 9]),
        control_expr_counts=np.array([2, 9]),
        min_cells_ctrl=5, min_cells_pert=0,
    )
    np.testing.assert_array_equal(flagged, [True, False])


def test_nonestimable_mask_can_be_disabled():
    flagged = _nonestimable_glm_mask(
        pert_expr_counts=np.array([0, 0]),
        control_expr_counts=np.array([0, 9]),
        min_cells_ctrl=0, min_cells_pert=0,
    )
    np.testing.assert_array_equal(flagged, [False, False])


def _separation_adata(tmp_path):
    """Genes spanning estimable, non-estimable and empty."""
    rng = np.random.default_rng(0)
    n, n_genes = 200, 6
    perturbed = np.zeros(n, dtype=bool)
    perturbed[: n // 2] = True
    counts = rng.negative_binomial(5, 5 / (5 + 30.0), size=(n, n_genes)).astype(np.float32)
    counts[perturbed, 1] = 0          # silenced by the perturbation
    counts[~perturbed, 2] = 0         # induced by the perturbation
    counts[:, 3] = 0                  # no counts anywhere
    counts[perturbed, 4] = 0
    counts[np.flatnonzero(perturbed)[0], 4] = 1   # one count: MLE exists
    names = ["normal", "silenced", "induced", "empty", "one_count", "sparse_one_sided"]
    counts[:, 5] = 0
    counts[np.flatnonzero(perturbed)[:3], 5] = 1  # counts in one arm only

    obs = pd.DataFrame(
        {"perturbation": np.where(perturbed, "g1", "control")},
        index=[f"c{i}" for i in range(n)],
    )
    path = tmp_path / "separation.h5ad"
    ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=names)).write_h5ad(path)
    return path, names


def _by_gene(result, names):
    genes = np.asarray(result.genes).ravel()
    index = {g: i for i, g in enumerate(genes)}
    return (
        index,
        np.asarray(result.logfoldchanges).ravel(),
        np.asarray(result.statistics).ravel(),
        np.asarray(result.pvalues).ravel(),
        np.asarray(result.pts).ravel(),
        np.asarray(result.pts_rest).ravel(),
    )


def test_nb_glm_does_not_estimate_effects_for_one_sided_pairs(tmp_path):
    """A pair absent from one arm gets no effect, statistic or p-value.

    The log-link GLM effect is a difference of log means, so with one arm at
    zero it has no finite maximiser and whatever the fitter returns is set by
    where it stopped.  Such pairs are reported as untested, exactly as genes
    with no counts anywhere already are.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)

    for gene in ("silenced", "induced", "empty", "sparse_one_sided"):
        i = index.get(gene)
        if i is None:
            continue  # dropped upstream, which is also "not reported"
        assert np.isnan(lfc[i]), f"{gene} should have no effect estimate"
        assert np.isnan(stat[i]), f"{gene} should have no statistic"
        assert np.isnan(pval[i]), f"{gene} should have no p-value"

    # Genes whose effect is identified are still tested.
    for gene in ("normal", "one_count"):
        i = index[gene]
        assert np.isfinite(lfc[i]), f"{gene} should still be tested"
        assert np.isfinite(stat[i])


def test_excluded_pairs_keep_their_expression_evidence(tmp_path):
    """Untested is not the same as unremarkable: pts must still show it."""
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)

    silenced = index["silenced"]
    assert np.isnan(lfc[silenced])
    assert pts[silenced] == pytest.approx(0.0)
    assert pts_rest[silenced] > 0.9, "the control arm's expression must remain visible"

    induced = index["induced"]
    assert pts[induced] > 0.9
    assert pts_rest[induced] == pytest.approx(0.0)


def test_excluded_pairs_keep_their_expression_evidence_without_the_control_cache(tmp_path):
    """The same, on the path that does not use the cached control statistics.

    That path built ``pts`` and ``pts_rest`` behind the validity mask, so the
    evidence the documentation points at for an excluded pair -- 0% of
    perturbed cells against 95% of control cells -- was zeroed away on
    precisely the pairs it was there for.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, use_control_cache=False, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)

    silenced = index["silenced"]
    assert np.isnan(pval[silenced]), "the pair is still untested"
    assert pts[silenced] == pytest.approx(0.0)
    assert pts_rest[silenced] > 0.9, "the control arm's expression must remain visible"

    induced = index["induced"]
    assert pts[induced] > 0.9
    assert pts_rest[induced] == pytest.approx(0.0)


def test_effect_is_never_reported_as_zero_for_an_excluded_pair(tmp_path):
    """Reporting 0 would describe a silenced gene as unchanged."""
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)
    silenced = index["silenced"]
    assert not (lfc[silenced] == 0.0), "an excluded effect must be NaN, not zero"


@pytest.mark.parametrize("use_control_cache", [True, False], ids=["cached", "uncached"])
def test_shrinkage_does_not_resurrect_an_excluded_pair(tmp_path, use_control_cache):
    """apeGLM shrinks the pairs that were fitted; it does not fit the rest.

    An excluded pair is dropped before any fit, so it has no MLE to shrink.
    The cached worker handed the shrinkage its zero-initialised coefficients
    anyway, which came back as a finite effect beside a NaN p-value -- and on
    both workers the standard error of an untested gene was replaced by the
    literal 1.0 the per-gene fallback returns.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, data_name=f"apeglm_{use_control_cache}",
        lfc_shrinkage_type="apeglm", use_control_cache=use_control_cache,
        verbose=False,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)
    se = np.asarray(ad.read_h5ad(res.result_path).layers["standard_error"]).ravel()

    for gene in ("silenced", "induced", "empty", "sparse_one_sided"):
        i = index.get(gene)
        if i is None:
            continue  # dropped upstream, which is also "not reported"
        assert np.isnan(pval[i]), f"{gene} should have no p-value"
        assert np.isnan(lfc[i]), f"{gene} should have no effect estimate under shrinkage"
        assert np.isnan(se[i]), f"{gene} should have no standard error, not {se[i]}"

    normal = index["normal"]
    assert np.isfinite(lfc[normal]), "shrinkage must still report the tested genes"
    assert np.isfinite(se[normal]) and se[normal] > 0


def test_min_cells_per_arm_zero_restores_the_previous_behaviour(tmp_path):
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    res = nb_glm_test(
        path, perturbation_column="perturbation", control_label="control",
        output_dir=tmp_path, data_name="disabled", verbose=False,
        min_cells_ctrl=0, min_cells_pert=0,
    )
    index, lfc, stat, pval, pts, pts_rest = _by_gene(res, names)
    silenced = index["silenced"]
    assert np.isfinite(lfc[silenced]), "the filter should be switchable off"
    assert np.isfinite(stat[silenced])


def test_asymmetric_thresholds_reach_nb_glm_test(tmp_path):
    """A demanding control threshold drops pairs a symmetric one would keep.

    The "one_count" gene has a single expressing perturbed cell and a fully
    expressed control arm: estimable, and tested by default.  Requiring more
    perturbed cells -- what an activation screen would want -- excludes it,
    while requiring more control cells does not.
    """
    path, names = _separation_adata(tmp_path)
    from crispyx.de import nb_glm_test

    def effect(**kwargs):
        res = nb_glm_test(
            path, perturbation_column="perturbation", control_label="control",
            output_dir=tmp_path, verbose=False, **kwargs,
        )
        index, lfc, *_ = _by_gene(res, names)
        return lfc[index["one_count"]]

    assert np.isfinite(effect(data_name="sym")), "estimable by default"
    assert np.isnan(effect(data_name="strict_pert", min_cells_pert=5))
    assert np.isfinite(effect(data_name="strict_ctrl", min_cells_ctrl=5))


# ---------------------------------------------------------------------------
# One meaning for "untested": NaN in every column derived from the comparison
# ---------------------------------------------------------------------------

def _make_dataset_partly_tested(tmp_path: Path):
    """A gene that ``min_cells_expressed`` excludes for KO1 but not for KO2.

    Two of the 80 control cells express it, so the control side is not "low"
    and the joint low-expression filter leaves it alone; only the total
    expressing-cell count separates the two perturbations.
    """
    rng = np.random.default_rng(3)
    n = 80
    counts = np.zeros((3 * n, 3), dtype=np.float64)
    # gene 0: expressed everywhere
    counts[:, 0] = rng.poisson(20, 3 * n)
    # gene 1: 2 control cells, none in KO1, 30 in KO2
    counts[[0, 1], 1] = 8.0
    counts[2 * n : 2 * n + 30, 1] = rng.poisson(10, 30) + 1
    # gene 2: expressed everywhere
    counts[:, 2] = rng.poisson(6, 3 * n)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n + ["KO1"] * n + ["KO2"] * n},
        index=[f"cell_{i}" for i in range(3 * n)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(3)])
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / "partly_tested.h5ad"
    adata.write(path)
    return path


_UNTESTED_KW = dict(
    perturbation_column="perturbation",
    control_label="ctrl",
    min_cells_expressed=10,
)


def _assert_untested_is_uniform(res, row):
    """No column derived from the comparison may disagree about testedness."""
    pvals = np.asarray(res.pvalues[row])
    lfc = np.asarray(res.logfoldchanges[row])
    stat = np.asarray(res.statistics[row])
    eff = np.asarray(res.effect_size[row])
    untested = np.isnan(pvals)
    assert untested.any(), "fixture should exclude at least one gene"
    for name, col in (("logfoldchanges", lfc), ("statistics", stat), ("effect_size", eff)):
        assert np.isnan(col[untested]).all(), f"{name} is finite where the p-value is NaN"
        assert np.isfinite(col[~untested]).all(), f"{name} is NaN where the p-value is not"
    # Nothing is spelled as "tested, no change".
    assert not (pvals[untested] == 1.0).any()
    assert not (lfc[untested] == 0.0).any()
    # Expression fractions describe the data and survive exclusion.
    assert np.isfinite(res.pts[row]).all()
    assert np.isfinite(res.pts_rest[row]).all()
    assert res.pts_rest[row][untested].max() > 0


@pytest.mark.parametrize("fn", ["wilcoxon_test", "t_test"])
def test_untested_genes_are_nan_in_every_derived_column(tmp_path, fn):
    path = _make_dataset_partly_tested(tmp_path)
    res = getattr(cx, fn)(path, output_dir=tmp_path, data_name=fn, **_UNTESTED_KW)
    ko1 = res.groups.index("KO1")
    assert np.isnan(res.pvalues[ko1][1]), "gene1 should be excluded for KO1"
    _assert_untested_is_uniform(res, ko1)
    # The same gene is tested for KO2, which has enough expressing cells.
    ko2 = res.groups.index("KO2")
    assert np.isfinite(res.pvalues[ko2][1])
    assert np.isfinite(res.logfoldchanges[ko2][1])


@pytest.mark.parametrize("fn", ["wilcoxon_test", "t_test"])
def test_a_row_does_not_depend_on_the_other_perturbations_in_the_run(tmp_path, fn):
    """A gene excluded here must not inherit a p-value from another row's block."""
    path = _make_dataset_partly_tested(tmp_path)
    both = getattr(cx, fn)(path, output_dir=tmp_path, data_name=f"{fn}_both", **_UNTESTED_KW)
    alone = getattr(cx, fn)(
        path, output_dir=tmp_path, data_name=f"{fn}_alone",
        perturbations=["KO1"], **_UNTESTED_KW,
    )
    row_both = both.groups.index("KO1")
    row_alone = alone.groups.index("KO1")
    for attr in ("pvalues", "statistics", "logfoldchanges", "effect_size"):
        np.testing.assert_allclose(
            np.asarray(getattr(both, attr)[row_both]),
            np.asarray(getattr(alone, attr)[row_alone]),
            atol=1e-10,
            err_msg=f"{attr} for KO1 changed with the run's composition",
        )


def test_wilcoxon_paths_agree_on_untested_genes(tmp_path):
    path = _make_dataset_partly_tested(tmp_path)
    standard = cx.wilcoxon_test(
        path, output_dir=tmp_path, data_name="std", memory_limit_gb=128, **_UNTESTED_KW
    )
    streaming = cx.wilcoxon_test(
        path, output_dir=tmp_path, data_name="stream", memory_limit_gb=1e-7, **_UNTESTED_KW
    )
    assert np.isnan(standard.pvalues).any()
    for attr in ("pvalues", "statistics", "logfoldchanges", "effect_size", "pts", "pts_rest"):
        np.testing.assert_allclose(
            np.asarray(getattr(standard, attr)),
            np.asarray(getattr(streaming, attr)),
            atol=1e-10,
            err_msg=f"{attr} differs between the standard and streaming paths",
        )
    _assert_untested_is_uniform(standard, standard.groups.index("KO1"))
    _assert_untested_is_uniform(streaming, streaming.groups.index("KO1"))
