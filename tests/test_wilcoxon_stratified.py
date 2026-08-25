"""Tests for the batch-stratified (van Elteren) Wilcoxon rank-sum test.

Covers:
- Correctness of the stratified path against an independent per-stratum
  reference implementation (rank-within-batch, unit-weight combination).
- Equivalence with the pooled Wilcoxon test when there is a single batch.
- Batch-effect correction: a confounded batch structure that makes the pooled
  test spuriously significant is corrected by the stratified test.
"""
from __future__ import annotations

import math
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
import pytest
import scipy.sparse as sp
import h5py
from scipy.stats import rankdata
import anndata as ad
import scanpy as sc

from crispyx.de import wilcoxon_test


# ---------------------------------------------------------------------------
# Reference implementation (independent, slow, per-gene)
# ---------------------------------------------------------------------------

def _van_elteren_reference(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    control_label: str,
    *,
    tie_correct: bool = True,
):
    """Rank-within-batch Wilcoxon combined with unit weights (van Elteren).

    Returns dicts keyed by perturbation label -> per-gene arrays of
    ``(z, pvalue, effect_size, u_statistic)``.
    """
    n_genes = X.shape[1]
    groups = [g for g in pd.Index(labels).unique().tolist() if g != control_label]
    ctrl_mask = labels == control_label
    batches = np.unique(batch)

    z_out, p_out, eff_out, u_out = {}, {}, {}, {}
    for group in groups:
        pmask = labels == group
        z_g = np.zeros(n_genes)
        p_g = np.ones(n_genes)
        eff_g = np.zeros(n_genes)
        u_g = np.zeros(n_genes)
        for g in range(n_genes):
            num = 0.0
            var = 0.0
            u_sum = 0.0
            n1n0 = 0.0
            for b in batches:
                bm = batch == b
                x = X[pmask & bm, g].astype(np.float64)
                y = X[ctrl_mask & bm, g].astype(np.float64)
                n1, n0 = len(x), len(y)
                if n1 == 0 or n0 == 0:
                    continue
                N = n1 + n0
                comb = np.concatenate([x, y])
                ranks = rankdata(comb)
                rank_sum = ranks[:n1].sum()
                expected = n1 * (N + 1) / 2.0
                if tie_correct:
                    _, counts = np.unique(comb, return_counts=True)
                    tie_sum = float(np.sum(counts ** 3 - counts))
                else:
                    tie_sum = 0.0
                var_b = (
                    n1 * n0 / 12.0 * ((N + 1) - tie_sum / (N * (N - 1)))
                    if N > 1
                    else 0.0
                )
                u_b = rank_sum - n1 * (n1 + 1) / 2.0
                num += rank_sum - expected
                var += var_b
                u_sum += u_b
                n1n0 += n1 * n0
            if var > 0:
                z = num / math.sqrt(var)
                pval = math.erfc(abs(z) / math.sqrt(2.0))
            else:
                z, pval = 0.0, 1.0
            z_g[g] = z
            p_g[g] = pval
            eff_g[g] = (u_sum / n1n0 - 0.5) if n1n0 > 0 else 0.0
            u_g[g] = u_sum
        z_out[group] = z_g
        p_out[group] = p_g
        eff_out[group] = eff_g
        u_out[group] = u_g
    return z_out, p_out, eff_out, u_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h5ad(
    tmp_path: Path,
    counts: np.ndarray,
    labels: list[str],
    batch: list[int],
) -> tuple[Path, np.ndarray]:
    """Write a log-normalised h5ad and return (path, dense log-normalised X)."""
    n_genes = counts.shape[1]
    obs = pd.DataFrame(
        {"perturbation": labels, "batch": batch},
        index=[f"c{i}" for i in range(counts.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.h5ad"
    adata.write(path)
    dense = np.asarray(adata.X.todense())
    return path, dense


_NO_FILTER = dict(
    min_cells_expressed=0,
    min_pct_both=0.0,
    min_mean_ctrl=0.0,
    min_mean_pert=0.0,
)


# ---------------------------------------------------------------------------
# 1. Correctness against reference
# ---------------------------------------------------------------------------

class TestStratifiedCorrectness:
    def test_matches_reference(self, tmp_path):
        rng = np.random.default_rng(0)
        n_ctrl, n_pert, cells = 120, 3, 60
        n_genes = 25
        n_cells = n_ctrl + n_pert * cells
        counts = (rng.random((n_cells, n_genes)) < 0.4) * rng.integers(
            1, 15, (n_cells, n_genes)
        )
        labels = ["control"] * n_ctrl + [
            f"pert_{i}" for i in range(n_pert) for _ in range(cells)
        ]
        # 4 batches assigned pseudo-randomly to every cell
        batch = rng.integers(0, 4, n_cells).tolist()

        path, X = _make_h5ad(tmp_path, counts, labels, batch)
        labels_arr = np.array(labels)
        batch_arr = np.array(batch)

        ref_z, ref_p, ref_eff, ref_u = _van_elteren_reference(
            X, labels_arr, batch_arr, "control", tie_correct=True
        )

        result = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            batch_column="batch",
            tie_correct=True,
            **_NO_FILTER,
        )

        gene_order = list(result.genes)
        for i, group in enumerate(result.groups):
            col = [gene_order.index(f"gene{j}") for j in range(n_genes)]
            got_z = result.statistics[i][col]
            got_p = result.pvalues[i][col]
            got_eff = result.effect_size[i][col]
            got_u = result.u_statistics[i][col]
            np.testing.assert_allclose(got_z, ref_z[group], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(got_p, ref_p[group], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(got_eff, ref_eff[group], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(got_u, ref_u[group], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 2. Single-batch equivalence with the pooled test
# ---------------------------------------------------------------------------

class TestSingleBatchEquivalence:
    def test_single_batch_equals_pooled(self, tmp_path):
        rng = np.random.default_rng(1)
        n_ctrl, n_pert, cells = 100, 2, 50
        n_genes = 20
        n_cells = n_ctrl + n_pert * cells
        counts = (rng.random((n_cells, n_genes)) < 0.4) * rng.integers(
            1, 12, (n_cells, n_genes)
        )
        labels = ["control"] * n_ctrl + [
            f"pert_{i}" for i in range(n_pert) for _ in range(cells)
        ]
        batch = [0] * n_cells  # single batch

        path, _ = _make_h5ad(tmp_path, counts, labels, batch)

        pooled = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            data_name="pooled",
            **_NO_FILTER,
        )
        strat = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            batch_column="batch",
            data_name="strat",
            **_NO_FILTER,
        )
        np.testing.assert_allclose(
            strat.statistics, pooled.statistics, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            strat.effect_size, pooled.effect_size, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            strat.pvalues, pooled.pvalues, rtol=1e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 3. Batch-effect correction
# ---------------------------------------------------------------------------

class TestBatchConfounding:
    def test_corrects_confounded_batch_effect(self, tmp_path):
        """A gene with no real perturbation effect but a strong batch effect
        that is confounded with group membership should look significant to the
        pooled test and non-significant to the stratified test."""
        rng = np.random.default_rng(2)
        n_genes = 8  # gene 0 is confounded; genes 1..7 are stable background
        # Batch 0: low baseline for gene 0; batch 1: high baseline for gene 0.
        # Control cells mostly in batch 0; perturbation cells mostly in batch 1,
        # so the pooled comparison is dominated by the batch shift, not the pert.
        def sample(n, gene0_base):
            mat = rng.poisson(6, (n, n_genes)).astype(np.int64)  # stable background
            mat[:, 0] = rng.poisson(gene0_base, n)               # confounded gene
            return mat

        ctrl_b0 = sample(180, 1)
        ctrl_b1 = sample(20, 60)
        pert_b0 = sample(20, 1)
        pert_b1 = sample(180, 60)

        counts = np.vstack([ctrl_b0, ctrl_b1, pert_b0, pert_b1])
        labels = (
            ["control"] * 200 + ["pert_0"] * 200
        )
        batch = [0] * 180 + [1] * 20 + [0] * 20 + [1] * 180

        path, _ = _make_h5ad(tmp_path, counts, labels, batch)

        pooled = wilcoxon_test(
            path, perturbation_column="perturbation", control_label="control",
            data_name="pooled", **_NO_FILTER,
        )
        strat = wilcoxon_test(
            path, perturbation_column="perturbation", control_label="control",
            batch_column="batch", data_name="strat", **_NO_FILTER,
        )
        pooled_p = float(pooled["pert_0"].pvalue[0])
        strat_p = float(strat["pert_0"].pvalue[0])

        # Pooled test is fooled by the confounded batch effect.
        assert pooled_p < 1e-3
        # Stratified test removes it: within batch there is no real effect.
        assert strat_p > 0.05
        assert strat_p > pooled_p


# ---------------------------------------------------------------------------
# 4. Cache / output-path safety
# ---------------------------------------------------------------------------

class TestStratifiedCacheSafety:
    def test_default_output_path_does_not_collide_with_pooled(self, tmp_path):
        rng = np.random.default_rng(3)
        n_genes = 6
        ctrl_b0 = rng.poisson(3, (40, n_genes))
        ctrl_b1 = rng.poisson(30, (8, n_genes))
        pert_b0 = rng.poisson(3, (8, n_genes))
        pert_b1 = rng.poisson(30, (40, n_genes))
        counts = np.vstack([ctrl_b0, ctrl_b1, pert_b0, pert_b1])
        labels = ["control"] * 48 + ["pert_0"] * 48
        batch = [0] * 40 + [1] * 8 + [0] * 8 + [1] * 40
        path, _ = _make_h5ad(tmp_path, counts, labels, batch)

        pooled = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            force=True,
            **_NO_FILTER,
        )
        strat = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            batch_column="batch",
            **_NO_FILTER,
        )

        assert pooled.result_path.name == "data_cx_wilcoxon.h5ad"
        assert strat.result_path.name == "data_cx_wilcoxon_stratified.h5ad"
        assert not np.allclose(pooled.statistics, strat.statistics, equal_nan=True)
        with h5py.File(strat.result_path, "r") as hf:
            assert hf["uns"].attrs["batch_column"] == "batch"
            assert bool(hf["uns"].attrs["stratified"]) is True

    def test_explicit_output_path_metadata_mismatch_reruns(self, tmp_path):
        rng = np.random.default_rng(4)
        n_genes = 5
        ctrl_b0 = rng.poisson(4, (36, n_genes))
        ctrl_b1 = rng.poisson(25, (6, n_genes))
        pert_b0 = rng.poisson(4, (6, n_genes))
        pert_b1 = rng.poisson(25, (36, n_genes))
        counts = np.vstack([ctrl_b0, ctrl_b1, pert_b0, pert_b1])
        labels = ["control"] * 42 + ["pert_0"] * 42
        batch = [0] * 36 + [1] * 6 + [0] * 6 + [1] * 36
        path, _ = _make_h5ad(tmp_path, counts, labels, batch)
        out = tmp_path / "shared_output.h5ad"

        pooled = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            output_path=out,
            force=True,
            **_NO_FILTER,
        )
        strat = wilcoxon_test(
            path,
            perturbation_column="perturbation",
            control_label="control",
            batch_column="batch",
            output_path=out,
            **_NO_FILTER,
        )

        assert strat.result_path == pooled.result_path == out
        assert not np.allclose(pooled.statistics, strat.statistics, equal_nan=True)
        with h5py.File(out, "r") as hf:
            assert hf["uns"].attrs["batch_column"] == "batch"
            assert bool(hf["uns"].attrs["stratified"]) is True


# ---------------------------------------------------------------------------
# 5. Untestable batch layouts
# ---------------------------------------------------------------------------

class TestUntestableBatchLayouts:
    def test_no_shared_control_batch_yields_nan_rank_statistics(self, tmp_path):
        rng = np.random.default_rng(5)
        counts = rng.poisson(5, (60, 4))
        labels = ["control"] * 30 + ["pert_0"] * 30
        batch = [0] * 30 + [1] * 30
        path, _ = _make_h5ad(tmp_path, counts, labels, batch)

        with pytest.warns(UserWarning, match="no batches containing both"):
            result = wilcoxon_test(
                path,
                perturbation_column="perturbation",
                control_label="control",
                batch_column="batch",
                **_NO_FILTER,
            )

        assert np.isnan(result.statistics[0]).all()
        assert np.isnan(result.pvalues[0]).all()
        assert np.isnan(result.pvalues_adj[0]).all()
        assert np.isnan(result.effect_size[0]).all()
        assert np.isnan(result.u_statistics[0]).all()
        assert np.isfinite(result.logfoldchanges[0]).any()
        with h5py.File(result.result_path, "r") as hf:
            assert hf["uns"].attrs["stratified_n_untestable_perturbations"] == 1


# ---------------------------------------------------------------------------
# 6. Interrupted checkpoint resume guard
# ---------------------------------------------------------------------------

class TestWilcoxonResumeGuard:
    def test_interrupted_wilcoxon_checkpoint_is_rejected(self, tmp_path):
        rng = np.random.default_rng(6)
        counts = rng.poisson(5, (80, 6))
        labels = ["control"] * 40 + ["pert_0"] * 40
        batch = [0, 1] * 40
        path, _ = _make_h5ad(tmp_path, counts, labels, batch)
        out = tmp_path / "interrupted_stratified.h5ad"
        out.with_suffix(".progress.json").write_text('{"last_gene_chunk": 0}')

        with pytest.raises(NotImplementedError, match="Resuming interrupted wilcoxon_test"):
            wilcoxon_test(
                path,
                perturbation_column="perturbation",
                control_label="control",
                batch_column="batch",
                output_path=out,
                resume=True,
                **_NO_FILTER,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
