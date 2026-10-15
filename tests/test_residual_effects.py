"""residual_effects: control-fitted linear model, effects as mean residuals.

Every result is checked against a dense in-memory reference: log1p(CP10k),
a one-hot design, least squares on the control cells, residuals averaged per
perturbation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

import crispyx as cx

GENES = [f"g{i}" for i in range(12)] + ["MT-CO1", "MT-ND1"]


def _screen(tmp_path, *, n_controls=300, n_per_target=40, shift=None, seed=0, fmt="csr",
            name="screen", lane_of_a=None):
    """Counts with a gene-specific lane effect, depth variation and three
    targets. ``shift`` maps a target to (gene index, multiplicative effect);
    ``lane_of_a`` puts every cell of target A in that lane."""
    rng = np.random.default_rng(seed)
    labels = ["ctrl"] * n_controls + [t for t in ("A", "B", "C") for _ in range(n_per_target)]
    n = len(labels)
    lane = rng.choice(["L1", "L2", "L3"], size=n)
    if lane_of_a is not None:
        lane[np.array(labels) == "A"] = lane_of_a
    lane_scale = pd.DataFrame(rng.lognormal(0, 0.6, size=(3, len(GENES))), index=["L1", "L2", "L3"])
    depth = rng.lognormal(0, 0.4, size=n)
    base = rng.uniform(0.5, 6, size=len(GENES))
    mean = depth[:, None] * base[None, :] * lane_scale.loc[lane].to_numpy()
    for target, (gene, factor) in (shift or {}).items():
        mean[np.array(labels) == target, gene] *= factor
    counts = rng.poisson(mean).astype(np.float32)
    obs = pd.DataFrame(
        {
            "perturbation": pd.Categorical(labels),
            "lane": pd.Categorical(lane),
            "score": rng.normal(size=n),
        },
        index=[f"cell{i}" for i in range(n)],
    )
    X = sp.csr_matrix(counts) if fmt == "csr" else sp.csc_matrix(counts) if fmt == "csc" else counts
    path = tmp_path / f"{name}.h5ad"
    ad.AnnData(X, obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    return path, counts, obs


def _reference(counts, obs, covariates, *, used=None, null="model", min_control_mean=0.1):
    """Dense reference: effect, z, residual sd, robust sd, per target."""
    used = np.ones(len(obs), bool) if used is None else used
    counts, obs = counts[used].astype(np.float64), obs.iloc[np.flatnonzero(used)]
    totals = counts.sum(axis=1)
    y = np.log1p(counts / totals[:, None] * 1e4)
    parts = [pd.Series(1.0, index=obs.index, name="intercept")]
    for name in covariates:
        if name == "log_total_counts":
            parts.append(pd.Series(np.log(totals), index=obs.index))
        elif name == "mito_fraction":
            mito = np.array([g.startswith("MT-") for g in GENES])
            parts.append(pd.Series(counts[:, mito].sum(axis=1) / totals, index=obs.index))
        elif pd.api.types.is_numeric_dtype(obs[name]):
            parts.append(obs[name].astype(float))
        else:
            parts.append(pd.get_dummies(obs[name].astype(str), drop_first=True, dtype=float))
    design = pd.concat(parts, axis=1).to_numpy()
    control = (obs["perturbation"] == "ctrl").to_numpy()
    coef, *_ = np.linalg.lstsq(design[control], y[control], rcond=None)
    residual = y - design @ coef
    rank = np.linalg.matrix_rank(design[control])
    sd = np.sqrt((residual[control] ** 2).sum(axis=0) / (control.sum() - rank))
    targets = [t for t in pd.unique(obs["perturbation"].astype(str)) if t != "ctrl"]
    n = np.array([(obs["perturbation"] == t).sum() for t in targets], dtype=float)
    effect = np.vstack([residual[(obs["perturbation"] == t).to_numpy()].mean(axis=0) for t in targets])
    scaled = effect * np.sqrt(n)[:, None]
    robust = 1.4826 * np.median(np.abs(scaled - np.median(scaled, axis=0)), axis=0)
    spread = sd if null == "model" else robust
    tested = (y[control].mean(axis=0) > min_control_mean) & (spread > 0)
    z = np.where(tested, scaled / spread, np.nan)
    return targets, effect, z, sd, robust


def _read(result):
    return ad.read_h5ad(result.path)


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
@pytest.mark.parametrize("chunk_size", [7, 1000])
def test_matches_dense_reference(tmp_path, fmt, chunk_size):
    path, counts, obs = _screen(tmp_path, fmt=fmt)
    covariates = ["lane", "score", "log_total_counts", "mito_fraction"]
    out = _read(cx.residual_effects(
        path, "perturbation", "ctrl", covariates=covariates, min_cells=1, chunk_size=chunk_size, verbose=False,
    ))
    targets, effect, z, sd, robust = _reference(counts, obs, covariates)
    assert out.obs_names.tolist() == targets
    np.testing.assert_allclose(out.X, effect, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(out.var["residual_sd"], sd, rtol=1e-8)
    np.testing.assert_allclose(out.var["robust_sd"], robust, rtol=1e-8)
    np.testing.assert_allclose(out.layers["z_score"], z, rtol=1e-7, equal_nan=True)
    p = 2 * stats.norm.sf(np.abs(z))
    np.testing.assert_allclose(out.layers["pvalue"], p, rtol=1e-6, equal_nan=True)
    assert (out.obs["n_cells"] == 40).all()
    assert list(out.uns["design_columns"]) == [
        "intercept", "lane[L2]", "lane[L3]", "score", "log_total_counts", "mito_fraction",
    ]


def test_robust_null(tmp_path):
    path, counts, obs = _screen(tmp_path)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["log_total_counts", "mito_fraction"], null="robust", min_cells=1, verbose=False))
    _, _, z, _, _ = _reference(counts, obs, ["log_total_counts", "mito_fraction"], null="robust")
    np.testing.assert_allclose(out.layers["z_score"], z, rtol=1e-7, equal_nan=True)
    assert out.uns["null"] == "robust"


def test_covariates_remove_composition(tmp_path):
    """Every cell of target A sits in one lane, whose genes differ from the
    pooled controls: A looks perturbed against them, but not once lane is in
    the model."""
    path, counts, obs = _screen(tmp_path, lane_of_a="L2", n_controls=1500, n_per_target=150)
    naive = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=[], min_cells=1,
                                      output_path=tmp_path / "naive.h5ad", verbose=False))
    adjusted = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=1,
                                         output_path=tmp_path / "adjusted.h5ad", verbose=False))
    _, ref_naive, *_ = _reference(counts, obs, [])
    _, ref_adjusted, *_ = _reference(counts, obs, ["lane"])
    np.testing.assert_allclose(naive.X, ref_naive, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(adjusted.X, ref_adjusted, rtol=1e-8, atol=1e-10)
    assert np.nanmax(np.abs(naive["A"].layers["z_score"])) > 6
    assert np.nanmax(np.abs(adjusted["A"].layers["z_score"])) < 3.5


def test_detects_a_real_effect(tmp_path):
    path, *_ = _screen(tmp_path, shift={"B": (3, 0.2)}, n_per_target=80)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "log_total_counts"],
                                    min_cells=1, verbose=False))
    assert out["B"].X[0, 3] < -1.0
    assert out["B"].layers["pvalue_adj"][0, 3] < 1e-6


def test_calibrated_without_effects(tmp_path):
    path, *_ = _screen(tmp_path, n_controls=2000, n_per_target=200, seed=5)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "log_total_counts"],
                                    min_cells=1, verbose=False))
    p = out.layers["pvalue"][np.isfinite(out.layers["pvalue"])]
    assert stats.kstest(p, "uniform").pvalue > 1e-3


def test_missing_labels_are_not_used(tmp_path):
    path, counts, obs = _screen(tmp_path)
    labels = obs["perturbation"].astype(object)
    left_out = np.zeros(len(obs), bool)
    left_out[::5] = True
    labels[left_out] = np.nan
    out = _read(cx.residual_effects(path, pd.Series(labels, index=obs.index, name="guide_target"), "ctrl",
                                    covariates=["lane"], min_cells=1, verbose=False))
    _, effect, *_ = _reference(counts, obs, ["lane"], used=~left_out)
    np.testing.assert_allclose(out.X, effect, rtol=1e-8, atol=1e-10)
    assert "guide_target" in out.obs.columns


def test_series_must_cover_every_cell(tmp_path):
    path, _, obs = _screen(tmp_path)
    with pytest.raises(ValueError, match="no entry"):
        cx.residual_effects(path, obs["perturbation"].iloc[1:], "ctrl", covariates=["lane"], verbose=False)


def test_rerun_reuses_and_relabel_recomputes(tmp_path, capsys):
    path, _, obs = _screen(tmp_path)
    cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=1, verbose=False)
    cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=1, verbose=True)
    assert "Loading existing result" in capsys.readouterr().out
    shuffled = pd.Series(np.random.default_rng(0).permutation(obs["perturbation"].astype(str).to_numpy()),
                         index=obs.index, name="perturbation")
    out = _read(cx.residual_effects(path, shuffled, "ctrl", covariates=["lane"], min_cells=1, verbose=True))
    assert "is not reused" in capsys.readouterr().out
    assert out.uns["crispyx"]["kind"] == "residual_effects"


def test_min_cells_and_perturbations(tmp_path):
    path, *_ = _screen(tmp_path)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], perturbations=["B"], min_cells=1, verbose=False))
    assert out.obs_names.tolist() == ["B"]
    with pytest.raises(ValueError, match="no perturbation has at least"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=41, verbose=False)
    with pytest.raises(ValueError, match="not found"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], perturbations=["Z"], verbose=False)


def test_untested_genes_keep_effects(tmp_path):
    path, *_ = _screen(tmp_path)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=1, min_control_mean=3.0, verbose=False))
    tested = out.var["tested"].to_numpy()
    assert 0 < tested.sum() < len(GENES)
    assert np.isnan(out.layers["z_score"][:, ~tested]).all()
    assert np.isfinite(out.X[:, ~tested]).all()
    assert (out.obs["n_genes_tested"] == tested.sum()).all()


def test_refuses_normalised_input(tmp_path):
    path, counts, obs = _screen(tmp_path)
    ad.AnnData(sp.csr_matrix(np.log1p(counts)), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    with pytest.raises(ValueError, match="raw counts"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], verbose=False)


def test_level_without_controls_raises(tmp_path):
    path, counts, obs = _screen(tmp_path)
    obs["lane"] = obs["lane"].astype(str)
    obs.loc[obs["perturbation"] == "A", "lane"] = "L9"
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    with pytest.raises(ValueError, match="L9"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], verbose=False)


def test_computed_covariate_name_clash_raises(tmp_path):
    path, counts, obs = _screen(tmp_path)
    obs["log_total_counts"] = 1.0
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    with pytest.raises(ValueError, match="computed from the counts"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["log_total_counts"], verbose=False)


def test_mito_fraction_matches_scanpy(tmp_path):
    sc = pytest.importorskip("scanpy")
    path, counts, obs = _screen(tmp_path)
    adata = ad.AnnData(counts.copy(), obs=obs.copy(), var=pd.DataFrame(index=GENES))
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["mito_fraction"], min_cells=1,
                                    verbose=False))
    expected = adata.obs["pct_counts_mt"].to_numpy() / 100
    _, effect, *_ = _reference(counts, obs.assign(mito=expected), ["mito"])
    np.testing.assert_allclose(out.X, effect, rtol=1e-6, atol=1e-8)


def test_round_trip_through_anndata(tmp_path):
    path, *_ = _screen(tmp_path)
    result = cx.tl.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=1, verbose=False)
    out = _read(result)
    assert set(out.layers) - {None} == {"z_score", "pvalue", "pvalue_adj"}  # anndata >= 0.13 lists X as None
    assert {"control_mean", "residual_sd", "robust_sd", "tested"} <= set(out.var.columns)
    assert out.uns["crispyx"]["schema"] == 1


def test_disk_usage_estimate(tmp_path):
    path, *_ = _screen(tmp_path)
    estimate = cx.estimate_disk_usage("residual_effects", path, perturbation_column="perturbation",
                                      control_label="ctrl", min_cells=1)
    assert estimate["output"].required_bytes == pytest.approx(3 * len(GENES) * 4 * 8 * 1.10)


def test_segments_share_only_their_edge_groups():
    from crispyx.residual import _segments

    rng = np.random.default_rng(0)
    for groups in (rng.integers(-1, 30, size=500), np.repeat([0, 1, 2], [50, 400, 50])):
        row_nnz = rng.integers(1, 100, size=groups.size)
        for workers in (1, 3, 8, 64):
            rows, bounds, edge_groups, controls, control_bounds = _segments(groups, row_nnz, workers)
            assert len(bounds) == len(control_bounds) == workers + 1
            assert np.all(np.diff(bounds) >= 0) and np.all(np.diff(control_bounds) >= 0)
            assert sorted(rows) == sorted(np.flatnonzero(groups > 0))
            assert sorted(controls) == sorted(np.flatnonzero(groups == 0))
            owners = [set(groups[rows[bounds[w]:bounds[w + 1]]]) for w in range(workers)]
            for a in range(workers):
                for b in range(a + 1, workers):
                    assert owners[a] & owners[b] <= set(edge_groups[a]) & set(edge_groups[b])


def test_segments_split_a_large_group():
    """A chunk holding one perturbation still gives every thread work."""
    from crispyx.residual import _segments

    groups = np.ones(400, dtype=np.int64)
    _, bounds, edge_groups, _, _ = _segments(groups, np.full(400, 5), 4)
    assert np.all(np.diff(bounds) == 100)
    assert (edge_groups[:, 0] == 1).all() and (edge_groups[:, 1] == -1).all()


@pytest.mark.parametrize("chunk_size", [7, 100, 1000])
def test_sorted_by_perturbation_matches_reference(tmp_path, chunk_size):
    """Groups larger than a chunk, split across threads."""
    path, counts, obs = _screen(tmp_path, n_per_target=150)
    order = np.argsort(obs["perturbation"].astype(str).to_numpy(), kind="stable")
    counts, obs = counts[order], obs.iloc[order]
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "log_total_counts"],
                                    min_cells=1, chunk_size=chunk_size, n_jobs=4, verbose=False))
    _, effect, *_ = _reference(counts, obs, ["lane", "log_total_counts"])
    np.testing.assert_allclose(out.X, effect, rtol=1e-8, atol=1e-10)


def test_collinear_but_identified_runs(tmp_path):
    """A covariate duplicated under another name: rank-deficient, but every
    perturbation's design lies within the controls', so effects are unique."""
    path, counts, obs = _screen(tmp_path)
    obs["lane_copy"] = obs["lane"].astype(str)
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "lane_copy"],
                                    min_cells=1, verbose=False))
    _, effect, *_ = _reference(counts, obs, ["lane"])
    np.testing.assert_allclose(out.X, effect, rtol=1e-8, atol=1e-10)


def test_unidentified_effect_raises(tmp_path):
    """Among the controls donor and lane coincide, so the fit cannot tell
    them apart; target A's cells pair them differently, so its effect would
    be arbitrary."""
    path, counts, obs = _screen(tmp_path)
    obs["lane"] = obs["lane"].astype(str)
    control = (obs["perturbation"] == "ctrl").to_numpy()
    lane = np.where(control, np.where(np.arange(len(obs)) % 2, "L1", "L2"), "L1")
    donor = np.where(control, np.where(lane == "L1", "D1", "D2"), "D1")
    donor[(obs["perturbation"] == "A").to_numpy()] = "D2"
    obs["lane"], obs["donor"] = lane, donor
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    with pytest.raises(ValueError, match=r"arbitrary: \['A'\]"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "donor"], min_cells=1,
                            verbose=False)


def test_refuses_normalised_rows_after_the_first_chunk(tmp_path):
    path, counts, obs = _screen(tmp_path)
    mixed = counts.astype(np.float64)
    mixed[200:] = np.log1p(mixed[200:])
    ad.AnnData(sp.csr_matrix(mixed), obs=obs, var=pd.DataFrame(index=GENES)).write(path)
    with pytest.raises(ValueError, match="raw counts"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], chunk_size=100,
                            verbose=False)


def test_memory_guard_counts_per_thread_arrays(tmp_path):
    """The per-thread cross-products exceed the budget though the
    perturbations x genes arrays alone would fit."""
    path, *_ = _screen(tmp_path)
    budget_gb = 8 * len(GENES) * 3 * 4 * 1.5 / 1e9
    with pytest.raises(MemoryError, match="thread"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "score"], min_cells=1,
                            n_jobs=1, memory_limit_gb=budget_gb, verbose=False)


def test_disk_usage_estimate_includes_scratch_copy(tmp_path):
    path, *_ = _screen(tmp_path, fmt="csc")
    kwargs = dict(perturbation_column="perturbation", control_label="ctrl", min_cells=1)
    converting = cx.estimate_disk_usage("residual_effects", path, format_mismatch_policy="convert", **kwargs)
    assert converting["scratch"].required_bytes > 0
    assert "scratch" not in cx.estimate_disk_usage("residual_effects", path, format_mismatch_policy="off", **kwargs)


def test_format_mismatch_policy_convert(tmp_path):
    path, counts, obs = _screen(tmp_path, fmt="csc")
    out = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"], min_cells=1,
                                    format_mismatch_policy="convert", verbose=False))
    _, effect, *_ = _reference(counts, obs, ["lane"])
    np.testing.assert_allclose(out.X, effect, rtol=1e-8, atol=1e-10)
    assert not list(tmp_path.glob(".cx_*"))
    with pytest.raises(ValueError, match="format_mismatch_policy"):
        cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane"],
                            format_mismatch_policy="never", verbose=False)


def test_results_do_not_depend_on_n_jobs(tmp_path):
    path, *_ = _screen(tmp_path)
    one = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "log_total_counts"],
                                    min_cells=1, n_jobs=1, output_path=tmp_path / "one.h5ad", verbose=False))
    many = _read(cx.residual_effects(path, "perturbation", "ctrl", covariates=["lane", "log_total_counts"],
                                     min_cells=1, n_jobs=4, output_path=tmp_path / "many.h5ad", verbose=False))
    np.testing.assert_allclose(one.X, many.X, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(one.layers["z_score"], many.layers["z_score"], rtol=1e-10, equal_nan=True)


def test_covariates_must_be_given(tmp_path):
    path, *_ = _screen(tmp_path)
    with pytest.raises(TypeError, match="covariates"):
        cx.residual_effects(path, "perturbation", "ctrl", verbose=False)
