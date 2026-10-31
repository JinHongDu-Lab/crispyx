"""tl.nb_offset_test: each perturbation on its own cells, the controls as a
fixed per-stratum baseline."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx

NO_FILTERS = dict(min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0, min_total_count=0)


def _screen(path: Path, *, n_ctrl=3000, n_strata=6, n_pert=10, cells_per=30, n_genes=60, alpha=0.3, seed=1,
            zero_stratum_gene=None):
    """Controls spread over strata with stratum effects, small perturbations
    with planted effects on the first 20 genes; returns the true betas and
    the per-gene base means."""
    rng = np.random.default_rng(seed)
    strata = np.concatenate([rng.integers(0, n_strata, n_ctrl), rng.integers(0, n_strata, n_pert * cells_per)])
    labels = np.array(["control"] * n_ctrl + [f"p{k}" for k in range(n_pert) for _ in range(cells_per)])
    n = labels.size
    depth = np.exp(rng.normal(0, 0.4, n))
    base = np.exp(rng.uniform(np.log(0.05), np.log(20), n_genes))
    stratum_effect = rng.normal(0, 0.5, (n_strata, n_genes))
    beta = np.zeros((n_pert, n_genes))
    planted = min(20, n_genes)
    beta[:, :planted] = rng.normal(0, 1.0, (n_pert, planted))
    pert = np.full(n, -1)
    pert[n_ctrl:] = np.repeat(np.arange(n_pert), cells_per)
    effect = np.where(pert[:, None] >= 0, beta[np.maximum(pert, 0)], 0.0)
    mu = depth[:, None] * base[None, :] * np.exp(stratum_effect[strata] + effect)
    counts = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(np.float32)
    if zero_stratum_gene is not None:
        stratum, gene = zero_stratum_gene
        counts[(strata == stratum) & (labels == "control"), gene] = 0
        counts[(strata == stratum) & (labels == "p0"), gene] = 5
    obs = pd.DataFrame({"perturbation": labels, "stratum": [f"s{b}" for b in strata]},
                       index=[f"c{i}" for i in range(n)])
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])).write(path)
    return beta, base


def _run(path, out, **kwargs):
    return ad.read_h5ad(cx.tl.nb_offset_test(
        path, perturbation_column="perturbation", control_label="control", output_path=out, verbose=False,
        force=True, **kwargs,
    ).result_path)


def test_agrees_with_the_joint_cell_glm_with_a_stratum_covariate(tmp_path):
    beta, base = _screen(tmp_path / "s.h5ad")
    offset = _run(tmp_path / "s.h5ad", tmp_path / "off.h5ad", strata="stratum", lfc_base="ln", **NO_FILTERS)
    joint = ad.read_h5ad(cx.nb_glm_test(
        tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control", covariates=["stratum"],
        dispersion=0.3, lfc_base="ln", output_path=tmp_path / "glm.h5ad", verbose=False, **NO_FILTERS,
    ).result_path)
    well = (base >= 0.5)[None, :] & np.isfinite(offset.X) & np.isfinite(joint.X)
    assert np.median(np.abs(offset.X - joint.X)[well]) < 0.05
    ratio = (offset.layers["standard_error"] / joint.layers["standard_error"])[well]
    assert 0.9 < np.median(ratio) < 1.1
    # the planted effects are recovered
    planted = well[:, :20]
    assert np.corrcoef(offset.X[:, :20][planted], beta[:, :20][planted])[0, 1] > 0.9
    # the control-only dispersion is near the truth on well-expressed genes
    assert 0.2 < np.median(offset.layers["dispersion"][0][base >= 0.5]) < 0.45
    assert offset.uns["method"] == "nb_offset" and offset.uns["n_strata"] == 6
    assert offset.layers["converged"][np.isfinite(offset.X)].all()
    assert (offset.obs["n_genes_tested"] > 50).all()


def test_one_stratum_matches_the_two_group_fit_at_a_fixed_dispersion(tmp_path):
    _screen(tmp_path / "s.h5ad", n_strata=1)
    offset = _run(tmp_path / "s.h5ad", tmp_path / "off.h5ad", dispersion=0.3, lfc_base="ln", **NO_FILTERS)
    two = ad.read_h5ad(cx.nb_glm_test(
        tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control", dispersion=0.3,
        lfc_base="ln", output_path=tmp_path / "glm.h5ad", verbose=False, **NO_FILTERS,
    ).result_path)
    ok = np.isfinite(offset.X) & np.isfinite(two.X)
    assert np.median(np.abs(offset.X - two.X)[ok]) < 0.05
    assert offset.uns["dispersion_source"] == "given"
    np.testing.assert_allclose(offset.layers["dispersion"], 0.3)


def test_a_stratum_without_control_counts_of_a_gene_does_not_send_beta_to_the_bound(tmp_path):
    _screen(tmp_path / "s.h5ad", zero_stratum_gene=(2, 30))
    offset = _run(tmp_path / "s.h5ad", tmp_path / "off.h5ad", strata="stratum", lfc_base="ln", **NO_FILTERS)
    value = offset.X[0, 30]
    assert np.isfinite(value) and value < 10  # the pseudo-cell keeps the baseline above zero


def test_cells_of_strata_without_controls_are_left_out(tmp_path):
    _screen(tmp_path / "s.h5ad")
    obs = cx.load_obs(tmp_path / "s.h5ad")
    obs["stratum"] = obs["stratum"].astype(str)
    obs.loc[obs["perturbation"] == "p1", "stratum"] = "orphan"          # all of p1 in a stratum with no control
    obs.loc[(obs["perturbation"] == "p2").to_numpy() & (np.arange(len(obs)) % 2 == 0), "stratum"] = "orphan"
    cx.write_obs(tmp_path / "s.h5ad", obs)
    with pytest.warns(UserWarning, match="no control cell"):
        result = cx.tl.nb_offset_test(
            tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control", strata="stratum",
            output_path=tmp_path / "off.h5ad", verbose=False, **NO_FILTERS,
        )
    got = ad.read_h5ad(result.result_path)
    assert np.isnan(got.X[got.obs_names.get_loc("p1")]).all()
    assert np.isfinite(got.X[got.obs_names.get_loc("p2")]).sum() > 40
    # the filters and pts describe the fitted cells, not the ones left out
    counts = ad.read_h5ad(tmp_path / "s.h5ad").X.toarray()
    fitted = ((obs["perturbation"] == "p2") & (obs["stratum"] != "orphan")).to_numpy()
    np.testing.assert_allclose(got.layers["pts"][got.obs_names.get_loc("p2")], (counts[fitted] > 0).mean(axis=0),
                               rtol=1e-6)


def test_a_csc_file_is_warned_about(tmp_path):
    _screen(tmp_path / "s.h5ad", n_ctrl=300, n_pert=3)
    a = ad.read_h5ad(tmp_path / "s.h5ad")
    ad.AnnData(sp.csc_matrix(a.X), obs=a.obs, var=a.var).write(tmp_path / "csc.h5ad")
    with pytest.warns(UserWarning, match="CSC format"):
        result = cx.tl.nb_offset_test(
            tmp_path / "csc.h5ad", perturbation_column="perturbation", control_label="control", strata="stratum",
            output_path=tmp_path / "off.h5ad", verbose=False, **NO_FILTERS,
        )
    assert np.isfinite(ad.read_h5ad(result.result_path).X).any()


def test_the_sorting_step_is_printed(tmp_path, monkeypatch, capsys):
    import crispyx.de as de

    _screen(tmp_path / "s.h5ad", n_ctrl=300, n_pert=3)
    monkeypatch.setattr(de, "needs_sorting_for_nbglm", lambda *a, **k: True)
    cx.tl.nb_offset_test(tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control",
                         strata="stratum", output_path=tmp_path / "off.h5ad", **NO_FILTERS)
    out = capsys.readouterr().out
    assert "[cx] nb_offset_test: the cells of each perturbation are scattered through s.h5ad" in out
    assert "[cx] sort_by_perturbation: Saving → " in out and (tmp_path / "s_sorted.h5ad").exists()


def test_result_is_independent_of_the_cell_block_size(tmp_path):
    _screen(tmp_path / "s.h5ad", n_pert=3, cells_per=200)
    big = _run(tmp_path / "s.h5ad", tmp_path / "a.h5ad", strata="stratum", **NO_FILTERS)
    small = _run(tmp_path / "s.h5ad", tmp_path / "b.h5ad", strata="stratum", memory_limit_gb=1e-5, **NO_FILTERS)
    np.testing.assert_allclose(big.X, small.X, rtol=1e-8, atol=1e-10, equal_nan=True)


def test_log2_layers_and_shrink_lfc_on_the_result(tmp_path):
    _screen(tmp_path / "s.h5ad")
    result = cx.tl.nb_offset_test(
        tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control", strata="stratum",
        output_path=tmp_path / "off.h5ad", verbose=False,
    )
    got = ad.read_h5ad(result.result_path)
    np.testing.assert_allclose(got.X * np.log(2), got.layers["logfoldchange_raw_ln"], equal_nan=True)
    assert result.method == "nb_offset" and result["p0"].logfoldchanges.shape == (60,)
    shrunk = ad.read_h5ad(cx.shrink_lfc(result.result_path, verbose=False).result_path)
    finite = np.isfinite(got.X)
    assert np.isfinite(shrunk.X[finite]).all()
    assert (np.abs(shrunk.X[finite]) <= np.abs(got.X[finite]) + 1e-9).all()
    with pytest.raises(ValueError, match="method='full'"):
        cx.shrink_lfc(result.result_path, method="full", verbose=False)


def test_given_dispersion_per_gene_and_bad_shapes(tmp_path):
    _screen(tmp_path / "s.h5ad", n_genes=12)
    alpha = np.linspace(0.1, 0.5, 12)
    got = _run(tmp_path / "s.h5ad", tmp_path / "off.h5ad", dispersion=alpha, **NO_FILTERS)
    np.testing.assert_allclose(got.layers["dispersion"][3], alpha)
    with pytest.raises(ValueError, match="expected one value or"):
        _run(tmp_path / "s.h5ad", tmp_path / "x.h5ad", dispersion=alpha[:5])
    with pytest.raises(ValueError, match="positive and finite"):
        _run(tmp_path / "s.h5ad", tmp_path / "x.h5ad", dispersion=-1.0)
    with pytest.raises(KeyError, match="'nope'"):
        _run(tmp_path / "s.h5ad", tmp_path / "x.h5ad", strata="nope")


def test_an_existing_result_is_reused(tmp_path):
    _screen(tmp_path / "s.h5ad", n_genes=12)
    kwargs = dict(perturbation_column="perturbation", control_label="control", strata="stratum",
                  output_path=tmp_path / "off.h5ad", verbose=False)
    first = cx.tl.nb_offset_test(tmp_path / "s.h5ad", **kwargs)
    mtime = first.result_path.stat().st_mtime_ns
    again = cx.tl.nb_offset_test(tmp_path / "s.h5ad", **kwargs)
    assert again.result_path.stat().st_mtime_ns == mtime
    assert cx.tl.nb_offset_test(tmp_path / "s.h5ad", force=True, **kwargs).result_path.stat().st_mtime_ns != mtime


def test_an_empty_selection_writes_an_empty_result_for_every_test(tmp_path):
    _screen(tmp_path / "s.h5ad", n_genes=12, n_pert=2)
    log = cx.pp.normalize_total_log1p(tmp_path / "s.h5ad", output_path=tmp_path / "log.h5ad", verbose=False)
    calls = [
        (cx.nb_glm_test, tmp_path / "s.h5ad", {}),
        (cx.tl.nb_offset_test, tmp_path / "s.h5ad", {"strata": "stratum"}),
        (cx.t_test, log.filename, {}),
        (cx.wilcoxon_test, log.filename, {}),
        (cx.wilcoxon_test, log.filename, {"batch_column": "stratum"}),
    ]
    for i, (fn, path, kwargs) in enumerate(calls):
        result = fn(path, perturbation_column="perturbation", control_label="control", perturbations=[],
                    output_path=tmp_path / f"empty{i}.h5ad", verbose=False, **kwargs)
        assert len(result) == 0 and np.asarray(result.pvalues).shape == (0, 12)
        assert ad.read_h5ad(result.result_path).shape == (0, 12)
    # a file with nothing but controls still raises when no selection is given
    obs = cx.load_obs(tmp_path / "s.h5ad")
    obs["perturbation"] = "control"
    cx.write_obs(tmp_path / "s.h5ad", obs)
    with pytest.raises(ValueError, match="No perturbation groups"):
        cx.nb_glm_test(tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control",
                       output_path=tmp_path / "none.h5ad", verbose=False)


def _units(path: Path, *, n_ctrl=1500, n_strata=8, n_pert=8, units_per=12, n_genes=50, alpha=0.2, seed=5):
    """Pseudo-bulk-like units: a stratum and a continuous covariate (log cells
    per unit) shape every unit's mean; the controls are many, the
    perturbations few units each."""
    rng = np.random.default_rng(seed)
    n = n_ctrl + n_pert * units_per
    strata = rng.integers(0, n_strata, n)
    logn = rng.normal(1.0, 0.3, n)
    labels = np.array(["control"] * n_ctrl + [f"p{k}" for k in range(n_pert) for _ in range(units_per)])
    base = np.exp(rng.uniform(np.log(1), np.log(50), n_genes))
    stratum_effect = rng.normal(0, 0.4, (n_strata, n_genes))
    slope = rng.normal(0.8, 0.2, n_genes)
    beta = np.zeros((n_pert, n_genes))
    beta[:, :15] = rng.normal(0, 0.8, (n_pert, 15))
    pert = np.full(n, -1)
    pert[n_ctrl:] = np.repeat(np.arange(n_pert), units_per)
    effect = np.where(pert[:, None] >= 0, beta[np.maximum(pert, 0)], 0.0)
    mu = base[None, :] * np.exp(stratum_effect[strata] + slope[None, :] * (logn[:, None] - 1.0) + effect)
    counts = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels, "stratum": [f"s{b}" for b in strata], "logn": logn},
                       index=[f"u{i}" for i in range(n)])
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])).write(path)
    return beta


def test_covariates_fit_the_controls_once_and_agree_with_the_joint_glm(tmp_path):
    beta = _units(tmp_path / "u.h5ad")
    offset = _run(tmp_path / "u.h5ad", tmp_path / "off.h5ad", covariates=["stratum", "logn"], dispersion=0.2,
                  lfc_base="ln", min_mu=0.5, **NO_FILTERS)
    joint = ad.read_h5ad(cx.nb_glm_test(
        tmp_path / "u.h5ad", perturbation_column="perturbation", control_label="control",
        covariates=["stratum", "logn"], dispersion=0.2, lfc_base="ln", min_mu=0.5,
        output_path=tmp_path / "glm.h5ad", verbose=False, **NO_FILTERS,
    ).result_path)
    ok = np.isfinite(offset.X) & np.isfinite(joint.X)
    assert ok.mean() > 0.95
    assert np.median(np.abs(offset.X - joint.X)[ok]) < 0.05
    assert 0.9 < np.median((offset.layers["standard_error"] / joint.layers["standard_error"])[ok]) < 1.1
    assert np.corrcoef(offset.X[:, :15][ok[:, :15]], beta[:, :15][ok[:, :15]])[0, 1] > 0.9
    assert list(offset.uns["covariates"]) == ["stratum", "logn"] and offset.uns["strata"] == ""
    # the dispersion estimated from the controls alone is near the truth
    estimated = _run(tmp_path / "u.h5ad", tmp_path / "est.h5ad", covariates=["stratum", "logn"], min_mu=0.5,
                     **NO_FILTERS)
    assert 0.1 < np.median(estimated.layers["dispersion"][0]) < 0.4
    assert estimated.uns["dispersion_source"] == "controls"


def test_covariate_levels_without_controls_drop_their_cells_and_both_designs_raise(tmp_path):
    _units(tmp_path / "u.h5ad")
    obs = cx.load_obs(tmp_path / "u.h5ad")
    obs["stratum"] = obs["stratum"].astype(str)
    obs.loc[obs["perturbation"] == "p1", "stratum"] = "orphan"
    cx.write_obs(tmp_path / "u.h5ad", obs)
    with pytest.warns(UserWarning, match="no control cell"):
        result = cx.tl.nb_offset_test(
            tmp_path / "u.h5ad", perturbation_column="perturbation", control_label="control",
            covariates=["stratum", "logn"], output_path=tmp_path / "off.h5ad", verbose=False, **NO_FILTERS,
        )
    got = ad.read_h5ad(result.result_path)
    assert np.isnan(got.X[got.obs_names.get_loc("p1")]).all()
    assert np.isfinite(got.X[got.obs_names.get_loc("p0")]).sum() > 40
    with pytest.raises(ValueError, match="not both"):
        cx.tl.nb_offset_test(tmp_path / "u.h5ad", perturbation_column="perturbation", control_label="control",
                             strata="stratum", covariates=["logn"], output_path=tmp_path / "x.h5ad", verbose=False)


def test_nb_glm_covariate_path_reports_its_fits_and_timing(tmp_path, capsys):
    _units(tmp_path / "u.h5ad", n_ctrl=300, n_pert=4, units_per=10, n_genes=20)
    cx.nb_glm_test(
        tmp_path / "u.h5ad", perturbation_column="perturbation", control_label="control", covariates=["stratum"],
        targets_per_fit=2, dispersion=0.2, output_path=tmp_path / "glm.h5ad", verbose=True, n_jobs=1, **NO_FILTERS,
    )
    out = capsys.readouterr().out
    assert "2 fits of up to 2 perturbation(s) with the 300 controls" in out
    assert "first fit took" in out and "1 more with 1 worker(s)" in out


def test_the_dispersion_is_unbiased_over_many_small_strata():
    """The residuals are about each stratum's own baseline, which costs a
    degree of freedom per stratum: subtracting the full Poisson term biased
    alpha down by 1/n_b, to the floor over hundreds of two-cell strata."""
    from crispyx.de import _offset_baseline

    rng = np.random.default_rng(3)
    alpha, n_genes = 0.3, 400
    for n_strata, per_stratum in ((300, 2), (200, 3), (6, 500)):
        n = n_strata * per_stratum
        strata = np.repeat(np.arange(n_strata), per_stratum)
        s = np.exp(rng.normal(0, 0.4, n))
        base = np.exp(rng.uniform(np.log(1), np.log(20), n_genes))
        mu = s[:, None] * base[None, :] * np.exp(rng.normal(0, 0.5, (n_strata, n_genes))[strata])
        y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(np.float64)
        baseline = _offset_baseline(
            iter([(np.arange(n), sp.csr_matrix(y))]), s, strata, n_strata, n_genes, dispersion=None,
        )
        assert 0.2 < np.median(baseline.dispersion_raw) < 0.4, (n_strata, per_stratum)
    # a stratum of one control contributes nothing: alone, no dispersion is estimable
    baseline = _offset_baseline(
        iter([(np.arange(3), sp.csr_matrix(y[:3]))]), s[:3], np.arange(3), 3, n_genes, dispersion=None,
    )
    assert np.isnan(baseline.dispersion_raw).all()


def test_a_selection_of_only_the_control_raises(tmp_path):
    _screen(tmp_path / "s.h5ad", n_genes=12, n_pert=2)
    for fn in (cx.nb_glm_test, cx.tl.nb_offset_test):
        with pytest.raises(ValueError, match="No perturbation groups"):
            fn(tmp_path / "s.h5ad", perturbation_column="perturbation", control_label="control",
               perturbations=["control"], output_path=tmp_path / "x.h5ad", verbose=False)


class _RecordingBacked:
    """A backed AnnData whose ``X`` records how each read indexes it."""

    def __init__(self, backed, keys: list):
        self._backed, self._keys = backed, keys

    def __getattr__(self, name):
        return getattr(self._backed, name)

    @property
    def X(self):
        inner, keys = self._backed.X, self._keys

        class _Proxy:
            def __getitem__(self, key):
                keys.append(key)
                return inner[key]

            def __getattr__(self, name):
                return getattr(inner, name)

        return _Proxy()


def test_a_sorted_input_is_read_one_slice_per_group(tmp_path, monkeypatch):
    """A slice of a backed CSR matrix is one read of its data and indices;
    an index array is reads per row, minutes over a million cells."""
    import crispyx.de as de

    _screen(tmp_path / "s.h5ad", n_ctrl=300, n_pert=3)
    monkeypatch.setattr(de, "needs_sorting_for_nbglm", lambda *a, **k: True)
    keys: list = []
    real_read_backed = de.read_backed
    monkeypatch.setattr(de, "read_backed", lambda path, *a, **k: _RecordingBacked(real_read_backed(path, *a, **k), keys))
    result = _run(tmp_path / "s.h5ad", tmp_path / "off.h5ad", strata="stratum", **NO_FILTERS)
    assert (tmp_path / "s_sorted.h5ad").exists() and np.isfinite(result.X).any()
    rows = [key[0] if isinstance(key, tuple) else key for key in keys]
    assert rows and all(isinstance(row, slice) for row in rows), [type(r) for r in rows]
    # the unsorted file's scattered groups are still read by index
    unsorted = _run(tmp_path / "s_sorted.h5ad", tmp_path / "off2.h5ad", strata="stratum", **NO_FILTERS)
    np.testing.assert_allclose(result.X, unsorted.X, equal_nan=True)


def test_the_cell_block_is_shared_between_the_threads(tmp_path, monkeypatch):
    """Every thread fits a block at once, so a thread's block is its share
    of the tenth of the budget, not the whole tenth n_threads times over."""
    import crispyx.de as de

    _screen(tmp_path / "s.h5ad", n_ctrl=200, n_pert=2, n_genes=60)
    seen: list[int] = []
    real = de._fit_offset_group

    def recording(*args, block_cells, **kwargs):
        seen.append(block_cells)
        return real(*args, block_cells=block_cells, **kwargs)

    monkeypatch.setattr(de, "_fit_offset_group", recording)
    budget_gb = 60 * 8 * 5 * 4096 * 10 / 1e9  # a tenth of it holds 4096 cells of 60 genes in five arrays
    _run(tmp_path / "s.h5ad", tmp_path / "one.h5ad", memory_limit_gb=budget_gb, n_jobs=1, **NO_FILTERS)
    _run(tmp_path / "s.h5ad", tmp_path / "four.h5ad", memory_limit_gb=budget_gb, n_jobs=4, **NO_FILTERS)
    assert seen[0] == 4096 and seen[-1] == 1024, seen
