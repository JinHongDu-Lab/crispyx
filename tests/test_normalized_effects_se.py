"""Standard errors of the stratified effects of ``compute_normalized_effects``
(Liu et al., 2026, doi:10.64898/2026.10.05.756839), against a direct
per-batch calculation of the published formulas."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _screen(tmp_path, sizes, seed=0):
    """Counts with a batch offset; ``sizes`` maps (group, batch) to cells."""
    rng = np.random.default_rng(seed)
    rows, records = [], []
    for (group, batch), n in sizes.items():
        rate = {"b1": 12, "b2": 4, "b3": 7}[batch] * (1.5 if group == "KO_A" else 1.0)
        rows.append(rng.poisson(rate, size=(n, 5)).astype(np.float32))
        records += [{"perturbation": group, "batch": batch}] * n
    counts = np.vstack(rows)
    obs = pd.DataFrame(records, index=[f"c{i}" for i in range(len(records))])
    path = tmp_path / "counts.h5ad"
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(5)])).write(path)
    return path, counts.astype(np.float64), obs


def _reference(counts, obs, target, batch_column):
    """Effect, model SE and robust SE by the published formulas, one batch at a time."""
    y = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    batches = obs[batch_column] if batch_column else pd.Series("all", index=obs.index)
    w, d, n_p, n_c, var_p, var_c = [], [], [], [], [], []
    for batch in pd.unique(batches):
        in_batch = (batches == batch).to_numpy()
        p = y[in_batch & (obs["perturbation"] == target).to_numpy()]
        c = y[in_batch & (obs["perturbation"] == "control").to_numpy()]
        if len(p) == 0 or len(c) == 0:
            continue
        w.append(len(p) * len(c) / (len(p) + len(c)))
        d.append(p.mean(axis=0) - c.mean(axis=0))
        n_p.append(len(p))
        n_c.append(len(c))
        var_p.append(p.var(axis=0) if len(p) >= 2 else None)
        var_c.append(c.var(axis=0) if len(c) >= 2 else None)
    w, d, n_p, n_c = np.array(w), np.array(d), np.array(n_p), np.array(n_c)
    W = w.sum()
    effect = (w[:, None] * d).sum(axis=0) / W

    def pooled(variances):
        used = [(wb, v) for wb, v in zip(w, variances) if v is not None]
        if not used:
            return np.full(d.shape[1], np.nan)
        return sum(wb * v for wb, v in used) / sum(wb for wb, _ in used)

    se = np.sqrt((pooled(var_p) * (w**2 / n_p).sum() + pooled(var_c) * (w**2 / n_c).sum()) / W**2)
    if len(w) >= 2:
        robust = np.sqrt(len(w) / (len(w) - 1) * ((w**2)[:, None] * (d - effect) ** 2).sum(axis=0) / W**2)
    else:
        robust = np.full(d.shape[1], np.nan)
    return effect, se, robust


def _run(path, tmp_path, batch_column, **kwargs):
    result = cx.pb.normalized_effects(path, perturbation_column="perturbation", control_label="control",
                                      batch_column=batch_column, output_path=tmp_path / "out.h5ad",
                                      verbose=False, **kwargs)
    out = ad.read_h5ad(result.path)
    result.close()
    return out


SIZES = {("control", "b1"): 9, ("control", "b2"): 5, ("control", "b3"): 6,
         ("KO_A", "b1"): 8, ("KO_A", "b2"): 1, ("KO_A", "b3"): 3,   # a one-cell batch
         ("KO_B", "b1"): 6, ("KO_B", "b2"): 6,                       # no b3 cells
         ("KO_C", "b3"): 4}                                          # a single batch


@pytest.mark.parametrize("batch_column", ["batch", None])
def test_standard_errors_match_the_published_formulas(tmp_path, batch_column):
    path, counts, obs = _screen(tmp_path, SIZES)
    out = _run(path, tmp_path, batch_column, chunk_size=7)
    for target in ("KO_A", "KO_B", "KO_C"):
        effect, se, robust = _reference(counts, obs, target, batch_column)
        row = out.obs_names.get_loc(target)
        np.testing.assert_allclose(out.X[row], effect, rtol=1e-10, atol=1e-12, err_msg=target)
        np.testing.assert_allclose(out.layers["standard_error"][row], se, rtol=1e-8, err_msg=target)
        np.testing.assert_allclose(out.layers["standard_error_robust"][row], robust, rtol=1e-8,
                                   equal_nan=True, err_msg=target)
    if batch_column is None:
        assert np.isnan(out.layers["standard_error_robust"]).all()  # one batch
    else:
        assert np.isnan(out.layers["standard_error_robust"][out.obs_names.get_loc("KO_C")]).all()


def test_without_batches_the_standard_error_is_the_two_sample_one(tmp_path):
    path, counts, obs = _screen(tmp_path, SIZES)
    out = _run(path, tmp_path, None)
    y = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    p, c = y[(obs["perturbation"] == "KO_B").to_numpy()], y[(obs["perturbation"] == "control").to_numpy()]
    np.testing.assert_allclose(out.layers["standard_error"][out.obs_names.get_loc("KO_B")],
                               np.sqrt(p.var(axis=0) / len(p) + c.var(axis=0) / len(c)), rtol=1e-8)


def test_the_effect_is_the_coefficient_of_a_regression_with_batch_intercepts(tmp_path):
    path, counts, obs = _screen(tmp_path, SIZES)
    out = _run(path, tmp_path, "batch")
    y = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    keep = obs["perturbation"].isin(["control", "KO_A"]).to_numpy()
    design = np.column_stack([
        (obs["perturbation"] == "KO_A").to_numpy()[keep],
        pd.get_dummies(obs["batch"][keep]).to_numpy(dtype=float),
    ]).astype(float)
    coef, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
    np.testing.assert_allclose(out.X[out.obs_names.get_loc("KO_A")], coef[0], rtol=1e-10, atol=1e-12)


def test_a_side_with_only_single_cell_batches_has_no_model_standard_error(tmp_path):
    sizes = {("control", "b1"): 5, ("control", "b2"): 5, ("KO_A", "b1"): 1, ("KO_A", "b2"): 1}
    path, _, _ = _screen(tmp_path, sizes)
    out = _run(path, tmp_path, "batch")
    assert np.isfinite(out.X).all()
    assert np.isnan(out.layers["standard_error"]).all()
    assert np.isfinite(out.layers["standard_error_robust"]).all()


def test_log_mean_has_no_standard_error(tmp_path):
    path, _, _ = _screen(tmp_path, SIZES)
    out = _run(path, tmp_path, "batch", method="log_mean")
    assert "standard_error" not in out.layers and "standard_error_robust" not in out.layers


def test_accumulators_in_memory_or_on_disk_give_the_same_result(tmp_path, monkeypatch):
    import crispyx.pseudobulk as pseudobulk

    placements = []
    original = pseudobulk._pair_accumulator

    def recording(*args, on_disk, **kwargs):
        placements.append(on_disk)
        return original(*args, on_disk=on_disk, **kwargs)

    monkeypatch.setattr(pseudobulk, "_pair_accumulator", recording)
    path, _, _ = _screen(tmp_path, SIZES)
    (tmp_path / "ram").mkdir()
    (tmp_path / "disk").mkdir()
    in_memory = _run(path, tmp_path / "ram", "batch")
    on_disk = _run(path, tmp_path / "disk", "batch", memory_limit_gb=1e-9, chunk_size=7)
    assert placements == [False, False, True, True]
    np.testing.assert_allclose(in_memory.X, on_disk.X, rtol=1e-12, atol=1e-15)
    for layer in ("standard_error", "standard_error_robust"):
        np.testing.assert_allclose(in_memory.layers[layer], on_disk.layers[layer], rtol=1e-12, equal_nan=True)
