"""Tests for the generic streaming batch-statistics API."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx.data import convert_to_csc


def _write_data(tmp_path: Path, *, sparse: bool = True) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    batches: list[str] = []
    for batch_index, batch in enumerate(("b1", "b2", "b3")):
        for group_index, group in enumerate(("ctrl", "A", "B")):
            n = 5 + batch_index + group_index
            values = rng.normal(
                loc=3 * batch_index + group_index,
                scale=0.5 + batch_index + 0.25 * group_index,
                size=(n, 7),
            )
            rows.extend(values)
            labels.extend([group] * n)
            batches.extend([batch] * n)
    X = np.asarray(rows, dtype=np.float64)
    matrix = sp.csr_matrix(X) if sparse else X
    obs = pd.DataFrame(
        {"perturbation": labels, "batch": batches},
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])])
    path = tmp_path / ("sparse.h5ad" if sparse else "dense.h5ad")
    ad.AnnData(matrix, obs=obs, var=var).write(path)
    return path, X, np.asarray(labels), np.asarray(batches)


def _moment_reducer() -> cx.BatchReducer:
    def initialize(width):
        return {"n": 0, "mean": np.zeros(width), "m2": np.zeros(width)}

    def update(state, block):
        n_b = block.shape[0]
        if n_b == 0:
            return None
        mean_b = block.mean(axis=0)
        m2_b = np.square(block - mean_b).sum(axis=0)
        if state["n"] == 0:
            state["n"] = n_b
            state["mean"][:] = mean_b
            state["m2"][:] = m2_b
            return None
        total = state["n"] + n_b
        delta = mean_b - state["mean"]
        state["m2"] += m2_b + delta * delta * state["n"] * n_b / total
        state["mean"] += delta * n_b / total
        state["n"] = total

    def finalize(state):
        return cx.BatchStatistic(
            np.sqrt(state["m2"] / (state["n"] - 1)),
            state["n"],
        )

    def compare(group_state, reference_state):
        n_g, n_r = group_state["n"], reference_state["n"]
        weight = n_g * n_r / (n_g + n_r)
        return cx.BatchStatistic(
            group_state["mean"] - reference_state["mean"],
            weight,
        )

    return cx.BatchReducer(initialize, update, finalize, compare)


def _read_result(result: cx.AnnData) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    backed = result.backed
    return (
        np.asarray(backed.X[:]),
        np.asarray(backed.layers["weight_sum"][:]),
        backed.obs.copy(),
        dict(backed.uns),
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_weighted_batch_std_matches_reference_and_streams(tmp_path, sparse):
    path, X, labels, batches = _write_data(tmp_path, sparse=sparse)
    result = cx.tl.batch_process(
        path,
        _moment_reducer(),
        groupby="perturbation",
        batch_column="batch",
        mode="group",
        statistic_name="std",
        chunk_size=2,
        cell_chunk_size=3,
        output_path=tmp_path / f"std_{sparse}.h5ad",
        force=True,
    )
    actual, weights, obs, uns = _read_result(result)

    expected = []
    expected_weight = []
    for group in ("ctrl", "A", "B"):
        values = []
        counts = []
        for batch in ("b1", "b2", "b3"):
            subset = X[(labels == group) & (batches == batch)]
            values.append(subset.std(axis=0, ddof=1))
            counts.append(subset.shape[0])
        expected.append(np.average(values, axis=0, weights=counts))
        expected_weight.append(np.full(X.shape[1], sum(counts)))

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights, expected_weight)
    assert obs["n_batches_used"].tolist() == [3, 3, 3]
    assert uns["statistic_name"] == "std"
    assert uns["perturbation_column"] == "perturbation"
    assert uns["stratified_n_batches"] == 3


def test_comparison_mode_infers_reference_and_uses_custom_weight(tmp_path):
    path, X, labels, batches = _write_data(tmp_path)
    result = cx.batch_process(
        path,
        _moment_reducer(),
        perturbation_column="perturbation",
        batch_column="batch",
        mode="comparison",
        statistic_name="mean_difference",
        perturbations=["B", "A"],
        chunk_size=3,
        cell_chunk_size=4,
        output_path=tmp_path / "comparison.h5ad",
        force=True,
    )
    actual, weights, obs, uns = _read_result(result)
    expected = []
    expected_weights = []
    for group in ("B", "A"):
        contrasts, harmonic = [], []
        for batch in ("b1", "b2", "b3"):
            group_values = X[(labels == group) & (batches == batch)]
            ctrl_values = X[(labels == "ctrl") & (batches == batch)]
            contrasts.append(group_values.mean(0) - ctrl_values.mean(0))
            harmonic.append(
                group_values.shape[0] * ctrl_values.shape[0]
                / (group_values.shape[0] + ctrl_values.shape[0])
            )
        expected.append(np.average(contrasts, axis=0, weights=harmonic))
        expected_weights.append(np.full(X.shape[1], sum(harmonic)))

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights, expected_weights)
    assert obs.index.tolist() == ["B", "A"]
    assert uns["control_label"] == "ctrl"


def test_alias_conflicts_match_de_conventions(tmp_path):
    path, *_ = _write_data(tmp_path)
    reducer = _moment_reducer()
    with pytest.raises(TypeError, match="perturbation_column.*groupby"):
        cx.batch_process(
            path,
            reducer,
            perturbation_column="perturbation",
            groupby="perturbation",
            batch_column="batch",
            statistic_name="std",
        )
    with pytest.raises(TypeError, match="control_label.*reference"):
        cx.batch_process(
            path,
            reducer,
            groupby="perturbation",
            control_label="ctrl",
            reference="ctrl",
            batch_column="batch",
            mode="comparison",
            statistic_name="difference",
        )


def test_no_shared_batch_is_nan_with_diagnostics(tmp_path):
    X = np.arange(24, dtype=float).reshape(6, 4)
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl"] * 3 + ["A"] * 3,
            "batch": ["b1"] * 3 + ["b2"] * 3,
        },
        index=[f"c{i}" for i in range(6)],
    )
    path = tmp_path / "unshared.h5ad"
    ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(4)])).write(path)
    with pytest.warns(UserWarning, match="no usable batch statistics"):
        result = cx.tl.batch_process(
            path,
            _moment_reducer(),
            groupby="perturbation",
            reference="ctrl",
            batch_column="batch",
            mode="comparison",
            statistic_name="difference",
            chunk_size=2,
            cell_chunk_size=2,
            output_path=tmp_path / "unshared_result.h5ad",
            force=True,
        )
    values, weights, obs_result, uns = _read_result(result)
    assert np.isnan(values).all()
    assert np.equal(weights, 0).all()
    assert obs_result["n_batches_used"].tolist() == [0]
    assert uns["stratified_n_untestable_perturbations"] == 1


def test_gene_wise_weights_and_bare_vector(tmp_path):
    path, *_ = _write_data(tmp_path)

    def initialize(width):
        return np.zeros(width), 0

    def update(state, block):
        total, count = state
        return total + block.sum(0), count + block.shape[0]

    def finalize(state):
        total, count = state
        means = total / count
        weights = np.arange(1, means.size + 1, dtype=float)
        return cx.BatchStatistic(means, weights)

    weighted = cx.BatchReducer(initialize, update, finalize)
    result = cx.batch_process(
        path,
        weighted,
        groupby="perturbation",
        perturbations=["A"],
        batch_column="batch",
        statistic_name="weighted_mean",
        chunk_size=7,
        cell_chunk_size=5,
        output_path=tmp_path / "weighted.h5ad",
        force=True,
    )
    _, weights, _, _ = _read_result(result)
    np.testing.assert_allclose(weights[0], 3 * np.arange(1, 8))

    bare = cx.BatchReducer(initialize, update, lambda state: state[0] / state[1])
    result = cx.batch_process(
        path,
        bare,
        groupby="perturbation",
        perturbations=["A"],
        batch_column="batch",
        statistic_name="equal_mean",
        output_path=tmp_path / "bare.h5ad",
        chunk_size=7,
        cell_chunk_size=5,
        force=True,
    )
    _, weights, _, _ = _read_result(result)
    np.testing.assert_allclose(weights, 3)


def test_invalid_reducer_output_and_weights(tmp_path):
    path, *_ = _write_data(tmp_path)

    bad_shape = cx.BatchReducer(lambda width: None, lambda state, block: None, lambda state: [1])
    with pytest.raises(ValueError, match="must have shape"):
        cx.batch_process(
            path,
            bad_shape,
            groupby="perturbation",
            perturbations=["A"],
            batch_column="batch",
            statistic_name="bad",
            chunk_size=2,
            output_path=tmp_path / "bad_shape.h5ad",
            force=True,
        )

    bad_weight = cx.BatchReducer(
        lambda width: width,
        lambda state, block: None,
        lambda width: cx.BatchStatistic(np.zeros(width), -1),
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        cx.batch_process(
            path,
            bad_weight,
            groupby="perturbation",
            perturbations=["A"],
            batch_column="batch",
            statistic_name="bad_weight",
            chunk_size=2,
            output_path=tmp_path / "bad_weight.h5ad",
            force=True,
        )


def test_existing_matching_result_reloads_unless_forced(tmp_path):
    path, *_ = _write_data(tmp_path)
    output_path = tmp_path / "cached.h5ad"
    kwargs = dict(
        groupby="perturbation",
        perturbations=["A"],
        batch_column="batch",
        statistic_name="std",
        output_path=output_path,
    )
    first = cx.batch_process(path, _moment_reducer(), force=True, **kwargs)
    first.close()
    mtime = output_path.stat().st_mtime
    time.sleep(0.01)
    second = cx.tl.batch_process(path, _moment_reducer(), **kwargs)
    second.close()
    assert output_path.stat().st_mtime == mtime


def test_missing_batch_labels_are_excluded_and_callback_errors_have_context(tmp_path):
    path, *_ = _write_data(tmp_path)
    adata = ad.read_h5ad(path)
    adata.obs["batch"] = adata.obs["batch"].astype(object)
    adata.obs.iloc[0, adata.obs.columns.get_loc("batch")] = None
    adata.write(path)
    with pytest.warns(UserWarning, match="missing 'batch'"):
        result = cx.batch_process(
            path,
            _moment_reducer(),
            groupby="perturbation",
            perturbations=["A"],
            batch_column="batch",
            statistic_name="std_missing",
            output_path=tmp_path / "missing.h5ad",
            force=True,
        )
    result.close()

    failing = cx.BatchReducer(
        lambda width: np.zeros(width),
        lambda state, block: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda state: state,
    )
    with pytest.warns(UserWarning, match="missing 'batch'"):
        with pytest.raises(RuntimeError, match="group 'A', batch 'b1'"):
            cx.batch_process(
                path,
                failing,
                groupby="perturbation",
                perturbations=["A"],
                batch_column="batch",
                statistic_name="failing",
                output_path=tmp_path / "failing.h5ad",
                force=True,
            )


def test_cache_is_invalidated_when_the_source_file_changes(tmp_path):
    """Regenerating the input in place must not return the previous result."""
    path, X, *_ = _write_data(tmp_path)
    kwargs = dict(
        groupby="perturbation",
        batch_column="batch",
        statistic_name="std",
        output_path=tmp_path / "cached_source.h5ad",
    )
    first = cx.batch_process(path, _moment_reducer(), force=True, **kwargs)
    before = np.asarray(first.backed.X[:]).copy()
    first.close()

    # Same path, same groups and batches, different values.
    obs = pd.DataFrame(
        {
            "perturbation": np.repeat(["ctrl", "A", "B"], X.shape[0] // 3),
            "batch": np.tile(["b1", "b2", "b3"], X.shape[0] // 3),
        },
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])])
    ad.AnnData(sp.csr_matrix(X * 10.0), obs=obs, var=var).write(path)

    second = cx.batch_process(path, _moment_reducer(), **kwargs)
    after = np.asarray(second.backed.X[:]).copy()
    second.close()

    assert not np.allclose(before, after)

    # An untouched source must still reload from cache.
    mtime = kwargs["output_path"].stat().st_mtime_ns
    time.sleep(0.01)
    third = cx.batch_process(path, _moment_reducer(), **kwargs)
    third.close()
    assert kwargs["output_path"].stat().st_mtime_ns == mtime


@pytest.mark.parametrize("mode", ["group", "comparison"])
def test_non_reducer_raises_type_error_in_both_modes(tmp_path, mode):
    path, *_ = _write_data(tmp_path)
    with pytest.raises(TypeError, match="reducer must be a BatchReducer instance"):
        cx.batch_process(
            path,
            object(),
            groupby="perturbation",
            batch_column="batch",
            mode=mode,
            statistic_name="std",
            output_path=tmp_path / "not_a_reducer.h5ad",
        )


def test_warns_when_disk_space_low(tmp_path, monkeypatch):
    """A near-full disk should warn but not block batch_process from completing."""
    import shutil
    import types

    path, *_ = _write_data(tmp_path)
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda p: types.SimpleNamespace(total=1000, used=999, free=1),
    )
    with pytest.warns(UserWarning, match="tl.batch_process"):
        result = cx.batch_process(
            path,
            _moment_reducer(),
            groupby="perturbation",
            batch_column="batch",
            statistic_name="std_low_disk",
            output_path=tmp_path / "low_disk.h5ad",
        )
    assert result.backed.n_obs > 0


def _crashing_reducer(crash_after_updates: int) -> cx.BatchReducer:
    """A moment reducer whose update() raises after a fixed call count.

    With cell_chunk_size larger than any (group, batch) cell count, each
    (group, batch) pair triggers exactly one update() call per gene chunk --
    so this deterministically fails partway through a specific gene chunk.
    """
    base = _moment_reducer()
    counter = {"n": 0}

    def update(state, block):
        counter["n"] += 1
        if counter["n"] > crash_after_updates:
            raise RuntimeError("simulated crash")
        return base.update(state, block)

    return replace(base, update=update)


def test_resume_after_interruption_matches_uninterrupted_run(tmp_path):
    path, *_ = _write_data(tmp_path)
    kwargs = dict(
        groupby="perturbation",
        batch_column="batch",
        statistic_name="std",
        chunk_size=2,         # 7 genes -> 4 gene chunks (2, 2, 2, 1)
        cell_chunk_size=100,  # >= total n_obs (63): one cell chunk, one update() per pair
        force=True,
    )

    reference = cx.batch_process(
        path, _moment_reducer(), output_path=tmp_path / "reference.h5ad", **kwargs,
    )
    reference_values = np.asarray(reference.backed.X[:]).copy()
    reference.close()

    output_path = tmp_path / "resumable.h5ad"
    # 3 groups x 3 batches = 9 update() calls per gene chunk; crash partway
    # through the 3rd gene chunk (after chunks 0 and 1 fully complete).
    with pytest.raises(RuntimeError, match="Reducer failed"):
        cx.batch_process(
            path, _crashing_reducer(crash_after_updates=9 * 2 + 3),
            output_path=output_path, **kwargs,
        )
    checkpoint_path = output_path.with_suffix(".progress.json")
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["last_gene_chunk"] == 1

    resumed = cx.batch_process(
        path, _moment_reducer(), output_path=output_path, resume=True, **kwargs,
    )
    resumed_values = np.asarray(resumed.backed.X[:]).copy()
    resumed.close()

    np.testing.assert_allclose(resumed_values, reference_values)
    assert not checkpoint_path.exists()  # cleaned up on successful completion


def test_resume_falls_back_to_scanning_output_when_checkpoint_corrupted(tmp_path):
    path, *_ = _write_data(tmp_path)
    kwargs = dict(
        groupby="perturbation",
        batch_column="batch",
        statistic_name="std",
        chunk_size=2,
        cell_chunk_size=100,
        force=True,
    )
    reference = cx.batch_process(
        path, _moment_reducer(), output_path=tmp_path / "reference2.h5ad", **kwargs,
    )
    reference_values = np.asarray(reference.backed.X[:]).copy()
    reference.close()

    output_path = tmp_path / "resumable_corrupt.h5ad"
    with pytest.raises(RuntimeError, match="Reducer failed"):
        cx.batch_process(
            path, _crashing_reducer(crash_after_updates=9 * 2 + 3),
            output_path=output_path, **kwargs,
        )
    checkpoint_path = output_path.with_suffix(".progress.json")
    assert checkpoint_path.exists()
    checkpoint_path.write_text("{not valid json")  # simulate corruption

    resumed = cx.batch_process(
        path, _moment_reducer(), output_path=output_path, resume=True, **kwargs,
    )
    resumed_values = np.asarray(resumed.backed.X[:]).copy()
    resumed.close()

    np.testing.assert_allclose(resumed_values, reference_values)


def test_multi_channel_output_populates_layers_and_x(tmp_path):
    path, X, labels, batches = _write_data(tmp_path)

    def initialize(width):
        return {"n": 0, "sum": np.zeros(width), "sumsq": np.zeros(width)}

    def update(state, block):
        state["n"] += block.shape[0]
        state["sum"] += block.sum(axis=0)
        state["sumsq"] += np.square(block).sum(axis=0)

    def compare(group_state, reference_state):
        def _mean_var(state):
            mean = state["sum"] / state["n"]
            var = state["sumsq"] / state["n"] - mean ** 2
            return mean, np.clip(var, 0, None)

        g_mean, g_var = _mean_var(group_state)
        r_mean, r_var = _mean_var(reference_state)
        n_g, n_r = group_state["n"], reference_state["n"]
        weight = n_g * n_r / (n_g + n_r)
        se = np.sqrt(g_var / n_g + r_var / n_r)
        return {
            "mean_diff": cx.BatchStatistic(g_mean - r_mean, weight=weight),
            "se": cx.BatchStatistic(se, weight=weight),
        }

    reducer = cx.BatchReducer(
        initialize, update, finalize=lambda state: state, compare=compare,
        channels=("mean_diff", "se"),
    )
    result = cx.batch_process(
        path, reducer,
        groupby="perturbation", reference="ctrl", batch_column="batch",
        mode="comparison", statistic_name="mean_se",
        perturbations=["A"], chunk_size=3, cell_chunk_size=20,
        output_path=tmp_path / "multi_channel.h5ad", force=True,
    )
    backed = result.backed
    assert set(backed.layers.keys()) & {"mean_diff", "se", "mean_diff_weight_sum", "se_weight_sum"} == {
        "mean_diff", "se", "mean_diff_weight_sum", "se_weight_sum",
    }
    np.testing.assert_allclose(np.asarray(backed.X[:]), np.asarray(backed.layers["mean_diff"][:]))
    assert backed.uns["channels"].tolist() == ["mean_diff", "se"]
    assert np.all(np.asarray(backed.layers["se"][:]) >= 0)
    result.close()


def test_channels_mismatch_raises(tmp_path):
    path, *_ = _write_data(tmp_path)

    # finalize() returns a dict but the reducer never declared `channels`.
    undeclared = cx.BatchReducer(
        lambda w: np.zeros(w), lambda s, b: None, lambda s: {"a": s},
    )
    with pytest.raises(TypeError, match="channels was not set"):
        cx.batch_process(
            path, undeclared, groupby="perturbation", perturbations=["A"],
            batch_column="batch", statistic_name="undeclared", chunk_size=2,
            output_path=tmp_path / "undeclared.h5ad", force=True,
        )

    # `channels` is declared but finalize() returns the wrong keys.
    wrong_keys = cx.BatchReducer(
        lambda w: np.zeros(w), lambda s, b: None, lambda s: {"wrong": s},
        channels=("mean", "se"),
    )
    with pytest.raises(ValueError, match="must have exactly the keys"):
        cx.batch_process(
            path, wrong_keys, groupby="perturbation", perturbations=["A"],
            batch_column="batch", statistic_name="wrong_keys", chunk_size=2,
            output_path=tmp_path / "wrong_keys.h5ad", force=True,
        )


def test_csr_source_warns_but_csc_does_not(tmp_path, caplog):
    import logging

    import crispyx.data as cxd

    path, *_ = _write_data(tmp_path, sparse=True)  # written as CSR by default
    cxd._SLOW_AXIS_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="crispyx.data"):
        result = cx.batch_process(
            path, _moment_reducer(),
            groupby="perturbation", batch_column="batch", statistic_name="std_csr",
            chunk_size=2, cell_chunk_size=20,
            output_path=tmp_path / "csr_result.h5ad", force=True,
        )
    result.close()
    assert any("slower" in r.getMessage() for r in caplog.records)

    caplog.clear()
    cxd._SLOW_AXIS_WARNED.clear()
    csc_result = convert_to_csc(path, output_path=tmp_path / "csc_source.h5ad", verbose=False)
    csc_result.close()
    with caplog.at_level(logging.WARNING, logger="crispyx.data"):
        result = cx.batch_process(
            tmp_path / "csc_source.h5ad", _moment_reducer(),
            groupby="perturbation", batch_column="batch", statistic_name="std_csc",
            chunk_size=2, cell_chunk_size=20,
            output_path=tmp_path / "csc_result.h5ad", force=True,
        )
    result.close()
    assert not any("slower" in r.getMessage() for r in caplog.records)


def test_format_mismatch_policy_convert_matches_native_csc(tmp_path):
    path, *_ = _write_data(tmp_path, sparse=True)
    kwargs = dict(
        groupby="perturbation", batch_column="batch", statistic_name="std",
        chunk_size=2, cell_chunk_size=20, force=True,
    )
    converted = cx.batch_process(
        path, _moment_reducer(), format_mismatch_policy="convert",
        output_path=tmp_path / "via_convert.h5ad", **kwargs,
    )
    converted_values = np.asarray(converted.backed.X[:]).copy()
    converted.close()

    native = cx.batch_process(
        path, _moment_reducer(), format_mismatch_policy="off",
        output_path=tmp_path / "native.h5ad", **kwargs,
    )
    native_values = np.asarray(native.backed.X[:]).copy()
    native.close()

    np.testing.assert_allclose(converted_values, native_values)
