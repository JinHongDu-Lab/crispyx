"""Tests for the generic streaming batch-statistics API."""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


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
