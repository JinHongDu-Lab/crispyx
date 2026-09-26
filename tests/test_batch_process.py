"""Tests for the generic streaming batch-statistics API."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx._checkpoint import (
    _find_last_completed_gene_chunk,
    _pack_bool_matrix,
    _unpack_bool_matrix,
)
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
        # Blocks arrive in the stored matrix's dtype and their shape is an
        # implementation detail, so accumulate in float64 -- see BatchReducer.
        block = np.asarray(block, dtype=np.float64)
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


def test_resume_skips_already_completed_gene_chunks(tmp_path):
    """A resume must not redo work for chunks the checkpoint says are done.

    Regression test for a bug where `_read_checkpoint` required DE's
    "completed"/"total" schema keys, which `batch_process`'s checkpoint
    never has -- every resume silently fell through to a full restart. A
    full run of this dataset calls update() 9 (group, batch) pairs x 4 gene
    chunks = 36 times; only the last 2 chunks (18 calls) should remain
    after chunks 0-1 are already checkpointed as complete.
    """
    path, *_ = _write_data(tmp_path)
    kwargs = dict(
        groupby="perturbation",
        batch_column="batch",
        statistic_name="std",
        chunk_size=2,
        cell_chunk_size=100,
        force=True,
    )
    output_path = tmp_path / "resumable_count.h5ad"
    with pytest.raises(RuntimeError, match="Reducer failed"):
        cx.batch_process(
            path, _crashing_reducer(crash_after_updates=9 * 2 + 3),
            output_path=output_path, **kwargs,
        )
    checkpoint_path = output_path.with_suffix(".progress.json")
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["last_gene_chunk"] == 1

    base = _moment_reducer()
    counter = {"n": 0}

    def counting_update(state, block):
        counter["n"] += 1
        return base.update(state, block)

    cx.batch_process(
        path, replace(base, update=counting_update),
        output_path=output_path, resume=True, **kwargs,
    )
    assert counter["n"] == 9 * 2


def test_find_last_completed_gene_chunk_requires_all_groups_written(tmp_path):
    """A chunk is only "complete" once every group's row was written.

    Regression test: the scan previously used np.any() over the whole
    (group, gene) block, so a crash mid-way through the per-group write
    loop for a chunk (some groups written, some not) was misclassified as
    a fully-completed chunk.
    """
    h5ad_path = tmp_path / "partial.h5ad"
    n_groups, n_genes, chunk_size = 5, 8, 2
    with h5py.File(h5ad_path, "w") as f:
        ds = f.create_dataset(
            "layers/weight_sum", shape=(n_groups, n_genes), dtype="float64", fillvalue=0.0,
        )
        ds[:, 0:4] = 1.0  # chunks 0 and 1: every group written
        ds[0:3, 4:6] = 1.0  # chunk 2: only groups 0-2 written, 3-4 still zero

    last = _find_last_completed_gene_chunk(
        h5ad_path, n_gene_chunks=4, chunk_size=chunk_size, n_genes=n_genes,
    )
    assert last == 1


def test_resume_with_different_chunk_size_restarts_instead_of_dropping_genes(tmp_path):
    """A chunk_size change across a resume must be detected, not silently
    misalign gene-chunk boundaries against what's already on disk.

    Regression test: `expected_metadata` previously omitted chunk_size, so
    resuming with a different chunk_size than the interrupted run went
    undetected and could permanently leave genes as NaN.
    """
    path, *_ = _write_data(tmp_path)
    common = dict(
        groupby="perturbation", batch_column="batch", statistic_name="std",
        cell_chunk_size=100, force=True,
    )
    reference = cx.batch_process(
        path, _moment_reducer(), output_path=tmp_path / "reference_cs.h5ad",
        chunk_size=3, **common,
    )
    reference_values = np.asarray(reference.backed.X[:]).copy()
    reference.close()

    output_path = tmp_path / "resumable_chunksize.h5ad"
    with pytest.raises(RuntimeError, match="Reducer failed"):
        cx.batch_process(
            path, _crashing_reducer(crash_after_updates=9 * 2 + 3),
            output_path=output_path, chunk_size=2, **common,
        )
    checkpoint_path = output_path.with_suffix(".progress.json")
    assert checkpoint_path.exists()

    with pytest.warns(UserWarning, match="does not match this call's parameters"):
        resumed = cx.batch_process(
            path, _moment_reducer(), output_path=output_path, resume=True,
            chunk_size=3, **{**common, "force": False},
        )
    resumed_values = np.asarray(resumed.backed.X[:]).copy()
    resumed.close()
    np.testing.assert_allclose(resumed_values, reference_values)


def test_resume_with_nothing_left_preserves_n_batches_used(tmp_path):
    """Resuming a run that's already fully complete must not zero out
    obs['n_batches_used'].

    Regression test: `batches_used` was always reinitialized to all-False
    and only set inside the per-chunk loop, which resume skips entirely
    for already-completed chunks -- a resume that finds nothing left to
    process silently overwrote correct historical counts with zeros.
    """
    path, *_ = _write_data(tmp_path)
    kwargs = dict(
        groupby="perturbation", batch_column="batch", statistic_name="std",
        chunk_size=2, cell_chunk_size=100, force=True,
    )
    output_path = tmp_path / "resumable_done.h5ad"
    result = cx.batch_process(path, _moment_reducer(), output_path=output_path, **kwargs)
    original_n_batches_used = np.asarray(result.backed.obs["n_batches_used"]).copy()
    result.close()
    assert (original_n_batches_used > 0).all()

    checkpoint_path = output_path.with_suffix(".progress.json")
    checkpoint_path.write_text(json.dumps({
        "total_gene_chunks": 4,
        "last_gene_chunk": 3,
        "batches_used": _pack_bool_matrix(np.ones((3, 3), dtype=bool)),
        "method": "batch_process",
        "statistic_name": "std",
        "mode": "group",
    }))

    resumed = cx.batch_process(path, _moment_reducer(), output_path=output_path, resume=True, **kwargs)
    resumed_n_batches_used = np.asarray(resumed.backed.obs["n_batches_used"]).copy()
    resumed.close()
    np.testing.assert_array_equal(resumed_n_batches_used, original_n_batches_used)


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
    # The first channel is X; it is not duplicated as a layer. AnnData 0.13
    # exposes X itself as layers[None], so only named layers are compared.
    layer_keys = {key for key in backed.layers.keys() if key is not None}
    assert layer_keys == {"se", "mean_diff_weight_sum", "se_weight_sum"}
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


def test_csr_source_warns_under_warn_policy_but_csc_does_not(tmp_path, caplog):
    import logging

    import crispyx.data as cxd

    path, *_ = _write_data(tmp_path, sparse=True)  # written as CSR by default
    cxd._SLOW_AXIS_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="crispyx.data"), pytest.warns(
        UserWarning, match=r"tl\.batch_process: X is stored as CSR.*re-reads the whole matrix"
    ):
        result = cx.batch_process(
            path, _moment_reducer(),
            groupby="perturbation", batch_column="batch", statistic_name="std_csr",
            chunk_size=2, cell_chunk_size=20, format_mismatch_policy="warn",
            output_path=tmp_path / "csr_result.h5ad", force=True,
        )
    result.close()
    assert len([r for r in caplog.records if "slower" in r.getMessage()]) == 1

    caplog.clear()
    cxd._SLOW_AXIS_WARNED.clear()
    csc_result = convert_to_csc(path, output_path=tmp_path / "csc_source.h5ad", verbose=False)
    csc_result.close()
    with caplog.at_level(logging.WARNING, logger="crispyx.data"), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = cx.batch_process(
            tmp_path / "csc_source.h5ad", _moment_reducer(),
            groupby="perturbation", batch_column="batch", statistic_name="std_csc",
            chunk_size=2, cell_chunk_size=20, format_mismatch_policy="warn",
            output_path=tmp_path / "csc_result.h5ad", force=True,
        )
    result.close()
    assert not any("slower" in r.getMessage() for r in caplog.records)


def test_convert_policy_converts_csr_source_and_removes_scratch_copy(tmp_path, caplog):
    import logging

    import crispyx.data as cxd

    path, *_ = _write_data(tmp_path, sparse=True)  # CSR
    out_dir = tmp_path / "out"
    cxd._SLOW_AXIS_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="crispyx.data"), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = cx.batch_process(
            path, _moment_reducer(),
            groupby="perturbation", batch_column="batch", statistic_name="std",
            chunk_size=2, cell_chunk_size=20, format_mismatch_policy="convert",
            output_path=out_dir / "result.h5ad", force=True,
        )
    result.close()
    # No slow-axis access happened, and the temporary CSC copy beside the
    # output is gone again.
    assert not any("slower" in r.getMessage() for r in caplog.records)
    assert sorted(p.name for p in out_dir.iterdir()) == ["result.h5ad"]


def test_default_auto_policy_streams_a_small_csr_source_without_converting(
    tmp_path, monkeypatch
):
    """On a file this small the re-reads cost microseconds, so "auto" must not
    pay for a conversion -- and must not warn about a cost that isn't there."""
    import crispyx.data as cxd

    path, *_ = _write_data(tmp_path, sparse=True)  # CSR
    out_dir = tmp_path / "out"
    calls: list[object] = []
    real_convert = cxd.convert_to_csc
    monkeypatch.setattr(
        cxd, "convert_to_csc",
        lambda *a, **k: (calls.append(a), real_convert(*a, **k))[1],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = cx.batch_process(
            path, _moment_reducer(),
            groupby="perturbation", batch_column="batch", statistic_name="std",
            chunk_size=2, cell_chunk_size=20,
            output_path=out_dir / "result.h5ad", force=True,
        )
    result.close()
    assert calls == []
    assert sorted(p.name for p in out_dir.iterdir()) == ["result.h5ad"]


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


def test_cell_order_and_slab_boundaries_do_not_change_results(tmp_path):
    """Cells are sorted by (group, batch) internally, so a shuffled source and a
    cell_chunk_size that splits segments across slabs must give the same
    statistics, weights and n_batches_used as the grouped source."""
    path, X, labels, batches = _write_data(tmp_path, sparse=True)
    rng = np.random.default_rng(5)
    perm = rng.permutation(X.shape[0])
    shuffled = tmp_path / "shuffled.h5ad"
    ad.AnnData(
        sp.csr_matrix(X[perm]),
        obs=pd.DataFrame(
            {"perturbation": labels[perm], "batch": batches[perm]},
            index=[f"cell_{i}" for i in perm],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])]),
    ).write(shuffled)

    def run(src, name, cell_chunk_size):
        result = cx.batch_process(
            src, _moment_reducer(), groupby="perturbation", reference="ctrl",
            batch_column="batch", mode="comparison", statistic_name="delta",
            chunk_size=3, cell_chunk_size=cell_chunk_size,
            output_path=tmp_path / f"{name}_result.h5ad", force=True,
        )
        values, weights, obs, _ = _read_result(result)
        result.close()
        return values, weights, obs["n_batches_used"].tolist()

    grouped = run(path, "grouped", cell_chunk_size=1000)
    split = run(path, "split", cell_chunk_size=4)
    shuffled_split = run(shuffled, "shuffled", cell_chunk_size=5)
    for other in (split, shuffled_split):
        np.testing.assert_allclose(other[0], grouped[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(other[1], grouped[1])
        assert other[2] == grouped[2]

    # Reference check against a direct computation.
    expected = []
    for group in ("A", "B"):
        num = np.zeros(X.shape[1])
        den = np.zeros(X.shape[1])
        for batch in ("b1", "b2", "b3"):
            g = X[(labels == group) & (batches == batch)]
            r = X[(labels == "ctrl") & (batches == batch)]
            w = g.shape[0] * r.shape[0] / (g.shape[0] + r.shape[0])
            num += w * (g.mean(axis=0) - r.mean(axis=0))
            den += w
        expected.append(num / den)
    np.testing.assert_allclose(grouped[0], expected, rtol=1e-12, atol=1e-12)


def test_output_datasets_are_chunked_along_gene_chunks(tmp_path):
    path, X, *_ = _write_data(tmp_path, sparse=True)
    result = cx.batch_process(
        path, _moment_reducer(), groupby="perturbation", batch_column="batch",
        statistic_name="std", chunk_size=3, output_path=tmp_path / "chunked.h5ad", force=True,
    )
    result.close()
    with h5py.File(tmp_path / "chunked.h5ad", "r") as f:
        for name in ("X", "layers/weight_sum"):
            assert f[name].chunks is not None
            assert f[name].chunks[1] == 3


def test_scanned_weight_layer_is_written_last_per_gene_chunk(tmp_path, monkeypatch):
    """The resume fallback scan keys off the first channel's weight layer, so
    that dataset must be the last one written for every gene chunk."""
    import h5py as _h5py

    writes: list[str] = []
    original = _h5py.Dataset.__setitem__

    def recording_setitem(self, key, value):
        if self.file.filename.endswith("order.h5ad"):
            writes.append(self.name)
        return original(self, key, value)

    monkeypatch.setattr(_h5py.Dataset, "__setitem__", recording_setitem)
    path, *_ = _write_data(tmp_path, sparse=True)
    reducer = cx.BatchReducer(
        lambda w: {"n": 0, "s": np.zeros(w)},
        lambda s, b: s.update(n=s["n"] + b.shape[0], s=s["s"] + b.sum(axis=0)),
        lambda s: {
            "mean": cx.BatchStatistic(s["s"] / s["n"], weight=s["n"]),
            "n": np.full(s["s"].shape, float(s["n"])),
        },
        channels=("mean", "n"),
    )
    result = cx.batch_process(
        path, reducer, groupby="perturbation", batch_column="batch",
        statistic_name="ordered", chunk_size=3, output_path=tmp_path / "order.h5ad",
        force=True, format_mismatch_policy="off",
    )
    result.close()
    chunk_writes = [w for w in writes if w in ("/X", "/layers/n",
                                               "/layers/mean_weight_sum", "/layers/n_weight_sum")]
    n_chunks = 3  # 7 genes, chunk_size 3
    assert len(chunk_writes) == 4 * n_chunks
    for i in range(n_chunks):
        per_chunk = chunk_writes[4 * i:4 * (i + 1)]
        assert per_chunk[-1] == "/layers/mean_weight_sum"
        assert set(per_chunk) == {"/X", "/layers/n",
                                  "/layers/mean_weight_sum", "/layers/n_weight_sum"}


def test_auto_chunk_size_is_capped_by_group_accumulators(tmp_path):
    """With n_groups groups the per-chunk accumulators are (n_groups, width)
    float64 arrays; the auto chunk_size must keep them within the budget."""
    path, *_ = _write_data(tmp_path, sparse=True)
    result = cx.batch_process(
        path, _moment_reducer(), groupby="perturbation", batch_column="batch",
        statistic_name="capped", memory_limit_gb=1e-6, cell_chunk_size=20,
        output_path=tmp_path / "capped.h5ad", force=True, format_mismatch_policy="off",
    )
    # 0.15 * 1e3 bytes // (2 arrays * 1 layer * 3 groups * 8 B) == 3 genes.
    assert int(result.backed.uns["chunk_size"]) == 3
    result.close()


def test_off_policy_does_not_silence_slow_axis_warning_for_other_callers(tmp_path, caplog):
    import logging

    import crispyx.data as cxd

    path, *_ = _write_data(tmp_path, sparse=True)  # CSR
    cxd._SLOW_AXIS_WARNED.clear()
    result = cx.batch_process(
        path, _moment_reducer(), groupby="perturbation", batch_column="batch",
        statistic_name="off", chunk_size=2, format_mismatch_policy="off",
        output_path=tmp_path / "off.h5ad", force=True,
    )
    result.close()
    backed = cxd.read_backed(path)
    try:
        with caplog.at_level(logging.WARNING, logger="crispyx.data"):
            for _ in cxd.iter_matrix_chunks(backed, axis=1, chunk_size=2, convert_to_dense=False):
                pass
    finally:
        backed.file.close()
    assert len([r for r in caplog.records if "slower" in r.getMessage()]) == 1


def test_convert_policy_falls_back_to_warn_when_scratch_disk_is_short(tmp_path, monkeypatch):
    import crispyx.data as cxd
    from crispyx._disk import DiskEstimate

    path, *_ = _write_data(tmp_path, sparse=True)  # CSR
    monkeypatch.setattr(
        cxd, "assess_bytes",
        lambda required, p: DiskEstimate(required_bytes=required, free_bytes=1.0, path=Path(p)),
    )
    out_dir = tmp_path / "short"
    with pytest.warns(UserWarning, match=r"streaming from the CSR source instead"):
        result = cx.batch_process(
            path, _moment_reducer(), groupby="perturbation", batch_column="batch",
            statistic_name="fallback", chunk_size=2, cell_chunk_size=20,
            format_mismatch_policy="convert",
            output_path=out_dir / "result.h5ad", force=True,
        )
    values = np.asarray(result.backed.X[:]).copy()
    result.close()
    assert np.isfinite(values).all()
    assert sorted(p.name for p in out_dir.iterdir()) == ["result.h5ad"]


def test_partial_output_of_a_killed_run_is_not_reused_as_a_cached_result(tmp_path):
    """The output file is created, with complete metadata and a NaN fill,
    before the first gene chunk runs. Only the completion marker distinguishes
    a finished result from one whose run was killed -- without that check a
    crashed run's all-NaN file comes back instantly as a "cached result"."""
    path, X, labels, batches = _write_data(tmp_path, sparse=True)
    out = tmp_path / "result.h5ad"
    common = dict(
        groupby="perturbation", batch_column="batch", mode="group",
        statistic_name="std", chunk_size=2, cell_chunk_size=20,
        output_path=out, format_mismatch_policy="off",
    )
    expected, *_ = _read_result(cx.batch_process(path, _moment_reducer(), force=True, **common))

    # Simulate the file a killed run leaves: metadata written, marker absent.
    with h5py.File(out, "r+") as f:
        del f["uns"]["crispyx_run_complete"]
        f["X"][:] = np.nan

    recomputed, *_ = _read_result(cx.batch_process(path, _moment_reducer(), **common))
    assert np.isfinite(recomputed).all()
    np.testing.assert_allclose(recomputed, expected, rtol=1e-12, atol=1e-12)


def test_an_unusable_existing_output_is_overwritten_with_a_warning(tmp_path):
    """A recompute fills the output in place, so it replaces what is there
    before it has a result to put in its place -- and if this run is killed
    too, neither survives. That has to be said out loud: the file may be a
    perfectly good result from a version before the completion marker."""
    path, *_ = _write_data(tmp_path, sparse=True)
    out = tmp_path / "result.h5ad"
    common = dict(
        groupby="perturbation", batch_column="batch", mode="group",
        statistic_name="std", chunk_size=2, cell_chunk_size=20,
        output_path=out, format_mismatch_policy="off",
    )
    cx.batch_process(path, _moment_reducer(), force=True, **common)
    # What an earlier version left behind: a complete result, no marker.
    with h5py.File(out, "r+") as f:
        del f["uns"]["crispyx_run_complete"]

    with pytest.warns(UserWarning, match="overwritten"):
        cx.batch_process(path, _moment_reducer(), **common)

    # force=True is the user asking for the rerun, so it says nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cx.batch_process(path, _moment_reducer(), force=True, **common)


def test_completed_run_is_still_reused_without_recomputing(tmp_path):
    path, *_ = _write_data(tmp_path, sparse=True)
    out = tmp_path / "cached.h5ad"
    common = dict(
        groupby="perturbation", batch_column="batch", statistic_name="std",
        chunk_size=2, cell_chunk_size=20, output_path=out,
        format_mismatch_policy="off",
    )
    first, *_ = _read_result(cx.batch_process(path, _moment_reducer(), force=True, **common))

    def _explode(state, block):  # pragma: no cover - must never be called
        raise AssertionError("recomputed a complete cached result")

    reducer = replace(_moment_reducer(), update=_explode)
    cached, *_ = _read_result(cx.batch_process(path, reducer, **common))
    np.testing.assert_array_equal(cached, first)


def test_float32_source_matches_a_float64_reference(tmp_path):
    """The library must not itself lose precision on a float32 file: with a
    reducer that accumulates in float64 (the documented contract), the result
    matches an in-memory float64 computation whatever the chunking."""
    rng = np.random.default_rng(5)
    n_cells, n_genes = 300, 40
    X = rng.lognormal(size=(n_cells, n_genes)).astype(np.float32)
    labels = np.array(["ctrl", "A", "B"])[rng.integers(0, 3, n_cells)]
    batches = np.array(["b1", "b2"])[rng.integers(0, 2, n_cells)]
    path = tmp_path / "float32.h5ad"
    ad.AnnData(
        sp.csr_matrix(X),
        obs=pd.DataFrame(
            {"perturbation": labels, "batch": batches},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    ).write(path)

    result = cx.batch_process(
        path, _moment_reducer(), groupby="perturbation", batch_column="batch",
        statistic_name="std", chunk_size=7, cell_chunk_size=13,
        output_path=tmp_path / "std.h5ad", force=True, format_mismatch_policy="off",
    )
    actual, _, obs, _ = _read_result(result)

    X64 = X.astype(np.float64)
    expected = []
    for group in obs.index.astype(str):
        values, counts = [], []
        for batch in ("b1", "b2"):
            subset = X64[(labels == group) & (batches == batch)]
            values.append(subset.std(axis=0, ddof=1))
            counts.append(subset.shape[0])
        expected.append(np.average(values, axis=0, weights=counts))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _sum_reducer(weight_maker) -> cx.BatchReducer:
    """Per-(group, batch) mean, with the weight built by ``weight_maker(n, width)``."""

    def initialize(width):
        return {"sum": np.zeros(width), "n": 0}

    def update(state, block):
        block = np.asarray(block, dtype=np.float64)
        state["sum"] += block.sum(axis=0)
        state["n"] += block.shape[0]

    def finalize(state):
        return cx.BatchStatistic(
            state["sum"] / state["n"], weight_maker(state["n"], state["sum"].size)
        )

    return cx.BatchReducer(initialize, update, finalize)


def test_scalar_weight_matches_the_same_weight_broadcast_per_gene(tmp_path):
    """The scalar-weight fast path must be bit-identical to the masked path.

    A scalar weight used to be broadcast to ``(width,)`` and then validated
    and boolean-masked per (group, batch) pair per channel; it is now kept
    scalar and applied unmasked. For a positive weight the mask was
    uniformly true, so the two must agree exactly, not merely closely.
    """
    path, *_ = _write_data(tmp_path)
    common = dict(
        groupby="perturbation", batch_column="batch", chunk_size=3,
        cell_chunk_size=4, force=True,
    )
    scalar = cx.batch_process(
        path, _sum_reducer(lambda n, width: float(n)),
        statistic_name="scalar_w", output_path=tmp_path / "scalar_w.h5ad", **common,
    )
    scalar_values, scalar_weights, scalar_obs, _ = _read_result(scalar)
    scalar.close()

    broadcast = cx.batch_process(
        path, _sum_reducer(lambda n, width: np.full(width, float(n))),
        statistic_name="array_w", output_path=tmp_path / "array_w.h5ad", **common,
    )
    broadcast_values, broadcast_weights, broadcast_obs, _ = _read_result(broadcast)
    broadcast.close()

    np.testing.assert_array_equal(scalar_values, broadcast_values)
    np.testing.assert_array_equal(scalar_weights, broadcast_weights)
    np.testing.assert_array_equal(
        scalar_obs["n_batches_used"].to_numpy(), broadcast_obs["n_batches_used"].to_numpy()
    )


def test_zero_scalar_weight_drops_the_batch_like_an_all_zero_vector(tmp_path):
    """Weight 0 must drop the (group, batch) pair on both weight paths."""
    path, *_ = _write_data(tmp_path)
    common = dict(
        groupby="perturbation", perturbations=["A"], batch_column="batch",
        chunk_size=3, cell_chunk_size=4, force=True,
    )
    # Only batch "b2" carries weight; the other two contribute nothing.
    def scalar_weight(n, width):
        return float(n) if n == 6 else 0.0

    def vector_weight(n, width):
        return np.full(width, float(n) if n == 6 else 0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scalar = cx.batch_process(
            path, _sum_reducer(scalar_weight),
            statistic_name="zero_scalar", output_path=tmp_path / "zero_scalar.h5ad", **common,
        )
        scalar_values, scalar_weights, scalar_obs, _ = _read_result(scalar)
        scalar.close()
        vector = cx.batch_process(
            path, _sum_reducer(vector_weight),
            statistic_name="zero_vector", output_path=tmp_path / "zero_vector.h5ad", **common,
        )
        vector_values, vector_weights, vector_obs, _ = _read_result(vector)
        vector.close()

    np.testing.assert_array_equal(scalar_values, vector_values)
    np.testing.assert_array_equal(scalar_weights, vector_weights)
    # One batch of "A" has 6 cells, so exactly one batch is counted as used.
    assert scalar_obs["n_batches_used"].tolist() == [1]
    assert vector_obs["n_batches_used"].tolist() == [1]


@pytest.mark.parametrize("bad", [np.inf, np.nan, -1.0])
def test_non_finite_or_negative_scalar_weights_are_rejected(tmp_path, bad):
    """The scalar path must reject the same weights the vector path does."""
    path, *_ = _write_data(tmp_path)
    for label, weight_maker in (
        ("scalar", lambda n, width: float(bad)),
        ("vector", lambda n, width: np.full(width, float(bad))),
    ):
        with pytest.raises(ValueError, match="finite and non-negative"):
            cx.batch_process(
                path, _sum_reducer(weight_maker),
                groupby="perturbation", perturbations=["A"], batch_column="batch",
                statistic_name=f"bad_{label}", chunk_size=3,
                output_path=tmp_path / f"bad_{label}.h5ad", force=True,
            )


def test_perturbation_subset_maps_cells_of_excluded_labels_to_no_group(tmp_path):
    """Cells whose label is not in ``perturbations`` must not join any group.

    The cell-to-group mapping is built from the distinct labels rather than
    per cell; a label missing from the requested subset has to keep coding
    as "unusable", the way the per-cell ``dict.get(label, -1)`` did.
    """
    path, X, labels, batches = _write_data(tmp_path)
    subset = cx.batch_process(
        path, _moment_reducer(), groupby="perturbation", perturbations=["A"],
        batch_column="batch", statistic_name="subset", chunk_size=3,
        cell_chunk_size=4, output_path=tmp_path / "subset.h5ad", force=True,
    )
    subset_values, _, subset_obs, _ = _read_result(subset)
    subset.close()
    assert subset_obs.index.tolist() == ["A"]

    # A file holding only the "A" cells must give the same row: if any "ctrl"
    # or "B" cell had leaked into group A, these would differ.
    keep = labels == "A"
    only_a = tmp_path / "only_a.h5ad"
    ad.AnnData(
        sp.csr_matrix(X[keep]),
        obs=pd.DataFrame(
            {"perturbation": labels[keep], "batch": batches[keep]},
            index=[f"cell_{i}" for i in np.flatnonzero(keep)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])]),
    ).write(only_a)
    isolated = cx.batch_process(
        only_a, _moment_reducer(), groupby="perturbation", batch_column="batch",
        statistic_name="isolated", chunk_size=3, cell_chunk_size=4,
        output_path=tmp_path / "isolated.h5ad", force=True,
    )
    isolated_values, _, _, _ = _read_result(isolated)
    isolated.close()
    np.testing.assert_allclose(subset_values, isolated_values)


def test_packed_batches_used_round_trips(tmp_path):
    """The checkpoint's packed bitmap must survive JSON unchanged."""
    rng = np.random.default_rng(3)
    matrix = rng.random((37, 5)) < 0.4
    payload = json.loads(json.dumps(_pack_bool_matrix(matrix)))
    np.testing.assert_array_equal(_unpack_bool_matrix(payload, (37, 5)), matrix)
    # A payload that does not describe this run's grid is refused, not guessed.
    assert _unpack_bool_matrix(payload, (37, 4)) is None
    assert _unpack_bool_matrix(np.argwhere(matrix).tolist(), (37, 5)) is None


def test_untestable_groups_are_found_across_several_weight_slices(tmp_path):
    """The end-of-run weight reduction reads the layer in gene-chunk slices.

    It must report exactly the groups the whole-layer reduction did, with a
    chunk_size small enough that several slices are needed and an untestable
    group sitting alongside testable ones.
    """
    X = np.arange(90, dtype=float).reshape(9, 10)
    obs = pd.DataFrame(
        {
            # "B" shares no batch with the control, so it is untestable;
            # "A" does, so it must survive the same reduction.
            "perturbation": ["ctrl"] * 3 + ["A"] * 3 + ["B"] * 3,
            "batch": ["b1"] * 3 + ["b1"] * 3 + ["b2"] * 3,
        },
        index=[f"c{i}" for i in range(9)],
    )
    path = tmp_path / "mixed.h5ad"
    ad.AnnData(
        X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(10)])
    ).write(path)

    with pytest.warns(UserWarning, match="no usable batch statistics"):
        result = cx.batch_process(
            path, _moment_reducer(), groupby="perturbation", reference="ctrl",
            batch_column="batch", mode="comparison", statistic_name="mixed",
            chunk_size=3, cell_chunk_size=2,
            output_path=tmp_path / "mixed_result.h5ad", force=True,
        )
    values, weights, obs_result, uns = _read_result(result)
    result.close()

    assert uns["stratified_n_untestable_perturbations"] == 1
    row = {label: i for i, label in enumerate(obs_result.index)}
    assert np.isnan(values[row["B"]]).all()
    assert np.equal(weights[row["B"]], 0).all()
    assert not np.isnan(values[row["A"]]).any()
    assert obs_result["n_batches_used"].tolist() == [1, 0]


def test_resume_keeps_the_stored_chunk_size_when_it_was_auto_selected(tmp_path):
    """An auto chunk_size must follow the output being resumed, not the budget.

    ``chunk_size`` is auto-selected from ``memory_limit_gb``, so resuming
    under a different memory allocation would pick different gene-chunk
    boundaries, fail the metadata match, and overwrite the partial output
    the call was asked to continue.
    """
    path, *_ = _write_data(tmp_path)
    common = dict(
        groupby="perturbation", batch_column="batch", statistic_name="std",
        cell_chunk_size=100,
    )
    output_path = tmp_path / "resume_auto_cs.h5ad"
    with pytest.raises(RuntimeError, match="Reducer failed"):
        cx.batch_process(
            path, _crashing_reducer(crash_after_updates=9 * 2 + 3),
            output_path=output_path, chunk_size=2, force=True, **common,
        )
    with h5py.File(output_path, "r") as f:
        assert int(np.asarray(f["uns/chunk_size"][()]).item()) == 2

    # Auto chunk_size under a memory budget that would pick a different
    # width: the run must adopt 2 and continue, with no overwrite warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        resumed = cx.batch_process(
            path, _moment_reducer(), output_path=output_path, resume=True,
            memory_limit_gb=8.0, **common,
        )
    resumed_values = np.asarray(resumed.backed.X[:]).copy()
    resumed_uns = dict(resumed.backed.uns)
    resumed.close()
    assert int(np.asarray(resumed_uns["chunk_size"]).item()) == 2

    reference = cx.batch_process(
        path, _moment_reducer(), output_path=tmp_path / "resume_auto_ref.h5ad",
        chunk_size=2, force=True, **common,
    )
    reference_values = np.asarray(reference.backed.X[:]).copy()
    reference.close()
    np.testing.assert_allclose(resumed_values, reference_values)


def test_resumable_run_warns_that_the_converted_copy_is_rebuilt_each_restart(tmp_path):
    """resume=True plus a converting policy is a per-restart cost, so it warns.

    The temporary fast-axis copy lives only for one call, so a run that needs
    several restarts -- which is what resume exists for -- pays the whole
    conversion again before any new chunk starts.
    """
    path, *_ = _write_data(tmp_path, sparse=True)  # CSR
    common = dict(
        groupby="perturbation", batch_column="batch", chunk_size=2,
        cell_chunk_size=20, format_mismatch_policy="convert",
    )
    with pytest.warns(UserWarning, match="rebuilt from scratch on every resume"):
        result = cx.batch_process(
            path, _moment_reducer(), statistic_name="resumable_csr", resume=True,
            output_path=tmp_path / "resumable_csr.h5ad", **common,
        )
    result.close()

    # Not a resumable run: converting is a one-off, so no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = cx.batch_process(
            path, _moment_reducer(), statistic_name="oneshot_csr", force=True,
            output_path=tmp_path / "oneshot_csr.h5ad", **common,
        )
    result.close()

    # A source already on its fast axis converts nothing, so resume is free.
    convert_to_csc(path, output_path=tmp_path / "csc_src.h5ad", verbose=False).close()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = cx.batch_process(
            tmp_path / "csc_src.h5ad", _moment_reducer(),
            statistic_name="resumable_csc", resume=True,
            output_path=tmp_path / "resumable_csc.h5ad", **common,
        )
    result.close()


def _write_one_batch_per_group(tmp_path: Path) -> Path:
    """Three groups, three batches, each group present in exactly one batch."""
    rng = np.random.default_rng(5)
    labels: list[str] = []
    batches: list[str] = []
    rows: list[np.ndarray] = []
    for index, (group, batch) in enumerate((("A", "b1"), ("B", "b2"), ("C", "b3"))):
        rows.extend(rng.normal(loc=index, scale=0.5, size=(6, 7)))
        labels.extend([group] * 6)
        batches.extend([batch] * 6)
    X = np.asarray(rows, dtype=np.float64)
    obs = pd.DataFrame(
        {"perturbation": labels, "batch": batches},
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])])
    path = tmp_path / "one_batch_per_group.h5ad"
    ad.AnnData(sp.csr_matrix(X), obs=obs, var=var).write(path)
    return path


def test_restart_from_scratch_does_not_inherit_the_stale_batches_used_grid(tmp_path):
    """A resume that restarts from scratch must recount batches, not adopt
    the discarded run's grid.

    Regression test: the checkpoint was read before the metadata check and
    never cleared, so the branch that deletes the checkpoint and recreates
    the output still restored `batches_used` from it -- seeding the fresh
    run with another run's counts and re-persisting them. The grid's shape
    check cannot catch this: it passes whenever the groups and batches are
    unchanged, which is exactly the common mismatch case (a regenerated
    source, or a different chunk_size).
    """
    path = _write_one_batch_per_group(tmp_path)
    common = dict(
        groupby="perturbation", batch_column="batch", statistic_name="std",
        cell_chunk_size=100,
    )
    reference = cx.batch_process(
        path, _moment_reducer(), output_path=tmp_path / "reference_grid.h5ad",
        chunk_size=3, force=True, **common,
    )
    truth = np.asarray(reference.backed.obs["n_batches_used"]).copy()
    reference.close()
    assert truth.tolist() == [1, 1, 1]  # each group is in one batch only

    # An earlier run under a different chunk_size, whose checkpoint claims
    # every group used every batch.
    output_path = tmp_path / "stale_grid.h5ad"
    first = cx.batch_process(
        path, _moment_reducer(), output_path=output_path,
        chunk_size=2, force=True, **common,
    )
    first.close()
    output_path.with_suffix(".progress.json").write_text(json.dumps({
        "total_gene_chunks": 4,
        "last_gene_chunk": 1,
        "batches_used": _pack_bool_matrix(np.ones((3, 3), dtype=bool)),
        "method": "batch_process",
        "statistic_name": "std",
        "mode": "group",
    }))

    with pytest.warns(UserWarning, match="does not match this call's parameters"):
        resumed = cx.batch_process(
            path, _moment_reducer(), output_path=output_path,
            chunk_size=3, resume=True, **common,
        )
    restarted = np.asarray(resumed.backed.obs["n_batches_used"]).copy()
    resumed.close()
    np.testing.assert_array_equal(restarted, truth)

    # ... and the count it writes back into the new checkpoint is the real one.
    reopened = cx.batch_process(
        path, _moment_reducer(), output_path=output_path,
        chunk_size=3, resume=True, **common,
    )
    np.testing.assert_array_equal(
        np.asarray(reopened.backed.obs["n_batches_used"]), truth
    )
    reopened.close()


def test_force_overwrites_an_unreadable_output_instead_of_raising(tmp_path):
    """force=True must not be blocked by the output it is about to replace.

    Regression test: the auto chunk_size path learned to read the stored
    chunk_size out of an existing output whenever resume=True, before the
    force check, so force=True on a truncated file (a run killed during the
    placeholder write) raised out of batch_process instead of overwriting.
    """
    path, *_ = _write_data(tmp_path)
    output_path = tmp_path / "truncated.h5ad"
    output_path.write_bytes(b"\x89HDF\r\n\x1a\n truncated")

    result = cx.batch_process(
        path, _moment_reducer(), output_path=output_path,
        groupby="perturbation", batch_column="batch", statistic_name="std",
        cell_chunk_size=100, force=True, resume=True,
    )
    values = np.asarray(result.backed.X[:]).copy()
    result.close()
    assert np.isfinite(values).all()
