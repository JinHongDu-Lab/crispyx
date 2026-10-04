"""A finished output is reused only when it was computed from the same inputs.

Every reusable output carries ``uns["crispyx"]`` (writer version, layout
schema and the call's fingerprint); a later call reuses it only
when the fingerprint and schema match. Regression tests for results returned
stale after the input was relabelled in place, an argument changed, or the
file was written by a crispyx with another layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx._provenance import SCHEMAS, read_stamp
from crispyx.data import sort_by_perturbation


def _write_counts(path: Path, labels: np.ndarray, *, log_normalise: bool, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n_genes = 12
    counts = rng.poisson(4, (labels.size, n_genes)).astype(np.float32)
    # A real effect in group A, so relabelling changes the answer.
    counts[labels == "A", :4] *= 3
    X = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4) if log_normalise else counts
    adata = ad.AnnData(
        sp.csr_matrix(X.astype(np.float32)),
        obs=pd.DataFrame(
            {"perturbation": labels, "batch": np.where(np.arange(labels.size) % 2, "b1", "b2")},
            index=[f"c{i}" for i in range(labels.size)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.write(path)
    return path


def _labels() -> np.ndarray:
    return np.array(["ctrl"] * 40 + ["A"] * 20 + ["B"] * 20)


def _rewrite_labels(path: Path, labels: np.ndarray) -> None:
    """Relabel the cells of ``path`` in place, as a permutation null does."""
    adata = ad.read_h5ad(path)
    adata.obs["perturbation"] = labels
    adata.write(path)
    # Guarantee a new mtime even on filesystems with coarse timestamps.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


_DE_METHODS = {
    "wilcoxon": (cx.wilcoxon_test, True, dict(min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, min_mean_pert=0.0)),
    "t_test": (cx.t_test, True, dict(min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, min_mean_pert=0.0)),
    "nb_glm": (cx.nb_glm_test, False, dict(min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0, min_mean_pert=0.0, n_jobs=1)),
}


def _run(method: str, path: Path, output: Path, **overrides):
    fn, _, defaults = _DE_METHODS[method]
    return fn(
        path, perturbation_column="perturbation", control_label="ctrl",
        output_path=output, verbose=True, **{**defaults, **overrides},
    )


@pytest.fixture(params=sorted(_DE_METHODS))
def de_setup(request, tmp_path):
    method = request.param
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=_DE_METHODS[method][1])
    return method, path, tmp_path / "result.h5ad"


_RESULT_FIELDS = ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "effect_size", "pts", "pts_rest")


def test_identical_rerun_reuses_the_result(de_setup, capsys):
    method, path, output = de_setup
    first = _run(method, path, output)
    mtime = output.stat().st_mtime_ns
    capsys.readouterr()
    second = _run(method, path, output)
    out = capsys.readouterr().out
    assert "Loading existing result" in out
    assert "force=True" in out
    assert output.stat().st_mtime_ns == mtime
    # The reloaded result is the run's result, field for field.
    for field in _RESULT_FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(second, field)), np.asarray(getattr(first, field)), err_msg=field
        )
    assert second.groups == first.groups
    assert second.method == first.method


def test_force_recomputes(de_setup, capsys):
    method, path, output = de_setup
    _run(method, path, output)
    os.utime(output, ns=(0, 0))
    capsys.readouterr()
    _run(method, path, output, force=True)
    assert "Loading existing result" not in capsys.readouterr().out
    assert output.stat().st_mtime_ns != 0


def test_relabelled_source_is_recomputed(de_setup, capsys):
    method, path, output = de_setup
    first = _run(method, path, output)
    _rewrite_labels(path, np.random.default_rng(1).permutation(_labels()))
    capsys.readouterr()
    second = _run(method, path, output)
    out = capsys.readouterr().out
    assert "is not reused: the input file has changed since" in out
    assert not np.allclose(second.pvalues, first.pvalues, equal_nan=True)


@pytest.mark.parametrize(
    "change",
    [dict(min_pct_ctrl=0.5), dict(perturbations=["A"])],
    ids=["filter", "perturbations"],
)
def test_changed_argument_is_recomputed(de_setup, capsys, change):
    method, path, output = de_setup
    _run(method, path, output)
    capsys.readouterr()
    second = _run(method, path, output, **change)
    name = next(iter(change))
    assert f"is not reused: argument '{name}' differs" in capsys.readouterr().out
    if name == "perturbations":
        assert second.groups == ["A"]


def test_operational_arguments_do_not_invalidate(tmp_path, capsys):
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "result.h5ad"
    _run("wilcoxon", path, output)
    capsys.readouterr()
    _run("wilcoxon", path, output, chunk_size=3, memory_limit_gb=1.0)
    assert "Loading existing result" in capsys.readouterr().out


def test_stamp_records_writer_and_inputs(tmp_path):
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "result.h5ad"
    _run("wilcoxon", path, output)
    stamp = ad.read_h5ad(output).uns["crispyx"]
    assert stamp["version"] == cx.__version__
    assert stamp["kind"] == "de_result"
    assert stamp["schema"] == SCHEMAS["de_result"]
    fingerprint = json.loads(stamp["fingerprint"])
    assert fingerprint["source"] == str(path.resolve())
    assert fingerprint["params"]["min_pct_ctrl"] == 0.0
    # How the run was carried out is not part of what it computed.
    assert "chunk_size" not in fingerprint["params"]
    assert "verbose" not in fingerprint["params"]


def test_any_spelling_of_the_input_path_reuses_the_result(tmp_path, monkeypatch, capsys):
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "result.h5ad"
    monkeypatch.chdir(tmp_path)
    _run("wilcoxon", Path("data.h5ad"), output)
    capsys.readouterr()
    _run("wilcoxon", path.resolve(), output)
    assert "Loading existing result" in capsys.readouterr().out


def test_scanpy_format_counts_for_the_result_not_the_checkpoint(tmp_path, capsys):
    from crispyx.de import _checkpoint_fingerprint

    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "result.h5ad"
    _run("t_test", path, output)
    capsys.readouterr()
    _run("t_test", path, output, scanpy_format=True)
    assert "argument 'scanpy_format' differs" in capsys.readouterr().out

    stamped = json.loads(read_stamp(output)["fingerprint"])
    unformatted = {**stamped, "params": {**stamped["params"], "scanpy_format": False}}
    assert _checkpoint_fingerprint(stamped) == _checkpoint_fingerprint(unformatted)


def _delete_uns_key(path: Path, key: str) -> None:
    with h5py.File(path, "r+") as handle:
        del handle[f"uns/{key}"]


def test_file_without_provenance_is_recomputed(tmp_path, capsys):
    """A file written by crispyx <= 0.1.6 (no stamp) is recomputed, not read
    under a layout it may not have."""
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "result.h5ad"
    _run("wilcoxon", path, output)
    _delete_uns_key(output, "crispyx")
    capsys.readouterr()
    _run("wilcoxon", path, output)
    assert "is not reused: it has no crispyx provenance" in capsys.readouterr().out
    assert read_stamp(output) is not None


def test_file_with_another_schema_is_recomputed(tmp_path, capsys):
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "result.h5ad"
    _run("t_test", path, output)
    with h5py.File(output, "r+") as handle:
        handle["uns/crispyx/schema"][()] = SCHEMAS["de_result"] + 1
    capsys.readouterr()
    _run("t_test", path, output)
    out = capsys.readouterr().out
    assert f"de_result layout {SCHEMAS['de_result'] + 1}" in out
    assert read_stamp(output)["schema"] == SCHEMAS["de_result"]


@pytest.mark.parametrize("scanpy_format", [False, True])
def test_one_group_result_round_trips(de_setup, scanpy_format):
    """A single-perturbation result (every case-control comparison) reads
    back with anndata and through the reuse path."""
    method, path, output = de_setup
    first = _run(method, path, output, perturbations=["A"], scanpy_format=scanpy_format)
    on_disk = ad.read_h5ad(output)
    assert on_disk.obs_names.tolist() == ["A"]
    assert on_disk.n_obs == 1
    reloaded = _run(method, path, output, perturbations=["A"], scanpy_format=scanpy_format)
    assert reloaded.groups == ["A"]
    np.testing.assert_array_equal(reloaded.pvalues, first.pvalues)


def test_pseudobulk_reuse_reads_no_matrix(tmp_path, monkeypatch, capsys):
    """A matching result is found before any pass over the input matrix."""
    import crispyx.pseudobulk as pb

    counts = _write_counts(tmp_path / "counts.h5ad", _labels(), log_normalise=False)
    logs = _write_counts(tmp_path / "logs.h5ad", _labels(), log_normalise=True)
    aggregate = dict(groupby=["perturbation", "batch"], method="sum", min_cells=1,
                     output_path=tmp_path / "pb.h5ad")
    effects = dict(perturbation_column="perturbation", batch_column="batch",
                   control_label="ctrl", min_cells=1, output_path=tmp_path / "effects.h5ad")
    cx.aggregate_pseudobulk(counts, **aggregate).close()
    cx.compute_pseudobulk_effects(logs, **effects).close()

    def unexpected(*args, **kwargs):
        raise AssertionError("streamed the input instead of reusing the result")

    monkeypatch.setattr(pb, "read_backed", unexpected)
    monkeypatch.setattr(pb, "aggregate_pseudobulk", unexpected)
    capsys.readouterr()
    cx.aggregate_pseudobulk(counts, **aggregate).close()
    cx.compute_pseudobulk_effects(logs, **effects).close()
    assert capsys.readouterr().out.count("Loading existing result") == 2


def test_aggregate_pseudobulk_tracks_its_inputs(tmp_path, capsys):
    path = _write_counts(tmp_path / "counts.h5ad", _labels(), log_normalise=False)
    output = tmp_path / "pb.h5ad"
    kwargs = dict(groupby=["perturbation", "batch"], method="sum", min_cells=1, output_path=output)

    first = cx.aggregate_pseudobulk(path, **kwargs)
    n_first = first.backed.n_obs
    first.close()
    capsys.readouterr()

    cx.aggregate_pseudobulk(path, **kwargs).close()
    assert "Loading existing result" in capsys.readouterr().out

    # perturbations= used to be absent from the cache key.
    subset = cx.aggregate_pseudobulk(path, perturbations=["A"], **kwargs)
    assert "argument 'perturbations' differs" in capsys.readouterr().out
    assert subset.backed.n_obs < n_first
    subset.close()

    _rewrite_labels(path, np.random.default_rng(2).permutation(_labels()))
    cx.aggregate_pseudobulk(path, perturbations=["A"], **kwargs).close()
    assert "the input file has changed since" in capsys.readouterr().out
    assert read_stamp(output)["kind"] == "pseudobulk"


def test_pseudobulk_effects_track_aggregation_arguments(tmp_path, capsys):
    path = _write_counts(tmp_path / "logs.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "effects.h5ad"
    kwargs = dict(
        perturbation_column="perturbation", batch_column="batch", control_label="ctrl",
        output_path=output,
    )
    cx.compute_pseudobulk_effects(path, min_cells=1, **kwargs).close()
    capsys.readouterr()
    cx.compute_pseudobulk_effects(path, min_cells=1, **kwargs).close()
    assert "Loading existing result" in capsys.readouterr().out
    cx.compute_pseudobulk_effects(path, min_cells=2, **kwargs).close()
    assert "argument 'min_cells' differs" in capsys.readouterr().out


def test_sorted_copy_of_an_older_source_is_not_reused(tmp_path):
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=False)
    sorted_path = tmp_path / "data_sorted.h5ad"
    sort_by_perturbation(path, "perturbation", "ctrl", output_path=sorted_path)
    mtime = sorted_path.stat().st_mtime_ns
    sort_by_perturbation(path, "perturbation", "ctrl", output_path=sorted_path)
    assert sorted_path.stat().st_mtime_ns == mtime

    relabelled = np.random.default_rng(3).permutation(_labels())
    _rewrite_labels(path, relabelled)
    sort_by_perturbation(path, "perturbation", "ctrl", output_path=sorted_path)
    resorted = ad.read_h5ad(sorted_path)
    source = ad.read_h5ad(path)
    # The sorted copy now holds the relabelled cells.
    pd.testing.assert_series_equal(
        resorted.obs["perturbation"].astype(str).sort_index(),
        source.obs["perturbation"].astype(str).sort_index(),
    )
    assert read_stamp(sorted_path)["kind"] == "sorted"


def test_batch_process_reuses_a_finished_result_across_chunk_widths(tmp_path, capsys):
    path = _write_counts(tmp_path / "data.h5ad", _labels(), log_normalise=True)
    output = tmp_path / "batch.h5ad"

    def reducer() -> cx.BatchReducer:
        def initialize(width):
            return {"sum": np.zeros(width), "n": 0}

        def update(state, block):
            state["sum"] += np.asarray(block, dtype=np.float64).sum(axis=0)
            state["n"] += block.shape[0]

        def finalize(state):
            return cx.BatchStatistic(state["sum"] / max(state["n"], 1), state["n"])

        return cx.BatchReducer(initialize, update, finalize)

    kwargs = dict(
        groupby="perturbation", batch_column="batch", statistic_name="mean",
        output_path=output,
    )
    cx.batch_process(path, reducer(), chunk_size=4, **kwargs).close()
    assert read_stamp(output)["kind"] == "batch"
    capsys.readouterr()
    cx.batch_process(path, reducer(), chunk_size=5, **kwargs).close()
    assert "Loading existing result" in capsys.readouterr().out

    # A rejected finished output is recomputed, not "resumed" with every
    # chunk already done.
    _delete_uns_key(output, "crispyx")
    with pytest.warns(UserWarning, match="restarting from scratch"):
        cx.batch_process(path, reducer(), chunk_size=4, resume=True, **kwargs).close()
    assert "it has no crispyx provenance" in capsys.readouterr().out
    assert read_stamp(output)["kind"] == "batch"

    _rewrite_labels(path, np.random.default_rng(4).permutation(_labels()))
    with pytest.warns(UserWarning, match="cannot be reused"):
        cx.batch_process(path, reducer(), chunk_size=5, **kwargs).close()
    assert "the input file has changed since" in capsys.readouterr().out
