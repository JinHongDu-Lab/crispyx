"""Interrupted DE runs resume from their checkpoint and match an uninterrupted run."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from crispyx import de
from crispyx._checkpoint import ResumableRun


class Interrupted(Exception):
    """Stands in for the process being killed."""


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    """Counts and log-normalized copies of a small synthetic screen."""
    root = tmp_path_factory.mktemp("screen")
    rng = np.random.default_rng(0)
    n_genes = 120
    labels = np.array(["control"] * 200 + [f"P{i}" for i in range(6) for _ in range(60)])
    batch = rng.choice(["b1", "b2"], labels.size)
    mu = rng.gamma(0.8, 2.0, n_genes)
    effect = np.ones((7, n_genes))
    effect[1:, :20] = rng.lognormal(0, 0.8, (6, 20))
    group = np.array([0 if lab == "control" else int(lab[1:]) + 1 for lab in labels])
    counts = rng.negative_binomial(2, 2 / (2 + mu[None] * effect[group])).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels, "batch": batch}, index=[f"c{i}" for i in range(labels.size)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var).write(root / "counts.h5ad")
    norm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    ad.AnnData(sp.csr_matrix(norm), obs=obs, var=var).write(root / "norm.h5ad")
    return root


def _run(method, screen, out: Path, **kw):
    common = dict(perturbation_column="perturbation", control_label="control", verbose=False, output_path=out)
    if method == "t_test":
        return de.t_test(screen / "norm.h5ad", n_jobs=1, checkpoint_interval=1, **common, **kw)
    if method == "nb_glm":
        return de.nb_glm_test(screen / "counts.h5ad", n_jobs=1, checkpoint_interval=1, **common, **kw)
    extra = {"batch_column": "batch"} if method == "wilcoxon_stratified" else {}
    return de.wilcoxon_test(
        screen / "norm.h5ad", chunk_size=25, checkpoint_interval=1, **common, **extra, **kw,
    )


def _interrupt_after(monkeypatch, n_saves: int) -> None:
    """Kill the run right after its ``n_saves``-th checkpoint is on disk."""
    real_save = ResumableRun.save
    calls = {"n": 0}

    def save(self, **progress):
        real_save(self, **progress)
        calls["n"] += 1
        if calls["n"] == n_saves:
            raise Interrupted

    monkeypatch.setattr(ResumableRun, "save", save)


def _record_runs(monkeypatch) -> list[ResumableRun]:
    runs: list[ResumableRun] = []
    real_init = ResumableRun.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        runs.append(self)

    monkeypatch.setattr(ResumableRun, "__init__", init)
    return runs


def _dataset_keys(f: h5py.File, key: str) -> list[str]:
    """``key`` itself, or every dataset under it when it is a group.

    With pandas 3, AnnData writes string columns as ``nullable-string-array``
    groups (``values`` + ``mask``) rather than a single dataset.
    """
    if isinstance(f[key], h5py.Dataset):
        return [key]
    found: list[str] = []

    def collect(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            found.append(f"{key}/{name}")

    f[key].visititems(collect)
    return found


def _assert_same_result(expected: Path, actual: Path) -> None:
    """Every stored value of the two DE results is bit-identical."""
    with h5py.File(expected, "r") as a, h5py.File(actual, "r") as b:
        tops = ["X"] + [f"layers/{k}" for k in a["layers"]] + [f"var/{k}" for k in a["var"]]
        keys = [key for top in tops for key in _dataset_keys(a, top)]
        assert sorted(a["layers"]) == sorted(b["layers"])
        for key in keys:
            x, y = a[key][()], b[key][()]
            assert x.dtype == y.dtype and x.shape == y.shape, key
            if x.dtype.kind == "O":  # variable-length strings
                assert x.tolist() == y.tolist(), key
            else:
                assert np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes(), key


def _resume_dir(out: Path) -> Path:
    return out.with_name(f".{out.name}.resume")


CASES = ["t_test", "nb_glm", "wilcoxon", "wilcoxon_stratified"]


@pytest.mark.parametrize("method", CASES)
def test_interrupted_run_resumes_to_the_uninterrupted_result(screen, tmp_path, monkeypatch, method):
    reference = tmp_path / "reference.h5ad"
    _run(method, screen, reference)

    out = tmp_path / "result.h5ad"
    checkpoint = out.with_suffix(".progress.json")
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run(method, screen, out)
    # Nothing at the output path until the run is complete, so a later call
    # cannot mistake the partial result for a finished one.
    assert not out.exists()
    assert checkpoint.exists() and _resume_dir(out).is_dir()

    runs = _record_runs(monkeypatch)
    result = _run(method, screen, out, resume=True)
    assert runs[-1].resumed, "the run started over instead of resuming"
    _assert_same_result(reference, out)
    assert result.result_path == out
    assert not checkpoint.exists() and not _resume_dir(out).exists()


@pytest.mark.parametrize("method", CASES)
def test_killed_while_finishing_leaves_no_output_and_resumes(screen, tmp_path, monkeypatch, method):
    reference = tmp_path / "reference.h5ad"
    _run(method, screen, reference)
    out = tmp_path / "result.h5ad"

    def failing_stamp(*args, **kwargs):
        raise Interrupted

    with monkeypatch.context() as m:
        m.setattr(de, "write_stamp", failing_stamp)
        with pytest.raises(Interrupted):
            _run(method, screen, out)
    # A file with matching uns but missing values would be loaded as done.
    assert not out.exists()

    runs = _record_runs(monkeypatch)
    _run(method, screen, out, resume=True)
    assert runs[-1].resumed
    if method in ("t_test", "nb_glm"):
        assert len(runs[-1].progress["completed"]) == 6
    _assert_same_result(reference, out)


def test_changed_parameters_start_fresh(screen, tmp_path, monkeypatch):
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run("t_test", screen, out)
    runs = _record_runs(monkeypatch)
    with pytest.warns(UserWarning, match="Not resuming .*argument 'min_pct_ctrl' differs"):
        _run("t_test", screen, out, resume=True, min_pct_ctrl=0.2)
    assert not runs[-1].resumed

    reference = tmp_path / "reference.h5ad"
    _run("t_test", screen, reference, min_pct_ctrl=0.2)
    _assert_same_result(reference, out)


def test_checkpoint_listing_an_argument_left_at_none_resumes(screen, tmp_path, monkeypatch, recwarn):
    """A checkpoint saved by a version with a since-removed ``None``-default
    argument (0.1.7's ``min_pct_both``) is the same call and resumes."""
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run("t_test", screen, out)
    checkpoint = out.with_suffix(".progress.json")
    saved = json.loads(checkpoint.read_text())
    saved["fingerprint"]["params"]["min_pct_both"] = None
    checkpoint.write_text(json.dumps(saved))
    runs = _record_runs(monkeypatch)
    _run("t_test", screen, out, resume=True)
    assert runs[-1].resumed
    assert not [w for w in recwarn if "Not resuming" in str(w.message)]


def test_resume_false_discards_a_stale_checkpoint(screen, tmp_path, monkeypatch):
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run("wilcoxon", screen, out)
    runs = _record_runs(monkeypatch)
    _run("wilcoxon", screen, out)  # resume=False
    assert not runs[-1].resumed
    assert not out.with_suffix(".progress.json").exists() and not _resume_dir(out).exists()


def test_memory_and_chunking_changes_still_resume(screen, tmp_path, monkeypatch):
    # Rerunning with a lower memory limit is the natural response to an OOM
    # kill; it must not throw the finished work away.
    reference = tmp_path / "reference.h5ad"
    _run("t_test", screen, reference)
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run("t_test", screen, out)
    runs = _record_runs(monkeypatch)
    _run("t_test", screen, out, resume=True, memory_limit_gb=1.0, cell_chunk_size=64)
    assert runs[-1].resumed
    # A different cell chunk sums in a different order, so rows fitted after
    # the resume may differ from the reference in the last bit.
    with h5py.File(reference, "r") as a, h5py.File(out, "r") as b:
        for key in ["X"] + [f"layers/{k}" for k in a["layers"]]:
            np.testing.assert_allclose(a[key][()], b[key][()], rtol=1e-5, atol=1e-5, err_msg=key)


@pytest.mark.parametrize("method", CASES)
def test_killed_after_writing_is_reused(screen, tmp_path, monkeypatch, method):
    """The output is stamped before it takes the output's name, so a kill
    while the result is being returned loses no work: the next call reuses
    the file, with its Scanpy structure."""
    out = tmp_path / "result.h5ad"

    def killed(*args, **kwargs):
        raise Interrupted

    with monkeypatch.context() as m:
        m.setattr(de, "RankGenesGroupsResult", killed)
        with pytest.raises(Interrupted):
            _run(method, screen, out, scanpy_format=True)
    assert not out.with_suffix(".progress.json").exists()
    runs = _record_runs(monkeypatch)
    result = _run(method, screen, out, scanpy_format=True)
    assert not runs, "the finished output was recomputed"
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["uns/rank_genes_groups/full/scores"][()], np.asarray(result.statistics))


@pytest.mark.parametrize("method", CASES)
def test_resume_restarts_when_the_work_file_is_unreadable(screen, tmp_path, monkeypatch, method):
    reference = tmp_path / "reference.h5ad"
    _run(method, screen, reference)
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 1)
        with pytest.raises(Interrupted):
            _run(method, screen, out)
    (_resume_dir(out) / "result.h5ad").write_bytes(b"killed mid-write")
    runs = _record_runs(monkeypatch)
    _run(method, screen, out, resume=True)
    assert not runs[-1].resumed
    _assert_same_result(reference, out)


def test_t_test_checkpoints_once_a_batch_crosses_the_interval(screen, tmp_path, monkeypatch):
    # Batches of 4 never land on a multiple of 3 before the last one.
    saved: list[int] = []
    real_save = ResumableRun.save

    def save(self, **progress):
        saved.append(len(progress["completed"]))
        real_save(self, **progress)

    monkeypatch.setattr(ResumableRun, "save", save)
    monkeypatch.setattr(de, "_resolve_n_jobs", lambda n_jobs: 4)
    de.t_test(
        screen / "norm.h5ad", perturbation_column="perturbation", control_label="control",
        verbose=False, output_path=tmp_path / "result.h5ad", checkpoint_interval=3,
    )
    assert saved[0] == 4


def test_t_test_failed_perturbation_is_nan_not_significant(screen, tmp_path):
    out = tmp_path / "result.h5ad"
    result = _run("t_test", screen, out, perturbations=["P0", "absent"])
    row = result.groups.index("absent")
    assert np.isnan(result.pvalues[row]).all() and np.isnan(result.pvalues_adj[row]).all()
    assert np.isfinite(result.pvalues[1 - row]).any()


@pytest.mark.parametrize("method", ["t_test", "nb_glm", "wilcoxon"])
def test_result_reads_its_rows_from_the_output_file(screen, tmp_path, method):
    out = tmp_path / "result.h5ad"
    result = _run(method, screen, out)
    with h5py.File(out, "r") as f:
        z = f["layers/z_score"][()]
    np.testing.assert_array_equal(result.statistics[2], z[2])
    np.testing.assert_array_equal(np.asarray(result.statistics), z)
    np.testing.assert_array_equal(result.order[2], np.argsort(-np.where(np.isfinite(z[2]), np.abs(z[2]), -np.inf), kind="mergesort"))
    label = result.groups[2]
    np.testing.assert_array_equal(result[label].statistic, z[2])
    assert (result.u_statistics is None) is (method != "wilcoxon")


def test_t_test_rejects_a_bad_corr_method_before_reading(screen, tmp_path, monkeypatch):
    def read_backed(*args, **kwargs):
        raise AssertionError("read the data before validating corr_method")

    monkeypatch.setattr(de, "read_backed", read_backed)
    with pytest.raises(ValueError, match="corr_method"):
        _run("t_test", screen, tmp_path / "result.h5ad", corr_method="bh")


def test_fingerprint_tells_array_arguments_apart(screen):
    from crispyx._checkpoint import run_fingerprint

    sf = pd.Series(np.linspace(0.5, 2.0, 1000))
    nudged = sf.copy()
    nudged.iloc[500] += 1e-9  # hidden by a truncated, rounded repr
    fp = lambda v: run_fingerprint(screen / "norm.h5ad", size_factors=v)
    assert fp(sf) == fp(sf.copy())
    assert fp(sf) != fp(nudged)
    assert fp(sf.to_numpy()) != fp(nudged.to_numpy())
    assert len(str(fp(np.ones(2_000_000)))) < 1000  # a digest, not the vector


@pytest.mark.parametrize(
    "n_perturbations, requested, expected",
    [(50, None, 1), (500, None, 10), (5000, None, 50), (500, 3, 3), (500, 0, 1)],
)
def test_checkpoint_interval(n_perturbations, requested, expected):
    from crispyx._checkpoint import _get_checkpoint_interval

    assert _get_checkpoint_interval(n_perturbations, requested) == expected


@pytest.mark.parametrize("in_memory", [True, False], ids=["small-in-memory", "large-from-file"])
def test_result_reads_a_replaced_output_afresh(screen, tmp_path, monkeypatch, in_memory):
    """Whether a result keeps its matrices in memory (small) or reads the
    file per access (large), it reports the file now at its path."""
    monkeypatch.setattr(de, "_IN_MEMORY_LAYER_FRACTION", 1.0 if in_memory else 0.0)
    out = tmp_path / "result.h5ad"
    first = _run("t_test", screen, out)
    before = first.pvalues[0]
    _run("t_test", screen, out, perturbations=["P3", "P4"])  # rewrites out, another row 0
    second = de.RankGenesGroupsResult(out)
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(second.pvalues[0], f["layers/pvalue"][0])
        np.testing.assert_array_equal(first.pvalues[0], f["layers/pvalue"][0])
    assert not np.array_equal(before, first.pvalues[0], equal_nan=True)


@pytest.mark.parametrize("in_memory", [True, False], ids=["small-in-memory", "large-from-file"])
def test_a_used_result_leaves_its_file_writable(screen, tmp_path, monkeypatch, in_memory):
    """A result holds no handle on its file between reads, so the file can
    be opened for writing (HDF5 refuses while the process holds it open)."""
    monkeypatch.setattr(de, "_IN_MEMORY_LAYER_FRACTION", 1.0 if in_memory else 0.0)
    out = tmp_path / "result.h5ad"
    result = _run("t_test", screen, out)
    _ = result.pvalues[0], result[result.groups[1]].statistic, np.asarray(result.order)
    with h5py.File(out, "r+") as f:
        f["uns"].attrs["note"] = "edited"


@pytest.mark.parametrize("method", ["wilcoxon", "wilcoxon_stratified"])
def test_wilcoxon_writes_column_blocks_and_resumes_in_them(screen, tmp_path, monkeypatch, method):
    """The result is chunked in column blocks matching the gene-chunk writes,
    with and without page eviction; the values do not depend on the chunk
    shape, and a run resumes in the same layout."""
    reference = tmp_path / "reference.h5ad"
    _run(method, screen, reference)
    with h5py.File(reference, "r") as f:
        assert f["X"].chunks[1] == 25  # the gene chunk width _run passes

    from crispyx import _memory

    monkeypatch.setattr(_memory, "_page_cache_is_charged", lambda: True)  # evict, as in a Slurm job
    monkeypatch.setattr(de, "_COLUMN_BLOCK_CHUNK_BYTES", 25 * 8 * 4)  # 4 rows x 25 genes
    out = tmp_path / "small_chunks.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run(method, screen, out)
    runs = _record_runs(monkeypatch)
    _run(method, screen, out, resume=True)
    assert runs[-1].resumed
    with h5py.File(out, "r") as f:
        assert f["X"].chunks == (4, 25) and f["layers/pvalue"].chunks == (4, 25)
    _assert_same_result(reference, out)


def test_row_blocks_cover_whole_chunk_bands():
    blocks = list(de._row_blocks(1000, 1 << 18, chunk_rows=7))  # 16 rows a block before rounding
    assert all(b.start % 7 == 0 for b in blocks)
    assert blocks[0].stop - blocks[0].start == 14
    assert blocks[-1].stop == 1000 and sum(b.stop - b.start for b in blocks) == 1000
