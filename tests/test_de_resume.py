"""Interrupted DE runs resume from their checkpoint and match an uninterrupted run."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


@pytest.fixture
def streaming(monkeypatch):
    """Force the group-batch streaming Wilcoxon path (batches of 2 groups)."""
    monkeypatch.setattr(de, "_should_use_streaming", lambda *a, **k: (True, 0, 0, 2))


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


def _assert_same_result(expected: Path, actual: Path) -> None:
    """Every stored value of the two DE results is bit-identical."""
    with h5py.File(expected, "r") as a, h5py.File(actual, "r") as b:
        keys = ["X"] + [f"layers/{k}" for k in a["layers"]] + [f"var/{k}" for k in a["var"]]
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


CASES = ["t_test", "nb_glm", "wilcoxon", "wilcoxon_stratified", "wilcoxon_streaming"]


@pytest.mark.parametrize("case", CASES)
def test_interrupted_run_resumes_to_the_uninterrupted_result(screen, tmp_path, monkeypatch, request, case):
    if case == "wilcoxon_streaming":
        request.getfixturevalue("streaming")
    method = "wilcoxon" if case == "wilcoxon_streaming" else case
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


def test_nb_glm_resumes_after_all_fits_when_the_final_write_failed(screen, tmp_path, monkeypatch):
    reference = tmp_path / "reference.h5ad"
    _run("nb_glm", screen, reference)
    out = tmp_path / "result.h5ad"
    real_write = ad.AnnData.write

    def failing_write(self, *args, **kwargs):
        raise Interrupted

    monkeypatch.setattr(ad.AnnData, "write", failing_write)
    with pytest.raises(Interrupted):
        _run("nb_glm", screen, out)
    monkeypatch.setattr(ad.AnnData, "write", real_write)

    runs = _record_runs(monkeypatch)
    _run("nb_glm", screen, out, resume=True)
    assert runs[-1].resumed and len(runs[-1].progress["completed"]) == 6
    _assert_same_result(reference, out)


def test_changed_parameters_start_fresh(screen, tmp_path, monkeypatch):
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 2)
        with pytest.raises(Interrupted):
            _run("t_test", screen, out)
    runs = _record_runs(monkeypatch)
    _run("t_test", screen, out, resume=True, min_pct_ctrl=0.2)
    assert not runs[-1].resumed

    reference = tmp_path / "reference.h5ad"
    _run("t_test", screen, reference, min_pct_ctrl=0.2)
    _assert_same_result(reference, out)


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


@pytest.mark.parametrize("case", ["wilcoxon", "wilcoxon_stratified"])
def test_wilcoxon_killed_while_writing_leaves_no_output(screen, tmp_path, monkeypatch, case):
    reference = tmp_path / "reference.h5ad"
    _run(case, screen, reference)
    out = tmp_path / "result.h5ad"
    real_create = de._create_array

    def create(group, name, **kwargs):
        if name == "pvalue_adj":
            raise Interrupted
        return real_create(group, name, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(de, "_create_array", create)
        with pytest.raises(Interrupted):
            _run(case, screen, out)
    # A file with matching uns but missing layers would be loaded as done.
    assert not out.exists()
    assert not list(tmp_path.glob(f".{out.name}.partial"))

    runs = _record_runs(monkeypatch)
    _run(case, screen, out, resume=True)
    assert runs[-1].resumed
    _assert_same_result(reference, out)


def test_streaming_wilcoxon_restarts_when_its_work_file_is_unreadable(screen, tmp_path, monkeypatch, streaming):
    reference = tmp_path / "reference.h5ad"
    _run("wilcoxon", screen, reference)
    out = tmp_path / "result.h5ad"
    with monkeypatch.context() as m:
        _interrupt_after(m, 1)
        with pytest.raises(Interrupted):
            _run("wilcoxon", screen, out)
    (_resume_dir(out) / "result.h5ad").write_bytes(b"killed mid-write")
    _run("wilcoxon", screen, out, resume=True)
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


def test_t_test_result_does_not_map_the_deleted_partial_files(screen, tmp_path):
    result = _run("t_test", screen, tmp_path / "result.h5ad")
    for array in (result.statistics, result.pvalues, result.pvalues_adj, result.logfoldchanges,
                  result.effect_size, result.pts, result.order):
        assert type(array) is np.ndarray and array.base is None


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
