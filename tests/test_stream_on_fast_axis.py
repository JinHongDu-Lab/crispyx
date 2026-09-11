"""Tests for the shared slow-axis streaming helper in ``crispyx.data``.

Covers the ``"auto"`` policy's convert-or-stream decision and the sweep that
reclaims temporary copies left behind by killed runs. The decision depends on
measured read throughput, so the tests pin the measurement rather than build
files big enough to be genuinely slow.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import crispyx.data as cxd
from crispyx.data import (
    _HOST_TAG,
    _SLOW_AXIS_MIN_CHUNKS,
    _SLOW_AXIS_MIN_SAVING_SECONDS,
    _STALE_SCRATCH_SECONDS,
    _measure_read_throughput,
    _sweep_stale_scratch_copies,
    resolve_auto_format_mismatch_policy,
    scratch_copy_bytes,
    stream_on_fast_axis,
)


def _write_csr(path: Path, *, n_obs: int = 40, n_vars: int = 60) -> Path:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32))
    ad.AnnData(X).write(path)
    return path


def _pin_throughput(monkeypatch, value):
    monkeypatch.setattr(cxd, "_measure_read_throughput", lambda path: value)


def _throughput_for(path: Path, *, n_chunks: int, seconds: float) -> float:
    """Throughput at which ``n_chunks`` re-reads of ``path`` cost ``seconds``."""
    _, nbytes = cxd._matrix_shape_and_nbytes(path)
    return (n_chunks - 1) * nbytes / seconds


class TestAutoDecision:
    def test_converts_when_the_re_reads_are_expensive(self, tmp_path, monkeypatch):
        path = _write_csr(tmp_path / "csr.h5ad")
        chunk_size = 10  # 6 gene chunks
        _pin_throughput(
            monkeypatch,
            _throughput_for(path, n_chunks=6, seconds=2 * _SLOW_AXIS_MIN_SAVING_SECONDS),
        )
        resolved = resolve_auto_format_mismatch_policy(path, axis=1, chunk_size=chunk_size)
        assert resolved.policy == "convert"
        assert "converting" in resolved.reason

    def test_streams_when_the_re_reads_are_cheap(self, tmp_path, monkeypatch):
        path = _write_csr(tmp_path / "csr.h5ad")
        _pin_throughput(monkeypatch, 5e9)  # a local SSD / page cache
        resolved = resolve_auto_format_mismatch_policy(path, axis=1, chunk_size=10)
        assert resolved.policy == "off"

    def test_never_converts_below_the_chunk_floor(self, tmp_path, monkeypatch):
        """With too few passes, one conversion cannot repay its own write --
        however slow the filesystem is. The user still hears about the cost."""
        path = _write_csr(tmp_path / "csr.h5ad")
        n_chunks = _SLOW_AXIS_MIN_CHUNKS - 1
        chunk_size = -(-60 // n_chunks)  # ceil, so exactly n_chunks chunks
        _pin_throughput(
            monkeypatch,
            _throughput_for(
                path, n_chunks=n_chunks, seconds=100 * _SLOW_AXIS_MIN_SAVING_SECONDS
            ),
        )
        resolved = resolve_auto_format_mismatch_policy(path, axis=1, chunk_size=chunk_size)
        assert resolved.policy == "warn"
        assert f"fewer than {_SLOW_AXIS_MIN_CHUNKS} chunks" in resolved.reason

    def test_streams_when_throughput_cannot_be_measured(self, tmp_path, monkeypatch):
        path = _write_csr(tmp_path / "csr.h5ad")
        _pin_throughput(monkeypatch, None)
        resolved = resolve_auto_format_mismatch_policy(path, axis=1, chunk_size=10)
        assert resolved.policy == "warn"
        assert "could not measure" in resolved.reason

    def test_probe_measures_a_real_file(self, tmp_path):
        path = _write_csr(tmp_path / "csr.h5ad")
        throughput = _measure_read_throughput(path)
        assert throughput is not None and throughput > 0

    def test_probe_reports_unmeasurable_rather_than_raising(self, tmp_path):
        missing = tmp_path / "nope.h5ad"
        assert _measure_read_throughput(missing) is None

    def test_a_file_is_measured_once_per_process(self, tmp_path, monkeypatch):
        """The probe warms the page cache, so measuring twice measures two
        different things: estimate_disk_usage would report a scratch copy the
        run it describes then declines, purely because the query went first."""
        path = _write_csr(tmp_path / "csr.h5ad")
        first = _measure_read_throughput(path)

        def _boom(_path):  # pragma: no cover - must never run
            raise AssertionError("re-probed a file already measured")

        monkeypatch.setattr(cxd, "_probe_read_throughput", _boom)
        assert _measure_read_throughput(path) == first

    def test_rewriting_the_file_re_measures_it(self, tmp_path, monkeypatch):
        path = _write_csr(tmp_path / "csr.h5ad")
        _measure_read_throughput(path)
        path.unlink()
        _write_csr(path, n_obs=80, n_vars=120)

        probed = []
        original = cxd._probe_read_throughput
        monkeypatch.setattr(
            cxd, "_probe_read_throughput",
            lambda p: (probed.append(p), original(p))[1],
        )
        _measure_read_throughput(path)
        assert probed == [path]


class TestStreamOnFastAxis:
    def test_auto_converts_end_to_end_and_cleans_up(self, tmp_path, monkeypatch):
        path = _write_csr(tmp_path / "csr.h5ad")
        scratch = tmp_path / "scratch"
        _pin_throughput(
            monkeypatch,
            _throughput_for(path, n_chunks=6, seconds=2 * _SLOW_AXIS_MIN_SAVING_SECONDS),
        )
        with stream_on_fast_axis(
            path, axis=1, policy="auto", fn_name="demo",
            scratch_dir=scratch, chunk_size=10,
        ) as streamed:
            assert streamed != path
            assert cxd.get_matrix_storage_format(streamed) == "csc"
            copy = streamed
        assert not copy.exists()

    def test_auto_streams_from_the_source_without_warning_when_cheap(
        self, tmp_path, monkeypatch
    ):
        path = _write_csr(tmp_path / "csr.h5ad")
        _pin_throughput(monkeypatch, 5e9)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with stream_on_fast_axis(
                path, axis=1, policy="auto", fn_name="demo",
                scratch_dir=tmp_path / "scratch", chunk_size=10,
            ) as streamed:
                assert streamed == path

    def test_auto_warns_when_it_declines_a_cost_worth_knowing_about(
        self, tmp_path, monkeypatch
    ):
        path = _write_csr(tmp_path / "csr.h5ad")
        n_chunks = _SLOW_AXIS_MIN_CHUNKS - 1
        _pin_throughput(
            monkeypatch,
            _throughput_for(
                path, n_chunks=n_chunks, seconds=100 * _SLOW_AXIS_MIN_SAVING_SECONDS
            ),
        )
        with pytest.warns(UserWarning, match="re-reads the whole matrix"):
            with stream_on_fast_axis(
                path, axis=1, policy="auto", fn_name="demo",
                scratch_dir=tmp_path / "scratch", chunk_size=-(-60 // n_chunks),
            ) as streamed:
                assert streamed == path

    def test_matched_format_never_probes(self, tmp_path, monkeypatch):
        """A CSR source streamed by rows is already on its fast axis; the
        policy -- and the measurement it would need -- is irrelevant."""
        path = _write_csr(tmp_path / "csr.h5ad")

        def _boom(_path):  # pragma: no cover - must never run
            raise AssertionError("throughput probed for a matched format")

        monkeypatch.setattr(cxd, "_measure_read_throughput", _boom)
        with stream_on_fast_axis(
            path, axis=0, policy="auto", fn_name="demo",
            scratch_dir=tmp_path / "scratch", chunk_size=10,
        ) as streamed:
            assert streamed == path

    def test_invalid_policy_raises(self, tmp_path):
        path = _write_csr(tmp_path / "csr.h5ad")
        with pytest.raises(ValueError, match="format_mismatch_policy"):
            with stream_on_fast_axis(
                path, axis=1, policy="atuo", fn_name="demo",
                scratch_dir=tmp_path, chunk_size=10,
            ):
                pass


class TestScratchCopyBytes:
    def test_tracks_the_auto_decision(self, tmp_path, monkeypatch):
        path = _write_csr(tmp_path / "csr.h5ad")
        _pin_throughput(monkeypatch, 5e9)
        assert scratch_copy_bytes(path, axis=1, policy="auto", chunk_size=10) is None
        _pin_throughput(
            monkeypatch,
            _throughput_for(path, n_chunks=6, seconds=2 * _SLOW_AXIS_MIN_SAVING_SECONDS),
        )
        assert scratch_copy_bytes(
            path, axis=1, policy="auto", chunk_size=10
        ) == pytest.approx(2 * path.stat().st_size)

    def test_none_for_a_matched_format_or_a_non_converting_policy(self, tmp_path):
        path = _write_csr(tmp_path / "csr.h5ad")
        assert scratch_copy_bytes(path, axis=0, policy="convert", chunk_size=10) is None
        assert scratch_copy_bytes(path, axis=1, policy="warn", chunk_size=10) is None
        assert scratch_copy_bytes(path, axis=1, policy="off", chunk_size=10) is None


class TestStaleScratchSweep:
    """A SIGKILL skips the ``finally`` that deletes the temporary copy, so a
    later run in that directory reclaims what dead runs left.

    The scratch directory is the output directory, which a cluster job array
    shares across nodes, so "the owning PID is not running" is only evidence
    of abandonment when the copy was written on this host -- and even then
    only alongside the age guard, which is what rules out a live owner the
    sweep cannot ask about.
    """

    #: Stands in for another node in the cluster; never this machine.
    OTHER_HOST = "nodea" if _HOST_TAG != "nodea" else "nodeb"

    def _touch(self, directory: Path, name: str, *, age: float = 0.0) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / name
        p.write_bytes(b"x")
        if age:
            stamp = time.time() - age
            os.utime(p, (stamp, stamp))
        return p

    def _name(self, pid: int, *, host: str = _HOST_TAG, salt: str = "abc123") -> str:
        return f".cx_demo_{pid}-{host}_{salt}.csc.h5ad"

    def _dead_pid(self) -> int:
        """A PID no process currently holds."""
        pid = 999_999
        while cxd._process_is_alive(pid):  # pragma: no cover - practically never loops
            pid -= 1
        return pid

    def test_removes_copies_of_dead_runs(self, tmp_path):
        old = 2 * _STALE_SCRATCH_SECONDS
        name = self._name(self._dead_pid())
        stale = self._touch(tmp_path, name, age=old)
        partial = self._touch(tmp_path, f"{name}.partial", age=old)
        _sweep_stale_scratch_copies(tmp_path, fn_name="demo")
        assert not stale.exists()
        assert not partial.exists()

    def test_keeps_copies_of_live_runs(self, tmp_path):
        old = 2 * _STALE_SCRATCH_SECONDS
        live = self._touch(tmp_path, self._name(os.getpid()), age=old)
        other = self._touch(tmp_path, self._name(os.getppid(), salt="def456"), age=old)
        _sweep_stale_scratch_copies(tmp_path, fn_name="demo")
        assert live.exists()
        assert other.exists()

    def test_keeps_a_recent_copy_whose_pid_is_already_gone(self, tmp_path):
        """A PID is reused, and on a shared filesystem it may never have named
        a process on this host at all. A copy still being written is recent,
        so age is what separates the two -- not the PID lookup."""
        recent = self._touch(tmp_path, self._name(self._dead_pid()))
        _sweep_stale_scratch_copies(tmp_path, fn_name="demo")
        assert recent.exists()

    def test_keeps_another_hosts_copy_while_it_is_being_written(self, tmp_path):
        """The regression this guards: node B must not delete the copy node A
        is midway through writing just because A's PID means nothing here."""
        remote = self._touch(tmp_path, self._name(self._dead_pid(), host=self.OTHER_HOST))
        _sweep_stale_scratch_copies(tmp_path, fn_name="demo")
        assert remote.exists()

    def test_reclaims_another_hosts_abandoned_copy_once_it_is_stale(self, tmp_path):
        remote = self._touch(
            tmp_path,
            self._name(os.getpid(), host=self.OTHER_HOST),  # alive here, meaningless there
            age=2 * _STALE_SCRATCH_SECONDS,
        )
        _sweep_stale_scratch_copies(tmp_path, fn_name="demo")
        assert not remote.exists()

    def test_never_touches_files_it_did_not_write(self, tmp_path):
        old = 2 * _STALE_SCRATCH_SECONDS
        dead = self._dead_pid()
        unrelated = self._touch(tmp_path, ".hidden_user_file.h5ad", age=old)
        result = self._touch(tmp_path, "result.h5ad", age=old)
        no_pid = self._touch(tmp_path, ".cx_demo_noise.h5ad", age=old)
        # Only a name this version writes is ours to delete: a PID with no
        # host beside it could have come from anywhere.
        no_host = self._touch(tmp_path, f".cx_demo_{dead}_abc123.csc.h5ad", age=old)
        _sweep_stale_scratch_copies(tmp_path, fn_name="demo")
        assert unrelated.exists() and result.exists()
        assert no_pid.exists() and no_host.exists()

    @pytest.mark.parametrize("seconds", [2 * _SLOW_AXIS_MIN_SAVING_SECONDS, 0.0])
    def test_sweeps_whenever_the_source_is_mismatched(self, tmp_path, monkeypatch, seconds):
        """Including when ``"auto"`` decides *not* to convert: the decision is
        a measurement, so a run that leaves a copy behind may be followed only
        by runs that stream, and those have to reclaim it."""
        path = _write_csr(tmp_path / "csr.h5ad")
        scratch = tmp_path / "scratch"
        stale = self._touch(
            scratch, self._name(self._dead_pid()), age=2 * _STALE_SCRATCH_SECONDS
        )
        _pin_throughput(
            monkeypatch,
            _throughput_for(path, n_chunks=6, seconds=seconds) if seconds else 5e9,
        )
        with stream_on_fast_axis(
            path, axis=1, policy="auto", fn_name="demo",
            scratch_dir=scratch, chunk_size=10,
        ):
            assert not stale.exists()

    def test_the_name_a_conversion_writes_is_one_the_sweep_can_parse(
        self, tmp_path, monkeypatch
    ):
        """Guards the copy's name and the sweep's pattern against drifting
        apart -- if they do, nothing reclaims anything and no test would
        otherwise notice."""
        path = _write_csr(tmp_path / "csr.h5ad")
        scratch = tmp_path / "scratch"
        _pin_throughput(
            monkeypatch,
            _throughput_for(path, n_chunks=6, seconds=2 * _SLOW_AXIS_MIN_SAVING_SECONDS),
        )
        with stream_on_fast_axis(
            path, axis=1, policy="auto", fn_name="demo",
            scratch_dir=scratch, chunk_size=10,
        ) as streamed:
            written = streamed.name
        assert written.startswith(f".cx_demo_{os.getpid()}-{_HOST_TAG}_")
        # The same name, left by a process that is gone and untouched for days.
        abandoned = self._touch(
            scratch,
            written.replace(f"_{os.getpid()}-", f"_{self._dead_pid()}-", 1),
            age=2 * _STALE_SCRATCH_SECONDS,
        )
        _sweep_stale_scratch_copies(scratch, fn_name="demo")
        assert not abandoned.exists()


def test_partial_name_of_a_scratch_copy_stays_sweepable(tmp_path):
    """The converter writes ``<name>.partial`` beside its target; for a
    dot-prefixed scratch copy that must not become ``..cx_*`` -- the sweep's
    pattern (and the user's eye) would both miss it."""
    from crispyx.data import _replace_on_success

    target = tmp_path / f".cx_demo_1234-{_HOST_TAG}_abc.csc.h5ad"
    with _replace_on_success(target) as partial:
        assert partial.name == f".cx_demo_1234-{_HOST_TAG}_abc.csc.h5ad.partial"
        partial.write_bytes(b"x")
    assert target.exists()
