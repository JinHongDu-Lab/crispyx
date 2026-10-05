"""Tests for the profiling module.

Timing is checked against a fake clock, so the assertions are exact and the
tests do not sleep; only the background memory sampler needs real time.
"""

import time
import tracemalloc

import numpy as np
import pytest

from crispyx.profiling import MemoryProfiler, Profiler, TimingProfiler


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, "perf_counter", fake)
    return fake


@pytest.fixture
def tracing():
    tracemalloc.start()
    yield
    tracemalloc.stop()


def test_timing_records_and_accumulates_sections(clock):
    profiler = Profiler(timing=True)
    profiler.start("a")
    clock.advance(2.0)
    assert profiler.stop("a") == 2.0
    profiler.start("b")
    clock.advance(3.0)
    profiler.stop("b")
    profiler.start("a")  # a repeated label accumulates
    clock.advance(0.5)
    profiler.stop("a")

    sections = profiler.get_stats()["timing"]["sections"]
    assert sections["a"]["seconds"] == pytest.approx(2.5)
    assert sections["b"]["seconds"] == pytest.approx(3.0)
    assert profiler.get_total_time() == pytest.approx(5.5)
    report = profiler.get_report()
    assert "a" in report and "b" in report


def test_disabled_profiler_records_nothing(clock):
    profiler = Profiler(timing=False, memory=False, sampling=False)
    profiler.start("a")
    clock.advance(1.0)
    assert profiler.stop("a") == 0.0
    profiler.snapshot("s")
    profiler.start_sampling()
    profiler.stop_sampling()
    stats = profiler.get_stats()
    assert "timing" not in stats
    assert not stats.get("memory", {}).get("snapshots")
    assert profiler._samples == []
    assert "not enabled" in profiler.get_report().lower()


def test_tracemalloc_snapshots_see_an_allocation(tracing):
    profiler = Profiler(memory=True, memory_method="tracemalloc")
    profiler._tracemalloc_start_time = time.perf_counter()
    profiler.snapshot("before")
    data = np.ones((1000, 1000))  # 8 MB
    profiler.snapshot("after")
    snapshots = profiler.get_stats()["memory"]["snapshots"]
    assert snapshots["after"]["current_mb"] - snapshots["before"]["current_mb"] > 7
    del data


def test_rss_snapshot_is_positive():
    profiler = Profiler(memory=True, memory_method="rss")
    profiler._total_start = time.perf_counter()
    profiler.snapshot("now")
    assert profiler.get_stats()["memory"]["snapshots"]["now"]["current_mb"] > 0


def test_sampling_collects_timestamped_memory_samples():
    profiler = Profiler(sampling=True, sample_interval=0.01)
    profiler.start_sampling()
    time.sleep(0.1)
    profiler.stop_sampling()
    assert len(profiler._samples) >= 2
    timestamps = [t for t, _ in profiler._samples]
    assert timestamps == sorted(timestamps)
    assert all(isinstance(mb, float) and mb > 0 for _, mb in profiler._samples)


def test_context_manager_adds_total_and_end():
    with Profiler(timing=True, memory=True, sampling=True, sample_interval=0.01) as profiler:
        profiler.start("work")
        data = np.random.default_rng(0).normal(size=(1000, 100))
        np.linalg.svd(data, full_matrices=False)
        profiler.snapshot("after_svd")
        profiler.stop("work")

    stats = profiler.get_stats()
    assert {"total", "work"} <= set(stats["timing"]["sections"])
    assert stats["timing"]["total_seconds"] > 0
    assert {"after_svd", "end"} <= set(stats["memory"]["snapshots"])
    assert stats["memory"]["peak_mb"] > 0
    assert "samples" in stats["memory"]
    report = profiler.get_report()
    assert "Memory Snapshots:" in report and "after_svd" in report
    assert "Peak memory:" in report
    assert "Memory sampling:" in report


def test_specialised_profilers_expose_their_records(clock, tracing):
    timer = TimingProfiler(enabled=True)
    timer.start("section")
    clock.advance(1.5)
    timer.stop("section")
    assert timer.timings["section"] == pytest.approx(1.5)

    off = TimingProfiler(enabled=False)
    off.start("section")
    off.stop("section")
    assert off.timings == {}

    memory = MemoryProfiler(enabled=True)
    memory._tracemalloc_start_time = time.perf_counter()
    memory.snapshot("test")
    assert "test" in memory.snapshots


@pytest.mark.parametrize("plot", ["plot_timeline", "plot_memory"])
def test_plots_return_axes(plot):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiler = Profiler(timing=True, sampling=True, sample_interval=0.01)
    profiler.start("total")
    profiler.start_sampling()
    for label in ("a", "b"):
        profiler.start(label)
        time.sleep(0.02)
        profiler.stop(label)
    profiler.stop_sampling()
    profiler.stop("total")

    ax = getattr(profiler, plot)()
    assert ax.has_data()
    plt.close("all")
