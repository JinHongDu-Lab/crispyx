"""Profiling utilities for timing and memory measurement.

This module provides a unified `Profiler` class for measuring execution time
and memory usage of code sections. It supports:

- Timing measurement with start/stop labels
- Memory tracking via tracemalloc (Python objects) or RSS (process-level)
- Continuous background memory sampling
- Visualization utilities for timing and memory plots

Example usage::

    from crispyx.profiling import Profiler
    
    # Basic timing
    profiler = Profiler(timing=True)
    profiler.start("data_loading")
    # ... load data ...
    profiler.stop("data_loading")
    print(profiler.get_report())
    
    # Combined timing and memory with context manager
    with Profiler(timing=True, memory=True) as p:
        p.start("processing")
        # ... process data ...
        p.snapshot("after_processing")
        p.stop("processing")
    
    # Access results programmatically
    stats = p.get_stats()
    print(f"Total time: {stats['timing']['total_seconds']}s")
    print(f"Peak memory: {stats['memory']['peak_mb']}MB")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = ["Profiler", "TimingProfiler", "MemoryProfiler"]


class Profiler:
    """Unified profiler for timing and memory measurement.
    
    This class combines timing profiling, memory snapshots, and continuous
    memory sampling into a single interface. It can be used as a context
    manager or with explicit start/stop calls.
    
    Parameters
    ----------
    timing : bool, default=False
        Enable timing measurement of code sections.
    memory : bool, default=False
        Enable memory tracking (snapshots at labeled points).
    memory_method : {"tracemalloc", "rss"}, default="tracemalloc"
        Method for memory measurement:
        - "tracemalloc": Python object memory via tracemalloc (more detailed)
        - "rss": Process resident set size via psutil (total process memory)
    sampling : bool, default=False
        Enable continuous background memory sampling. Runs a daemon thread
        that records memory usage at `sample_interval` intervals.
    sample_interval : float, default=0.1
        Seconds between samples when `sampling=True`.
    top_n : int, default=10
        Number of top allocations to report (only for tracemalloc).
        
    Examples
    --------
    >>> with Profiler(timing=True, memory=True) as p:
    ...     p.start("section1")
    ...     # ... code ...
    ...     p.stop("section1")
    >>> print(p.get_report())
    """
    
    def __init__(
        self,
        timing: bool = False,
        memory: bool = False,
        memory_method: Literal["tracemalloc", "rss"] = "tracemalloc",
        sampling: bool = False,
        sample_interval: float = 0.1,
        top_n: int = 10,
    ):
        self.timing_enabled = timing
        self.memory_enabled = memory
        self.memory_method = memory_method
        self.sampling_enabled = sampling
        self.sample_interval = sample_interval
        self.top_n = top_n
        
        # Timing state
        self._timings: dict[str, float] = {}
        self._start_times: dict[str, float] = {}
        self._total_start: float | None = None
        
        # Memory state
        self._snapshots: dict[str, dict] = {}
        self._peak_memory_mb: float = 0.0
        self._tracemalloc_start_time: float | None = None
        
        # Sampling state
        self._samples: list[tuple[float, float]] = []  # (timestamp, memory_mb)
        self._sampling_thread: threading.Thread | None = None
        self._stop_sampling_event: threading.Event | None = None
    
    def __enter__(self):
        """Start profiling context."""
        if self.memory_enabled and self.memory_method == "tracemalloc":
            import tracemalloc
            tracemalloc.start()
            self._tracemalloc_start_time = time.perf_counter()
        
        if self.sampling_enabled:
            self.start_sampling()
        
        if self.timing_enabled:
            self._total_start = time.perf_counter()
            self.start("total")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling context."""
        if self.timing_enabled and "total" in self._start_times:
            self.stop("total")
        
        if self.sampling_enabled:
            self.stop_sampling()
        
        if self.memory_enabled:
            self.snapshot("end")
            if self.memory_method == "tracemalloc":
                import tracemalloc
                tracemalloc.stop()
        
        return False
    
    # =========================================================================
    # Timing methods
    # =========================================================================
    
    def start(self, label: str) -> None:
        """Start timing a labeled section."""
        if not self.timing_enabled:
            return
        if self._total_start is None:
            self._total_start = time.perf_counter()
        self._start_times[label] = time.perf_counter()
    
    def stop(self, label: str) -> float:
        """Stop timing a labeled section and return elapsed time."""
        if not self.timing_enabled:
            return 0.0
        if label not in self._start_times:
            logger.warning(f"Profiler.stop() called for unstarted label: {label}")
            return 0.0
        
        elapsed = time.perf_counter() - self._start_times[label]
        if label in self._timings:
            self._timings[label] += elapsed  # Accumulate if called multiple times
        else:
            self._timings[label] = elapsed
        del self._start_times[label]
        return elapsed
    
    def get_total_time(self) -> float:
        """Get total elapsed time since first start() call."""
        if not self.timing_enabled or self._total_start is None:
            return 0.0
        return time.perf_counter() - self._total_start
    
    # =========================================================================
    # Memory snapshot methods
    # =========================================================================
    
    def snapshot(self, label: str) -> None:
        """Take a memory snapshot at the current point."""
        if not self.memory_enabled:
            return
        
        timestamp = time.perf_counter() - (self._tracemalloc_start_time or self._total_start or time.perf_counter())
        
        if self.memory_method == "tracemalloc":
            import tracemalloc
            snap = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
            self._snapshots[label] = {
                "snapshot": snap,
                "timestamp_s": timestamp,
                "current_mb": current_mb,
                "peak_mb": peak_mb,
            }
        else:  # rss
            current_mb = self._get_rss_mb()
            self._snapshots[label] = {
                "snapshot": None,
                "timestamp_s": timestamp,
                "current_mb": current_mb,
                "peak_mb": current_mb,  # RSS doesn't track peak
            }
            peak_mb = current_mb
        
        self._peak_memory_mb = max(self._peak_memory_mb, peak_mb)
        
        logger.debug(
            f"Memory snapshot '{label}': current={current_mb:.1f}MB, peak={peak_mb:.1f}MB"
        )
    
    def _get_rss_mb(self) -> float:
        """Get current process RSS in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback for systems without psutil
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on Linux
            except ImportError:
                logger.warning("Neither psutil nor resource available for RSS measurement")
                return 0.0
    
    def reset_peak(self) -> None:
        """Reset peak memory tracking to current memory level.
        
        This is useful when profiling a specific operation after loading data,
        so that peak memory measures only the operation's memory usage,
        not the memory used by previously loaded data.
        """
        if not self.memory_enabled and not self.sampling_enabled:
            return
        
        if self.memory_method == "tracemalloc":
            import tracemalloc
            tracemalloc.reset_peak()
            _, current_peak = tracemalloc.get_traced_memory()
            self._peak_memory_mb = current_peak / 1024 / 1024
        else:  # rss
            # For RSS, we can only reset our tracked peak (can't reset OS stats)
            current_mb = self._get_rss_mb()
            self._peak_memory_mb = current_mb
            
        # Also clear previous samples if sampling is active
        if self.sampling_enabled:
            self._samples = []
    
    # =========================================================================
    # Continuous sampling methods
    # =========================================================================
    
    def start_sampling(self) -> None:
        """Start background memory sampling thread."""
        if not self.sampling_enabled:
            return
        if self._sampling_thread is not None:
            return  # Already running
        
        self._stop_sampling_event = threading.Event()
        self._samples = []
        start_time = time.perf_counter()
        
        def _sample_loop():
            while not self._stop_sampling_event.is_set():
                timestamp = time.perf_counter() - start_time
                memory_mb = self._get_rss_mb()
                self._samples.append((timestamp, memory_mb))
                self._peak_memory_mb = max(self._peak_memory_mb, memory_mb)
                self._stop_sampling_event.wait(self.sample_interval)
        
        self._sampling_thread = threading.Thread(target=_sample_loop, daemon=True)
        self._sampling_thread.start()
    
    def stop_sampling(self) -> None:
        """Stop background memory sampling thread."""
        if self._stop_sampling_event is not None:
            self._stop_sampling_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=1.0)
            self._sampling_thread = None
    
    # =========================================================================
    # Results methods
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get profiling statistics as a dict.

        Returns
        -------
        dict
            Dictionary with keys ``"timing"`` and ``"memory"``.

            The ``"timing"`` entry contains ``"total_seconds"`` (float) and
            ``"sections"`` (a mapping of label to ``{"seconds": float,
            "percent": float}``).

            The ``"memory"`` entry contains ``"peak_mb"`` (float),
            ``"snapshots"`` (a mapping of label to snapshot data),
            ``"samples"`` (list of timestamp/memory pairs if sampling is
            enabled), and ``"top_allocations"`` (if tracemalloc is used).
        """
        stats = {}
        
        # Timing stats
        if self.timing_enabled:
            total = self._timings.get("total", self.get_total_time())
            timing_stats = {
                "total_seconds": round(total, 3),
                "sections": {},
            }
            for label, elapsed in self._timings.items():
                pct = (elapsed / total * 100) if total > 0 else 0
                timing_stats["sections"][label] = {
                    "seconds": round(elapsed, 3),
                    "percent": round(pct, 1),
                }
            stats["timing"] = timing_stats
        
        # Memory stats
        if self.memory_enabled or self.sampling_enabled:
            memory_stats = {
                "peak_mb": round(self._peak_memory_mb, 2),
                "snapshots": {},
            }
            
            for label, snap_data in self._snapshots.items():
                memory_stats["snapshots"][label] = {
                    "timestamp_s": round(snap_data["timestamp_s"], 3),
                    "current_mb": round(snap_data["current_mb"], 2),
                }
            
            if self.sampling_enabled and self._samples:
                memory_stats["samples"] = [
                    (round(t, 3), round(m, 2)) for t, m in self._samples
                ]
            
            # Top allocations from tracemalloc
            if self.memory_method == "tracemalloc" and "end" in self._snapshots:
                snap = self._snapshots["end"].get("snapshot")
                if snap is not None:
                    top_stats = snap.statistics("lineno")[:self.top_n]
                    memory_stats["top_allocations"] = [
                        {
                            "file": str(stat.traceback),
                            "size_mb": round(stat.size / 1024 / 1024, 2),
                            "count": stat.count,
                        }
                        for stat in top_stats
                    ]
            
            stats["memory"] = memory_stats
        
        return stats
    
    def get_report(self) -> str:
        """Generate a human-readable profiling report."""
        if not self.timing_enabled and not self.memory_enabled:
            return "Profiling was not enabled."
        
        lines = ["=" * 60, "Profiling Report", "=" * 60]
        
        # Timing section
        if self.timing_enabled and self._timings:
            total = self._timings.get("total", self.get_total_time())
            lines.append(f"\nTotal time: {total:.2f}s\n")
            lines.append("Section Breakdown:")
            lines.append("-" * 50)
            
            sorted_timings = sorted(
                [(k, v) for k, v in self._timings.items() if k != "total"],
                key=lambda x: -x[1]
            )
            for label, elapsed in sorted_timings:
                pct = (elapsed / total * 100) if total > 0 else 0
                bar_len = int(pct / 2)
                bar = "█" * bar_len + "░" * (25 - bar_len)
                lines.append(f"  {label:30s} {elapsed:7.2f}s ({pct:5.1f}%) {bar}")
        
        # Memory section
        if self.memory_enabled and self._snapshots:
            lines.append("\nMemory Snapshots:")
            lines.append("-" * 50)
            for label, snap_data in self._snapshots.items():
                lines.append(
                    f"  {label:25s} t={snap_data['timestamp_s']:7.2f}s  "
                    f"current={snap_data['current_mb']:8.1f}MB"
                )
            lines.append(f"\nPeak memory: {self._peak_memory_mb:.1f}MB")
        
        # Sampling summary
        if self.sampling_enabled and self._samples:
            lines.append(f"\nMemory sampling: {len(self._samples)} samples collected")
            if self._samples:
                min_mem = min(m for _, m in self._samples)
                max_mem = max(m for _, m in self._samples)
                lines.append(f"  Range: {min_mem:.1f}MB - {max_mem:.1f}MB")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    # =========================================================================
    # Visualization methods
    # =========================================================================
    
    def plot_timeline(self, ax=None):
        """Plot timing breakdown as horizontal bar chart.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; cannot create plot")
            return None
        
        if not self.timing_enabled or not self._timings:
            logger.warning("No timing data to plot")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by elapsed time
        sorted_timings = sorted(
            [(k, v) for k, v in self._timings.items() if k != "total"],
            key=lambda x: x[1]
        )
        labels = [k for k, v in sorted_timings]
        times = [v for k, v in sorted_timings]
        
        colors = plt.cm.viridis([i / len(labels) for i in range(len(labels))])
        ax.barh(labels, times, color=colors)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Timing Breakdown by Section")
        
        # Add time labels on bars
        for i, (label, t) in enumerate(sorted_timings):
            ax.text(t + 0.1, i, f"{t:.2f}s", va="center", fontsize=9)
        
        plt.tight_layout()
        return ax
    
    def plot_memory(self, ax=None):
        """Plot memory usage over time (requires sampling mode).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; cannot create plot")
            return None
        
        if not self._samples:
            logger.warning("No memory samples to plot. Enable sampling=True.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        timestamps = [t for t, m in self._samples]
        memory = [m for t, m in self._samples]
        
        ax.plot(timestamps, memory, "b-", linewidth=1.5)
        ax.fill_between(timestamps, memory, alpha=0.3)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage Over Time")
        
        # Mark peak
        peak_idx = memory.index(max(memory))
        ax.axhline(y=max(memory), color="r", linestyle="--", alpha=0.5)
        ax.annotate(
            f"Peak: {max(memory):.1f}MB",
            xy=(timestamps[peak_idx], max(memory)),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
        )
        
        plt.tight_layout()
        return ax


# =============================================================================
# Specialized profiler subclasses
# =============================================================================

class TimingProfiler(Profiler):
    """Timing-only profiler.

    Thin wrapper around ``Profiler`` with ``timing=True``.
    """

    def __init__(self, enabled: bool = False):
        super().__init__(timing=enabled)
        self.enabled = enabled
        self.timings = self._timings


class MemoryProfiler(Profiler):
    """Memory-only profiler.

    Thin wrapper around ``Profiler`` with ``memory=True``.
    """

    def __init__(self, enabled: bool = False, top_n: int = 10):
        super().__init__(memory=enabled, top_n=top_n)
        self.enabled = enabled
        self.snapshots = self._snapshots


# =============================================================================
# Standalone visualization utilities
# =============================================================================

def plot_benchmark_comparison(
    profiler_results: list[dict],
    labels: list[str],
    metric: Literal["timing", "memory"] = "timing",
    ax=None,
):
    """Compare profiling results from multiple runs side-by-side.
    
    Useful for comparing before/after optimization, different parameters,
    or different methods (crispyx vs PyDESeq2).
    
    Parameters
    ----------
    profiler_results : list[dict]
        List of profiler stats dicts from `Profiler.get_stats()`.
    labels : list[str]
        Labels for each run (e.g., ["before", "after"]).
    metric : {"timing", "memory"}
        Which metric to compare.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
        
    Examples
    --------
    >>> stats_before = profiler_before.get_stats()
    >>> stats_after = profiler_after.get_stats()
    >>> plot_benchmark_comparison([stats_before, stats_after], ["Before", "After"])
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; cannot create plot")
        return None
    
    if not profiler_results:
        logger.warning("No profiler results to compare")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if metric == "timing":
        # Collect all section names
        all_sections = set()
        for result in profiler_results:
            if "timing" in result:
                all_sections.update(result["timing"].get("sections", {}).keys())
        all_sections.discard("total")
        sections = sorted(all_sections)
        
        if not sections:
            logger.warning("No timing sections to compare")
            return ax
        
        # Build data matrix
        n_runs = len(profiler_results)
        n_sections = len(sections)
        x = np.arange(n_sections)
        width = 0.8 / n_runs
        
        for i, (result, label) in enumerate(zip(profiler_results, labels)):
            timing = result.get("timing", {}).get("sections", {})
            values = [timing.get(s, {}).get("seconds", 0) for s in sections]
            offset = (i - n_runs / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label)
        
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Timing Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(sections, rotation=45, ha="right")
        ax.legend()
    
    else:  # memory
        # Compare peak memory
        peaks = []
        for result in profiler_results:
            peak = result.get("memory", {}).get("peak_mb", 0)
            peaks.append(peak)
        
        x = np.arange(len(labels))
        ax.bar(x, peaks, color=plt.cm.viridis([0.3, 0.7][:len(labels)]))
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Memory Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        # Add value labels
        for i, peak in enumerate(peaks):
            ax.text(i, peak + 5, f"{peak:.1f}MB", ha="center", fontsize=10)
    
    plt.tight_layout()
    return ax
