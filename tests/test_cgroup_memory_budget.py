"""The auto-sizing budgets must respect a cgroup ceiling, not host memory.

Slurm, Docker and Kubernetes cap a job's memory with a cgroup while
``psutil.virtual_memory()`` keeps reporting the whole machine. Budgeting from
the host reading is how an auto-sized run gets OOM-killed on a shared compute
node with every crispyx budget still apparently satisfied.
"""

from __future__ import annotations

import numpy as np
import pytest

from crispyx import _memory
from crispyx.data import _conversion_buffer_budget_bytes


@pytest.fixture
def cgroup_files(tmp_path, monkeypatch):
    """Point the cgroup reader at files under tmp_path."""

    def _write(v2: str | None = None, v1: str | None = None) -> None:
        paths = []
        if v2 is not None:
            p = tmp_path / "memory.max"
            p.write_text(v2)
            paths.append((str(p), ("max",)))
        else:
            paths.append((str(tmp_path / "absent-v2"), ("max",)))
        if v1 is not None:
            p = tmp_path / "memory.limit_in_bytes"
            p.write_text(v1)
            paths.append((str(p), ()))
        else:
            paths.append((str(tmp_path / "absent-v1"), ()))
        monkeypatch.setattr(_memory, "_CGROUP_PATHS", tuple(paths))

    return _write


def test_no_cgroup_reads_as_no_limit(cgroup_files):
    cgroup_files()
    assert _memory._cgroup_memory_limit_bytes() is None


def test_cgroup_v2_limit_is_read(cgroup_files):
    cgroup_files(v2=str(200 * 1024**3))
    assert _memory._cgroup_memory_limit_bytes() == pytest.approx(200 * 1024**3)


def test_cgroup_v2_max_means_unlimited_and_falls_through_to_v1(cgroup_files):
    cgroup_files(v2="max", v1=str(64 * 1024**3))
    assert _memory._cgroup_memory_limit_bytes() == pytest.approx(64 * 1024**3)


def test_cgroup_v1_unlimited_sentinel_is_ignored(cgroup_files):
    # cgroup v1 writes a value near 2**63 rather than a word when unlimited.
    cgroup_files(v1=str(2**63 - 4096))
    assert _memory._cgroup_memory_limit_bytes() is None


def test_unparseable_cgroup_file_is_ignored(cgroup_files):
    cgroup_files(v2="not-a-number")
    assert _memory._cgroup_memory_limit_bytes() is None


def test_detected_memory_is_capped_by_the_cgroup(cgroup_files, monkeypatch):
    """A 200G job on a node reporting 1 TB free must budget from 200G."""
    cgroup_files(v2=str(200 * 1024**3))
    monkeypatch.setattr(
        _memory, "_cgroup_memory_limit_bytes", lambda: float(200 * 1024**3)
    )
    import psutil

    class _HostMemory:
        available = 1024 * 1024**3  # 1 TB free on the shared node

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _HostMemory())
    assert _memory._detected_available_bytes() == pytest.approx(200 * 1024**3)
    assert _memory._resolve_memory_limit_bytes(None) == pytest.approx(200 * 1024**3)


def test_explicit_limit_larger_than_the_cgroup_is_capped(monkeypatch):
    """An over-large memory_limit_gb cannot buy memory the cgroup won't grant.

    Passing memory_limit_gb=400 to a job allocated 200G is a real and
    already-observed mistake; it must not size buffers past the allocation.
    """
    monkeypatch.setattr(
        _memory, "_cgroup_memory_limit_bytes", lambda: float(200 * 1024**3)
    )
    assert _memory._resolve_memory_limit_bytes(400) == pytest.approx(200 * 1024**3)
    # Under the ceiling, the explicit value still wins.
    assert _memory._resolve_memory_limit_bytes(100) == pytest.approx(100e9)


def test_conversion_buffer_budget_is_capped_by_the_cgroup(monkeypatch):
    import crispyx.data as cxd

    monkeypatch.setattr(cxd, "_cgroup_memory_limit_bytes", lambda: float(200 * 1024**3))
    # Half of the requested 400 GB would be 200 GB, over the whole allocation.
    assert _conversion_buffer_budget_bytes(400) == pytest.approx(0.5 * 200 * 1024**3)
    # Half of a request that fits is unchanged.
    assert _conversion_buffer_budget_bytes(80) == pytest.approx(40e9)


def test_no_cgroup_leaves_every_budget_untouched(monkeypatch):
    import crispyx.data as cxd

    monkeypatch.setattr(_memory, "_cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(cxd, "_cgroup_memory_limit_bytes", lambda: None)
    assert _memory._resolve_memory_limit_bytes(170) == pytest.approx(170e9)
    assert _conversion_buffer_budget_bytes(170) == pytest.approx(85e9)
    assert np.isfinite(_memory._resolve_memory_limit_bytes(None))
