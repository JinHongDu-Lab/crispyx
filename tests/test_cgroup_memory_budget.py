"""The auto-sizing budgets must respect a cgroup allowance, not host memory.

Slurm, Docker and Kubernetes cap a job's memory with a cgroup while
``psutil.virtual_memory()`` keeps reporting the whole machine. Budgeting from
the host reading is how an auto-sized run gets OOM-killed on a shared compute
node with every crispyx budget still apparently satisfied.

The trees built here are the real layouts: a container's own cgroup is
namespaced onto the hierarchy root, while a Slurm job's sits several levels
down and the root publishes no limit at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crispyx import _memory
from crispyx.data import _conversion_buffer_budget_bytes

GIB = 1024**3


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    """A fake cgroup hierarchy: write ``/proc/self/cgroup`` and any files."""
    v2_mount = tmp_path / "v2"
    v1_mount = tmp_path / "v1" / "memory"
    proc = tmp_path / "proc_self_cgroup"
    v2_mount.mkdir(parents=True)
    v1_mount.mkdir(parents=True)
    monkeypatch.setattr(_memory, "_PROC_SELF_CGROUP", str(proc))
    monkeypatch.setattr(_memory, "_CGROUP_V2_MOUNT", str(v2_mount))
    monkeypatch.setattr(_memory, "_CGROUP_V1_MEMORY_MOUNT", str(v1_mount))

    class _Hierarchy:
        v2 = v2_mount
        v1 = v1_mount

        @staticmethod
        def proc_says(*lines: str) -> None:
            proc.write_text("\n".join(lines) + "\n")

        @staticmethod
        def write(directory: Path, **files: str) -> Path:
            directory.mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                (directory / name.replace("_", ".", 1)).write_text(content)
            return directory

    return _Hierarchy


def test_no_cgroup_reads_as_no_limit(cgroup):
    # /proc/self/cgroup absent: not Linux, or no cgroup support.
    assert _memory._cgroup_available_bytes() is None


def test_slurm_v2_job_limit_below_the_root_is_found(cgroup):
    """The regression: a Slurm job's ceiling is not at the hierarchy root.

    Slurm does not enter a cgroup namespace, so /sys/fs/cgroup is the host
    root -- which has no memory.max at all -- and the job's ceiling lives
    under system.slice/slurmstepd.scope/job_<id>/.
    """
    cgroup.proc_says("0::/system.slice/slurmstepd.scope/job_4242/step_0/user/task_0")
    cgroup.write(cgroup.v2 / "system.slice/slurmstepd.scope/job_4242",
                 memory_max=str(200 * GIB), memory_current="0")
    cgroup.write(cgroup.v2 / "system.slice/slurmstepd.scope/job_4242/step_0/user/task_0",
                 memory_max="max")
    assert not (cgroup.v2 / "memory.max").exists()  # the root knows nothing
    assert _memory._cgroup_available_bytes() == pytest.approx(200 * GIB)


def test_slurm_v1_job_limit_below_the_root_is_found(cgroup):
    cgroup.proc_says("7:memory:/slurm/uid_1000/job_4242/step_0")
    cgroup.write(cgroup.v1 / "slurm/uid_1000/job_4242",
                 memory_limit_in_bytes=str(200 * GIB), memory_usage_in_bytes="0")
    cgroup.write(cgroup.v1 / "slurm/uid_1000/job_4242/step_0")
    assert _memory._cgroup_available_bytes() == pytest.approx(200 * GIB)


def test_container_cgroup_is_read_from_the_namespaced_root(cgroup):
    """Docker/Kubernetes: the path in /proc/self/cgroup is the host's."""
    cgroup.proc_says("0::/docker/2f9c1e4b5a6d")  # does not exist under the mount
    cgroup.write(cgroup.v2, memory_max=str(64 * GIB), memory_current="0")
    assert _memory._cgroup_available_bytes() == pytest.approx(64 * GIB)


def test_the_tightest_ancestor_limit_wins(cgroup):
    """A parent's ceiling binds its children, however large the leaf's is."""
    cgroup.proc_says("0::/pod/container")
    cgroup.write(cgroup.v2 / "pod", memory_max=str(100 * GIB), memory_current="0")
    cgroup.write(cgroup.v2 / "pod/container",
                 memory_max=str(200 * GIB), memory_current="0")
    assert _memory._cgroup_available_bytes() == pytest.approx(100 * GIB)


def test_v2_max_means_unlimited(cgroup):
    cgroup.proc_says("0::/")
    cgroup.write(cgroup.v2, memory_max="max", memory_current=str(8 * GIB))
    assert _memory._cgroup_available_bytes() is None


def test_v1_unlimited_sentinel_is_ignored(cgroup):
    # cgroup v1 writes a value near 2**63 rather than a word when unlimited.
    cgroup.proc_says("7:memory:/")
    cgroup.write(cgroup.v1, memory_limit_in_bytes=str(2**63 - 4096))
    assert _memory._cgroup_available_bytes() is None


def test_unparseable_cgroup_file_is_ignored(cgroup):
    cgroup.proc_says("0::/")
    cgroup.write(cgroup.v2, memory_max="not-a-number")
    assert _memory._cgroup_available_bytes() is None


def test_headroom_excludes_memory_the_job_already_holds(cgroup):
    """The ceiling is what the job was granted; what is left is what matters.

    A job holding 150 GB of a 200 GB allocation can allocate 50 GB more, not
    200 -- budgeting the ceiling OOMs exactly as budgeting host memory does.
    """
    cgroup.proc_says("0::/")
    cgroup.write(cgroup.v2, memory_max=str(200 * GIB), memory_current=str(150 * GIB))
    assert _memory._cgroup_available_bytes() == pytest.approx(50 * GIB)


def test_reclaimable_page_cache_does_not_count_as_held(cgroup):
    """Streaming an h5ad charges page cache to the cgroup; it is not a loss.

    Without this, every chunk size would collapse as soon as one large file
    had been read.
    """
    cgroup.proc_says("0::/")
    cgroup.write(
        cgroup.v2,
        memory_max=str(200 * GIB),
        memory_current=str(160 * GIB),
        memory_stat=f"anon {20 * GIB}\nfile {140 * GIB}\ninactive_file {140 * GIB}\n",
    )
    assert _memory._cgroup_available_bytes() == pytest.approx(180 * GIB)


def test_v1_reclaimable_page_cache_uses_the_v1_stat_key(cgroup):
    cgroup.proc_says("7:memory:/")
    cgroup.write(
        cgroup.v1,
        memory_limit_in_bytes=str(200 * GIB),
        memory_usage_in_bytes=str(160 * GIB),
        memory_stat=f"total_inactive_file {140 * GIB}\n",
    )
    assert _memory._cgroup_available_bytes() == pytest.approx(180 * GIB)


def test_a_partially_torn_down_job_tree_still_finds_the_job_limit(cgroup):
    """The descent stops at the deepest directory that exists, not at the root.

    A step's leaf directories come and go while the job's own cgroup stays;
    falling straight back to the mount when the full path is missing would
    lose the job ceiling and report the host again.
    """
    cgroup.proc_says("0::/system.slice/slurmstepd.scope/job_4242/step_0/user/task_0")
    cgroup.write(cgroup.v2 / "system.slice/slurmstepd.scope/job_4242",
                 memory_max=str(200 * GIB), memory_current="0")
    # step_0/user/task_0 was never created.
    assert _memory._cgroup_available_bytes() == pytest.approx(200 * GIB)


def test_an_unmounted_hierarchy_reads_as_no_limit(cgroup, monkeypatch):
    """/proc/self/cgroup can name a hierarchy that is not mounted here."""
    cgroup.proc_says("0::/some/path")
    monkeypatch.setattr(_memory, "_CGROUP_V2_MOUNT", "/nonexistent/cgroup")
    assert _memory._cgroup_available_bytes() is None


@pytest.mark.parametrize("layout, files", [
    ("v2 root publishes no memory.max", {}),
    ("v1 root publishes the unlimited sentinel",
     {"memory_limit_in_bytes": str(2**63 - 4096)}),
])
def test_an_uncapped_linux_host_is_unchanged_by_cgroup_detection(cgroup, layout, files):
    """Bare metal must budget from psutil exactly as it did before.

    The overwhelmingly common case: a Linux box with a cgroup hierarchy
    mounted but no memory ceiling anywhere on this process's path.
    """
    if files:
        cgroup.proc_says("7:memory:/user.slice/user-1000.slice/session-2.scope")
        cgroup.write(cgroup.v1, **files)
        cgroup.write(cgroup.v1 / "user.slice/user-1000.slice/session-2.scope")
    else:
        cgroup.proc_says("0::/user.slice/user-1000.slice/session-2.scope")
        cgroup.write(cgroup.v2 / "user.slice/user-1000.slice/session-2.scope")
    assert _memory._cgroup_available_bytes() is None, layout


def test_detected_memory_is_capped_by_the_cgroup(cgroup, monkeypatch):
    """A 200G job on a node reporting 1 TB free must budget from 200G."""
    cgroup.proc_says("0::/")
    cgroup.write(cgroup.v2, memory_max=str(200 * GIB), memory_current="0")
    import psutil

    class _HostMemory:
        available = 1024 * GIB  # 1 TB free on the shared node

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _HostMemory())
    assert _memory._detected_available_bytes() == pytest.approx(200 * GIB)
    assert _memory._resolve_memory_limit_bytes(None) == pytest.approx(200 * GIB)


def test_explicit_limit_larger_than_the_cgroup_is_capped(monkeypatch):
    """An over-large memory_limit_gb cannot buy memory the cgroup won't grant.

    Passing memory_limit_gb=400 to a job allocated 200G is a real and
    already-observed mistake; it must not size buffers past the allocation.
    """
    monkeypatch.setattr(_memory, "_cgroup_available_bytes", lambda: float(200 * GIB))
    assert _memory._resolve_memory_limit_bytes(400) == pytest.approx(200 * GIB)
    # Under the ceiling, the explicit value still wins.
    assert _memory._resolve_memory_limit_bytes(100) == pytest.approx(100e9)


def test_conversion_buffer_budget_is_capped_by_the_cgroup(monkeypatch):
    import crispyx.data as cxd

    monkeypatch.setattr(cxd, "_cgroup_available_bytes", lambda: float(200 * GIB))
    # Half of the requested 400 GB would be 200 GB, over the whole allocation.
    assert _conversion_buffer_budget_bytes(400) == pytest.approx(0.5 * 200 * GIB)
    # Half of a request that fits is unchanged.
    assert _conversion_buffer_budget_bytes(80) == pytest.approx(40e9)


def test_no_cgroup_leaves_every_budget_untouched(monkeypatch):
    import crispyx.data as cxd

    monkeypatch.setattr(_memory, "_cgroup_available_bytes", lambda: None)
    monkeypatch.setattr(cxd, "_cgroup_available_bytes", lambda: None)
    assert _memory._resolve_memory_limit_bytes(170) == pytest.approx(170e9)
    assert _conversion_buffer_budget_bytes(170) == pytest.approx(85e9)
    assert np.isfinite(_memory._resolve_memory_limit_bytes(None))
