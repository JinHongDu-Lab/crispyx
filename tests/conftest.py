"""Pytest configuration for the crispyx test suite."""

from __future__ import annotations

import os

# Pin NUMBA_NUM_THREADS before any import so umap-learn / pynndescent cannot
# raise "Cannot set NUMBA_NUM_THREADS to a different value once the threads
# have been launched" across the full test session.
# NUMBA_THREADING_LAYER is handled by crispyx.__init__ (workqueue, to avoid
# the libiomp5 / libomp conflict when torch is co-installed).
os.environ.setdefault("NUMBA_NUM_THREADS", "4")


def pytest_sessionstart(session):  # pragma: no cover - test harness setup
    try:
        from joblib.externals.loky import process_executor
    except Exception:
        return

    if getattr(process_executor._check_system_limits, "_crispyx_patched", False):
        return

    original = process_executor._check_system_limits

    def _check_system_limits_safe():
        try:
            return original()
        except PermissionError:
            return None

    _check_system_limits_safe._crispyx_patched = True  # type: ignore[attr-defined]
    process_executor._check_system_limits = _check_system_limits_safe
