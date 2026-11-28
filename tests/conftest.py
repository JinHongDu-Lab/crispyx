"""Pytest configuration for the crispyx test suite."""

from __future__ import annotations

import os

import pytest

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


@pytest.fixture(autouse=True)
def _empty_process(monkeypatch):
    """An explicit memory_limit_gb reads as the budget it gives: tests pass
    tiny limits to force banded and on-disk paths, which crispyx's import
    footprint (0.3-0.4 GB) would exceed. Tests of that term set it."""
    from crispyx import _memory

    monkeypatch.setattr(_memory, "_baseline_bytes", lambda: 0)


@pytest.fixture
def pydeseq2_helper():
    """Look up one of PyDESeq2's GLM helpers by name (skipping the test
    without PyDESeq2). 0.6.0 moved ``irls_solver``, ``fit_lin_mu`` and
    ``wald_test`` to ``pydeseq2.glm`` and ``nb_nll`` to
    ``pydeseq2.distributions``; it needs Python 3.12, so CI's 3.10 and 3.11
    jobs keep 0.5's ``pydeseq2.utils``."""
    import importlib

    pytest.importorskip("pydeseq2")

    def lookup(name: str):
        for module in ("pydeseq2.glm", "pydeseq2.distributions", "pydeseq2.utils"):
            try:
                return getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                continue
        raise ImportError(f"PyDESeq2 has no {name!r} in pydeseq2.glm, .distributions or .utils")

    return lookup
