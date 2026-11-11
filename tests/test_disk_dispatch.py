"""Tests for the disk-space estimation and warning utility (crispyx._disk)."""

from __future__ import annotations

import sys
import shutil
import types
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from crispyx._disk import (
    DiskEstimate,
    assess_bytes,
    estimate_bytes,
    estimate_conversion_bytes,
    estimate_sparse_output_bytes,
    warn_if_disk_space_low,
)


# ---------------------------------------------------------------------------
# estimate_bytes / estimate_sparse_output_bytes / estimate_conversion_bytes
# ---------------------------------------------------------------------------

class TestEstimateBytes:
    def test_scales_with_extra_dims(self):
        base = estimate_bytes(1000, 20000)
        assert estimate_bytes(1000, 20000, 2) == pytest.approx(base * 2)  # n_layers=2

    def test_exact_with_default_overhead(self):
        assert estimate_bytes(100, 2000, itemsize=8) == 100 * 2000 * 8

    def test_default_itemsize_is_float64(self):
        assert estimate_bytes(10, 10) == estimate_bytes(10, 10, itemsize=8)

    def test_overhead_multiplies_final_size(self):
        assert estimate_bytes(100, 2000, overhead=1.10) == pytest.approx(100 * 2000 * 8 * 1.10)

    def test_no_dims_falls_back_to_one_element(self):
        # Degenerate but should not raise: product of an empty dims sequence
        # is 1, so this returns a single element's worth of bytes.
        assert estimate_bytes() == 8.0


class TestEstimateSparseOutputBytes:
    def test_scales_with_nnz(self):
        assert estimate_sparse_output_bytes(1000) == pytest.approx(1000 * (4 + 4) * 1.10)

    def test_zero_nnz_is_zero(self):
        assert estimate_sparse_output_bytes(0) == 0.0

    def test_custom_itemsizes(self):
        result = estimate_sparse_output_bytes(100, value_itemsize=8, index_itemsize=8, overhead=1.0)
        assert result == 100 * 16


class TestEstimateConversionBytes:
    def test_conversion_bytes_is_double_the_source(self, tmp_path):
        f = tmp_path / "source.h5ad"
        f.write_bytes(b"0" * 10_000)
        assert estimate_conversion_bytes(f) == 20_000

    def test_accepts_str_path(self, tmp_path):
        f = tmp_path / "source.h5ad"
        f.write_bytes(b"0" * 500)
        assert estimate_conversion_bytes(str(f)) == 1000


# ---------------------------------------------------------------------------
# DiskEstimate / assess_bytes
# ---------------------------------------------------------------------------

class TestDiskEstimate:
    def test_sufficient_when_plenty_of_room(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=10**15, used=0, free=10**15),
        )
        estimate = assess_bytes(1e6, tmp_path)
        assert estimate.sufficient is True
        assert estimate.required_gb == pytest.approx(1e6 / 1e9)
        assert estimate.free_gb == pytest.approx(10**15 / 1e9)

    def test_insufficient_when_headroom_thin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=100, used=95, free=5),
        )
        # required (4.9) exceeds 90% of the 5 bytes free (4.5).
        estimate = assess_bytes(4.9, tmp_path)
        assert estimate.sufficient is False

    def test_str_reports_verdict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=10**15, used=0, free=10**15),
        )
        estimate = assess_bytes(1e6, tmp_path)
        assert "OK" in str(estimate)

    def test_walks_up_to_nearest_existing_ancestor(self, tmp_path, monkeypatch):
        # The target path itself doesn't exist yet (it's about to be
        # written); the free-space check must resolve to the nearest real
        # ancestor directory instead of raising.
        seen = {}

        def fake_disk_usage(p):
            seen["path"] = Path(p)
            return types.SimpleNamespace(total=10**15, used=0, free=10**15)

        monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
        assess_bytes(1e6, tmp_path / "not_yet_written.h5ad")
        assert seen["path"] == tmp_path.resolve()

    def test_is_frozen_dataclass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=10**15, used=0, free=10**15),
        )
        estimate = assess_bytes(1e6, tmp_path)
        with pytest.raises(Exception):
            estimate.required_bytes = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cross-platform robustness: shutil.disk_usage (and path resolution) can
# fail in ways that are platform- or filesystem-specific -- an unreachable
# network mount, a permission error on a Windows junction, an ejected
# drive. These simulate that generically via OSError/PermissionError
# (a subclass of OSError) rather than relying on a real Windows/network
# environment, since the code path only branches on the exception type.
# ---------------------------------------------------------------------------

def _raising(exc: BaseException):
    """A callable that raises *exc* regardless of the arguments it's given.

    Used with ``monkeypatch.setattr`` to make ``shutil.disk_usage`` or
    ``Path.resolve`` fail on demand -- plain ``lambda``s can't contain a
    ``raise`` statement.
    """
    def _fn(*_args, **_kwargs):
        raise exc
    return _fn


class TestUnknownFreeSpace:
    def test_free_bytes_none_when_disk_usage_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage", _raising(PermissionError("network mount unavailable")),
        )
        estimate = assess_bytes(1e9, tmp_path)
        assert estimate.free_bytes is None
        assert estimate.free_gb is None

    def test_sufficient_fails_open_when_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage", _raising(OSError("disk_usage unavailable")),
        )
        estimate = assess_bytes(1e15, tmp_path)  # absurdly large; still "sufficient"
        assert estimate.sufficient is True

    def test_str_reports_unknown_rather_than_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage", _raising(OSError("disk_usage unavailable")),
        )
        estimate = assess_bytes(1e9, tmp_path)
        assert "unknown" in str(estimate)
        assert "0.0 GB free" not in str(estimate)

    def test_path_resolution_failure_does_not_raise(self, tmp_path, monkeypatch):
        # A broken symlink or an inaccessible network mount can make
        # Path.resolve() itself raise; _safe_resolve falls back to the
        # unresolved (but already-absolute, from tmp_path) path, so the
        # free-space lookup can still succeed on the fallback -- the
        # invariant under test is simply that nothing raises.
        monkeypatch.setattr(Path, "resolve", _raising(OSError("broken symlink")))
        estimate = assess_bytes(1e9, tmp_path / "sub" / "file.h5ad")
        assert isinstance(estimate, DiskEstimate)

    def test_warn_if_disk_space_low_stays_silent_when_disk_usage_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage", _raising(OSError("disk_usage unavailable")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_disk_space_low(required_bytes=1e9, output_path=tmp_path)

    def test_warn_if_disk_space_low_still_flags_large_write_when_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage", _raising(OSError("disk_usage unavailable")),
        )
        with pytest.warns(UserWarning, match="approximately"):
            warn_if_disk_space_low(required_bytes=25 * 1e9, output_path=tmp_path, large_file_gb=20.0)


# ---------------------------------------------------------------------------
# warn_if_disk_space_low
# ---------------------------------------------------------------------------

class TestWarnIfDiskSpaceLow:
    def test_warns_when_headroom_thin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=100, used=95, free=5),
        )
        with pytest.warns(UserWarning, match="No space left on device|estimated disk usage"):
            warn_if_disk_space_low(required_bytes=4.9, output_path=tmp_path)

    def test_warns_for_large_output_even_with_headroom(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=10**15, used=0, free=10**15),
        )
        with pytest.warns(UserWarning, match="approximately"):
            warn_if_disk_space_low(required_bytes=25 * 1e9, output_path=tmp_path, large_file_gb=20.0)

    def test_no_warning_when_plenty_of_room(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=10**15, used=0, free=10**15),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_disk_space_low(required_bytes=1e6, output_path=tmp_path)

    def test_context_label_prefixes_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=100, used=95, free=5),
        )
        with pytest.warns(UserWarning, match="my_context: "):
            warn_if_disk_space_low(required_bytes=4.9, output_path=tmp_path, context="my_context")

    def test_insufficient_takes_priority_over_large_file(self, tmp_path, monkeypatch):
        # A tiny disk that is also "large" relative to what's free should
        # get the actionable "no space" message, not just "heads up".
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda p: types.SimpleNamespace(total=1e9, used=0, free=1e9),
        )
        with pytest.warns(UserWarning, match="estimated disk usage"):
            warn_if_disk_space_low(required_bytes=25 * 1e9, output_path=tmp_path, large_file_gb=20.0)
