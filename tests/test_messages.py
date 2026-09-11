"""Tests for the shared verbose-print/warning formatting in crispyx._messages."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from crispyx._disk import DiskEstimate, format_bytes
from crispyx._messages import (
    print_disk_estimate,
    print_done,
    print_reading,
    print_saving,
    vprint,
    warn,
)


def _pretend_public_function():
    """Stand-in for a real crispyx function's body that raises a warning."""
    warn("pretend_fn", "boom")


class TestVprint:
    def test_prints_at_or_above_level(self, capsys):
        vprint(1, "pp.demo", "hello")
        assert capsys.readouterr().out == "[cx] pp.demo: hello\n"

    def test_silent_below_level(self, capsys):
        vprint(0, "pp.demo", "hello")
        assert capsys.readouterr().out == ""

    def test_bool_true_counts_as_level_one(self, capsys):
        vprint(True, "pp.demo", "hello")
        assert "[cx] pp.demo: hello" in capsys.readouterr().out

    def test_custom_level_gate(self, capsys):
        vprint(1, "pp.demo", "detail", level=2)
        assert capsys.readouterr().out == ""
        vprint(2, "pp.demo", "detail", level=2)
        assert "detail" in capsys.readouterr().out


class TestReadingSavingDone:
    def test_print_reading_format(self, capsys):
        print_reading(1, "pp.normalize_total_log1p", "/data/screen.h5ad")
        assert capsys.readouterr().out == "[cx] pp.normalize_total_log1p: Reading /data/screen.h5ad\n"

    def test_print_saving_uses_literal_arrow(self, capsys):
        print_saving(1, "pp.normalize_total_log1p", "/out/screen.h5ad")
        out = capsys.readouterr().out
        assert out == "[cx] pp.normalize_total_log1p: Saving → /out/screen.h5ad\n"

    def test_print_done_format(self, capsys):
        print_done(1, "pp.qc_summary", "100/100 cells kept (100%)")
        assert capsys.readouterr().out == "[cx] pp.qc_summary: Done  100/100 cells kept (100%)\n"

    def test_all_three_silent_by_default_level(self, capsys):
        print_reading(0, "pp.x", "p")
        print_saving(0, "pp.x", "p")
        print_done(0, "pp.x", "m")
        assert capsys.readouterr().out == ""


class TestFormatBytes:
    """Sizes must stay readable at both ends of the range the package spans:
    a 50 MB subset's matrix and a 40 GB screen's."""

    @pytest.mark.parametrize(
        "n_bytes, expected",
        [
            (0, "0 B"),
            (512, "512 B"),
            (49_300_000, "49.3 MB"),
            (492_600_000, "492.6 MB"),
            (40e9, "40.0 GB"),
            (3.2e12, "3.2 TB"),
            (None, "unknown"),
        ],
    )
    def test_scales_the_unit_to_the_value(self, n_bytes, expected):
        assert format_bytes(n_bytes) == expected

    def test_sub_gigabyte_estimate_does_not_render_as_zero(self, capsys):
        estimate = DiskEstimate(
            required_bytes=49_300_000, free_bytes=2.8e12, path=Path("/tmp")
        )
        print_disk_estimate(1, "pp.demo", estimate)
        out = capsys.readouterr().out
        assert "49.3 MB" in out and "0.0 GB" not in out
        assert "49.3 MB" in str(estimate)


class TestWarn:
    def test_message_is_context_prefixed(self):
        with pytest.warns(UserWarning, match=r"^my_context: something happened$"):
            warn("my_context", "something happened")

    def test_default_category_is_user_warning(self):
        with pytest.warns(UserWarning):
            warn("ctx", "msg")

    def test_custom_category(self):
        with pytest.warns(DeprecationWarning, match="ctx: msg"):
            warn("ctx", "msg", category=DeprecationWarning)

    def test_stacklevel_attributes_to_caller_of_warn(self):
        """warn()'s default stacklevel=2 should attribute the warning to
        whichever function *calls* the function containing warn() -- the
        same place a direct `warnings.warn(msg, stacklevel=2)` in that
        function's body would attribute to -- not to _messages.py, and not
        to _pretend_public_function's own line either.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            this_call_line = sys._getframe().f_lineno + 1
            _pretend_public_function()

        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert caught[0].lineno == this_call_line
