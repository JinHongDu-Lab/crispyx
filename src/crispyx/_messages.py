"""Shared formatting for crispyx's "[cx] {name}: {message}" verbose
convention and context-prefixed warnings.

Two independent channels stay as they are: ``logger.debug``/``logger.info``
remain for developer-oriented detail and are untouched by this module. This
module only standardizes the user-facing ``print``/``warnings.warn`` side,
which had drifted (different arrow glyphs, inconsistent Reading/Saving/Done
structure, print prefixes that didn't track a renamed function) across the
~44 call sites that used to hand-roll it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._disk import DiskEstimate


def vprint(verbose: int | bool, name: str, message: str, *, level: int = 1) -> None:
    """Print ``[cx] {name}: {message}`` when ``int(verbose) >= level``."""
    if int(verbose) >= level:
        print(f"[cx] {name}: {message}")


def print_reading(verbose: int | bool, name: str, path) -> None:
    """``vprint`` for the standard "about to read this file" announcement."""
    vprint(verbose, name, f"Reading {path}")


def print_saving(verbose: int | bool, name: str, path) -> None:
    """``vprint`` for the standard "about to write this file" announcement."""
    vprint(verbose, name, f"Saving → {path}")


def print_done(verbose: int | bool, name: str, message: str) -> None:
    """``vprint`` for the standard completion summary, e.g. shape counts."""
    vprint(verbose, name, f"Done  {message}")


def print_disk_estimate(verbose: int | bool, name: str, estimate: "DiskEstimate") -> None:
    """``vprint`` companion to ``warn_if_disk_space_low``, shown whether or
    not the warning fired -- the warning stays unconditional (safety-relevant),
    this line is the ``verbose``-gated confirmation of what was estimated.
    """
    if estimate.free_bytes is None:
        message = f"estimated disk usage: {estimate.required_gb:.1f} GB (free space unknown)"
    else:
        message = f"estimated disk usage: {estimate.required_gb:.1f} GB ({estimate.free_gb:.1f} GB free)"
    vprint(verbose, name, message)


def warn(context: str, message: str, *, category: type[Warning] = UserWarning, stacklevel: int = 2) -> None:
    """``warnings.warn`` with the ``"{context}: "`` prefix ``_disk.py`` already uses.

    Warnings are never gated on ``verbose`` -- they signal something the
    caller should know about regardless of verbosity, the same way
    ``_disk.py``'s disk-space warnings already work.
    """
    warnings.warn(f"{context}: {message}", category, stacklevel=stacklevel + 1)


__all__ = [
    "vprint",
    "print_reading",
    "print_saving",
    "print_done",
    "print_disk_estimate",
    "warn",
]
