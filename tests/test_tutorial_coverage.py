"""The basic tutorials (``docs/notebooks/basics_*.ipynb``) run or name every
public function, so a new one cannot be added without showing it there."""

from __future__ import annotations

import json
import re
from pathlib import Path

import crispyx as cx
import crispyx._namespaces as namespaces

NOTEBOOKS = sorted((Path(__file__).parents[1] / "docs" / "notebooks").glob("basics_*.ipynb"))
NAMESPACES = ("pp", "pb", "tl", "pl")

#: Public names no basic tutorial has to show, each with its reason.
EXCLUDED = {
    "cx.plot_benchmark_comparison": "plots benchmark tables, which no tutorial produces",
}


def _public_names() -> list[str]:
    """Every namespace method, and every top-level name that is not one of
    the functions the namespaces wrap (those are shown under their
    namespace name)."""
    names = [
        f"cx.{namespace}.{method}"
        for namespace in NAMESPACES
        for method in dir(getattr(cx, namespace))
        if not method.startswith("_")
    ]
    names += [
        f"cx.{name}"
        for name in cx.__all__
        if name not in (*NAMESPACES, "__version__") and not hasattr(namespaces, name)
    ]
    return names


def _tutorial_text() -> str:
    cells = [cell for notebook in NOTEBOOKS for cell in json.loads(notebook.read_text())["cells"]]
    return "\n".join("".join(cell["source"]) for cell in cells)


def _shown(name: str, text: str) -> bool:
    return re.search(re.escape(name) + r"(?!\w)", text) is not None


def test_every_public_function_is_in_a_basic_tutorial():
    assert NOTEBOOKS, "no docs/notebooks/basics_*.ipynb found"
    text = _tutorial_text()
    missing = [name for name in _public_names() if name not in EXCLUDED and not _shown(name, text)]
    assert not missing, (
        f"not run or named in {', '.join(notebook.name for notebook in NOTEBOOKS)}: {', '.join(missing)}. "
        "Add each to the notebook of its area (a code cell, or its closing function table)."
    )


def test_exclusions_are_still_needed():
    text = _tutorial_text()
    public = set(_public_names())
    stale = [name for name in EXCLUDED if name not in public or _shown(name, text)]
    assert not stale, f"remove from EXCLUDED (no longer public, or now shown in a tutorial): {', '.join(stale)}"
