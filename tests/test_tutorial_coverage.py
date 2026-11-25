"""The basic tutorials (``docs/notebooks/basics_*.ipynb``) run or name every
public function, so a new one cannot be added without showing it there."""

from __future__ import annotations

import inspect
import json
import re
import types
from pathlib import Path

import crispyx as cx

NOTEBOOKS = sorted((Path(__file__).parents[1] / "docs" / "notebooks").glob("basics_*.ipynb"))
NAMESPACES = ("pp", "pb", "tl", "pl")

#: Public names no basic tutorial has to show, each with its reason.
EXCLUDED = {
    "cx.plot_benchmark_comparison": "plots benchmark tables, which no tutorial produces",
}


#: Top-level functions a namespace method calls on the way, without being
#: the function it wraps.
HELPERS = {"resolve_data_path"}


def _names_in(code: types.CodeType) -> set[str]:
    """The global names a function's code refers to, nested functions included."""
    nested = [const for const in code.co_consts if isinstance(const, types.CodeType)]
    return set(code.co_names).union(*map(_names_in, nested))


def _public_names() -> list[str]:
    """Every namespace method, and every top-level name that is not a
    function a namespace method wraps (those are shown under their namespace
    name). A function counts as wrapped when a method's code calls it and it
    is not one of :data:`HELPERS`; classes and type aliases never do."""
    methods = [
        (namespace, method)
        for namespace in NAMESPACES
        for method in dir(getattr(cx, namespace))
        if not method.startswith("_")
    ]
    called = set().union(*(
        _names_in(getattr(type(getattr(cx, namespace)), method).__code__) for namespace, method in methods
    ))
    wrapped = {name for name in called - HELPERS if inspect.isfunction(getattr(cx, name, None))}
    names = [f"cx.{namespace}.{method}" for namespace, method in methods]
    names += [f"cx.{name}" for name in cx.__all__ if name not in (*NAMESPACES, "__version__", *wrapped)]
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
