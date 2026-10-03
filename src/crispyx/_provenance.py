"""Provenance stamped into crispyx outputs, and the check that reuses them.

Every output a later call may reuse instead of recomputing carries
``uns["crispyx"]``: the crispyx ``version`` that wrote it, whether that was an
``editable`` (development) install, the output ``kind`` and its layout
``schema``, and the ``fingerprint`` of the call that produced it (a
:func:`crispyx._checkpoint.run_fingerprint`, stored as JSON). An existing
output is reused only when its kind, schema and fingerprint all match the
call being made, so an edited input, a changed argument or a file written
by a crispyx with another layout is recomputed rather than returned stale.

The version and install type are recorded, not compared: a new release with
the same layout reuses an old result, while ``editable`` makes results from
a development checkout visible after the fact.
"""

from __future__ import annotations

import functools
import json
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import anndata as ad
import h5py

KEY = "crispyx"

#: Layout version of each output kind. Bump an entry whenever what that kind
#: stores on disk changes, so existing files of the old layout are recomputed
#: instead of being read under the new one.
SCHEMAS = {
    "de_result": 1,
    "pseudobulk": 1,
    "pseudobulk_effects": 1,
    "batch": 1,
    "sorted": 1,
    "standardized": 1,
}

# Fingerprint keys that identify the input file (see ``run_fingerprint``).
_SOURCE_KEYS = ("source", "source_size", "source_mtime_ns")


@functools.cache
def _is_editable_install() -> bool:
    try:
        text = distribution("crispyx").read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    if not text:
        return False
    return bool(json.loads(text).get("dir_info", {}).get("editable", False))


def stamp(kind: str, fingerprint: dict) -> dict:
    """The ``uns["crispyx"]`` entry for an output of ``kind`` made by the
    call whose identity is ``fingerprint``."""
    from . import __version__

    return {
        "version": __version__,
        "editable": _is_editable_install(),
        "kind": kind,
        "schema": SCHEMAS[kind],
        "fingerprint": json.dumps(fingerprint, sort_keys=True),
    }


def write_stamp(path: Path, provenance: dict) -> None:
    """Set ``uns["crispyx"]`` of the existing h5ad at ``path`` in place, for
    an output that is not written through anndata. The entry is encoded by
    anndata (into a scratch file) and copied across, so it reads back like
    one written with the rest of the file."""
    with tempfile.TemporaryDirectory(prefix="cx_stamp_") as tmpdir:
        donor = Path(tmpdir) / "stamp.h5ad"
        ad.AnnData(uns={KEY: provenance}).write(donor)
        with h5py.File(donor, "r") as src, h5py.File(path, "r+") as dest:
            uns = dest["uns"]
            if KEY in uns:
                del uns[KEY]
            src.copy(src[f"uns/{KEY}"], uns, name=KEY)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "item"):
        return _decode(value.item())
    return value


def read_stamp(path: Path) -> dict | None:
    """``uns["crispyx"]`` of the h5ad at ``path``, or ``None`` if it has none
    (or cannot be opened)."""
    try:
        with h5py.File(path, "r") as handle:
            group = handle.get(f"uns/{KEY}")
            if not isinstance(group, h5py.Group):
                return None
            return {
                name: _decode(obj[()])
                for name, obj in group.items()
                if isinstance(obj, h5py.Dataset) and obj.shape == ()
            }
    except OSError:
        return None


def reuse_mismatch(path: Path, kind: str, fingerprint: dict) -> str | None:
    """Why the output at ``path`` cannot stand in for this call, or ``None``
    when it can.

    ``fingerprint`` must already be JSON-normalised, as
    :func:`~crispyx._checkpoint.run_fingerprint` returns it.
    """
    stored = read_stamp(path)
    if stored is None:
        return "it has no crispyx provenance (written by crispyx 0.1.6 or earlier, or not by crispyx)"
    if stored.get("kind") != kind or stored.get("schema") != SCHEMAS[kind]:
        return (
            f"it was written by crispyx {stored.get('version', '?')} as "
            f"{stored.get('kind', '?')} layout {stored.get('schema', '?')}; "
            f"this version writes {kind} layout {SCHEMAS[kind]}"
        )
    try:
        previous = json.loads(stored.get("fingerprint", ""))
    except (TypeError, json.JSONDecodeError):
        return "its recorded inputs cannot be read"
    if previous == fingerprint:
        return None
    return _describe_difference(previous, fingerprint)


def _describe_difference(previous: dict, current: dict) -> str:
    if previous.get("source") != current.get("source"):
        return f"it was computed from {previous.get('source')}"
    if any(previous.get(key) != current.get(key) for key in _SOURCE_KEYS):
        return "the input file has changed since"
    old_params = previous.get("params", {})
    new_params = current.get("params", {})
    for name in sorted(set(old_params) | set(new_params)):
        if old_params.get(name) != new_params.get(name):
            return f"argument '{name}' differs"
    for name in sorted(set(previous) | set(current)):
        if previous.get(name) != current.get(name):
            return f"'{name}' differs"
    return "its recorded inputs differ"
