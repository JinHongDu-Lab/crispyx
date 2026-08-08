"""Public on-demand disk-usage query: ``crispyx.estimate_disk_usage()``.

This sits at the same import-graph layer as ``_namespaces.py``: it imports
the disk-usage resolver from every module that owns one, and nothing
imports it back. It cannot live inside ``_disk.py`` itself, since
``pseudobulk.py``/``de.py``/``batch.py``/``data.py`` already import *from*
``_disk.py`` for the automatic ``warn_if_disk_space_low`` warnings -- adding
the reverse import would be circular.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from . import batch as _batch
from . import data as _data
from . import de as _de
from . import pseudobulk as _pseudobulk
from ._disk import DiskEstimate, assess_bytes
from .data import AnnData, resolve_data_path

_RESOLVERS: dict[str, Callable[..., dict[str, float]]] = {
    "compute_normalized_effects": _pseudobulk._estimate_shape_for_normalized_effects,
    "aggregate_pseudobulk": _pseudobulk._estimate_shape_for_aggregate_pseudobulk,
    "compute_pseudobulk_effects": _pseudobulk._estimate_shape_for_pseudobulk_effects,
    "t_test": _de._estimate_shape_for_t_test,
    "wilcoxon_test": _de._estimate_shape_for_wilcoxon_test,
    "nb_glm_test": _de._estimate_shape_for_nb_glm_test,
    "shrink_lfc": _de._estimate_shape_for_shrink_lfc,
    "batch_process": _batch._estimate_shape_for_batch_process,
    "convert_to_csc": _data._estimate_shape_for_conversion,
    "convert_to_csr": _data._estimate_shape_for_conversion,
    "normalize_total_log1p": _data._estimate_shape_for_normalize_total_log1p,
}

# Locations that live in the system temp directory rather than next to the
# output file. Anything not listed here is assessed against the output path.
_TEMPDIR_LOCATIONS = {"tempdir"}


def estimate_disk_usage(
    func: str | Callable,
    data: "str | Path | AnnData",
    **kwargs,
) -> dict[str, DiskEstimate]:
    """Estimate the disk space a crispyx function will need, before running it.

    This is a standalone, on-demand query -- separate from the automatic
    warning crispyx already emits from inside a real call when its own
    estimate looks tight. It never touches ``X``: resolvers read only cheap
    ``obs``/``uns`` metadata in backed mode to reproduce the group/batch
    counts the real function would compute in its own preamble.

    Parameters
    ----------
    func
        Either the function's name (e.g. ``"compute_normalized_effects"``,
        ``"t_test"``, ``"convert_to_csc"``) or the function object itself
        (e.g. ``crispyx.t_test``). Both are accepted: the string form is the
        primary interface, and accepting the callable too costs one
        ``getattr`` while removing a class of typo'd-string bugs.
    data
        Path to an h5ad file, or a backed/wrapped AnnData -- same meaning as
        every other crispyx function's ``data`` parameter.
    **kwargs
        The subset of the target function's keyword arguments that affect
        group/shape counts (e.g. ``perturbation_column``, ``control_label``,
        ``batch_column``, ``groupby``, ``perturbations``, ``min_cells``).
        Performance-only keywords the caller might also pass (``chunk_size``,
        ``memory_limit_gb``, ``verbose``, ``output_path``, ...) are accepted
        and ignored, since they don't change the estimate.

    Returns
    -------
    dict[str, DiskEstimate]
        Keyed by filesystem location -- ``"tempdir"`` for disk-backed
        intermediate accumulators, ``"output"`` for the final result file.
        Not every function uses both; a conversion function like
        ``convert_to_csc`` only has ``"output"``.

    Examples
    --------
    >>> import crispyx as cx
    >>> cx.estimate_disk_usage(
    ...     "compute_normalized_effects", "screen.h5ad",
    ...     perturbation_column="guide_target", batch_column="gem_group",
    ... )
    {'tempdir': ..., 'output': ...}
    """
    name = getattr(func, "__name__", func)
    try:
        resolver = _RESOLVERS[name]
    except KeyError:
        raise ValueError(
            f"No disk-usage estimator registered for {name!r}. Supported: "
            f"{sorted(_RESOLVERS)}"
        ) from None

    path = resolve_data_path(data)
    required_by_location = resolver(path, **kwargs)
    tempdir = Path(tempfile.gettempdir())
    return {
        location: assess_bytes(
            required_bytes, tempdir if location in _TEMPDIR_LOCATIONS else path.parent,
        )
        for location, required_bytes in required_by_location.items()
    }


__all__ = ["estimate_disk_usage"]
