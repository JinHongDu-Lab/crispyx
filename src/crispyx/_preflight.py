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
from .data import AnnData, resolve_data_path, resolve_output_path

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
    data: str | Path | AnnData,
    **kwargs,
) -> dict[str, DiskEstimate]:
    """Estimate the disk space a crispyx function will need, before running it.

    This is a standalone, on-demand query -- separate from the automatic
    warning crispyx already emits from inside a real call when its own
    estimate looks tight. Resolvers read only cheap ``obs``/``uns`` metadata
    in backed mode to reproduce the group/batch counts the real function
    would compute in its own preamble; they never read ``X`` in full. The one
    exception is ``format_mismatch_policy="auto"`` (the default) on a source
    stored off the axis the function streams: deciding whether that call
    would write a temporary fast-axis copy means measuring the filesystem, so
    the query makes the same bounded 64 MB probe read of ``X`` the real call
    makes. The measurement is cached per file, so a call made later in the
    same process does not pay for it again and cannot resolve the decision
    differently than this query did.

    Also available as ``cx.tl.estimate_disk_usage`` for Scanpy-style
    namespace discovery (the same pattern as ``compute_overlap``, which is
    both top-level and under ``cx.tl``); the two are the same function and
    always agree.

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
        ``format_mismatch_policy`` matters too, as do ``chunk_size`` and
        ``memory_limit_gb`` through it: those two set how many chunks the run
        streams, which is what decides whether it converts the source and so
        whether ``"scratch"`` appears at all. Pass the ones you will pass to
        the real call; omitted, they resolve the way the real call resolves
        them.
        Keywords that genuinely cannot change the estimate (``verbose``,
        ``n_jobs``, ...) are accepted and ignored.

    Returns
    -------
    dict[str, DiskEstimate]
        Keyed by filesystem location -- ``"tempdir"`` for disk-backed
        intermediate accumulators in ``$TMPDIR``, ``"output"`` for the final
        result file, and ``"scratch"`` for the temporary fast-axis copy a
        ``format_mismatch_policy="convert"`` call writes beside the output.
        Not every function uses all of them; a conversion function like
        ``convert_to_csc`` only has ``"output"``. ``"output"`` and
        ``"scratch"`` are assessed where the target function would write:
        pass the same ``output_path`` (or ``output_dir``) you will pass to
        it; with neither, the source file's directory.

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
    # Same resolution the target functions apply to their own output (the
    # suffix only affects the filename, never the directory).
    output_parent = resolve_output_path(
        path, suffix="estimate",
        output_path=kwargs.get("output_path"), output_dir=kwargs.get("output_dir"),
        data_name=kwargs.get("data_name"),
    ).parent
    return {
        location: assess_bytes(
            required_bytes, tempdir if location in _TEMPDIR_LOCATIONS else output_parent,
        )
        for location, required_bytes in required_by_location.items()
    }


__all__ = ["estimate_disk_usage"]
