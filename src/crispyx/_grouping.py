"""Shared argument handling for grouped analyses."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _group_seed(random_state: int, key: tuple[str, ...]) -> np.random.SeedSequence:
    """Deterministic per-group seed, independent of dict/iteration order.

    Derives from a stable hash of ``key`` (``hashlib.blake2b``, not Python's
    randomized ``hash()``) so a given ``(random_state, key)`` pair always
    produces the same seed across processes and runs. Shared by any grouped
    sampling that needs one independent RNG stream per group (pseudo-bulk
    bootstrap, stratified subsampling).
    """
    digest = hashlib.blake2b(
        "\x1f".join(key).encode("utf-8"), digest_size=8
    ).digest()
    key_seed = int.from_bytes(digest, "little", signed=False)
    return np.random.SeedSequence(
        [int(random_state), key_seed & 0xFFFFFFFF, key_seed >> 32]
    )


def resolve_group_reference_aliases(
    *,
    perturbation_column: str | None,
    groupby: str | None,
    control_label: str | None,
    reference: str | None,
    fn_name: str,
) -> tuple[str, str | None]:
    """Resolve the Scanpy-style grouping and reference aliases.

    This is shared by differential-expression functions and generic grouped
    statistics so their public argument behavior and error messages stay in
    sync.
    """
    if groupby is not None and perturbation_column is not None:
        raise TypeError(
            f"{fn_name}() received both 'perturbation_column' and 'groupby'; "
            "they are aliases for the same parameter — pass only one."
        )
    if groupby is not None:
        perturbation_column = groupby
    if perturbation_column is None:
        raise TypeError(
            f"{fn_name}() requires either 'perturbation_column' or its alias 'groupby'."
        )

    if reference is not None and control_label is not None:
        raise TypeError(
            f"{fn_name}() received both 'control_label' and 'reference'; "
            "they are aliases for the same parameter — pass only one."
        )
    if reference is not None:
        control_label = reference
    return perturbation_column, control_label


def canonical_call_args(
    call_args: dict, *, perturbation_column: str, control_label: str | None,
) -> dict:
    """``call_args`` (a function's ``locals()`` at entry) with the grouping
    column and the control under their canonical names, as
    :func:`resolve_group_reference_aliases` resolved them, so a call made
    through ``groupby``/``reference`` fingerprints the same as one made
    through ``perturbation_column``/``control_label``."""
    args = {k: v for k, v in call_args.items() if k not in ("groupby", "reference")}
    return {**args, "perturbation_column": perturbation_column, "control_label": control_label}


# Cells with a missing label carry this, which no real group can be named, so
# they fall into no comparison (``str(NaN)`` would make a group named "nan").
UNLABELLED = "\0unlabelled"


def cell_labels(obs: pd.DataFrame, labels: str | pd.Series) -> tuple[pd.Series, str]:
    """Each cell's value, in ``obs`` order (NaN = the cell is not used), and its
    name: ``labels`` is an ``obs`` column, or a ``pd.Series`` indexed by cell
    (``obs_names``), which leaves the file untouched."""
    names = pd.Index(obs.index).astype(str)
    if isinstance(labels, pd.Series):
        series = labels.set_axis(labels.index.astype(str))
        if not series.index.is_unique:
            raise ValueError("A labels Series must be indexed by cell, each cell once.")
        absent = names[~names.isin(series.index)]
        if len(absent):
            raise ValueError(
                f"The labels Series has no label for {len(absent)} of {len(names)} cells "
                f"(e.g. {absent[:3].tolist()}); index it by obs_names, with NaN for cells to leave out."
            )
        return series.reindex(names), str(labels.name or "perturbation")
    if labels not in obs.columns:
        raise KeyError(f"Column '{labels}' was not found in adata.obs. Available columns: {list(obs.columns)}")
    return obs[labels].set_axis(names), str(labels)


def label_strings(labels: pd.Series) -> np.ndarray:
    """Per-cell labels as strings, missing ones as :data:`UNLABELLED`."""
    return labels.astype(object).where(labels.notna(), UNLABELLED).astype(str).to_numpy()
