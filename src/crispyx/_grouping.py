"""Shared argument handling for grouped analyses."""

from __future__ import annotations

import hashlib

import numpy as np


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
