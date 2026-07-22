"""Shared argument handling for grouped analyses."""

from __future__ import annotations


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
