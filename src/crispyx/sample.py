"""Streaming, stratified/cluster subsampling for large ``.h5ad`` datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from . import _messages
from ._grouping import _group_seed
from .data import (
    AnnData,
    read_backed,
    resolve_data_path,
    resolve_output_path,
    write_filtered_subset,
)


@dataclass
class _StratumOutcome:
    key: tuple[str, ...]
    pool_size: int
    target: int
    kept: int
    status: str  # "sampled" | "dropped" | "kept_in_full"


def _normalise_groupby(groupby: str | Sequence[str] | None) -> list[str]:
    if groupby is None:
        return []
    if isinstance(groupby, str):
        columns = [groupby]
    else:
        columns = [str(column) for column in groupby]
    if any(not column for column in columns):
        raise ValueError("groupby must contain only non-empty obs column names")
    if len(set(columns)) != len(columns):
        raise ValueError("groupby must not contain duplicate column names")
    return columns


def subsample(
    data: str | Path | AnnData | ad.AnnData,
    *,
    n: int | None = None,
    frac: float | None = None,
    groupby: str | Sequence[str] | None = None,
    unit: str = "cell",
    drop_insufficient: bool = True,
    random_state: int = 0,
    chunk_size: int = 4096,
    output_path: str | Path | None = None,
    data_name: str | None = None,
    verbose: int | bool = True,
) -> AnnData:
    """Stream a stratified or cluster-sampled subset of an h5ad file to disk.

    Cells are never loaded to decide the subset — the sampling mask is built
    entirely from ``.obs`` (fast, metadata-only), then handed to
    :func:`crispyx.data.write_filtered_subset` to stream ``X``, ``layers``,
    ``obsm``/``obsp``/``varm``, and ``uns`` into the output the same way
    every other crispyx filtering function does.

    Parameters
    ----------
    data
        Path to an h5ad file, :class:`crispyx.AnnData`, or a backed AnnData.
    n, frac
        Exactly one must be given. ``n`` draws exactly this many items from
        every stratum; ``frac`` draws ``round(frac * pool_size)`` items,
        matching ``pandas.DataFrameGroupBy.sample(n=, frac=)`` semantics —
        both are per-stratum quantities, not a global total.
    groupby
        One obs column, a sequence of obs columns, or ``None`` (default).
        Defines the strata that are sampled independently of each other.
        ``None`` means a single global stratum (the whole dataset).
    unit
        ``"cell"`` (default) draws individual cells. Any other value must
        name an obs column (e.g. ``"batch"``): each unique value of that
        column *within a stratum* becomes one indivisible sampling item — a
        chosen unit keeps ALL of its cells in that stratum, an unchosen unit
        keeps NONE. ``n``/``frac`` then count units, not cells.
    drop_insufficient
        When a stratum's item pool is smaller than the requested count: if
        True (default), the stratum is excluded entirely from the output; if
        False, all of its cells are kept instead. Either way, every affected
        stratum is reported via a warning, independent of ``verbose``.
    random_state
        Seed for the per-stratum RNG. Sampling is deterministic for a fixed
        seed and independent of stratum iteration order or ``chunk_size``.
    chunk_size
        Cells streamed per chunk when writing the output.
    output_path
        Explicit output path. If None, derived from ``data_name``.
    data_name
        Custom output name suffix.
    verbose
        Print how many cells were kept and across how many strata.

    Returns
    -------
    AnnData
        Backed AnnData wrapper pointing to the subsampled output file.
    """
    if (n is None) == (frac is None):
        raise TypeError("subsample() requires exactly one of 'n' or 'frac'.")
    if n is not None:
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        n = int(n)
    if frac is not None:
        if not (0 < frac <= 1):
            raise ValueError("frac must be in (0, 1]")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer")

    columns = _normalise_groupby(groupby)
    path = resolve_data_path(data)

    backed = read_backed(path)
    try:
        missing_columns = [c for c in columns if c not in backed.obs.columns]
        if missing_columns:
            raise KeyError(
                f"Grouping column(s) {missing_columns} were not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        if unit != "cell" and unit not in backed.obs.columns:
            raise KeyError(
                f"unit={unit!r} is not 'cell' and was not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        if columns:
            multi = pd.MultiIndex.from_frame(backed.obs[columns].astype(str))
            stratum_codes, stratum_keys_raw = pd.factorize(multi, sort=False)
            stratum_keys = [
                tuple(str(v) for v in key) if isinstance(key, tuple) else (str(key),)
                for key in stratum_keys_raw.tolist()
            ]
        else:
            stratum_codes = np.zeros(n_obs, dtype=np.int64)
            stratum_keys = [()]
        unit_values = (
            backed.obs[unit].astype(str).to_numpy() if unit != "cell" else None
        )
    finally:
        backed.file.close()

    cell_mask = np.zeros(n_obs, dtype=bool)
    outcomes: list[_StratumOutcome] = []

    rows_by_stratum = pd.Series(np.arange(n_obs)).groupby(stratum_codes).indices
    for stratum_code, rows in rows_by_stratum.items():
        key = stratum_keys[stratum_code]
        rows = np.asarray(rows)

        if unit == "cell":
            pool = rows
            pool_size = pool.size
        else:
            local_units = unit_values[rows]
            pool = np.unique(local_units)
            pool_size = pool.size

        target = n if n is not None else int(round(frac * pool_size))

        if pool_size < target:
            if drop_insufficient:
                selected = pool[:0]
                kept = 0
                status = "dropped"
            else:
                selected = pool
                kept = pool_size
                status = "kept_in_full"
        else:
            rng = np.random.default_rng(_group_seed(random_state, key))
            choice = rng.choice(pool_size, size=target, replace=False)
            selected = pool[choice]
            kept = target
            status = "sampled"

        outcomes.append(_StratumOutcome(key, pool_size, target, kept, status))

        if unit == "cell":
            cell_mask[selected] = True
        elif selected.size:
            local_units = unit_values[rows]
            cell_mask[rows[np.isin(local_units, selected)]] = True

    affected = [o for o in outcomes if o.status != "sampled"]
    if affected:
        label = lambda key: "(all)" if not key else "/".join(key)
        preview = "; ".join(
            f"{label(o.key)}: pool={o.pool_size} < requested={o.target} -> "
            f"{'dropped' if o.status == 'dropped' else 'kept in full'}"
            for o in affected[:20]
        )
        more = f" (+{len(affected) - 20} more)" if len(affected) > 20 else ""
        _messages.warn(
            "pp.subsample",
            f"{len(affected)}/{len(outcomes)} stratum/strata had fewer than the "
            f"requested count and were "
            f"{'dropped' if drop_insufficient else 'kept in full'}: {preview}{more}",
        )

    kept_total = int(cell_mask.sum())
    _messages.print_done(
        verbose, "pp.subsample",
        f"{kept_total}/{n_obs} cells kept ({100 * kept_total / n_obs:.0f}%) "
        f"across {len(outcomes)} stratum/strata (unit={unit!r})",
    )

    resolved_output = resolve_output_path(
        path, suffix="subsample", data_name=data_name, output_path=output_path,
    )
    gene_mask = np.ones(n_vars, dtype=bool)
    write_filtered_subset(
        path,
        cell_mask=cell_mask,
        gene_mask=gene_mask,
        output_path=resolved_output,
        chunk_size=chunk_size,
    )
    return AnnData(resolved_output)
