"""Plotting utilities for crispyx with Scanpy-style helpers.

These functions are designed to work with on-disk AnnData objects and
avoid loading full count matrices into memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Literal, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import (
    AnnData,
    OverlapResult,
    iter_matrix_chunks,
    normalize_total_block,
    read_backed,
    resolve_data_path,
)
from .qc import QualityControlResult

logger = logging.getLogger(__name__)

PlotInput = str | Path | AnnData | ad.AnnData


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _decode_strings(values: Iterable) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _decode_scalar(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        if value.size == 1:
            return _decode_scalar(value.reshape(-1)[0])
    return value


def _resolve_path(data: PlotInput) -> Path:
    return resolve_data_path(data)


def _read_uns_scalar(path: Path, key: str):
    try:
        with h5py.File(path, "r") as handle:
            if "uns" not in handle or key not in handle["uns"]:
                return None
            dataset = handle["uns"][key]
            if not isinstance(dataset, h5py.Dataset):
                return None
            return _decode_scalar(dataset[()])
    except Exception:
        return None


def _read_group_names(path: Path, key: str) -> list[str]:
    with h5py.File(path, "r") as handle:
        rgg_path = f"uns/{key}"
        if rgg_path in handle:
            rgg = handle[rgg_path]
            if "names" in rgg:
                names_ds = rgg["names"]
                if names_ds.dtype.names is not None:
                    return [str(name) for name in names_ds.dtype.names]
                if names_ds.shape:
                    return _decode_strings(names_ds[()])
    backed = read_backed(path)
    try:
        return backed.obs_names.astype(str).tolist()
    finally:
        backed.file.close()


def _read_gene_names(path: Path) -> np.ndarray:
    backed = read_backed(path)
    try:
        if "genes" in backed.uns:
            genes = np.asarray(backed.uns["genes"]).astype(str)
        else:
            genes = backed.var_names.astype(str).to_numpy()
    finally:
        backed.file.close()
    return genes


def _read_var_column(path: Path, column: str) -> pd.Series | None:
    backed = read_backed(path)
    try:
        if column not in backed.var.columns:
            return None
        return backed.var[column].copy()
    finally:
        backed.file.close()


def _infer_rgg_params(path: Path, key: str) -> dict:
    params: dict[str, object] = {}
    with h5py.File(path, "r") as handle:
        rgg_path = f"uns/{key}"
        if rgg_path in handle:
            rgg = handle[rgg_path]
            if "params" in rgg:
                attrs = rgg["params"].attrs
                for attr in ("groupby", "method", "reference", "tie_correct", "corr_method", "use_raw"):
                    if attr in attrs:
                        params[attr] = _decode_scalar(attrs[attr])
    params.setdefault("groupby", _read_uns_scalar(path, "perturbation_column") or "group")
    params.setdefault("reference", _read_uns_scalar(path, "control_label") or "reference")
    params.setdefault("corr_method", _read_uns_scalar(path, "pvalue_correction") or "benjamini-hochberg")
    params.setdefault("method", _read_uns_scalar(path, "method") or "unknown")
    params.setdefault("tie_correct", False)
    params.setdefault("use_raw", False)
    return params


def _to_recarray(arrays: list[np.ndarray], names: Sequence[str]) -> np.recarray:
    return np.rec.fromarrays(arrays, names=[str(name) for name in names])


def _build_rgg_from_full(
    rgg: h5py.Group,
    groups: list[str],
    group_indices: list[int],
    genes: np.ndarray,
    n_genes: int | None,
) -> dict:
    full = rgg["full"]
    if "scores" not in full:
        raise KeyError("rank_genes_groups/full is missing required 'scores' dataset")

    order_ds = rgg.get("order")
    metrics = [
        key
        for key in (
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
            "pts",
            "pts_rest",
            "auc",
            "u_stat",
        )
        if key in full
    ]

    arrays_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}
    name_arrays: list[np.ndarray] = []

    for idx, _group in zip(group_indices, groups):
        if order_ds is not None:
            order = order_ds[idx]
        else:
            order = np.arange(len(genes), dtype=int)
        if n_genes is not None:
            order = order[:n_genes]
        name_arrays.append(genes[order].astype(str))
        for metric in metrics:
            row = full[metric][idx]
            arrays_by_metric[metric].append(np.take(row, order))

    rgg_dict = {
        "names": _to_recarray(name_arrays, groups),
    }
    for metric, arrays in arrays_by_metric.items():
        rgg_dict[metric] = _to_recarray(arrays, groups)
    return rgg_dict


def _build_rgg_from_recarray(
    rgg: h5py.Group,
    groups: list[str],
    n_genes: int | None,
) -> dict:
    names_ds = rgg["names"]
    names_arr = names_ds[()]
    available = list(names_arr.dtype.names or [])
    if not available:
        raise KeyError("rank_genes_groups names dataset is not structured")

    for group in groups:
        if group not in available:
            raise KeyError(f"Group '{group}' not found in rank_genes_groups names")

    limit = n_genes or names_arr.shape[0]

    name_arrays = []
    for group in groups:
        raw = names_arr[group][:limit]
        name_arrays.append(np.asarray(_decode_strings(raw), dtype=object))

    rgg_dict = {
        "names": _to_recarray(name_arrays, groups),
    }

    for key in (
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pts",
        "pts_rest",
        "auc",
        "u_stat",
    ):
        if key not in rgg:
            continue
        metric_arr = rgg[key][()]
        rgg_dict[key] = _to_recarray([metric_arr[group][:limit] for group in groups], groups)

    return rgg_dict


def _materialize_rank_genes_groups_uns(
    path: Path,
    *,
    key: str,
    groups: list[str],
    n_genes: int | None,
) -> dict:
    genes = _read_gene_names(path)
    params = _infer_rgg_params(path, key)
    all_groups = _read_group_names(path, key)
    group_indices = [all_groups.index(group) for group in groups]

    with h5py.File(path, "r") as handle:
        rgg_path = f"uns/{key}"
        if rgg_path not in handle:
            return _materialize_rank_genes_groups_from_layers(
                path,
                groups=groups,
                genes=genes,
                n_genes=n_genes,
                params=params,
            )
        rgg = handle[rgg_path]
        if "full" in rgg:
            rgg_dict = _build_rgg_from_full(rgg, groups, group_indices, genes, n_genes)
        else:
            rgg_dict = _build_rgg_from_recarray(rgg, groups, n_genes)

    rgg_dict["params"] = params
    return rgg_dict


def _materialize_rank_genes_groups_from_layers(
    path: Path,
    *,
    groups: list[str],
    genes: np.ndarray,
    n_genes: int | None,
    params: dict,
) -> dict:
    backed = read_backed(path)
    try:
        obs_names = backed.obs_names.astype(str).tolist()
        group_indices = [obs_names.index(group) for group in groups]

        def pick_layer(options: Sequence[str]) -> str | None:
            for name in options:
                if name in backed.layers:
                    return name
            return None

        score_layer = pick_layer(["z_score", "u_statistic", "u_stat", "scores"])
        if score_layer is None:
            raise KeyError("No score layer found for rank_genes_groups materialization")

        layer_map = {
            "scores": score_layer,
            "logfoldchanges": pick_layer(["logfoldchange", "logfoldchanges"]),
            "pvals": pick_layer(["pvalue", "pvals"]),
            "pvals_adj": pick_layer(["pvalue_adj", "pvals_adj"]),
            "pts": pick_layer(["pts"]),
            "pts_rest": pick_layer(["pts_rest"]),
        }

        arrays_by_metric: dict[str, list[np.ndarray]] = {key: [] for key in layer_map if layer_map[key]}
        name_arrays: list[np.ndarray] = []

        for idx in group_indices:
            scores = np.asarray(backed.layers[score_layer][idx]).ravel()
            order = np.argsort(-np.abs(scores), kind="mergesort")
            if n_genes is not None:
                order = order[:n_genes]
            name_arrays.append(genes[order].astype(str))
            for metric, layer_name in layer_map.items():
                if layer_name is None:
                    continue
                values = np.asarray(backed.layers[layer_name][idx]).ravel()
                arrays_by_metric[metric].append(np.take(values, order))
    finally:
        backed.file.close()

    rgg_dict = {"names": _to_recarray(name_arrays, groups)}
    for metric, arrays in arrays_by_metric.items():
        rgg_dict[metric] = _to_recarray(arrays, groups)
    rgg_dict["params"] = params
    return rgg_dict


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def materialize_rank_genes_groups(
    data: PlotInput,
    *,
    key: str = "rank_genes_groups",
    groups: Sequence[str] | None = None,
    n_genes: int | None = None,
    gene_symbols: str | None = None,
) -> ad.AnnData:
    """Create a minimal in-memory AnnData with Scanpy-style rank_genes_groups.

    This helper is intended for plotting only. It never loads the expression
    matrix from disk; instead it constructs an empty sparse matrix and injects
    Scanpy-compatible ``uns['rank_genes_groups']``.
    """
    path = _resolve_path(data)
    all_groups = _read_group_names(path, key)
    if groups is None:
        selected_groups = all_groups
    else:
        selected_groups = [str(group) for group in groups]
        missing = [group for group in selected_groups if group not in all_groups]
        if missing:
            raise KeyError(f"Groups not found in rank_genes_groups: {missing}")

    genes = _read_gene_names(path)
    rgg = _materialize_rank_genes_groups_uns(
        path,
        key=key,
        groups=selected_groups,
        n_genes=n_genes,
    )

    obs = pd.DataFrame(index=pd.Index(selected_groups, name="group"))
    groupby = rgg.get("params", {}).get("groupby")
    if groupby:
        obs[str(groupby)] = selected_groups

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    if gene_symbols is not None:
        var_col = _read_var_column(path, gene_symbols)
        if var_col is not None:
            var[gene_symbols] = var_col.to_numpy()
        else:
            logger.warning("gene_symbols column '%s' not found in var", gene_symbols)

    X = sp.csr_matrix((len(selected_groups), len(genes)), dtype=np.float32)
    adata = ad.AnnData(X, obs=obs, var=var)
    adata.uns[key] = rgg
    return adata


def rank_genes_groups(
    data: PlotInput,
    groups: Sequence[str] | None = None,
    *,
    n_genes: int = 20,
    gene_symbols: str | None = None,
    key: str = "rank_genes_groups",
    **kwargs,
):
    """Scanpy-style rank_genes_groups plot from on-disk crispyx results."""
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("scanpy is required for rank_genes_groups plotting") from exc

    adata = materialize_rank_genes_groups(
        data,
        key=key,
        groups=groups,
        n_genes=n_genes,
        gene_symbols=gene_symbols,
    )
    return sc.pl.rank_genes_groups(
        adata,
        groups=groups,
        n_genes=n_genes,
        gene_symbols=gene_symbols,
        key=key,
        **kwargs,
    )


def rank_genes_groups_df(
    data: PlotInput,
    group: str | Sequence[str],
    *,
    key: str = "rank_genes_groups",
    n_genes: int | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame for rank_genes_groups results from disk."""
    path = _resolve_path(data)
    genes = _read_gene_names(path)

    groups = [str(group)] if isinstance(group, str) else [str(g) for g in group]
    group_names = _read_group_names(path, key)

    dfs: list[pd.DataFrame] = []
    with h5py.File(path, "r") as handle:
        rgg_path = f"uns/{key}"
        if rgg_path not in handle:
            rgg = _materialize_rank_genes_groups_uns(
                path,
                key=key,
                groups=groups,
                n_genes=n_genes,
            )
            for group_name in groups:
                df = pd.DataFrame({"names": rgg["names"][group_name]})
                for key_name in (
                    "scores",
                    "logfoldchanges",
                    "pvals",
                    "pvals_adj",
                    "pts",
                    "pts_rest",
                    "auc",
                    "u_stat",
                ):
                    if key_name in rgg:
                        df[key_name] = rgg[key_name][group_name]
                df["group"] = group_name
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        rgg = handle[rgg_path]
        if "full" in rgg:
            full = rgg["full"]
            order_ds = rgg.get("order")
            for group_name in groups:
                if group_name not in group_names:
                    raise KeyError(f"Group '{group_name}' not found in rank_genes_groups")
                idx = group_names.index(group_name)
                if order_ds is not None:
                    order = order_ds[idx]
                else:
                    order = np.arange(len(genes), dtype=int)
                if n_genes is not None:
                    order = order[:n_genes]

                df = pd.DataFrame({"names": genes[order].astype(str)})
                for key_name in (
                    "scores",
                    "logfoldchanges",
                    "pvals",
                    "pvals_adj",
                    "pts",
                    "pts_rest",
                    "auc",
                    "u_stat",
                ):
                    if key_name in full:
                        row = np.take(full[key_name][idx], order)
                        df[key_name] = row
                df["group"] = group_name
                dfs.append(df)
        else:
            names_ds = rgg["names"]
            names_arr = names_ds[()]
            available = list(names_arr.dtype.names or [])
            if not available:
                raise KeyError("rank_genes_groups names dataset is not structured")
            limit = n_genes or names_arr.shape[0]
            for group_name in groups:
                if group_name not in available:
                    raise KeyError(f"Group '{group_name}' not found in rank_genes_groups")
                names = pd.Series(names_arr[group_name][:limit]).astype(str).to_numpy()
                df = pd.DataFrame({"names": names})
                for key_name in (
                    "scores",
                    "logfoldchanges",
                    "pvals",
                    "pvals_adj",
                    "pts",
                    "pts_rest",
                    "auc",
                    "u_stat",
                ):
                    if key_name in rgg:
                        metric_arr = rgg[key_name][()]
                        df[key_name] = metric_arr[group_name][:limit]
                df["group"] = group_name
                dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    return result


# -----------------------------------------------------------------------------
# Differential expression plots
# -----------------------------------------------------------------------------


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; cannot create plot")
        return None
    return plt


def plot_volcano(
    *,
    data: PlotInput | None = None,
    group: str | None = None,
    de_df: pd.DataFrame | None = None,
    key: str = "rank_genes_groups",
    p_cut: float = 0.05,
    lfc_cut: float = 1.0,
    ax=None,
    show: bool | None = None,
    savepath: str | Path | None = None,
):
    """Volcano plot for a single group.

    Provide ``de_df`` directly or supply ``data`` and ``group`` to read from disk.
    """
    plt = _require_matplotlib()
    if plt is None:
        return None

    if de_df is None:
        if data is None or group is None:
            raise ValueError("Provide either de_df or both data and group")
        de_df = rank_genes_groups_df(data, group=group, key=key)
    if group is None:
        if "group" in de_df.columns and not de_df["group"].empty:
            group = str(de_df["group"].iloc[0])
        else:
            group = ""

    if "pvals_adj" in de_df.columns:
        pvals = de_df["pvals_adj"].to_numpy()
    elif "pvals" in de_df.columns:
        pvals = de_df["pvals"].to_numpy()
    else:
        raise KeyError("p-values not found in de_df (expected pvals or pvals_adj)")

    if "logfoldchanges" not in de_df.columns:
        raise KeyError("logfoldchanges not found in de_df")

    df = de_df.copy()
    df["neglog10_p"] = -np.log10(np.clip(pvals, 1e-300, None))
    lfc = df["logfoldchanges"].to_numpy()
    sig = (pvals < p_cut) & (np.abs(lfc) >= lfc_cut)
    up = sig & (lfc > 0)
    down = sig & (lfc < 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(df.loc[~sig, "logfoldchanges"], df.loc[~sig, "neglog10_p"], s=6, alpha=0.4)
    ax.scatter(df.loc[down, "logfoldchanges"], df.loc[down, "neglog10_p"], s=8, alpha=0.9)
    ax.scatter(df.loc[up, "logfoldchanges"], df.loc[up, "neglog10_p"], s=8, alpha=0.9)

    ax.axhline(-np.log10(p_cut), linestyle="--", linewidth=1)
    ax.axvline(-lfc_cut, linestyle="--", linewidth=1)
    ax.axvline(lfc_cut, linestyle="--", linewidth=1)

    ax.set_title(f"Volcano: {group}")
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10(adj p)")

    if savepath:
        ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return ax


def plot_top_genes_bar(
    *,
    data: PlotInput | None = None,
    group: str | None = None,
    de_df: pd.DataFrame | None = None,
    key: str = "rank_genes_groups",
    topn: int = 15,
    ax=None,
    show: bool | None = None,
    savepath: str | Path | None = None,
):
    """Horizontal bar plot of top-ranked genes for a group."""
    plt = _require_matplotlib()
    if plt is None:
        return None

    if de_df is None:
        if data is None or group is None:
            raise ValueError("Provide either de_df or both data and group")
        de_df = rank_genes_groups_df(data, group=group, key=key)
    if group is None:
        if "group" in de_df.columns and not de_df["group"].empty:
            group = str(de_df["group"].iloc[0])
        else:
            group = ""

    if "scores" not in de_df.columns:
        raise KeyError("scores not found in de_df")

    df = de_df.sort_values("scores", ascending=False).head(topn).iloc[::-1]
    if "logfoldchanges" in df.columns:
        colors = np.where(df["logfoldchanges"].to_numpy() > 0, "tab:red", "tab:blue")
    else:
        colors = "tab:gray"

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.barh(df["names"], df["scores"], color=colors)
    ax.set_title(f"Top {topn} genes: {group}")
    ax.set_xlabel("score")

    if savepath:
        ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return ax


def _compute_library_sizes(adata: ad.AnnData, chunk_size: int) -> np.ndarray:
    n_obs = adata.n_obs
    library_size = np.zeros(n_obs, dtype=np.float64)
    for slc, block in iter_matrix_chunks(adata, axis=0, chunk_size=chunk_size, convert_to_dense=False):
        if sp.issparse(block):
            sums = np.asarray(block.sum(axis=1)).ravel()
        else:
            sums = np.asarray(block).sum(axis=1)
        library_size[slc] = sums
    return library_size


def _mean_expression_by_group(
    adata: ad.AnnData,
    gene_indices: np.ndarray,
    group_mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    chunk_size: int,
    mean_mode: str,
    target_sum: float,
) -> tuple[np.ndarray, np.ndarray]:
    subset = adata[:, gene_indices]
    n_genes = len(gene_indices)
    group_sum = np.zeros(n_genes, dtype=np.float64)
    ref_sum = np.zeros(n_genes, dtype=np.float64)

    if mean_mode == "log1p":
        library_size = _compute_library_sizes(adata, chunk_size=chunk_size)
    else:
        library_size = None

    for slc, block in iter_matrix_chunks(subset, axis=0, chunk_size=chunk_size, convert_to_dense=False):
        if mean_mode == "log1p":
            block, _ = normalize_total_block(
                block,
                library_size=library_size[slc],
                target_sum=target_sum,
            )
            block = np.log1p(block)
        else:
            if sp.issparse(block):
                block = block.toarray()
            else:
                block = np.asarray(block)

        if group_mask[slc].any():
            group_sum += block[group_mask[slc]].sum(axis=0)
        if ref_mask[slc].any():
            ref_sum += block[ref_mask[slc]].sum(axis=0)

    n_group = int(group_mask.sum())
    n_ref = int(ref_mask.sum())
    if n_group == 0 or n_ref == 0:
        raise ValueError("Group or reference has zero cells")

    return group_sum / n_group, ref_sum / n_ref


def plot_ma(
    *,
    data: PlotInput,
    de_result: PlotInput | None = None,
    group: str,
    reference: str | None = None,
    perturbation_column: str | None = None,
    key: str = "rank_genes_groups",
    de_df: pd.DataFrame | None = None,
    mean_mode: str = "raw",
    target_sum: float = 1e4,
    n_genes: int | None = None,
    p_cut: float = 0.05,
    lfc_cut: float = 1.0,
    chunk_size: int = 1024,
    ax=None,
    show: bool | None = None,
    savepath: str | Path | None = None,
):
    """MA plot using raw counts or normalized log1p means.

    Parameters
    ----------
    data
        Path or backed AnnData containing raw counts.
    de_result
        Path or AnnData with rank_genes_groups results. Defaults to ``data``.
    mean_mode
        "raw" or "log1p" (normalized log1p means).
    """
    plt = _require_matplotlib()
    if plt is None:
        return None

    if mean_mode not in {"raw", "log1p"}:
        raise ValueError("mean_mode must be 'raw' or 'log1p'")

    if de_result is None:
        de_result = data

    if de_df is None:
        de_df = rank_genes_groups_df(de_result, group=group, key=key, n_genes=n_genes)

    if "logfoldchanges" not in de_df.columns:
        raise KeyError("logfoldchanges not found in de_df")

    path = _resolve_path(data)
    backed = read_backed(path)
    try:
        var_names = backed.var_names.astype(str)
        if perturbation_column is None:
            params = _infer_rgg_params(_resolve_path(de_result), key)
            perturbation_column = str(params.get("groupby", "group"))
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' not found in adata.obs"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        if reference is None:
            params = _infer_rgg_params(_resolve_path(de_result), key)
            reference = str(params.get("reference", "reference"))
    finally:
        backed.file.close()

    genes = de_df["names"].astype(str).to_numpy()
    gene_indexer = pd.Index(var_names).get_indexer(genes)
    valid_mask = gene_indexer >= 0

    if not np.any(valid_mask):
        raise ValueError("None of the DE genes were found in the data var_names")

    genes = genes[valid_mask]
    gene_indexer = gene_indexer[valid_mask]
    lfc = de_df.iloc[valid_mask, :].loc[:, "logfoldchanges"].to_numpy()
    pvals_adj = (
        de_df.iloc[valid_mask, :].loc[:, "pvals_adj"].to_numpy()
        if "pvals_adj" in de_df.columns
        else None
    )

    backed = read_backed(path)
    try:
        group_mask = labels == group
        ref_mask = labels == reference

        mean_group, mean_ref = _mean_expression_by_group(
            backed,
            gene_indexer,
            group_mask,
            ref_mask,
            chunk_size=chunk_size,
            mean_mode=mean_mode,
            target_sum=target_sum,
        )
    finally:
        backed.file.close()

    if mean_mode == "raw":
        A = np.log1p((mean_group + mean_ref) / 2.0)
    else:
        A = (mean_group + mean_ref) / 2.0

    sig = None
    if pvals_adj is not None:
        sig = (pvals_adj < p_cut) & (np.abs(lfc) >= lfc_cut)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if sig is None:
        ax.scatter(A, lfc, s=6, alpha=0.6)
    else:
        ax.scatter(A[~sig], lfc[~sig], s=6, alpha=0.4)
        ax.scatter(A[sig], lfc[sig], s=8, alpha=0.9)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axhline(lfc_cut, linestyle="--", linewidth=1)
    ax.axhline(-lfc_cut, linestyle="--", linewidth=1)

    ax.set_title(f"MA plot: {group} vs {reference}")
    ax.set_xlabel("log1p(mean expression)" if mean_mode == "raw" else "mean log1p expression")
    ax.set_ylabel("log2FC")

    if savepath:
        ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return ax


# -----------------------------------------------------------------------------
# QC plots
# -----------------------------------------------------------------------------


def plot_qc_perturbation_counts(
    *,
    data: PlotInput,
    perturbation_column: str,
    cell_mask: np.ndarray | None = None,
    top_n: int | None = None,
    ax=None,
    show: bool | None = None,
    savepath: str | Path | None = None,
):
    """Plot per-perturbation cell counts (optionally after QC filtering)."""
    plt = _require_matplotlib()
    if plt is None:
        return None

    path = _resolve_path(data)
    backed = read_backed(path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' not found in adata.obs"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()

    if cell_mask is not None:
        labels = labels[cell_mask]

    counts = pd.Series(labels).value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.bar(counts.index.astype(str), counts.to_numpy())
    ax.set_ylabel("Cells")
    ax.set_xlabel("Perturbation")
    ax.set_title("Perturbation composition")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    if savepath:
        ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return ax


def plot_qc_summary(
    qc_result: QualityControlResult,
    *,
    bins: int = 50,
    min_genes: int | None = None,
    min_cells_per_gene: int | None = None,
    ax=None,
    show: bool | None = None,
    savepath: str | Path | None = None,
):
    """Plot QC summary distributions from a QualityControlResult."""
    plt = _require_matplotlib()
    if plt is None:
        return None

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        if isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 2:
            axes = ax
            fig = ax[0].figure
        else:
            raise ValueError("ax must be a sequence of two matplotlib axes")

    axes[0].hist(qc_result.cell_gene_counts, bins=bins, color="tab:blue", alpha=0.7)
    axes[0].set_title("Genes per cell")
    axes[0].set_xlabel("genes")
    axes[0].set_ylabel("cells")
    if min_genes is not None:
        axes[0].axvline(min_genes, linestyle="--", color="black")

    axes[1].hist(qc_result.gene_cell_counts, bins=bins, color="tab:green", alpha=0.7)
    axes[1].set_title("Cells per gene")
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("genes")
    if min_cells_per_gene is not None:
        axes[1].axvline(min_cells_per_gene, linestyle="--", color="black")

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return axes


# -----------------------------------------------------------------------------
# PCA Plotting Functions
# -----------------------------------------------------------------------------


def plot_pca(
    data: PlotInput,
    *,
    color: str | Sequence[str] | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    sort_order: bool = True,
    groups: str | Sequence[str] | None = None,
    projection: str = "2d",
    components: str | Sequence[str] | None = None,
    palette=None,
    na_color: str = "lightgray",
    na_in_legend: bool = True,
    size: float | None = None,
    frameon: bool | None = None,
    legend_fontsize: int | float | str | None = None,
    legend_fontweight: int | str | None = None,
    legend_loc: str = "right margin",
    legend_fontoutline: int | None = None,
    colorbar_loc: str | None = "right",
    ncols: int = 4,
    wspace: float | None = None,
    hspace: float = 0.25,
    title: str | Sequence[str] | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    ax=None,
    return_fig: bool | None = None,
    **kwargs,
):
    """Plot PCA scatter from on-disk crispyx/backed AnnData or in-memory AnnData.
    
    Wrapper around scanpy.pl.pca that works with backed and in-memory AnnData.
    Loads only the PCA embeddings and specified color columns.
    
    Parameters
    ----------
    data
        Path to h5ad, crispyx.AnnData, or anndata.AnnData with X_pca computed.
    color
        Keys for annotations of observations in .obs or variables in .var.
    components
        e.g. '1,2' or ['1,2', '3,4']. Default first 2 components.
    projection
        '2d' or '3d'.
    palette
        Color palette for categorical annotations.
    size
        Point size.
    show
        Show the figure.
    save
        Save the figure.
    **kwargs
        Passed to scanpy.pl.pca.
    
    Returns
    -------
    matplotlib.axes.Axes or list of Axes, or None if show=True.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required for PCA plotting") from exc

    # Handle both backed and in-memory AnnData
    if isinstance(data, ad.AnnData) and not hasattr(data, 'path'):
        # In-memory AnnData: use directly
        adata = data
    else:
        # Backed data: resolve path and load
        path = _resolve_path(data)
        adata = read_backed(path)
    
    # Check X_pca exists
    if "X_pca" not in adata.obsm:
        raise ValueError(
            "X_pca not found in adata.obsm. Run cx.pp.pca() first."
        )
    
    # Load into memory for plotting (just embeddings + obs)
    adata_plot = ad.AnnData(
        X=sp.csr_matrix((adata.n_obs, adata.n_vars), dtype=np.float32),
        obs=adata.obs.copy() if hasattr(adata.obs, 'copy') else pd.DataFrame(adata.obs),
    )
    adata_plot.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])
    
    # Copy uns['pca'] if present
    if "pca" in adata.uns:
        adata_plot.uns["pca"] = dict(adata.uns["pca"])
    
    return sc.pl.pca(
        adata_plot,
        color=color,
        use_raw=use_raw,
        layer=layer,
        sort_order=sort_order,
        groups=groups,
        projection=projection,
        components=components,
        palette=palette,
        na_color=na_color,
        na_in_legend=na_in_legend,
        size=size,
        frameon=frameon,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        legend_fontoutline=legend_fontoutline,
        colorbar_loc=colorbar_loc,
        ncols=ncols,
        wspace=wspace,
        hspace=hspace,
        title=title,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


def plot_pca_variance_ratio(
    data: PlotInput,
    *,
    n_pcs: int | None = None,
    log: bool = False,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Plot variance ratio explained by each PC.
    
    Wrapper around scanpy.pl.pca_variance_ratio that works with backed AnnData.
    
    Parameters
    ----------
    data
        Path to h5ad, crispyx.AnnData, or anndata.AnnData with PCA computed.
    n_pcs
        Number of PCs to show. Default shows all computed.
    log
        Plot on log scale.
    show
        Show the figure.
    save
        Save the figure.
    
    Returns
    -------
    matplotlib.axes.Axes or None if show=True.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required for PCA plotting") from exc

    # Handle both backed and in-memory AnnData
    if isinstance(data, ad.AnnData) and not hasattr(data, 'path'):
        adata = data
    else:
        path = _resolve_path(data)
        adata = read_backed(path)
    
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        raise ValueError(
            "PCA variance info not found. Run cx.pp.pca() first."
        )
    
    # Create minimal AnnData with just PCA uns
    adata_plot = ad.AnnData(
        X=sp.csr_matrix((1, 1), dtype=np.float32),
    )
    adata_plot.uns["pca"] = dict(adata.uns["pca"])
    
    # Only pass n_pcs if specified, otherwise let scanpy use its default
    kwargs = {"log": log, "show": show, "save": save}
    if n_pcs is not None:
        kwargs["n_pcs"] = n_pcs
    
    return sc.pl.pca_variance_ratio(adata_plot, **kwargs)


def plot_pca_loadings(
    data: PlotInput,
    *,
    components: int | str | Sequence[int] | None = None,
    include_lowest: bool = True,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Plot gene loadings for principal components.
    
    Wrapper around scanpy.pl.pca_loadings that works with backed and in-memory AnnData.
    
    Parameters
    ----------
    data
        Path to h5ad, crispyx.AnnData, or anndata.AnnData with PCA computed.
    components
        Which PCs to plot loadings for. e.g. [1, 2, 3] or '1,2,3'.
        Default shows first few components.
    include_lowest
        Show genes with lowest loadings (most negative) as well.
    show
        Show the figure.
    save
        Save the figure.
    
    Returns
    -------
    matplotlib.axes.Axes or None if show=True.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required for PCA plotting") from exc

    # Handle both backed and in-memory AnnData
    if isinstance(data, ad.AnnData) and not hasattr(data, 'path'):
        adata = data
    else:
        path = _resolve_path(data)
        adata = read_backed(path)
    
    if "PCs" not in adata.varm:
        raise ValueError(
            "PCA loadings (varm['PCs']) not found. Run cx.pp.pca() first."
        )
    
    # Load var and PCs for plotting
    adata_plot = ad.AnnData(
        X=sp.csr_matrix((1, adata.n_vars), dtype=np.float32),
        var=adata.var.copy() if hasattr(adata.var, 'copy') else pd.DataFrame(adata.var),
    )
    adata_plot.varm["PCs"] = np.asarray(adata.varm["PCs"])
    
    if "pca" in adata.uns:
        adata_plot.uns["pca"] = dict(adata.uns["pca"])
    
    return sc.pl.pca_loadings(
        adata_plot,
        components=components,
        include_lowest=include_lowest,
        show=show,
        save=save,
    )


# -----------------------------------------------------------------------------
# UMAP Plotting Functions
# -----------------------------------------------------------------------------


def plot_umap(
    data: PlotInput,
    *,
    color: str | Sequence[str] | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    sort_order: bool = True,
    groups: str | Sequence[str] | None = None,
    components: str | Sequence[int] | None = None,
    palette=None,
    na_color: str = "lightgray",
    na_in_legend: bool = True,
    size: float | None = None,
    frameon: bool | None = None,
    legend_fontsize: int | float | str | None = None,
    legend_fontweight: int | str | None = None,
    legend_loc: str = "right margin",
    legend_fontoutline: int | None = None,
    colorbar_loc: str | None = "right",
    ncols: int = 4,
    wspace: float | None = None,
    hspace: float = 0.25,
    title: str | Sequence[str] | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    ax=None,
    return_fig: bool | None = None,
    **kwargs,
):
    """Plot UMAP embedding from on-disk crispyx/backed AnnData or in-memory AnnData.
    
    Wrapper around scanpy.pl.umap that works with backed and in-memory AnnData.
    Loads only the UMAP embeddings and specified color columns.
    
    Parameters
    ----------
    data
        Path to h5ad, crispyx.AnnData, or anndata.AnnData with X_umap computed.
    color
        Keys for annotations of observations in .obs or variables in .var.
    components
        Which dimensions to use (e.g. [0, 1] for first two). Default first 2.
    palette
        Color palette for categorical annotations.
    size
        Point size.
    show
        Show the figure.
    save
        Save the figure.
    **kwargs
        Passed to scanpy.pl.umap.
    
    Returns
    -------
    matplotlib.axes.Axes or list of Axes, or None if show=True.
    
    Examples
    --------
    >>> import crispyx as cx
    >>> adata = cx.read_backed("data.h5ad")
    >>> cx.pl.umap(adata, color="perturbation")
    
    See Also
    --------
    cx.tl.umap : Compute UMAP embedding.
    cx.pp.neighbors : Compute neighbor graph (required for UMAP).
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required for UMAP plotting") from exc

    # Handle both backed and in-memory AnnData
    if isinstance(data, ad.AnnData) and not hasattr(data, 'path'):
        # In-memory AnnData: use directly
        adata = data
    else:
        # Backed data: resolve path and load
        path = _resolve_path(data)
        adata = read_backed(path)
    
    # Check X_umap exists
    if "X_umap" not in adata.obsm:
        raise ValueError(
            "X_umap not found in adata.obsm. Run cx.tl.umap() first."
        )
    
    # Load into memory for plotting (just embeddings + obs)
    adata_plot = ad.AnnData(
        X=sp.csr_matrix((adata.n_obs, adata.n_vars), dtype=np.float32),
        obs=adata.obs.copy() if hasattr(adata.obs, 'copy') else pd.DataFrame(adata.obs),
    )
    adata_plot.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"])
    
    # Copy uns['umap'] if present
    if "umap" in adata.uns:
        adata_plot.uns["umap"] = dict(adata.uns["umap"])
    
    return sc.pl.umap(
        adata_plot,
        color=color,
        use_raw=use_raw,
        layer=layer,
        sort_order=sort_order,
        groups=groups,
        components=components,
        palette=palette,
        na_color=na_color,
        na_in_legend=na_in_legend,
        size=size,
        frameon=frameon,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        legend_fontoutline=legend_fontoutline,
        colorbar_loc=colorbar_loc,
        ncols=ncols,
        wspace=wspace,
        hspace=hspace,
        title=title,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


# =============================================================================
# Overlap heatmap (Feature 5)
# =============================================================================

def plot_overlap_heatmap(
    result,
    *,
    metric="jaccard",
    ax=None,
    cmap="Blues",
    annot=True,
    fmt=None,
    title=None,
):
    """Heatmap of pairwise overlap statistics.

    Parameters
    ----------
    result
        :class:`~crispyx.data.OverlapResult` from
        :func:`~crispyx.data.compute_overlap`.
    metric
        ``"jaccard"`` (default) or ``"count"``.
    ax
        Existing :class:`matplotlib.axes.Axes`; a new figure is created if
        ``None``.
    cmap
        Matplotlib colormap name.
    annot
        Annotate each cell with the numeric value.
    fmt
        Number format. Defaults to ``".2f"`` for Jaccard and ``"d"`` for
        counts.
    title
        Figure title. Defaults to ``"Jaccard similarity"`` or
        ``"Overlap counts"``.

    Returns
    -------
    matplotlib.axes.Axes or None
    """
    plt = _require_matplotlib()
    if plt is None:
        return None

    data = result.jaccard_matrix if metric == "jaccard" else result.count_matrix
    default_fmt = ".2f" if metric == "jaccard" else "d"
    fmt = fmt or default_fmt
    title = title or (
        "Jaccard similarity" if metric == "jaccard" else "Overlap counts"
    )
    n = len(data)

    try:
        import seaborn as sns  # type: ignore[import]
        if ax is None:
            _, ax = plt.subplots(figsize=(max(4, n * 0.8), max(3, n * 0.7)))
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            linewidths=0.5,
            square=True,
            vmin=0,
            vmax=1.0 if metric == "jaccard" else None,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(title)
    except ImportError:
        if ax is None:
            _, ax = plt.subplots(figsize=(max(4, n * 0.8), max(3, n * 0.7)))
        arr = data.to_numpy(dtype=float)
        vmax = 1.0 if metric == "jaccard" else None
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_xticklabels(data.columns, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(data.index)
        if annot:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(
                        j, i, f"{arr[i, j]:{fmt}}",
                        ha="center", va="center", fontsize=8,
                    )
        ax.figure.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(title)

    return ax
