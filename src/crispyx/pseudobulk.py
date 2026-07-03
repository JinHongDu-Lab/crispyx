"""Pseudo-bulk effect size estimators operating directly on ``.h5ad`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd

from .data import (
    AnnData,
    calculate_optimal_chunk_size,
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
)


def _resolve_candidates(
    labels: np.ndarray,
    control_label: str,
    perturbations: Iterable[str] | None,
) -> list[str]:
    if perturbations is None:
        unique = pd.Index(labels).unique().tolist()
    else:
        unique = [str(p) for p in perturbations]
    return [label for label in unique if label != control_label]


def compute_average_log_expression(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    chunk_size: int | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    verbose: int | bool = False,
) -> AnnData:
    """Compute average log-normalised expression per perturbation relative to control.

    For each perturbation group, computes the per-gene mean of log1p-normalised
    expression and stores the difference relative to the control group as the
    effect size.

    Parameters
    ----------
    data
        Path to an h5ad file, or a backed/in-memory AnnData object.
    perturbation_column
        Column in ``adata.obs`` that identifies perturbation groups.
    control_label
        Label of the control group.  If ``None``, inferred from common
        patterns (``'non-targeting'``, ``'control'``, etc.).
    gene_name_column
        Column in ``adata.var`` with gene symbols.  If ``None``, uses
        ``adata.var_names``.
    perturbations
        Subset of perturbation labels to include.  If ``None``, all
        non-control groups are processed.
    chunk_size
        Number of cells to process per chunk.  If ``None``, auto-determined
        from dataset shape.
    data_name
        Custom stem for the output filename.  If ``None``, the input file
        stem is used with a ``_cx_avg_log_effects`` suffix.
    output_path
        Exact path for the output h5ad file.  When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for the output file.  Defaults to the input file's
        directory.  *Deprecated* – use ``output_path`` instead.  Will be
        removed in the next major version.
    verbose
        Verbosity level.  ``0`` / ``False`` is silent; ``1`` / ``True``
        prints a summary line.

    Returns
    -------
    AnnData
        On-disk AnnData where ``X`` contains the effect-size matrix
        (perturbation mean minus control mean in log-normalised space),
        ``layers['perturbation_mean']`` contains per-perturbation means,
        and ``uns['control_mean']`` contains the control mean vector.
    """

    path = resolve_data_path(data)
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_average_log_expression: Reading {path}")
    backed = read_backed(path)
    try:
        # Calculate adaptive chunk_size if not provided
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(backed.n_obs, backed.n_vars)
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        groups = [control_label] + candidates
        sums = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
        counts = {label: 0 for label in groups}
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            slice_labels = labels[slc]
            normalised_block, _ = normalize_total_block(block)
            log_block = np.log1p(normalised_block)
            for label in groups:
                mask = slice_labels == label
                if not np.any(mask):
                    continue
                sums[label] += log_block[mask].sum(axis=0)
                counts[label] += int(mask.sum())
    finally:
        backed.file.close()

    if counts[control_label] == 0:
        raise ValueError("Control group contains no cells")
    control_mean = sums[control_label] / counts[control_label]

    effect_matrix = []
    pert_means = []
    for label in candidates:
        if counts[label] == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        mean = sums[label] / counts[label]
        pert_means.append(mean)
        effect_matrix.append(mean - control_mean)

    if not effect_matrix:
        obs_index = pd.Index([], name="perturbation")
        adata = ad.AnnData(
            np.zeros((0, gene_symbols.shape[0])),
            obs=pd.DataFrame(index=obs_index),
            var=pd.DataFrame(index=gene_symbols),
        )
        output_path = resolve_output_path(
            path, suffix="avg_log_effects", output_dir=output_dir, data_name=data_name,
            output_path=output_path,
        )
        if int(verbose) >= 1:
            print(f"[cx] pb.compute_average_log_expression: 0 perturbations × {gene_symbols.shape[0]} genes")
            print(f"[cx] pb.compute_average_log_expression: Saving → {output_path}")
        adata.write(output_path)
        return AnnData(output_path)

    effect_matrix_np = np.vstack(effect_matrix)
    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix_np, obs=obs, var=var)
    adata.layers["perturbation_mean"] = np.vstack(pert_means)
    adata.uns["control_mean"] = control_mean
    output_path = resolve_output_path(
        path, suffix="avg_log_effects", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_average_log_expression: {len(candidates)} perturbations × {len(gene_symbols)} genes")
        print(f"[cx] pb.compute_average_log_expression: Saving → {output_path}")
    adata.write(output_path)
    return AnnData(output_path)


def compute_pseudobulk_expression(
    data: str | Path | AnnData | ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str | None = None,
    gene_name_column: str | None = None,
    perturbations: Iterable[str] | None = None,
    baseline_count: float = 1.0,
    chunk_size: int | None = None,
    data_name: str | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
    verbose: int | bool = False,
) -> AnnData:
    """Compute pseudo-bulk log-fold changes relative to control.

    Aggregates normalised counts per perturbation group into a pseudo-bulk
    profile (sum divided by cell count), applies log1p scaling with a
    ``baseline_count`` offset, and stores the difference relative to the
    control group as the log-fold change effect size.

    Parameters
    ----------
    data
        Path to an h5ad file, or a backed/in-memory AnnData object.
    perturbation_column
        Column in ``adata.obs`` that identifies perturbation groups.
    control_label
        Label of the control group.  If ``None``, inferred from common
        patterns (``'non-targeting'``, ``'control'``, etc.).
    gene_name_column
        Column in ``adata.var`` with gene symbols.  If ``None``, uses
        ``adata.var_names``.
    perturbations
        Subset of perturbation labels to include.  If ``None``, all
        non-control groups are processed.
    baseline_count
        Pseudo-count added before log transformation
        (``log1p(baseline_count * mean_counts)``).  Default ``1.0``.
    chunk_size
        Number of cells to process per chunk.  If ``None``, auto-determined
        from dataset shape.
    data_name
        Custom stem for the output filename.  If ``None``, the input file
        stem is used with a ``_cx_pseudobulk_effects`` suffix.
    output_path
        Exact path for the output h5ad file.  When provided, ``output_dir``
        and ``data_name`` are ignored.
    output_dir
        Directory for the output file.  Defaults to the input file's
        directory.  *Deprecated* – use ``output_path`` instead.  Will be
        removed in the next major version.
    verbose
        Verbosity level.  ``0`` / ``False`` is silent; ``1`` / ``True``
        prints a summary line.

    Returns
    -------
    AnnData
        On-disk AnnData where ``X`` contains the pseudo-bulk log-fold change
        matrix (perturbation pseudo-bulk minus control pseudo-bulk),
        ``layers['perturbation_bulk']`` contains per-perturbation pseudo-bulk
        vectors, ``uns['control_bulk']`` the control pseudo-bulk vector, and
        ``uns['baseline_count']`` the scaling offset used.
    """

    if baseline_count <= 0:
        raise ValueError("baseline_count must be positive")

    path = resolve_data_path(data)
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_pseudobulk_expression: Reading {path}")
    backed = read_backed(path)
    try:
        # Calculate adaptive chunk_size if not provided
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(backed.n_obs, backed.n_vars)
        gene_symbols = ensure_gene_symbol_column(backed, gene_name_column)
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        control_label = resolve_control_label(labels, control_label)
        n_genes = backed.n_vars
        candidates = _resolve_candidates(labels, control_label, perturbations)
        groups = [control_label] + candidates
        sums = {label: np.zeros(n_genes, dtype=np.float64) for label in groups}
        counts = {label: 0 for label in groups}
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size):
            slice_labels = labels[slc]
            normalised_block, _ = normalize_total_block(block)
            for label in groups:
                mask = slice_labels == label
                if not np.any(mask):
                    continue
                sums[label] += normalised_block[mask].sum(axis=0)
                counts[label] += int(mask.sum())
    finally:
        backed.file.close()

    if counts[control_label] == 0:
        raise ValueError("Control group contains no cells")
    control_bulk = np.log1p(baseline_count * sums[control_label] / counts[control_label])

    effect_matrix = []
    pert_bulks = []
    for label in candidates:
        if counts[label] == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        bulk = np.log1p(baseline_count * sums[label] / counts[label])
        pert_bulks.append(bulk)
        effect_matrix.append(bulk - control_bulk)

    if not effect_matrix:
        obs_index = pd.Index([], name="perturbation")
        adata = ad.AnnData(
            np.zeros((0, gene_symbols.shape[0])),
            obs=pd.DataFrame(index=obs_index),
            var=pd.DataFrame(index=gene_symbols),
        )
        adata.uns["control_bulk"] = control_bulk
        adata.uns["baseline_count"] = float(baseline_count)
        output_path = resolve_output_path(
            path, suffix="pseudobulk_effects", output_dir=output_dir, data_name=data_name,
            output_path=output_path,
        )
        if int(verbose) >= 1:
            print(f"[cx] pb.compute_pseudobulk_expression: 0 perturbations × {gene_symbols.shape[0]} genes")
            print(f"[cx] pb.compute_pseudobulk_expression: Saving → {output_path}")
        adata.write(output_path)
        return AnnData(output_path)

    effect_matrix_np = np.vstack(effect_matrix)
    gene_symbols = pd.Index(gene_symbols).astype(str)
    obs_index = pd.Index(candidates, name="perturbation").astype(str)
    obs = pd.DataFrame({perturbation_column: obs_index.to_list()}, index=obs_index)
    var = pd.DataFrame(index=gene_symbols)
    adata = ad.AnnData(effect_matrix_np, obs=obs, var=var)
    adata.layers["perturbation_bulk"] = np.vstack(pert_bulks)
    adata.uns["control_bulk"] = control_bulk
    adata.uns["baseline_count"] = float(baseline_count)
    output_path = resolve_output_path(
        path, suffix="pseudobulk_effects", output_dir=output_dir, data_name=data_name,
        output_path=output_path,
    )
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_pseudobulk_expression: {len(candidates)} perturbations × {len(gene_symbols)} genes")
        print(f"[cx] pb.compute_pseudobulk_expression: Saving → {output_path}")
    adata.write(output_path)
    return AnnData(output_path)

