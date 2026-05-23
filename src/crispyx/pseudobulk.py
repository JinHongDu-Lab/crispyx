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
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    verbose: int | bool = False,
) -> AnnData:
    """Compute average log-normalised expression per perturbation relative to control."""

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
            path, suffix="avg_log_effects", output_dir=output_dir, data_name=data_name
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
    output_path = resolve_output_path(path, suffix="avg_log_effects", output_dir=output_dir, data_name=data_name)
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
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    verbose: int | bool = False,
) -> AnnData:
    """Compute pseudo-bulk log-fold changes relative to control."""

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
            path, suffix="pseudobulk_effects", output_dir=output_dir, data_name=data_name
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
    output_path = resolve_output_path(path, suffix="pseudobulk_effects", output_dir=output_dir, data_name=data_name)
    if int(verbose) >= 1:
        print(f"[cx] pb.compute_pseudobulk_expression: {len(candidates)} perturbations × {len(gene_symbols)} genes")
        print(f"[cx] pb.compute_pseudobulk_expression: Saving → {output_path}")
    adata.write(output_path)
    return AnnData(output_path)

