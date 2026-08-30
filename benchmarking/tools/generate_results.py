"""Generate benchmark results reports from cached benchmark data.

This module handles the evaluation and reporting of benchmark results,
separated from the benchmark execution logic in run_benchmarks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

# Import from centralized modules
from .constants import (
    SHRINKAGE_METADATA,
    LFCSHRINK_METHODS,
    NB_GLM_METHODS,
    ALL_DE_METHODS_FOR_HEATMAP,
    METHODS_WITH_SHRUNK_OUTPUT,
    METHOD_DISPLAY_NAMES,
    DE_METHOD_DISPLAY_ORDER,
    CATEGORY_DISPLAY_ORDER,
    STANDARD_DE_COLUMNS,
)
from .cache import (
    load_method_result,
    load_cached_results,
    save_method_result,
    load_cache_config,
    get_expected_output_path,
    resolve_result_path,
    is_scalar_na,
    has_valid_result,
)
from .formatting import (
    is_scalar_notna,
    format_method_name,
    get_method_package,
    format_full_method_name,
    is_crispyx_method,
    get_shrinkage_type,
    get_method_sort_key,
    get_category_sort_key,
    get_method_category,
    get_performance_emoji,
    get_accuracy_emoji,
    format_mean_std,
    format_pct,
    format_diff,
    frame_to_markdown_table,
    standardise_de_dataframe,
)


# ============================================================================
# Shrinkage Quality Utilities
# ============================================================================

def _compute_shrinkage_quality_standalone(
    shrunk_result_path: Path,
    method_name: str,
) -> Optional[Dict[str, Any]]:
    """Compute shrinkage quality metrics from a standalone lfcShrink result file.
    
    This function extracts both raw and shrunk LFC values from the same h5ad file,
    using the 'logfoldchange_raw' layer for raw values and 'X' (or 'logfoldchange')
    for shrunk values. This allows computing shrinkage quality even when the base
    NB-GLM method failed or was not run.
    
    Parameters
    ----------
    shrunk_result_path : Path
        Path to lfcShrink result h5ad file (must contain 'logfoldchange_raw' layer)
    method_name : str
        Name of the method for labeling
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with shrinkage quality metrics, or None if computation fails
    """
    import anndata as ad
    
    try:
        if not str(shrunk_result_path).endswith('.h5ad'):
            return None
            
        adata = ad.read_h5ad(str(shrunk_result_path))
        
        # Check for required layers - lfcShrink results store raw LFC in logfoldchange_raw
        if 'logfoldchange_raw' not in adata.layers:
            return None
        
        all_ratios = []
        all_inflation_counts = []
        all_total_counts = []
        
        for pert_idx in range(adata.n_obs):
            # Get raw LFC from logfoldchange_raw layer
            raw_lfc = adata.layers['logfoldchange_raw'][pert_idx, :]
            if hasattr(raw_lfc, 'toarray'):
                raw_lfc = raw_lfc.toarray().flatten()
            else:
                raw_lfc = np.asarray(raw_lfc).flatten()
            
            # Get shrunk LFC from X (effect_size) or logfoldchange layer
            if 'logfoldchange' in adata.layers:
                shrunk_lfc = adata.layers['logfoldchange'][pert_idx, :]
            else:
                shrunk_lfc = adata.X[pert_idx, :]
            if hasattr(shrunk_lfc, 'toarray'):
                shrunk_lfc = shrunk_lfc.toarray().flatten()
            else:
                shrunk_lfc = np.asarray(shrunk_lfc).flatten()
            
            # Compute ratio for genes with |raw| > 0.1
            valid_mask = np.isfinite(raw_lfc) & np.isfinite(shrunk_lfc) & (np.abs(raw_lfc) > 0.1)
            if valid_mask.sum() == 0:
                continue
            
            ratio = np.abs(shrunk_lfc[valid_mask]) / np.abs(raw_lfc[valid_mask])
            all_ratios.extend(ratio.tolist())
            all_inflation_counts.append((ratio > 1.0).sum())
            all_total_counts.append(len(ratio))
        
        if not all_ratios:
            return None
        
        ratios = np.array(all_ratios)
        total_inflation = sum(all_inflation_counts)
        total_genes = sum(all_total_counts)
        
        return {
            "method": method_name,
            "pct_inflated": 100.0 * total_inflation / total_genes if total_genes > 0 else 0,
            "max_inflation": float(ratios.max()) if len(ratios) > 0 else 1.0,
            "median_ratio": float(np.median(ratios)) if len(ratios) > 0 else 1.0,
            "genes_analyzed": total_genes,
        }
        
    except Exception as e:
        return None


def _compute_shrinkage_quality(
    raw_result_path: Path,
    shrunk_result_path: Path,
    method_name: str,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Compute shrinkage quality metrics for a method.
    
    Compares raw vs shrunk LFC values to detect aberrant shrinkage where
    |shrunk| > |raw| (LFC magnitude increases instead of decreasing).
    
    Parameters
    ----------
    raw_result_path : Path
        Path to raw (unshrunk) DE results
    shrunk_result_path : Path
        Path to shrunk DE results
    method_name : str
        Name of the method for labeling
    output_dir : Path
        Output directory
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with shrinkage quality metrics, or None if computation fails
    """
    import anndata as ad
    
    try:
        # Load raw and shrunk results
        if str(raw_result_path).endswith('.h5ad'):
            raw_adata = ad.read_h5ad(str(raw_result_path))
            shrunk_adata = ad.read_h5ad(str(shrunk_result_path))
            
            # For h5ad files, extract LFC from layers or X
            all_ratios = []
            all_inflation_counts = []
            all_total_counts = []
            
            for pert_idx, pert in enumerate(raw_adata.obs['perturbation']):
                # Get raw LFC
                if 'logfoldchange' in raw_adata.layers:
                    raw_lfc = raw_adata.layers['logfoldchange'][pert_idx, :]
                    if hasattr(raw_lfc, 'toarray'):
                        raw_lfc = raw_lfc.toarray().flatten()
                else:
                    continue
                
                # Get shrunk LFC
                shrunk_lfc = shrunk_adata.X[pert_idx, :]
                if hasattr(shrunk_lfc, 'toarray'):
                    shrunk_lfc = shrunk_lfc.toarray().flatten()
                
                # Compute ratio for genes with |raw| > 0.1
                valid_mask = np.abs(raw_lfc) > 0.1
                if valid_mask.sum() == 0:
                    continue
                
                ratio = np.abs(shrunk_lfc[valid_mask]) / np.abs(raw_lfc[valid_mask])
                all_ratios.extend(ratio.tolist())
                all_inflation_counts.append((ratio > 1.0).sum())
                all_total_counts.append(len(ratio))
            
            if not all_ratios:
                return None
            
            ratios = np.array(all_ratios)
            total_inflation = sum(all_inflation_counts)
            total_genes = sum(all_total_counts)
            
            return {
                "method": method_name,
                "pct_inflated": 100.0 * total_inflation / total_genes if total_genes > 0 else 0,
                "max_inflation": float(ratios.max()) if len(ratios) > 0 else 1.0,
                "median_ratio": float(np.median(ratios)) if len(ratios) > 0 else 1.0,
                "genes_analyzed": total_genes,
            }
        
        elif str(raw_result_path).endswith('.csv') and str(shrunk_result_path).endswith('.csv'):
            # For CSV files (pertpy)
            raw_df = pd.read_csv(raw_result_path)
            shrunk_df = pd.read_csv(shrunk_result_path)
            
            # Standardize column names
            raw_df = raw_df.rename(columns={"log2FoldChange": "log_fc", "log_fc": "log_fc"})
            shrunk_df = shrunk_df.rename(columns={"log2FoldChange": "log_fc", "log_fc": "log_fc"})
            
            if 'log_fc' not in raw_df.columns or 'log_fc' not in shrunk_df.columns:
                return None
            if 'gene' not in raw_df.columns or 'gene' not in shrunk_df.columns:
                return None
            if 'perturbation' not in raw_df.columns or 'perturbation' not in shrunk_df.columns:
                return None
            
            # Merge on gene and perturbation
            merged = raw_df.merge(
                shrunk_df[['gene', 'perturbation', 'log_fc']],
                on=['gene', 'perturbation'],
                suffixes=('_raw', '_shrunk')
            )
            
            # Filter for |raw| > 0.1
            merged = merged[np.abs(merged['log_fc_raw']) > 0.1]
            if len(merged) == 0:
                return None
            
            # Compute ratio
            ratio = np.abs(merged['log_fc_shrunk']) / np.abs(merged['log_fc_raw'])
            
            return {
                "method": method_name,
                "pct_inflated": 100.0 * (ratio > 1.0).sum() / len(ratio),
                "max_inflation": float(ratio.max()),
                "median_ratio": float(ratio.median()),
                "genes_analyzed": len(ratio),
            }
        
        return None
        
    except Exception as e:
        return None


# ============================================================================
# DE Result Loading and Conversion Utilities
# ============================================================================

def _anndata_to_de_dict(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format.
    
    Properly extracts 1D dense arrays from sparse matrices in adata.X and adata.layers.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing DE results in layers (crispyx format)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping perturbation names to SimpleNamespace objects with
        genes, effect_size, statistic, and pvalue attributes
    """
    from types import SimpleNamespace
    
    def _extract_row(matrix, idx: int) -> np.ndarray | None:
        """Extract row from matrix (sparse or dense) as 1D dense array."""
        if matrix is None:
            return None
        if idx >= matrix.shape[0]:
            return None
        try:
            row = matrix[idx, :]
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            if "logfoldchange" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif "logfoldchanges" in adata.layers:
                # crispyx t-test/Wilcoxon store LFCs in "logfoldchanges" (plural)
                effect_size_values = _extract_row(adata.layers["logfoldchanges"], idx)
            elif adata.X is not None:
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    elif "rank_genes_groups" in adata.uns:
        rgg = adata.uns["rank_genes_groups"]
        if hasattr(rgg["names"], "dtype") and hasattr(rgg["names"].dtype, "names"):
            groups = list(rgg["names"].dtype.names)
        else:
            groups = []
        
        for group in groups:
            if "names" in rgg:
                genes = rgg["names"][group]
            else:
                genes = adata.uns.get("genes", adata.var_names.to_numpy())

            stream_result_dict[group] = SimpleNamespace(
                genes=genes,
                effect_size=rgg["logfoldchanges"][group] if "logfoldchanges" in rgg else None,
                statistic=rgg["scores"][group] if "scores" in rgg else None,
                pvalue=rgg["pvals"][group] if "pvals" in rgg else None,
            )
    
    return stream_result_dict


def _anndata_to_de_dict_raw(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format, using RAW (unshrunken) LFCs.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing DE results in layers (crispyx format)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping perturbation names to SimpleNamespace objects with
        genes, effect_size (raw), statistic, and pvalue attributes
    """
    from types import SimpleNamespace
    
    def _extract_row(matrix, idx: int) -> np.ndarray | None:
        if matrix is None:
            return None
        if idx >= matrix.shape[0]:
            return None
        try:
            row = matrix[idx, :]
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            if "logfoldchange_raw" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange_raw"], idx)
            elif "logfoldchange" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif "logfoldchanges" in adata.layers:
                # crispyx t-test/Wilcoxon store LFCs in "logfoldchanges" (plural)
                effect_size_values = _extract_row(adata.layers["logfoldchanges"], idx)
            elif adata.X is not None:
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    
    return stream_result_dict


def _make_gene_index_unique(gene_index: pd.Index) -> pd.Index:
    """Append positional suffix to duplicate gene names (e.g. MATR3, MATR3-1).

    Prevents many-to-many merges in ``compute_de_comparison_metrics`` when
    datasets contain duplicate gene symbols (e.g. Feng gwsf/gwsnf/ts).
    """
    if not gene_index.duplicated().any():
        return gene_index
    counts: dict[str, int] = {}
    unique: list[str] = []
    for g in gene_index:
        if g in counts:
            counts[g] += 1
            unique.append(f"{g}-{counts[g]}")
        else:
            counts[g] = 0
            unique.append(g)
    return pd.Index(unique)


def _streaming_de_to_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a streaming differential expression mapping to a tidy DataFrame."""
    frames = []
    _gene_cache: dict[int, pd.Index] = {}
    for perturbation, entry in result.items():
        genes = getattr(entry, "genes", None)
        if genes is None:
            continue
        genes_id = id(genes)
        if genes_id in _gene_cache:
            gene_index = _gene_cache[genes_id]
        else:
            gene_index = _make_gene_index_unique(pd.Index(genes).astype(str))
            _gene_cache[genes_id] = gene_index
        n_rows = len(gene_index)
        frame = pd.DataFrame(
            {
                "perturbation": [str(perturbation)] * n_rows,
                "gene": gene_index,
                "effect_size": getattr(entry, "effect_size", pd.NA),
                "statistic": getattr(entry, "statistic", pd.NA),
                "pvalue": getattr(entry, "pvalue", pd.NA),
            }
        )
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=STANDARD_DE_COLUMNS)


def standardise_de_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return ``df`` with standard differential expression column names."""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_DE_COLUMNS)

    result = df.copy()
    lower_to_original = {col.lower(): col for col in result.columns}

    def _resolve_column(candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                return original
        return None

    rename: Dict[str, str] = {}
    perturbation_col = _resolve_column(["perturbation", "group", "cluster", "label", "contrast"])
    gene_col = _resolve_column(["gene", "genes", "name", "names", "feature", "variable"])
    effect_col = _resolve_column(["effect_size", "logfoldchange", "logfoldchanges", "logfc", "lfc", "log_fc", "coefficient"])
    stat_col = _resolve_column(["statistic", "statistics", "stat", "score", "scores", "wald_statistic", "zscore", "t_stat", "t_value", "t_statistic", "f", "f_value", "u_stat"])
    pvalue_col = _resolve_column(["pvalue", "p_value", "pval", "pvals", "pvalue_raw", "pvalue_adj", "pvals_adj"])

    if perturbation_col is not None:
        rename[perturbation_col] = "perturbation"
    if gene_col is not None:
        rename[gene_col] = "gene"
    if effect_col is not None:
        rename[effect_col] = "effect_size"
    if stat_col is not None:
        rename[stat_col] = "statistic"
    if pvalue_col is not None:
        rename[pvalue_col] = "pvalue"

    result = result.rename(columns=rename)

    for column in STANDARD_DE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[STANDARD_DE_COLUMNS]
    result["perturbation"] = result["perturbation"].astype(str).str.strip()
    result["gene"] = result["gene"].astype(str).str.strip()
    # Deduplicate gene names within each perturbation to prevent many-to-many
    # merges when datasets contain duplicate gene symbols (e.g. Feng datasets).
    if result.duplicated(subset=["perturbation", "gene"]).any():
        cumcounts = result.groupby(["perturbation", "gene"], sort=False).cumcount()
        mask = cumcounts > 0
        if mask.any():
            result = result.copy()
            result.loc[mask, "gene"] = result.loc[mask, "gene"] + "-" + cumcounts[mask].astype(str)
    result["effect_size"] = pd.to_numeric(result["effect_size"], errors="coerce")
    result["statistic"] = pd.to_numeric(result["statistic"], errors="coerce")
    result["pvalue"] = pd.to_numeric(result["pvalue"], errors="coerce")
    return result


# ============================================================================
# Markdown Generation
# ============================================================================

def _generate_improved_markdown(
    perf_df: pd.DataFrame,
    perf_comp_results: List[Dict[str, Any]],
    accuracy_results: List[Dict[str, Any]],
    overlap_heatmaps: Optional[Dict[str, Path]] = None,
    shrinkage_quality: Optional[List[Dict[str, Any]]] = None,
    skipped_comparisons: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate improved markdown with categorized tables and emoji indicators."""
    
    md = "# Benchmark Results\n\n"
    
    # =========================================================================
    # Section 1: Performance by Category
    # =========================================================================
    md += "## 1. Performance\n\n"
    
    if not perf_df.empty:
        perf_df = perf_df.copy()
        perf_df["_category"] = perf_df["method"].apply(lambda x: get_method_category(x)[0])
        perf_df["_sort_order"] = perf_df["method"].apply(lambda x: get_method_category(x)[2])
        perf_df = perf_df.sort_values("_sort_order")
        
        perf_df["Package"] = perf_df["method"].apply(get_method_package)
        perf_df["Method"] = perf_df["method"].apply(format_method_name)
        
        categories = perf_df["_category"].unique()
        
        for category in categories:
            cat_df = perf_df[perf_df["_category"] == category].copy()
            
            md += f"### {category}\n\n"
            
            if "Preprocessing" in category or "QC" in category:
                cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "cells_kept", "genes_kept"]
            else:
                cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "groups", "genes"]
            
            # Add rerun columns if any row in this category has rerun data
            has_rerun = (
                "rerun_elapsed_seconds" in cat_df.columns
                and cat_df["rerun_elapsed_seconds"].notna().any()
            )
            if has_rerun:
                cols.append("rerun_elapsed_seconds")
                if "rerun_peak_memory_mb" in cat_df.columns and cat_df["rerun_peak_memory_mb"].notna().any():
                    cols.append("rerun_peak_memory_mb")
            
            cols = [c for c in cols if c in cat_df.columns]
            display_df = cat_df[cols].copy()
            
            rename_map = {
                "status": "Status",
                "elapsed_seconds": "Total (s)",
                "peak_memory_mb": "Memory (MB)",
                "cells_kept": "Cells",
                "genes_kept": "Genes",
                "groups": "Groups",
                "genes": "Genes",
                "rerun_elapsed_seconds": "Rerun (s)",
                "rerun_peak_memory_mb": "Rerun Mem (MB)",
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
            
            def _safe_int(x):
                """Safely convert to int, handling arrays, lists, and NA values."""
                if is_scalar_na(x):
                    return x
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return int(x)
                if isinstance(x, str):
                    try:
                        return int(float(x))
                    except (ValueError, TypeError):
                        return x
                # For lists, arrays, etc., return length or the original
                if isinstance(x, (list, np.ndarray)):
                    return len(x)
                return x
            
            for col in display_df.columns:
                if col in ["Cells", "Genes", "Groups"]:
                    display_df[col] = display_df[col].apply(_safe_int)
                elif col in display_df.select_dtypes(include=["number"]).columns:
                    display_df[col] = display_df[col].round(2)
            
            md += frame_to_markdown_table(display_df)
            md += "\n\n"
    
    # =========================================================================
    # Section 2: Performance Comparison
    # =========================================================================
    md += "## 2. Performance Comparison\n\n"
    
    if perf_comp_results:
        crispyx_comps = []
        other_comps = []
        
        for comp in perf_comp_results:
            comparison = comp["comparison"]
            method_a = comparison.split(" vs ")[0]
            if is_crispyx_method(method_a):
                crispyx_comps.append(comp)
            else:
                other_comps.append(comp)
        
        if crispyx_comps:
            md += "### crispyx vs Reference Tools\n\n"
            md += "_crispyx as baseline. Negative values = crispyx is faster/uses less memory._\n\n"
            
            comp_by_category: Dict[str, List[Dict[str, Any]]] = {}
            for comp in crispyx_comps:
                comparison = comp["comparison"]
                method_a = comparison.split(" vs ")[0]
                category = get_method_category(method_a)[0]
                
                if category not in comp_by_category:
                    comp_by_category[category] = []
                comp_by_category[category].append(comp)
            
            # Sort categories by predefined order
            sorted_categories = sorted(comp_by_category.keys(), key=get_category_sort_key)
            
            for category in sorted_categories:
                comps = comp_by_category[category]
                md += f"#### {category}\n\n"
                
                is_nb_glm = "NB GLM" in category
                rows = []
                for comp in comps:
                    comparison = comp["comparison"]
                    comp_type = comp.get("comp_type", "de")
                    parts = comparison.split(" vs ")
                    method_a_raw = parts[0]
                    method_b_raw = parts[1]
                    method_a = format_method_name(method_a_raw)
                    method_b = format_full_method_name(method_b_raw)
                    
                    time_pct = comp.get("time_pct")
                    mem_pct = comp.get("mem_pct")
                    time_diff = comp.get("time_diff_s")
                    mem_diff = comp.get("mem_diff_mb")
                    
                    time_emoji = get_performance_emoji(time_pct, is_lower_better=True)
                    mem_emoji = get_performance_emoji(mem_pct, is_lower_better=True)
                    
                    row = {
                        "crispyx method": method_a,
                        "compared to": method_b,
                        "Time Δ": format_diff(time_diff, "s"),
                        "Time %": format_pct(time_pct),
                        "": time_emoji,
                        "Mem Δ": format_diff(mem_diff, " MB"),
                        "Mem %": format_pct(mem_pct),
                        " ": mem_emoji,
                    }
                    rows.append(row)
                
                # Sort rows by method order: t-test, Wilcoxon, NB-GLM
                rows.sort(key=lambda r: (get_method_sort_key(r["crispyx method"]), r["crispyx method"], r["compared to"]))
                md += frame_to_markdown_table(pd.DataFrame(rows))
                md += "\n\n"
        
        if other_comps:
            md += "### Tool Comparisons\n\n"
            md += "_Comparisons between external tools._\n\n"
            
            rows = []
            for comp in other_comps:
                comparison = comp["comparison"]
                parts = comparison.split(" vs ")
                method_a = format_method_name(parts[0])
                method_b = format_method_name(parts[1])
                
                time_pct = comp.get("time_pct")
                mem_pct = comp.get("mem_pct")
                time_diff = comp.get("time_diff_s")
                mem_diff = comp.get("mem_diff_mb")
                
                time_emoji = get_performance_emoji(time_pct, is_lower_better=True)
                mem_emoji = get_performance_emoji(mem_pct, is_lower_better=True)
                
                rows.append({
                    "package A": get_method_package(parts[0]),
                    "method A": method_a,
                    "package B": get_method_package(parts[1]),
                    "method B": method_b,
                    "Time Δ (A-B)": format_diff(time_diff, "s"),
                    "Time % (A/B)": format_pct(time_pct),
                    "": time_emoji,
                    "Mem Δ (A-B)": format_diff(mem_diff, " MB"),
                    "Mem % (A/B)": format_pct(mem_pct),
                    " ": mem_emoji,
                })
            
            md += frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n\n"
    
    # =========================================================================
    # Section 3: Accuracy Comparison
    # =========================================================================
    md += "## 3. Accuracy\n\n"
    md += "_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_\n\n"
    
    if accuracy_results:
        crispyx_accs = []
        other_accs = []
        
        for acc in accuracy_results:
            comparison = acc["comparison"]
            method_a = comparison.split(" vs ")[0]
            if is_crispyx_method(method_a):
                crispyx_accs.append(acc)
            else:
                other_accs.append(acc)
        
        if crispyx_accs:
            acc_by_category: Dict[str, List[Dict[str, Any]]] = {}
            
            for acc in crispyx_accs:
                comparison = acc["comparison"]
                method_a = comparison.split(" vs ")[0]
                category = get_method_category(method_a)[0]
                
                if category not in acc_by_category:
                    acc_by_category[category] = []
                acc_by_category[category].append(acc)
            
            # Sort categories by predefined order
            sorted_categories = sorted(acc_by_category.keys(), key=get_category_sort_key)
            
            for category in sorted_categories:
                accs = acc_by_category[category]
                md += f"### {category}\n\n"
                
                is_qc = "QC" in category or "Preprocessing" in category
                is_nb_glm = "NB GLM" in category
                
                if is_qc:
                    rows = []
                    for acc in accs:
                        comparison = acc["comparison"]
                        parts = comparison.split(" vs ")
                        method_a = format_method_name(parts[0])
                        method_b = format_full_method_name(parts[1])
                        
                        cells_diff = acc.get("cells_diff", 0)
                        genes_diff = acc.get("genes_diff", 0)
                        
                        cells_emoji = "✅" if cells_diff == 0 else ("⚠️" if abs(cells_diff) < 10 else "❌")
                        genes_emoji = "✅" if genes_diff == 0 else ("⚠️" if abs(genes_diff) < 10 else "❌")
                        
                        rows.append({
                            "crispyx method": method_a,
                            "compared to": method_b,
                            "Cells Δ": f"{int(cells_diff):+d}" if is_scalar_notna(cells_diff) else "-",
                            "": cells_emoji,
                            "Genes Δ": f"{int(genes_diff):+d}" if is_scalar_notna(genes_diff) else "-",
                            " ": genes_emoji,
                        })
                    
                    md += frame_to_markdown_table(pd.DataFrame(rows))
                else:
                    rows = []
                    for acc in accs:
                        comparison = acc["comparison"]
                        comp_type = acc.get("comp_type", "de")
                        parts = comparison.split(" vs ")
                        method_a_raw = parts[0]
                        method_b_raw = parts[1]
                        method_a = format_method_name(method_a_raw)
                        method_b = format_full_method_name(method_b_raw)
                        
                        effect_p_mean = acc.get("effect_pearson_corr_mean")
                        effect_p_std = acc.get("effect_pearson_corr_std")
                        effect_s_mean = acc.get("effect_spearman_corr_mean")
                        effect_s_std = acc.get("effect_spearman_corr_std")
                        stat_p_mean = acc.get("statistic_pearson_corr_mean")
                        stat_p_std = acc.get("statistic_pearson_corr_std")
                        stat_s_mean = acc.get("statistic_spearman_corr_mean")
                        stat_s_std = acc.get("statistic_spearman_corr_std")
                        pval_p_mean = acc.get("pvalue_log_pearson_corr_mean")
                        pval_p_std = acc.get("pvalue_log_pearson_corr_std")
                        pval_s_mean = acc.get("pvalue_log_spearman_corr_mean")
                        pval_s_std = acc.get("pvalue_log_spearman_corr_std")
                        
                        row = {
                            "crispyx method": method_a,
                            "compared to": method_b,
                            "Eff ρ": format_mean_std(effect_p_mean, effect_p_std),
                            "": get_accuracy_emoji(effect_p_mean),
                            "Eff ρₛ": format_mean_std(effect_s_mean, effect_s_std),
                            " ": get_accuracy_emoji(effect_s_mean),
                            "Stat ρ": format_mean_std(stat_p_mean, stat_p_std),
                            "  ": get_accuracy_emoji(stat_p_mean),
                            "Stat ρₛ": format_mean_std(stat_s_mean, stat_s_std),
                            "   ": get_accuracy_emoji(stat_s_mean),
                            "log-Pval ρ": format_mean_std(pval_p_mean, pval_p_std),
                            "    ": get_accuracy_emoji(pval_p_mean),
                            "log-Pval ρₛ": format_mean_std(pval_s_mean, pval_s_std),
                            "     ": get_accuracy_emoji(pval_s_mean),
                        }
                        rows.append(row)
                    
                    # Sort rows by method order: t-test, Wilcoxon, NB-GLM
                    rows.sort(key=lambda r: (get_method_sort_key(r["crispyx method"]), r["crispyx method"], r["compared to"]))
                    md += frame_to_markdown_table(pd.DataFrame(rows))
                
                md += "\n\n"
        
        if other_accs:
            md += "### Tool Comparisons\n\n"
            
            rows = []
            for acc in other_accs:
                comparison = acc["comparison"]
                parts = comparison.split(" vs ")
                method_a = format_method_name(parts[0])
                method_b = format_method_name(parts[1])
                
                effect_p_mean = acc.get("effect_pearson_corr_mean")
                effect_p_std = acc.get("effect_pearson_corr_std")
                effect_s_mean = acc.get("effect_spearman_corr_mean")
                effect_s_std = acc.get("effect_spearman_corr_std")
                stat_p_mean = acc.get("statistic_pearson_corr_mean")
                stat_p_std = acc.get("statistic_pearson_corr_std")
                stat_s_mean = acc.get("statistic_spearman_corr_mean")
                stat_s_std = acc.get("statistic_spearman_corr_std")
                pval_p_mean = acc.get("pvalue_log_pearson_corr_mean")
                pval_p_std = acc.get("pvalue_log_pearson_corr_std")
                pval_s_mean = acc.get("pvalue_log_spearman_corr_mean")
                pval_s_std = acc.get("pvalue_log_spearman_corr_std")
                
                rows.append({
                    "package A": get_method_package(parts[0]),
                    "method A": method_a,
                    "package B": get_method_package(parts[1]),
                    "method B": method_b,
                    "Eff ρ": format_mean_std(effect_p_mean, effect_p_std),
                    "": get_accuracy_emoji(effect_p_mean),
                    "Eff ρₛ": format_mean_std(effect_s_mean, effect_s_std),
                    " ": get_accuracy_emoji(effect_s_mean),
                    "Stat ρ": format_mean_std(stat_p_mean, stat_p_std),
                    "  ": get_accuracy_emoji(stat_p_mean),
                    "Stat ρₛ": format_mean_std(stat_s_mean, stat_s_std),
                    "   ": get_accuracy_emoji(stat_s_mean),
                    "log-Pval ρ": format_mean_std(pval_p_mean, pval_p_std),
                    "    ": get_accuracy_emoji(pval_p_mean),
                    "log-Pval ρₛ": format_mean_std(pval_s_mean, pval_s_std),
                    "     ": get_accuracy_emoji(pval_s_mean),
                })
            
            md += frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n\n"
    
    # =========================================================================
    # Section 4: Gene Set Overlap
    # =========================================================================
    md += "## 4. Gene Set Overlap\n\n"
    md += "_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_\n\n"
    
    if accuracy_results:
        for metric_type, metric_label in [("effect", "Effect Size"), ("pvalue", "P-value")]:
            md += f"### {metric_label} Overlap\n\n"
            
            rows = []
            for acc in accuracy_results:
                comparison = acc["comparison"]
                comp_type = acc.get("comp_type", "de")
                parts = comparison.split(" vs ")
                method_a_raw = parts[0]
                method_b_raw = parts[1]
                
                # Only include crispyx comparisons in this table
                if not is_crispyx_method(method_a_raw):
                    continue
                
                # Skip certain methods based on metric type:
                # - P-value: exclude crispyx _pydeseq2 variants and lfcShrink (p-values are identical to base)
                # - Effect size: exclude crispyx _pydeseq2 base without lfcShrink (LFCs are identical to base)
                # Note: "pertpy_de_pydeseq2" in method_b should NOT be excluded - it's the reference!
                if metric_type == "pvalue":
                    # Exclude crispyx pydeseq2 variants from p-value tables (same p-values as base)
                    # Only check method_a (crispyx side), not method_b (reference tool side)
                    if "crispyx" in method_a_raw and "pydeseq2" in method_a_raw:
                        continue
                    if comp_type == "de_lfcshrink":
                        continue
                elif metric_type == "effect":
                    # Exclude crispyx pydeseq2 base (without lfcShrink) - only keep pydeseq2 with lfcShrink
                    is_crispyx_pydeseq2_base = (
                        "crispyx" in method_a_raw and "pydeseq2" in method_a_raw and
                        comp_type != "de_lfcshrink"
                    )
                    if is_crispyx_pydeseq2_base:
                        continue
                
                method_a = format_method_name(method_a_raw)
                method_b = format_full_method_name(method_b_raw)
                
                # Remove "(lfcShrink)" suffix from method_b display name since we have a column for it
                method_b = method_b.replace(" (lfcShrink)", "")
                
                k50 = acc.get(f"{metric_type}_top_k_overlap")
                k100_mean = acc.get(f"{metric_type}_top_100_overlap_mean")
                k100_std = acc.get(f"{metric_type}_top_100_overlap_std")
                k500_mean = acc.get(f"{metric_type}_top_500_overlap_mean")
                k500_std = acc.get(f"{metric_type}_top_500_overlap_std")
                
                if k50 is None and k100_mean is None and k500_mean is None:
                    continue
                
                def _get_overlap_emoji(val: Optional[float]) -> str:
                    if val is None or is_scalar_na(val):
                        return ""
                    if val >= 0.7:
                        return "✅"
                    elif val >= 0.5:
                        return "⚠️"
                    else:
                        return "❌"
                
                rows.append({
                    "crispyx method": method_a,
                    "compared to": method_b,
                    "Top-50": f"{k50:.3f}" if k50 is not None and is_scalar_notna(k50) else "-",
                    "": _get_overlap_emoji(k50),
                    "Top-100": format_mean_std(k100_mean, k100_std),
                    " ": _get_overlap_emoji(k100_mean),
                    "Top-500": format_mean_std(k500_mean, k500_std),
                    "  ": _get_overlap_emoji(k500_mean),
                })
            
            if rows:
                # Sort rows by method order: t-test, Wilcoxon, NB-GLM
                rows.sort(key=lambda r: (get_method_sort_key(r["crispyx method"]), r["crispyx method"], r["compared to"]))
                md += frame_to_markdown_table(pd.DataFrame(rows))
            else:
                md += "_No overlap data available._\n"
            md += "\n\n"
            
            # Add note about skipped comparisons for P-value section
            if metric_type == "pvalue" and skipped_comparisons:
                # Filter for DE comparisons that would have affected this table
                de_skipped = [
                    sc for sc in skipped_comparisons 
                    if sc.get("comp_type") in ("de",) and "crispyx" in sc.get("method_a", "")
                ]
                if de_skipped:
                    md += "_Note: Some methods are missing due to errors:_\n"
                    for sc in de_skipped:
                        method_a = format_method_name(sc.get("method_a", ""))
                        method_b = format_full_method_name(sc.get("method_b", ""))
                        reason = sc.get("reason", "unknown error")
                        md += f"- {method_a} vs {method_b}: _{reason}_\n"
                    md += "\n"
    else:
        md += "_No overlap data available._\n\n"
    
    # Embed top-100 overlap heatmaps
    if overlap_heatmaps:
        md += "### Overlap Heatmaps (Top-100)\n\n"
        
        for metric, label in [("effect", "Effect Size"), ("pvalue", "P-value")]:
            heatmap_key = f"benchmark_{metric}_top_100_overlap.png"
            if heatmap_key in overlap_heatmaps:
                md += f"#### {label}\n\n"
                md += f"![{label} Top-100 Overlap]({heatmap_key})\n\n"
    
    # =========================================================================
    # Section 5: Shrinkage Quality
    # =========================================================================
    if shrinkage_quality:
        md += "## 5. Shrinkage Quality\n\n"
        md += "_Shrinkage should reduce LFC magnitude toward zero. Genes with |shrunk| > |raw| indicate aberrant shrinkage._\n\n"
        md += "_✅ <1% inflated | ⚠️ 1-10% inflated | ❌ >10% inflated_\n\n"
        
        rows = []
        for sq in shrinkage_quality:
            pct = sq.get("pct_inflated", 0)
            max_inf = sq.get("max_inflation", 1.0)
            median = sq.get("median_ratio", 1.0)
            genes = sq.get("genes_analyzed", 0)
            method = sq.get("method", "")
            
            # Get emoji for percentage inflated
            if pct < 1.0:
                pct_emoji = "✅"
            elif pct < 10.0:
                pct_emoji = "⚠️"
            else:
                pct_emoji = "❌"
            
            rows.append({
                "Method": format_full_method_name(method),
                "% Inflated": f"{pct:.1f}%",
                "": pct_emoji,
                "Max Inflation": f"{max_inf:.2f}×",
                "Median Ratio": f"{median:.3f}",
                "Genes": f"{genes:,}",
            })
        
        if rows:
            md += frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n"
            md += "_Note: A proper apeGLM implementation should have 0% inflated genes. "
            md += "The 'Median Ratio' shows how much shrinkage is applied on average (lower = more shrinkage)._\n"
        md += "\n"
    
    # =========================================================================
    # Legend
    # =========================================================================
    md += "---\n\n"
    md += "**Legend:**\n"
    md += "- **Performance:** ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse\n"
    md += "- **Accuracy:** ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8\n"
    md += "- **Overlap:** ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5\n"
    md += "- **Shrinkage:** ✅ <1% inflated | ⚠️ 1-10% inflated | ❌ >10% inflated\n\n"
    md += "**Abbreviations:**\n"
    md += "- ρ = Pearson correlation, ρₛ = Spearman correlation\n"
    md += "- log-Pval = correlations on -log₁₀(p) transformed values\n"
    md += "- sf=per = per-comparison size factor estimation (matches PyDESeq2)\n\n"
    md += "**Notes:**\n"
    md += "- Correlation and overlap values shown as mean±std across perturbations\n"
    md += "- crispyx lfcShrink uses `method='stats'` (Gaussian approximation) which is numerically stable and ~35× faster than `method='full'`.\n"
    md += "- P-value overlap excludes lfcShrink methods since shrinkage only affects effect sizes, not p-values.\n"
    md += "- **Warning:** PyDESeq2 may produce aberrant shrinkage when dispersion trend fitting fails. crispyx shrinkage is more robust.\n"
    
    return md


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_benchmarks(output_dir: Path) -> None:
    """Evaluate benchmark results and generate report.
    
    This function loads cached benchmark results, computes comparison metrics,
    generates overlap heatmaps, and writes summary files.
    
    Output files (all with benchmark_ prefix):
    - benchmark_performance.csv: Performance metrics table
    - benchmark_accuracy.csv: Accuracy comparison metrics
    - benchmark_report.md: Full markdown report
    - benchmark_effect_top_{k}_overlap.png: Effect size overlap heatmaps
    - benchmark_pvalue_top_{k}_overlap.png: P-value overlap heatmaps
    
    Parameters
    ----------
    output_dir : Path
        Directory containing .benchmark_cache with cached results
    """
    from .comparison import compute_de_comparison_metrics
    
    results_dir = output_dir / ".benchmark_cache"
    if not results_dir.exists():
        return
        
    # Load all results
    results = load_cached_results(output_dir)
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # Rename max_memory_mb to peak_memory_mb if needed
    if "max_memory_mb" in df.columns and "peak_memory_mb" not in df.columns:
        df = df.rename(columns={"max_memory_mb": "peak_memory_mb"})
    
    # Generate Performance Table
    # Include step-wise profiling columns for NB-GLM methods (2-phase: fit + shrinkage)
    perf_cols = [
        "method", "status", "elapsed_seconds", "spawn_overhead_seconds",
        "import_seconds", "load_seconds", "process_seconds", 
        "de_seconds", "convert_seconds", "save_seconds",
        # Step-wise timing for NB-GLM methods
        "base_seconds", "shrinkage_seconds",
        # Memory metrics (including step-wise for NB-GLM)
        "peak_memory_mb", "avg_memory_mb",
        "base_peak_memory_mb", "shrinkage_peak_memory_mb",
        "cells_kept", "genes_kept", "groups"
    ]
    perf_df = df[[c for c in perf_cols if c in df.columns]].copy()
    if "method" in perf_df.columns:
        perf_df = perf_df.sort_values("method")
    
    # Load reference extraction marker to annotate rerun data
    marker_path = output_dir / ".reference_extracted"
    rerun_marker: Dict[str, Any] = {}
    if marker_path.exists():
        try:
            with open(marker_path, 'r') as f:
                rerun_marker = json.load(f).get("extracted_methods", {})
        except (json.JSONDecodeError, OSError):
            pass
    
    # Annotate performance rows for methods that were rerun without limits
    if rerun_marker and "method" in perf_df.columns and "status" in perf_df.columns:
        for idx, row in perf_df.iterrows():
            method = row["method"]
            status = row.get("status", "")
            if method in rerun_marker and status in ("timeout", "error", "memory_limit"):
                rerun_info = rerun_marker[method]
                perf_df.at[idx, "status"] = f"{status} (rerun: no limits)"
                rerun_time = rerun_info.get("elapsed_seconds")
                rerun_mem = rerun_info.get("peak_memory_mb")
                if rerun_time is not None:
                    perf_df.at[idx, "rerun_elapsed_seconds"] = rerun_time
                if rerun_mem is not None:
                    perf_df.at[idx, "rerun_peak_memory_mb"] = rerun_mem
    
    accuracy_results = []
    perf_comp_results = []
    skipped_comparisons = []  # Track skipped comparisons with reasons
    de_results_for_heatmaps: Dict[str, pd.DataFrame] = {}
    
    # Define comparisons
    # Note: We compare like-with-like:
    # - base crispyx vs base tools (edgeR, PyDESeq2)
    # - lfcShrink crispyx vs lfcShrink PyDESeq2 (standalone shrinkage methods)
    # - lfcShrink crispyx also vs edgeR (as a reference baseline)
    comparisons = [
        ("crispyx_qc_filtered", "scanpy_qc_filtered", "qc"),
        # DE GLM - base NB-GLM vs external tools (base LFCs)
        ("crispyx_de_nb_glm", "edger_de_glm", "de"),
        ("crispyx_de_nb_glm", "pertpy_de_pydeseq2", "de"),
        # DE GLM - standalone lfcShrink vs PyDESeq2 lfcShrink
        ("crispyx_de_lfcshrink", "pertpy_de_lfcshrink", "de_lfcshrink"),
        # DE GLM - external tool comparison (base)
        ("edger_de_glm", "pertpy_de_pydeseq2", "de"),
        # DE Tests
        ("crispyx_de_t_test", "scanpy_de_t_test", "de"),
        ("crispyx_de_wilcoxon", "scanpy_de_wilcoxon", "de"),
    ]
    
    # Add conditional comparisons for optional PyDESeq2-parity lfcShrink method (only if it was run)
    # pydeseq2 variant uses per-comparison size factor/dispersion estimation (matches PyDESeq2 behavior exactly)
    optional_method_results = df[df["method"] == "crispyx_de_lfcshrink_pydeseq2"]
    if not optional_method_results.empty:
        row = optional_method_results.iloc[0]
        if has_valid_result(row, output_dir):
            # PyDESeq2-parity lfcShrink vs PyDESeq2 lfcShrink (for parity testing)
            comparisons.extend([
                ("crispyx_de_lfcshrink_pydeseq2", "pertpy_de_lfcshrink", "de_lfcshrink"),
                # Also compare default vs PyDESeq2-parity variants
                ("crispyx_de_lfcshrink", "crispyx_de_lfcshrink_pydeseq2", "de_lfcshrink"),
            ])
    
    def _load_de_result(method_name: str, result_path_val: Optional[str], use_raw_lfc: bool = False) -> Optional[pd.DataFrame]:
        """Load and standardize a DE result from file.
        
        Uses resolve_result_path to fallback to expected path if result_path is missing.
        """
        # Resolve the path with fallback to expected output path
        result_path = resolve_result_path(method_name, result_path_val, output_dir)
        if result_path is None:
            return None
            
        try:
            if str(result_path).endswith('.h5ad'):
                import anndata as ad
                adata = ad.read_h5ad(str(result_path))
                if use_raw_lfc:
                    result_dict = _anndata_to_de_dict_raw(adata)
                else:
                    result_dict = _anndata_to_de_dict(adata)
                return _streaming_de_to_frame(result_dict)
            else:
                result_df = pd.read_csv(result_path)
                return standardise_de_dataframe(result_df)
        except Exception as e:
            print(f"Warning: Could not load DE result for {method_name}: {e}")
            return None
    
    # Explicitly collect all DE methods for heatmaps (base results)
    # Also check for outputs from reference extraction (no cache entry but file exists)
    for method_name in ALL_DE_METHODS_FOR_HEATMAP:
        if method_name in de_results_for_heatmaps:
            continue  # Already loaded
        
        method_res = df[df["method"] == method_name]
        if not method_res.empty:
            row = method_res.iloc[0]
            if has_valid_result(row, output_dir):
                result_path_val = row.get("result_path")
                method_df = _load_de_result(method_name, result_path_val)
                if method_df is not None:
                    de_results_for_heatmaps[method_name] = method_df
                    continue
        
        # No cache entry or invalid - check for output file directly (reference extraction)
        resolved_path = resolve_result_path(method_name, None, output_dir)
        if resolved_path is not None:
            method_df = _load_de_result(method_name, str(resolved_path))
            if method_df is not None:
                de_results_for_heatmaps[method_name] = method_df
    
    # Note: With standalone lfcShrink methods, shrunk results are now loaded directly
    # via ALL_DE_METHODS_FOR_HEATMAP (crispyx_de_lfcshrink, pertpy_de_lfcshrink, etc.)
    # No need for separate shrunk_result_path handling anymore.
    
    for method_a_name, method_b_name, comp_type in comparisons:
        method_a_res = df[df["method"] == method_a_name]
        method_b_res = df[df["method"] == method_b_name]
        
        # Check if reference outputs exist even without cache entries
        # This allows accuracy comparison with outputs from reference extraction
        method_a_has_output = False
        method_b_has_output = False
        method_a_resolved = None
        method_b_resolved = None
        
        if not method_a_res.empty:
            method_a_path_val = method_a_res.iloc[0].get("result_path")
            method_a_resolved = resolve_result_path(method_a_name, method_a_path_val, output_dir)
            method_a_has_output = method_a_resolved is not None
        else:
            # No cache entry - check for output file directly (from reference extraction)
            method_a_resolved = resolve_result_path(method_a_name, None, output_dir)
            method_a_has_output = method_a_resolved is not None
        
        if not method_b_res.empty:
            method_b_path_val = method_b_res.iloc[0].get("result_path")
            method_b_resolved = resolve_result_path(method_b_name, method_b_path_val, output_dir)
            method_b_has_output = method_b_resolved is not None
        else:
            # No cache entry - check for output file directly (from reference extraction)
            method_b_resolved = resolve_result_path(method_b_name, None, output_dir)
            method_b_has_output = method_b_resolved is not None
        
        # Determine if we can proceed with comparison
        # For accuracy comparison: we only need output files to exist
        # For performance comparison: we need cache entries with timing data
        can_compare_accuracy = method_a_has_output and method_b_has_output
        can_compare_performance = (
            not method_a_res.empty and not method_b_res.empty and
            has_valid_result(method_a_res.iloc[0], output_dir) and
            has_valid_result(method_b_res.iloc[0], output_dir)
        )
        
        if not can_compare_accuracy:
            # Track skipped comparison due to missing output
            missing = []
            if not method_a_has_output:
                if method_a_res.empty:
                    missing.append(f"{method_a_name} (not run, no output)")
                else:
                    missing.append(f"{method_a_name} (no output file)")
            if not method_b_has_output:
                if method_b_res.empty:
                    missing.append(f"{method_b_name} (not run, no output)")
                else:
                    missing.append(f"{method_b_name} (no output file)")
            skipped_comparisons.append({
                "method_a": method_a_name,
                "method_b": method_b_name,
                "comp_type": comp_type,
                "reason": f"missing output: {', '.join(missing)}",
            })
            continue
        
        # Get row data for methods that have cache entries
        a_row = method_a_res.iloc[0] if not method_a_res.empty else None
        b_row = method_b_res.iloc[0] if not method_b_res.empty else None
        
        # Performance comparison only if we have cache entries with timing data
        if can_compare_performance:
            a_time = a_row.get("elapsed_seconds", np.nan) if a_row is not None else np.nan
            b_time = b_row.get("elapsed_seconds", np.nan) if b_row is not None else np.nan
            
            a_mem = a_row.get("peak_memory_mb", np.nan) if a_row is not None else np.nan
            b_mem = b_row.get("peak_memory_mb", np.nan) if b_row is not None else np.nan
            
            comp = {
                "comparison": f"{method_a_name} vs {method_b_name}",
                "comp_type": comp_type,
                "method_a_time_s": a_time,
                "method_b_time_s": b_time,
                "time_diff_s": a_time - b_time if is_scalar_notna(a_time) and is_scalar_notna(b_time) else None,
                "time_pct": (a_time / b_time * 100) if is_scalar_notna(a_time) and is_scalar_notna(b_time) and b_time > 0 else None,
                "method_a_mem_mb": a_mem,
                "method_b_mem_mb": b_mem,
                "mem_diff_mb": a_mem - b_mem if is_scalar_notna(a_mem) and is_scalar_notna(b_mem) else None,
                "mem_pct": (a_mem / b_mem * 100) if is_scalar_notna(a_mem) and is_scalar_notna(b_mem) and b_mem > 0 else None,
            }
            perf_comp_results.append(comp)
        
        # Accuracy comparison uses the resolved paths (can work from reference extraction)
        try:
            # Use already-resolved paths (handles cache entry or direct file detection)
            method_a_path_val = str(method_a_resolved) if method_a_resolved else None
            method_b_path_val = str(method_b_resolved) if method_b_resolved else None
            
            if method_a_resolved is None or method_b_resolved is None:
                print(f"Skipping comparison {method_a_name} vs {method_b_name}: missing result file")
                continue
            
            if comp_type == "qc":
                # QC comparison requires cache entries with cells_kept/genes_kept
                if a_row is None or b_row is None:
                    print(f"Skipping QC comparison {method_a_name} vs {method_b_name}: missing cache data")
                    continue
                acc = {
                    "comparison": f"{method_a_name} vs {method_b_name}",
                    "comp_type": comp_type,
                    "cells_diff": float(a_row["cells_kept"] - b_row["cells_kept"]),
                    "genes_diff": float(a_row["genes_kept"] - b_row["genes_kept"]),
                }
                accuracy_results.append(acc)
                
            elif comp_type == "de":
                method_a_df = _load_de_result(method_a_name, method_a_path_val)
                method_b_df = _load_de_result(method_b_name, method_b_path_val)
                
                if method_a_df is not None and method_a_name not in de_results_for_heatmaps:
                    de_results_for_heatmaps[method_a_name] = method_a_df
                if method_b_df is not None and method_b_name not in de_results_for_heatmaps:
                    de_results_for_heatmaps[method_b_name] = method_b_df
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name}: could not load results")
                    continue
                
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name}", "comp_type": comp_type}
                acc.update(metrics)
                accuracy_results.append(acc)
            
            elif comp_type == "de_raw":
                method_a_df = _load_de_result(method_a_name, method_a_path_val, use_raw_lfc=True)
                method_b_df = _load_de_result(method_b_name, method_b_path_val, use_raw_lfc=True)
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name}: could not load results")
                    continue
                
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name}", "comp_type": comp_type}
                acc.update(metrics)
                accuracy_results.append(acc)
            
            elif comp_type == "de_lfcshrink":
                method_a_df = _load_de_result(method_a_name, method_a_path_val, use_raw_lfc=False)
                method_b_df = _load_de_result(method_b_name, method_b_path_val, use_raw_lfc=False)
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name} (lfcShrink): could not load results")
                    continue
                
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name}", "comp_type": comp_type}
                acc.update(metrics)
                accuracy_results.append(acc)
                
        except Exception as e:
            print(f"Error comparing {method_a_name} vs {method_b_name}: {e}")
    
    # Generate overlap heatmaps
    overlap_heatmaps: Dict[str, Path] = {}
    if de_results_for_heatmaps:
        try:
            from .visualization import generate_overlap_heatmaps
            overlap_heatmaps = generate_overlap_heatmaps(
                de_results_for_heatmaps,
                output_dir,
                k_values=(50, 100, 500),
            )
        except Exception as e:
            print(f"Warning: Could not generate overlap heatmaps: {e}")
    
    # Compute shrinkage quality metrics
    shrinkage_quality: List[Dict[str, Any]] = []
    
    # Define pairs of (raw_method, shrunk_method) for shrinkage quality analysis
    shrinkage_pairs = [
        # crispyx: base NB-GLM vs lfcShrink
        ("crispyx_de_nb_glm", "crispyx_de_lfcshrink"),
        # crispyx PyDESeq2-parity variant
        ("crispyx_de_nb_glm_pydeseq2", "crispyx_de_lfcshrink_pydeseq2"),
        # pertpy/PyDESeq2
        ("pertpy_de_pydeseq2", "pertpy_de_lfcshrink"),
    ]
    
    # Track which shrunk methods we've already processed
    processed_shrunk_methods = set()
    
    for raw_method, shrunk_method in shrinkage_pairs:
        raw_row = df[df["method"] == raw_method]
        shrunk_row = df[df["method"] == shrunk_method]
        
        if shrunk_row.empty:
            continue
        
        shrunk_path = resolve_result_path(
            shrunk_method, shrunk_row.iloc[0].get("result_path"), output_dir
        )
        
        if shrunk_path is None:
            continue
        
        # Try paired approach first (raw + shrunk files)
        sq = None
        if not raw_row.empty:
            raw_path = resolve_result_path(
                raw_method, raw_row.iloc[0].get("result_path"), output_dir
            )
            if raw_path is not None:
                sq = _compute_shrinkage_quality(raw_path, shrunk_path, shrunk_method, output_dir)
        
        # Fallback: try standalone computation from shrunk file alone
        # (lfcShrink h5ad files contain both raw and shrunk LFCs in layers)
        if sq is None:
            sq = _compute_shrinkage_quality_standalone(shrunk_path, shrunk_method)
        
        if sq is not None:
            shrinkage_quality.append(sq)
            processed_shrunk_methods.add(shrunk_method)
    
    # Also check for any standalone lfcShrink methods that weren't in pairs
    standalone_shrink_methods = [
        "crispyx_de_lfcshrink",
        "crispyx_de_lfcshrink_pydeseq2",
        "pertpy_de_lfcshrink",
    ]
    
    for shrunk_method in standalone_shrink_methods:
        if shrunk_method in processed_shrunk_methods:
            continue
        
        shrunk_row = df[df["method"] == shrunk_method]
        if shrunk_row.empty:
            continue
        
        shrunk_path = resolve_result_path(
            shrunk_method, shrunk_row.iloc[0].get("result_path"), output_dir
        )
        
        if shrunk_path is None:
            continue
        
        sq = _compute_shrinkage_quality_standalone(shrunk_path, shrunk_method)
        if sq is not None:
            shrinkage_quality.append(sq)
            
    # Save tables with benchmark_ prefix
    perf_df.to_csv(output_dir / "benchmark_performance.csv", index=False)
    if accuracy_results:
        acc_df = pd.DataFrame(accuracy_results)
        acc_df.to_csv(output_dir / "benchmark_accuracy.csv", index=False)
        
    # Generate improved Markdown
    md = _generate_improved_markdown(
        perf_df, 
        perf_comp_results, 
        accuracy_results,
        overlap_heatmaps,
        shrinkage_quality,
        skipped_comparisons,
    )
        
    with open(output_dir / "benchmark_report.md", "w") as f:
        f.write(md)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI entry point for regenerating benchmark reports."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from cached results."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing .benchmark_cache with cached results",
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Directory {args.output_dir} does not exist")
        return
    
    cache_dir = args.output_dir / ".benchmark_cache"
    if not cache_dir.exists():
        print(f"Error: No .benchmark_cache directory found in {args.output_dir}")
        return
    
    print(f"Generating benchmark report for {args.output_dir}")
    evaluate_benchmarks(args.output_dir)
    print(f"Report generated: {args.output_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
