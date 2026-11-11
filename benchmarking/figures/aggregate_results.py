#!/usr/bin/env python3
"""
Aggregate benchmark results from all datasets.

Scans benchmarking/results/*/benchmark_performance.csv and benchmark_accuracy.csv
to create unified DataFrames for visualization.
"""

import os
from pathlib import Path
import pandas as pd


def get_results_dir() -> Path:
    """Get the benchmarking results directory."""
    return Path(__file__).parent.parent / "results"


def aggregate_performance_data(results_dir: Path = None) -> pd.DataFrame:
    """
    Aggregate benchmark_performance.csv from all dataset directories.
    
    Returns:
        DataFrame with columns: dataset, method, status, elapsed_seconds, 
        peak_memory_mb, avg_memory_mb, cells_kept, genes_kept, groups, ...
    """
    if results_dir is None:
        results_dir = get_results_dir()
    
    all_data = []
    
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        perf_file = dataset_dir / "benchmark_performance.csv"
        if not perf_file.exists():
            print(f"Skipping {dataset_dir.name}: no benchmark_performance.csv")
            continue
        
        try:
            df = pd.read_csv(perf_file)
            df.insert(0, "dataset", dataset_dir.name)
            all_data.append(df)
            print(f"Loaded {dataset_dir.name}: {len(df)} methods")
        except Exception as e:
            print(f"Error loading {perf_file}: {e}")
    
    if not all_data:
        raise ValueError("No performance data found")
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def aggregate_accuracy_data(results_dir: Path = None) -> pd.DataFrame:
    """
    Aggregate benchmark_accuracy.csv from all dataset directories.
    
    Returns:
        DataFrame with columns: dataset, comparison, comp_type, 
        effect_pearson_corr_mean, pvalue_top_100_overlap_mean, ...
    """
    if results_dir is None:
        results_dir = get_results_dir()
    
    all_data = []
    
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        acc_file = dataset_dir / "benchmark_accuracy.csv"
        if not acc_file.exists():
            continue
        
        try:
            df = pd.read_csv(acc_file)
            df.insert(0, "dataset", dataset_dir.name)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {acc_file}: {e}")
    
    if not all_data:
        print("Warning: No accuracy data found")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def get_dataset_metadata(results_dir: Path = None) -> pd.DataFrame:
    """
    Extract dataset metadata (cells, genes, perturbations) from performance data.
    
    Returns:
        DataFrame with columns: dataset, cells, genes, perturbations
    """
    if results_dir is None:
        results_dir = get_results_dir()
    
    perf_df = aggregate_performance_data(results_dir)
    
    # Get cells/genes from QC methods
    qc_data = perf_df[perf_df["method"].str.contains("qc_filtered")]
    
    metadata = []
    for dataset in perf_df["dataset"].unique():
        dataset_df = perf_df[perf_df["dataset"] == dataset]
        qc_df = qc_data[qc_data["dataset"] == dataset]
        
        # Get cells and genes from QC
        cells = None
        genes = None
        if not qc_df.empty:
            cells = qc_df["cells_kept"].dropna().iloc[0] if "cells_kept" in qc_df.columns and not qc_df["cells_kept"].dropna().empty else None
            genes = qc_df["genes_kept"].dropna().iloc[0] if "genes_kept" in qc_df.columns and not qc_df["genes_kept"].dropna().empty else None
        
        # Get perturbations from DE methods
        groups = dataset_df["groups"].dropna()
        perturbations = int(groups.iloc[0]) if not groups.empty else None
        
        metadata.append({
            "dataset": dataset,
            "cells": int(cells) if cells is not None else None,
            "genes": int(genes) if genes is not None else None,
            "perturbations": perturbations,
        })
    
    return pd.DataFrame(metadata)


def categorize_method(method: str) -> dict:
    """
    Categorize a method by tool and analysis type.
    
    Returns:
        Dict with 'tool', 'analysis_type', 'method_type'
    """
    if method.startswith("crispyx_"):
        tool = "crispyx"
    elif method.startswith("scanpy_"):
        tool = "Scanpy"
    elif method.startswith("edger_"):
        tool = "edgeR"
    elif method.startswith("pertpy_"):
        tool = "Pertpy"
    else:
        tool = "unknown"
    
    if "qc" in method:
        analysis_type = "QC"
    elif "pb_" in method:
        analysis_type = "Pseudobulk"
    elif "de_" in method or "glm" in method:
        analysis_type = "DE"
    else:
        analysis_type = "other"
    
    if "t_test" in method:
        method_type = "t-test"
    elif "wilcoxon" in method:
        method_type = "Wilcoxon"
    elif "nb_glm" in method or "pydeseq2" in method or "edger" in method:
        method_type = "GLM"
    elif "qc" in method:
        method_type = "QC"
    elif "avg_log" in method:
        method_type = "avg_log"
    elif "pseudobulk" in method:
        method_type = "pseudobulk"
    else:
        method_type = method
    
    return {
        "tool": tool,
        "analysis_type": analysis_type,
        "method_type": method_type,
    }


def enrich_performance_data(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Add tool and analysis type columns to performance data."""
    categorization = perf_df["method"].apply(categorize_method).apply(pd.Series)
    return pd.concat([perf_df, categorization], axis=1)


def save_aggregated_data(output_dir: Path = None):
    """Save aggregated data to CSV files."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance data
    perf_df = aggregate_performance_data()
    perf_df = enrich_performance_data(perf_df)
    perf_file = output_dir / "aggregated_performance.csv"
    perf_df.to_csv(perf_file, index=False)
    print(f"Saved: {perf_file}")
    
    # Accuracy data
    acc_df = aggregate_accuracy_data()
    if not acc_df.empty:
        acc_file = output_dir / "aggregated_accuracy.csv"
        acc_df.to_csv(acc_file, index=False)
        print(f"Saved: {acc_file}")
    
    # Dataset metadata
    meta_df = get_dataset_metadata()
    meta_file = output_dir / "dataset_metadata.csv"
    meta_df.to_csv(meta_file, index=False)
    print(f"Saved: {meta_file}")
    
    return perf_df, acc_df, meta_df


if __name__ == "__main__":
    perf_df, acc_df, meta_df = save_aggregated_data()
    
    print("\n=== Dataset Metadata ===")
    print(meta_df.to_string(index=False))
    
    print("\n=== Methods per Dataset ===")
    print(perf_df.groupby(["dataset", "status"]).size().unstack(fill_value=0))
