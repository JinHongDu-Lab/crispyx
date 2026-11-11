#!/usr/bin/env python3
"""
Generate benchmark figure for scientific publication.

Creates a 3x2 figure demonstrating crispyx benefits:
A) Runtime comparison (speed)
B) Memory usage comparison (efficiency)
C) Scalability analysis (scaling behavior)
D) Method success rate (robustness)
E) Accuracy validation (no accuracy loss)
F) NB-GLM comparison (advanced DE)

Uses Wong colorblind-safe palette from Nature Methods (Wong, 2011).
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from aggregate_results import (
    aggregate_performance_data,
    aggregate_accuracy_data,
    get_dataset_metadata,
    enrich_performance_data,
)


# =============================================================================
# Color Scheme: Wong colorblind-safe palette (Nature Methods, 2011)
# =============================================================================
WONG_COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Tool colors using Wong palette
TOOL_COLORS = {
    "crispyx": WONG_COLORS["blue"],       # Blue - primary tool
    "Scanpy": WONG_COLORS["orange"],      # Orange - main comparison
    "edgeR": WONG_COLORS["bluish_green"], # Green
    "Pertpy": WONG_COLORS["reddish_purple"],  # Purple
}

# Status colors
STATUS_COLORS = {
    "success": WONG_COLORS["bluish_green"],
    "error": WONG_COLORS["vermillion"],
    "timeout": WONG_COLORS["orange"],
    "memory_limit": WONG_COLORS["reddish_purple"],
}

# Method display names
METHOD_DISPLAY = {
    "crispyx_qc_filtered": "crispyx QC",
    "crispyx_de_t_test": "crispyx t-test",
    "crispyx_de_wilcoxon": "crispyx Wilcoxon",
    "crispyx_de_nb_glm": "crispyx NB-GLM",
    "crispyx_pb_avg_log": "crispyx avg-log",
    "crispyx_pb_pseudobulk": "crispyx pseudobulk",
    "scanpy_qc_filtered": "Scanpy QC",
    "scanpy_de_t_test": "Scanpy t-test",
    "scanpy_de_wilcoxon": "Scanpy Wilcoxon",
    "edger_de_glm": "edgeR GLM",
    "pertpy_de_pydeseq2": "PyDESeq2",
}

# Dataset display order (by size)
DATASET_ORDER = [
    "Tian-crispra",
    "Tian-crispri", 
    "Adamson",
    "Nadig-HEPG2",
    "Frangieh",
    "Nadig-JURKAT",
    "Replogle-E-k562",
    "Replogle-E-rpe1",
    "Feng-gwsf",
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_display_name(method: str) -> str:
    """Get display name for a method."""
    return METHOD_DISPLAY.get(method, method)


def get_tool_color(method: str) -> str:
    """Get color for a method based on its tool."""
    if method.startswith("crispyx"):
        return TOOL_COLORS["crispyx"]
    elif method.startswith("scanpy"):
        return TOOL_COLORS["Scanpy"]
    elif method.startswith("edger"):
        return TOOL_COLORS["edgeR"]
    elif method.startswith("pertpy"):
        return TOOL_COLORS["Pertpy"]
    return WONG_COLORS["black"]


def filter_complete_datasets(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to datasets with complete benchmark results."""
    # Exclude datasets without performance CSVs
    complete_datasets = []
    for dataset in perf_df["dataset"].unique():
        dataset_df = perf_df[perf_df["dataset"] == dataset]
        # Must have at least some successful methods
        if (dataset_df["status"] == "success").sum() >= 3:
            complete_datasets.append(dataset)
    
    return perf_df[perf_df["dataset"].isin(complete_datasets)]


# =============================================================================
# Subfigure A: Runtime Comparison
# =============================================================================

def plot_runtime_comparison(ax, perf_df: pd.DataFrame):
    """
    Panel A: Runtime comparison bar chart.
    Shows all DE methods across datasets.
    """
    # Filter to DE methods
    de_methods = [
        "crispyx_de_t_test", "scanpy_de_t_test",
        "crispyx_de_wilcoxon", "scanpy_de_wilcoxon",
        "crispyx_de_nb_glm", "pertpy_de_pydeseq2", "edger_de_glm",
    ]
    
    df = perf_df[perf_df["method"].isin(de_methods)].copy()
    
    # Get datasets in order
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    
    # Prepare data for grouped bar chart
    x = np.arange(len(datasets))
    width = 0.12
    n_methods = len(de_methods)
    
    # Max time for timeout display
    max_time = 21600  # 6 hours
    
    for i, method in enumerate(de_methods):
        method_df = df[df["method"] == method]
        runtimes = []
        statuses = []
        
        for dataset in datasets:
            row = method_df[method_df["dataset"] == dataset]
            if row.empty:
                runtimes.append(0)
                statuses.append("missing")
            else:
                status = row["status"].iloc[0]
                statuses.append(status)
                if status == "success":
                    runtimes.append(row["elapsed_seconds"].iloc[0])
                elif status == "timeout":
                    runtimes.append(max_time)
                else:
                    runtimes.append(0)
        
        offset = (i - n_methods/2 + 0.5) * width
        color = get_tool_color(method)
        
        # Create bars
        bars = ax.bar(x + offset, runtimes, width, 
                     label=get_display_name(method),
                     color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        
        # Add hatching for non-success
        for j, (bar, status) in enumerate(zip(bars, statuses)):
            if status == "timeout":
                bar.set_hatch("//")
                bar.set_edgecolor("black")
            elif status in ("error", "memory_limit"):
                bar.set_hatch("xx")
                bar.set_alpha(0.3)
    
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_title("A) Runtime Comparison (DE Methods)", fontsize=11, fontweight="bold")
    
    # Legend
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    
    # Add reference lines
    ax.axhline(y=60, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.axhline(y=3600, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.text(len(datasets)-0.5, 60, "1 min", va="bottom", ha="right", fontsize=7, color="gray")
    ax.text(len(datasets)-0.5, 3600, "1 hr", va="bottom", ha="right", fontsize=7, color="gray")


# =============================================================================
# Subfigure B: Memory Usage Comparison
# =============================================================================

def plot_memory_comparison(ax, perf_df: pd.DataFrame):
    """
    Panel B: Peak memory usage comparison.
    Shows all methods across datasets.
    """
    # Methods to compare
    methods = [
        "crispyx_qc_filtered", "scanpy_qc_filtered",
        "crispyx_de_t_test", "scanpy_de_t_test",
        "crispyx_de_wilcoxon", "scanpy_de_wilcoxon",
    ]
    
    df = perf_df[perf_df["method"].isin(methods)].copy()
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    
    x = np.arange(len(datasets))
    width = 0.13
    n_methods = len(methods)
    
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        memory = []
        
        for dataset in datasets:
            row = method_df[method_df["dataset"] == dataset]
            if row.empty or pd.isna(row["peak_memory_mb"].iloc[0]):
                memory.append(0)
            else:
                memory.append(row["peak_memory_mb"].iloc[0] / 1024)  # Convert to GB
        
        offset = (i - n_methods/2 + 0.5) * width
        color = get_tool_color(method)
        
        ax.bar(x + offset, memory, width,
               label=get_display_name(method),
               color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    
    ax.set_ylabel("Peak Memory (GB)", fontsize=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_title("B) Memory Usage Comparison", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)


# =============================================================================
# Subfigure C: Scalability Analysis
# =============================================================================

def plot_scalability(ax, perf_df: pd.DataFrame, meta_df: pd.DataFrame):
    """
    Panel C: Scalability - runtime vs dataset size.
    """
    # Merge metadata
    df = perf_df.merge(meta_df, on="dataset")
    df["dataset_size"] = df["cells"] * df["perturbations"]
    df = df.dropna(subset=["dataset_size", "elapsed_seconds"])
    df = df[df["status"] == "success"]
    
    # Methods to plot
    methods = ["crispyx_de_wilcoxon", "scanpy_de_wilcoxon"]
    
    for method in methods:
        method_df = df[df["method"] == method].sort_values("dataset_size")
        if method_df.empty:
            continue
        
        ax.scatter(method_df["dataset_size"], method_df["elapsed_seconds"],
                  label=get_display_name(method),
                  color=get_tool_color(method), s=60, alpha=0.8, edgecolors="white")
        
        # Fit line
        if len(method_df) >= 2:
            z = np.polyfit(np.log10(method_df["dataset_size"]), 
                          np.log10(method_df["elapsed_seconds"]), 1)
            p = np.poly1d(z)
            x_fit = np.linspace(method_df["dataset_size"].min(), 
                               method_df["dataset_size"].max(), 100)
            ax.plot(x_fit, 10**p(np.log10(x_fit)), 
                   color=get_tool_color(method), alpha=0.5, linestyle="--", linewidth=1.5)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (cells × perturbations)", fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_title("C) Scalability: Wilcoxon Test", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


# =============================================================================
# Subfigure D: Method Success Rate Heatmap
# =============================================================================

def plot_success_heatmap(ax, perf_df: pd.DataFrame):
    """
    Panel D: Heatmap showing method success/failure across datasets.
    """
    # All methods
    all_methods = [
        "crispyx_qc_filtered", "scanpy_qc_filtered",
        "crispyx_de_t_test", "scanpy_de_t_test",
        "crispyx_de_wilcoxon", "scanpy_de_wilcoxon",
        "crispyx_de_nb_glm", "pertpy_de_pydeseq2", "edger_de_glm",
    ]
    
    datasets = [d for d in DATASET_ORDER if d in perf_df["dataset"].unique()]
    
    # Create status matrix
    status_map = {"success": 1, "error": -1, "timeout": -0.5, "memory_limit": -0.75}
    matrix = np.zeros((len(all_methods), len(datasets)))
    
    for i, method in enumerate(all_methods):
        for j, dataset in enumerate(datasets):
            row = perf_df[(perf_df["method"] == method) & (perf_df["dataset"] == dataset)]
            if row.empty:
                matrix[i, j] = 0  # missing
            else:
                matrix[i, j] = status_map.get(row["status"].iloc[0], 0)
    
    # Custom colormap: red (fail) -> yellow (timeout) -> green (success)
    colors = [STATUS_COLORS["error"], STATUS_COLORS["timeout"], 
              "#EEEEEE", STATUS_COLORS["success"]]
    cmap = LinearSegmentedColormap.from_list("status", colors)
    
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(all_methods)))
    ax.set_yticklabels([get_display_name(m) for m in all_methods], fontsize=8)
    ax.set_title("D) Method Success Rate", fontsize=11, fontweight="bold")
    
    # Legend patches
    legend_elements = [
        mpatches.Patch(facecolor=STATUS_COLORS["success"], label="Success"),
        mpatches.Patch(facecolor=STATUS_COLORS["timeout"], label="Timeout"),
        mpatches.Patch(facecolor=STATUS_COLORS["error"], label="Error/OOM"),
        mpatches.Patch(facecolor="#EEEEEE", edgecolor="gray", label="Not run"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, 
             bbox_to_anchor=(1.0, 1.0), framealpha=0.9)


# =============================================================================
# Subfigure E: Accuracy Validation
# =============================================================================

def plot_accuracy_validation(ax, acc_df: pd.DataFrame):
    """
    Panel E: Correlation between crispyx and Scanpy results.
    """
    # Filter to crispyx vs scanpy comparisons
    de_comparisons = acc_df[acc_df["comp_type"] == "de"].copy()
    de_comparisons = de_comparisons[
        de_comparisons["comparison"].str.contains("crispyx") & 
        de_comparisons["comparison"].str.contains("scanpy")
    ]
    
    if de_comparisons.empty:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center", fontsize=12)
        ax.set_title("E) Accuracy Validation", fontsize=11, fontweight="bold")
        return
    
    # Extract method pairs
    de_comparisons["method_pair"] = de_comparisons["comparison"].apply(
        lambda x: "t-test" if "t_test" in x else "Wilcoxon" if "wilcoxon" in x else "NB-GLM"
    )
    
    # Group by method pair and dataset
    summary = de_comparisons.groupby(["method_pair", "dataset"]).agg({
        "effect_pearson_corr_mean": "first",
        "effect_spearman_corr_mean": "first",
    }).reset_index()
    
    # Create box plot
    method_pairs = ["t-test", "Wilcoxon"]
    x = np.arange(len(method_pairs))
    width = 0.35
    
    pearson_vals = [summary[summary["method_pair"] == mp]["effect_pearson_corr_mean"].values 
                   for mp in method_pairs]
    spearman_vals = [summary[summary["method_pair"] == mp]["effect_spearman_corr_mean"].values 
                    for mp in method_pairs]
    
    bp1 = ax.boxplot(pearson_vals, positions=x - width/2, widths=width*0.8,
                     patch_artist=True, boxprops=dict(facecolor=WONG_COLORS["sky_blue"]))
    bp2 = ax.boxplot(spearman_vals, positions=x + width/2, widths=width*0.8,
                     patch_artist=True, boxprops=dict(facecolor=WONG_COLORS["yellow"]))
    
    ax.set_xticks(x)
    ax.set_xticklabels(method_pairs, fontsize=10)
    ax.set_ylabel("Correlation (crispyx vs Scanpy)", fontsize=10)
    ax.set_ylim(0.999, 1.0001)
    ax.set_title("E) Accuracy Validation", fontsize=11, fontweight="bold")
    
    # Reference line
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.999, color="gray", linestyle=":", alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=WONG_COLORS["sky_blue"], label="Pearson r"),
        mpatches.Patch(facecolor=WONG_COLORS["yellow"], label="Spearman ρ"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)


# =============================================================================
# Subfigure F: Speedup Summary
# =============================================================================

def plot_speedup_summary(ax, perf_df: pd.DataFrame):
    """
    Panel F: Speedup factor of crispyx vs Scanpy/others.
    """
    # Calculate speedup for each dataset and method pair
    speedup_data = []
    
    comparisons = [
        ("crispyx_de_t_test", "scanpy_de_t_test", "t-test"),
        ("crispyx_de_wilcoxon", "scanpy_de_wilcoxon", "Wilcoxon"),
        ("crispyx_qc_filtered", "scanpy_qc_filtered", "QC"),
    ]
    
    datasets = [d for d in DATASET_ORDER if d in perf_df["dataset"].unique()]
    
    for crispyx_method, other_method, label in comparisons:
        for dataset in datasets:
            crispyx_row = perf_df[(perf_df["method"] == crispyx_method) & 
                                  (perf_df["dataset"] == dataset) &
                                  (perf_df["status"] == "success")]
            other_row = perf_df[(perf_df["method"] == other_method) & 
                                (perf_df["dataset"] == dataset) &
                                (perf_df["status"] == "success")]
            
            if not crispyx_row.empty and not other_row.empty:
                crispyx_time = crispyx_row["elapsed_seconds"].iloc[0]
                other_time = other_row["elapsed_seconds"].iloc[0]
                speedup = other_time / crispyx_time
                
                speedup_data.append({
                    "dataset": dataset,
                    "comparison": label,
                    "speedup": speedup,
                })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    if speedup_df.empty:
        ax.text(0.5, 0.5, "No speedup data", ha="center", va="center", fontsize=12)
        ax.set_title("F) Speedup Summary", fontsize=11, fontweight="bold")
        return
    
    # Create grouped bar chart
    comparisons_list = ["t-test", "Wilcoxon", "QC"]
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = [WONG_COLORS["blue"], WONG_COLORS["sky_blue"], WONG_COLORS["bluish_green"]]
    
    for i, (comp, color) in enumerate(zip(comparisons_list, colors)):
        comp_df = speedup_df[speedup_df["comparison"] == comp]
        speedups = [comp_df[comp_df["dataset"] == d]["speedup"].iloc[0] 
                   if not comp_df[comp_df["dataset"] == d].empty else 0
                   for d in datasets]
        
        offset = (i - len(comparisons_list)/2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f"{comp}", color=color, alpha=0.85)
        
        # Add value labels for significant speedups
        for j, (bar, val) in enumerate(zip(bars, speedups)):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{val:.0f}×", ha="center", va="bottom", fontsize=6, rotation=90)
    
    ax.set_ylabel("Speedup (× faster)", fontsize=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_title("F) Speedup: crispyx vs Scanpy", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(0, None)


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_benchmark_figure(output_dir: Path = None):
    """Generate the complete benchmark figure."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    print("Loading data...")
    perf_df = aggregate_performance_data()
    perf_df = enrich_performance_data(perf_df)
    perf_df = filter_complete_datasets(perf_df)
    
    acc_df = aggregate_accuracy_data()
    meta_df = get_dataset_metadata()
    
    print(f"Datasets: {perf_df['dataset'].nunique()}")
    print(f"Methods: {perf_df['method'].nunique()}")
    
    # Create figure with 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Benchmark Results: crispyx vs Reference Methods", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    # Generate each panel
    print("Generating Panel A: Runtime comparison...")
    plot_runtime_comparison(axes[0, 0], perf_df)
    
    print("Generating Panel B: Memory comparison...")
    plot_memory_comparison(axes[0, 1], perf_df)
    
    print("Generating Panel C: Scalability...")
    plot_scalability(axes[1, 0], perf_df, meta_df)
    
    print("Generating Panel D: Success heatmap...")
    plot_success_heatmap(axes[1, 1], perf_df)
    
    print("Generating Panel E: Accuracy validation...")
    plot_accuracy_validation(axes[2, 0], acc_df)
    
    print("Generating Panel F: Speedup summary...")
    plot_speedup_summary(axes[2, 1], perf_df)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / "benchmark_figure.pdf"
    png_path = output_dir / "benchmark_figure.png"
    svg_path = output_dir / "benchmark_figure.svg"
    
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    
    plt.close(fig)
    
    return fig


if __name__ == "__main__":
    generate_benchmark_figure()
