#!/usr/bin/env python3
"""
Generate individual benchmark figures for scientific publication.

Creates separate figures for each panel:
A) Runtime comparison (speed)
B) Memory usage for DE methods (efficiency) - split by method type
C) Scalability analysis with annotations (scaling behavior)
D) Method success rate heatmap (robustness)
E) Accuracy validation scatter plots (no accuracy loss)
F) Speedup summary (quantified benefits)

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

# Full dataset names for display
DATASET_DISPLAY = {
    "Tian-crispra": "Tian CRISPRa",
    "Tian-crispri": "Tian CRISPRi",
    "Adamson": "Adamson",
    "Nadig-HEPG2": "Nadig HepG2",
    "Frangieh": "Frangieh",
    "Nadig-JURKAT": "Nadig Jurkat",
    "Replogle-E-k562": "Replogle Essential K562",
    "Replogle-E-rpe1": "Replogle Essential RPE1",
    "Feng-gwsf": "Feng GW-SF",
}


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
    complete_datasets = []
    for dataset in perf_df["dataset"].unique():
        dataset_df = perf_df[perf_df["dataset"] == dataset]
        if (dataset_df["status"] == "success").sum() >= 3:
            complete_datasets.append(dataset)
    return perf_df[perf_df["dataset"].isin(complete_datasets)]


def save_figure(fig, output_dir: Path, name: str):
    """Save figure in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / name}.pdf, .png")
    plt.close(fig)


# =============================================================================
# Figure A: Runtime Comparison (3 subplots by test type)
# =============================================================================

def generate_figure_a(perf_df: pd.DataFrame, output_dir: Path):
    """
    Figure A: Runtime comparison bar chart for DE methods.
    Split into 3 subplots: t-test, Wilcoxon, GLM.
    Colors distinguish tools (crispyx vs Scanpy vs others).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    datasets = [d for d in DATASET_ORDER if d in perf_df["dataset"].unique()]
    max_time = 21600
    
    # Define method groups
    method_groups = [
        ("A1: t-test", ["crispyx_de_t_test", "scanpy_de_t_test"]),
        ("A2: Wilcoxon", ["crispyx_de_wilcoxon", "scanpy_de_wilcoxon"]),
        ("A3: GLM", ["crispyx_de_nb_glm", "pertpy_de_pydeseq2", "edger_de_glm"]),
    ]
    
    for ax_idx, (title, methods) in enumerate(method_groups):
        ax = axes[ax_idx]
        df = perf_df[perf_df["method"].isin(methods)].copy()
        
        x = np.arange(len(datasets))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
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
            
            offset = (i - len(methods)/2 + 0.5) * width
            color = get_tool_color(method)
            
            bars = ax.bar(x + offset, runtimes, width, 
                         label=get_display_name(method),
                         color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            
            # Add symbols for failed methods
            for j, (bar, status) in enumerate(zip(bars, statuses)):
                bar_x = x[j] + offset
                if status == "timeout":
                    # Show ">" above the bar to indicate exceeded time limit
                    ax.text(bar_x, max_time * 1.2, ">", ha="center", va="bottom",
                           fontsize=10, fontweight="bold", color=color)
                elif status in ("error", "memory_limit"):
                    # Show "×" marker at baseline for error/OOM
                    ax.scatter([bar_x], [1], marker="x", s=60, color=color, 
                              zorder=5, linewidths=2)
        
        ax.set_yscale("log")
        ax.set_ylabel("Runtime (seconds)" if ax_idx == 0 else "", fontsize=10)
        ax.set_xlabel("Dataset", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets], 
                          rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        
        # Reference lines
        ax.axhline(y=60, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(y=3600, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "figure_a_runtime")


# =============================================================================
# Figure B: Memory Usage for DE Methods (3 subfigures by test type)
# =============================================================================

def generate_figure_b(perf_df: pd.DataFrame, output_dir: Path):
    """
    Figure B: Memory usage comparison for DE methods.
    Split into 3 subfigures: t-test, Wilcoxon, GLM.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    datasets = [d for d in DATASET_ORDER if d in perf_df["dataset"].unique()]
    
    # Define method groups
    method_groups = [
        ("B1: t-test", ["crispyx_de_t_test", "scanpy_de_t_test"]),
        ("B2: Wilcoxon", ["crispyx_de_wilcoxon", "scanpy_de_wilcoxon"]),
        ("B3: GLM", ["crispyx_de_nb_glm", "pertpy_de_pydeseq2", "edger_de_glm"]),
    ]
    
    for ax_idx, (title, methods) in enumerate(method_groups):
        ax = axes[ax_idx]
        df = perf_df[perf_df["method"].isin(methods)].copy()
        
        x = np.arange(len(datasets))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            method_df = df[df["method"] == method]
            memory = []
            statuses = []
            
            for dataset in datasets:
                row = method_df[method_df["dataset"] == dataset]
                if row.empty:
                    memory.append(0)
                    statuses.append("missing")
                else:
                    status = row["status"].iloc[0]
                    statuses.append(status)
                    # Only show memory for successful methods
                    if status == "success" and not pd.isna(row["peak_memory_mb"].iloc[0]):
                        memory.append(row["peak_memory_mb"].iloc[0] / 1024)  # GB
                    else:
                        memory.append(0)
            
            offset = (i - len(methods)/2 + 0.5) * width
            color = get_tool_color(method)
            
            bars = ax.bar(x + offset, memory, width,
                         label=get_display_name(method),
                         color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            
            # Add symbols for failed methods
            for j, (bar, status) in enumerate(zip(bars, statuses)):
                bar_x = x[j] + offset
                if status == "timeout":
                    # Show ">" symbol for timeout
                    ax.text(bar_x, 0.3, ">", ha="center", va="bottom",
                           fontsize=12, fontweight="bold", color=color)
                elif status in ("error", "memory_limit"):
                    # Show "×" marker for error/OOM
                    ax.text(bar_x, 0.3, "×", ha="center", va="bottom",
                           fontsize=12, fontweight="bold", color=color)
        
        ax.set_ylabel("Peak Memory (GB)" if ax_idx == 0 else "", fontsize=10)
        ax.set_xlabel("Dataset", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets], 
                          rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "figure_b_memory")


# =============================================================================
# Figure C: Scalability Analysis with Annotations
# =============================================================================

def generate_figure_c(perf_df: pd.DataFrame, meta_df: pd.DataFrame, output_dir: Path):
    """
    Figure C: Scalability - runtime vs dataset size with dataset labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    df = perf_df.merge(meta_df, on="dataset")
    df["dataset_size"] = df["cells"] * df["perturbations"]
    df = df.dropna(subset=["dataset_size", "elapsed_seconds"])
    df = df[df["status"] == "success"]
    
    methods = ["crispyx_de_wilcoxon", "scanpy_de_wilcoxon"]
    
    for method in methods:
        method_df = df[df["method"] == method].sort_values("dataset_size")
        if method_df.empty:
            continue
        
        color = get_tool_color(method)
        
        # Scatter points
        ax.scatter(method_df["dataset_size"], method_df["elapsed_seconds"],
                  label=get_display_name(method),
                  color=color, s=80, alpha=0.9, edgecolors="white", zorder=3)
        
        # Add dataset labels
        for _, row in method_df.iterrows():
            display_name = DATASET_DISPLAY.get(row["dataset"], row["dataset"])
            # Offset slightly to avoid overlap
            offset_x = 1.15 if method == "crispyx_de_wilcoxon" else 0.87
            ax.annotate(display_name, 
                       (row["dataset_size"] * offset_x, row["elapsed_seconds"]),
                       fontsize=6, alpha=0.8,
                       color=color,
                       ha="left" if method == "crispyx_de_wilcoxon" else "right")
        
        # Fit line
        if len(method_df) >= 2:
            z = np.polyfit(np.log10(method_df["dataset_size"]), 
                          np.log10(method_df["elapsed_seconds"]), 1)
            p = np.poly1d(z)
            x_fit = np.linspace(method_df["dataset_size"].min(), 
                               method_df["dataset_size"].max(), 100)
            ax.plot(x_fit, 10**p(np.log10(x_fit)), 
                   color=color, alpha=0.5, linestyle="--", linewidth=2)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (cells × perturbations)", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title("C) Scalability: Wilcoxon Test", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    save_figure(fig, output_dir, "figure_c_scalability")


# =============================================================================
# Figure D: Method Success Rate Heatmap with White Borders
# =============================================================================

def generate_figure_d(perf_df: pd.DataFrame, output_dir: Path):
    """
    Figure D: Heatmap showing method success/failure with white cell borders.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_methods = [
        "crispyx_qc_filtered", "scanpy_qc_filtered",
        "crispyx_de_t_test", "scanpy_de_t_test",
        "crispyx_de_wilcoxon", "scanpy_de_wilcoxon",
        "crispyx_de_nb_glm", "pertpy_de_pydeseq2", "edger_de_glm",
    ]
    
    datasets = [d for d in DATASET_ORDER if d in perf_df["dataset"].unique()]
    
    status_map = {"success": 1, "error": -1, "timeout": -0.5, "memory_limit": -0.75}
    matrix = np.zeros((len(all_methods), len(datasets)))
    
    for i, method in enumerate(all_methods):
        for j, dataset in enumerate(datasets):
            row = perf_df[(perf_df["method"] == method) & (perf_df["dataset"] == dataset)]
            if row.empty:
                matrix[i, j] = 0
            else:
                matrix[i, j] = status_map.get(row["status"].iloc[0], 0)
    
    # Custom colormap
    colors = [STATUS_COLORS["error"], STATUS_COLORS["timeout"], 
              "#EEEEEE", STATUS_COLORS["success"]]
    cmap = LinearSegmentedColormap.from_list("status", colors)
    
    # Use seaborn heatmap for better control
    dataset_labels = [DATASET_DISPLAY.get(d, d) for d in datasets]
    sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=-1, vmax=1,
                linewidths=2, linecolor="white",
                xticklabels=dataset_labels,
                yticklabels=[get_display_name(m) for m in all_methods],
                cbar=False)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_title("D) Method Success Rate", fontsize=12, fontweight="bold")
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=STATUS_COLORS["success"], edgecolor="white", label="Success"),
        mpatches.Patch(facecolor=STATUS_COLORS["timeout"], edgecolor="white", label="Timeout"),
        mpatches.Patch(facecolor=STATUS_COLORS["error"], edgecolor="white", label="Error/OOM"),
        mpatches.Patch(facecolor="#EEEEEE", edgecolor="gray", label="Not run"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
             bbox_to_anchor=(1.15, 1.0), framealpha=0.9)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "figure_d_success_heatmap")


# =============================================================================
# Figure E: Accuracy Validation - Effect Size Scatter Plot
# =============================================================================

def generate_figure_e(perf_df: pd.DataFrame, output_dir: Path):
    """
    Figure E: Scatter plot comparing effect sizes between crispyx and Scanpy.
    Focus on a single representative dataset (Replogle Essential K562 - largest with success).
    Shows t-test and Wilcoxon side by side.
    """
    import anndata as ad
    
    # Use Replogle-E-k562 as the representative dataset (largest with complete results)
    target_dataset = "Replogle-E-k562"
    results_dir = Path(__file__).parent.parent / "results" / target_dataset / "de"
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    method_pairs = [
        ("t-test", "crispyx_de_t_test.h5ad", "scanpy_de_t_test.csv"),
        ("Wilcoxon", "crispyx_de_wilcoxon.h5ad", "scanpy_de_wilcoxon.csv"),
    ]
    
    for ax_idx, (test_name, crispyx_file, scanpy_file) in enumerate(method_pairs):
        ax = axes[ax_idx]
        
        crispyx_path = results_dir / crispyx_file
        scanpy_path = results_dir / scanpy_file
        
        if not crispyx_path.exists() or not scanpy_path.exists():
            ax.text(0.5, 0.5, f"Data not available\n{crispyx_path.name}", 
                   ha="center", va="center", fontsize=10)
            ax.set_title(f"E{ax_idx+1}: {test_name}", fontsize=11, fontweight="bold")
            continue
        
        try:
            # Load crispyx results
            crispyx_adata = ad.read_h5ad(crispyx_path)
            
            # Load scanpy results
            scanpy_df = pd.read_csv(scanpy_path)
            
            # Get effect sizes (log fold changes) for comparison
            # Sample a subset of perturbations for clarity
            perturbations = list(crispyx_adata.uns.get("rank_genes_groups", {}).get("names", {}).dtype.names or [])[:10]
            
            if not perturbations:
                # Try alternative structure
                ax.text(0.5, 0.5, "Could not extract perturbations", 
                       ha="center", va="center", fontsize=10)
                continue
            
            # Collect effect sizes
            crispyx_effects = []
            scanpy_effects = []
            
            for pert in perturbations:
                # Get crispyx logfoldchanges
                try:
                    crispyx_lfc = crispyx_adata.uns["rank_genes_groups"]["logfoldchanges"][pert]
                    genes = crispyx_adata.uns["rank_genes_groups"]["names"][pert]
                    
                    # Get scanpy logfoldchanges for same perturbation
                    scanpy_pert = scanpy_df[scanpy_df["group"] == pert] if "group" in scanpy_df.columns else None
                    
                    if scanpy_pert is not None and not scanpy_pert.empty:
                        # Match by gene names
                        for g, lfc in zip(genes[:100], crispyx_lfc[:100]):  # Top 100 genes
                            scanpy_row = scanpy_pert[scanpy_pert["names"] == g]
                            if not scanpy_row.empty:
                                crispyx_effects.append(lfc)
                                scanpy_effects.append(scanpy_row["logfoldchanges"].iloc[0])
                except Exception:
                    continue
            
            if len(crispyx_effects) < 10:
                ax.text(0.5, 0.5, "Insufficient data for comparison", 
                       ha="center", va="center", fontsize=10)
                continue
            
            crispyx_effects = np.array(crispyx_effects)
            scanpy_effects = np.array(scanpy_effects)
            
            # Create scatter plot with density coloring
            ax.scatter(scanpy_effects, crispyx_effects, 
                      alpha=0.3, s=10, c=TOOL_COLORS["crispyx"], edgecolors="none")
            
            # Add diagonal line
            lims = [min(crispyx_effects.min(), scanpy_effects.min()),
                   max(crispyx_effects.max(), scanpy_effects.max())]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label="y = x")
            
            # Calculate correlation
            pearson_r = np.corrcoef(crispyx_effects, scanpy_effects)[0, 1]
            
            # Add correlation annotation
            ax.text(0.05, 0.95, f"Pearson r = {pearson_r:.6f}", 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment="top", fontweight="bold",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            ax.set_xlabel("Scanpy log fold-change", fontsize=10)
            ax.set_ylabel("crispyx log fold-change", fontsize=10)
            ax.set_title(f"E{ax_idx+1}: {test_name} Effect Sizes", fontsize=11, fontweight="bold")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3, linestyle=":")
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading data:\n{str(e)[:50]}", 
                   ha="center", va="center", fontsize=9)
            ax.set_title(f"E{ax_idx+1}: {test_name}", fontsize=11, fontweight="bold")
    
    fig.suptitle(f"Effect Size Comparison: {DATASET_DISPLAY.get(target_dataset, target_dataset)}", 
                fontsize=12, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "figure_e_accuracy")


# =============================================================================
# Figure F: Speedup Summary
# =============================================================================

def generate_figure_f(perf_df: pd.DataFrame, output_dir: Path):
    """
    Figure F: Speedup factor of crispyx vs Scanpy/others.
    Shows failure indicators when methods fail.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    comparisons = [
        ("crispyx_de_t_test", "scanpy_de_t_test", "t-test"),
        ("crispyx_de_wilcoxon", "scanpy_de_wilcoxon", "Wilcoxon"),
    ]
    
    datasets = [d for d in DATASET_ORDER if d in perf_df["dataset"].unique()]
    
    # Collect speedup data with status tracking
    speedup_data = []
    for crispyx_method, other_method, label in comparisons:
        for dataset in datasets:
            crispyx_row = perf_df[(perf_df["method"] == crispyx_method) & 
                                  (perf_df["dataset"] == dataset)]
            other_row = perf_df[(perf_df["method"] == other_method) & 
                                (perf_df["dataset"] == dataset)]
            
            crispyx_status = crispyx_row["status"].iloc[0] if not crispyx_row.empty else "missing"
            other_status = other_row["status"].iloc[0] if not other_row.empty else "missing"
            
            if crispyx_status == "success" and other_status == "success":
                crispyx_time = crispyx_row["elapsed_seconds"].iloc[0]
                other_time = other_row["elapsed_seconds"].iloc[0]
                speedup = other_time / crispyx_time
                status = "success"
            else:
                speedup = 0
                # Determine which method failed
                if crispyx_status != "success":
                    status = crispyx_status
                else:
                    status = other_status
            
            speedup_data.append({
                "dataset": dataset,
                "comparison": label,
                "speedup": speedup,
                "status": status,
            })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    comparisons_list = ["t-test", "Wilcoxon"]
    x = np.arange(len(datasets))
    width = 0.35
    
    colors = [WONG_COLORS["blue"], WONG_COLORS["sky_blue"]]
    
    for i, (comp, color) in enumerate(zip(comparisons_list, colors)):
        comp_df = speedup_df[speedup_df["comparison"] == comp]
        speedups = []
        statuses = []
        
        for d in datasets:
            row = comp_df[comp_df["dataset"] == d]
            if row.empty:
                speedups.append(0)
                statuses.append("missing")
            else:
                speedups.append(row["speedup"].iloc[0])
                statuses.append(row["status"].iloc[0])
        
        offset = (i - len(comparisons_list)/2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f"{comp}", 
                     color=color, alpha=0.85, edgecolor="white")
        
        # Add speedup labels and failure indicators
        for j, (bar, val, status) in enumerate(zip(bars, speedups, statuses)):
            bar_x = x[j] + offset
            if status == "success" and val > 1.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f"{val:.1f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")
            elif status == "timeout":
                ax.text(bar_x, 0.1, ">", ha="center", va="bottom",
                       fontsize=12, fontweight="bold", color=color)
            elif status in ("error", "memory_limit"):
                ax.text(bar_x, 0.1, "×", ha="center", va="bottom",
                       fontsize=12, fontweight="bold", color=color)
    
    ax.set_ylabel("Speedup (× faster)", fontsize=11)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets], rotation=45, ha="right", fontsize=8)
    ax.set_title("F) Speedup: crispyx vs Scanpy", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="No speedup")
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    save_figure(fig, output_dir, "figure_f_speedup")


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all individual figures."""
    output_dir = Path(__file__).parent
    
    print("Loading data...")
    perf_df = aggregate_performance_data()
    perf_df = enrich_performance_data(perf_df)
    perf_df = filter_complete_datasets(perf_df)
    
    acc_df = aggregate_accuracy_data()
    meta_df = get_dataset_metadata()
    
    print(f"Datasets: {perf_df['dataset'].nunique()}")
    print(f"Methods: {perf_df['method'].nunique()}")
    print()
    
    print("Generating Figure A: Runtime comparison...")
    generate_figure_a(perf_df, output_dir)
    
    print("Generating Figure B: Memory comparison (3 panels)...")
    generate_figure_b(perf_df, output_dir)
    
    print("Generating Figure C: Scalability with annotations...")
    generate_figure_c(perf_df, meta_df, output_dir)
    
    print("Generating Figure D: Success heatmap...")
    generate_figure_d(perf_df, output_dir)
    
    print("Generating Figure E: Accuracy scatter plots...")
    generate_figure_e(perf_df, output_dir)
    
    print("Generating Figure F: Speedup summary...")
    generate_figure_f(perf_df, output_dir)
    
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
