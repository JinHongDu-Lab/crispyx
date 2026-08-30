"""Rerun Scanpy methods without resource limits.

This script runs Scanpy QC, t-test, and Wilcoxon methods without time or memory
limits, allowing them to complete even on large datasets where they would
normally timeout or OOM in the benchmark. The outputs are saved to the same
locations as normal benchmarks, and reports are automatically regenerated.

Key features:
- No time/memory limits (methods run to completion)
- Does NOT write to .benchmark_cache (preserves benchmark integrity)
- Outputs to same de/ and preprocessing/ directories
- Automatically regenerates benchmark reports with new accuracy comparisons
- Tracks extraction status via .reference_extracted marker file
- Can be run before or after normal benchmarks

Usage:
    python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml
    python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml --methods scanpy_de_t_test
    python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml --no-report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure the local package is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crispyx.data import (
    read_backed,
    resolve_control_label,
    calculate_adaptive_qc_thresholds,
    standardize_dataset,
)

from .run_benchmarks import (
    BenchmarkConfig,
    QCParams,
    run_scanpy_qc,
    run_scanpy_de,
    REPO_ROOT,
)
from .profiling import get_peak_memory_mb

from .generate_results import evaluate_benchmarks


# Default reference methods to extract
DEFAULT_REFERENCE_METHODS = [
    "scanpy_qc_filtered",
    "scanpy_de_t_test",
    "scanpy_de_wilcoxon",
]


def _get_marker_path(output_dir: Path) -> Path:
    """Get path to reference extraction marker file."""
    return output_dir / ".reference_extracted"


def _load_marker(output_dir: Path) -> Dict[str, Any]:
    """Load reference extraction marker data."""
    marker_path = _get_marker_path(output_dir)
    if marker_path.exists():
        try:
            with open(marker_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_marker(output_dir: Path, data: Dict[str, Any]) -> None:
    """Save reference extraction marker data."""
    marker_path = _get_marker_path(output_dir)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _check_output_exists(method_name: str, output_dir: Path) -> bool:
    """Check if output file for a method already exists."""
    preprocessing_dir = output_dir / "preprocessing"
    de_dir = output_dir / "de"
    
    expected_paths = {
        "scanpy_qc_filtered": preprocessing_dir / "scanpy_qc_filtered.h5ad",
        "scanpy_de_t_test": de_dir / "scanpy_de_t_test.csv",
        "scanpy_de_wilcoxon": de_dir / "scanpy_de_wilcoxon.csv",
    }
    
    expected_path = expected_paths.get(method_name)
    if expected_path is None:
        return False
    
    return expected_path.exists() and expected_path.stat().st_size > 0


def extract_scanpy_qc(
    dataset_path: Path,
    output_dir: Path,
    perturbation_column: str,
    control_label: str,
    qc_params: Dict[str, int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Extract Scanpy QC results without resource limits."""
    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"  Running scanpy_qc_filtered...")
    
    mem_before = get_peak_memory_mb()
    start_time = time.perf_counter()
    try:
        result = run_scanpy_qc(
            dataset_path=dataset_path,
            min_genes=qc_params["min_genes"],
            min_cells_per_perturbation=qc_params["min_cells_per_perturbation"],
            min_cells_per_gene=qc_params["min_cells_per_gene"],
            perturbation_column=perturbation_column,
            control_label=control_label,
            output_dir=preprocessing_dir,
        )
        elapsed = time.perf_counter() - start_time
        peak_memory_mb = get_peak_memory_mb() - mem_before
        
        if verbose:
            print(f"    ✓ Completed in {elapsed:.1f}s (peak mem delta: {peak_memory_mb:.1f} MB)")
            print(f"      Cells: {result.get('cells_kept', 'N/A')}, Genes: {result.get('genes_kept', 'N/A')}")
        
        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "peak_memory_mb": peak_memory_mb,
            "result": result,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"    ✗ Failed after {elapsed:.1f}s: {e}")
        return {
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def extract_scanpy_de(
    dataset_path: Path,
    output_dir: Path,
    perturbation_column: str,
    control_label: str,
    method: str,
    preprocess: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Extract Scanpy DE results without resource limits."""
    de_dir = output_dir / "de"
    de_dir.mkdir(parents=True, exist_ok=True)
    
    method_name = f"scanpy_de_{method.replace('-', '_')}"
    if verbose:
        print(f"  Running {method_name}...")
    
    mem_before = get_peak_memory_mb()
    start_time = time.perf_counter()
    try:
        result = run_scanpy_de(
            dataset_path=dataset_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            method=method,
            output_dir=de_dir,
            preprocess=preprocess,
        )
        elapsed = time.perf_counter() - start_time
        peak_memory_mb = get_peak_memory_mb() - mem_before
        
        if verbose:
            print(f"    ✓ Completed in {elapsed:.1f}s (peak mem delta: {peak_memory_mb:.1f} MB)")
            print(f"      Groups: {result.get('groups', 'N/A')}")
        
        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "peak_memory_mb": peak_memory_mb,
            "result": result,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"    ✗ Failed after {elapsed:.1f}s: {e}")
        return {
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_reference_extraction(
    config: BenchmarkConfig,
    methods: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run reference extraction for specified methods.
    
    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration (from YAML)
    methods : List[str] | None
        Methods to extract. If None, uses DEFAULT_REFERENCE_METHODS.
    force : bool
        If True, re-extract even if output already exists.
    verbose : bool
        If True, print progress information.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results for each method (status, elapsed_seconds, etc.)
    """
    methods_to_run = methods if methods is not None else DEFAULT_REFERENCE_METHODS
    results: Dict[str, Dict[str, Any]] = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Rerun Scanpy: {config.dataset_name}")
        print(f"{'='*60}")
        print(f"Dataset: {config.dataset_path}")
        print(f"Output: {config.output_dir}")
        print(f"Methods: {', '.join(methods_to_run)}")
        print()
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standardize dataset (uses cache if available)
    if verbose:
        print("Standardizing dataset...")
    
    standardized_path = standardize_dataset(
        dataset_path=config.dataset_path,
        perturbation_column=config.perturbation_column,
        control_label=config.control_label,
        gene_name_column=config.gene_name_column,
        output_dir=config.output_dir,
        force=config.force_restandardize,
    )
    
    if verbose:
        print(f"  ✓ Standardized: {standardized_path.name}")
    
    # Calculate or use provided QC params
    if config.qc_params is None:
        import anndata as ad
        if verbose:
            print(f"\nCalculating adaptive QC parameters...")
        
        adata_temp = ad.read_h5ad(standardized_path, backed='r')
        try:
            qc_params = calculate_adaptive_qc_thresholds(
                adata_temp, "perturbation", mode=config.adaptive_qc_mode,
                chunk_size=config.chunk_size
            )
        finally:
            adata_temp.file.close()
        
        if verbose:
            print(f"  ✓ min_genes: {qc_params['min_genes']}")
            print(f"  ✓ min_cells_per_perturbation: {qc_params['min_cells_per_perturbation']}")
            print(f"  ✓ min_cells_per_gene: {qc_params['min_cells_per_gene']}")
    else:
        qc_params = {
            "min_genes": config.qc_params.min_genes,
            "min_cells_per_perturbation": config.qc_params.min_cells_per_perturbation,
            "min_cells_per_gene": config.qc_params.min_cells_per_gene,
            "chunk_size": config.chunk_size if config.chunk_size else config.qc_params.chunk_size,
        }
    
    # Resolve control label
    backed = read_backed(standardized_path)
    try:
        labels = backed.obs["perturbation"].astype(str).to_numpy().tolist()
        control_label = resolve_control_label(labels, "control", verbose=False)
    finally:
        backed.file.close()
    
    # Use the preprocessed file produced by crispyx during the main benchmark
    # (crispyx_qc_filtered -> crispyx_preprocess). It is already QC-filtered and
    # log-normalized, matching exactly what scanpy DE methods receive in the benchmark.
    preprocessing_dir = config.output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = preprocessing_dir / f"preprocessed_{standardized_path.name}"

    needs_preprocessed = any(
        m in methods_to_run for m in ("scanpy_de_t_test", "scanpy_de_wilcoxon")
    )
    if needs_preprocessed and not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Crispyx preprocessed dataset not found: {preprocessed_path}\n"
            "Run the crispyx benchmark first so that crispyx_qc_filtered and "
            "crispyx_preprocess produce this file."
        )
    
    # Load existing marker data
    marker_data = _load_marker(config.output_dir)
    extracted_methods = marker_data.get("extracted_methods", {})
    
    if verbose:
        print(f"\nExtracting reference outputs...")
    
    # Run each method
    for method_name in methods_to_run:
        # Check if already extracted
        if not force and _check_output_exists(method_name, config.output_dir):
            if verbose:
                print(f"  ⏭️  Skipping {method_name} (output exists)")
            results[method_name] = {
                "status": "skipped",
                "reason": "output_exists",
            }
            continue
        
        # Run the method
        if method_name == "scanpy_qc_filtered":
            result = extract_scanpy_qc(
                dataset_path=standardized_path,
                output_dir=config.output_dir,
                perturbation_column="perturbation",
                control_label=control_label,
                qc_params=qc_params,
                verbose=verbose,
            )
        elif method_name == "scanpy_de_t_test":
            result = extract_scanpy_de(
                dataset_path=preprocessed_path,
                output_dir=config.output_dir,
                perturbation_column="perturbation",
                control_label=control_label,
                method="t-test",
                preprocess=False,  # Already preprocessed
                verbose=verbose,
            )
        elif method_name == "scanpy_de_wilcoxon":
            result = extract_scanpy_de(
                dataset_path=preprocessed_path,
                output_dir=config.output_dir,
                perturbation_column="perturbation",
                control_label=control_label,
                method="wilcoxon",
                preprocess=False,  # preprocessed_path is already normalized
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"  ⚠️  Unknown method: {method_name}")
            result = {
                "status": "error",
                "error": f"Unknown method: {method_name}",
            }
        
        results[method_name] = result
        
        # Update marker for successful extractions
        if result.get("status") == "success":
            extracted_methods[method_name] = {
                "timestamp": time.time(),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "peak_memory_mb": result.get("peak_memory_mb"),
            }
    
    # Save marker data
    marker_data["extracted_methods"] = extracted_methods
    marker_data["last_extraction"] = time.time()
    _save_marker(config.output_dir, marker_data)
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("Extraction Summary")
        print(f"{'='*60}")
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        skip_count = sum(1 for r in results.values() if r.get("status") == "skipped")
        error_count = sum(1 for r in results.values() if r.get("status") == "error")
        print(f"  ✓ Success: {success_count}")
        if skip_count:
            print(f"  ⏭️  Skipped: {skip_count}")
        if error_count:
            print(f"  ✗ Errors: {error_count}")
        
        # Show total time
        total_time = sum(
            r.get("elapsed_seconds", 0) 
            for r in results.values() 
            if r.get("status") == "success"
        )
        print(f"\n  Total extraction time: {total_time:.1f}s")
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun Scanpy methods without resource limits and regenerate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rerun all Scanpy methods for one dataset
  python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml
  
  # Rerun only specific methods
  python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml \\
      --methods scanpy_de_t_test scanpy_de_wilcoxon
  
  # Force re-run even if outputs exist
  python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml --force
  
  # Skip report regeneration
  python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml --no-report
  
  # Quiet mode (minimal output)
  python -m benchmarking.tools.rerun_scanpy --config config/Adamson.yaml --quiet
""",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=f"Methods to extract (default: {', '.join(DEFAULT_REFERENCE_METHODS)})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if output already exists",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip regenerating benchmark reports after extraction",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config_result = BenchmarkConfig.from_yaml(args.config)
    configs = config_result if isinstance(config_result, list) else [config_result]
    
    # Process methods - handle comma-separated values
    methods = None
    if args.methods is not None:
        methods = []
        for m in args.methods:
            # Split on comma to handle "a,b,c" as separate methods
            methods.extend(m.split(","))
        methods = [m.strip() for m in methods if m.strip()]
        if not methods:
            methods = None
    
    all_results = {}
    successful_configs = []
    
    for config in configs:
        results = run_reference_extraction(
            config=config,
            methods=methods,
            force=args.force,
            verbose=not args.quiet,
        )
        all_results[config.dataset_name] = results
        
        # Track configs with at least one success for report regeneration
        if any(r.get("status") == "success" for r in results.values()):
            successful_configs.append(config)
    
    # Regenerate reports for successful extractions
    if not args.no_report and successful_configs:
        if not args.quiet:
            print("\n" + "=" * 60)
            print("Regenerating Benchmark Reports")
            print("=" * 60)
        
        for config in successful_configs:
            if not args.quiet:
                print(f"\n  Regenerating report for {config.dataset_name}...")
            try:
                evaluate_benchmarks(config.output_dir)
                if not args.quiet:
                    print(f"    ✓ Report: {config.output_dir}/benchmark_report.md")
            except Exception as e:
                print(f"    ✗ Failed to regenerate report: {e}")
    
    # Exit with error if any failures
    has_errors = any(
        r.get("status") == "error"
        for dataset_results in all_results.values()
        for r in dataset_results.values()
    )
    
    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
