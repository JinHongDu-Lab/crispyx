"""Caching utilities for benchmark results.

This module consolidates all caching functions used by the benchmarking
infrastructure to store and retrieve method results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import CACHE_VERSION


# ============================================================================
# Serialization Helpers
# ============================================================================

def is_scalar_na(value: Any) -> bool:
    """Check if a value is NA/NaN, handling arrays properly.
    
    For arrays, returns False (arrays are not scalar NA values).
    For scalars, returns True if NA/NaN/None.
    """
    # Handle None explicitly
    if value is None:
        return True
    # Handle numpy arrays - they are not scalar NA
    if isinstance(value, np.ndarray):
        return False
    # Handle pandas Series/DataFrame - not scalar NA
    if hasattr(value, '__len__') and hasattr(value, 'dtype'):
        return False
    # Try pandas isna for scalars
    try:
        result = pd.isna(value)
        # If result is a scalar bool, return it
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        # If result is array-like, this wasn't a scalar - return False
        return False
    except (TypeError, ValueError):
        return False


def make_json_serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable type.
    
    Handles numpy arrays, numpy scalars, pandas types, Path objects, etc.
    """
    # Handle None
    if value is None:
        return None
    
    # Handle numpy arrays - convert to list
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    # Handle numpy scalar types
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    
    # Handle Path objects
    if isinstance(value, Path):
        return str(value)
    
    # Handle pandas NA types
    if is_scalar_na(value):
        return None
    
    # Handle lists recursively
    if isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    
    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    
    # Return as-is for standard JSON types (str, int, float, bool)
    return value


# ============================================================================
# Path Resolution
# ============================================================================

def get_expected_output_path(method_name: str, output_dir: Path) -> Optional[Path]:
    """Get the expected output path for a benchmark method.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Path | None
        Expected output file path, or None if cannot be determined
    """
    # Phase-based directories
    preprocessing_dir = output_dir / "preprocessing"
    de_dir = output_dir / "de"
    
    # crispyx methods with module prefix
    if method_name == "crispyx_qc_filtered":
        return preprocessing_dir / "crispyx_qc_filtered.h5ad"
    elif method_name == "crispyx_pb_avg_log":
        return preprocessing_dir / "crispyx_pb_avg_log.h5ad"
    elif method_name == "crispyx_pb_pseudobulk":
        return preprocessing_dir / "crispyx_pb_pseudobulk.h5ad"
    elif method_name == "crispyx_de_t_test":
        return de_dir / "crispyx_de_t_test.h5ad"
    elif method_name == "crispyx_de_wilcoxon":
        return de_dir / "crispyx_de_wilcoxon.h5ad"
    elif method_name == "crispyx_de_nb_glm":
        return de_dir / "crispyx_de_nb_glm.h5ad"
    elif method_name == "crispyx_de_nb_glm_pydeseq2":
        return de_dir / "crispyx_de_nb_glm_pydeseq2_nb_glm.h5ad"
    elif method_name == "crispyx_de_lfcshrink":
        return de_dir / "crispyx_de_nb_glm_shrunk.h5ad"
    elif method_name == "crispyx_de_lfcshrink_pydeseq2":
        return de_dir / "crispyx_de_nb_glm_shrunk_pydeseq2.h5ad"
    
    # Scanpy methods with module prefix
    elif method_name == "scanpy_qc_filtered":
        return preprocessing_dir / "scanpy_qc_filtered.h5ad"
    elif method_name == "scanpy_de_t_test":
        return de_dir / "scanpy_de_t_test.csv"
    elif method_name == "scanpy_de_wilcoxon":
        return de_dir / "scanpy_de_wilcoxon.csv"
    
    # Reference tool CSV outputs
    elif method_name == "edger_de_glm":
        return de_dir / "edger_de_glm.csv"
    elif method_name == "pertpy_de_pydeseq2":
        return de_dir / "pertpy_de_pydeseq2.csv"
    elif method_name == "pertpy_de_lfcshrink":
        return de_dir / "pertpy_de_pydeseq2_shrunk.csv"
    
    return None


def resolve_result_path(
    method_name: str,
    result_path_val: Optional[str],
    output_dir: Path
) -> Optional[Path]:
    """Resolve the result path with fallback to expected path.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    result_path_val : Optional[str]
        Cached result_path value (may be None or NaN)
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Path]
        Resolved path to result file, or None if not found
    """
    # First try using the cached result_path
    if result_path_val is not None and not is_scalar_na(result_path_val):
        path_str = str(result_path_val)
        
        # Strip /workspace/ prefix from Docker paths (backward compatibility)
        if path_str.startswith("/workspace/"):
            path_str = path_str[len("/workspace/"):]
        
        # First try as an absolute path or path relative to workspace root
        result_path = Path(path_str)
        if result_path.exists():
            return result_path
        
        # Try as relative path from output_dir
        result_path = output_dir / path_str
        if result_path.exists():
            return result_path
        
        # Try extracting just the filename and looking in expected locations
        filename = Path(path_str).name
        for subdir in ["de", "qc", "pb", "preprocessing"]:
            potential_path = output_dir / subdir / filename
            if potential_path.exists():
                return potential_path
    
    # Fallback to expected output path
    expected_path = get_expected_output_path(method_name, output_dir)
    if expected_path is not None and expected_path.exists():
        return expected_path
    
    return None


# ============================================================================
# Cache Save/Load Functions
# ============================================================================

def save_method_result(method_name: str, row_dict: Dict[str, Any], output_dir: Path) -> None:
    """Save individual method benchmark result to cache.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    row_dict : Dict[str, Any]
        Dictionary containing benchmark results (status, runtime, memory, etc.)
    output_dir : Path
        Output directory for the dataset
    """
    cache_dir = output_dir / ".benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{method_name}.json"
    temp_file = cache_dir / f".{method_name}.json.tmp"
    
    try:
        # Convert non-serializable types to JSON-compatible formats
        serializable_dict = {}
        for key, value in row_dict.items():
            serializable_dict[key] = make_json_serializable(value)
        
        # Atomic write: write to temp file, then rename
        with temp_file.open('w') as f:
            json.dump(serializable_dict, f, indent=2, sort_keys=True)
        temp_file.rename(cache_file)
    except Exception as exc:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        # Log warning but don't fail the benchmark
        print(f"Warning: Failed to save cache for {method_name}: {exc}")


def load_method_result(method_name: str, output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load individual method benchmark result from cache.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Cached result dictionary, or None if cache doesn't exist or is corrupted
    """
    cache_file = output_dir / ".benchmark_cache" / f"{method_name}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with cache_file.open('r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Corrupted cache file for {method_name}, will re-run: {exc}")
        # Delete corrupted cache file
        try:
            cache_file.unlink()
        except Exception:
            pass
        return None


def validate_and_recover_cache_result(
    result: Dict[str, Any],
    cache_file: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Validate cache result and recover status if output file exists.
    
    If the cache shows timeout/error but the output file exists and was modified
    after the cache was written, update the status to 'recovered' and note that
    the execution actually succeeded despite the timeout.
    
    Parameters
    ----------
    result : Dict[str, Any]
        The cached result dictionary
    cache_file : Path
        Path to the cache file
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Dict[str, Any]
        Updated result dictionary (possibly with corrected status)
    """
    method_name = result.get("method")
    status = result.get("status")
    
    # Only check for recovery if the status indicates failure
    if status not in ("timeout", "error", "memory_limit"):
        return result
    
    # Need a valid method name to check for expected path
    if method_name is None:
        return result
    
    # Check if output file exists
    expected_path = get_expected_output_path(method_name, output_dir)
    if expected_path is None or not expected_path.exists():
        return result
    
    # Check if output file was modified after cache file was written
    try:
        cache_mtime = cache_file.stat().st_mtime
        output_mtime = expected_path.stat().st_mtime
        
        if output_mtime > cache_mtime:
            # Output was created after the cache entry - the method actually completed!
            print(f"  🔧 Recovering {method_name}: output exists despite '{status}' cache status")
            
            # Update the result to reflect successful completion
            recovered_result = result.copy()
            recovered_result["status"] = "recovered"
            recovered_result["original_status"] = status
            recovered_result["original_error"] = result.get("error")
            recovered_result["error"] = None
            recovered_result["result_path"] = str(expected_path)
            
            # Try to update the cache file with the corrected status
            try:
                save_method_result(method_name, recovered_result, output_dir)
            except Exception:
                pass  # Don't fail if we can't update cache
            
            return recovered_result
    except Exception:
        pass  # On any error, just return the original result
    
    return result


def load_cached_results(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all cached benchmark results.
    
    This also validates that timeout/error results are still accurate by checking
    if output files were created after the cache was written (e.g., if a Docker
    container continued running after being marked as timed out).
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    List[Dict[str, Any]]
        List of cached result dictionaries
    """
    cache_dir = output_dir / ".benchmark_cache"
    if not cache_dir.exists():
        return []
    
    cached_results = []
    for cache_file in cache_dir.glob("*.json"):
        # Skip config.json, temp files, and comparison cache files
        if (cache_file.name in ("config.json", ".config.json.tmp") or
            cache_file.name.startswith(".") or
            cache_file.name.endswith("_comparison.json")):
            continue
        
        try:
            with cache_file.open('r') as f:
                result = json.load(f)
                # Only add results that have a method field (exclude comparison metadata)
                if "method" in result:
                    # Check if the cache shows an error/timeout but the output file exists
                    # This can happen if Docker container continued after subprocess timeout
                    result = validate_and_recover_cache_result(result, cache_file, output_dir)
                    cached_results.append(result)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Skipping corrupted cache file {cache_file.name}: {exc}")
            continue
    
    return cached_results


def save_cache_config(
    output_dir: Path,
    qc_params: Optional[Dict[str, Any]],
    standardized_path: str
) -> None:
    """Save cache configuration for validation.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
    qc_params : Optional[Dict[str, Any]]
        QC parameters used (or None if adaptive)
    standardized_path : str
        Path to standardized dataset
    """
    cache_dir = output_dir / ".benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = cache_dir / "config.json"
    temp_file = cache_dir / ".config.json.tmp"
    
    config = {
        "cache_version": CACHE_VERSION,
        "qc_params": qc_params,
        "standardized_dataset_path": standardized_path,
        "timestamp": time.time(),
    }
    
    try:
        with temp_file.open('w') as f:
            json.dump(config, f, indent=2, sort_keys=True)
        temp_file.rename(config_file)
    except Exception as exc:
        if temp_file.exists():
            temp_file.unlink()
        print(f"Warning: Failed to save cache config: {exc}")


def load_cache_config(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load cache configuration.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Cached config, or None if doesn't exist or is corrupted
    """
    config_file = output_dir / ".benchmark_cache" / "config.json"
    
    if not config_file.exists():
        return None
    
    try:
        with config_file.open('r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def invalidate_cache(output_dir: Path, reason: str = "") -> None:
    """Clear the benchmark cache directory.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
    reason : str
        Reason for invalidation (for logging)
    """
    cache_dir = output_dir / ".benchmark_cache"
    
    if not cache_dir.exists():
        return
    
    if reason:
        print(f"  Invalidating cache: {reason}")
    
    # Remove all files in cache directory
    for cache_file in cache_dir.glob("*"):
        try:
            if cache_file.is_file():
                cache_file.unlink()
        except Exception as exc:
            print(f"Warning: Failed to remove cache file {cache_file.name}: {exc}")


def check_output_exists(method_name: str, output_dir: Path) -> bool:
    """Check if output file for a method already exists.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    bool
        True if output exists and is non-empty, or if valid cache exists
    """
    # First check if data output exists
    expected_path = get_expected_output_path(method_name, output_dir)
    if expected_path is not None and expected_path.exists():
        # Check file is non-empty (at least 1KB for h5ad files, any size for json)
        min_size = 1 if expected_path.suffix == '.json' else 1024
        if expected_path.stat().st_size >= min_size:
            return True
    
    # If no data output, check if cached benchmark result exists
    cached_result = load_method_result(method_name, output_dir)
    return cached_result is not None


def has_valid_result(row: pd.Series, output_dir: Path) -> bool:
    """Check if a method row has valid results that can be loaded.
    
    Returns True if:
    - status is "success" or "recovered", OR
    - status is "skipped_existing" and result file exists
    
    This allows cached/skipped methods to be included in reports and visualizations.
    
    Parameters
    ----------
    row : pd.Series
        Row from the benchmark results DataFrame
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    bool
        True if the method has valid results that can be loaded
    """
    status = row.get("status")
    if status in ("success", "recovered"):
        return True
    if status == "skipped_existing":
        # Check if result file exists
        method_name = row.get("method")
        result_path_val = row.get("result_path")
        resolved_path = resolve_result_path(method_name, result_path_val, output_dir)
        if resolved_path is not None and resolved_path.exists():
            return True
    return False
