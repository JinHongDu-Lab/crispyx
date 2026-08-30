"""Utility functions for comparing differential expression results."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Import constants from centralized module
from .constants import DE_METRIC_KEYS, TOP_K_VALUES

# Threshold for detecting F-statistics (which are always positive)
# If all reference statistics are positive and the ratio of max to min
# exceeds this threshold, we consider it likely to be F-statistics
_F_STATISTIC_DETECTION_RATIO = 100.0

_DEFAULT_TOP_K = 50
_LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "is_hit",
    "hit",
    "hits",
    "label",
    "labels",
    "truth",
    "is_target",
    "target",
    "targets",
    "positive",
    "is_positive",
    "ground_truth",
)
_POSITIVE_LABEL_VALUES: set[str] = {
    "true",
    "t",
    "yes",
    "y",
    "1",
    "positive",
    "pos",
    "hit",
    "hits",
    "target",
    "targets",
    "perturbed",
    "responsive",
}


def _is_likely_f_statistic(series: pd.Series) -> bool:
    """Detect if a statistic series is likely F-statistics (always non-negative).
    
    F-statistics from edgeR's quasi-likelihood F-test are always non-negative,
    unlike Wald statistics (coef/SE) which can be positive or negative.
    Direct comparison of these is meaningless for raw correlation.
    
    Returns True if the series appears to contain F-statistics.
    """
    valid = series.dropna()
    if len(valid) < 10:
        return False
    
    # F-statistics are always non-negative (can be 0)
    if (valid < 0).any():
        return False
    
    # Wald statistics are symmetric around 0, F-statistics are right-skewed
    # Check if all values are non-negative AND there's right skew
    mean_val = valid.mean()
    median_val = valid.median()
    
    # F-statistics: mean > median (right skew), all non-negative
    # Wald statistics: mean ≈ median (symmetric around 0)
    
    # Key check: Wald statistics will have negative values
    # F-statistics will not. Since we already checked for negatives above,
    # we check if the distribution looks like it could have come from 
    # absolute values of a symmetric distribution
    
    # Check the ratio of values above vs below the mean
    # For symmetric Wald: roughly 50% above mean
    # For F-stat: more concentrated toward lower values
    above_mean_ratio = (valid > mean_val).mean()
    
    # F-statistics are chi-squared-like: many small values, few large ones
    # So above_mean_ratio should be < 0.5 (typically 0.3-0.4)
    if above_mean_ratio < 0.45:
        return True
    
    return False


def _safe_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> Optional[float]:
    valid = series_a.notna() & series_b.notna()
    if valid.sum() < 2:
        return None
    try:
        corr = series_a[valid].corr(series_b[valid], method=method)
    except Exception:  # pragma: no cover - defensive guard
        return None
    return float(corr) if pd.notna(corr) else None


def _compute_per_perturbation_corr(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    method: str,
    transform_log: bool = False,
) -> tuple[Optional[float], Optional[float], int]:
    """Compute correlation per perturbation and return mean, std, and count.
    
    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame with perturbation column and value columns
    col_a : str
        Column name for method A values
    col_b : str
        Column name for method B values
    method : str
        Correlation method ('pearson' or 'spearman')
    transform_log : bool
        If True, apply -log10 transformation before computing correlation
        
    Returns
    -------
    tuple[Optional[float], Optional[float], int]
        (mean_correlation, std_correlation, n_perturbations)
        Returns (None, None, 0) if no valid correlations can be computed
    """
    if col_a not in merged.columns or col_b not in merged.columns:
        return None, None, 0
    
    correlations = []
    
    for pert, group in merged.groupby("perturbation"):
        series_a = pd.to_numeric(group[col_a], errors="coerce")
        series_b = pd.to_numeric(group[col_b], errors="coerce")
        
        if transform_log:
            series_a = -np.log10(np.clip(series_a.to_numpy(), 1e-300, 1))
            series_b = -np.log10(np.clip(series_b.to_numpy(), 1e-300, 1))
            series_a = pd.Series(series_a, index=group.index)
            series_b = pd.Series(series_b, index=group.index)
        
        corr = _safe_corr(series_a, series_b, method)
        if corr is not None:
            correlations.append(corr)
    
    if not correlations:
        return None, None, 0
    
    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0
    
    return mean_corr, std_corr, len(correlations)


def _top_k_indices_fast(
    values: np.ndarray,
    k: int,
    use_absolute: bool,
    ascending: bool,
    genes: Optional[np.ndarray] = None,
    tiebreak_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get indices of top-k values using fast numpy operations.
    
    Uses np.argpartition for O(n) selection instead of O(n log n) full sort,
    with optional tie-breaking by *tiebreak_values* (e.g. |effect_size|,
    descending) then by gene name when ties exist.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of values to rank
    k : int
        Number of top entries to select (must be <= len(values))
    use_absolute : bool
        If True, rank by absolute values
    ascending : bool
        If True, smaller values are "top" (for p-values)
    genes : np.ndarray, optional
        Gene names for tertiary tie-breaking.
    tiebreak_values : np.ndarray, optional
        Secondary tie-breaking values (ranked by descending absolute value).
        Typically |effect_size| so that, among genes with equal p-values,
        the ones with the largest fold-change are selected first.
    
    Returns
    -------
    np.ndarray
        Indices of top-k elements
    """
    n = len(values)
    if k >= n:
        # Return all indices sorted
        if use_absolute:
            sort_vals = np.abs(values)
            order = np.argsort(sort_vals) if ascending else np.argsort(-sort_vals)
        else:
            order = np.argsort(values) if ascending else np.argsort(-values)
        return order
    
    # Prepare values for ranking
    if use_absolute:
        rank_vals = np.abs(values)
    else:
        rank_vals = values.copy()
    
    # For descending order, negate values
    if not ascending:
        rank_vals = -rank_vals
    
    # Use argpartition for O(n) top-k selection
    # argpartition puts the k smallest elements in the first k positions (unordered)
    part_idx = np.argpartition(rank_vals, k)[:k]
    
    # Sort the k candidates to get proper order
    top_k_vals = rank_vals[part_idx]
    sorted_order = np.argsort(top_k_vals)
    top_k_idx = part_idx[sorted_order]
    
    # Apply tie-breaking only if needed (ties exist in top-k values)
    if tiebreak_values is not None or genes is not None:
        final_vals = rank_vals[top_k_idx]
        if len(final_vals) > 1:
            diffs = np.diff(final_vals)
            has_ties = np.any(np.abs(diffs) < 1e-15)
            
            if has_ties:
                # Build lexsort keys: last key is primary (ascending rank_vals),
                # second-to-last is secondary (|effect_size| descending), etc.
                keys: list[np.ndarray] = []
                if genes is not None:
                    keys.append(genes[top_k_idx])
                if tiebreak_values is not None:
                    # Negate so larger |effect_size| comes first
                    keys.append(-np.abs(tiebreak_values[top_k_idx]))
                keys.append(final_vals)
                sort_keys = np.lexsort(tuple(keys))
                top_k_idx = top_k_idx[sort_keys]
    
    return top_k_idx


def _compute_overlap_vectorized(
    values_a: np.ndarray,
    values_b: np.ndarray,
    k: int,
    use_absolute: bool,
    ascending: bool,
    genes: Optional[np.ndarray] = None,
    tiebreak_a: Optional[np.ndarray] = None,
    tiebreak_b: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Compute top-k overlap between two value arrays.
    
    Optimized version using numpy operations instead of pandas.
    
    Parameters
    ----------
    tiebreak_a, tiebreak_b : np.ndarray, optional
        Secondary sort values for methods A and B (e.g. effect_size).
        Used to break ties in the primary ranking column.
    
    Returns
    -------
    Optional[float]
        Overlap ratio (0-1) or None if computation fails
    """
    # Remove NaN values (keep only rows where both are valid)
    valid_mask = ~(np.isnan(values_a) | np.isnan(values_b))
    if not valid_mask.any():
        return None
    
    valid_a = values_a[valid_mask]
    valid_b = values_b[valid_mask]
    valid_genes = genes[valid_mask] if genes is not None else None
    valid_tb_a = tiebreak_a[valid_mask] if tiebreak_a is not None else None
    valid_tb_b = tiebreak_b[valid_mask] if tiebreak_b is not None else None
    
    n = len(valid_a)
    if n == 0:
        return None
    
    actual_k = min(k, n)
    if actual_k <= 0:
        return None
    
    # Get top-k indices for each method
    # We need to track original positions, so work with indices
    idx_a = _top_k_indices_fast(valid_a, actual_k, use_absolute, ascending, valid_genes, valid_tb_a)
    idx_b = _top_k_indices_fast(valid_b, actual_k, use_absolute, ascending, valid_genes, valid_tb_b)
    
    # Compute overlap
    overlap = len(set(idx_a) & set(idx_b))
    return float(overlap) / float(actual_k)


def _top_k_overlap(
    series_a: pd.Series,
    series_b: pd.Series,
    top_k: int = 50,
    use_absolute: bool = False,
    ascending: bool = True,
) -> Optional[float]:
    """Compute top-k overlap between two pandas Series.
    
    Wrapper around _compute_overlap_vectorized for backward compatibility.
    
    Parameters
    ----------
    series_a : pd.Series
        Values from method A
    series_b : pd.Series
        Values from method B
    top_k : int
        Number of top entries to compare
    use_absolute : bool
        If True, rank by absolute values
    ascending : bool
        If True, smaller values are "top" (for p-values)
    
    Returns
    -------
    Optional[float]
        Overlap ratio (0-1) or None if computation fails
    """
    # Align series on common index
    common_idx = series_a.index.intersection(series_b.index)
    if len(common_idx) == 0:
        return None
    
    values_a = series_a.loc[common_idx].to_numpy()
    values_b = series_b.loc[common_idx].to_numpy()
    
    return _compute_overlap_vectorized(
        values_a, values_b, top_k, use_absolute, ascending
    )


def _compute_per_perturbation_overlap(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    top_k: int,
    use_absolute: bool,
    ascending: bool,
) -> tuple[Optional[float], Optional[float], int]:
    """Compute top-k overlap per perturbation and return mean, std, and count.
    
    Optimized version that pre-converts data types and uses vectorized operations.
    
    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame with perturbation column and value columns
    col_a : str
        Column name for method A values
    col_b : str
        Column name for method B values
    top_k : int
        Number of top genes to consider. Will be capped at min(top_k, n_genes)
        for each perturbation.
    use_absolute : bool
        If True, rank by absolute values (for effect sizes)
    ascending : bool
        If True, sort ascending (for p-values); if False, sort descending
        
    Returns
    -------
    tuple[Optional[float], Optional[float], int]
        (mean_overlap, std_overlap, n_perturbations)
        Returns (None, None, 0) if no valid overlaps can be computed
    """
    if col_a not in merged.columns or col_b not in merged.columns:
        return None, None, 0
    
    # Pre-convert to numpy arrays for speed (do this once, not per perturbation)
    values_a_all = pd.to_numeric(merged[col_a], errors="coerce").to_numpy()
    values_b_all = pd.to_numeric(merged[col_b], errors="coerce").to_numpy()
    
    # Check if gene column exists for tie-breaking
    has_genes = "gene" in merged.columns
    genes_all = merged["gene"].to_numpy() if has_genes else None
    
    # Get perturbation groups using numpy for speed
    # Handle NaN values in perturbation column by converting to string first
    perturbations = merged["perturbation"].fillna("__NA__").astype(str).to_numpy()
    unique_perts, inverse_idx = np.unique(perturbations, return_inverse=True)
    
    overlaps = []
    
    for pert_idx in range(len(unique_perts)):
        # Get mask for this perturbation
        mask = inverse_idx == pert_idx
        
        values_a = values_a_all[mask]
        values_b = values_b_all[mask]
        genes = genes_all[mask] if has_genes else None
        
        overlap = _compute_overlap_vectorized(
            values_a, values_b, top_k, use_absolute, ascending, genes
        )
        if overlap is not None:
            overlaps.append(overlap)
    
    if not overlaps:
        return None, None, 0
    
    overlaps_arr = np.array(overlaps)
    mean_overlap = float(np.mean(overlaps_arr))
    std_overlap = float(np.std(overlaps_arr, ddof=1)) if len(overlaps_arr) > 1 else 0.0
    
    return mean_overlap, std_overlap, len(overlaps)


def _compute_per_perturbation_overlap_multi_k(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    k_values: tuple[int, ...],
    use_absolute: bool,
    ascending: bool,
    tiebreak_col_a: Optional[str] = None,
    tiebreak_col_b: Optional[str] = None,
) -> Dict[int, Optional[float]]:
    """Compute top-k overlap per perturbation for multiple k values in one pass.
    
    This is an optimized version that iterates over perturbations once and
    computes overlaps for all k values, avoiding redundant iterations.
    
    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame with perturbation column and value columns
    col_a : str
        Column name for method A values
    col_b : str
        Column name for method B values
    k_values : tuple[int, ...]
        Tuple of k values to compute
    use_absolute : bool
        If True, rank by absolute values (for effect sizes)
    ascending : bool
        If True, sort ascending (for p-values); if False, sort descending
        
    Returns
    -------
    Dict[int, Optional[float]]
        Dictionary mapping k values to mean overlap (None if computation fails)
    """
    if col_a not in merged.columns or col_b not in merged.columns:
        return {k: None for k in k_values}
    
    # Pre-convert to numpy arrays for speed
    values_a_all = pd.to_numeric(merged[col_a], errors="coerce").to_numpy()
    values_b_all = pd.to_numeric(merged[col_b], errors="coerce").to_numpy()
    
    # Check if gene column exists for tie-breaking
    has_genes = "gene" in merged.columns
    genes_all = merged["gene"].to_numpy() if has_genes else None
    
    # Optional effect-size tiebreaker arrays
    tb_a_all = (
        pd.to_numeric(merged[tiebreak_col_a], errors="coerce").to_numpy()
        if tiebreak_col_a and tiebreak_col_a in merged.columns
        else None
    )
    tb_b_all = (
        pd.to_numeric(merged[tiebreak_col_b], errors="coerce").to_numpy()
        if tiebreak_col_b and tiebreak_col_b in merged.columns
        else None
    )
    
    # Get perturbation groups using numpy for speed
    perturbations = merged["perturbation"].fillna("__NA__").astype(str).to_numpy()
    unique_perts, inverse_idx = np.unique(perturbations, return_inverse=True)
    
    # Collect overlaps per k value
    overlaps_by_k: Dict[int, list] = {k: [] for k in k_values}
    max_k = max(k_values)
    
    for pert_idx in range(len(unique_perts)):
        # Get mask for this perturbation
        mask = inverse_idx == pert_idx
        
        values_a = values_a_all[mask]
        values_b = values_b_all[mask]
        n_genes = len(values_a)
        genes = genes_all[mask] if has_genes else None
        tb_a = tb_a_all[mask] if tb_a_all is not None else None
        tb_b = tb_b_all[mask] if tb_b_all is not None else None
        
        # Skip if not enough genes for any k
        if n_genes == 0:
            continue
        
        # Compute overlap for each k value
        for k in k_values:
            actual_k = min(k, n_genes)
            if actual_k <= 0:
                continue
            overlap = _compute_overlap_vectorized(
                values_a, values_b, actual_k, use_absolute, ascending, genes,
                tb_a, tb_b,
            )
            if overlap is not None:
                overlaps_by_k[k].append(overlap)
    
    # Compute means for each k
    result: Dict[int, Optional[float]] = {}
    for k in k_values:
        overlaps = overlaps_by_k[k]
        if overlaps:
            result[k] = float(np.mean(overlaps))
        else:
            result[k] = None
    
    return result


def compute_pairwise_overlap_matrix(
    de_results: Dict[str, pd.DataFrame],
    top_k: int = 100,
    metric: str = "effect",
) -> tuple[pd.DataFrame, int]:
    """Compute pairwise top-k overlap matrix between all DE methods.
    
    For each pair of methods, computes the mean top-k overlap across all
    shared perturbations. The matrix is symmetric with 1.0 on the diagonal.
    
    Parameters
    ----------
    de_results : Dict[str, pd.DataFrame]
        Dictionary mapping method names to their DE result DataFrames.
        Each DataFrame should have columns: perturbation, gene, effect_size, pvalue
    top_k : int
        Number of top genes to consider for overlap (default: 100)
    metric : str
        Which metric to use for ranking: "effect" (by |effect_size|) or
        "pvalue" (by ascending p-value). Default: "effect"
        
    Returns
    -------
    tuple[pd.DataFrame, int]
        (overlap_matrix, effective_k) where:
        - overlap_matrix: DataFrame with method names as index and columns,
          containing mean overlap values. NaN for missing/incompatible pairs.
        - effective_k: The actual k used (may be less than top_k if fewer genes)
    """
    method_names = list(de_results.keys())
    n_methods = len(method_names)
    
    # Initialize matrix with NaN
    matrix = pd.DataFrame(
        np.full((n_methods, n_methods), np.nan),
        index=method_names,
        columns=method_names,
    )
    
    # Set diagonal to 1.0 (perfect self-overlap)
    for method in method_names:
        matrix.loc[method, method] = 1.0
    
    # Track effective k (minimum across all comparisons)
    effective_k = top_k
    
    # Configure ranking based on metric type
    if metric == "effect":
        col_name = "effect_size"
        use_absolute = True
        ascending = False
    else:  # pvalue
        col_name = "pvalue"
        use_absolute = False
        ascending = True
    
    # Compute pairwise overlaps
    for i, method_a in enumerate(method_names):
        for j, method_b in enumerate(method_names):
            if i >= j:  # Skip diagonal and lower triangle (will mirror)
                continue
            
            df_a = de_results[method_a]
            df_b = de_results[method_b]
            
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            
            # Standardize column names if needed
            if col_name not in df_a.columns or col_name not in df_b.columns:
                continue
            
            # Merge on perturbation + gene (compute on gene intersection)
            merged = df_a.merge(
                df_b,
                on=["perturbation", "gene"],
                suffixes=("_a", "_b"),
                how="inner",
            )
            
            if merged.empty:
                continue
            
            col_a = f"{col_name}_a"
            col_b = f"{col_name}_b"
            
            mean_overlap, _, n_pert = _compute_per_perturbation_overlap(
                merged, col_a, col_b, top_k, use_absolute, ascending
            )
            
            if mean_overlap is not None:
                matrix.loc[method_a, method_b] = mean_overlap
                matrix.loc[method_b, method_a] = mean_overlap
                
                # Track minimum genes per perturbation for effective k (optimized)
                pert_counts = merged.groupby("perturbation", sort=False).size()
                min_genes = pert_counts.min() if len(pert_counts) > 0 else top_k
                effective_k = min(effective_k, min_genes, top_k)
    
    return matrix, effective_k


def compute_pairwise_overlap_matrices_batch(
    de_results: Dict[str, pd.DataFrame],
    k_values: tuple[int, ...] = (50, 100, 500),
    metrics: tuple[str, ...] = ("effect", "pvalue"),
) -> Dict[str, tuple[pd.DataFrame, int]]:
    """Compute pairwise overlap matrices for multiple k values and metrics in one pass.
    
    This is an optimized version that caches merged DataFrames and computes
    overlaps for all k values in a single pass over the data.
    
    Parameters
    ----------
    de_results : Dict[str, pd.DataFrame]
        Dictionary mapping method names to their DE result DataFrames
    k_values : tuple[int, ...]
        Top-k values to compute (default: (50, 100, 500))
    metrics : tuple[str, ...]
        Metrics to compute: "effect" and/or "pvalue"
        
    Returns
    -------
    Dict[str, tuple[pd.DataFrame, int]]
        Dictionary mapping "{metric}_top_{k}" keys to (matrix, effective_k) tuples
    """
    method_names = list(de_results.keys())
    n_methods = len(method_names)
    
    # Initialize result matrices for all combinations
    results: Dict[str, tuple[pd.DataFrame, int]] = {}
    matrices: Dict[str, pd.DataFrame] = {}
    effective_ks: Dict[str, int] = {}
    
    for metric in metrics:
        for k in k_values:
            key = f"{metric}_top_{k}"
            matrices[key] = pd.DataFrame(
                np.full((n_methods, n_methods), np.nan),
                index=method_names,
                columns=method_names,
            )
            # Set diagonal to 1.0
            for method in method_names:
                matrices[key].loc[method, method] = 1.0
            effective_ks[key] = k
    
    # Cache merged DataFrames to avoid redundant merge operations
    merged_cache: Dict[tuple[str, str], pd.DataFrame] = {}
    
    # Compute pairwise overlaps
    for i, method_a in enumerate(method_names):
        for j, method_b in enumerate(method_names):
            if i >= j:  # Skip diagonal and lower triangle
                continue
            
            df_a = de_results[method_a]
            df_b = de_results[method_b]
            
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            
            # Create cache key (order-independent)
            cache_key = (min(method_a, method_b), max(method_a, method_b))
            
            # Get or create merged DataFrame
            if cache_key not in merged_cache:
                # Merge once for all metrics and k values
                merged = df_a.merge(
                    df_b,
                    on=["perturbation", "gene"],
                    suffixes=("_a", "_b"),
                    how="inner",
                )
                if not merged.empty:
                    merged_cache[cache_key] = merged
            
            if cache_key not in merged_cache:
                continue
                
            merged = merged_cache[cache_key]
            
            # Get min genes per perturbation once for effective_k
            pert_counts = merged.groupby("perturbation", sort=False).size()
            min_genes = int(pert_counts.min()) if len(pert_counts) > 0 else max(k_values)
            
            # Compute overlaps for each metric using multi-k function (single pass per metric)
            for metric in metrics:
                if metric == "effect":
                    col_name = "effect_size"
                    use_absolute = True
                    ascending = False
                else:  # pvalue
                    col_name = "pvalue"
                    use_absolute = False
                    ascending = True
                
                col_a = f"{col_name}_a"
                col_b = f"{col_name}_b"
                
                if col_a not in merged.columns or col_b not in merged.columns:
                    continue
                
                # Compute all k values in one pass over perturbations
                overlaps_by_k = _compute_per_perturbation_overlap_multi_k(
                    merged, col_a, col_b, k_values, use_absolute, ascending
                )
                
                for k, mean_overlap in overlaps_by_k.items():
                    key = f"{metric}_top_{k}"
                    if mean_overlap is not None:
                        matrices[key].loc[method_a, method_b] = mean_overlap
                        matrices[key].loc[method_b, method_a] = mean_overlap
                        effective_ks[key] = min(effective_ks[key], min_genes, k)
    
    # Build results dictionary
    for key in matrices:
        results[key] = (matrices[key], effective_ks[key])
    
    return results


def _normalise_labels(series: pd.Series) -> Optional[pd.Series]:
    if series.empty:
        return None

    data = series.dropna()
    if data.empty:
        return None

    if data.dtype == bool:
        mapped = data.astype(int)
    else:
        numeric = pd.to_numeric(data, errors="coerce")
        numeric_unique = {float(value) for value in numeric.dropna().unique().tolist()}
        if numeric.notna().sum() >= 2 and numeric_unique <= {0.0, 1.0} and len(numeric_unique) == 2:
            mapped = numeric.astype(int)
        else:
            lowered = data.astype(str).str.lower()
            positive_mask = lowered.isin(_POSITIVE_LABEL_VALUES)
            if not positive_mask.any() or positive_mask.all():
                return None
            mapped = positive_mask.astype(int)

    if mapped.nunique(dropna=True) < 2:
        return None

    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[mapped.index] = mapped.astype(float)
    return output


def _find_label_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    for column in frame.columns:
        base = column.replace("_stream", "").replace("_reference", "")
        if base in _LABEL_COLUMN_CANDIDATES:
            labels = _normalise_labels(frame[column])
            if labels is not None:
                return labels
    return None


def _prepare_auc_scores(series: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna()
    if valid.sum() < 2:
        return None
    clipped = np.clip(numeric[valid].to_numpy(dtype=float), 1e-300, None)
    transformed = -np.log10(clipped)
    return pd.Series(transformed, index=numeric.index[valid], dtype=float)


def _compute_binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    sum_ranks_pos = float(ranks[positives].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_de_comparison_metrics(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> Dict[str, Optional[float]]:
    """Return a dictionary of metrics comparing two differential expression tables.
    
    Computes correlation, overlap, and statistical comparison metrics between two
    differential expression result DataFrames. Both inputs must contain columns:
    'perturbation', 'gene', 'effect_size', 'statistic', and 'pvalue'.
    
    Label-based metrics (pvalue_method_a_auroc, pvalue_method_b_auroc) require
    ground truth labels in a column such as 'is_hit', 'label', or 'ground_truth'.
    These AUROC metrics are typically only available in benchmark/validation
    scenarios where true positive perturbations are known.
    
    Args:
        method_a: DataFrame with first method's DE results
        method_b: DataFrame with second method's DE results
        top_k: Number of top hits to consider for overlap metrics (default: 50)
    
    Returns:
        Dictionary with 14 metric keys defined in DE_METRIC_KEYS. Metrics include:
        - Correlation (Pearson, Spearman) for effect sizes, statistics, and p-values
        - Top-k overlap for ranking concordance
        - Max absolute differences
        - AUROC for p-values (only when ground truth labels are present)
    """

    metrics: Dict[str, Optional[float]] = {key: None for key in DE_METRIC_KEYS}

    if method_a is None or method_b is None or method_a.empty or method_b.empty:
        return metrics

    merged = method_a.merge(
        method_b,
        on=["perturbation", "gene"],
        suffixes=("_method_a", "_method_b"),
        how="inner",
    )
    if merged.empty:
        return metrics

    merged = merged.copy()
    pair_index = merged["perturbation"].astype(str) + "||" + merged["gene"].astype(str)
    merged.index = pair_index

    column_settings = {
        "effect": {
            "method_a": "effect_size_method_a",
            "method_b": "effect_size_method_b",
            "use_absolute": True,
            "ascending": False,
        },
        "statistic": {
            "method_a": "statistic_method_a",
            "method_b": "statistic_method_b",
            "use_absolute": True,
            "ascending": False,
        },
        "pvalue": {
            "method_a": "pvalue_method_a",
            "method_b": "pvalue_method_b",
            "use_absolute": False,
            "ascending": True,
        },
    }

    # Detect if we're comparing Wald statistics (signed) vs F-statistics (unsigned)
    statistic_type_mismatch = False
    if "statistic_method_a" in merged.columns and "statistic_method_b" in merged.columns:
        method_a_stat = pd.to_numeric(merged["statistic_method_a"], errors="coerce")
        method_b_stat = pd.to_numeric(merged["statistic_method_b"], errors="coerce")
        method_a_is_f = _is_likely_f_statistic(method_a_stat)
        method_b_is_f = _is_likely_f_statistic(method_b_stat)
        if method_a_is_f != method_b_is_f:
            statistic_type_mismatch = True
            metrics["statistic_type_mismatch"] = 1.0

    for name, settings in column_settings.items():
        method_a_col = settings["method_a"]
        method_b_col = settings["method_b"]
        if method_a_col not in merged.columns or method_b_col not in merged.columns:
            continue
        method_a_series = pd.to_numeric(merged[method_a_col], errors="coerce")
        method_b_series = pd.to_numeric(merged[method_b_col], errors="coerce")
        valid = method_a_series.notna() & method_b_series.notna()
        if valid.sum() == 0:
            continue
        diff = (method_a_series[valid] - method_b_series[valid]).abs()
        if not diff.empty:
            metrics[f"{name}_max_abs_diff"] = float(diff.max())
        
        # For statistics, if there's a type mismatch (Wald vs F), skip raw correlation
        # but still compute overlap using absolute values
        if name == "statistic" and statistic_type_mismatch:
            # Use absolute values for correlation when comparing Wald vs F
            # Create a copy with absolute values for per-perturbation correlation
            merged_abs = merged.copy()
            merged_abs[method_a_col] = pd.to_numeric(merged[method_a_col], errors="coerce").abs()
            merged_abs[method_b_col] = pd.to_numeric(merged[method_b_col], errors="coerce").abs()
            
            mean_p, std_p, n_pert = _compute_per_perturbation_corr(merged_abs, method_a_col, method_b_col, "pearson")
            metrics[f"{name}_pearson_corr_mean"] = mean_p
            metrics[f"{name}_pearson_corr_std"] = std_p
            
            mean_s, std_s, _ = _compute_per_perturbation_corr(merged_abs, method_a_col, method_b_col, "spearman")
            metrics[f"{name}_spearman_corr_mean"] = mean_s
            metrics[f"{name}_spearman_corr_std"] = std_s
        elif name == "pvalue":
            # For p-values, compute correlations on -log10 transformed values
            mean_p, std_p, n_pert = _compute_per_perturbation_corr(
                merged, method_a_col, method_b_col, "pearson", transform_log=True
            )
            metrics["pvalue_log_pearson_corr_mean"] = mean_p
            metrics["pvalue_log_pearson_corr_std"] = std_p
            
            mean_s, std_s, _ = _compute_per_perturbation_corr(
                merged, method_a_col, method_b_col, "spearman", transform_log=True
            )
            metrics["pvalue_log_spearman_corr_mean"] = mean_s
            metrics["pvalue_log_spearman_corr_std"] = std_s
            metrics["n_perturbations"] = float(n_pert)
        else:
            mean_p, std_p, n_pert = _compute_per_perturbation_corr(merged, method_a_col, method_b_col, "pearson")
            metrics[f"{name}_pearson_corr_mean"] = mean_p
            metrics[f"{name}_pearson_corr_std"] = std_p
            
            mean_s, std_s, _ = _compute_per_perturbation_corr(merged, method_a_col, method_b_col, "spearman")
            metrics[f"{name}_spearman_corr_mean"] = mean_s
            metrics[f"{name}_spearman_corr_std"] = std_s
            
            if name == "effect":
                metrics["n_perturbations"] = float(n_pert)
        
        # Compute per-perturbation overlap for all k values (including default top_k).
        # Using per-perturbation overlap avoids spurious 0% overlap when many p-values
        # underflow to exactly 0.0, causing arbitrary tie-breaking in global ranking.
        if name in ("effect", "pvalue"):
            # Compute for all k values in one pass (optimized multi-k function)
            k_values = (top_k, 100, 500)
            # For p-value overlap, break ties by |effect_size| so that
            # both methods agree on rankings when p-values are equal.
            tb_a = tb_b = None
            if name == "pvalue":
                tb_a = "effect_size_method_a"
                tb_b = "effect_size_method_b"
            overlaps_by_k = _compute_per_perturbation_overlap_multi_k(
                merged,
                method_a_col,
                method_b_col,
                k_values=k_values,
                use_absolute=settings["use_absolute"],
                ascending=settings["ascending"],
                tiebreak_col_a=tb_a,
                tiebreak_col_b=tb_b,
            )
            for k, mean_ovl in overlaps_by_k.items():
                if k == top_k:
                    # Use mean for the primary top_k_overlap metric
                    metrics[f"{name}_top_k_overlap"] = mean_ovl
                else:
                    metrics[f"{name}_top_{k}_overlap_mean"] = mean_ovl
                    # Note: std is not computed by multi-k function for performance
                    # metrics[f"{name}_top_{k}_overlap_std"] = std_ovl
        else:
            # For statistic, use global overlap (less sensitive to ties)
            metrics[f"{name}_top_k_overlap"] = _top_k_overlap(
                method_a_series,
                method_b_series,
                top_k=top_k,
                use_absolute=settings["use_absolute"],
                ascending=settings["ascending"],
            )

    label_series = _find_label_series(merged)
    if label_series is not None:
        label_series = label_series.astype(float)
        if "pvalue_method_a" in merged.columns:
            method_a_scores = _prepare_auc_scores(merged["pvalue_method_a"])
            if method_a_scores is not None:
                labels = label_series.dropna()
                valid_idx = labels.index.intersection(method_a_scores.index)
                if len(valid_idx) >= 2:
                    y_true = labels.loc[valid_idx].astype(int).to_numpy()
                    if np.unique(y_true).size == 2:
                        y_score = method_a_scores.loc[valid_idx].to_numpy()
                        metrics["pvalue_method_a_auroc"] = _compute_binary_roc_auc(
                            y_true, y_score
                        )
        if "pvalue_method_b" in merged.columns:
            method_b_scores = _prepare_auc_scores(merged["pvalue_method_b"])
            if method_b_scores is not None:
                labels = label_series.dropna()
                valid_idx = labels.index.intersection(method_b_scores.index)
                if len(valid_idx) >= 2:
                    y_true = labels.loc[valid_idx].astype(int).to_numpy()
                    if np.unique(y_true).size == 2:
                        y_score = method_b_scores.loc[valid_idx].to_numpy()
                        metrics["pvalue_method_b_auroc"] = _compute_binary_roc_auc(
                            y_true, y_score
                        )

    return metrics

