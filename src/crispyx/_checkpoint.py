"""Checkpoint and progress utilities for streaming, resumable computations.

This module provides atomic checkpointing and progress tracking shared by
streaming differential expression tests and by ``batch_process``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

# Check for tqdm availability
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _write_checkpoint_atomic(
    checkpoint_path: Path,
    data: dict,
) -> None:
    """Write checkpoint data atomically using temp file + rename.
    
    This ensures checkpoint file is never corrupted on crash.
    """
    # Ensure parent directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temporary file in the same directory
    tmp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        # Atomic rename
        os.rename(tmp_path, checkpoint_path)
    except Exception:
        # Clean up temp file on error
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def _read_checkpoint(checkpoint_path: Path) -> dict | None:
    """Read checkpoint file, returning None if missing or corrupted.
    
    Returns
    -------
    dict or None
        Checkpoint data if valid, None if file is missing or corrupted.
    """
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        # Validate required fields
        if not isinstance(data, dict):
            return None
        if "completed" not in data or "total" not in data:
            return None
        return data
    except (json.JSONDecodeError, IOError, OSError):
        return None


def _scan_h5ad_completed(
    h5ad_path: Path,
    all_candidates: list[str],
    result_dataset: str = "uns/rank_genes_groups/full/scores",
) -> list[str]:
    """Scan h5ad file to detect completed perturbations by non-zero/non-NaN rows.
    
    This is a fallback when checkpoint file is missing or corrupted.
    
    Parameters
    ----------
    h5ad_path
        Path to the output h5ad file.
    all_candidates
        List of all perturbation labels (in order).
    result_dataset
        HDF5 dataset path to check for results. Should have shape (n_groups, n_genes).
        
    Returns
    -------
    list[str]
        List of perturbation labels that have been completed.
    """
    completed = []
    if not h5ad_path.exists():
        return completed
    
    try:
        with h5py.File(h5ad_path, "r") as f:
            # Try to access the result dataset
            if result_dataset in f:
                ds = f[result_dataset]
                n_groups = ds.shape[0]
                for idx in range(min(n_groups, len(all_candidates))):
                    row = ds[idx, :]
                    # Check if row has any non-NaN, non-zero values
                    if np.any(np.isfinite(row) & (row != 0)):
                        completed.append(all_candidates[idx])
            else:
                # Try alternative: check layers in X matrix
                if "X" in f:
                    X = f["X"]
                    if hasattr(X, "shape") and len(X.shape) == 2:
                        n_groups = X.shape[0]
                        for idx in range(min(n_groups, len(all_candidates))):
                            row = X[idx, :]
                            if np.any(np.isfinite(row) & (row != 0)):
                                completed.append(all_candidates[idx])
    except Exception as e:
        logger.warning(f"Failed to scan h5ad for completed perturbations: {e}")
    
    return completed


def _find_last_completed_gene_chunk(
    h5ad_path: Path,
    n_gene_chunks: int,
    chunk_size: int,
    n_genes: int,
    weight_dataset: str = "layers/weight_sum",
) -> int:
    """Scan an output file's weight layer for the last fully-written gene
    chunk, as a fallback when the checkpoint JSON is missing or corrupted.

    Used by ``batch_process``, whose gene chunks are written strictly in
    order (unlike DE's per-candidate resume, where any candidate can
    complete independently): a chunk is "done" once its column slice of
    ``weight_dataset`` holds any finite, positive weight. Scanning stops at
    the first not-yet-written chunk. Returns -1 if none are complete.
    """
    if not h5ad_path.exists():
        return -1
    try:
        with h5py.File(h5ad_path, "r") as f:
            if weight_dataset not in f:
                return -1
            ds = f[weight_dataset]
            last = -1
            for i in range(n_gene_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, n_genes)
                column = np.asarray(ds[:, start:end])
                if np.any(np.isfinite(column) & (column > 0)):
                    last = i
                else:
                    break
            return last
    except Exception as e:
        logger.warning(f"Failed to scan h5ad for completed gene chunks: {e}")
        return -1


def _get_resumable_candidates(
    checkpoint_path: Path,
    h5ad_path: Path,
    all_candidates: list[str],
    retry_failed: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Get candidates to process, accounting for previous progress.

    Parameters
    ----------
    checkpoint_path
        Path to the progress JSON file.
    h5ad_path
        Path to the output h5ad file.
    all_candidates
        List of all perturbation labels to process.
    retry_failed
        If True, previously failed perturbations will be retried.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        (candidates_to_run, completed, failed)
        - candidates_to_run: perturbations that need to be processed
        - completed: perturbations already completed
        - failed: perturbations that failed (for logging)
    """
    checkpoint = _read_checkpoint(checkpoint_path)

    if checkpoint is not None:
        completed = checkpoint.get("completed", [])
        failed = checkpoint.get("failed", [])
        logger.info(f"Resuming: {len(completed)}/{len(all_candidates)} already completed")
        if failed:
            logger.info(f"  {len(failed)} previously failed perturbations")
    else:
        # Checkpoint missing or corrupted - try scanning h5ad
        if h5ad_path.exists():
            logger.warning(
                f"Checkpoint file missing or corrupted at {checkpoint_path}. "
                f"Scanning h5ad file to detect completed perturbations..."
            )
            completed = _scan_h5ad_completed(h5ad_path, all_candidates)
            failed = []
            if completed:
                logger.info(f"Detected {len(completed)} completed perturbations from h5ad scan")
        else:
            completed = []
            failed = []
    
    # Determine which candidates to run
    completed_set = set(completed)
    failed_set = set(failed) if not retry_failed else set()
    
    candidates_to_run = [
        c for c in all_candidates 
        if c not in completed_set and c not in failed_set
    ]
    
    return candidates_to_run, completed, failed


def _get_checkpoint_interval(n_perturbations: int, checkpoint_interval: int | None) -> int:
    """Determine checkpoint interval based on dataset size.
    
    Parameters
    ----------
    n_perturbations
        Total number of perturbations.
    checkpoint_interval
        User-specified interval, or None for auto.
        
    Returns
    -------
    int
        Number of perturbations to process between checkpoints.
    """
    if checkpoint_interval is not None:
        return max(1, checkpoint_interval)
    # Auto: every 1 for small datasets, every 10 for larger ones
    if n_perturbations < 100:
        return 1
    elif n_perturbations < 1000:
        return 10
    else:
        return 50


class _DummyProgress:
    """Dummy progress bar that does nothing (for when verbose=False)."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n: int = 1):
        pass
    
    def set_postfix(self, **kwargs):
        pass


def _create_progress_context(
    total: int,
    desc: str,
    verbose: int | bool,
    *,
    unit: str = "perturbation",
) -> "_tqdm | _DummyProgress":
    """Create a progress bar context manager.

    Returns tqdm progress bar if verbose>=1 and tqdm is available,
    otherwise returns a dummy context manager.
    """
    if int(verbose) >= 1 and HAS_TQDM and total > 0:
        return _tqdm(total=total, desc=desc, unit=unit)
    return _DummyProgress()
