"""Pre-sort a dataset by perturbation so NB-GLM's timed run reuses the copy.

crispyx's ``nb_glm_test`` first writes a copy of its input sorted by
perturbation (``<input>_sorted.h5ad``) when the input is not already sorted.
On genome-wide datasets that copy is ~50-60 GB and takes many hours on
Lustre, which would otherwise count against the NB-GLM method's time limit.
This makes the identical call up front, as its own untimed job; the
benchmark's ``crispyx_de_nb_glm`` then reuses the copy (crispyx >= 0.1.7
reuses it only when it was made from this exact source with the same
labels), so its runtime covers fitting only.

With PRESORT_IN_MEMORY=1, crispyx's own sorted-copy writer is fed from an
in-memory copy of the matrix (one sequential read) instead of fancy-indexing
the backed file row by row, which on Lustre ran at ~1 GB/h for Huang-HCT116
(~210 GB). The writer, layout and provenance are unchanged, so NB-GLM reuses
the copy as before. Needs RAM for the whole matrix (use a 768 GB node).

Usage (inside the benchmark container, from the repo root):
    python -m benchmarking.tools.presort_nbglm benchmarking/config/VCC.yaml
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import yaml

import anndata as ad
import crispyx.data
from crispyx.data import get_matrix_storage_format, needs_sorting_for_nbglm, sort_by_perturbation

REPO_ROOT = Path(__file__).resolve().parents[2]

# After standardization every dataset uses these labels, and the benchmark
# passes them to nb_glm_test (see create_benchmark_suite).
PERTURBATION_COLUMN = "perturbation"
CONTROL_LABEL = "control"


class _InMemoryMatrix:
    """Stands in for crispyx's backed AnnData inside the sorted-copy writer."""

    def __init__(self, path):
        t0 = time.perf_counter()
        self._adata = ad.read_h5ad(path)
        print(f"Loaded {path} into memory ({time.perf_counter() - t0:.0f} s)", flush=True)
        self.n_obs, self.n_vars, self.X = self._adata.n_obs, self._adata.n_vars, self._adata.X
        self.file = self  # the writer closes backed.file when done

    def close(self):
        pass


def _use_in_memory_sorted_writer() -> None:
    """Make crispyx's sparse sorted-copy writer read rows from memory."""
    write_sorted_sparse = crispyx.data._write_sorted_sparse

    def writer(**kwargs):
        read_backed = crispyx.data.read_backed
        crispyx.data.read_backed = _InMemoryMatrix
        try:
            return write_sorted_sparse(**kwargs)
        finally:
            crispyx.data.read_backed = read_backed

    crispyx.data._write_sorted_sparse = writer


def main(config_path: str) -> None:
    config = yaml.safe_load(open(config_path))
    dataset_stem = Path(config["dataset_path"]).stem
    output_dir = REPO_ROOT / config["output_dir"]
    if os.environ.get("NBGLM_INPUT") == "qc":
        # The NB-GLM input since crispyx 0.2.3: the QC-filtered counts (plan
        # 2026-10-25 §4), sorted next to it as crispyx_qc_filtered_sorted.h5ad.
        _sort(output_dir / "preprocessing" / "crispyx_qc_filtered.h5ad")
        return
    standardized = output_dir / ".cache" / f"standardized_{dataset_stem}.h5ad"
    if not standardized.exists():
        raise FileNotFoundError(f"Standardized file not found: {standardized}")
    # NB-GLM reads the CSR copy when the standardized file is not CSR
    # (same rule as _run_single_benchmark).
    nbglm_input = standardized
    if get_matrix_storage_format(standardized) != "csr":
        nbglm_input = (
            output_dir / "preprocessing"
            / f"standardized_csr_{standardized.stem.removeprefix('standardized_')}.h5ad"
        )
        # Same call (and mtime-based reuse) as the benchmark's own CSR step.
        from benchmarking.tools.run_benchmarks import run_preprocess_csr

        nbglm_input.parent.mkdir(parents=True, exist_ok=True)
        run_preprocess_csr(standardized, nbglm_input, chunk_size=config.get("chunk_size"))
    _sort(nbglm_input)


def _sort(nbglm_input: Path) -> None:
    if not nbglm_input.exists():
        raise FileNotFoundError(f"NB-GLM input not found: {nbglm_input}")
    if not needs_sorting_for_nbglm(nbglm_input, perturbation_column=PERTURBATION_COLUMN):
        print(f"{nbglm_input.name}: already sorted by perturbation, nothing to do")
        return

    if os.environ.get("PRESORT_IN_MEMORY") == "1":
        _use_in_memory_sorted_writer()
    t0 = time.perf_counter()
    sorted_path = sort_by_perturbation(
        nbglm_input,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=CONTROL_LABEL,
        output_path=nbglm_input.parent / f"{nbglm_input.stem}_sorted.h5ad",
    )
    print(f"Sorted copy: {sorted_path} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1])
