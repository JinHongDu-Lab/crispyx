"""Standardize a dataset with a verified file copy, before the benchmark run.

crispyx's ``standardize_dataset`` copies the raw ``.h5ad`` with
``shutil.copy2`` and then edits the copy in place. On HPC Lustre that copy
has twice come back short without raising (Marson-CD4T: 7 of 50 GB;
Huang-HCT116: 53 of 209 GB), so the next step failed with "truncated file".
This makes the same ``standardize_dataset`` call as
``_run_single_benchmark`` but with a chunked copy whose size is checked,
leaving ``.cache/standardized_<name>.h5ad`` for the benchmark to reuse.

Usage (inside the benchmark container, from the repo root):
    python -m benchmarking.tools.prestandardize benchmarking/config/Huang-HCT116.yaml
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import crispyx.data
from benchmarking.tools.run_benchmarks import BenchmarkConfig

_COPY_CHUNK_BYTES = 64 * 1024 * 1024


def _verified_copy(src, dst, *, follow_symlinks=True):
    """Chunked copy that fails loudly instead of returning a short file."""
    expected = os.path.getsize(src)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while chunk := fin.read(_COPY_CHUNK_BYTES):
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    copied = os.path.getsize(dst)
    if copied != expected:
        raise OSError(f"Short copy of {src}: {copied} of {expected} bytes")
    shutil.copystat(src, dst, follow_symlinks=follow_symlinks)
    return dst


def main(config_path: str) -> None:
    config = BenchmarkConfig.from_yaml(Path(config_path))
    standardized = config.output_dir / ".cache" / f"standardized_{config.dataset_path.stem}.h5ad"
    if standardized.exists() and not config.force_restandardize:
        print(f"{standardized.name} already exists, nothing to do")
        return
    crispyx.data.shutil.copy2 = _verified_copy
    path = crispyx.data.standardize_dataset(
        dataset_path=config.dataset_path,
        perturbation_column=config.perturbation_column,
        control_label=config.control_label,
        gene_name_column=config.gene_name_column,
        output_dir=config.output_dir,
        force=config.force_restandardize,
    )
    print(f"Standardized: {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
