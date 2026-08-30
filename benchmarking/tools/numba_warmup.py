"""Numba JIT warm-up script — executed during Docker image build.

Triggers JIT compilation of all Wilcoxon Numba kernels for both float32
(native h5ad dtype after O2 dtype alignment) and float64 paths, then
caches the compiled artifacts inside the image layer.  This eliminates
the ~25-minute cold-start compilation on the first benchmark run.
"""

import sys
import os

# When run from the Dockerfile, crispyx is installed as an editable package
# under /workspace/src; ensure it is importable.
sys.path.insert(0, "/workspace/src")

import numpy as np
from crispyx._kernels import (
    _presort_control_nonzeros,
    _compute_ctrl_tie_sums,
    _wilcoxon_batch_perts_presorted_numba,
    _wilcoxon_single_pert_presorted,
    _ZERO_PARTITION_THRESHOLD,
)

rng = np.random.default_rng(42)

for dt in (np.float32, np.float64):
    ctrl = rng.exponential(1.0, (100, 32)).astype(dt)
    ctrl[ctrl < 0.5] = 0.0
    flat, off, nnz, nz = _presort_control_nonzeros(ctrl)
    tie_sums = _compute_ctrl_tie_sums(flat, off, nnz)

    stk = rng.exponential(0.5, (150, 32)).astype(dt)
    stk[stk < 0.5] = 0.0
    offsets = np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)
    vm = np.ones((6, 32), dtype=np.bool_)
    u = np.zeros((6, 32), dtype=np.float64)
    z = np.zeros((6, 32), dtype=np.float64)
    p = np.ones((6, 32), dtype=np.float64)
    e = np.zeros((6, 32), dtype=np.float64)

    _wilcoxon_batch_perts_presorted_numba(
        ctrl, flat, off, nnz, nz, tie_sums, stk, offsets, vm,
        True, _ZERO_PARTITION_THRESHOLD, u, z, p, e,
    )

print("Numba JIT cache warm-up complete (float32 + float64).")
