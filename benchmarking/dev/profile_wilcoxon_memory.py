"""Profile memory usage at each key step of wilcoxon_test for Feng-gwsf.

Run with: docker run ... python3.11 /workspace/benchmarking/dev/profile_wilcoxon_memory.py
"""
import gc
import sys
import time
import os
import psutil

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/workspace")

proc = psutil.Process()


def rss():
    return proc.memory_info().rss / 1e9


def report(label):
    gb = rss()
    print(f"[MEM] {label}: {gb:.2f} GB RSS", flush=True)


# Report Numba thread count
n_threads = int(os.environ.get("NUMBA_NUM_THREADS", 8))
print(f"[INFO] NUMBA_NUM_THREADS={n_threads}", flush=True)


print(f"[INFO] Python {sys.version}", flush=True)
report("start")

import numpy as np
import scipy.sparse as sp
import anndata as ad

report("imports done")

import numba as nb
print(f"[INFO] Numba {nb.__version__}", flush=True)

# Import kernels to trigger JIT compilation
from crispyx._kernels import _presort_control_nonzeros, _wilcoxon_presorted_ctrl_numba
# Warm up JIT with tiny data
_ctrl_warm = np.zeros((10, 5), dtype=np.float64)
_flat, _off, _nz, _z0 = _presort_control_nonzeros(_ctrl_warm)
_pert_warm = np.zeros((3, 5), dtype=np.float64)
_u = np.zeros(5); _zz = np.zeros(5); _p = np.ones(5); _e = np.zeros(5)
_vg = np.ones(5, dtype=bool)
_wilcoxon_presorted_ctrl_numba(_ctrl_warm, _flat, _off, _nz, _z0, _pert_warm, _vg, True, 0.0, _u, _zz, _p, _e)
report("Numba JIT warm-up done")

DATA = "/workspace/benchmarking/results/Feng-gwsf/preprocessing/preprocessed_standardized_Feng-gwsf.h5ad"

backed = ad.read_h5ad(DATA, backed="r")
report("backed open")
n_obs, n_vars = backed.n_obs, backed.n_vars
labels = backed.obs["perturbation"].astype(str).to_numpy()
control_label = "control"
candidates = sorted([x for x in np.unique(labels) if x != control_label])
control_mask = labels == control_label
control_idx = np.where(control_mask)[0]
pert_idx = {label: np.where(labels == label)[0] for label in candidates}
n_groups = len(candidates)
print(f"[INFO] n_obs={n_obs}, n_vars={n_vars}, n_groups={n_groups}", flush=True)
report("obs loaded")

import tempfile
from pathlib import Path

tmpdir = tempfile.mkdtemp()
tmpdir_path = Path(tmpdir)


def _create_memmap(name, dtype, fill=0.0):
    path = tmpdir_path / f"{name}.dat"
    mmap = np.memmap(path, dtype=dtype, mode="w+", shape=(n_groups, n_vars))
    if fill != 0:
        mmap[:] = fill
    else:
        mmap.fill(0)
    return mmap


report("before memmaps")
effect_matrix = _create_memmap("effect", np.float64)
report("after effect_matrix")
u_matrix = _create_memmap("u_stat", np.float64)
report("after u_matrix")
pvalue_matrix = _create_memmap("pvalue", np.float64, fill=1.0)
report("after pvalue_matrix (fill=1.0)")
z_matrix = _create_memmap("z_score", np.float64)
lfc_matrix = _create_memmap("logfoldchange", np.float64)
pts_matrix = _create_memmap("pts", np.float32)
pts_rest_matrix = _create_memmap("pts_rest", np.float32)
order_matrix = np.memmap(tmpdir_path / "order.dat", dtype=np.int64, mode="w+", shape=(n_groups, n_vars))
report("all memmaps created")

chunk_size = 128
chunk_count = 0
from crispyx.data import iter_matrix_chunks

for slc, block in iter_matrix_chunks(backed, axis=1, chunk_size=chunk_size, convert_to_dense=False):
    chunk_count += 1
    if chunk_count % 20 == 1 or chunk_count <= 3:
        report(f"chunk {chunk_count} START (genes {slc.start}-{slc.stop})")

    csr_block = sp.csr_matrix(block, dtype=np.float64)

    if chunk_count <= 3:
        report(f"chunk {chunk_count} after csr_block")

    # Control stats
    control_values = csr_block[control_mask, :]
    control_expr = np.asarray(control_values.getnnz(axis=0)).ravel()

    # Pre-compute pert stats (triggers 2254 sparse slices)
    pert_expr_counts, pert_means, pert_n_cells = [], [], []
    for lbl in candidates:
        gv = csr_block[pert_idx[lbl], :]
        pert_n_cells.append(gv.shape[0])
        pert_expr_counts.append(np.asarray(gv.getnnz(axis=0)).ravel())
        pert_means.append(np.asarray(gv.mean(axis=0)).ravel() if gv.nnz else np.zeros(csr_block.shape[1]))

    if chunk_count <= 3:
        report(f"chunk {chunk_count} after pert loop")

    # Union valid genes
    any_valid = np.zeros(csr_block.shape[1], dtype=bool)
    for idx, lbl in enumerate(candidates):
        total_expr = control_expr + pert_expr_counts[idx]
        low_expr = (control_expr < 0) & (pert_expr_counts[idx] < 0)
        valid = (total_expr >= 0) & ~low_expr
        any_valid |= valid
    valid_gene_indices = np.where(any_valid)[0]
    n_valid = len(valid_gene_indices)

    # Dense conversion step
    all_valid_dense = csr_block[:, valid_gene_indices].toarray().astype(np.float64)
    if chunk_count <= 3:
        report(f"chunk {chunk_count} after all_valid_dense ({all_valid_dense.nbytes/1e9:.2f} GB)")

    control_dense = all_valid_dense[control_idx, :]
    if chunk_count <= 3:
        report(f"chunk {chunk_count} after control_dense ({control_dense.nbytes/1e9:.2f} GB, shape={control_dense.shape})")

    # Pre-sort control (this is the step that crashed before kernel call)
    ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz, ctrl_n_z = _presort_control_nonzeros(control_dense)
    if chunk_count <= 3:
        report(f"chunk {chunk_count} after _presort_control_nonzeros")

    # Run kernel for first perturbation group only (to measure peak)
    lbl0 = candidates[0]
    pert_dense = all_valid_dense[pert_idx[lbl0], :]
    valid_genes_arr = np.ones(n_valid, dtype=bool)  # all valid for profiling
    u_out = np.zeros(n_valid, dtype=np.float64)
    z_out = np.zeros(n_valid, dtype=np.float64)
    p_out = np.ones(n_valid, dtype=np.float64)
    eff_out = np.zeros(n_valid, dtype=np.float64)
    _wilcoxon_presorted_ctrl_numba(
        control_dense, ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz, ctrl_n_z,
        pert_dense, valid_genes_arr, True, 0.0, u_out, z_out, p_out, eff_out,
    )
    if chunk_count <= 3:
        report(f"chunk {chunk_count} after kernel (1 pert)")

    # Free big allocs
    del all_valid_dense, control_dense, csr_block, block, control_values
    del ctrl_sorted_flat, ctrl_offsets, ctrl_n_nz, ctrl_n_z
    del pert_dense, u_out, z_out, p_out, eff_out
    del pert_expr_counts, pert_means, pert_n_cells, any_valid
    gc.collect()

    if chunk_count <= 3:
        report(f"chunk {chunk_count} after gc")

    if chunk_count == 10:
        report("=== AFTER 10 CHUNKS ===")
        break

report("loop done")
backed.file.close()
