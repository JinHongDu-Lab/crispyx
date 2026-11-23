"""Dimension reduction module for crispyx.

Streaming PCA, and neighbor and UMAP computation on the PCA embedding,
following Scanpy-style API patterns.

PCA's peak memory is :func:`_pca_peak_bytes`; every step's is in the docs'
memory model (docs/large_data.rst, ``memory-model``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from anndata.io import sparse_dataset
from scipy import sparse
from scipy.linalg import eigh
from scipy.linalg.blas import dsyrk
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import anndata as AnnData

from ._memory import _resolve_memory_limit_bytes
from .data import (
    iter_matrix_chunks,
    write_obsm_to_h5ad,
    write_varm_to_h5ad,
    write_uns_dict_to_h5ad,
    write_obsp_to_h5ad,
    AnnData as CrispyxAnnData,
)

logger = logging.getLogger(__name__)

PCA_METHODS = ("covariance", "incremental")

#: Share of the memory budget PCA plans to use: the process holds other
#: things, and the formulas below are fitted, not exact.
_PCA_BUDGET_FRACTION = 0.5
#: Fixed peak above the pre-call size, measured: HDF5 buffers and read blocks.
_PCA_OVERHEAD_BYTES = 64e6


def _pca_peak_bytes(
    method: str, n_vars: int, n_comps: int, chunk_size: int, n_obs_in_memory: int = 0,
    read_row_bytes: float = 0.0,
) -> float:
    """Peak bytes PCA holds above the process's size before the call: the
    method's fixed state, one chunk's working set, a fixed overhead (HDF5
    buffers, read blocks) and, when they are returned rather than written to
    the file, the scores.

    Fitted to peak resident memory measured on a Linux server, 50,000 and
    200,000 cells x 2,000-10,000 genes, chunks of 512 and 2,048: the formula
    is at most 6% below a measurement and up to 45% above (most at 2,000
    genes, 512-cell chunks, where the fixed overhead dominates);
    docs/large_data.rst, ``memory-model``.
    The covariance working set is the read block and its float64 dense copy;
    the incremental one adds IncrementalPCA's stacked matrix and SVD.
    ``read_row_bytes`` is one input row at full width, read before the genes
    are subset (pass it when PCA uses a subset of the genes).
    """
    scores = 4.0 * n_obs_in_memory * n_comps + read_row_bytes * chunk_size
    if method == "covariance":
        return 8.0 * n_vars * n_vars + 16.0 * chunk_size * n_vars + _PCA_OVERHEAD_BYTES + scores
    rows = chunk_size + n_comps
    return 50.0 * rows * n_vars + 32.0 * rows * rows + _PCA_OVERHEAD_BYTES + scores


def _pca_chunk_size(
    n_vars: int,
    n_comps: int = 50,
    *,
    n_obs_in_memory: int = 0,
    read_row_bytes: float = 0.0,
    method: str = "auto",
    memory_limit_gb: float | None = None,
    min_chunk: int = 256,
    max_chunk: int = 4096,
) -> tuple[int, str]:
    """Chunk size and method for :func:`pca` within ``memory_limit_gb``.

    ``auto`` takes the exact covariance method whenever its peak at
    ``min_chunk`` fits in half the budget, and IncrementalPCA (approximate
    and slower) only when it does not. The chunk is
    ``max_chunk`` unless its peak (:func:`_pca_peak_bytes`) exceeds half the
    budget, then the largest that fits: on any machine with the memory, the
    same file gets the same chunk, and so the same IncrementalPCA result. A forced
    ``method="covariance"`` whose matrix does not fit the whole budget raises
    :class:`MemoryError` rather than being killed partway.
    """
    if method not in ("auto", *PCA_METHODS):
        raise ValueError(f"method must be 'auto', 'covariance' or 'incremental', got {method!r}")
    budget = _resolve_memory_limit_bytes(memory_limit_gb)
    usable = _PCA_BUDGET_FRACTION * budget
    covariance_bytes = 8.0 * n_vars * n_vars

    def peak(chunk: int, method: str) -> float:
        return _pca_peak_bytes(method, n_vars, n_comps, chunk, n_obs_in_memory, read_row_bytes)

    if method == "auto":
        method = "covariance" if peak(min_chunk, "covariance") <= usable else "incremental"

    if method == "covariance" and peak(min_chunk, method) > budget:
        raise MemoryError(
            f"PCA with method='covariance' on {n_vars} genes needs {peak(min_chunk, method) / 1e9:.1f} GB "
            f"(a {covariance_bytes / 1e9:.1f} GB gene x gene matrix), over the "
            f"{budget / 1e9:.1f} GB memory budget. Use method='incremental', or restrict PCA to "
            "highly variable genes (cx.pp.highly_variable_genes, use_highly_variable=True)."
        )
    # The peak grows monotonically with the chunk: bisect for the largest that fits.
    lo, hi = min_chunk, max_chunk
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if peak(mid, method) <= usable else (lo, mid - 1)
    chunk = lo
    if chunk < max_chunk:
        logger.info(f"PCA: chunk lowered to {chunk} to fit the memory budget")
    logger.info(
        f"PCA: method={method}, chunk={chunk}, peak ~{peak(chunk, method) / 1e9:.2f} GB "
        f"of a {budget / 1e9:.1f} GB budget ({n_vars} genes, {n_comps} components)"
    )
    return chunk, method


def _pca_gene_indices(var, use_highly_variable: bool) -> np.ndarray | None:
    """Columns entering PCA: the highly variable genes when asked and marked,
    else all (``None``)."""
    if use_highly_variable and "highly_variable" in var.columns:
        gene_indices = np.flatnonzero(np.asarray(var["highly_variable"], dtype=bool))
        if gene_indices.size == 0:
            raise ValueError(
                "var['highly_variable'] marks no genes: rerun cx.pp.highly_variable_genes, "
                "or pass use_highly_variable=False to use every gene."
            )
        logger.info(f"Using {gene_indices.size} highly variable genes for PCA")
        return gene_indices
    logger.info(f"Using all {len(var)} genes for PCA")
    return None


def _row_blocks(
    matrix, chunk_size: int, gene_indices: np.ndarray | None, desc: str, show_progress: bool,
) -> Iterator[tuple[slice, np.ndarray | sparse.spmatrix]]:
    """Row chunks of ``matrix`` restricted to ``gene_indices``, as read:
    sparse stays sparse, and a dense chunk may be a view of the input."""
    with tqdm(total=matrix.shape[0], desc=desc, disable=not show_progress, unit="cells") as bar:
        for slc, block in iter_matrix_chunks(None, matrix=matrix, chunk_size=chunk_size, convert_to_dense=False):
            if gene_indices is not None:
                block = block[:, gene_indices]
            bar.update(slc.stop - slc.start)
            yield slc, block


def _dense_blocks(
    matrix, chunk_size: int, gene_indices: np.ndarray | None, *, min_rows: int = 1,
    desc: str = "", show_progress: bool = True,
) -> Iterator[tuple[slice, np.ndarray]]:
    """:func:`_row_blocks` as fresh Fortran-ordered float64 arrays, which BLAS
    ``dsyrk`` and LAPACK take without a copy and the caller may change in
    place. A final chunk shorter than ``min_rows`` is merged into the one
    before it, so IncrementalPCA sees every cell."""
    n_obs = matrix.shape[0]
    tail = n_obs % chunk_size
    # Only a short tail is held back for: the chunk before it waits to be joined.
    hold_start = n_obs - tail - chunk_size if 0 < tail < min_rows and n_obs > chunk_size else -1
    held: tuple[slice, np.ndarray] | None = None
    for slc, block in _row_blocks(matrix, chunk_size, gene_indices, desc, show_progress):
        if sparse.issparse(block):
            block = block.astype(np.float64).toarray(order="F")  # casts the non-zeros only
        else:
            block = np.array(block, dtype=np.float64, order="F", copy=True)
        if slc.start == hold_start:
            held = (slc, block)
        elif held is not None:
            yield slice(held[0].start, slc.stop), np.asfortranarray(np.vstack([held[1], block]))
            held = None
        else:
            yield slc, block


def _fit_covariance(blocks, n_obs: int, n_vars: int, n_comps: int) -> dict:
    """Exact PCA from the gene x gene covariance, accumulated in place.

    Each chunk is shifted by the first chunk's column means, so ``E[x²] -
    E[x]²`` does not cancel for genes whose mean dwarfs their spread, and
    ``dsyrk`` adds its ``Xᵀ X`` into the upper triangle of one Fortran-ordered
    float64 matrix. That is centred on the remaining mean shift in column
    blocks and handed to ``eigh`` with ``overwrite_a``: one g x g array in all.
    """
    xtx = np.zeros((n_vars, n_vars), dtype=np.float64, order="F")
    sums = np.zeros(n_vars, dtype=np.float64)
    shift = None
    for _, block in blocks:
        if shift is None:
            shift = block.mean(axis=0)
        block -= shift
        sums += block.sum(axis=0)
        xtx = dsyrk(1.0, block, beta=1.0, c=xtx, trans=1, lower=0, overwrite_c=1)
    offset = sums / n_obs
    xtx /= n_obs
    step = max(1, (1 << 21) // max(n_vars, 1))  # 16 MB of outer product at a time
    for j in range(0, n_vars, step):
        xtx[:, j:j + step] -= np.outer(offset, offset[j:j + step])
    total_variance = float(np.trace(xtx))
    eigenvalues, eigenvectors = eigh(
        xtx, lower=False, overwrite_a=True, check_finite=False,
        subset_by_index=[n_vars - n_comps, n_vars - 1],
    )
    del xtx
    eigenvalues = eigenvalues[::-1]
    return {
        "mean": shift + offset,
        "components": np.ascontiguousarray(eigenvectors[:, ::-1].T),
        # Sample variance (n - 1), as IncrementalPCA, scikit-learn and Scanpy report.
        "variance": eigenvalues * (n_obs / max(n_obs - 1, 1)),
        "variance_ratio": eigenvalues / total_variance,
    }


def _fit_incremental(blocks, n_comps: int) -> dict:
    """Approximate PCA with scikit-learn's IncrementalPCA, a chunk at a time."""
    ipca = IncrementalPCA(n_components=n_comps, copy=False)  # each block is a fresh copy
    for _, block in blocks:
        ipca.partial_fit(block)
    return {
        "mean": ipca.mean_,
        "components": ipca.components_,
        "variance": ipca.explained_variance_,
        "variance_ratio": ipca.explained_variance_ratio_,
        "noise_variance": ipca.noise_variance_,
    }


def _fit_pca(
    matrix, method: str, n_comps: int, chunk_size: int, gene_indices: np.ndarray | None,
    show_progress: bool,
) -> dict:
    n_obs = matrix.shape[0]
    n_vars = matrix.shape[1] if gene_indices is None else gene_indices.size
    if method == "covariance":
        blocks = _dense_blocks(matrix, chunk_size, gene_indices, desc="Computing covariance", show_progress=show_progress)
        fit = _fit_covariance(blocks, n_obs, n_vars, n_comps)
    else:
        blocks = _dense_blocks(
            matrix, max(chunk_size, n_comps), gene_indices, min_rows=n_comps,
            desc="Learning PCA", show_progress=show_progress,
        )
        fit = _fit_incremental(blocks, n_comps)
    # scikit-learn's sign convention (svd_flip on the components): each
    # component's largest loading is positive, whatever sign LAPACK returned.
    components = fit["components"]
    signs = np.sign(components[np.arange(components.shape[0]), np.argmax(np.abs(components), axis=1)])
    signs[signs == 0] = 1.0
    fit["components"] = components * signs[:, None]
    return fit


def _project(matrix, fit: dict, chunk_size: int, gene_indices: np.ndarray | None, out, show_progress: bool) -> None:
    """Write each chunk's scores ``(x - mean) @ componentsᵀ`` into ``out``, an
    array or an HDF5 dataset of shape (n_obs, n_comps), as
    ``x @ componentsᵀ - mean @ componentsᵀ``: a sparse chunk is multiplied as
    it is read, and the input is never changed."""
    components_t = np.ascontiguousarray(fit["components"].T)
    centre = fit["mean"] @ components_t
    for slc, block in _row_blocks(matrix, chunk_size, gene_indices, "Transforming data", show_progress):
        scores = np.asarray(block @ components_t, dtype=np.float64)
        scores -= centre
        out[slc] = scores.astype(np.float32)


def _full_pcs(fit: dict, gene_indices: np.ndarray | None, n_vars_total: int) -> np.ndarray:
    """``varm['PCs']``: the components over every gene, zero outside those used."""
    pcs = np.zeros((n_vars_total, fit["components"].shape[0]), dtype=np.float32)
    pcs[slice(None) if gene_indices is None else gene_indices, :] = fit["components"].T
    return pcs


def _pca_uns(fit: dict, use_highly_variable: bool, method: str, chunk: int) -> dict:
    return {
        "variance": fit["variance"],
        "variance_ratio": fit["variance_ratio"],
        "use_highly_variable": use_highly_variable,
        "method": method,
        "n_comps": int(fit["components"].shape[0]),
        "chunk_size": int(chunk),
    }


def _pca_scratch_path(path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}.X_pca.partial")


def _check_writable(path) -> None:
    """Fail before the passes, not after them, when the results could not be
    written: HDF5 opens a file for writing only if no other handle -- in this
    process or another -- has it open."""
    try:
        h5py.File(path, "r+").close()
    except OSError as err:
        raise OSError(
            f"{path} is open elsewhere (another AnnData object or process holds it), so PCA "
            "could not write its results; close it, or pass copy=True to get them in memory."
        ) from err


def _open_x(f: h5py.File):
    elem = f["X"]
    return sparse_dataset(elem) if isinstance(elem, h5py.Group) else elem


def _input_row_bytes(matrix) -> float:
    """Bytes of one stored row at full width: non-zeros x (value + index) for
    sparse, the row itself for dense."""
    n_obs, n_cols = matrix.shape
    if sparse.issparse(matrix):
        return matrix.nnz / max(n_obs, 1) * (matrix.data.itemsize + matrix.indices.itemsize)
    group = getattr(matrix, "group", None)
    if group is not None:  # anndata's backed sparse dataset
        data, indices = group["data"], group["indices"]
        return data.shape[0] / max(n_obs, 1) * (data.dtype.itemsize + indices.dtype.itemsize)
    return float(n_cols * np.dtype(matrix.dtype).itemsize)


def pca(
    adata: "AnnData",
    n_comps: int = 50,
    method: str = "auto",
    use_highly_variable: bool = True,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    random_state: int = 0,
    copy: bool = False,
    show_progress: bool = True,
) -> "AnnData" | None:
    """Compute Principal Component Analysis (PCA), streaming the cells from disk.

    On a crispyx backed file (:class:`crispyx.AnnData`, from
    ``cx.read_h5ad_ondisk`` or a path through ``cx.pp.pca``) the scores are
    written to ``obsm['X_pca']`` a chunk at a time, so memory does not grow
    with the number of cells. See the module docstring for each method's peak.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_comps
        Number of principal components to compute. Default 50.
    method
        - ``'auto'``: ``'covariance'`` whenever it fits in half the memory
          budget, ``'incremental'`` only when it does not. The choice depends
          on the budget; pass the method to fix it across machines.
        - ``'covariance'``: exact PCA from the gene x gene covariance (one
          ``8 g²``-byte matrix, g = genes entering PCA). Raises
          :class:`MemoryError` up front when that does not fit.
        - ``'incremental'``: scikit-learn's IncrementalPCA, approximate,
          memory linear in the genes.
    use_highly_variable
        If True and 'highly_variable' exists in var, restrict to HVGs.
        Default True. Without HVGs, PCA runs on every gene.
    chunk_size
        Number of cells to process per chunk. If None, 4096, or the largest
        chunk whose peak fits in half the memory budget when that is smaller
        (logged; the chunk the fit used, at least ``n_comps`` for
        IncrementalPCA, is stored in ``uns['pca']['chunk_size']``). The covariance
        result does not depend on it; IncrementalPCA's does, slightly.
    memory_limit_gb
        Memory budget in GB. ``None`` detects the memory available, capped
        by the job's allocation when known (pass the job's ``--mem`` on a
        cluster that does not expose it).
    random_state
        Not used: neither method is randomized. Kept for Scanpy's signature.
    copy
        If True, load the data into memory and return a copy with the PCA
        results; otherwise modify ``adata`` (or its file) in place.
    show_progress
        Show progress bars during computation. Default True.

    Returns
    -------
    adata : AnnData | None
        If copy=True, returns modified AnnData. Otherwise modifies in place.

    Modifies adata
    --------------
    obsm['X_pca']
        PCA-transformed data (n_obs × n_comps).
    varm['PCs']
        Principal components (n_vars × n_comps); zero outside the genes used.
    uns['pca']
        Dict with 'variance' (sample variance of each component),
        'variance_ratio', 'use_highly_variable', 'method', 'n_comps' and
        'chunk_size'. Components follow scikit-learn's sign convention: each
        one's largest loading is positive.

    Examples
    --------
    >>> import crispyx as cx
    >>> adata = cx.read_h5ad_ondisk("data.h5ad")
    >>> cx.pp.pca(adata, n_comps=50)
    """
    stream = isinstance(adata, CrispyxAnnData) and not copy
    if copy:
        is_backed = isinstance(adata, CrispyxAnnData) or getattr(adata, "isbacked", False)
        adata = adata.to_memory() if is_backed else adata.copy()

    var = adata.var
    gene_indices = _pca_gene_indices(var, use_highly_variable)
    n_vars_total = len(var)
    n_vars = n_vars_total if gene_indices is None else gene_indices.size

    if stream:
        # Read-only for the passes (other readers may share the file, and a
        # standardized file's X is an external link to its source); opened
        # for writing only to store the results.
        path = adata.path
        adata.close()
        _check_writable(path)
        f = h5py.File(path, "r")
        matrix = _open_x(f)
    else:
        f = None
        matrix = adata.X
    try:
        n_obs = matrix.shape[0]
        n_comps = min(n_comps, n_vars, n_obs)
        planned_chunk, selected_method = _pca_chunk_size(
            n_vars, n_comps, n_obs_in_memory=0 if stream else n_obs,
            read_row_bytes=0.0 if gene_indices is None else _input_row_bytes(matrix),
            method=method, memory_limit_gb=memory_limit_gb,
        )
        if chunk_size is None:
            chunk_size = planned_chunk
        else:
            logger.info(f"PCA: chunk_size={chunk_size} as given")
        fit = _fit_pca(matrix, selected_method, n_comps, chunk_size, gene_indices, show_progress)
        fit_chunk = max(chunk_size, n_comps) if selected_method == "incremental" else chunk_size

        if stream:
            # Scores go to a scratch file beside the output (n x k, small) and
            # are copied in, compressed chunks as they are, once complete: a
            # run killed partway leaves the output as it was.
            scratch = _pca_scratch_path(path)
            scratch.unlink(missing_ok=True)
            try:
                with h5py.File(scratch, "w") as tmp:
                    out = tmp.create_dataset(
                        "X_pca", shape=(n_obs, n_comps), dtype=np.float32,
                        chunks=(min(chunk_size, n_obs), n_comps), compression="gzip", compression_opts=4,
                    )
                    _project(matrix, fit, chunk_size, gene_indices, out, show_progress)
                    out.attrs["encoding-type"] = "array"
                    out.attrs["encoding-version"] = "0.2.0"
                f.close()
                # One write session, after every pass: X_pca, PCs and uns together.
                with h5py.File(path, "r+") as dst, h5py.File(scratch, "r") as tmp:
                    obsm = dst.require_group("obsm")
                    if "X_pca" in obsm:
                        del obsm["X_pca"]
                    obsm.copy(tmp["X_pca"], "X_pca")
                    write_varm_to_h5ad(dst, "PCs", _full_pcs(fit, gene_indices, n_vars_total))
                    write_uns_dict_to_h5ad(dst, "pca", _pca_uns(fit, use_highly_variable, selected_method, fit_chunk))
            finally:
                scratch.unlink(missing_ok=True)
        else:
            X_pca = np.empty((n_obs, n_comps), dtype=np.float32)
            _project(matrix, fit, chunk_size, gene_indices, X_pca, show_progress)
    finally:
        if f is not None and f.id.valid:
            f.close()

    explained = f"{fit['variance_ratio'].sum():.2%}"
    if stream:
        logger.info(f"PCA complete: {n_comps} components, variance explained: {explained} (written to {path})")
        return None
    adata.obsm["X_pca"] = X_pca
    adata.varm["PCs"] = _full_pcs(fit, gene_indices, n_vars_total)
    adata.uns["pca"] = _pca_uns(fit, use_highly_variable, selected_method, fit_chunk)
    logger.info(f"PCA complete: {n_comps} components, variance explained: {explained}")
    return adata if copy else None


def _compute_connectivities_umap(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_obs: int,
    n_neighbors: int,
) -> sparse.csr_matrix:
    """Compute UMAP-style connectivities from KNN graph.
    
    Follows the UMAP fuzzy simplicial set construction.
    """
    from scipy.sparse import coo_matrix
    
    # Simple UMAP-style connectivities: 1 / (1 + distance)
    # More sophisticated version would use local connectivity
    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = knn_indices.ravel()
    
    # Avoid division by zero
    dists = knn_distances.ravel()
    dists = np.maximum(dists, 1e-10)
    
    # Simple connectivity: exponential decay
    # sigma = local bandwidth (use mean of k-th neighbor distance)
    sigma = np.mean(knn_distances[:, -1])
    sigma = max(sigma, 1e-10)
    
    data = np.exp(-dists / sigma)
    
    connectivities = coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    connectivities = connectivities.tocsr()
    
    # Symmetrize: (A + A.T) / 2
    connectivities = (connectivities + connectivities.T) / 2
    
    return connectivities


def _compute_distances_sparse(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_obs: int,
    n_neighbors: int,
) -> sparse.csr_matrix:
    """Convert KNN indices/distances to sparse distance matrix."""
    from scipy.sparse import coo_matrix
    
    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = knn_indices.ravel()
    data = knn_distances.ravel()
    
    distances = coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    return distances.tocsr()


def neighbors(
    adata: "AnnData",
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    method: str = "umap",
    random_state: int = 0,
    copy: bool = False,
    show_progress: bool = True,
) -> "AnnData" | None:
    """Compute k-nearest neighbors graph from embeddings.
    
    Uses pre-computed embeddings (typically PCA) to build a KNN graph.
    The embeddings are loaded into memory for efficient distance computation.
    
    For backed data (crispyx.AnnData wrapper), results are written directly to
    the h5ad file using a close-write-reopen pattern, keeping .X on disk.
    
    Parameters
    ----------
    adata
        The annotated data matrix with embeddings in .obsm.
    n_neighbors
        Number of neighbors in the KNN graph. Default 15.
    n_pcs
        Number of PCs to use from the embedding. If None, uses all.
    use_rep
        Key in .obsm to use for distance computation. Default 'X_pca'.
    metric
        Distance metric. Default 'euclidean'. Supports 'euclidean',
        'cosine', 'manhattan', etc.
    method
        KNN algorithm: 'umap' (uses pynndescent, fast approximate) or
        'sklearn' (exact but slower). Default 'umap'.
    random_state
        Random seed for reproducibility.
    copy
        If True, return a copy with results. Otherwise modify in place.
    show_progress
        Show progress information. Default True.
    
    Returns
    -------
    adata : AnnData | None
        If copy=True, returns modified AnnData. Otherwise modifies in place.
    
    Modifies adata
    --------------
    obsp['distances']
        Sparse distance matrix (n_obs × n_obs).
    obsp['connectivities']
        Sparse connectivity matrix (n_obs × n_obs).
    uns['neighbors']
        Dict with parameters: n_neighbors, method, metric, use_rep.
    
    Examples
    --------
    >>> import crispyx as cx
    >>> adata = cx.read_backed("data.h5ad")
    >>> cx.pp.pca(adata, n_comps=50)
    >>> cx.pp.neighbors(adata, n_neighbors=15)
    >>> adata.obsp['connectivities']
    <sparse matrix (n_obs, n_obs)>
    """
    # Detect if this is a crispyx.AnnData wrapper (backed, with .path)
    is_crispyx_wrapper = isinstance(adata, CrispyxAnnData)
    
    if copy:
        if is_crispyx_wrapper:
            adata = adata.to_memory()
        elif hasattr(adata, 'file') and adata.file is not None:
            adata = adata.to_memory()
        else:
            adata = adata.copy()
    
    # Get embeddings
    if use_rep not in adata.obsm:
        raise ValueError(
            f"'{use_rep}' not found in adata.obsm. "
            f"Run cx.pp.pca() first or specify a valid use_rep. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    X = adata.obsm[use_rep]
    
    # Load to memory if needed (backed obsm)
    if hasattr(X, 'to_memory'):
        X = X.to_memory()
    X = np.asarray(X)
    
    # Subset PCs if requested
    if n_pcs is not None and n_pcs < X.shape[1]:
        X = X[:, :n_pcs]
        logger.info(f"Using first {n_pcs} components from {use_rep}")
    
    n_obs, n_dims = X.shape
    logger.info(
        f"Computing {n_neighbors}-NN graph on {n_obs} cells × {n_dims} dims "
        f"using method='{method}'"
    )
    
    # Compute KNN
    if method == "umap":
        try:
            from pynndescent import NNDescent
        except ImportError:
            raise ImportError(
                "pynndescent is required for method='umap'. "
                "Install with: pip install pynndescent"
            )
        
        # Build index and query
        index = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            verbose=show_progress,
        )
        knn_indices, knn_distances = index.neighbor_graph
        
    elif method == "sklearn":
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm="auto",
        )
        nn.fit(X)
        knn_distances, knn_indices = nn.kneighbors(X)
        
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'umap' or 'sklearn'.")
    
    # Build sparse matrices
    distances = _compute_distances_sparse(knn_indices, knn_distances, n_obs, n_neighbors)
    connectivities = _compute_connectivities_umap(knn_indices, knn_distances, n_obs, n_neighbors)
    
    # Prepare uns dict
    neighbors_uns = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": n_neighbors,
            "method": method,
            "metric": metric,
            "use_rep": use_rep,
            "n_pcs": n_pcs if n_pcs is not None else X.shape[1],
        },
    }
    
    # Store results: use close-write-reopen for crispyx wrapper, direct for in-memory
    if is_crispyx_wrapper and not copy:
        # Close file handle
        path = adata.path
        adata.close()
        
        # Write results to h5ad file
        write_obsp_to_h5ad(path, "distances", distances)
        write_obsp_to_h5ad(path, "connectivities", connectivities)
        write_uns_dict_to_h5ad(path, "neighbors", neighbors_uns)
        
        # File will be reopened lazily on next access
        logger.info(
            f"Neighbors complete: {n_neighbors} neighbors, "
            f"{connectivities.nnz} connections (written to {path})"
        )
    else:
        # In-memory AnnData: store directly
        adata.obsp["distances"] = distances
        adata.obsp["connectivities"] = connectivities
        adata.uns["neighbors"] = neighbors_uns
        
        logger.info(
            f"Neighbors complete: {n_neighbors} neighbors, "
            f"{connectivities.nnz} connections"
        )
    
    if copy:
        return adata
    return None


def umap(
    adata: "AnnData",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    neighbors_key: str = "neighbors",
    random_state: int = 0,
    copy: bool = False,
) -> "AnnData" | None:
    """Compute UMAP embedding from pre-computed neighbor graph.
    
    Uses the neighbor graph stored in adata.obsp (computed by cx.pp.neighbors)
    to create a 2D UMAP embedding. This is memory-efficient because only the
    neighbor graph and embedding need to be in memory, not the full expression
    matrix.
    
    For backed data (crispyx.AnnData wrapper), results are written directly to
    the h5ad file using a close-write-reopen pattern, keeping .X on disk.
    
    Parameters
    ----------
    adata
        The annotated data matrix with a neighbor graph computed.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        result in a more clustered embedding. Default 0.5.
    spread
        The effective scale of embedded points. In combination with min_dist
        this determines how clustered/clumped the embedding is. Default 1.0.
    n_components
        Number of UMAP dimensions. Default 2.
    neighbors_key
        Key in .uns where neighbor information is stored. Default 'neighbors'.
    random_state
        Random seed for reproducibility.
    copy
        If True, return a copy with results. Otherwise modify in place.
    
    Returns
    -------
    adata : AnnData | None
        If copy=True, returns modified AnnData. Otherwise modifies in place.
    
    Modifies adata
    --------------
    obsm['X_umap']
        UMAP embedding (n_obs × n_components).
    uns['umap']
        Dict with UMAP parameters.
    
    Examples
    --------
    >>> import crispyx as cx
    >>> adata = cx.read_backed("data.h5ad")
    >>> cx.pp.pca(adata, n_comps=50)
    >>> cx.pp.neighbors(adata, n_neighbors=15)
    >>> cx.tl.umap(adata)
    >>> adata.obsm['X_umap'].shape
    (n_obs, 2)
    
    Notes
    -----
    This function wraps scanpy.tl.umap. The neighbor graph must be computed
    first using cx.pp.neighbors(). Memory requirements scale linearly with
    the number of cells: approximately 0.75MB per 1000 cells for 15 neighbors.
    
    See Also
    --------
    cx.pp.neighbors : Compute k-nearest neighbors graph.
    cx.pl.umap : Plot UMAP embedding.
    """
    import scanpy as sc
    
    # Detect if this is a crispyx.AnnData wrapper (backed, with .path)
    is_crispyx_wrapper = isinstance(adata, CrispyxAnnData)
    
    # Check for neighbor graph
    if neighbors_key not in adata.uns:
        raise ValueError(
            f"'{neighbors_key}' not found in adata.uns. "
            f"Run cx.pp.neighbors() first."
        )
    
    # For backed data, we need to load neighbors into memory
    if is_crispyx_wrapper and not copy:
        path = adata.path
        n_obs = adata.n_obs
        
        # Load neighbor graph components into memory
        connectivities = adata.obsp["connectivities"]
        if sparse.issparse(connectivities):
            connectivities = connectivities.tocsr()
        else:
            connectivities = sparse.csr_matrix(connectivities)
        
        distances = adata.obsp["distances"]
        if sparse.issparse(distances):
            distances = distances.tocsr()
        else:
            distances = sparse.csr_matrix(distances)
        
        neighbors_uns = dict(adata.uns[neighbors_key])
        
        # Create a minimal in-memory AnnData with neighbors and optionally X_pca
        import anndata
        obsm_dict = {}
        init_pos = "spectral"
        
        # Include X_pca if available for spectral initialization
        if "X_pca" in adata.obsm:
            X_pca = adata.obsm["X_pca"]
            X_pca = np.asarray(X_pca)
            obsm_dict["X_pca"] = X_pca
        else:
            # Use random initialization if no X_pca
            init_pos = "random"
            logger.info("X_pca not found, using random initialization for UMAP")
        
        adata_mem = anndata.AnnData(
            X=sparse.csr_matrix((n_obs, 1)),  # Minimal X (not used)
            obsp={
                "connectivities": connectivities,
                "distances": distances,
            },
            uns={neighbors_key: neighbors_uns},
        )
        
        # Add obsm separately to avoid shape mismatch issues
        for key, val in obsm_dict.items():
            adata_mem.obsm[key] = val
        
        logger.info(
            f"Computing UMAP embedding on {n_obs} cells "
            f"(min_dist={min_dist}, spread={spread})"
        )
        
        # Run scanpy's UMAP on the minimal AnnData
        sc.tl.umap(
            adata_mem,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            init_pos=init_pos,
            neighbors_key=neighbors_key,
            random_state=random_state,
        )
        
        # Extract results
        X_umap = adata_mem.obsm["X_umap"]
        umap_uns = adata_mem.uns.get("umap", {
            "params": {
                "min_dist": min_dist,
                "spread": spread,
                "n_components": n_components,
                "random_state": random_state,
            }
        })
        
        # Close crispyx wrapper and write results
        adata.close()
        
        write_obsm_to_h5ad(path, "X_umap", X_umap)
        write_uns_dict_to_h5ad(path, "umap", umap_uns)
        
        logger.info(
            f"UMAP complete: {n_components} components (written to {path})"
        )
        
        return None
    
    else:
        # In-memory or copy mode
        if copy:
            if is_crispyx_wrapper:
                adata = adata.to_memory()
            elif hasattr(adata, 'file') and adata.file is not None:
                adata = adata.to_memory()
            else:
                adata = adata.copy()
        
        logger.info(
            f"Computing UMAP embedding on {adata.n_obs} cells "
            f"(min_dist={min_dist}, spread={spread})"
        )
        
        # Run scanpy's UMAP
        sc.tl.umap(
            adata,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            neighbors_key=neighbors_key,
            random_state=random_state,
        )
        
        logger.info(f"UMAP complete: {n_components} components")
        
        if copy:
            return adata
        return None
