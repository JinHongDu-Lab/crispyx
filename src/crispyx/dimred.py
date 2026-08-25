"""Dimension reduction module for crispyx.

Provides streaming/on-disk PCA and neighbor computation to avoid memory issues
with large datasets. Follows Scanpy-style API patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import anndata as AnnData

from .data import (
    calculate_pca_chunk_size,
    _to_dense,
    write_obsm_to_h5ad,
    write_varm_to_h5ad,
    write_uns_dict_to_h5ad,
    write_obsp_to_h5ad,
    AnnData as CrispyxAnnData,
)

logger = logging.getLogger(__name__)


def _streaming_pca_sparse_cov(
    adata: "AnnData",
    n_comps: int = 50,
    chunk_size: int = 2048,
    use_highly_variable: bool = True,
    return_info: bool = False,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """Compute PCA using sparse covariance method.
    
    Efficient for datasets with moderate gene counts (< ~15K genes).
    Computes X^T @ X in a streaming fashion, exploiting sparsity.
    
    Parameters
    ----------
    adata
        AnnData object with expression data.
    n_comps
        Number of principal components to compute.
    chunk_size
        Number of cells per chunk.
    use_highly_variable
        If True and 'highly_variable' exists in var, use only HVGs.
    return_info
        If True, return additional info dict.
    show_progress
        Show progress bar.
    
    Returns
    -------
    X_pca
        PCA-transformed data (n_obs × n_comps).
    components
        Principal components (n_comps × n_vars).
    variance_ratio
        Variance explained ratio for each component.
    info
        Optional dict with mean, variance, etc. if return_info=True.
    """
    X = adata.X
    n_obs, n_vars_total = adata.shape
    
    # Determine which genes to use
    if use_highly_variable and "highly_variable" in adata.var.columns:
        gene_mask = adata.var["highly_variable"].values
        gene_indices = np.where(gene_mask)[0]
        n_vars = len(gene_indices)
        logger.info(f"Using {n_vars} highly variable genes for PCA")
    else:
        gene_indices = None
        n_vars = n_vars_total
        logger.info(f"Using all {n_vars} genes for PCA")
    
    n_comps = min(n_comps, n_vars, n_obs)
    
    # Streaming: compute sums and X^T @ X
    gene_sums = np.zeros(n_vars, dtype=np.float64)
    XTX = np.zeros((n_vars, n_vars), dtype=np.float64)
    
    n_chunks = (n_obs + chunk_size - 1) // chunk_size
    pbar_desc = "Computing covariance"
    
    for chunk_start in tqdm(
        range(0, n_obs, chunk_size),
        total=n_chunks,
        desc=pbar_desc,
        disable=not show_progress,
    ):
        chunk_end = min(chunk_start + chunk_size, n_obs)
        chunk = X[chunk_start:chunk_end, :]
        
        # Subset to selected genes
        if gene_indices is not None:
            chunk = chunk[:, gene_indices]
        
        # Convert to dense if sparse
        chunk_dense = _to_dense(chunk)
        
        # Accumulate sums
        gene_sums += chunk_dense.sum(axis=0)
        
        # Accumulate X^T @ X (exploits sparsity if input was sparse)
        XTX += chunk_dense.T @ chunk_dense
    
    # Compute mean
    mean = gene_sums / n_obs
    
    # Compute covariance: (X^T @ X) / n - mean @ mean^T
    cov = XTX / n_obs - np.outer(mean, mean)
    
    # Eigendecomposition (get top n_comps)
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = eigh(
        cov,
        subset_by_index=[n_vars - n_comps, n_vars - 1],
    )
    
    # Reverse to get descending order
    eigenvalues = eigenvalues[::-1]
    components = eigenvectors[:, ::-1].T  # (n_comps, n_vars)
    
    # Variance explained ratio
    total_variance = np.trace(cov)
    variance_ratio = eigenvalues / total_variance
    
    # Transform data: second pass through data
    X_pca = np.zeros((n_obs, n_comps), dtype=np.float32)
    
    for chunk_start in tqdm(
        range(0, n_obs, chunk_size),
        total=n_chunks,
        desc="Transforming data",
        disable=not show_progress,
    ):
        chunk_end = min(chunk_start + chunk_size, n_obs)
        chunk = X[chunk_start:chunk_end, :]
        
        if gene_indices is not None:
            chunk = chunk[:, gene_indices]
        
        chunk_dense = _to_dense(chunk)
        
        # Center and project
        chunk_centered = chunk_dense - mean
        X_pca[chunk_start:chunk_end] = (chunk_centered @ components.T).astype(np.float32)
    
    info = None
    if return_info:
        info = {
            "mean": mean,
            "variance": eigenvalues,
            "variance_ratio": variance_ratio,
            "gene_indices": gene_indices,
        }
    
    return X_pca, components, variance_ratio, info


def _streaming_pca_incremental(
    adata: "AnnData",
    n_comps: int = 50,
    chunk_size: int = 1024,
    use_highly_variable: bool = True,
    return_info: bool = False,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """Compute PCA using IncrementalPCA from sklearn.
    
    Memory-efficient for datasets with many genes (> ~15K genes).
    Streams through data without loading all into memory.
    
    Parameters
    ----------
    adata
        AnnData object with expression data.
    n_comps
        Number of principal components to compute.
    chunk_size
        Number of cells per chunk. Must be >= n_comps.
    use_highly_variable
        If True and 'highly_variable' exists in var, use only HVGs.
    return_info
        If True, return additional info dict.
    show_progress
        Show progress bar.
    
    Returns
    -------
    X_pca
        PCA-transformed data (n_obs × n_comps).
    components
        Principal components (n_comps × n_vars).
    variance_ratio
        Variance explained ratio for each component.
    info
        Optional dict with mean, noise_variance, etc. if return_info=True.
    """
    X = adata.X
    n_obs, n_vars_total = adata.shape
    
    # Determine which genes to use
    if use_highly_variable and "highly_variable" in adata.var.columns:
        gene_mask = adata.var["highly_variable"].values
        gene_indices = np.where(gene_mask)[0]
        n_vars = len(gene_indices)
        logger.info(f"Using {n_vars} highly variable genes for PCA")
    else:
        gene_indices = None
        n_vars = n_vars_total
        logger.info(f"Using all {n_vars} genes for PCA")
    
    n_comps = min(n_comps, n_vars, n_obs)
    
    # Ensure chunk_size >= n_comps for IncrementalPCA
    actual_chunk_size = max(chunk_size, n_comps)
    if actual_chunk_size != chunk_size:
        logger.info(f"Adjusted chunk size from {chunk_size} to {actual_chunk_size} (>= n_comps)")
    
    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_comps)
    
    n_chunks = (n_obs + actual_chunk_size - 1) // actual_chunk_size
    
    # First pass: partial_fit to learn components
    for chunk_start in tqdm(
        range(0, n_obs, actual_chunk_size),
        total=n_chunks,
        desc="Learning PCA",
        disable=not show_progress,
    ):
        chunk_end = min(chunk_start + actual_chunk_size, n_obs)
        chunk = X[chunk_start:chunk_end, :]
        
        if gene_indices is not None:
            chunk = chunk[:, gene_indices]
        
        chunk_dense = _to_dense(chunk)
        
        # Skip if chunk smaller than n_comps (last chunk edge case)
        if chunk_dense.shape[0] < n_comps:
            logger.debug(f"Skipping small final chunk of size {chunk_dense.shape[0]}")
            continue
        
        ipca.partial_fit(chunk_dense)
    
    # Extract components
    components = ipca.components_  # (n_comps, n_vars)
    variance_ratio = ipca.explained_variance_ratio_
    
    # Second pass: transform data
    X_pca = np.zeros((n_obs, n_comps), dtype=np.float32)
    
    for chunk_start in tqdm(
        range(0, n_obs, actual_chunk_size),
        total=n_chunks,
        desc="Transforming data",
        disable=not show_progress,
    ):
        chunk_end = min(chunk_start + actual_chunk_size, n_obs)
        chunk = X[chunk_start:chunk_end, :]
        
        if gene_indices is not None:
            chunk = chunk[:, gene_indices]
        
        chunk_dense = _to_dense(chunk)
        X_pca[chunk_start:chunk_end] = ipca.transform(chunk_dense).astype(np.float32)
    
    info = None
    if return_info:
        info = {
            "mean": ipca.mean_,
            "variance": ipca.explained_variance_,
            "variance_ratio": variance_ratio,
            "gene_indices": gene_indices,
            "noise_variance": ipca.noise_variance_,
        }
    
    return X_pca, components, variance_ratio, info


def pca(
    adata: "AnnData",
    n_comps: int = 50,
    method: str = "auto",
    use_highly_variable: bool = True,
    chunk_size: int | None = None,
    random_state: int = 0,
    copy: bool = False,
    show_progress: bool = True,
) -> "AnnData" | None:
    """Compute Principal Component Analysis (PCA) on backed AnnData.
    
    Streaming implementation that works with on-disk data to avoid memory issues.
    Automatically selects the optimal method based on dataset characteristics.
    
    For backed data (crispyx.AnnData wrapper), results are written directly to
    the h5ad file using a close-write-reopen pattern, keeping .X on disk.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    n_comps
        Number of principal components to compute. Default 50.
    method
        PCA method to use:
        - 'auto': Automatically select based on gene count and memory
        - 'sparse_cov': Use sparse covariance method (fast for ≤15K genes)
        - 'incremental': Use IncrementalPCA (memory-efficient for >15K genes)
    use_highly_variable
        If True and 'highly_variable' exists in var, restrict to HVGs.
        Default True.
    chunk_size
        Number of cells to process per chunk. If None, automatically
        calculated based on available memory.
    random_state
        Random seed (not currently used, for API compatibility).
    copy
        If True, return a copy of adata with PCA results.
        If False, modify adata in place and return None.
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
        Principal components (n_vars × n_comps). Only includes selected genes.
    uns['pca']
        Dict with 'variance', 'variance_ratio', 'use_highly_variable'.
    
    Examples
    --------
    >>> import crispyx as cx
    >>> adata = cx.read_backed("data.h5ad")
    >>> cx.pp.pca(adata, n_comps=50)
    >>> adata.obsm['X_pca'].shape
    (n_obs, 50)
    """
    import anndata
    
    # Detect if this is a crispyx.AnnData wrapper (backed, with .path)
    is_crispyx_wrapper = isinstance(adata, CrispyxAnnData)
    
    if copy:
        # For copy mode, load into memory
        if is_crispyx_wrapper:
            adata = adata.to_memory()
        elif hasattr(adata, 'file') and adata.file is not None:
            adata = adata.to_memory()
        else:
            adata = adata.copy()
    
    n_obs, n_vars = adata.shape
    
    # Determine gene count for PCA (after HVG filtering)
    if use_highly_variable and "highly_variable" in adata.var.columns:
        n_vars_pca = adata.var["highly_variable"].sum()
    else:
        n_vars_pca = n_vars
    
    # Calculate chunk size and method if not provided
    if chunk_size is None:
        chunk_size, selected_method = calculate_pca_chunk_size(
            n_obs=n_obs,
            n_vars=n_vars_pca,
            n_comps=n_comps,
            method=method,
        )
    else:
        # Still need to select method
        if method == "auto":
            _, selected_method = calculate_pca_chunk_size(
                n_obs=n_obs,
                n_vars=n_vars_pca,
                n_comps=n_comps,
                method=method,
            )
        else:
            selected_method = method
    
    logger.info(f"Running PCA with method='{selected_method}', chunk_size={chunk_size}")
    
    # Run PCA
    if selected_method == "sparse_cov":
        X_pca, components, variance_ratio, info = _streaming_pca_sparse_cov(
            adata,
            n_comps=n_comps,
            chunk_size=chunk_size,
            use_highly_variable=use_highly_variable,
            return_info=True,
            show_progress=show_progress,
        )
    else:
        X_pca, components, variance_ratio, info = _streaming_pca_incremental(
            adata,
            n_comps=n_comps,
            chunk_size=chunk_size,
            use_highly_variable=use_highly_variable,
            return_info=True,
            show_progress=show_progress,
        )
    
    # Prepare PCs array
    if use_highly_variable and "highly_variable" in adata.var.columns:
        pcs_full = np.zeros((n_vars, n_comps), dtype=np.float32)
        pcs_full[info["gene_indices"], :] = components.T
    else:
        pcs_full = components.T.astype(np.float32)
    
    # Prepare uns dict
    pca_uns = {
        "variance": info["variance"],
        "variance_ratio": variance_ratio,
        "use_highly_variable": use_highly_variable,
        "method": selected_method,
        "n_comps": n_comps,
    }
    
    # Store results: use close-write-reopen for crispyx wrapper, direct for in-memory
    if is_crispyx_wrapper and not copy:
        # Close file handle
        path = adata.path
        adata.close()
        
        # Write results to h5ad file
        write_obsm_to_h5ad(path, "X_pca", X_pca)
        write_varm_to_h5ad(path, "PCs", pcs_full)
        write_uns_dict_to_h5ad(path, "pca", pca_uns)
        
        # File will be reopened lazily on next access
        logger.info(
            f"PCA complete: {n_comps} components, "
            f"variance explained: {variance_ratio.sum():.2%} (written to {path})"
        )
    else:
        # In-memory AnnData: store directly
        adata.obsm["X_pca"] = X_pca
        adata.varm["PCs"] = pcs_full
        adata.uns["pca"] = pca_uns
        
        logger.info(
            f"PCA complete: {n_comps} components, "
            f"variance explained: {variance_ratio.sum():.2%}"
        )
    
    if copy:
        return adata
    return None


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
