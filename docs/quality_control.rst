Quality control and dimension reduction
=======================================

Filtering cells, perturbations and genes, selecting highly variable genes, and embedding cells, all streamed from disk. Normalisation is part of these steps; :doc:`large_data` covers the storage formats they prefer.


Quality control
---------------

Call :func:`crispyx.pp.qc_summary` to filter cells, perturbations,
and genes. The function writes a filtered AnnData file to disk and returns a
new :class:`cx.AnnData` view pointing at the result so the next step can reuse
the same handle without reopening the path. When ``control_label`` is omitted,
perturbations containing strings such as ``ctrl`` or ``nontarget`` are chosen
automatically and logged for reproducibility. Likewise, omitting
``gene_name_column`` falls back to ``adata.var_names`` with a helpful message.
Individual helpers such as :func:`crispyx.pp.filter_cells` are also
available for customised pipelines.

Highly variable genes
---------------------

Before PCA, select highly variable genes (HVGs) so the embedding is built
from the genes that actually carry signal rather than the full gene set:

.. code-block:: python

   # Raw-count file -> HVGs via the seurat_v3/vst method (default flavor)
   cx.pp.highly_variable_genes(
       adata_raw,
       perturbation_column="perturbation",
       n_top_genes=2000,
   )

   # Then normalize and run PCA -- it already restricts to var["highly_variable"]
   cx.pp.normalize_total_log1p(adata_raw, output_path="normalized.h5ad")
   cx.pp.pca(adata_norm, n_comps=50)

Two flavors are available via ``flavor=``:

* ``"seurat_v3"`` (default; Stuart et al. 2019): ranks genes by standardized
  variance fit with a LOESS smoother. Expects **raw counts** and requires
  ``n_top_genes``.
* ``"mean_dispersion"`` (Satija et al. 2015): bins genes by mean expression
  and z-normalizes dispersion within each bin. Expects **log1p-normalized
  data**; needs no extra dependency (``scikit-misc``, used only by
  ``"seurat_v3"``'s LOESS fit).

Both flavors stream the data in ``O(n_genes)`` memory, dispatching on
storage format the same way the QC functions do (row-chunked for
CSR/dense, column-chunked for CSC), and write ``var["highly_variable"]``,
``var["means"]``, ``var["variances"]``, and ``var["variances_norm"]`` back
to the file.

**Control-cells-only default.** Unlike scanpy/Seurat, ``highly_variable_genes``
defaults to computing gene statistics from **control cells only**
(``cell_mask="control"``), resolved from ``perturbation_column`` and
``control_label`` the same way every other perturbation-aware crispyx
function resolves them. This is a CRISPR/Perturb-seq-specific choice: over
all cells, on-target perturbation effects are often the single largest
source of per-gene variance, so the top-N variable-gene list -- and the PCA
embedding built from it -- ends up structured around *which perturbation a
cell received* rather than baseline cell-state heterogeneity (cell cycle,
cell size, etc.). Selecting HVGs from control cells only avoids that
confound while still applying the resulting gene set to the full dataset
for PCA. The mask is resolved directly from ``obs`` (no ``.X`` access) and
threaded into the streaming pass as a boolean mask, so restricting to
control cells costs nothing beyond the pass that would run regardless of
which cells are included -- no subset is ever copied or materialized.

.. code-block:: python

   # Default: HVGs from control cells only
   cx.pp.highly_variable_genes(adata_raw, perturbation_column="perturbation")

   # Opt out explicitly to use every cell (the scanpy/Seurat default)
   cx.pp.highly_variable_genes(adata_raw, cell_mask=None, n_top_genes=2000)

   # Or restrict to a custom subset (e.g. one batch, a QC-pass mask)
   cx.pp.highly_variable_genes(adata_raw, cell_mask=my_boolean_mask, n_top_genes=2000)

``cell_mask="control"`` requires ``perturbation_column`` -- pass it
explicitly, or set ``cell_mask=None`` to use all cells without perturbation
metadata. If the resolved control label matches zero cells, the call raises
rather than silently proceeding with a degenerate (``n=0``) statistic.

Dimension reduction
-------------------

For visualization and clustering, CRISPYx provides streaming PCA and KNN
graph construction that works with on-disk data:

.. code-block:: python

   # Streaming PCA (auto-selects optimal method based on gene count)
   cx.pp.pca(adata_norm, n_comps=50)
   
   # Build KNN graph from PCA embeddings
   cx.pp.neighbors(adata_norm, n_neighbors=15)

The PCA implementation uses a hybrid approach:

* **Sparse covariance** (``method='sparse_cov'``): ~5× faster for datasets with
  ≤15K genes. Exploits sparsity in the Xᵀ @ X computation.
* **IncrementalPCA** (``method='incremental'``): Lower memory for datasets with
  >15K genes. Uses sklearn's streaming PCA with partial_fit().
* **Automatic selection** (``method='auto'``, default): Chooses the optimal method
  based on gene count and available memory.

PCA results are stored in:

* ``adata.obsm['X_pca']``: Cell embeddings (n_cells × n_comps)
* ``adata.varm['PCs']``: Gene loadings (n_genes × n_comps)
* ``adata.uns['pca']``: Variance info and method metadata

KNN results are stored in:

* ``adata.obsp['distances']``: Sparse distance matrix
* ``adata.obsp['connectivities']``: Sparse connectivity matrix (UMAP-style)
* ``adata.uns['neighbors']``: Parameters dict

**Close-Write-Reopen Pattern**: When using ``cx.read_h5ad_ondisk()`` to load
backed data, PCA and neighbors results are written directly to the h5ad file.
This keeps ``.X`` on disk while persisting embeddings, loadings, and neighbor
graphs for later use. No ``copy=True`` is needed in typical workflows.
