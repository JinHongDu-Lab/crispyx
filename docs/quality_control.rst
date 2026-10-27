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

Plot the QC summary:

.. code-block:: python

   qc = cx.pp.qc_summary(
       adata_ro,
       perturbation_column="perturbation",
       min_genes=100,
       min_cells_per_perturbation=15,
       min_cells_per_gene=10,
   )
   cx.pl.qc_perturbation_counts(
       data=adata_ro,
       perturbation_column="perturbation",
       cell_mask=qc.cell_mask,
   )
   cx.pl.qc_summary(qc, min_genes=100, min_cells_per_gene=10)

.. _check-controls:

Check the control cells
-----------------------

Every effect is measured against the controls, so it pays to check that they
behave alike before testing. Two questions come up:

- **Do the control guides agree?** :func:`crispyx.pp.control_guide_correlation`
  sums each non-targeting guide's counts, takes each guide's deviation from
  the guides' average (``log1p`` counts per 10,000), and correlates the
  deviations; :func:`crispyx.pl.control_guide_correlation` draws the matrix,
  clustered. Guides that differ only by noise correlate near 0. A block of
  correlated guides shares variation -- a batch, an off-target effect -- that
  a test reads as signal when those guides are the controls.
- **May cells without a detected guide join the controls?** Screens with a
  low assignment rate often pool "unassigned" cells into the controls
  (Feng *et al.* 2026, *Cell Genomics*, use all of them).
  :func:`crispyx.pp.compare_controls` tests each non-targeting guide against
  the unassigned cells and, as a yardstick, against the other non-targeting
  guides (``nb_glm_test`` with any covariates and settings you pass). If the
  unassigned cells are like the guides, both comparisons give about as many
  hits.

.. code-block:: python

   guides = cx.load_obs(path)["guide"].astype(str)
   controls = sorted(g for g in guides.unique() if g.startswith("NonTarget"))

   corr = cx.pp.control_guide_correlation(path, guide_column="guide",
                                          controls=controls, reference="unassigned")
   cx.pl.control_guide_correlation(corr)

   table = cx.pp.compare_controls(path, guide_column="guide", controls=controls,
                                  reference="unassigned", work_dir="control_check")

Both work on single cells or on pseudo-bulk counts; on pseudo-bulk (units per
guide and batch from ``cx.pb.aggregate``, ``size_factor_method="deseq2"``, the
batch as a covariate) the comparison is calibrated and fast.

**Example** (:doc:`notebooks/control_checks`, with the correlation heatmap
and each guide's hits). The targeted screen of Feng *et al.* has 535,000
control cells, 98 % of them unassigned. In pseudo-bulk per guide and inlet,
with the inlet as a covariate:

.. list-table::
   :header-rows: 1

   * - Each non-targeting guide against
     - Genes differing
     - Genes with \|z\| > 3
   * - the unassigned cells
     - 132-438
     - 1.5-2.3 %
   * - the other non-targeting guides
     - 0-16
     - 0.13-0.39 %

- The guides agreed with one another (median r -0.02, at most 0.26).
- Pooled, 849 genes differed between guides and unassigned cells, including
  the CRISPRi construct (dCas9-KRAB-MeCP2, lower in unassigned cells),
  donor-line genes (XIST, MEG3) and the screen's own targets (lower in
  unassigned cells for 68 % of them, against 55 % of other genes).
- Many unassigned cells carry less CRISPRi machinery or an undetected guide.
  Here, the non-targeting cells are the safer controls.

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

Plot the embedding:

.. code-block:: python

   # Plot variance explained per component
   cx.pl.pca_variance_ratio(adata_norm, n_pcs=20)

   # PCA scatter colored by perturbation
   cx.pl.pca(adata_norm, color='perturbation', components='1,2')

   # Gene loadings for top components
   cx.pl.pca_loadings(adata_norm, components=[1, 2, 3])

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
