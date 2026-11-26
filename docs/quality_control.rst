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

Per-cell metrics, outliers and doublets
---------------------------------------

For a dataset with several samples and no perturbation column (a tumour
cohort, a tissue atlas), compute Scanpy's QC metrics, flag cells far from their
own sample's median, and score doublets one sample at a time:

.. code-block:: python

   var = cx.load_var("counts.h5ad")
   var["mt"] = var.index.str.startswith("MT-")
   cx.write_var("counts.h5ad", var)
   cx.pp.calculate_qc_metrics("counts.h5ad", qc_vars=["mt"])  # writes obs and var columns

   flags = cx.pp.mad_outliers(
       "counts.h5ad", ["log1p_total_counts", "log1p_n_genes_by_counts", "pct_counts_mt"],
       groupby="sample", nmads=5,
   )
   obs = cx.load_obs("counts.h5ad")
   bad = (
       flags["log1p_total_counts_low"] | flags["log1p_n_genes_by_counts_low"]
       | (flags["pct_counts_mt_high"] & (obs["pct_counts_mt"] > 10))
   )
   cx.pp.scrublet("counts.h5ad", batch_key="sample", cell_mask=~bad.to_numpy())

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Does
   * - :func:`crispyx.pp.calculate_qc_metrics`
     - Scanpy's ``n_genes_by_counts``, ``total_counts``,
       ``pct_counts_<set>`` (for each boolean ``var`` column in
       ``qc_vars``) and the per-gene metrics, in one pass over the matrix in
       its stored layout. Writes them to the file unless ``inplace=False``.
   * - :func:`crispyx.pp.mad_outliers`
     - ``<key>_low`` / ``<key>_high`` flags for values more than ``nmads``
       median absolute deviations from the median of the cell's ``groupby``
       group. Reads ``obs`` only.
   * - :func:`crispyx.pp.scrublet`
     - :func:`scanpy.pp.scrublet` on each ``batch_key`` group read from disk
       in turn, so memory holds one sample. Writes ``doublet_score`` and
       ``predicted_doublet``. Needs ``pip install 'crispyx[scrublet]'``.

Fewer cells, fewer counts
-------------------------

Two functions reduce a dataset before normalisation, each streamed to a new
file:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Function
     - Decides
     - Typical use
   * - :func:`crispyx.pp.subsample`
     - Which cells stay: ``n`` or ``frac`` drawn within each stratum of
       ``groupby``. With ``unit="batch"`` (any ``obs`` column) whole units
       are kept or dropped. A stratum smaller than ``n`` is dropped, or kept
       whole with ``drop_insufficient=False``.
     - The same number of cells for every perturbation
   * - :func:`crispyx.pp.downsample_counts`
     - How many counts stay in a cell: thinned to ``counts_per_cell``
       without replacement.
     - Equal sequencing depth across datasets

.. code-block:: python

   capped = cx.pp.subsample(
       "data/filtered.h5ad", n=200, groupby="perturbation", random_state=0,
       output_path="data/subsampled.h5ad",
   )

Assign guides to cells
----------------------

A screen that ships its guide-capture counts rather than per-cell calls needs
one rule to turn them into labels. :func:`crispyx.pp.assign_guides` gives each
cell its most abundant guide when that guide has at least ``min_umi`` UMIs
(default 3) and at least ``min_ratio`` times the runner-up's (default 2); any
other cell, with no guide, two guides at similar counts, or a multiplet, is
left ``NA``. Such a cell is not a control: whether the unassigned cells may
join the controls is the question :func:`crispyx.pp.compare_controls`
answers below.

.. code-block:: python

   # cells x guides UMI matrix: an array, an AnnData, or an h5ad streamed in blocks
   guide = cx.pp.assign_guides("data/guide_counts.h5ad", min_umi=3, min_ratio=2.0)
   obs = cx.load_obs("data/counts.h5ad")
   obs["guide"] = guide.reindex(obs.index)
   cx.write_obs("data/counts.h5ad", obs)
   # then map guide names to targets: normalise_perturbation_labels(target_pattern=...)

Check the file before analysis
------------------------------

:func:`crispyx.pp.check_data` reads ``obs``, ``var`` and a few sampled blocks
of ``X`` and reports the mistakes that mislead an analysis without making it
fail: a control synonym (``NTC``, ``non-targeting``, ``scramble``, ``GFP``
...) left as a perturbation beside the control label, labels that are still
guide names, labels that differ only by case, symbols truncated at a hyphen,
targets that name no gene in ``var``, duplicate gene names, genes split over
several rows (``_PAR_Y`` copies, ``var_names_make_unique`` suffixes, symbols
on more than one row), ``X`` that does not hold counts, cells whose row sum
exceeds their recorded total, and inconsistent sparse pointers. Each finding is an ``Issue`` with a level;
``raise_on_errors`` stops a pipeline on the errors.

.. code-block:: python

   issues = cx.pp.check_data(
       "data/counts.h5ad",
       perturbation_column="perturbation",
       control_label="control",
   )
   cx.raise_on_errors(issues)   # ValueError listing the errors, if any

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
- The differences are small. Re-estimated with the non-targeting controls
  alone, the screen's effects hardly change (r 0.97; knockdowns of the
  targets' own genes identical), except on the construct genes, which shift
  by up to 0.05; the smaller control set finds a quarter fewer hits
  (:doc:`notebooks/residual_effects_targeted_feng`, Part 4). Pooling the
  unassigned cells is acceptable here if effects on the construct genes are
  discounted.

The check finds differences, not their size. When it finds them, re-estimate
the effects of interest with each set of controls before choosing.

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

Both flavors stream the data a chunk at a time (rows × genes for CSR/dense,
cells × a few genes for CSC), dispatching on storage format the same way
the QC functions do, and write ``var["highly_variable"]``,
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

**Several samples.** ``batch_key`` selects genes within each sample and
combines them as Scanpy does, so differences between samples (donors,
patients) do not fill the list. Both passes over the matrix accumulate every
sample at once, so the cost does not grow with the number of samples.

.. code-block:: python

   cx.pp.highly_variable_genes(
       "counts.h5ad", batch_key="sample", cell_mask=None, n_top_genes=2000,
       flavor="seurat_v3_paper",
   )

``var["highly_variable_nbatches"]`` counts the samples that select a gene. The
two ``seurat_v3`` flavors differ only in how they rank genes across samples:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``flavor``
     - Genes ranked by
   * - ``"seurat_v3"``
     - median rank among the samples that select the gene, then the number of samples
   * - ``"seurat_v3_paper"``
     - number of samples, then median rank (Seurat's ``SelectIntegrationFeatures``)

.. _quality-control-pca:

Dimension reduction
-------------------

For visualization and clustering, CRISPYx provides streaming PCA and KNN
graph construction that works with on-disk data:

.. code-block:: python

   # Streaming PCA (auto-selects optimal method based on gene count)
   cx.pp.pca(adata_norm, n_comps=50)
   
   # Build KNN graph from PCA embeddings
   cx.pp.neighbors(adata_norm, n_neighbors=15)

   # Leiden clusters of that graph, in obs["leiden"]
   cx.tl.leiden(adata_norm, resolution=1.0)

:func:`crispyx.tl.leiden` reads only the graph, never ``X``, and runs
:func:`scanpy.tl.leiden` with ``flavor="igraph"`` on it (``pip install
'crispyx[leiden]'``). Markers of the clusters: :ref:`one-vs-rest`.

Plot the embedding:

.. code-block:: python

   # Plot variance explained per component
   cx.pl.pca_variance_ratio(adata_norm, n_pcs=20)

   # PCA scatter colored by perturbation
   cx.pl.pca(adata_norm, color='perturbation', components='1,2')

   # Gene loadings for top components
   cx.pl.pca_loadings(adata_norm, components=[1, 2, 3])

PCA streams the cells twice (fit, then project) with one of two methods:

* **Covariance** (``method='covariance'``): exact PCA from the gene × gene
  covariance, accumulated in place with BLAS. It holds one ``8 × genes²``-byte
  matrix (32 MB for 2,000 HVGs, 2 GB for 16,000 genes, 8.7 GB for 33,000), so
  it suits HVGs. Forced when its peak (chiefly the matrix) exceeds the memory
  budget, it raises ``MemoryError`` before the passes.
* **IncrementalPCA** (``method='incremental'``): scikit-learn's approximate
  streaming PCA, memory linear in the genes, for gene sets whose covariance
  matrix does not fit. It is 2.5-7 times slower than covariance, and only its
  leading components are accurate (table below); later ones also change
  with ``chunk_size``.
* **Automatic selection** (``method='auto'``, default): covariance whenever
  its peak fits in half of ``memory_limit_gb`` (detected when not given), and
  incremental only when it does not. The chunk is 4,096 cells unless that does not fit
  in half the budget; ``uns['pca']`` records the method and chunk used.

On Replogle et al.'s genome-wide K562 screen (1,989,569 cells, log-normalised,
2,000 highly variable genes, 50 components; a Linux server, 16 BLAS threads):

.. list-table::
   :header-rows: 1

   * - Method
     - Time
     - Peak memory
     - Agreement with Scanpy (arpack)
   * - crispyx ``covariance``
     - 591 s
     - 0.71 GB
     - every component \|cos\| > 0.99999, explained variance within 1e-6
   * - crispyx ``incremental``
     - 1,489 s
     - 0.90 GB
     - first 10 components within 0.25° of covariance, first 20 within
       1.4°, first 30 within 7°; components 31-50 differ
   * - Scanpy ``sc.pp.pca`` (arpack), file loaded
     - 619-736 s
     - 101 GB
     - two seeds agree within 0.02°

Covariance on all 8,246 genes took 949 s and 1.6 GB.

Neither method is randomized: the same file, method and chunk give the same
result on every run. Covariance gives the same result at any chunk size and
thread count (scores within 1e-7), and matches an exact PCA to float32
precision. Components follow scikit-learn's sign convention (each one's
largest loading positive), and ``uns['pca']['variance']`` is the sample
variance, as in Scanpy.

Restrict PCA to highly variable genes (``cx.pp.highly_variable_genes`` first,
``use_highly_variable=True``, the default): without a ``highly_variable``
column it runs on every gene. The peak memory of each step is in
:ref:`memory-model`.

PCA results are stored in:

* ``adata.obsm['X_pca']``: Cell embeddings (n_cells × n_comps)
* ``adata.varm['PCs']``: Gene loadings (n_genes × n_comps)
* ``adata.uns['pca']``: Variance info and method metadata

KNN results are stored in:

* ``adata.obsp['distances']``: Sparse distance matrix
* ``adata.obsp['connectivities']``: Sparse connectivity matrix (UMAP-style)
* ``adata.uns['neighbors']``: Parameters dict

**Results go to the file**: on a file opened with ``cx.read_h5ad_ondisk()``,
or given as a path, PCA, neighbors and UMAP write their results into the h5ad
file. PCA writes ``X_pca`` a chunk at a time, so its memory grows with the
number of cells only by a CSR file's 8-byte row index. ``copy=True`` instead loads the whole file into memory
and returns it, as Scanpy's ``copy`` does.
