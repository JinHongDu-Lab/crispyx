Comparison with other tools
===========================

This page explains when to use crispyx, when to use another tool, and what
the practical differences are across common workflows.

crispyx vs Scanpy
-----------------

**Summary:** crispyx produces identical results to Scanpy for t-test and
Wilcoxon differential expression (Pearson *r* > 0.9999), but uses substantially
less time and memory by streaming from disk instead of loading the full count
matrix into RAM. This assumes adequate scratch disk; see :ref:`disk-space`.

When to use crispyx instead of Scanpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Your dataset has more than ~200,000 cells, or your count matrix exceeds the
  available RAM on your machine or HPC job.
- You are running multiple DE methods on the same dataset and need reproducible,
  low-memory pipelines.
- You want to keep data on disk between steps and avoid repeated loading.

When to use Scanpy instead of crispyx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Your dataset fits comfortably in RAM and you need a broad ecosystem of
  single-cell analysis tools beyond QC and DE (trajectory inference, RNA
  velocity, spatial transcriptomics, etc.).
- You need tight integration with third-party Scanpy extensions that require
  in-memory AnnData.

Example: Wilcoxon DE on a large screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # crispyx: streams from disk, minimal memory
   import crispyx as cx
   adata = cx.read_h5ad_ondisk("large_screen.h5ad")
   adata = cx.pp.qc_summary(adata, perturbation_column="perturbation")
   adata = cx.tl.rank_genes_groups(adata, perturbation_column="perturbation",
                                   method="wilcoxon")

   # Scanpy: loads entire matrix into RAM
   import scanpy as sc
   adata = sc.read_h5ad("large_screen.h5ad")  # may OOM on large datasets
   sc.tl.rank_genes_groups(adata, groupby="perturbation", method="wilcoxon")

The crispyx result for logfoldchanges, scores, and p-values matches Scanpy
with Pearson *r* > 0.9999 across all tested datasets. That includes Scanpy's
convention for the fold change of a gene absent from one arm, where the
reported magnitude is set by the formula's ``1e-9`` constant rather than by
the data; see :doc:`faq` before ranking on ``logfoldchanges``.

crispyx vs Pertpy / PyDESeq2
-----------------------------

**Summary:** crispyx's NB-GLM is DESeq2's model fitted one perturbation at a
time against the controls, at cell level or on pseudo-bulk, with covariates
and apeGLM shrinkage, streamed under a memory budget. Given the same size
factors and mean floor its statistics agree with PyDESeq2's (z correlation
0.997-0.9999, median difference in z 0.0004-0.004 on six screens; see
:doc:`nb_glm`). It completes genome-wide screens where Pertpy/PyDESeq2 exceeds
the available memory.

When to use crispyx NB-GLM instead of Pertpy/PyDESeq2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Your dataset has many perturbation groups (hundreds to thousands) or many
  cells per group, making PyDESeq2's per-group fitting memory-intensive.
- You need NB-GLM DE on a genome-wide screen (>10,000 perturbations).
- You want a streaming pipeline that integrates QC, pseudobulk, and DE in a
  single on-disk workflow.

When to use Pertpy/PyDESeq2 instead of crispyx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- You need arbitrary contrasts or Cook's-distance outlier filtering; crispyx
  fits one reference group against the controls per fit.
- Your dataset is small enough that memory is not a constraint.
- You need Pertpy's other perturbation analysis methods (augur, mixscape, etc.).

Example: NB-GLM DE with crispyx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import crispyx as cx

   adata = cx.read_h5ad_ondisk("screen.h5ad")
   adata = cx.pp.qc_summary(adata, perturbation_column="perturbation")
   adata = cx.tl.rank_genes_groups(
       adata,
       perturbation_column="perturbation",
       method="nb_glm",
       memory_limit_gb=32,
   )

crispyx vs edgeR
-----------------

**Summary:** edgeR (via rpy2) times out or errors on most large CRISPR screen
datasets. crispyx completes reliably at that scale. edgeR remains the
standard for bulk RNA-seq and small pseudobulk experiments; for large
perturbation screens with many groups, crispyx is more practical.

When to use edgeR instead of crispyx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- You are working with bulk RNA-seq data or have a small number of perturbation
  groups (< 50) where edgeR's quasi-likelihood F-test is well-calibrated.
- You need an R-native workflow integrated with DESeq2, limma, or other
  Bioconductor packages.

Limitations of crispyx
-----------------------

- crispyx does not replace the full Scanpy or Pertpy ecosystem. It focuses on
  QC, normalization, pseudobulk, and differential expression for large CRISPR
  screens. Trajectory inference, RNA velocity, and spatial transcriptomics are
  outside its scope.
- The NB-GLM tests one perturbation (or a few, ``targets_per_fit``) against
  the controls per fit, with categorical and continuous covariates; it does
  not fit arbitrary contrasts or filter outliers by Cook's distance.
- crispyx does not read raw sequencing outputs (Cell Ranger matrices,
  MatrixMarket, CSV): it starts from an ``.h5ad`` file, which Scanpy or
  anndata write from those.
- On a dataset that fits in RAM, the memory savings do not matter. The
  reasons to use crispyx there are the methods: cell-level NB-GLM with
  covariates, covariate-adjusted residual effects, the control checks and
  the data-preparation utilities.

See also
--------

- :doc:`quickstart` and the guide pages it links — complete workflow examples
- :doc:`covariate_adjustment` — adjusting for batches, donors and other covariates
- :doc:`faq` — troubleshooting and common questions
