Comparison with other tools
===========================

This page explains when to use crispyx, when to use another tool, and what
the practical differences are across common workflows.

crispyx vs Scanpy
-----------------

**Summary:** crispyx produces identical results to Scanpy for t-test and
Wilcoxon differential expression (Pearson *r* > 0.9999), but uses 2–43×
less time and 2–6× less memory by streaming from disk instead of loading the
full count matrix into RAM. These figures assume adequate scratch disk; see
:ref:`disk-space` in the usage guide.

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
with Pearson *r* > 0.9999 across all tested datasets.

Performance comparison
~~~~~~~~~~~~~~~~~~~~~~~

+--------------------+-----------------------+------------------------+
| Dataset            | crispyx Wilcoxon      | Scanpy Wilcoxon        |
+====================+=======================+========================+
| Adamson (65K cells)| 123 s, 3.2 GB peak    | 1305 s, 3.3 GB peak    |
+--------------------+-----------------------+------------------------+
| Feng-gwsf (321K)   | 10761 s, 17.2 GB peak | timeout (>9 hours)     |
+--------------------+-----------------------+------------------------+
| Feng-gwsnf (1.97M) | completes             | timeout / OOM          |
+--------------------+-----------------------+------------------------+

crispyx vs Pertpy / PyDESeq2
-----------------------------

**Summary:** crispyx's NB-GLM is approximately 2× faster than Pertpy/PyDESeq2
and uses substantially less memory. On genome-wide datasets, Pertpy/PyDESeq2
typically exceeds 120 GB peak memory or fails outright; crispyx completes
within 64 GB. Pearson *r* > 0.97 vs PyDESeq2 for log-fold-change estimates.

When to use crispyx NB-GLM instead of Pertpy/PyDESeq2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Your dataset has many perturbation groups (hundreds to thousands) or many
  cells per group, making PyDESeq2's per-group fitting memory-intensive.
- You need NB-GLM DE on a genome-wide screen (>10,000 perturbations).
- You want a streaming pipeline that integrates QC, pseudobulk, and DE in a
  single on-disk workflow.

When to use Pertpy/PyDESeq2 instead of crispyx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- You need the full PyDESeq2 feature set (custom design matrices, contrasts,
  Cook's distance filtering, etc.).
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

Performance comparison (NB-GLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+-------------------+---------------------------+
| Dataset              | crispyx NB-GLM    | Pertpy/PyDESeq2           |
+======================+===================+===========================+
| Adamson (65K cells)  | 2472 s, 3.5 GB    | 5318 s, 33.8 GB           |
+----------------------+-------------------+---------------------------+
| Feng-gwsf (321K)     | completes         | memory limit (>124 GB)    |
+----------------------+-------------------+---------------------------+
| Feng-gwsnf (1.97M)   | completes         | memory limit / fails      |
+----------------------+-------------------+---------------------------+

crispyx vs edgeR
-----------------

**Summary:** edgeR (via rpy2) times out or errors on most large CRISPR screen
datasets tested in the benchmark. crispyx completes all 12 tested datasets.
edgeR remains the standard for bulk RNA-seq and small pseudobulk experiments;
for large perturbation screens with many groups, crispyx is more practical.

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
- The NB-GLM uses a simplified model (per-gene intercept and perturbation
  effect) without support for complex design matrices or batch covariates
  beyond what is in the current API.
- On small datasets that fit in RAM, crispyx offers no meaningful advantage
  over Scanpy or Pertpy. Use Scanpy or Pertpy for datasets under ~50,000 cells
  if memory is not a concern.

See also
--------

- :doc:`benchmarking` — full benchmark results and reproduction instructions
- :doc:`usage` — complete workflow examples
- :doc:`faq` — troubleshooting and common questions
