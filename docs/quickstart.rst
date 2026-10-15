Quick start
===========

crispyx provides a Scanpy-style API for streaming CRISPR screen analysis.
Each operation reads data from disk so large ``.h5ad`` files can be processed
on commodity hardware without loading the full count matrix into memory.

The typical workflow, with the guide page for each step:

1. **Load** – open a dataset on disk (:doc:`data_preparation`)
2. **QC** – filter cells, perturbations, and genes (:doc:`quality_control`)
3. **Preprocess** – normalise and log-transform, streaming (:doc:`large_data`
   covers the storage formats each step prefers)
4. **HVG selection** – pick highly variable genes (:doc:`quality_control`)
5. **Dimension reduction** – PCA and KNN graph construction (:doc:`quality_control`)
6. **Pseudo-bulk** – aggregate per perturbation (:doc:`pseudobulk`)
7. **Differential expression** – t-test, Wilcoxon, or NB-GLM (:doc:`differential_expression`)
8. **Plot** – visualise results with Scanpy-style helpers (:doc:`differential_expression`)

Screens run in batches, donors or lines: :doc:`covariate_adjustment` compares
the ways crispyx adjusts for them. The :doc:`crispyx tutorial
<notebooks/crispyx_tutorial>` runs the whole workflow on a small dataset.


A first run
-----------

.. code-block:: python

   import crispyx as cx

   adata_ro = cx.read_h5ad_ondisk("data/demo_benchmark.h5ad")
   adata_ro = cx.pp.qc_summary(
       adata_ro,
       perturbation_column="perturbation",
       min_genes=100,
       min_cells_per_perturbation=15,
       min_cells_per_gene=10,
   )
   adata_pb = cx.pb.normalized_effects(
       adata_ro,
       groupby="perturbation",
   )
   adata_ro = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="wilcoxon",
   )
   print(adata_ro.uns["rank_genes_groups"])  # preview without loading everything
   de_results = adata_ro.uns["rank_genes_groups"].load()
   de_full = de_results["full"]
   var_table = adata_ro.var.load()
