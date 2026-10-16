Loading and preparing data
==========================

Opening a dataset without loading its matrix, and the utilities that tidy its metadata before analysis.


Loading data
------------

Use :func:`crispyx.read_h5ad_ondisk` to open a dataset without
materialising the expression matrix and print the first few rows of metadata.
The examples below assume a dataset at ``data/demo_benchmark.h5ad``; substitute
the path to your own ``.h5ad`` file. The returned :class:`cx.AnnData` object keeps a
backed AnnData handle alive lazily and automatically closes it when the wrapper
is garbage collected. Explicitly call ``adata.close()`` to release the file as
soon as you finish the preview.

Data preparation utilities
--------------------------

crispyx 0.7.5 adds five utility families for cleaning heterogeneous datasets
before running QC or differential expression.

Editing backed metadata without loading X
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load only the ``obs`` or ``var`` table from a backed file, edit it in Python,
and write it back — without ever reading the expression matrix:

.. code-block:: python

   # Load obs metadata (X is never read)
   obs = cx.load_obs("data/counts.h5ad")
   obs["batch"] = obs["batch"].str.upper()

   # Write back (must have the same number of rows)
   cx.write_obs("data/counts.h5ad", obs)

   # Same for var
   var = cx.load_var("data/counts.h5ad")
   var["gene_symbols"] = var["gene_symbols"].str.upper()
   cx.write_var("data/counts.h5ad", var)

Gene name standardisation
~~~~~~~~~~~~~~~~~~~~~~~~~

Normalise Ensembl version suffixes, mitochondrial prefixes, and optionally
map IDs to HGNC symbols via ``mygene`` (``pip install mygene``):

.. code-block:: python

   # Strip version suffix + mt normalisation (in-place)
   cx.standardise_gene_names(
       "data/counts.h5ad",
       column="ensembl_id",          # var column; None uses var_names
       strip_version=True,           # ENSG00000123.4 → ENSG00000123
       normalise_mt_prefix=True,     # mt-ND1 → MT-ND1
   )

   # Online Ensembl → symbol lookup (returns Series without modifying file)
   symbols = cx.standardise_gene_names(
       "data/counts.h5ad",
       column="ensembl_id",
       lookup_symbols=True,
       species="human",
       unmapped_action="warn",
       inplace=False,
   )

Perturbation label normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strip guide prefixes/suffixes and unify diverse control labels:

.. code-block:: python

   cx.normalise_perturbation_labels(
       "data/counts.h5ad",
       column="perturbation",
       strip_prefixes=["sg-", "sg"],
       strip_suffixes=["_KO", "_KD", "_P1P2"],
       canonical_control="NTC",     # maps ctrl/scramble/NTC/non-targeting → NTC
   )

   # Or return normalised labels without writing
   labels = cx.normalise_perturbation_labels(
       "data/counts.h5ad",
       column="perturbation",
       inplace=False,
   )

Auto-detecting metadata columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let crispyx infer which obs/var columns hold perturbation labels and gene
symbols:

.. code-block:: python

   # Detect individually
   pert_col = cx.detect_perturbation_column("data/counts.h5ad")
   gene_col  = cx.detect_gene_symbol_column("data/counts.h5ad")

   # Or in one call
   cols = cx.infer_columns("data/counts.h5ad")
   print(cols)
   # {"perturbation_column": "perturbation", "gene_name_column": "gene_symbols"}

   # Pass detected columns directly to downstream functions
   adata = cx.tl.rank_genes_groups(
       "data/counts.h5ad",
       perturbation_column=cols["perturbation_column"],
       gene_name_column=cols["gene_name_column"],
       method="wilcoxon",
   )

Overlap analysis
~~~~~~~~~~~~~~~~

Compare sets of genes or perturbations across datasets:

.. code-block:: python

   result = cx.tl.compute_overlap({
       "Adamson": set(adamson_genes),
       "Replogle": set(replogle_genes),
       "Nadig":    set(nadig_genes),
   })

   print(result.jaccard_matrix)
   print(result.count_matrix)
   print(result.set_sizes)

   # Plot as a heatmap
   ax = cx.pl.overlap_heatmap(result, metric="jaccard", cmap="Blues")
   ax = cx.pl.overlap_heatmap(result, metric="count", annot=True, fmt="d")
