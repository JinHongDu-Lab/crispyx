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

Published screens arrive in different shapes: metadata columns named
differently, guide names instead of target genes, controls under several
names, counts replaced by normalised values. The utilities below bring a file
to the form the analysis functions expect, editing metadata without reading
``X`` and streaming ``X`` where it has to change. Run
:func:`crispyx.pp.check_data` afterwards (:doc:`quality_control`) to confirm
the result.

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

The lookup names each ID with today's annotation. For a screen annotated
against an older Ensembl release that can rename a gene, or give one symbol to
two IDs that were since merged; joining several screens on such symbols pairs
the wrong genes. Check the result with :func:`crispyx.pp.check_data`, and keep
the Ensembl IDs in ``var`` as the key for joins.

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

When the column holds guide names, ``target_pattern`` takes the target gene
out of each name through one capture group. Write the pattern for the
library's grammar and anchor it on the part *after* the gene: splitting at the
first separator would turn ``NKX2-5`` into ``NKX2`` and merge ``HLA-A`` and
``HLA-B`` into one ``HLA`` target.

.. code-block:: python

   # hCRISPRi-v2 names: GENE_-_172659581.23-P1P2 ; controls: non-targeting_00006
   cx.normalise_perturbation_labels(
       "data/counts.h5ad",
       column="guide",
       strip_suffix_regex=r"_\d+$",           # non-targeting_00006 → non-targeting
       target_pattern=r"^(.+?)_[+-]_\d+",      # NKX2-5_-_172659581.23-P1P2 → NKX2-5
       canonical_control="control",
   )

   # GENE_spacer names: HLA_B_GTGGGAGCCTGGGGGTGAGG (the library writes '-' as '_')
   labels = cx.normalise_perturbation_labels(
       "data/counts.h5ad",
       column="guide",
       target_pattern=r"^(.+)_[ACGT]{18,25}$",
       inplace=False,
   ).str.replace("_", "-")

Recovering counts from normalised data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some screens are published only as ``log1p`` of total-count-normalised
values, while ``nb_glm_test``, pseudo-bulk count sums and
``downsample_counts`` need the counts. :func:`crispyx.pp.recover_counts`
undoes the transform cell by cell, streamed: ``expm1`` leaves each cell's
counts multiplied by one factor, which is recovered from the cell's recorded
total when an ``obs`` column has it (exact), or from its smallest non-zero
value otherwise (exact whenever the cell's smallest count is 1, which holds
for practically every real cell). A value that does not come out an integer
makes the call raise rather than round.

.. code-block:: python

   counts = cx.pp.recover_counts(
       "data/log1p_cp10k.h5ad",
       output_path="data/counts.h5ad",
       n_counts_column="n_counts",   # omit if the file carries no totals
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
