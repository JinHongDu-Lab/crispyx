Usage Guide
===========

crispyx provides a Scanpy-style API for streaming CRISPR screen analysis.
Each operation reads data from disk so large ``.h5ad`` files can be processed
on commodity hardware without loading the full count matrix into memory.

The typical workflow is:

1. **Load** – open a dataset on disk
2. **QC** – filter cells, perturbations, and genes
3. **Preprocess** – normalise and log-transform (streaming)
4. **Dimension reduction** – PCA and KNN graph construction
5. **Pseudo-bulk** – aggregate per perturbation
6. **Differential expression** – t-test, Wilcoxon, or NB-GLM
7. **Plot** – visualise results with Scanpy-style helpers

Quick start
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
   adata_pb = cx.pb.expression_effects(
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

Setting up
----------

Install the project in editable mode with optional dependencies:

.. code-block:: bash

   pip install -e .[test]

Loading data
------------

Use :func:`crispyx.read_h5ad_ondisk` to open a dataset without
materialising the expression matrix and print the first few rows of metadata.
A synthetic demo dataset can be generated via
``python benchmarking/generate_demo_dataset.py`` and stored at
``data/demo_benchmark.h5ad``. The returned :class:`cx.AnnData` object keeps a
backed AnnData handle alive lazily and automatically closes it when the wrapper
is garbage collected. Explicitly call ``adata.close()`` to release the file as
soon as you finish the preview.

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

Batch-level pseudo-bulk profiles
--------------------------------

Use :func:`crispyx.pb.aggregate` to retain an absolute profile for every
observed combination of grouping columns.  Grouping by perturbation and batch
creates repeated measurements suitable for downstream replicate-aware
analyses:

.. code-block:: python

   profiles = cx.pb.aggregate(
       "screen.h5ad",
       groupby=["perturbation", "batch"],
       method="mean_log1p",
       min_cells=5,
       output_path="pseudobulk/screen_by_batch.h5ad",
   )

``method="mean_log1p"`` computes the mean of per-cell log1p values. Raw count
input is log-transformed automatically; continuous non-negative input is
treated as already transformed. ``method="sum"`` is stricter and accepts only
finite, non-negative integer counts, typically supplied with
``layer="counts"``. Set ``bootstrap_size=N`` to sample exactly ``N`` cells once
with replacement from every retained group. The output records grouping and
cell-count columns in ``obs`` and full aggregation provenance under
``uns['crispyx_pseudobulk']``.

``perturbations`` restricts which profiles are returned. A profile is kept when
any of its grouping values matches, so the argument selects on whichever column
holds the labels irrespective of its position in ``groupby``, and every observed
combination of the remaining columns is preserved. With
``groupby=["perturbation", "batch"]``, passing ``perturbations=["KO_A"]`` returns
one profile per batch for ``KO_A``.

Within-batch effects can be calculated from the saved profiles without
re-reading the single-cell matrix:

.. code-block:: python

   effects = cx.pb.effects(
       profiles,
       groupby="perturbation",
       batch_column="batch",
       reference="control",
   )

The default retains one target-minus-control row per perturbation and batch.
Pass ``aggregate_batches=True`` to combine those effects with harmonic-count
weights. Cell-level input is also accepted and is aggregated internally.

Normalised effect sizes in one pass
-----------------------------------

:func:`crispyx.pb.expression_effects` goes from a cell-level file to an effect size
in a single streaming pass. Unlike :func:`crispyx.pb.effects`, which contrasts on
whatever scale its input already carries, this function **normalises library size
itself** -- so do not pre-normalise as well.

``method`` selects which scale the averaging happens on. These are different
estimators, not different spellings: averaging before or after the log gives
different answers whenever expression varies within a group.

.. code-block:: python

   # mean of per-cell log1p values
   effects = cx.pb.expression_effects(
       adata_ro,
       groupby="perturbation",
       method="mean_log1p",
   )

   # log of the mean of normalised counts
   effects = cx.pb.expression_effects(
       adata_ro,
       groupby="perturbation",
       method="log_mean",
       baseline_count=1e4,
   )

The effect is in ``X``, the per-perturbation profile in
``layers['perturbation_profile']``, and the pooled reference in
``uns['control_profile']``. Control-label inference and gene-name fallbacks apply
here as elsewhere, so a well-annotated dataset needs little boilerplate.

Batch-corrected effect sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When cells originate from multiple batches (e.g. 10x lanes, gem-groups, or
experimental replicates), pooling all cells can confound the effect size: if a
perturbation and the control are unevenly represented across batches, a
batch-specific expression profile leaks into the estimated effect.

Pass ``batch_column`` to compute the effect *within each
batch* and combine the per-batch differences with harmonic-count weights
``w_b = n_pert_b · n_ctrl_b / (n_pert_b + n_ctrl_b)``:

.. code-block:: python

   effects = cx.pb.expression_effects(
       adata_ro,
       groupby="perturbation",
       method="log_mean",
       batch_column="batch",          # column in adata.obs
   )

Key properties:

* Batches where a perturbation has no cells (or no control cells) are skipped
  for that perturbation. A perturbation that shares no batch with the control
  raises a ``ValueError``.
* ``layers['perturbation_profile']`` holds the **batch-corrected**
  per-perturbation expression (the harmonic-weighted average of the within-batch
  means), and ``layers['control_profile_matched']`` holds the per-perturbation
  weight-matched control reference, so the identity
  ``X = perturbation_profile − control_profile_matched`` holds exactly.
  ``uns['control_profile']`` retains the pooled control reference.
* The batch column name and the batch labels encountered are recorded in
  ``uns['batch_column']`` and ``uns['batch_ids']``.
* When ``batch_column`` is ``None`` (default), a single pooled effect is
  computed and no ``*_matched`` layer is written. Supplying the column is itself
  the request for batch correction; there is no separate flag.
* **Bounded memory.** The correction keeps the streaming, single-pass design:
  the only quantity that grows with the number of batches -- the
  per-``(perturbation, batch)`` sum accumulator -- is spilled to a disk-backed
  ``numpy.memmap`` and the scatter-add is vectorised. Peak RAM stays
  ``O(chunk × n_genes + n_batches × n_genes + n_perturbations × n_genes)``
  regardless of the number of gem-groups, so genome-wide screens with hundreds
  of batches do not blow up memory.

Memory budget and chunk size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the differential-expression functions, the pseudo-bulk estimators
stream cells in chunks. The cell ``chunk_size`` defaults to ``None`` and is
then auto-selected from the dataset shape and the available memory. Pass
``memory_limit_gb`` to cap the budget in SLURM / cgroup-constrained
environments (the value is forwarded to the chunk-size heuristic exactly as in
``cx.tl.t_test`` / ``cx.tl.nb_glm_test``):

.. code-block:: python

   adata_pb = cx.pb.expression_effects(
       adata_ro,
       groupby="perturbation",
       batch_column="batch",
       memory_limit_gb=128,   # cap the streaming chunk budget
   )

``memory_limit_gb`` and ``chunk_size`` only affect performance/peak memory —
the computed values are identical regardless of the chosen chunk size. Passing
an explicit ``chunk_size`` overrides the auto-selection (and ignores
``memory_limit_gb``).

Dimension Reduction
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

CSC preprocessing for Wilcoxon
------------------------------

For large datasets, convert the preprocessed CSR file to CSC format before
running Wilcoxon DE. CSR storage forces a full scan of all ``data`` and
``indices`` arrays for each gene chunk (O(total_nnz) per chunk); CSC storage
makes each chunk access O(nnz_in_chunk), so total I/O drops from
``n_chunks × file_size`` to ``file_size``. This gives approximately 18×
speedup on large screens (Feng-gwsf: 3.35 h → ~11 min):

.. code-block:: python

   # Convert normalized CSR file to CSC (streaming, no full-matrix load)
   adata_csc = cx.pp.convert_to_csc(
       adata_norm,
       output_dir="results/",
   )
   # adata_csc is returned immediately and unchanged if input is already CSC.
   # Use adata_csc as input to rank_genes_groups() for fast Wilcoxon.

The function auto-detects whether the source file is already CSC and returns
it unchanged with no I/O. In the benchmark pipeline, CSC conversion is
bundled inside ``crispyx_de_wilcoxon`` (via ``run_wilcoxon_with_csc``) so
that the reported wall-time includes both conversion and DE — giving a single
honest total cost rather than a split accounting that would make Wilcoxon
appear faster than it is. Benchmark results include sub-columns
``csc_conversion_seconds``, ``wilcoxon_seconds``, and ``was_already_csc``
for fine-grained breakdown.

.. note::

   For small datasets (files < ~12.5 GB), the CSC conversion overhead is
   negligible (< 1 s on fast NVMe) and the total wilcoxon time is dominated
   by process startup (2–3 s). For large screens (Feng-gwsf 15 GB, Feng-gwsnf
   27 GB) the CSC conversion itself takes ~60–120 s but eliminates the ~18×
   repeated full-file scans that CSR imposes.

CSR preprocessing for NB-GLM
-----------------------------

NB-GLM operations (size factors, control matrix loading, per-perturbation
slicing) are all row-wise. CSC or dense storage makes each row access
O(total_nnz) per slice, causing severe slowdowns or hangs. Convert to CSR
before running NB-GLM:

.. code-block:: python

   # Convert standardised file to CSR (streaming, no full-matrix load)
   adata_csr = cx.pp.convert_to_csr(
       adata,
       output_dir="results/",
   )
   # Returns immediately if already CSR.

   # NB-GLM on CSR file
   result = cx.nb_glm_test(
       adata_csr,
       perturbation_column="perturbation",
   )

The function uses format-aware streaming: CSC sources are read in
column-chunks (axis=1) and scattered into CSR buffers; dense sources are
read in row-chunks (axis=0). For large datasets, the streaming control
statistics function automatically activates — fitting the intercept-only
model in chunks of 4,096 control cells instead of densifying the full
control matrix. This keeps peak memory at O(chunk_size × n_genes) rather
than O(n_control × n_genes). When ``freeze_control`` is auto-enabled,
streaming is used unconditionally. DESeq2-style size factors
(``size_factor_method="deseq2"``) also stream automatically when the
intermediate counts array would exceed 4 GB, computing geometric means
and per-cell median ratios in chunks. After each streaming phase,
``drop_file_cache()`` evicts file data from the kernel page cache so that
cgroup-limited environments (e.g. SLURM) do not count cached pages toward
the memory limit.

.. note::

   If ``nb_glm_test()`` detects CSC storage, it emits a ``UserWarning``
   advising conversion to CSR. In the benchmark pipeline, CSR conversion
   is handled automatically via the ``crispyx_standardize_csr`` step.

Differential expression
-----------------------

Invoke :func:`crispyx.tl.rank_genes_groups` to compare perturbations
against the control population while matching the familiar
:func:`scanpy.tl.rank_genes_groups` interface. Choose ``method="wilcoxon"`` (the
default) for a Mann-Whitney U test, ``method="wald"`` for the streaming Wald
test, or ``method="nb_glm"`` to fit the negative binomial GLM that supports
covariates. The helper reuses the automatic control inference so a missing
``control_label`` triggers the same adaptive search used earlier, and the
returned :class:`cx.AnnData` wrapper stores previews of the results in
``.uns`` so printing ``adata.uns["rank_genes_groups"]`` shows the top genes
per perturbation while ``.load()`` retrieves the full tables on demand.

Scanpy-compatible parameter aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three DE functions (:func:`crispyx.tl.t_test`, :func:`crispyx.tl.wilcoxon_test`,
:func:`crispyx.tl.nb_glm_test`) and :func:`crispyx.tl.rank_genes_groups` accept
the Scanpy-style aliases ``groupby`` (for ``perturbation_column``) and
``reference`` (for ``control_label``), making it easier to port code from
Scanpy workflows:

.. code-block:: python

   # Scanpy-style call (aliases)
   result = cx.tl.rank_genes_groups(
       adata_ro,
       groupby="perturbation",   # alias for perturbation_column
       reference="ctrl",         # alias for control_label
       method="wilcoxon",
   )

   # Equivalent canonical call
   result = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       control_label="ctrl",
       method="wilcoxon",
   )

The canonical names (``perturbation_column`` and ``control_label``) remain the
primary names and are not deprecated. Supplying both a canonical name and its
alias at the same time raises ``TypeError``.

Batch-stratified Wilcoxon (van Elteren test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When cells come from multiple batches (e.g. 10x lanes, gem-groups, or
experimental replicates), a pooled Wilcoxon test can suffer from rank
inflation: cells from a high-count batch will systematically receive higher
ranks, creating spurious hits correlated with batch rather than perturbation.

Pass ``batch_column`` to perform a **batch-stratified (van Elteren) Wilcoxon**
test instead: cells are ranked *within each batch separately* and the
per-stratum U statistics are summed, removing batch-driven rank inflation.

.. code-block:: python

   result = cx.tl.wilcoxon_test(
       adata_csc,
       perturbation_column="perturbation",
       batch_column="batch",          # column in adata.obs
   )

Key properties:

* Low-expression filtering, log-fold changes, and ``pts`` remain pooled across
  all cells — only the rank test is stratified.
* Perturbations that share no batch with any control cell are automatically
  marked untestable (NaN p-values) and logged.
* Diagnostic metadata stored in ``uns``: ``stratified_n_batches``,
  ``stratified_n_control_batches``, ``stratified_n_untestable_perturbations``,
  ``stratified_min_shared_batches_per_perturbation``, and
  ``stratified_median_shared_batches_per_perturbation``.
* The output file receives the suffix ``_cx_wilcoxon_stratified.h5ad`` to
  distinguish it from the pooled result.


on-disk results. The plotting functions materialise only the metadata needed
for plotting, keeping the expression matrix on disk.

PCA Visualization
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run PCA first (if not already done)
   cx.pp.pca(adata_norm, n_comps=50)
   
   # Plot variance explained per component
   cx.pl.pca_variance_ratio(adata_norm, n_pcs=20)
   
   # PCA scatter colored by perturbation
   cx.pl.pca(adata_norm, color='perturbation', components='1,2')
   
   # Gene loadings for top components
   cx.pl.pca_loadings(adata_norm, components=[1, 2, 3])

Differential Expression Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Rank genes groups plot (Scanpy-style)
   cx.pl.rank_genes_groups(adata_de, n_genes=20, sharey=False)

   # Convert DE results into a tidy DataFrame for custom plots
   df = cx.pl.rank_genes_groups_df(adata_de, group="perturbation_A", n_genes=200)

   # Volcano and top-genes plots
   cx.pl.volcano(de_df=df, group="perturbation_A")
   cx.pl.top_genes_bar(de_df=df, group="perturbation_A", topn=15)

   # MA plot using raw counts or normalized log1p means
   cx.pl.ma(
       data=adata_ro,  # raw counts
       de_result=adata_de,
       group="perturbation_A",
       reference="control",
       perturbation_column="perturbation",
       mean_mode="raw",  # or "log1p"
   )

   # QC plotting (composition + summary distributions)
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

NB-GLM options
~~~~~~~~~~~~~~

The negative binomial GLM (``method="nb_glm"``) supports several options:

* **Adaptive chunk size** (default): When ``chunk_size=None`` (the default),
  crispyx automatically calculates an optimal chunk size based on dataset
  dimensions and ``memory_limit_gb``. Small/medium datasets use the maximum
  chunk size (256) for speed, while large memory-constrained datasets use
  smaller chunks to avoid OOM errors. You can still set ``chunk_size``
  explicitly to override automatic selection.

* **Frozen control mode** (``freeze_control``): For datasets with large control
  populations (>100K cells), the control matrix can consume 30+ GB of memory.
  When ``freeze_control=True``, control statistics are pre-computed once and
  shared across workers via memory-mapped files, reducing per-worker memory
  from ~32 GB to <1 GB. This enables full parallelization on large datasets.
  
  **Auto-detection** (default): When ``freeze_control=None``, crispyx automatically
  enables frozen control mode when:
  
  1. Control matrix exceeds 10 GB (control_n × n_genes × 8 bytes > 10 GB)
  2. Standard mode would limit parallelization to <4 workers
  
  This means large datasets like Feng (110K control cells) automatically use
  frozen control mode without user intervention, while smaller datasets maintain
  full flexibility.

* **LFC shrinkage**: For improved accuracy, apply adaptive Cauchy prior 
  shrinkage to log-fold changes using ``shrink_lfc()`` after running 
  ``nb_glm_test()``. This preserves large effects while shrinking 
  small/uncertain effects toward zero.

* **Dispersion sharing** (``share_dispersion=True``): Estimate dispersion once 
  using all cells (similar to PyDESeq2's approach). This provides more stable 
  estimates when sample sizes are small or when you expect homogeneous 
  dispersion across perturbations.

* **Memory limit** (``memory_limit_gb``): Specify the maximum memory available
  for the analysis. For ``nb_glm_test()``, this affects chunk size calculation and
  worker count estimation. For ``wilcoxon_test()``, it controls whether the
  streaming path is used for large datasets (>30% of budget triggers streaming).
  For ``t_test()``, it controls automatic cell chunk size calculation.
  For ``shrink_lfc()``, it limits parallel workers in ``method="full"``.
  For HPC environments with fixed allocations, set this to your SLURM ``--mem``
  value (e.g., ``memory_limit_gb=128``).

* **Scanpy format** (``scanpy_format=True``): Write Scanpy-compatible 
  ``uns["rank_genes_groups"]`` structure for interoperability with 
  ``sc.get.rank_genes_groups_df()`` and similar utilities. This option is 
  available for ``t_test()``, ``wilcoxon_test()``, and ``nb_glm_test()``.
  Default is ``False`` for performance.

.. code-block:: python

   # Basic NB-GLM (faster, per-perturbation dispersion)
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
   )

   # NB-GLM with shared dispersion
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
       share_dispersion=True,
   )

LFC shrinkage
~~~~~~~~~~~~~

For more accurate log-fold change estimates, apply apeGLM shrinkage after
running the NB-GLM test. This two-step workflow matches DESeq2/PyDESeq2
best practices:

.. code-block:: python

   # Step 1: Run NB-GLM test
   result = cx.nb_glm_test(
       adata_ro,
       perturbation_column="perturbation",
   )
   
   # Step 2: Apply LFC shrinkage
   shrunk = cx.shrink_lfc(
       result.result_path,
       prior_scale_mode="global",  # or "per_comparison"
   )

The ``prior_scale_mode`` parameter controls how the shrinkage prior is estimated:

* ``"global"`` (default): Estimate a single prior scale from all comparisons.
  Recommended for CRISPR screens with many perturbations.
* ``"per_comparison"``: Estimate prior scale separately for each perturbation.
  May be more accurate when effect sizes vary substantially across perturbations.

You can also use ``cx.tl.shrink_lfc()`` for API consistency with other tools:

.. code-block:: python

   shrunk = cx.tl.shrink_lfc(
       result.result_path,
       prior_scale_mode="global",
   )

.. _auto-reload:

Auto-reload and re-run control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three DE functions (``wilcoxon_test``, ``t_test``, ``nb_glm_test``) write
their results to an ``.h5ad`` file and remember the output path.  On a
subsequent call with the same parameters, if the output file already exists
the function **loads and returns the saved result instead of rerunning the
analysis** — making notebook and script reruns instant:

.. code-block:: python

   # First call — runs the full analysis and writes crispyx_wilcoxon.h5ad
   result = cx.wilcoxon_test(
       "data.h5ad",
       perturbation_column="perturbation",
       verbose=True,
   )

   # Second call — reloads from disk in milliseconds
   result = cx.wilcoxon_test(
       "data.h5ad",
       perturbation_column="perturbation",
       verbose=True,
   )
   # [crispyx] Loading existing result: data/crispyx_wilcoxon.h5ad
   # [crispyx] Pass force=True to rerun the analysis.

To rerun unconditionally (e.g. after changing ``min_pct_ctrl``,
``memory_limit_gb``, or any other parameter), pass ``force=True``:

.. code-block:: python

   result = cx.wilcoxon_test(
       "data.h5ad",
       perturbation_column="perturbation",
       min_pct_ctrl=0.05,
       min_pct_pert=0.05,
       force=True,   # overwrite the existing file
   )

.. note::

   The auto-reload check is based purely on file existence, not on a
   comparison of the parameters used to produce it.  Always pass ``force=True``
   when you intentionally run with different settings.

Resume and checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~

Long-running differential expression analyses can be resumed after interruption
using the ``resume`` and ``checkpoint_interval`` parameters:

.. code-block:: python

   # Enable checkpointing during long runs
   result = cx.nb_glm_test(
       adata_ro,
       perturbation_column="perturbation",
       resume=True,
       checkpoint_interval=10,  # Save progress every 10 perturbations
   )

If interrupted, simply re-run the same command - completed perturbations will
be skipped automatically. The checkpoint file ``<output>_progress.json`` is
written atomically to prevent corruption.

Benchmarking
------------

The ``benchmarking`` directory ships with a reusable script that measures time
and memory usage across the main methods:

.. code-block:: bash

   cd benchmarking
   ./run_benchmark.sh config/Adamson.yaml

See :doc:`benchmarking` for configuration options and output structure.

Data Preparation Utilities
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
