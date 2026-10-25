Differential expression
=======================

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
-----------------------------------

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
--------------------------------------------

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

For effect sizes and standard errors to go with the stratified p-values, and how
this compares with the other ways of adjusting for batches, see
:doc:`covariate_adjustment`.

Permutation nulls without copying the data
------------------------------------------

``wilcoxon_test`` also takes the labels themselves as a ``pd.Series`` indexed
by cell, so a label-permutation null needs no relabelled copy of the file:

.. code-block:: python

   labels = cx.load_obs("screen_csc.h5ad")["perturbation"]
   rng = np.random.default_rng(0)
   for i in range(100):
       shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index, name="shuffle")
       null = cx.tl.wilcoxon_test(
           "screen_csc.h5ad", perturbation_column=shuffled, control_label="control",
           output_path=f"null/shuffle_{i}.h5ad",
       )

The Series is aligned to ``obs_names`` by its index and needs a label for
every cell; its ``name`` names the column in the result. Each shuffle is a
different call, so a result at a fixed ``output_path`` is recomputed for a new
Series and reused for the same one. Convert a CSR file to CSC once beforehand
(:func:`crispyx.convert_to_csc`): otherwise every call makes its own temporary
CSC copy.


on-disk results. The plotting functions materialise only the metadata needed
for plotting, keeping the expression matrix on disk.

PCA visualization
-----------------

.. code-block:: python

   # Run PCA first (if not already done)
   cx.pp.pca(adata_norm, n_comps=50)
   
   # Plot variance explained per component
   cx.pl.pca_variance_ratio(adata_norm, n_pcs=20)
   
   # PCA scatter colored by perturbation
   cx.pl.pca(adata_norm, color='perturbation', components='1,2')
   
   # Gene loadings for top components
   cx.pl.pca_loadings(adata_norm, components=[1, 2, 3])

Differential expression visualization
-------------------------------------

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
--------------

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

* **Dispersion scope** (``dispersion_scope``): ``"global"`` (the default)
  estimates each gene's dispersion once and uses it in every comparison -- from
  all cells without covariates, and from the control cells with the covariate
  design when ``covariates`` are given. ``"per_comparison"`` runs DESeq2's
  estimator on each comparison's cells (the controls and one perturbation), as
  PyDESeq2 does when run per perturbation. Without covariates the controls are
  summarised once, so a comparison costs its own cells only (about half a
  second on 32 cores for a genome-wide screen's perturbation) and controls too
  large for memory are streamed.

* **Size factors** (``size_factor_method``, ``size_factors``): see
  :ref:`size-factors` below.

* **Memory limit** (``memory_limit_gb``): Specify the maximum memory available
  for the analysis; it is capped by what the job's cgroup will still grant.
  For ``nb_glm_test()``, this affects chunk size calculation and
  worker count estimation. For ``wilcoxon_test()``, it sets the gene chunk
  width (one chunk's cells and every perturbation's results for its genes are
  held at a time). For ``t_test()``, it controls automatic cell chunk size
  calculation. For ``shrink_lfc()``, it limits parallel workers in
  ``method="full"``. The results themselves are never held in memory: each
  perturbation's (or gene chunk's) values are written into the output file as
  they are computed. For HPC environments with fixed allocations, set this to
  your SLURM ``--mem`` value (e.g., ``memory_limit_gb=128``).

* **Results are read from the output file.** The returned
  ``RankGenesGroupsResult`` reads its matrices from the file: a matrix within
  1% of the memory budget (any result on a personal machine, short of a
  genome-wide screen) is read once and kept in memory; a larger one is read on
  access, so ``result.pvalues[i]`` and ``result["KO1"]`` cost one
  perturbation's row however large the result is, and
  ``np.asarray(result.pvalues)`` reads the matrix. ``result.order`` is
  computed for the rows read. A result holds no open handle on its file, and it
  follows the file at its path: a later call that rewrites the same
  ``output_path`` changes what an earlier result object returns; copy the
  values (``np.asarray``) first if you need both.

* **Scanpy format** (``scanpy_format=True``): Write Scanpy-compatible 
  ``uns["rank_genes_groups"]`` structure for interoperability with 
  ``sc.get.rank_genes_groups_df()`` and similar utilities. This option is 
  available for ``t_test()``, ``wilcoxon_test()``, and ``nb_glm_test()``.
  Default is ``False`` for performance.

.. code-block:: python

   # NB-GLM, one dispersion per gene shared by every comparison (the default)
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
   )

.. _size-factors:

Size factors
------------

A size factor puts each cell's counts on a common scale: the model's mean
for gene *g* in cell *i* is ``s_i * exp(beta_g + effect)``, so a cell with
twice the factor is expected to show twice the counts of every gene. The
factors are computed once for all cells; where a method needs a reference,
the reference is the **controls**, so a control's factor does not depend on
which perturbations the file holds or which one is compared. Multiplying all
factors by a constant changes no test (the intercept absorbs it); they are
reported at geometric mean 1.

``size_factor_method`` chooses among three definitions:

``"libsize"`` (default)
    Each cell's total count. It measures sequencing depth precisely (a sum
    over thousands of counts) and needs no reference. It assumes a
    perturbation does not change the cell's total captured RNA: when a
    perturbation strongly changes a few very highly expressed genes, or
    transcription overall, the total moves with it and every other gene
    looks shifted the opposite way.

``"geomeans"``
    causarray's median of ratios (``comp_size_factor(method="geomeans")``),
    with the controls as the reference: a cell's factor is the median of
    its ratios to the controls' geometric means (of their non-zero counts)
    over the genes with a control mean of at least 2 that the cell counts.
    A minority of strongly changed genes does not move a median, so it
    resists the composition shifts library size does not. Keeping only
    well-counted genes is what lets it follow depth: most single-cell
    counts are 1 or 2, and a median of their ratios captures only part of
    a depth difference.

``"deseq2"``
    DESeq2's median of ratios, on the genes counted in every control cell
    (the reference ``estimateSizeFactors(geoMeans=...)`` takes, from the
    controls; DESeq2's default takes it from all samples). A perturbed cell
    uses the reference genes counted in every cell of its perturbation.
    This is the method for **pseudo-bulk and bulk** counts, where thousands
    of genes are counted in every sample. Single cells rarely have more
    than a few dozen such genes (29 on Adamson, 6 on Huang-HCT116); with
    fewer than five it falls back to ``"geomeans"``.

Pass ``size_factors`` to use your own (scran's, or another tool's to compare
like with like) instead.

**How they behave: semi-synthetic screens with known truth.** Fake
perturbations are random groups of 150 real control cells (from Adamson, and
from 20,000 controls of Huang-HCT116), changed in known ways; every other
(perturbation, gene) pair is a true null:

* **depth** -- every count of the group thinned to 60 %: a pure difference in
  sequencing depth, no gene changed;
* **weak** -- 30 genes changed 1.4-2-fold, half up and half down;
* **strong up / strong down** -- 300 of the 1,000 most expressed genes up or
  down 3-fold, which changes the cells' total RNA (a composition shift).

Each method ran per-comparison NB-GLM with default settings on the single
cells, and on pseudo-bulk units of 25 cells of one label (6 units per
perturbation, 185 and 680 control units). The false-positive rate is the share
of null pairs with p < 0.05 (5 % expected); Adamson / Huang:

.. list-table:: False-positive rate (%), single cells
   :header-rows: 1

   * - scenario
     - ``"libsize"``
     - ``"geomeans"``
     - ``"deseq2"``
   * - depth
     - 4.9 / 4.0
     - 3.6 / 3.8 (dense genes 14 / 15)
     - 5.6 / 3.9
   * - weak
     - 1.7 / 1.3
     - 1.7 / 1.3
     - 1.7 / 1.2
   * - strong up
     - **40.7 / 39.7**
     - 15.9 / 11.6
     - 6.9 / 20.8
   * - strong down
     - 16.4 / 12.5
     - 15.4 / 13.4
     - 11.9 / 27.0

.. list-table:: False-positive rate (%), pseudo-bulk
   :header-rows: 1

   * - scenario
     - ``"libsize"``
     - ``"geomeans"``
     - ``"deseq2"``
   * - depth
     - 3.6 / 3.8
     - 3.6 / 3.8
     - 3.7 / 3.8
   * - weak
     - 3.4 / 3.0
     - 3.4 / 3.0
     - 3.4 / 3.0
   * - strong up
     - **45.5 / 43.2**
     - 3.3 / 3.3
     - 3.3 / 3.3
   * - strong down
     - **20.8 / 16.0**
     - 3.3 / 3.3
     - 3.4 / 3.3

What each method is good for:

* **Library size** follows depth best: under pure depth thinning its tests on
  well-counted genes (control mean >= 1) are calibrated in single cells (5 %
  of null pairs at p < 0.05) where ``"geomeans"`` reaches 14-15 %: a median
  of ratios over a cell's non-zero counts is pushed up when thinning turns
  counts into zeros. It fails under composition shifts, in single cells and
  pseudo-bulk alike: up to 46 % of null pairs and 1,100-7,700 false hits
  (q < 0.1) per perturbation, because the total moves with the changed genes.
* **Geometric means** and **DESeq2's ratio** resist composition shifts on
  pseudo-bulk completely (3.3 % of null pairs, ~25 false hits per perturbation
  against 300 changed genes, all of which are found): pseudo-bulk counts are
  large, so a median of ratios is both depth-accurate and robust.
* In **single cells no method stays calibrated under a strong composition
  shift** (well-counted genes: 31-100 % of null pairs at p < 0.05): a median
  over counts of 1-5 resists a shift far less than over pseudo-bulk counts.
  The ratio method is the least affected when many genes are counted in every
  control (Adamson, 29 genes) and the most when few are (Huang, 6 genes,
  some of them among the changed ones).
* Under **weak perturbations** all three agree, at both levels.

**On real cells.** Fake perturbations drawn from the deepest 30 % of
Huang-HCT116 and Adamson controls (a real depth difference, which also brings
some biology along) tell the same story for depth: mean null z on
well-counted genes +0.11 / -0.08 with library size, +0.97 / +0.29 with
``"geomeans"``, +0.94 / -0.48 with ``"deseq2"``, and +5.6 / +5.3 with a median
of ratios over every expressed gene (``"sparse"`` before 0.2.3, removed:
6,578 false hits per fake perturbation on Huang, against 22 with library
size).

**The mean floor matters for single cells.** ``nb_glm_test`` floors fitted
means at ``min_mu=0.5``, DESeq2's default for bulk counts. On single cells
that makes tests conservative (1.2-1.7 % of null pairs at p < 0.05 for weak
perturbations, power 0.42-0.44) and biases genes with means below about one
count when groups differ in depth (mean null z +1.2 on Adamson). With
``min_mu=1e-6``, DESeq2's own setting for single-cell counts, the weak
scenario is calibrated (4.5-5.1 %, power 0.51-0.53) and the depth bias goes;
the comparisons between size factors are unchanged.

**In practice:**

* single cells: keep the default (``"libsize"``); consider ``min_mu=1e-6``;
* pseudo-bulk and bulk: use ``"geomeans"`` or ``"deseq2"``;
* perturbations with large effects on highly expressed genes (strong
  activation, fitness genes): no single-cell size factor is reliable for
  them; aggregate to pseudo-bulk units (several guides or donors per
  perturbation) and use ``"geomeans"``, or at least compare each such
  perturbation's median log fold change over all genes across methods -- a
  whole-perturbation shift with ``"libsize"`` that ``"geomeans"`` does not
  show is a composition shift;
* to compare with PyDESeq2 or DESeq2, give them the same size factors
  (``dds.obs["size_factors"]``, ``sizeFactors(dds) <- ...``); a
  per-perturbation PyDESeq2 run otherwise recomputes them on each
  comparison's cells, and that difference, not the estimator, then
  dominates the comparison.

These are semi-synthetic screens built from two datasets' control cells; a
benchmark across the real screens (and their pseudo-bulk forms) is planned.

LFC shrinkage
-------------

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
