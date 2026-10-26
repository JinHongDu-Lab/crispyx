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
  crispyx calculates the number of cells read per block for the size factors
  from the dataset's dimensions and ``memory_limit_gb``. You can still set
  ``chunk_size`` explicitly to override automatic selection.

* **Dispersions, per comparison**: each perturbation is tested against the
  controls with DESeq2's estimator run on that comparison's cells, as
  PyDESeq2 does when run per perturbation: gene-wise Cox-Reid dispersions, a
  mean-dispersion trend, maximum a posteriori dispersions under a log-normal
  prior around it (genes far above the trend keep their gene-wise value), then
  the Wald test. Given the same size factors the statistics agree with
  PyDESeq2's to about 0.001 in z. Cook's distance outlier replacement is not
  done: on four Adamson perturbations against 7,629 controls, PyDESeq2's
  replaced outliers in 4-15 of ~18,800 genes, moved z by more than 0.5 in
  2-14 and changed the hits at q < 0.1 by 0-3. Without covariates the controls are summarised once, in one
  streaming pass, so a comparison costs its own cells only (about half a
  second on 32 cores for a genome-wide screen's perturbation) and controls of
  any size fit. With ``covariates`` (or ``lfc_shrinkage_type="apeglm"``) each
  comparison is fitted on all its cells, with the controls held in memory;
  controls too large for that raise ``MemoryError``, and aggregating to
  pseudo-bulk first (``cx.pb.aggregate``) is the way through. ``dispersion``
  fixes one value for every gene; ``shrink_dispersion=False`` tests at the
  gene-wise estimates.

* **Cell level or pseudo-bulk?** Cells that share a guide, donor or lane are
  not independent replicates, which is why pseudo-bulk analysis (counts
  summed per guide x batch or donor with ``cx.pb.aggregate(method="sum")``,
  then ``nb_glm_test`` with ``size_factor_method="deseq2"``) is the usual
  recommendation when replicate units exist. On two Tian iPSC-neuron screens
  (CRISPRi and CRISPRa) we compared the two levels directly:

  - Cell-level NB-GLM stayed calibrated against fake targets made from
    control guides (each non-targeting guide tested against the others):
    0.12-0.25 % of null pairs at \|z\| > 3, against 0.27 % expected.
  - It found far more hits than pseudo-bulk with 4 units per perturbation
    (2 guides x 2 batches), e.g. 118 against 19 genes for SPAST. The
    pseudo-bulk fold changes agreed with the cell-level ones (r 0.97-0.99);
    pseudo-bulk loses power through wider standard errors when units are
    few, and by counting the differences between a perturbation's guides as
    noise. crispyx, PyDESeq2 and R DESeq2 called the same pseudo-bulk hits.
  - The fake-target check cannot show whether cell level's extra hits rely
    on a perturbation's guides agreeing (control guides have no knockdown
    to differ in), and on several Tian targets the two guides' fold changes
    at the cell-level hits agreed poorly.

  In practice: test at cell level by default; aggregate to pseudo-bulk when
  each perturbation has several replicate units, or to check that a hit
  holds across guides and donors; and run a fake-target check (control
  guides relabelled as targets) on your own screen either way. These are two
  screens and ten perturbations each; a comparison across screens is
  planned.

* **LFC shrinkage**: For improved accuracy, apply adaptive Cauchy prior
  shrinkage to log-fold changes using ``shrink_lfc()`` after running
  ``nb_glm_test()``. This preserves large effects while shrinking
  small/uncertain effects toward zero.

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

   # NB-GLM, DESeq2's dispersions estimated per comparison
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

**How they behave.** :doc:`notebooks/size_factor_comparison` compares the
three on semi-synthetic screens with known truth -- fake perturbations of
real control cells from Adamson and Huang-HCT116, changed by depth thinning,
weakly (30 genes), or by a strong composition shift (300 highly expressed
genes 3-fold up or down) -- in single cells and in pseudo-bulk units of 25
cells, and on real-cell fake perturbations. The false-positive rate is the
share of null (perturbation, gene) pairs with p < 0.05; the evidence came
from single cells first, and pseudo-bulk was compared after. In short:

* **Library size** follows depth best: under pure depth thinning its tests
  are calibrated in single cells (5-6 %), where ``"geomeans"`` reaches
  14-16 % on well-counted genes. It fails under composition shifts, in single
  cells and pseudo-bulk alike (up to 47 % of null pairs), because the total
  moves with the changed genes.
* **Geometric means** and **DESeq2's ratio** resist composition shifts on
  pseudo-bulk completely (3-5 % of null pairs, every changed gene found):
  pseudo-bulk counts are large, so a median of ratios is both depth-accurate
  and robust.
* In **single cells no method stays calibrated under a strong composition
  shift**: a median over counts of 1-5 resists a shift far less than over
  pseudo-bulk counts.
* Under **weak perturbations** all three agree, at both levels.
* A median of ratios over every expressed gene (``"sparse"`` before 0.2.3,
  and DESeq2's ``poscounts``) mistakes depth for effect: fake perturbations of
  the deepest controls get thousands of false hits each.

**The mean floor** (``min_mu``). NB-GLM holds fitted means at least ``min_mu``
while fitting, as DESeq2 does. DESeq2's 0.5 suits bulk counts; on sparse
counts many genes have means below one, the floor binds more often in the
shallower group, and tests become conservative and biased for those genes when
groups differ in depth. In the notebook's single-cell screens, 0.5 holds weak
perturbations' tests at 1.2-1.7 % of null pairs (power 0.42-0.44) and biases
moderately expressed genes under depth thinning; 1e-6, DESeq2's own
single-cell setting, gives 4.5-5.1 % (power 0.51-0.53) and no depth bias. The
default, ``min_mu=None``, therefore chooses from the counts themselves: **1e-6
when at least half of the counts are zero, 0.5 otherwise** (the value used is
in ``uns["min_mu"]``). The rule is about sparsity, not the kind of data:
Adamson's 25-cell pseudo-bulk units get 0.5, but Huang-HCT116's, whose 36,000
genes are mostly very sparse, are still more than half zeros and get 1e-6 --
appropriately, as their tests are then closer to 5 %. Pass a number to
override it; pass 0.5 to match a default PyDESeq2 or DESeq2 run.

**In practice:**

* single cells: keep the defaults (``"libsize"``, the floor chosen from the
  counts);
* pseudo-bulk and bulk: use ``"geomeans"`` or ``"deseq2"``;
* perturbations with large effects on highly expressed genes (strong
  activation, fitness genes): no single-cell size factor is reliable for
  them; aggregate to pseudo-bulk units (several guides or donors per
  perturbation) and use ``"geomeans"``, or at least compare each such
  perturbation's median log fold change over all genes across methods -- a
  whole-perturbation shift with ``"libsize"`` that ``"geomeans"`` does not
  show is a composition shift;
* to compare with PyDESeq2 or DESeq2, give them the same size factors
  (``dds.obs["size_factors"]``, ``sizeFactors(dds) <- ...``) and the same
  floor (``min_mu`` / ``minmu``); a per-perturbation PyDESeq2 run otherwise
  recomputes the factors on each comparison's cells, and that difference,
  not the estimator, then dominates the comparison. Set crispyx's
  expression filters to 0 (``min_pct_ctrl``, ``min_pct_pert``,
  ``min_mean_ctrl``, ``min_mean_pert``, ``min_cells_ctrl``,
  ``min_cells_pert``, ``min_total_count``) so the same genes are tested, and
  compare p-values: crispyx's ``pvalue_adj`` is Benjamini-Hochberg per
  perturbation without DESeq2's independent filtering, and Cook's outliers
  are not replaced (``refit_cooks=False``, ``cooks_filter=False``).

The comparison uses two datasets' control cells and one seed; a benchmark
across the real screens (and their pseudo-bulk forms) is planned.

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
