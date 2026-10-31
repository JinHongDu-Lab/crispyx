.. _nb-glm:

NB-GLM: DESeq2 per comparison
=============================

``method="nb_glm"`` (or :func:`crispyx.tl.nb_glm_test`) fits DESeq2's negative
binomial GLM. Give it raw counts: it raises on normalised or log-transformed
values.

.. code-block:: python

   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
   )

Dispersions
-----------

Each perturbation is tested against the controls with DESeq2's steps, run on
that comparison's cells as PyDESeq2 does per perturbation: gene-wise Cox-Reid
dispersions, a mean-dispersion trend, maximum a posteriori dispersions (genes
far above the trend keep their gene-wise value), then the Wald test. Cook's
distance outlier replacement is not done (see :ref:`pydeseq2-parity`).

* **Without covariates**, the controls are summarised once, in one streaming
  pass, so a comparison costs only its own cells and any number of controls
  fits. All 18,293 perturbations of Huang-HCT116 (3.4 million cells, 166,000
  controls) take 6.3 h and 2.8 GB on 32 cores.
* **With covariates** (or ``lfc_shrinkage_type="apeglm"``, or
  ``targets_per_fit > 1``), each comparison is fitted on all its cells. The
  controls are held sparse and shared by the workers, which are the cores
  asked for, each with an equal share of ``memory_limit_gb``; a fit holds
  its genes dense a block at a time, the block sized to half that share, so
  memory grows with the cells times the block rather than times every gene.
  The time still grows with the number of controls, and every fit refits
  them: the first fit's wall time and the number of fits are printed at
  ``verbose=1``, so a multi-day run is visible before it is committed to.
  Where the controls are many and the perturbations small,
  :ref:`nb_offset_test <nb-offset>` fits the controls once.
* ``dispersion=<value>`` fixes one dispersion for every gene;
  ``shrink_dispersion=False`` tests at the gene-wise estimates.

.. _targets-per-fit:

Several perturbations per model
-------------------------------

To reproduce a DESeq2 or PyDESeq2 analysis that puts many targets in one
model, set ``targets_per_fit=k``. The perturbations are fitted ``k`` at a time
with all the controls, one indicator column each. The dispersions are
estimated once per model, mostly from the controls, and each perturbation is
tested on its own column. The default, 1, is DESeq2 run per comparison.

With covariates and many control units this is much faster: about 0.3 s per
target with ``k=50`` against 20 s one at a time, on the genome-wide CD4+
T-cell pseudo-bulk of Zhu *et al.* (2026, *Cell*), who fitted 50 targets per
model (see :doc:`notebooks/residual_effects_genomewide_zhu`).

.. _nb-offset:

Many small perturbations against a stratified control set
----------------------------------------------------------

A genome-wide screen pooled over many 10x lanes and donors can leave each
perturbation with 20-40 cells spread about one per stratum, so that no
pseudo-bulk unit of five cells can be formed and every perturbation falls
back to the cell level. There ``nb_glm_test(covariates=["stratum"])`` holds
every control cell in each fit and solves a design with one column per
stratum: with hundreds of thousands of controls and hundreds of strata, days
to weeks for thousands of perturbations.

:func:`crispyx.tl.nb_offset_test` fits the same model with the stratum
effects taken from the controls rather than estimated jointly. One pass over
the control cells gives each stratum's mean per gene (counts per unit size
factor) and, unless ``dispersion`` is given, a control-only dispersion per
gene (method of moments pooled over strata, the parametric trend below 0.5
counts per cell). Each perturbation is then a one-parameter negative binomial
fit on its own cells, ``y_i ~ NB(s_i m_bg exp(beta), alpha_g)``, by Fisher
scoring; the cost is linear in the perturbed cells, however many controls
there are.

.. code-block:: python

   result = cx.tl.nb_offset_test(
       "data/counts.h5ad",
       perturbation_column="perturbation",
       control_label="control",
       strata="stratum",            # e.g. lane x donor, one string column
       dispersion=None,             # or a per-gene array from the controls
   )
   shrunk = cx.shrink_lfc(result.result_path)   # the result has nb_glm_test's layout

``strata`` is the closed form for one categorical column and builds no
design matrix. For any other design, such as pseudo-bulk units with a
stratum and the log cells per unit, ``covariates`` fits the controls' GLM
once, with DESeq2's dispersion steps on the controls alone, and takes each
unit's baseline from it; ``nb_glm_test`` with the same covariates refits the
whole control set in every fit of ``targets_per_fit`` perturbations, which
on a screen of 10,000 perturbations, 7,000 control units and 90 strata is
about 45 minutes per fit of 50.

.. code-block:: python

   result = cx.tl.nb_offset_test(
       "data/pseudobulk.h5ad",
       perturbation_column="perturbation",
       control_label="control",
       covariates=["stratum", "log10_n_cells"],
       size_factor_method="deseq2",
   )

When each stratum holds thousands of control cells against a perturbation's
handful, the controls dominate the stratum effects and the two fits agree:
on a targeted iPSC-derived screen with every perturbation on the cell
fallback, genes with a control mean of at least 0.5 counts per cell gave
``beta`` correlation 0.9992, median \|Δbeta\| 0.0035 and a median standard
error ratio of 0.99; over all tested genes the offset standard error is
larger at low counts (median ratio 1.15), where ``nb_glm_test`` shrinks
each comparison's dispersion toward its trend and the offset test uses the
control estimate. A stratum whose controls happen to have no count of a gene
gets one pseudo-cell at the pooled control mean, so a single count among the
perturbed cells does not send ``beta`` to its upper bound (the joint fit's
stratum coefficient diverges there). Cells of a stratum without any control
carry no baseline and are left out, with a warning.

Other options
-------------

* ``chunk_size``: cells read per block when computing size factors (2,048 by
  default).
* ``memory_limit_gb``: the memory the analysis may use, capped by what the
  job's cgroup still grants. On a Slurm job, set it to your ``--mem`` value.
  It sets the worker count for ``nb_glm_test`` and ``shrink_lfc``
  (``method="full"``), the gene chunk width for ``wilcoxon_test``, and the
  cell chunk size for ``t_test``. Results are never held in memory: each
  perturbation's (or gene chunk's) values are written to the output file as
  they are computed.
* ``scanpy_format=True`` (``t_test``, ``wilcoxon_test``, ``nb_glm_test``)
  also writes the Scanpy ``uns["rank_genes_groups"]`` structure, for
  ``sc.get.rank_genes_groups_df()`` and similar utilities. Default ``False``.
* Shrink the fold changes afterwards with ``shrink_lfc()``; see
  `LFC shrinkage`_.

**Results are read from the output file.** A result matrix within 1 % of the
memory budget is read once and kept in memory. A larger one is read on access:
``result.pvalues[i]`` and ``result["KO1"]`` read one perturbation's row, and
``np.asarray(result.pvalues)`` reads the matrix. A result holds no open file
handle and follows the file at its path, so a later call that rewrites the same
``output_path`` changes what an earlier result returns. Copy the values
(``np.asarray``) first if you need both.

.. _cell-or-pseudobulk:

Cell level or pseudo-bulk?
--------------------------

Cells that share a guide, donor or lane are not independent replicates. That
is why pseudo-bulk analysis is the usual recommendation when replicate units
exist: sum counts per guide x batch or donor with
``cx.pb.aggregate(method="sum")``, then run ``nb_glm_test`` with
``size_factor_method="deseq2"``.

**What to do:**

1. Check the control cells (:ref:`check-controls`).
2. Run a fake-target check on your screen: relabel control guides as targets,
   as in :doc:`notebooks/residual_effects_genomewide_zhu`.
3. Use pseudo-bulk with the replicate unit as a covariate if cell level fails
   the check and the screen has replicate units with enough cells, or if a
   cell-level fit with covariates takes too long over many controls.
4. Otherwise use cell level, which is more powerful, and check that each hit
   holds across guides.

**The evidence.** We ran both levels on fake targets made of control guides
(no knockdown, but the guide and unit differences real targets have) in ten
screens. Null pairs are expected at \|z\| > 3 in 0.27 % and at p < 0.05 in
5 %. Cell-level NB-GLM:

.. list-table::
   :header-rows: 1

   * - Screens
     - \|z\| > 3
     - p < 0.05
     - Fake targets with hits
   * - Tian CRISPRi, CRISPRa
     - 0.16-0.17 %
     -
     -
   * - Replogle K562 genome-wide, K562 and RPE1 essential; Nadig Jurkat, HepG2
     - 0.31-0.40 %
     - 4.6-5.3 %
     - 40-55 %
   * - Nourreddine iPSC
     - 0.66 %
     - 6.5 %
     - 90 % (median 7 hits)
   * - Frangieh
     - 2 %
     - 10 %
     - all (median 204 hits)
   * - Feng *et al.* (targeted)
     - 2.5 %
     - 11 %
     - all

* Cell level counts guide-to-guide differences that are not the knockdown as
  signal; replicate units absorb them. Feng's controls are 98 % cells without
  a detected guide (see :ref:`check-controls`).
* Pseudo-bulk stayed close to calibrated where it could form units: 0.04-0.36 %
  at \|z\| > 3 (Nourreddine 0.52 %). With the replicate unit as a covariate,
  6-20 % of fake targets had any hit (Nourreddine 56 %).
* Pseudo-bulk needs enough cells per unit. On Nadig HepG2, whose cells spread
  thinly over many GEM groups, only 11 of 174 fake targets formed units of 5
  cells.
* Where both are calibrated, cell level finds more. On Tian, with 4 units per
  perturbation (2 guides x 2 batches), it found 118 genes for SPAST against 19
  for pseudo-bulk, with the same fold changes (r 0.97-0.99): few units widen
  the standard errors.
* With covariates, a cell-level fit holds every control cell in memory, dense.
  A hundred thousand control cells or more need hundreds of gigabytes, so
  ``nb_glm_test`` raises ``MemoryError``; use pseudo-bulk there, or, where no
  pseudo-bulk unit can be formed, :ref:`nb_offset_test <nb-offset>`.

.. _size-factors:

Size factors
------------

A size factor puts each cell's counts on a common scale: the model's mean
for gene *g* in cell *i* is ``s_i * exp(beta_g + effect)``, so a cell with
twice the factor is expected to show twice the counts of every gene. The
factors are computed once for all cells. Where a method needs a reference, the
reference is the **controls**, so a control's factor does not depend on which
perturbations are compared. Scaling all factors by a constant changes no test;
they are reported at geometric mean 1.

``size_factor_method`` chooses among three definitions:

``"libsize"`` (default)
    Each cell's total count. It measures sequencing depth precisely and needs
    no reference. It assumes a perturbation does not change the cell's total
    RNA: when a perturbation strongly changes a few highly expressed genes,
    the total moves with them and every other gene looks shifted the opposite
    way.

``"geomeans"``
    causarray's median of ratios (``comp_size_factor(method="geomeans")``)
    against the controls' geometric means of their non-zero counts, over the
    genes with a control mean of at least 2 that the cell counts. A median
    resists a minority of strongly changed genes. Keeping only well-counted
    genes lets it follow depth: a median of ratios of counts of 1 or 2
    captures only part of a depth difference.

``"deseq2"``
    DESeq2's median of ratios on the genes counted in every control cell
    (``estimateSizeFactors(geoMeans=...)`` with the controls as reference;
    DESeq2's default uses all samples). A perturbed cell uses the reference
    genes counted in every cell of its perturbation. This is the method for
    **pseudo-bulk and bulk** counts. Single cells rarely have more than a few
    dozen such genes (29 on Adamson, 6 on Huang-HCT116); with fewer than five
    it falls back to ``"geomeans"``.

Pass ``size_factors`` to use your own (scran's, or another tool's) instead.

**Which to use.** :doc:`notebooks/size_factor_comparison` compares the three
on semi-synthetic screens with known truth, in single cells and pseudo-bulk.
Library size follows depth best but fails under strong composition shifts (up
to 47 % of null pairs at p < 0.05, in single cells and pseudo-bulk alike). On
pseudo-bulk, ``"geomeans"`` and ``"deseq2"`` stay calibrated in every
scenario. In single cells, no method stays calibrated under a strong
composition shift.

* Single cells: keep the defaults (``"libsize"``, the floor chosen from the
  counts).
* Pseudo-bulk and bulk: use ``"geomeans"`` or ``"deseq2"``.
* Perturbations with large effects on highly expressed genes (strong
  activation, fitness genes): aggregate to pseudo-bulk units (several guides
  or donors per perturbation) and use ``"geomeans"``. At least compare such a
  perturbation's median log fold change over all genes across methods: a
  shift with ``"libsize"`` that ``"geomeans"`` does not show is a composition
  shift.

.. _mean-floor:

The mean floor
--------------

NB-GLM holds fitted means at least ``min_mu`` while fitting, as DESeq2 does.
DESeq2's 0.5 suits bulk counts. On sparse counts it makes tests conservative
and biases genes with means below one when groups differ in depth; 1e-6,
DESeq2's single-cell setting, does not (numbers in
:doc:`notebooks/size_factor_comparison`).

The default, ``min_mu=None``, chooses from the counts: **1e-6 when at least
half of the counts are zero, 0.5 otherwise**. The value used is in
``uns["min_mu"]``. The rule is about sparsity, not the kind of data: very
sparse pseudo-bulk units (Huang-HCT116's 25-cell units) still get 1e-6. Pass a
number to override it, e.g. 0.5 to match a default DESeq2 or PyDESeq2 run.

.. _pydeseq2-parity:

Matching PyDESeq2 or DESeq2
---------------------------

Given the same size factors and mean floor, crispyx's statistics agree with
PyDESeq2's: median difference in z 0.0004-0.004 on six screens (z correlation
0.997-0.9999), about 0.001 on simulated data. With each tool's default floor
(1e-6 here on single cells, 0.5 in PyDESeq2) the median difference is 0.3-0.4.
To compare like with like:

1. Give both the same size factors (``dds.obs["size_factors"]`` in PyDESeq2,
   ``sizeFactors(dds) <- ...`` in DESeq2). A per-perturbation PyDESeq2 run
   otherwise recomputes them on each comparison's cells.
2. Give both the same floor (``min_mu`` / ``minmu``).
3. Set crispyx's expression filters to 0 (``min_pct_ctrl``, ``min_pct_pert``,
   ``min_mean_ctrl``, ``min_mean_pert``, ``min_cells_ctrl``,
   ``min_cells_pert``, ``min_total_count``) so the same genes are tested.
4. Compare p-values, not ``pvalue_adj``: crispyx applies Benjamini-Hochberg per
   perturbation without DESeq2's independent filtering.
5. Turn off Cook's outlier handling in PyDESeq2 (``refit_cooks=False``,
   ``cooks_filter=False``). crispyx does not replace outliers: on four Adamson
   perturbations PyDESeq2 replaced them in 4-15 of ~18,800 genes and changed
   the hits at q < 0.1 by 0-3.

Gene-wise dispersions can still differ on sparse genes, most often in
pseudo-bulk: where the likelihood has a second mode at the lower bound
(1e-8), DESeq2's local search can stop there, while crispyx reports the
higher-likelihood dispersion. The tests hardly differ: on Tian CRISPRa
pseudo-bulk the median z difference was 0.001-0.015, the hits differed by a few
genes (crispyx found 13 for SPAST, PyDESeq2 16), and fake targets got the same
false hits.

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

The default ``method="stats"`` takes each entry's posterior mode from its
MLE and standard error in closed form (:func:`crispyx.glm.cauchy_map`: the
real roots of a cubic, the one with the lowest objective), so a result of a
hundred million entries shrinks in under a minute and every entry is at the
exact optimum.

For effects that did not come from one ``nb_glm_test`` file (several results
combined, a subset, another estimator's output with a standard error),
:func:`crispyx.tl.shrink_lfc_arrays` applies the same shrinkage to arrays
and returns the prior scale it used. Two arguments handle a case the prior
gets wrong: under a genome-wide prior scale (0.02 on a screen of thousands
of perturbations), apeGLM acts like a threshold, taking \|z\| of 3-5 to about
zero, and that also zeroes each perturbation's own knockdown, where a shift
is expected. ``exclude`` keeps those entries as given and leaves them out of
the prior estimate; ``floor`` raises a fold change at the fitted-mean floor
(every count zero in the perturbed cells: -7 on pseudo-bulk with
``min_mu=0.01``, -12 to -16 on cells) to a stated bound such as
``ln(0.5 / expected count)``, the strongest knockdown zero counts can show.

.. code-block:: python

   ln_lfc = result.result["layers/logfoldchange_raw_ln"]   # or X with lfc_base="ln"
   ln_se = result.result["layers/standard_error_ln"]
   own_target = np.zeros(ln_lfc.shape, dtype=bool)         # each row's target gene column
   shrunk = cx.tl.shrink_lfc_arrays(ln_lfc, ln_se, exclude=own_target)
   shrunk.lfc, shrunk.se, shrunk.prior_scale
