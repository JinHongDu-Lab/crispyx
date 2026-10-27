Covariate adjustment
====================

In a large screen, a target's cells and the control cells rarely come from the
same donors, lanes or cell lines in the same proportions. A plain comparison with
the controls then reports those differences as an effect of the perturbation.
crispyx offers several ways to adjust for recorded covariates. They differ in
what they can adjust for, the scale of the effect they report, and how their
significance is calibrated.

All of them adjust only for covariates that were recorded. If a target's cells
may differ from the controls in ways no recorded covariate captures (an
unrecorded batch, a cell state), use `causarray <https://causarray.readthedocs.io>`_
(Du *et al.*, 2026, *Briefings in Bioinformatics*,
`doi:10.1093/bib/bbag175 <https://doi.org/10.1093/bib/bbag175>`_), which estimates
unmeasured confounders and adjusts for them.

.. note::

   Adjust for how cells were processed (donor, lane, batch, cell line), which no
   perturbation changes. Avoid what the perturbation itself can change, such as
   sequencing depth, mitochondrial share or cell cycle: adjusting for it removes
   part of the effect you want to measure.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 18 17 12 18

   * - Approach
     - Adjusts for
     - Effect
     - Significance
     - Speed
     - Tutorial
   * - Stratified effect + Wilcoxon
     - one categorical column
     - difference of mean ``log1p(CP10k)``, with standard errors
     - rank test within strata
     - fast
     - :doc:`batch-corrected <notebooks/stratified_wilcoxon_tutorial>`
   * - ``residual_effects``, ``null="robust"``
     - any covariates
     - mean residual, ``log1p(CP10k)``
     - spread across targets
     - fastest (one pass)
     - :doc:`genome-wide <notebooks/residual_effects_tutorial>`
   * - ``residual_effects``, ``null="model"``
     - any covariates
     - mean residual, ``log1p(CP10k)``
     - control fit's noise
     - fastest (one pass)
     - :doc:`targeted <notebooks/residual_effects_feng_tutorial>`
   * - NB-GLM on pseudo-bulk
     - any covariates, replicates
     - log2 fold change
     - Wald test, replicate units
     - slow
     - :doc:`genome-wide <notebooks/residual_effects_tutorial>`
   * - NB-GLM on single cells
     - any covariates
     - log2 fold change
     - Wald test, cells as units
     - slowest
     - —
   * - ``cx.pb.effects`` per batch
     - one categorical column
     - difference on the input's scale
     - none
     - fast
     - —

Stratified effects and the batch-corrected Wilcoxon test
--------------------------------------------------------

Each batch is compared with its own controls, and the per-batch results are
combined, as in Liu *et al.* (2026, *bioRxiv*,
`doi:10.64898/2026.10.05.756839 <https://doi.org/10.64898/2026.10.05.756839>`_).
``cx.pb.normalized_effects(..., method="mean_log1p", batch_column=...)``
gives the effect: the difference of mean ``log1p(CP10k)`` within each batch,
combined with weights ``n_t n_c / (n_t + n_c)``, the estimate of a regression
with one intercept per batch and one perturbation coefficient. It also writes
two standard errors: ``layers["standard_error"]`` assumes independent cells and
one within-batch variance per arm, and ``layers["standard_error_robust"]`` is
taken from the spread of the per-batch effects (it needs two batches or more).
``cx.wilcoxon_test(..., batch_column=...)`` gives the p-values: cells are ranked
within each batch and the Mann-Whitney statistics are summed across batches (van
Elteren's test).

.. code-block:: python

   effects = cx.pb.normalized_effects(path, perturbation_column="target",
                                      control_label="non-targeting",
                                      method="mean_log1p", batch_column="batch")
   tests = cx.wilcoxon_test(path, perturbation_column="target",
                            control_label="non-targeting", batch_column="batch")

- **Strengths:** makes no assumption about the distribution of expression, and
  is robust to outlying cells. A batch effect cannot leak into the result, since
  cells are only ever compared within a batch.
- **Weaknesses:** one categorical column only (combine several into one column
  beforehand); continuous covariates cannot be adjusted for. Cells in batches
  without controls are not used, and a target sharing no batch with the controls
  is untested. Ranks of mostly zero counts carry less information than means: on
  the Feng et al. targeted screen it recovered 61–67% of the published hits
  where ``residual_effects`` recovered 97–99%. The Wilcoxon result's own
  ``logfoldchanges`` are pooled over batches; take effects and standard errors
  from ``normalized_effects``. The model-based standard error treats cells as
  independent, so it is too small when cells of one guide or donor move
  together; the robust one does not assume that but is noisy with few batches.
- **Tutorials:** :doc:`notebooks/stratified_wilcoxon_tutorial` (effects,
  standard errors and p-values per batch on the Feng *et al.* targeted
  screen, with a calibration check on fake targets); Part 3 of
  :doc:`notebooks/residual_effects_feng_tutorial` compares pooled and
  stratified controls for the Wilcoxon test alone.

``residual_effects``: effects from a control-fitted model
---------------------------------------------------------

``cx.tl.residual_effects`` (Feng *et al.*, 2026, *Cell Genomics*,
`doi:10.1016/j.xgen.2025.101076 <https://doi.org/10.1016/j.xgen.2025.101076>`_)
fits each gene's ``log1p(CP10k)`` to the covariates on the control cells only;
a target's effect is the mean, over its cells, of observed minus fitted
expression. It takes any mix of categorical and numeric ``obs`` columns, plus
``"log_total_counts"`` and ``"mito_fraction"`` computed from the counts, in one
pass over the raw counts.

.. code-block:: python

   result = cx.tl.residual_effects(path, "target", "non-targeting",
                                   covariates=["donor", "lane"], null="robust")

The two nulls turn an effect into a z-score differently:

- ``null="robust"`` divides by each gene's spread across targets (1.4826 x the
  median absolute deviation). It assumes most targets leave a given gene
  unchanged, which holds in a genome-wide screen and fails in a focused panel
  where every target is active. On fake targets built from non-targeting guides
  it is calibrated for a typical target (4.7% of p-values below 0.05).
- ``null="model"`` divides by the control fit's residual standard deviation,
  Feng *et al.*'s test. Its noise comes from the controls, so even groups of 10
  cells are tested reliably. It treats cells as independent, while cells sharing
  a guide or a donor are not: on Zhu *et al.*'s genome-wide CD4+ T-cell screen
  it gave about 300 hits to every fake target.

- **Strengths:** about two minutes for every target of a 7-million-cell screen;
  continuous and categorical covariates; small groups, including per-line or
  per-donor effects (label cells ``target|line``).
- **Weaknesses:** the effect is a difference of ``log1p(CP10k)``, not a log2 fold
  change. Every covariate level must occur among the controls (perturbed cells in
  other levels are left out). The robust null needs a genome-wide screen; the
  model null over-calls when cells are not independent.
- **Tutorials:** :doc:`notebooks/residual_effects_tutorial` (genome-wide,
  ``null="robust"``, calibration on fake targets) and
  :doc:`notebooks/residual_effects_feng_tutorial` (targeted, ``null="model"``,
  reproduces the published effects and calls).

NB-GLM with covariates
----------------------

``cx.nb_glm_test(..., covariates=[...])`` fits a negative binomial model to the
counts with the covariates and a perturbation indicator in the design, as DESeq2
does; ``cx.shrink_lfc`` shrinks the fold changes afterwards. It runs on single
cells or on pseudo-bulk counts. With many control units, fit many targets per
model with ``targets_per_fit`` (:ref:`targets-per-fit`).

On **pseudo-bulk**, counts are summed per guide and donor (or other replicate
unit) with ``cx.pb.aggregate(..., method="sum")``. Each unit is a replicate, so
the test accounts for variation between guides and donors.

.. code-block:: python

   units = cx.pb.aggregate(path, groupby=["guide", "donor"], method="sum",
                           output_path="units.h5ad")
   # add each unit's target and log10(cells) to its obs first (cx.write_obs);
   # the tutorial shows how
   result = cx.nb_glm_test("units.h5ad", perturbation_column="target",
                           control_label="non-targeting",
                           covariates=["log10_n_cells", "donor"],
                           size_factor_method="deseq2",
                           targets_per_fit=50)  # as Zhu et al.

- **Strengths:** a count model with log2 fold changes and standard errors. On
  pseudo-bulk it reproduces Zhu *et al.*'s PyDESeq2 estimates of their
  CD4+ T-cell screen (median per-target r 0.9999) and is the best calibrated of
  these methods on fake targets.
- **Weaknesses:** slower: on pseudo-bulk, about 0.3 seconds per target on 32
  cores with ``targets_per_fit=50``, so about an hour for a genome-wide screen.
  Pseudo-bulk needs replicate units (several guides or donors per target); with
  one donor and one guide there is little to estimate the variation from. On
  single cells, cells are again treated as independent, and with covariates the
  control cells must fit in memory (see :ref:`cell-or-pseudobulk`).
- **Tutorial:** Part 2 of :doc:`notebooks/residual_effects_tutorial`.

Per-batch pseudo-bulk effects
-----------------------------

``cx.pb.effects(..., batch_column=..., aggregate_batches=True)`` contrasts each
target's pseudo-bulk profile with the same batch's control profile and combines
the batches with the same weights as above. It reports effects only, with no
test: use it to look at effect sizes per batch (``aggregate_batches=False``) or
to feed a downstream model.

Custom statistics per batch
---------------------------

``cx.tl.batch_process`` streams any statistic you define within each batch, or
for each target against the controls of its batch, and combines the batches with
weights you choose. Use it when none of the above computes what you need. See
:doc:`notebooks/batch_statistics_tutorial`. ``cx.t_test`` itself does not
stratify by batch.

Which to use
------------

- **Genome-wide screen:** ``residual_effects`` with ``null="robust"`` on every
  target for a first pass, then the NB-GLM on pseudo-bulk for the targets that
  matter, or for all of them when compute allows. Check calibration on fake
  targets made of non-targeting guides, as the CD4+ T-cell tutorial does.
- **Targeted screen**, where most targets are active: ``residual_effects`` with
  ``null="model"``, or the NB-GLM on pseudo-bulk when there are replicate units.
- **One batch column and a test free of distributional assumptions:** stratified
  effects with the batch-corrected Wilcoxon test.
- **Effects within each donor, line or batch:** ``residual_effects`` with
  ``target|batch`` labels, or ``cx.pb.effects`` with ``aggregate_batches=False``.
- **Confounding you did not record:** causarray.

Tutorials
---------

Each approach above on a real screen, with its calibration checked:

- :doc:`notebooks/stratified_wilcoxon_tutorial`
- :doc:`notebooks/residual_effects_feng_tutorial`
- :doc:`notebooks/residual_effects_tutorial`
