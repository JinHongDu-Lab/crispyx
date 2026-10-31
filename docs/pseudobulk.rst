Pseudo-bulk and effect sizes
============================

Aggregating cells into pseudo-bulk profiles, and effect sizes against the controls, pooled or within each batch.


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

``method`` has no default, because each choice is wrong for the other use.
``method="sum"`` sums raw counts -- what count models such as DESeq2, edgeR,
:func:`crispyx.nb_glm_test` or causarray need -- and accepts only finite,
non-negative integer counts, typically supplied with ``layer="counts"``.
``method="mean_log1p"`` computes the mean of per-cell log1p values for
contrasts on the log scale; raw count input is log-transformed automatically
and continuous non-negative input is treated as already transformed. Set
``bootstrap_size=N`` to sample exactly ``N`` cells once with replacement from
every retained group.

Each profile is named by its group key -- the value for one grouping column,
the values joined with ``"|"`` for several (``"KO_A|batch1"``). ``obs`` holds
the grouping columns, the cell counts ``n_cells`` and ``n_cells_aggregated``,
and every other ``obs`` column of the input that is constant within each
profile, so donor-level covariates (sex, age, disease) arrive with the
profiles; columns that vary within a profile (a per-cell QC metric) are left
out and listed at ``verbose=1``. A description of the aggregation is under
``uns['crispyx_pseudobulk']``, and the input and arguments it was computed
from under ``uns['crispyx']``. ``X`` is written as CSR when at most half of
its values are non-zero (summed units of a few cells each, over tens of
thousands of genes), and dense otherwise, so a genome-wide screen's
pseudo-bulk is a fraction of the dense size to write and to read in every
fit.

``perturbations`` restricts which profiles are returned. A profile is kept when
any of its grouping values matches, so the argument selects on whichever column
holds the labels irrespective of its position in ``groupby``, and every observed
combination of the remaining columns is preserved. With
``groupby=["perturbation", "batch"]``, passing ``perturbations=["KO_A"]`` returns
one profile per batch for ``KO_A``.

Check cell-type composition before pooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pooling cell types into one pseudo-bulk per donor turns any difference in
their proportions between conditions into apparent differential expression:
if cases have fewer cells of a highly expressing subtype, its genes look
down-regulated in every case donor, and a latent-factor adjustment cannot
separate the two. Aggregating by donor *and* cell type keeps the subtypes
apart, and its ``n_cells`` column is the composition table:

.. code-block:: python

   pb = cx.pb.aggregate(
       "brain.h5ad", groupby=["donor", "subclass"], method="sum", min_cells=1,
   )
   obs = pb.backed.obs
   cells = obs.pivot(index="donor", columns="subclass", values="n_cells").fillna(0)
   share = cells.div(cells.sum(axis=1), axis=0)          # per-donor proportions
   disease = obs.groupby("donor", observed=True)["disease"].first()
   share.groupby(disease).mean().T                       # mean share per condition

A subclass whose share differs clearly between conditions should be analysed
on its own (one pseudo-bulk per donor and subclass) rather than pooled.
``disease`` is available here because it is constant within each profile.

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

:func:`crispyx.pb.normalized_effects` goes from a cell-level file to an effect size
in a single streaming pass. Unlike :func:`crispyx.pb.effects`, which contrasts on
whatever scale its input already carries, this function **normalises library size
itself** -- so do not pre-normalise as well.

``method`` selects which scale the averaging happens on. These are different
estimators, not different spellings: averaging before or after the log gives
different answers whenever expression varies within a group.

.. code-block:: python

   # mean of per-cell log1p values
   effects = cx.pb.normalized_effects(
       adata_ro,
       groupby="perturbation",
       method="mean_log1p",
   )

   # log of the mean of normalised counts
   effects = cx.pb.normalized_effects(
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

   effects = cx.pb.normalized_effects(
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
* **Standard errors.** ``method="mean_log1p"`` also writes
  ``layers['standard_error']`` (cells independent within a batch) and
  ``layers['standard_error_robust']`` (from the spread of the per-batch
  effects), after Liu et al. (2026). :doc:`covariate_adjustment` compares them
  with the other approaches, and
  :doc:`notebooks/stratified_wilcoxon_tutorial` checks their calibration.
* **Bounded memory.** The correction keeps the streaming, single-pass design.
  The only quantities that grow with the number of batches are the
  per-``(perturbation, batch)`` sums, ``n_pairs × n_genes × 8`` bytes each
  (two with standard errors), where ``n_pairs`` is the number of observed
  ``(perturbation, batch)`` combinations. They stay in memory when they fit in
  half of ``memory_limit_gb`` (detected when omitted) and otherwise go to a
  disk-backed ``numpy.memmap`` in ``$TMPDIR``, with a warning if free space
  there looks tight; see :ref:`disk-space` to check usage up front.

Memory budget and chunk size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the differential-expression functions, the pseudo-bulk estimators
stream cells in chunks. The cell ``chunk_size`` defaults to ``None`` and is
then auto-selected from the dataset shape and the available memory. Pass
``memory_limit_gb`` to cap the budget in SLURM / cgroup-constrained
environments (the value is forwarded to the chunk-size heuristic exactly as in
``cx.tl.t_test`` / ``cx.tl.nb_glm_test``):

.. code-block:: python

   adata_pb = cx.pb.normalized_effects(
       adata_ro,
       groupby="perturbation",
       batch_column="batch",
       memory_limit_gb=128,   # cap the streaming chunk budget
   )

``memory_limit_gb`` and ``chunk_size`` only affect performance/peak memory —
the computed values are identical regardless of the chosen chunk size. Passing
an explicit ``chunk_size`` overrides the auto-selection (and ignores
``memory_limit_gb``).
