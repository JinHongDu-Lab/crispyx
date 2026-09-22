Changelog
=========

Version 0.1.5
-------------

This release reworks the GLM solver. Estimates change for lowly-expressed
genes -- toward, not away from, the reference implementations -- so results
from 0.1.4 on covariate-adjusted or low-count genes will not reproduce
exactly. ``nb_glm_test`` keeps its own ``min_mu`` default of 0.5, so the
DESeq2-compatible path is unchanged.

* **Negative-binomial estimates are no longer biased on lowly-expressed
  genes.** ``min_mu`` was being applied not only as the floor on the fitted
  mean that DESeq2 defines, but also as a clamp on the IRLS weights, on the
  variance, and on several division guards. Clamping the weights inflates the
  leverage of every cell whose fitted mean falls below the floor: at
  ``mu = 0.1, alpha = 1`` the correct weight is 0.091 and the clamp made it
  0.5. The clamped fit did not merely differ from the right answer, it solved
  a different estimating equation: on genes below one count per cell it left a
  score of 50-100 where the maximum likelihood estimate has a score of zero,
  and roughly 40-100 units of excess deviance. The same fits now leave a score
  of ~1e-6. (The score equation is used here rather than agreement with
  another implementation because it needs no second implementation, and
  cross-solver agreement is a poor instrument on sparse count data.)
  ``min_mu`` now floors the fitted mean and nothing else, and its default on
  ``NBGLMBatchFitter`` and ``NBGLMFitter`` is 0. ``nb_glm_test`` keeps its own
  default of 0.5, so the DESeq2-compatible path is unchanged.
* **(gene, perturbation) pairs with no counts in one arm are no longer given
  an effect estimate.** A log-link GLM estimates the effect as a difference of
  log means, so a pair absent from one arm has no finite effect: the
  likelihood has no interior maximum and the coefficient runs to the boundary.
  What was reported there came from wherever the fitter stopped, not from the
  data -- on one such gene the effect was -18.9 with ``min_mu=0`` and -5.1 with
  ``min_mu=0.5``, and the Wald statistic moved from 0.08 to 32.6 on identical
  counts, that is from "no evidence" to "wildly significant" purely as an
  artefact of the mean floor. (The collapse at ``min_mu=0`` is the
  Hauck-Donner effect: as the coefficient diverges its standard error grows
  faster still, so the Wald test loses power on the strongest possible
  signal.) These pairs are now reported as untested -- ``NaN`` effect,
  statistic and p-value -- exactly as genes with no counts anywhere already
  were, and are excluded from the multiple-testing correction rather than
  entering it with artefactual p-values. They are *not* reported as an effect
  of zero, which would describe a completely silenced gene as unchanged; the
  observation remains visible in ``pts`` and ``pts_rest``, and
  ``lfc_shrinkage_type="apeglm"`` still gives a bounded estimate where one is
  wanted. Controlled by the new ``nb_glm_test`` parameters
  ``min_cells_ctrl`` and ``min_cells_pert``, both defaulting to 1 -- symmetric,
  and the exact boundary between an effect that exists and one that does not.
  They are separate because the informative direction depends on the screen;
  ``crispyx._statistics._nonestimable_glm_mask`` documents which asymmetry
  suits CRISPRi and which suits CRISPRa. Set either to 0 to disable that side.
* **Two standard-error bugs from the same cause.** ``NBGLMFitter`` floored
  standard errors at ``sqrt(min_mu)``, forcing every reported standard error
  to at least 0.707 at the old default, and floored the Cook's-distance
  leverage denominator at ``min_mu``.
* **Wide designs are two orders of magnitude faster.** The per-gene Hessian
  was formed with a three-operand ``numpy.einsum``, which stops routing
  through BLAS once its intermediate exceeds an internal budget and falls back
  to a nested loop. Formed with a chunked ``gemm`` instead, ``fit_batch`` on
  1,000 cells and 500 genes went from 16.05 s to 0.15 s at design width 41 and
  from 35.38 s to 0.23 s at width 61; below width 21 there is no material
  difference. Results agree to 7e-12.
* **New:** ``StructuredGLMBatchFitter``, ``fit_glm_onehot`` and
  ``detect_onehot_block`` for designs of the form ``[covariates | one-hot
  groups]`` -- a perturbation screen, or any design with a many-level
  categorical covariate. The disjoint group supports make the per-gene Hessian
  arrowhead-structured, so each Newton step goes through the Schur complement
  of the diagonal block: exact (verified against a dense per-gene solve to
  1.4e-15) and much cheaper. Measured against the dense fitter at 1,200 cells
  and 800 genes: 14.2x at 31 covariates and 200 groups, 6.2x at 12 and 100,
  3.3x at 2 and 50, 2.5x at 12 and 20. It also converges where the dense path
  does not -- 744 of 800 genes against 31 of 800 on the widest design --
  because a group carrying no counts has an unbounded coefficient that the
  group ridge and clip make finite and defined.
* **New:** ``family="poisson"`` on ``NBGLMBatchFitter``, and
  ``fixed_dispersion`` on ``fit_batch``. A Poisson fit is now a genuine
  Poisson fit rather than a negative-binomial fit with a small estimated
  dispersion.
* **``nb_glm_test`` with a many-level categorical covariate takes the
  structured path.** A ``batch``, ``donor`` or ``lane`` covariate is one-hot
  encoded into the design, so this is the common case rather than an exotic
  one. Routing is on the measured advantage -- the group columns must
  outnumber the remaining covariates -- and the intercept and the
  perturbation column are held out of the group block so that the coefficient
  under test is never subject to the group clip. Log-fold-changes agree with
  the dense path to 1e-3.
* **The IRLS loop itself is sturdier.** Newton steps that increase a gene's
  deviance are shortened rather than accepted (see below); converged genes are
  frozen and dropped from later iterations; the dispersion is estimated around
  the IRLS rather than re-estimated inside every iteration, and the model is
  refitted with it so the returned coefficients and dispersion describe the
  same model; and the normal equations are solved in a unit-root-mean-square
  column basis. Convergence now requires both the relative deviance change and
  the largest coefficient change to fall below ``tol``.
* **Step shortening does what it says, and the convergence flag means what it
  says.** Each retry interpolates between the current iterate and the full
  Newton point, halving the distance, and a shortened step is taken only if it
  improves on the deviance the iteration started from. Interpolating towards
  the previously shortened point instead compounds the shortening -- the third
  retry lands at ``2^-6`` of the step rather than ``2^-3`` -- so a gene needing
  repeated damping stops moving and is then reported as converged *because*
  nothing moved, at a point that is not a stationary point of the deviance;
  ``de.py`` gates on that flag. A gene for which no step in the range improves
  the deviance is now left where it was and reported as not converged, rather
  than being moved uphill. The exception is a gene resting on the ``min_mu``
  floor: the floored cells' means no longer move with the coefficients, so
  they carry no gradient while the normal equations still count them, and the
  line search finding nothing to take describes the floor rather than the fit.
  Such a gene is still reported. A gene that reaches the ``eta`` divergence
  clips is not exempted, because there "not converged" is the useful answer.
* **The Numba path for the intercept-plus-perturbation design fits at the
  dispersion it reports.** The kernel holds the dispersion fixed, and it was
  being given a placeholder of 0.1 while the dispersion reported beside the
  result came from a separate estimate made afterwards. On that design the
  coefficients are the two group means whatever the dispersion is, but the
  standard errors are not: measured on an overdispersed fixture the reported
  standard errors were up to 3.9x too small, so every Wald statistic built on
  them was up to 3.9x too large. The kernel is now run twice, as the
  NumPy path already was -- once at the dispersion the warm-start means imply,
  then at the dispersion the fitted means imply -- and its standard errors now
  match a fit at the reported dispersion to 5e-15.
* **The negative-binomial deviance no longer loses its value to
  cancellation.** It was formed as the difference of ``(y + r) log(y + r)``
  and ``(y + r) log(mu + r)``, each of order ``r n log r``. At the ``alpha``
  clip floor ``r`` is 1e8, so for a near-Poisson gene the answer was the small
  difference of two very large sums: the absolute error was 0.1 at 3,000 cells
  and 182 at 100,000, against a deviance of ~1e5. That is far above the
  ``1e-6`` relative tolerance the convergence test uses, so such a gene could
  never converge and its damping was triggered by noise. Evaluated as
  ``(y + r) log1p((mu - y) / (y + r))`` the same quantities are accurate to
  7e-7 at 100,000 cells.
* **``ridge_penalty`` means the same thing whatever the design's column
  scales.** The normal equations are solved in a preconditioned basis, where a
  ridge of ``r`` penalises the caller's coefficient ``j`` by
  ``r * scale_j**2``. The penalty is now divided by the column scale, so a
  caller who sets ``ridge_penalty`` deliberately gets the penalty they asked
  for. No effect at the 1e-6 default.
* **``StructuredGLMBatchFitter.fit_batch`` fits genes in batches**, with a
  ``gene_batch_size`` argument that matches ``NBGLMBatchFitter``'s and
  defaults to sizing the ``(n_samples, batch)`` work arrays for ~100 MB. A
  Newton step holds a dozen of them, which at 3,000 cells and 8,563 genes was
  around 2 GB. Batching changes nothing numerically beyond the summation order
  BLAS chooses. The starting point also no longer depends on the design having
  an exact column of ones: the constant column carries the starting predictor
  and ``eta`` is derived from the coefficients, so a design with no such
  column starts from a point its coefficients describe.
* **Less discarded work in the structured solver.** The negative-binomial path
  ran a full Poisson fit and an intermediate NB fit only for their fitted
  means, and computed -- then threw away -- the per-gene standard errors of
  both; neither now forms them. ``schur_solve`` re-formed the two
  ``(n_genes, n_features, n_groups)`` arrays that ``schur_complement`` had
  just built, and the IRLS loops copied the counts of the active gene set
  twice per iteration, including on the first iteration where the active set
  is every gene.
* **Where the ``min_mu`` floor binds, IRLS is run as the fixed-point iteration
  it is.** A cell held at the fitted-mean floor has ``d mu / d beta = 0``: it
  contributes nothing to the gradient of the deviance while still contributing
  its full weight to ``X'WX``, so the Newton step solves one model while the
  deviance measures another and can point uphill at a good fit. The line
  search above is the wrong instrument for those genes, and no step length
  repairs the direction. They now take the full Newton step and are judged by
  DESeq2's own criterion, ``|dev - dev_old| / (|dev| + 0.1) < tol``, paired
  with the usual coefficient-change test -- which is what DESeq2, and
  crispyx's own Numba kernel, have always done. A gene with floored cells that
  is descending normally keeps the strict test; the floor only decides what a
  *failed* step means. Measured against PyDESeq2, coefficients on
  floor-binding genes go from 2.8e-02 away to under 1e-06, every gene
  converges where one to four used to fall short, the 200-iteration cap stops
  being reached, and ``min_mu=0.5`` is no longer slower than ``min_mu=0``.
  On a covariate-adjusted run of a real screen (Adamson, 9,496 genes) this
  returns 103 genes per perturbation that were being reported as ``NaN``, at
  3.4 s against 5.7 s. They are not marginal genes -- their median ``pts``
  in the control arm is 0.69 against 0.30 for the genes already reported, and
  44 of them reach ``padj < 0.05``, taking that comparison's hit list from
  2,094 to 2,150. A gene that did not converge was previously given no result
  at all -- ``NaN`` effect, statistic, p-value, log-fold-change and standard
  error -- and was left out of the multiple-testing correction, so this is a
  change to what is tested, not only to what is estimated. Genes reported
  before are unchanged to within 2e-05 in log-fold-change, and their adjusted
  p-values move by a median of -0.3% from the larger correction.
* **``min_mu`` no longer reaches the reported standard errors.** The floor
  steadies the iteration; it is not part of the model whose uncertainty is
  reported, and DESeq2 keeps it out -- ``irls_solver`` returns an
  unthresholded ``mu`` and ``wald_test`` rebuilds the weights from it.
  crispyx built them from the floored mean, which overstates how much a
  floored cell knows: against PyDESeq2's ``wald_test`` on identical
  coefficients the standard errors on floor-binding genes were a median of
  24% and at worst 35% too small, and the Wald statistics correspondingly too
  large. Rebuilt from the unfloored mean they agree to 0.00%. The weights
  inside the iteration and the leverage behind Cook's distance keep the floor,
  as DESeq2 does. This affects ``nb_glm_test`` with covariates, the structured
  solver and the fitter API; a two-group comparison without covariates already
  recomputed its standard errors without the floor and is unchanged.

  End to end through ``nb_glm_test`` the effect is much smaller than those
  figures suggest, because the dispersion is re-estimated after the fit and
  absorbs most of the change: on the Adamson run above, p-values for genes
  reported both before and after moved by a median factor of 1.0000 (10th to
  90th percentile 0.97 to 1.02), 7 genes lost significance and 19 gained it.
  The larger figures are what you see with the dispersion held fixed, which is
  the right way to size the defect but not the change a user sees.

* **``pts`` and ``pts_rest`` are reported for every gene**, not only for the
  genes an effect was estimated for. They are descriptions of the data rather
  than inferences from a fit, and they are what makes a pair whose effect is
  not estimable -- a complete knockdown, expressed in 95% of control cells and
  none of the perturbed ones -- visible in the output, which the filtering
  documentation already said they were.

Version 0.1.4
-------------

*Released 2026-09-14.*

* **``batch_process`` no longer reads the whole weight layer into memory to
  finish a run.** The last step of a run reduced an ``(n_groups, n_genes)``
  layer -- 5 GB on a screen with ~18k perturbations and ~35k genes -- to one
  boolean per group, by materialising it in a single allocation, at the point
  in a long run where memory is least available. The reduction now streams
  the layer in gene-chunk slices (following the HDF5 chunking the
  output is already created with), so the peak is one chunk rather than the
  whole layer. The reported groups are unchanged.
* **``batch_process`` resumes on the gene-chunk width its output was written
  with**, when ``chunk_size`` was auto-selected. That width is derived from
  the memory budget, so resuming the same run under a different ``--mem``
  used to pick different chunk boundaries, fail the metadata match, and
  overwrite the partial output the call was asked to continue. An
  *explicitly* passed ``chunk_size`` that differs from the stored one still
  restarts with a warning, as before -- only the auto-selected case follows
  the file.
* **The per-``(group, batch)`` combine loop is roughly 2.5x faster.** A
  scalar ``BatchStatistic.weight`` -- what every reducer in the docs, the
  tests and practice returns -- was broadcast to one value per gene, then
  validated and boolean-masked per pair per channel: hundreds of thousands of
  redundant array allocations per gene chunk. Scalar weights now stay scalar
  through validation and accumulation, which is bit-identical to the masked
  path for a positive weight. Two further per-run scans were removed: the
  cell-to-group mapping is built from the distinct labels rather than with a
  dict lookup per cell, and the "perturbation contains no cells" check uses a
  set instead of scanning a list once per group (quadratic in the group
  count). On a synthetic 18,000-group profile the ``batch_process`` call goes
  15.5 s -> 7.0 s (600 genes) and 12.3 s -> 4.6 s (1800 genes); results are
  unchanged.
* A resumed ``batch_process`` progress bar starts at the chunk it resumes
  from, instead of printing a ``0/total`` line and then jumping. In a log
  file the old form read as a run that had restarted from nothing and then
  skipped ahead.
* **Memory budgets respect a cgroup ceiling.** Every auto-sizing path --
  gene and cell chunk sizes, the CSR<->CSC conversion buffers, the DE and QC
  budgets -- sized itself from ``psutil.virtual_memory().available``, which
  reports the *host's* memory. Under Slurm, Docker or Kubernetes a job
  allocated 200 GB on a shared node sees a machine with far more than that
  free, so an auto-sized run could budget past its allocation and be
  OOM-killed with every crispyx budget still apparently satisfied. All of
  these now read through one accessor that resolves the process's own cgroup
  from ``/proc/self/cgroup`` -- a Slurm job's ceiling sits several levels
  below the hierarchy root, which publishes none at all, so reading the root
  would have found a container's limit and never a job's -- and caps the
  reading by the tightest cgroup v2 (``memory.max``) or v1
  (``memory.limit_in_bytes``) ceiling on that path, less the memory the
  cgroup already holds (reclaimable page cache excluded, so streaming a large
  h5ad does not shrink the next chunk). An explicit ``memory_limit_gb``
  larger than what remains is capped the same way, so passing
  ``memory_limit_gb=400`` to a 200 GB job no longer sizes buffers for
  400 GB. Nothing changes off a cgroup.
* **A resumable run is warned when it is about to convert.** The temporary
  fast-axis copy lives for one call, so a computation that needs several
  restarts -- which is what ``resume=True`` is for -- rebuilds the whole copy
  on every one of them, before any new work starts. A job under a walltime
  shorter than its computation is guaranteed to hit this. ``batch_process``
  now emits one ``UserWarning`` when a resumable call converts, pointing at
  ``cx.pp.convert_to_csc``; ``docs/faq.rst`` gains a section on converting
  once for resumable or multi-step work. ``wilcoxon_test`` does not warn,
  because it refuses to resume from a checkpoint at all. The copy's per-call
  lifetime is deliberately unchanged.
* **Resume checkpoints shrink by ~75x.** ``batch_process`` stored its
  ``batches_used`` grid as a JSON coordinate list: at 17,978 groups x 4
  batches that is 913 KB of pretty-printed JSON, rewritten after every gene
  chunk (the interval is 1 below 100 chunks). Packed as a bitmap it is
  12 KB. **A checkpoint written by an earlier version keeps its gene-chunk
  progress but loses its batch record.** A run resumed across the upgrade
  still restarts on the first unfinished gene chunk, so nothing completed is
  recomputed -- but ``obs['n_batches_used']`` then counts only the batches
  seen after the resume, and a ``UserWarning`` says so. It cannot be
  reconstructed afterwards, because the weight layer is summed across
  batches. Finish an in-flight resumable run on the version
  that started it, or pass ``force=True`` for an exact recount.

Version 0.1.3
-------------

*Released 2026-09-10.*

* **Gene-streaming functions no longer re-read a CSR source once per gene
  chunk.** ``wilcoxon_test`` (all three paths: standard, group-batch
  streaming, batch-stratified) and ``batch_process`` stream ``X`` by gene
  (column) chunks. On a CSR-stored file -- what every crispyx writer
  produces -- anndata serves a column slice by reading the *whole*
  ``data``/``indices`` arrays into memory and filtering, so each of the
  ``n_gene_chunks`` chunks paid a full read of the file (tens of minutes
  per chunk on a network filesystem for a multi-million-cell screen, and
  the matrix's full size in transient RAM). Both functions now take
  ``format_mismatch_policy`` -- new on ``wilcoxon_test`` -- and all three
  functions that take it (with ``normalize_total_log1p``) default to a new
  value, ``"auto"``: the source is converted once to a temporary fast-axis
  copy *beside the output file* (not ``$TMPDIR``), and removed before
  returning, **when that is measurably cheaper than streaming off the fast
  axis**. Converting costs one full read plus one full write while streaming
  costs one full read per chunk, so which wins is a property of the
  filesystem rather than of the data: ``"auto"`` times a bounded 64 MB prefix
  read of the source, projects ``(n_gene_chunks - 1) ×`` the implied
  full-read time, and converts only when that exceeds 60 s and there are at
  least 4 chunks (fewer cannot repay the extra write however slow the
  filesystem is). On a 500 MB file on local disk, where a full re-read costs
  ~0.1 s, this streams as-is and saves the ~5 s an unconditional conversion
  spent; on the multi-million-cell network-filesystem case it still converts.
  ``"convert"`` keeps its meaning of *always* convert, and the chosen branch
  and the numbers behind it are printed at ``verbose>=1``. ``"warn"`` now
  emits a ``UserWarning`` that quantifies the cost (matrix size × chunk
  count) instead of only a logger line; ``"off"`` is unchanged. The three
  hand-rolled copies of this logic
  (``normalize_total_log1p``, ``batch_process``, and none for
  ``wilcoxon_test``) are replaced by one ``crispyx.data.stream_on_fast_axis``
  helper. ``estimate_disk_usage`` reports the temporary copy under a new
  ``"scratch"`` location and assesses it (and ``"output"``) where the file
  will actually be written -- it resolves ``output_path`` / ``output_dir`` /
  ``data_name`` exactly as the target function does, and under ``"auto"`` it
  runs the same convert-or-stream decision (including its 64 MB probe read,
  the one case where the query touches ``X`` at all), so the ``"scratch"``
  entry appears when the real call would make a copy. Because that decision
  is a measurement, the measurement is cached per file for the life of the
  process: the query and the run it describes see one number rather than two,
  and the probe is paid once. Pass ``chunk_size`` / ``memory_limit_gb`` to
  the query if you will pass them to the call -- they set the chunk count,
  which is what the decision turns on. When the free space
  beside the output cannot hold the temporary copy, ``"convert"`` falls back
  to ``"warn"`` behaviour (one warning naming the shortfall, then streaming
  from the source) instead of failing with ``ENOSPC`` partway through the
  conversion. ``format_mismatch_policy`` is validated up front by every
  entry point (``wilcoxon_test`` previously checked it only after the
  cached-result return; the disk-usage resolvers silently ignored a typo),
  and the ``"off"`` policy no longer mutes the process-wide slow-axis
  logger warning for unrelated later calls -- callers that resolved the
  format decision pass ``iter_matrix_chunks(..., warn_slow_axis=False)``
  instead. ``normalize_total_log1p(format_mismatch_policy="convert")`` on a
  CSC source now copies ``uns`` / ``layers`` / ``obsm`` / ``varm`` / ``obsp``
  / ``varp`` from the source file rather than from the X-only temporary
  copy, where they were silently dropped. A run killed outright cannot delete
  its temporary copy -- ``SIGKILL`` skips the ``finally`` that would, and
  ``atexit`` would not fire either -- so every call that sees a mismatched
  source first sweeps the abandoned copies in its scratch directory (not only
  the calls that convert: under ``"auto"`` a directory may be swept by runs
  that never convert again). Otherwise an out-of-memory-killed job left a
  hidden file the size of its matrix next to the output, forever. A copy is
  abandoned only once it has gone a day untouched -- a conversion in progress
  rewrites its copy continuously -- and, for copies this machine wrote, once
  its owning process is gone. Both the PID and a tag for the host are part of
  the ``.cx_<function>_<pid>-<host>_*`` name: the scratch directory is the
  output directory, which a cluster job array shares across nodes, and a PID
  read on the wrong node says nothing about whether that copy is live.
* **``batch_process`` no longer returns a killed run's output as a cached
  result.** The output file is created -- with complete ``uns`` metadata and
  a NaN fill -- before the first gene chunk is processed, and a run that died
  before its first checkpoint left exactly that file behind. The next
  identical call matched its metadata and returned it: an all-NaN result, in
  milliseconds, with no indication anything was wrong (the behaviour is
  present in 0.1.2 as well). A completion marker is now written after the
  last gene chunk and required by the cache check, so an unfinished output is
  recomputed (or resumed, with ``resume=True``) instead. Outputs written by
  earlier versions carry no marker and are recomputed once. A recompute fills
  the output file in place -- that is what lets a killed run resume -- so it
  replaces the existing file before it has a result to put there; if that
  rerun is killed too, neither result survives. It now says so in a warning
  naming the file, which also covers the more familiar case of rerunning with
  changed parameters. ``force=True`` is the user asking for the rerun and
  stays silent.
* **Sizes are reported in a unit that suits them.** Every user-facing disk
  and slow-axis message went through a fixed ``GB`` format, so the
  quantified slow-axis warning read ``0.0 GB of data+indices, ~0 GB in
  total`` for anything under a gigabyte -- exactly the messages meant to
  explain a cost. One ``crispyx._disk.format_bytes`` now scales the unit
  (``49.3 MB``, ``40.0 GB``) across ``DiskEstimate``, the disk-space
  warnings, the ``verbose`` disk line, and the slow-axis messages.
* **``convert_to_csc`` / ``convert_to_csr`` are memory-bounded and
  dtype-preserving.** Previously the whole converted matrix
  (``total_nnz × 8`` bytes) was buffered in RAM before a single write, which
  made the automatic conversion above unsafe for files near the node's
  memory. Both converters gain ``memory_limit_gb``: the output buffers use
  at most half of it, and a matrix that does not fit is converted in
  contiguous column (CSC) or row (CSR) *bands*, each costing one extra
  streaming pass over the source -- ``K`` bands for a matrix ``K`` times the
  budget instead of an OOM. A dense-to-CSR conversion now writes chunk by
  chunk with no whole-matrix buffer at all. The converters also stop
  silently casting values to ``float32``: the output keeps the source's
  value dtype, so a format change never changes results (the old cast made
  a converted float64 matrix disagree with its source at the 1e-7 level).
  Because the output is now pre-sized and filled band by band, it is
  written to a ``.<name>.partial`` file beside ``output_path`` and renamed
  into place only on completion; an interrupted conversion (Ctrl-C, OOM,
  ``ENOSPC``) leaves no output file instead of a structurally valid one
  with zero-filled bands. ``cx.pp.convert_to_csc`` / ``cx.pp.convert_to_csr``
  accept and forward ``memory_limit_gb`` like the top-level functions.
* **``batch_process`` inner loop is ``O(n_pairs)`` per gene chunk instead of
  ``O(n_cells)``.** Cells are sorted by ``(group, batch)`` once; each gene
  chunk is then row-permuted once and the reducer's ``update`` is called
  once per contiguous ``(group, batch)`` segment (per densified slab),
  replacing a mask scan plus a scipy fancy-index per pair per 4096-cell
  chunk -- which, with tens of thousands of groups, amounted to one call per
  cell. Combined statistics are written as one ``(n_groups, width)`` block
  per layer per gene chunk into output datasets whose HDF5 chunks are
  aligned to the gene chunks, instead of ``n_groups × n_layers`` strided
  7 KB row writes per chunk. Results are unchanged (the ``BatchReducer``
  contract is untouched; existing reducers need no change); a synthetic
  4000-group × 6-batch × 120k-cell run went from 25 s to 5.5 s of pure
  compute. One consequence is worth knowing about: a reducer now receives
  the cells of a ``(group, batch)`` pair in fewer, larger blocks, and a
  reducer that accumulates in the block's own dtype is *less* accurate on a
  ``float32`` file for it (summing thousands of float32 rows at once rather
  than hundreds -- ~3e-6 instead of ~4e-7 against a float64 reference on a
  17k-cell file). Block sizes were never part of the contract; the
  ``BatchReducer`` docstring now says so explicitly and shows the
  ``np.asarray(block, dtype=np.float64)`` that makes a reducer independent of
  them (and accurate to 1e-14). Peak memory stays bounded: rows are gathered
  per densified slab
  (never a second full copy of the gene-chunk block), the combined values
  are divided in place, and the automatic ``chunk_size`` is additionally
  capped so the ``(n_groups, chunk_size)`` accumulators fit the per-chunk
  budget. The weight layer the resume fallback scan keys off is written
  last for each gene chunk, so a chunk the scan reports complete has all
  of its datasets written.
* **``wilcoxon_test`` drops ``n_jobs``.** It was accepted but never read on
  any Wilcoxon path (parallelism comes from the numba ``prange`` rank
  kernels, which already use every CPU the process is allowed to run on).
  Also removed from ``rank_genes_groups(method="wilcoxon")``'s accepted
  keywords. The one-time per-group row lookup in all three Wilcoxon paths
  is now a single factorize/argsort instead of ``n_groups`` full scans of
  the label array (minutes at 18k groups × 2M cells).
* **``normalize_total_log1p(format_mismatch_policy="convert")`` names its
  output correctly.** The default output name was derived from the
  temporary CSR copy's random filename instead of the source's.
* ``cx.tl.batch_process`` now forwards ``resume``, ``checkpoint_interval``
  and ``format_mismatch_policy`` (previously only reachable via
  ``cx.batch_process``).

Version 0.1.2
-------------

*Released 2026-08-25.*

* **New: ``cx.pp.highly_variable_genes``** – streaming, disk-backed highly
  variable gene (HVG) selection, the producer half of the
  ``var["highly_variable"]`` contract ``cx.pp.pca`` already consumed. Two
  flavors, both dispatching on storage format the same way the QC functions
  do (row-chunked for CSR/dense, column-chunked for CSC), in ``O(n_genes)``
  memory:

  * ``"seurat_v3"`` (default; Stuart et al. 2019) -- ranks genes by
    standardized variance fit via a degree-2 LOESS smoother (the new
    ``scikit-misc`` runtime dependency). Expects raw counts; requires
    ``n_top_genes``. Two data passes are inherent to the method (the clip
    threshold used in pass 2 depends on every gene's pass-1 moments).
    Selection uses an exact rank (matching scanpy's own tie-breaking), so
    ``n_top_genes`` is always honored exactly even when several genes tie
    at the cutoff -- a real occurrence on production data, where multiple
    low-count genes can share an identical normalized variance.
  * ``"mean_dispersion"`` (Satija et al. 2015) -- bins genes by mean
    expression and z-normalizes dispersion within each bin. Expects
    log1p-normalized data; needs no extra dependency. Single data pass.

  Defaults to computing gene statistics from **control cells only**
  (``cell_mask="control"``, resolved from ``perturbation_column``/
  ``control_label``) rather than all cells -- a CRISPR/Perturb-seq-specific
  choice, since over all cells, on-target perturbation effects can dominate
  the variable-gene list and structure downstream PCA around *which
  perturbation a cell received* rather than baseline cell-state
  heterogeneity. Pass ``cell_mask=None`` for the scanpy/Seurat all-cells
  default, or an explicit boolean array for a custom subset; the mask is
  resolved from ``obs`` alone and threaded into the streaming pass at no
  extra cost. Writes ``var["highly_variable"]``, ``var["means"]``,
  ``var["variances"]``, and ``var["variances_norm"]``. Verified against
  scanpy on real datasets (including outlier/edge-case genes) with an exact
  match of both the selected gene set and the normalized-variance values.

Version 0.1.1
-------------

*Released 2026-08-17.*

* **``cx.tl.batch_process`` now streams gene-major in a single pass** via
  the same ``iter_matrix_chunks(axis=1, ...)`` access pattern
  ``wilcoxon_test`` already uses, instead of re-reading the full cell axis
  once per gene chunk. This is a strict speed improvement with no memory
  regression, and native/cheap for a CSC-stored source. A new
  ``format_mismatch_policy`` parameter (``"warn"`` / ``"convert"`` /
  ``"off"``, matching ``normalize_total_log1p``) controls what happens when
  the source is CSR instead.
* **``cx.tl.batch_process`` gains ``resume``/``checkpoint_interval``**,
  extending the same atomic-checkpoint, corruption-safe-read infrastructure
  ``t_test``/``wilcoxon_test``/``nb_glm_test`` already use to the generic
  streaming-statistics API. The unit of resumable progress is a gene chunk;
  results are written directly into the (pre-sized) output file as each
  chunk finishes, and a missing/corrupted checkpoint falls back to scanning
  that output file for the last completed chunk.
* **``BatchReducer`` supports multiple named channels from one pass.**
  Setting ``channels=(...)`` lets ``finalize``/``compare`` return a dict of
  related statistics computed from the same streaming state -- for example
  a mean difference and its standard error, so a caller can form
  ``t = mean_diff / se`` without a second pass over the data. Each channel
  is combined across batches independently and written to its own
  ``layers[name]``; the first channel is also copied into ``X``. Existing
  reducers returning a single ``BatchStatistic``/array are unaffected.

Version 0.1.0
-------------

*Released 2026-08-13.*

* **New: ``cx.pp.subsample``** – streaming, stratified/cluster subsampling.
  The mask is computed entirely from ``.obs`` metadata (no matrix pass
  needed to decide which cells survive), then streamed out via the same
  writer every other filtering function uses. ``groupby`` stratifies (one
  column, several columns, or ``None`` for a single global stratum);
  ``unit="cell"`` (default) draws individual cells, while passing an
  ``obs`` column name instead (e.g. ``unit="batch"``) switches to cluster
  sampling, where a chosen unit's cells are kept in full and an unchosen
  unit's are dropped in full. ``n`` (exact count) or ``frac`` (proportion)
  is drawn independently per stratum, matching
  ``pandas.DataFrameGroupBy.sample(n=, frac=)`` semantics.
  ``drop_insufficient`` controls what happens to a stratum smaller than the
  requested count (drop it entirely by default, or keep it in full), and
  every affected stratum is reported via a warning regardless of
  ``verbose``. Sampling is deterministic for a fixed ``random_state`` and
  independent of ``chunk_size``.
* **New: ``cx.pp.downsample_counts``** – streaming, dependency-free
  equivalent of ``scanpy.pp.downsample_counts(..., replace=False)``: thins
  every cell's total count down to a target via exact sampling without
  replacement (a cell already at or below the target is left unchanged).
  Complements ``subsample`` on the orthogonal axis — ``subsample`` decides
  *which cells* survive, ``downsample_counts`` decides *how many counts*
  survive within a surviving cell. A single streaming pass with a
  resizable HDF5 output avoids a separate counting pass over the source.
* **Fix: filtered/subsampled/normalized outputs now keep every AnnData slot.**
  ``write_filtered_subset`` — the shared streaming writer behind
  ``cx.pp.filter_cells``, ``cx.pp.filter_genes``, ``cx.pp.filter_perturbations``,
  ``cx.pp.qc_summary``, and the new ``cx.pp.subsample`` — previously wrote
  only ``X``, ``obs``, and ``var``, silently dropping ``layers``, ``obsm``,
  ``varm``, ``obsp``, ``varp``, and ``uns`` from every filtered output. It now
  streams ``layers`` the same way as ``X`` and carries
  ``obsm``/``varm``/``obsp``/``varp``/``uns`` through (subset on whichever axis
  applies); a source ``.raw`` is not copied, and a warning says so instead of
  the data silently disappearing. ``cx.pp.downsample_counts`` and
  ``cx.pp.normalize_total_log1p`` carry the same slots through unchanged, with
  the same ``.raw`` warning — including for an all-empty ``X``, which
  previously skipped the slot copy-through entirely.
* **Fix: ``cx.pp.downsample_counts`` per-cell thinning seed collisions.**
  The per-row RNG seed was truncated to 32 bits, which collides often enough
  at the "hundreds of thousands to millions of cells" scale this function
  targets that distinct cells could draw bit-identical thinning outcomes.
  Seeds now use the full 64-bit range, and the thinning kernel itself now
  draws via ``numpy.random.Generator.multivariate_hypergeometric`` in one
  call per row instead of a hand-rolled cumsum/choice/searchsorted/bincount
  sequence.
* **Fix: ``cx.pp.downsample_counts`` on dense-stored input.** A dense-stored
  ``X`` was previously always cast to ``float32`` regardless of its actual
  on-disk dtype (e.g. ``int32``); it's now read and preserved like the sparse
  path already did. ``X`` must hold non-negative integer counts — non-count
  (e.g. already-normalized) input now raises instead of being silently
  truncated and mostly no-op'd.
* **``write_filtered_subset`` is now exported at the top level**
  (``crispyx.write_filtered_subset``), reflecting that it is already relied
  on directly by real pipelines, not just an internal implementation
  detail of the filtering functions above.
* **Removed: ``compute_average_log_expression`` and
  ``compute_pseudobulk_expression``** (and the ``cx.pb.average_log_expression`` /
  ``cx.pb.pseudobulk`` namespace methods), deprecated in 0.0.9 with an explicit
  promise to remove them in 0.1.0. Use ``compute_normalized_effects`` /
  ``cx.pb.normalized_effects`` with ``method="mean_log1p"`` or
  ``method="log_mean"`` respectively instead.

Version 0.0.10
--------------

*Released 2026-08-08.*

* **Disk-space awareness** – crispyx now estimates the disk space a
  streaming call is about to need and warns -- without blocking the call --
  when free space on the relevant filesystem looks tight or the write is
  unusually large. This covers the disk-backed intermediate accumulators
  behind ``cx.pb.normalized_effects`` (batch-corrected path),
  ``cx.pb.aggregate``, ``cx.pb.effects``, ``cx.tl.t_test``,
  ``cx.tl.wilcoxon_test``, ``cx.tl.nb_glm_test``, ``cx.tl.batch_process``, and
  quality-control filtering, plus the ~2x transient disk requirement of
  whole-file CSR/CSC conversion (:func:`crispyx.convert_to_csc`,
  :func:`crispyx.convert_to_csr`, and
  ``normalize_total_log1p(..., format_mismatch_policy="convert")``). The
  check is automatic and has no configurable budget analogous to
  ``memory_limit_gb``: it always reads real free space via
  ``shutil.disk_usage`` and exists purely as a feasibility heads-up, not a
  resource allocator.
* **New: ``crispyx.estimate_disk_usage``** – an on-demand, standalone query
  to check disk usage *before* committing to a run:
  ``cx.estimate_disk_usage(func, data, **kwargs)`` accepts a function name
  (e.g. ``"compute_normalized_effects"``, ``"t_test"``,
  ``"convert_to_csc"``) or the function object itself, plus the same
  arguments the real call would take, and returns the estimated bytes
  required versus free space at each filesystem location involved (e.g.
  ``$TMPDIR`` for intermediate accumulators, the output directory for the
  final result). It reads only cheap ``obs``/``uns`` metadata in backed
  mode and never touches the expression matrix. Also available as
  ``cx.tl.estimate_disk_usage`` for Scanpy-style namespace discovery (the
  same pattern already used for ``compute_overlap``). See :ref:`disk-space`
  in the usage guide.
* **Cross-platform robustness** – disk-space checks now degrade gracefully
  instead of raising when free space cannot be determined at all (an
  unreachable network mount, a permission error on a Windows junction, a
  drive ejected mid-check): the affected ``DiskEstimate`` reports
  ``free_bytes=None`` and ``sufficient=True`` (fail open) rather than
  crashing the caller's real computation. The "large write" heads-up still
  fires in this case since it doesn't depend on free space.
* Documentation now notes that the memory/speed figures throughout the
  README, docs, and tutorial assume adequate free scratch disk for
  streaming intermediates and output files.
* **``verbose`` now defaults to ``True``** across the package (was ``False``
  on most differential-expression and pseudo-bulk functions). A first-time
  call already reports what it did -- what file is being read, what was
  inferred, what was written -- without passing ``verbose=`` explicitly.
  Pass ``verbose=False`` (or ``0``) for the previous silent behaviour. This
  is a behavioural default change, not a signature change: no parameter was
  removed or renamed.
* **Filtering feedback** – :func:`crispyx.pp.filter_cells`,
  :func:`crispyx.pp.filter_genes`, :func:`crispyx.pp.filter_perturbations`,
  and :func:`crispyx.pp.qc_summary` now report kept/total counts and warn
  when a filter removes more than half the data (cells, genes, or
  perturbations), a common sign of a misconfigured threshold.
* **Progress bars** extended beyond differential expression to CSC/CSR
  conversion, ``cx.pb.aggregate``, ``cx.tl.batch_process``, and the QC
  streaming passes. They use ``tqdm`` when available and degrade to a
  no-op otherwise, gated on the same ``verbose`` as everything else.
* **Chunk-size and streaming-strategy reporting** – functions that
  auto-select a chunk size, or choose between a single-pass and a
  streaming strategy internally, now say so at the default verbosity
  (e.g. ``chunk_size=4096 (auto)``, ``Strategy — column-streaming``).
* **Disk-usage confirmation** – every ``warn_if_disk_space_low`` call site
  now also prints a ``verbose``-gated confirmation of the estimate
  computed (required GB vs. free GB), shown whether or not the
  unconditional warning fired.
* Fixed a naming regression in :func:`crispyx.pp.qc_summary`'s verbose
  output (it printed ``qc.quality_control:`` instead of
  ``pp.qc_summary:``, left over from before the function was renamed).
* Warnings for missing batch/grouping values and untestable groups in
  ``cx.tl.batch_process``, ``cx.pb.aggregate``, ``cx.pb.effects``, and
  ``cx.tl.wilcoxon_test`` (batch-stratified) are now prefixed with their
  originating function, matching the convention already used by the
  disk-space warnings.
* See the new :ref:`Messaging and verbosity <messaging-and-verbosity>`
  section in the usage guide for the full picture of what prints, what
  warns, and what stays at logger level.

Version 0.0.9
-------------

*Released 2026-07-30.*

* **License change** – crispyx 0.0.9 and later is distributed under a Modified
  MIT License, which adds two attribution conditions for commercial use. All MIT
  freedoms are retained and no fee or royalty is imposed. Versions up to and
  including 0.0.8 remain under the unmodified MIT License; that grant is
  perpetual and is not withdrawn. See ``LICENSE`` for the terms.
* **Unified normalized effects** – ``compute_normalized_effects`` /
  ``cx.pb.normalized_effects`` replaces the two earlier one-command estimators with a
  single function selected by ``method``. ``method="mean_log1p"`` averages per-cell
  ``log1p`` values (mean of logs); ``method="log_mean"`` averages normalised counts and
  then applies ``log1p(baseline_count * mean)`` (log of mean). Both normalise library
  size themselves, and both return the effect in ``X`` with
  ``layers['perturbation_profile']``, plus
  ``layers['control_profile_matched']`` when ``batch_column`` is given, so that
  ``X == perturbation_profile - control_profile_matched`` exactly. Supplying
  ``batch_column`` is itself the request for batch correction; there is no flag.

  ``compute_average_log_expression`` and ``compute_pseudobulk_expression`` remain as
  deprecated aliases with their original layer and ``uns`` names, and now emit a
  ``DeprecationWarning``. They will be removed in 0.1.0.

  Note that ``cx.pb.effects`` deliberately does **not** normalise: it computes a contrast
  on whatever scale its input already carries. Normalise beforehand with
  ``cx.pp.normalize_total_log1p``, or use ``cx.pb.normalized_effects`` to have it done in
  one pass.
* **Generic streaming batch statistics** – ``batch_process`` /
  ``cx.tl.batch_process`` applies a user-supplied mergeable reducer within
  experimental batches without loading the complete cell-by-gene matrix. A
  ``BatchReducer`` provides ``initialize`` / ``update`` / ``finalize`` callbacks
  for per-group statistics, plus an optional ``compare`` callback for
  group-versus-reference contrasts in ``mode="comparison"``. Finalized batch
  statistics are combined as ``sum(weight * values) / sum(weight)``, and only
  batches containing both the group and the reference contribute to a contrast.
  Argument names follow the differential-expression API (``groupby`` aliases
  ``perturbation_column``; ``reference`` aliases ``control_label``). Cached
  results are keyed on the input path and modification time, so regenerating a
  source file invalidates its cached statistic; ``force=True`` remains necessary
  when a reducer's implementation changes without changing ``statistic_name``.
* **Batch-level absolute pseudo-bulk profiles** – ``aggregate_pseudobulk`` /
  ``cx.pb.aggregate`` groups by one or more observation columns and retains one
  profile for every observed combination. It supports strict raw-count sums,
  mean log1p expression, a five-cell default threshold, deterministic
  one-resample bootstrapping, source-layer selection, and versioned provenance
  metadata. ``perturbations`` keeps a profile when any of its grouping values
  matches, so it selects on whichever column holds the labels regardless of its
  position in ``groupby`` and preserves every combination of the others.
* **Explicit pseudo-bulk effects** – ``compute_pseudobulk_effects`` /
  ``cx.pb.effects`` consumes a saved crispyx pseudo-bulk artifact directly or
  aggregates cell-level input first. It returns within-batch target-minus-
  reference effects by default and can explicitly combine batches using the
  existing harmonic-count weighting.
* Tuple-level differential-expression results were intentionally not added;
  ``wilcoxon_test(batch_column=...)`` remains the batch-stratified test over all
  cells and batches.

Version 0.0.8
-------------

*Released 2026-07-14.*

* **Fix ``write_obs`` / ``write_var`` row-count check under the
  ``nullable-string-array`` encoding** – the shape guard read ``len()`` of the
  index element, which for the group encoding used by anndata >= 0.13 /
  pandas >= 3.0 counts the group's ``values`` / ``mask`` members (always 2)
  rather than the number of rows.  This made valid writes raise
  ``ValueError: DataFrame has N rows but the file has 2 cells`` (or ``genes``)
  and caused genuine shape mismatches to go undetected.  The check now resolves
  the index element correctly for both flat-dataset and group encodings, and
  honours the ``_index`` attribute for renamed indices.  ``standardise_gene_names``
  with ``inplace=True`` is fixed as a consequence.

Version 0.0.7
-------------

*Released 2026-07-14.*

* **Compatibility with anndata >= 0.13 / pandas >= 3.0** – the lightweight
  HDF5 metadata readers used by ``load_obs`` / ``load_var`` /
  ``standardise_gene_names`` / ``normalise_perturbation_labels`` /
  ``detect_perturbation_column`` / ``detect_gene_symbol_column`` /
  ``infer_columns`` and by the automatic DE-result reload path now understand
  the ``nullable-string-array`` group encoding.  With pandas >= 3.0, string
  index and columns default to the nullable ``StringDtype``, which anndata
  >= 0.13 writes to ``.h5ad`` as a group (``values`` + ``mask``) rather than a
  flat dataset; the readers previously assumed a flat dataset and raised
  ``TypeError: Accessing a group is done with bytes or str``.  Nullable
  integer/boolean columns and categorical categories stored in this encoding
  are handled as well.  Files written by older anndata / pandas versions
  continue to read unchanged.

Version 0.0.6
-------------

*Released 2026-07-13.*

* **Batch-corrected pseudo-bulk effect sizes** –
  ``compute_average_log_expression`` / ``cx.pb.average_log_expression`` and
  ``compute_pseudobulk_expression`` / ``cx.pb.pseudobulk`` now accept a
  ``batch_column`` parameter.  When provided, effects are computed *within
  each batch* and combined across batches with harmonic-count weights
  (``w_b = n_pert_b * n_ctrl_b / (n_pert_b + n_ctrl_b)``), removing
  batch-driven confounding when a perturbation and the control are unevenly
  represented across batches.  Batches where a perturbation has no cells (or no
  control cells) are skipped; a perturbation that shares no batch with the
  control raises ``ValueError``.  The batch column name and encountered batch
  labels are recorded in ``uns['batch_column']`` and ``uns['batch_ids']``.
  When ``batch_column`` is ``None`` (default), behaviour is unchanged.

* **Batch-corrected per-perturbation mean layers** – when ``batch_column`` is
  set, ``layers['perturbation_mean']`` / ``layers['perturbation_bulk']`` hold
  the **batch-corrected** per-perturbation expression (harmonic-weighted average
  of the within-batch means) instead of the pooled mean, and a new
  ``layers['control_mean_matched']`` / ``layers['control_bulk_matched']`` holds
  the per-perturbation weight-matched control reference, so
  ``X = perturbation_mean − control_mean_matched`` holds exactly.
  ``uns['control_mean']`` / ``uns['control_bulk']`` still carry the pooled
  control reference.  When ``batch_column`` is ``None`` (default), the pooled
  ``perturbation_mean`` is kept and no ``*_matched`` layer is written.

* **Bounded-memory batch path** – the per-``(perturbation, batch)`` sum
  accumulator -- the only quantity that grows with the number of batches -- is
  spilled to a disk-backed ``np.memmap`` and the streaming scatter-add is
  vectorised, so peak RAM stays ``O(chunk x n_genes + n_batches x n_genes +
  n_perturbations x n_genes)`` regardless of the number of gem-groups.

* **Memory budget for pseudo-bulk estimators** – ``cx.pb.average_log_expression``
  and ``cx.pb.pseudobulk`` now accept a ``memory_limit_gb`` argument and their
  namespace ``chunk_size`` default is ``None`` (auto-selected), matching the
  differential-expression functions.  The cell chunk size is auto-determined
  from the dataset shape and ``min(system memory, memory_limit_gb)``; passing an
  explicit ``chunk_size`` overrides it.  Only performance / peak memory is
  affected — computed values are identical regardless of the chunk size.

Version 0.0.5
-------------

*Released 2026-07-03.*

* **Batch-stratified (van Elteren) Wilcoxon test** – ``wilcoxon_test`` now
  accepts a ``batch_column`` parameter.  When provided, cells are ranked
  *within each batch* separately and the per-stratum U statistics are combined
  with unit weights (equivalent to a van Elteren test), removing rank
  inflation caused by batch effects.  Low-expression filtering, log-fold
  changes, and ``pts`` remain pooled across all cells; only the rank test is
  stratified.  Perturbations that share no batch with any control cell are
  marked untestable (NaN p-values).  Diagnostic metadata
  (``stratified_n_batches``, ``stratified_n_control_batches``,
  ``stratified_n_untestable_perturbations``, etc.) are stored in
  ``adata.uns``.

* **``output_path`` parameter for pseudo-bulk functions** –
  ``compute_average_log_expression`` and ``compute_pseudobulk_expression``
  now accept an explicit ``output_path`` argument, consistent with all other
  crispyx functions.  The old ``output_dir`` kwarg is retained for backward
  compatibility but is deprecated and will be removed in the next major
  version.

* **Format-aware masks-only QC (CSC fix)** – ``quality_control_summary`` with
  ``output_dir=None`` (masks only) now routes CSC inputs through a
  column-oriented counting path instead of row-slicing a backed CSC matrix,
  which was ``O(total_nnz)`` per chunk (~100x slower at genome scale). Output
  masks and statistics are byte-identical to the CSR path.

* **``normalize_total_log1p`` gains ``format_mismatch_policy``** – controls how a
  CSC source (slow for cell-streaming) is handled: ``"warn"`` (default, one
  actionable log message), ``"convert"`` (transparently stream via a
  bounded-memory temporary CSR copy, removed before returning), or ``"off"``.

* **Slow-axis guardrail** – ``iter_matrix_chunks`` now emits a single warning
  when a backed matrix is streamed against its slow axis (CSC by rows or CSR by
  columns), pointing to ``convert_to_csr`` / ``convert_to_csc``.

Version 0.0.4
-------------

*Released 2026-05-14.*

* **Scanpy-compatible ``groupby`` / ``reference`` parameter aliases** –
  ``t_test``, ``wilcoxon_test``, ``nb_glm_test``, and
  ``cx.tl.rank_genes_groups`` now accept ``groupby`` as an alias for
  ``perturbation_column`` and ``reference`` as an alias for
  ``control_label``, matching the parameter names used by Scanpy's
  ``sc.tl.rank_genes_groups``.  The original names remain the canonical
  names and are not deprecated.  Passing both a canonical name and its
  alias raises ``TypeError``.

* **Internal DRY refactor** – four private helpers (``_resolve_de_aliases``,
  ``_try_load_existing_de_result``, ``_print_de_summary``,
  ``_print_de_perturbation_verbose``) consolidate previously triplicated
  boilerplate across the three DE functions.  No behaviour change for
  existing callers.

* **Verbose improvements** – all three DE test functions accept
  ``verbose: int | bool``.  ``verbose=1`` prints a per-run summary
  (perturbations completed, mean genes tested).  ``verbose=2`` additionally
  prints per-perturbation gene-count lines.

* **Decoupled per-condition pct thresholds** – ``min_pct_both`` is complemented
  by independent ``min_pct_ctrl`` (default ``0.01``) and ``min_pct_pert``
  (default ``0.002``) parameters across all three DE test functions
  (``t_test``, ``wilcoxon_test``, ``nb_glm_test``) and the internal
  ``_low_expr_in_both_mask`` helper.  The lower ``min_pct_pert`` default
  prevents over-filtering genes induced from near-zero baseline
  (e.g. transcription-factor target genes).  The old ``min_pct_both``
  kwarg is retained as a convenience alias that silently sets both
  ``min_pct_ctrl`` and ``min_pct_pert`` to the same value.

* **Dual-condition pert filter with enabled ``min_mean_pert``** – The
  perturbed-side filter now always applies a dual condition:
  ``(pct_p < min_pct_pert) AND (mean_p < min_mean_pert)``.  The default
  ``min_mean_pert`` is raised from ``0.0`` (v0.0.3) to ``0.005`` so that
  genes with very few but high-count expressing cells (possible doublets or
  ambient RNA) are correctly excluded.  Existing code can restore the
  v0.0.3 behaviour by passing ``min_mean_pert=0.0``.

* **NaN initialisation for filtered-gene p-values (Wilcoxon)** – The
  standard single-pass Wilcoxon path previously initialised the chunk
  p-value array with ``np.ones`` (p=1.0) rather than ``np.nan``, causing
  filtered genes to appear as nominally non-significant rather than missing.
  The array is now initialised with ``np.full(..., np.nan)``, consistent
  with the streaming path and with ``t_test`` / ``nb_glm_test``.

Version 0.0.3
-------------

*Released 2026-05-13.*

* **Auto-reload for DE results** – ``wilcoxon_test``, ``t_test``, and
  ``nb_glm_test`` now accept a ``force: bool = False`` parameter.  When
  ``False`` (default) and the expected output ``.h5ad`` file already exists on
  disk, the functions load and return the saved result instead of rerunning the
  analysis.  Set ``force=True`` to rerun unconditionally and overwrite the
  existing file.  Combined with ``verbose=True``, a notice is printed to
  stdout identifying the reloaded file path.

* **Fixed ``RecursionError`` when pickling DE results** – ``AnnData.__getattr__``
  now guards against access before ``__init__`` has run (e.g. during
  ``pickle.load``), eliminating infinite recursion.  ``AnnData`` gains
  ``__getstate__`` / ``__setstate__`` so only the file path and access mode are
  serialised; the HDF5 handle is reopened lazily after unpickling.
  ``RankGenesGroupsResult`` and ``DifferentialExpressionResult`` likewise gain
  ``__getstate__`` / ``__setstate__`` that exclude the ``AnnData`` handle and
  group cache from the pickle payload, allowing round-trip serialisation with
  ``pickle.dumps`` / ``pickle.loads``.

* **Asymmetric low-expression filter** – DE tests (t-test, Wilcoxon, NB-GLM)
  now accept a ``min_mean_pert`` parameter (default ``0.0``). With the
  default, the mean-expression check is applied only to the *control* group;
  the perturbed group is filtered on fraction-of-expressing-cells
  (``min_pct_both``) alone. This prevents the filter from discarding genes
  that are induced from near-zero baseline expression, which is common in
  unbalanced CRISPR-screen comparisons. To reproduce the v0.0.2 behaviour
  pass ``min_mean_pert=min_mean_ctrl`` (e.g. ``min_mean_pert=0.05``).

Version 0.0.2
-------------

*Released 2026-04-28.*

* **Per-condition low-expression filter for DE tests** – t-test, Wilcoxon, and
  NB-GLM now accept ``min_pct_both`` (default ``0.01``) and ``min_mean_both``
  (default ``0.05``) parameters. A gene is excluded from a perturbation
  comparison (reported as NaN in ``pvalue`` / ``effect`` / ``logfoldchanges``)
  when the fraction of expressing cells *and* the mean expression are both
  below the respective thresholds in *both* the perturbation and control
  groups. Setting both thresholds to ``0.0`` recovers the 0.0.1 behaviour
  exactly. ``pts`` and mean expression values are always retained.

Version 0.0.1
-------------

*Initial release.*

* Streaming QC and preprocessing (filter cells, perturbations, genes;
  normalize and log-transform without loading the full matrix)
* Pseudo-bulk aggregation: average log expression and pseudo-bulk count
  matrices
* Differential expression: t-test, Wilcoxon rank-sum, NB-GLM with apeGLM
  LFC shrinkage, multi-core support, and adaptive memory management
* Dimension reduction: memory-efficient PCA and KNN graph construction on
  backed data
* Scanpy-compatible API and plotting: ``cx.pp``, ``cx.pb``, ``cx.tl``,
  ``cx.pl`` namespaces; rank genes plots, volcano, MA, PCA, UMAP, QC
  summaries, and overlap heatmaps
* Data preparation utilities: edit backed metadata, standardise gene names,
  normalise perturbation labels, auto-detect metadata columns
* HPC support: resume/checkpoint for long-running jobs, configurable
  ``memory_limit_gb``, Docker and Singularity support
* Benchmarking suite across 12 CRISPR screen datasets
