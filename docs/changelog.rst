Changelog
=========

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
