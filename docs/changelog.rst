Changelog
=========

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
