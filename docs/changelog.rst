Changelog
=========

Unreleased
----------

* **``nb_glm_test`` refuses input that is not raw counts.** Given
  normalised or log-transformed values it ran and returned results that
  looked plausible but were not: on a log1p copy of 1,716 Adamson cells the
  log fold changes correlated 0.93 with those from the counts, and the
  z-scores had half their spread. It now raises before any work, after
  reading one slice of the matrix.
* **Size factors default to library size; ``"geomeans"`` is added;
  ``"deseq2"`` takes the controls as the reference; ``"sparse"``,
  ``size_factor_scope`` and ``scale_size_factors`` are removed.**
  ``size_factor_method`` is ``"libsize"`` (each cell's total count, the
  default), ``"geomeans"`` (causarray's median of ratios on genes with a
  control mean of at least 2, against the controls' geometric means) or
  ``"deseq2"`` (DESeq2's median of ratios on the genes counted in every
  control, for pseudo-bulk and bulk; it falls back to ``"geomeans"`` with
  fewer than five such genes). Every reference is the controls, so a
  control's factor does not depend on the perturbations compared.
  ``"sparse"`` -- a median of ratios over every expressed gene, the default
  until now and what ``"deseq2"`` fell back to on single-cell data -- followed
  only part of a depth difference: on fake perturbations drawn from the
  deepest control cells it called nearly every well-counted gene up (mean z
  +5.6 and 6,578 false hits per fake on Huang-HCT116, against +0.11 and 22
  with library size), and real perturbations that differ in depth got
  inflated hit counts (176 against 9). Per-comparison size factors
  (``size_factor_scope``) recomputed every control's factor in each
  comparison, rested on the handful of genes counted in every cell compared,
  and could not run at scale; ``scale_size_factors`` changed no test. The user
  guide compares the methods (single-cell data; pseudo-bulk not yet).
  NB-GLM results computed with default size factors change.
* **Per-comparison NB-GLM runs at the scale of a genome-wide screen, as
  DESeq2 would.** Without covariates, ``dispersion_scope="per_comparison"``
  (without in-call shrinkage) now summarises the
  controls once -- a count histogram per gene and size-factor-binned
  moments -- and fits each perturbation from that summary and its own
  cells, following PyDESeq2's steps (linear-model means, gene-wise Cox-Reid
  dispersions, parametric trend, prior variance, MAP, outlier rule, IRLS
  with the mean floor and ridge, Wald test). Given the same size factors it
  matches PyDESeq2 to a median 1e-3 in z (z r >= 0.9999 on Adamson and
  Tian). On 20,000 Huang-HCT116 controls
  and 20 perturbations it takes 16 s and 3.4 GB on a 32-core Mac, against
  14 min per perturbation for the previous per-comparison code and 300 s
  and 58 GB for ``dispersion_scope="global"``; controls too large for
  memory are streamed. Standard errors no longer come out about 5 % small
  (they were computed at a gene-wise dispersion the LFC was not refitted
  at). Results with ``dispersion_scope="per_comparison"`` change.
* The dispersion trend falls back, as PyDESeq2's does, to the trimmed mean
  of the gene-wise dispersions when the parametric fit fails, instead of
  keeping a near-zero coefficient or fitting a polynomial.

Version 0.2.2
-------------

*Released 2026-10-25.*

* **``aggregate_pseudobulk`` and ``compute_pseudobulk_effects`` no longer
  load their result into memory to write it.** Both accumulated into
  temporary memory-mapped files and then read them whole to write the
  output. Profiles are now copied into the output a block of rows at a
  time, and effects are written a block of 512 genes at a time straight
  into the output (HDF5-chunked by gene block), with no temporary files.
  Values are bit-identical.
* **``convert_to_csc`` and ``convert_to_csr`` are 5-8x faster.** Each
  chunk's non-zeros were placed by a stable sort; one linear pass in
  source order places them the same way. On 300,000 FengTS cells
  (1.5 billion non-zeros, local SSD) the CSC conversion takes 14.5 s
  instead of 122 s, and 19 s instead of 119 s in four column bands; files
  are byte-identical. Most of a Wilcoxon run on a large screen was this
  conversion (78-86% on Huang).
* **Under a memory limit, a conversion in several bands reads its source
  once.** When the output buffers exceed half of ``memory_limit_gb`` the
  converters work in bands, and each band re-read the source. Where page
  cache counts against the limit (a Slurm or cgroup job) those re-reads
  come from storage. There the first band is now built in memory while the
  other bands' non-zeros are spilled to one file per band in the temporary
  directory (about 12 bytes each for float32 data), then read back band by
  band. A RAM-backed temporary directory (``tmpfs``) is skipped, since the
  spill would count against the same limit; the output's directory is used
  instead when it has room and, on the source's filesystem, only where the
  spill's write and read cost less than the re-reads; otherwise the
  re-reads stay. The soft open-file limit is raised toward the hard one
  when the bands need it. On a Slurm-based HPC cluster (source on Lustre,
  spill on node-local ``/tmp``), a 17.6 GB, 429,000-cell file converts in 5 bands in 330-400 s,
  against 600-720 s re-reading and 711 s for 0.2.1, and in 1 band in
  ~240 s. The full Huang-HCT116 screen (206 GB, 3.4 million cells, 3 bands,
  110 GB spilled) converts in 8,144-8,350 s, against 21,542 s for 0.2.1.
  Off such a limit a re-read is served from page cache and spilling was
  slower (Mac SSD, 4 bands: 26.5 s against 20.8 s), so nothing changes
  there. Files are byte-identical either way.
* **In-memory AnnData is accepted** by ``t_test``, ``wilcoxon_test``,
  ``nb_glm_test``, the QC filters and ``quality_control_summary``, the
  three pseudobulk functions, ``residual_effects``, ``subsample``,
  ``normalize_total_log1p`` and ``downsample_counts`` (and their ``cx.pp`` /
  ``cx.pb`` / ``cx.tl`` forms). The object is written to a scratch file in
  the temporary directory when it is on disk and has room (else beside the
  output), streamed like any input, and removed when the call returns or
  fails; a later run removes one a killed run left. ``output_path``, by
  keyword or position, is required with such input (except by
  ``quality_control_summary``, which then returns its masks only) and
  ``resume`` is refused, since both are defined by the input file. Large
  objects are still better saved once and passed as a path.
* **The GLM's linear algebra goes through factorisations.** No coefficient
  or standard error is computed from an explicit inverse or pseudo-inverse
  any more: two-column fits solve their 2x2 systems by an ``LDL'``
  factorisation (replacing Cramer's rule with its determinant clamped at
  1e-12), standard errors come from Cholesky solves, a singular
  weighted least-squares fit falls back to an SVD of the weighted design,
  and ``check_full_rank`` reads dependent columns off one QR. On Adamson
  this moves results by rounding only (1e-10 relative; the per-comparison
  MAP dispersion, refined to a 1e-4 tolerance, by up to 5e-6).
* **``crispyx.__version__`` is the running code's version.** It was read
  from the installed distribution's metadata, so a source checkout run with
  another crispyx installed reported -- and stamped into its outputs'
  provenance -- the installed version. The version is now set once, in
  ``crispyx/_version.py``, which the package and its build metadata both
  read.
* **Warnings point at the line that called crispyx.** Each warning named a
  fixed number of frames up, which a helper or the in-memory wrapper made
  land inside crispyx (with an absolute path to its source). Warnings are
  now attributed to the first frame outside the package.
* **Fixed: ``nb_glm_test``'s MAP dispersion is Cox-Reid adjusted.** The
  gene-wise dispersions were Cox-Reid adjusted but the MAP step that shrinks
  them toward the trend maximised the plain likelihood, which biases a
  dispersion low by about (n - p) / n for a design of p columns on n
  samples. With a categorical covariate of many levels on few samples (two
  control and two knockdown replicates per batch, 30 or 100 batches) the
  dispersions came out at 0.44-0.50x the truth and null z-scores had a
  robust sd of 1.30-1.36; they are now 1.01x and 1.00, as PyDESeq2's are.
  The prior variance also uses the design's residual degrees of freedom
  rather than assuming two columns. Results change only where the MAP step
  runs: with ``covariates`` and ``dispersion_scope="global"``, and with
  ``dispersion_scope="per_comparison"``. The default (no covariates, global
  dispersion) agrees to rounding (1e-10 relative). On Adamson with ``dispersion_scope=
  "per_comparison"``, dispersions rise by a median 2.4% and agree slightly
  better with PyDESeq2.
* **``nb_glm_test`` names a confounded covariate.** A comparison whose
  design is rank-deficient (every sample of a knockdown in a batch with no
  controls) used to be fitted as a singular system; it now fails with the
  offending column named, is NaN in the result, and is listed in the
  run's warning while the other comparisons go on. A categorical covariate
  level absent from a comparison's cells no longer adds an all-zero column.
  The ``covariates`` docstring says numeric columns are continuous, so a
  batch must be given as strings or a categorical.
* **``convert_to_csc`` and ``convert_to_csr`` evict the file they write.**
  The converted copy, often as large as the source, was written without
  eviction, so under a memory limit that charges page cache (a SLURM job)
  its pages stayed charged to the job into the step that reads it, a
  Wilcoxon test on the CSC copy. Each band is now written in 1 GB slices,
  evicted as they go, as the documentation already said. Nothing changes on
  a personal machine; files are byte-identical.

Version 0.2.1
-------------

*Released 2026-10-23.*

* **Differential expression stays inside ``memory_limit_gb``.** ``t_test``,
  ``nb_glm_test``, ``wilcoxon_test`` and ``shrink_lfc`` no longer hold, or map,
  a perturbations × genes result: each perturbation's row (or gene chunk's
  columns) is written into the result file as it is computed, a file
  allocated up front beside the output that becomes the output when the run
  finishes. Under a memory limit, each checkpoint evicts its pages, so they
  are not charged to a SLURM job's memory. Results are bit-identical to
  0.2.0's.
* **Changed: ``RankGenesGroupsResult`` reads its result file.** A matrix
  within 1% of the memory budget is read once and kept (as fast as before);
  a larger one is read on access (``result.pvalues[i]`` reads one row,
  ``np.asarray(result.pvalues)`` the matrix, ``order`` is computed per row),
  so returning a large result no longer costs its size in memory, and
  ``result["KO1"]`` works however large the result is (the lazy result
  ``wilcoxon_test`` returned for large outputs raised ``IndexError`` there).
  A later call that rewrites the same ``output_path`` changes what an earlier
  result object reads. ``u_statistics`` is ``None`` outside ``wilcoxon_test``
  (it was a matrix of zeros), and so is Scanpy's ``u_stat`` for ``t_test``
  and ``nb_glm_test``; ``t_test``'s Scanpy ``order`` now ranks untested genes
  last, like the other methods.
* **``wilcoxon_test`` writes its results in column blocks.** Each gene chunk
  of results was one small write per perturbation into a row-major file,
  each needing a page read before the write; on a network filesystem that
  is a round trip per perturbation per chunk (on Lustre: 219 s a gene
  chunk with the pages cached, 918 s when evicted, against 22 s now on the
  same node and load). The result's layers are HDF5-chunked as
  (rows, gene chunk width), ~1 MiB each, on every machine (on a local disk
  the two layouts write alike); values are unchanged. On a 1,500-perturbation
  Nourreddine subset writing to Lustre the test now takes 796 s, against
  1,089 s for 0.1.8. Both Wilcoxon paths print a read / compute / write
  split of their gene-chunk loop.
* **Removed: ``wilcoxon_test``'s group-batch streaming path.** With the
  results on disk, the single-pass path's memory is one gene chunk, whose
  width now accounts for every perturbation's results.
* **Faster, smaller ``nb_glm_test`` workers.** Each comparison's worker gets
  its own rows rather than every cell's label (pickled once per comparison,
  ~3.4 million strings on Huang-HCT116), checks that every perturbation has
  cells in one pass instead of one per perturbation, and results are written
  as they finish rather than in submission order. Requires ``joblib>=1.4``. ``uns["pvalue_correction"]`` is now
  recorded, as for the other methods.
* **QC's row-streaming cache is plain files** (``cache_mode="disk"``, was
  ``"memmap"``) in ``$TMPDIR``, read and written sequentially and evicted
  every GB, as are the outputs of the subset, concatenation, sort,
  normalisation and downsampling writers.
* **Fixed: QC read its whole output back into memory** to look up one
  ``obs`` column (a lazy ``obs``/``var``/``uns`` accessor materialised the
  AnnData, ``X`` included): 151 GB for Huang-HCT116, where QC peaked at
  134 GB and was killed on HEK293T after writing its output. The accessors
  now read only their element, and ``perturbation_keep`` counts labels once
  instead of comparing every label against every cell (8 min on
  Replogle-GW-k562).
* **``standardize_dataset`` no longer copies the dataset.** Only the
  metadata changes, so only the metadata is written: the standardized
  file's ``X``, layers and ``raw.X`` are HDF5 external links into the
  source (relative to the cache directory), read transparently by anndata
  and crispyx. Standardizing a 350 GB screen on Lustre took ~12 hours of
  copying, and the copy had come back short without raising three times;
  it is now seconds. The source must stay where it is.
* **Fixed: control inference picked a gene.** A control label is inferred
  only when exactly one label looks like one; several (Huang et al.'s
  ``Non-Targeting`` beside the chymotrypsin-like gene ``CTRL``) raise and ask
  for ``control_label``. An exact ``NTC`` now counts as control-like.
* **Streaming reads evict as they go, under a memory limit.**
  ``iter_matrix_chunks`` drops the pages it has read from the page cache every
  GB, so one pass over a file larger than the job no longer fills its cgroup
  with cache (QC on Huang-HCT116: 117 GB of it). Like every eviction in
  crispyx, it happens only when a cgroup memory limit is visible or the
  process runs in a Slurm job; elsewhere the kernel keeps the cache.
* **When the cgroup limit is not visible** (a job inside a container),
  the memory budget falls back to Slurm's ``SLURM_MEM_PER_NODE`` /
  ``SLURM_MEM_PER_CPU``, and asks for ``memory_limit_gb`` when under Slurm
  without either.
* **One memory budget.** ``quality_control_summary``, ``nb_glm_test``'s
  worker and dense-matrix sizing and ``shrink_lfc``'s worker count read
  ``memory_limit_gb`` capped by the cgroup's headroom (the memory available
  when ``None``), instead of each detecting memory its own way.

Version 0.2.0
-------------

*Released 2026-10-15.*

* **Docs reorganised.** The usage guide is split into task pages under a new
  *User guide* section (loading and preparing data; quality control and
  dimension reduction; pseudo-bulk and effect sizes; differential expression;
  covariate adjustment; custom batch statistics; working with large data), and
  a *Quick start* page opens *Getting started*. The covariate-adjustment
  tutorials are listed under the page comparing the approaches, retitled after
  the method each one illustrates; *Reference* keeps the API, the comparison
  with other tools and the FAQ. ``usage.html`` no longer exists.
* **New: standard errors from ``pb.normalized_effects``** for
  ``method="mean_log1p"``, after Liu et al. (2026,
  `doi:10.64898/2026.10.05.756839 <https://doi.org/10.64898/2026.10.05.756839>`_):
  ``layers["standard_error"]`` (cells independent within a batch, one
  within-batch variance per arm pooled over batches with two or more cells)
  and ``layers["standard_error_robust"]`` (from the spread of the per-batch
  effects; NaN with fewer than two batches). Without ``batch_column`` the
  data is one batch, so the effect and its standard error are the plain
  two-sample ones; that path now streams through the same code, one pass
  however many perturbations there are.
  Its per-(perturbation, batch) sums stay in memory when they fit half the
  memory budget, and go to temporary files otherwise (they always went to
  disk); the results are then finished a block of perturbations at a time,
  so nothing of the size of the output is held beside it;
  ``perturbation_column`` and ``batch_column`` may be ``pd.Series``
  indexed by cell, as in ``wilcoxon_test`` and ``residual_effects``. Checked
  against the effects and standard errors Liu et al. published for the Feng
  et al. targeted screen: they agree to float32 precision. New tutorial,
  *Batch-corrected effects and Wilcoxon tests*, on that screen.
* **Fixed: ``t_test`` with a one-cell group.** A perturbation (or control)
  with a single cell has no variance estimate; it was treated as zero, so
  the test used the other side's standard error alone and reported nearly
  every gene as significant. Such comparisons are now left untested (NaN),
  as in scanpy.
* **Fixed: an ``output_path`` equal to the input file destroyed it.**
  ``pp.subsample`` (and the QC filter) wrote the empty output skeleton over
  the source and then failed reading it; DE functions replaced the input with
  their result; ``normalize_total_log1p``, ``convert_to_csc``/``convert_to_csr``
  and ``sort_by_perturbation`` overwrote it too. Every writer now raises
  ``ValueError`` instead, also when a default output name (from
  ``data_name``) is the input's. Subset
  writes also go to a partial file renamed on success, so an interrupted
  ``subsample`` or QC filter leaves no half-written output.
* **Fixed: cells with a missing perturbation label** were tested as a group
  named ``"nan"`` by ``t_test``, ``wilcoxon_test`` and ``nb_glm_test``. They
  are now in no comparison, as in ``residual_effects``. A labels Series with
  ``NaN`` for some cells is accepted on the same terms.
* **Fixed: ``residual_effects`` z-scores were too large with few controls.**
  Every perturbation's effect shares the error of the control fit, which
  neither null counted: with 50 controls against 200 cells per target, 41%
  of null tests had p < 0.05. ``null="model"`` now divides by
  ``sigma_g * sqrt(1/n_t + h_t)``, with ``h_t`` the control fit's leverage at
  the perturbation's mean design (small with many controls: in the Feng et
  al. tutorial, 10.5% of target x gene pairs are significant instead of
  10.6%); ``null="robust"`` centres on the median
  over perturbations. ``obs["leverage"]`` and ``var["robust_center"]`` are
  written so either z can be recomputed.
* **Fixed: ``residual_effects`` could drop a covariate silently.** The rank
  cutoff ran on ``X^T X``, squaring its condition number, so a covariate of
  large magnitude but modest spread was projected out and effects came out
  wrong. Covariates are now centred and scaled over the controls first.
* **Fixed: ``nb_glm_test`` with controls too large for memory** returned
  all-NaN results whenever covariates, per-comparison size factors or
  per-comparison dispersion were used: only the control cache streams the
  controls from disk, and every other path received no control matrix. It
  now raises ``MemoryError`` with the alternatives up front, and any
  perturbation that fails raises a ``UserWarning`` instead of only a log line.
* **Fixed: ``nb_glm_test``'s fallback from ``size_factor_scope="per_comparison"``**
  on data too large to load used unit size factors (no normalisation at
  all) rather than the global ones its message promised. It now computes
  global size factors, warns, and records ``"global"`` in
  ``uns["size_factor_scope"]``. The per-comparison MAP step also used the
  global offsets instead of the comparison's.
* **Fixed: ``nb_glm_test``'s control-cache fit** estimated its coefficients
  at a per-comparison moments dispersion and reported their SE at the global
  one; it now fits at the global dispersion, matching the uncached fit (log2
  fold changes move by up to ~0.01). A fixed ``dispersion=`` with
  ``shrink_dispersion=False`` was ignored on this path and is now used.
* **Fixed: ``nb_glm_test`` with covariates and ``dispersion_scope="global"``**
  gave a gene almost absent among the controls (one a perturbation switches
  on) a huge dispersion extrapolated from the trend, costing it its test.
  Genes with fewer than 10 control counts now have their dispersion estimated
  within each comparison; ``layers["dispersion"]`` is written instead of
  ``var["dispersion"]`` when any gene needs it.
* **Fixed: ``pb.aggregate`` with nullable covariates.** A donor-level
  ``boolean``/``Int64`` column with missing values, as ``concat_h5ad`` makes
  for a column only some files have, was carried over as an object array
  anndata could not write. Dtypes are now kept. Finding the constant columns
  no longer runs a grouped ``nunique`` over every obs column (one sort, then
  a comparison per column), and a column of unhashable values counts as
  varying instead of raising. Integer columns are compared exactly (64-bit
  IDs above 2**53 are distinct).
* **Fixed: ``concat_h5ad`` and subset writes loaded every layer into
  memory**, once per file for every pass: they opened each file as a backed
  AnnData, which anndata fills with every layer, ``obsm`` and ``obsp``. They
  now read metadata and stream each matrix through h5py, so no layer is ever
  held in memory, as documented.
* **Changed: ``cx.read_backed`` (and ``cx.AnnData.backed``) leave the layers
  on disk.** ``anndata.read_h5ad(backed="r")`` keeps ``X`` on disk but reads
  every layer into memory, each as large as ``X``; every crispyx function
  opens its input this way, so a file with a ``counts`` layer cost that
  layer's size in memory before any work began (124 MB held vs 3 MB on a
  20,000 x 500 test file with two layers). Layers are now sparse datasets
  or HDF5 datasets read as they are sliced, like ``X``; the other slots are
  read as before, and ``to_memory()`` loads everything. A CSC layer streamed
  by rows is now re-read per chunk, as a CSC ``X`` already was.
  ``read_backed`` also takes ``mode="r+"``.
* **Fixed: CSC inputs to ``concat_h5ad`` and ``write_filtered_subset``**
  (``pp.subsample``, the QC filter) were streamed by rows, re-reading the
  whole matrix for every chunk. Both take ``format_mismatch_policy``
  (default ``"auto"``) and stream a CSC ``X`` from a temporary CSR copy
  beside the output, like the other streaming functions. ``concat_h5ad``
  checks the shared cells by reading a CSC ``X`` by columns, with no copy,
  and makes each file's copy only while writing that file's cells, so the
  scratch space needed is that of the largest file, not of all of them.
* **Fixed: ``concat_h5ad`` with an obs column of numbers in one file and text
  in another** failed to write, after a full pass over every matrix. Such
  columns, categorical or not, are now stored as text, and obs is merged
  before any matrix is read. Categorical columns keep their categories' order and ``ordered``
  flag when every file agrees (they were re-sorted and unordered), and
  otherwise take the categories in order of first appearance.
* **Fixed: a run killed while rewriting ``obs`` could not be resumed.** The
  streaming Wilcoxon test deleted its work file's ``obs`` before copying the
  new one in; killed in between, every resume failed with ``KeyError``.
  ``write_obs``/``write_var`` (now also used there) write the new table
  beside the old and swap it in by renames; a swap cut short is finished by
  the next crispyx read or write of the file (``cx.AnnData``,
  ``read_backed``, ``load_obs``, ...), after which anndata reads it too. They encode with anndata's ``write_elem``
  instead of a hand-written encoder that turned missing text into ``""``,
  could not store nullable columns and overflowed past 32,767 categories.
* **Changed: raw-count checks accept float rounding noise** (``2.0000001``)
  and ``downsample_counts`` rounds such values instead of truncating them.
  The noise allowed is absolute (``1e-6``, or two units in the last place of
  the value's dtype), so a large value with a fractional part such as
  ``612345.4`` is still refused.
* **Changed: crispyx requires anndata 0.11 or later** (``anndata.io``).
* **Changed: ``residual_effects`` leaves out perturbed cells in a covariate
  level no control shares**, instead of refusing the whole run; infinite
  numeric covariates raise, and so do missing ones in the cells used (a
  missing value is not a level of its own). ``perturbations`` given as a generator no longer
  prevents reuse of an existing result (also for the DE functions).

Version 0.1.10
--------------

*Released 2026-10-15.*

* **New: ``cx.tl.residual_effects``** (also ``cx.residual_effects``), the
  covariate-adjusted effects of Feng et al. (2026, `doi:10.1016/j.xgen.2025.101076
  <https://doi.org/10.1016/j.xgen.2025.101076>`_). Each gene is fitted to
  the covariates over the control cells, and a perturbation's effect is its
  cells' mean residual. z-scores use the control fit's noise
  (``null="model"``) or the spread across perturbations (``null="robust"``).
  ``covariates`` is required: ``obs`` columns, plus ``log_total_counts`` and
  ``mito_fraction`` computed from the counts. Adjust for donor, lane or
  batch; many perturbations change depth and mitochondrial share, so
  adjusting for those removes part of the effect.
  One pass over the raw counts on disk; cells with a missing label are not
  used. Every chunk is checked for raw counts, and collinear covariates raise
  when they would make a perturbation's effect arbitrary.
  ``format_mismatch_policy`` and ``cx.estimate_disk_usage`` cover the
  temporary CSR copy of a CSC source. On 7.4 million CD4+ T cells (27.6
  billion non-zeros) one call takes under 2 minutes on 32 cores with the file
  in the page cache. New tutorial on that screen: ``residual_effects`` and
  the authors' own model (``nb_glm_test`` on guide x donor pseudo-bulk,
  which reproduces their estimates) against the authors' estimates, and a
  calibration check on fake targets made of non-targeting guides. A second
  tutorial reproduces Feng et al.'s published effects and significance calls
  for their targeted iPSC screen (1.16 million cells), per target and per
  target in each cell line, and compares crispyx's Wilcoxon test.
* **Fixed: ``nb_glm_test`` with covariates** crashed with ``TypeError``
  (missing ``min_mu``) whenever four or more perturbations were fitted on
  worker processes.
* **Faster: ``nb_glm_test`` with covariates** estimates each gene's
  dispersion once, on the control cells with the covariates (DESeq2's MAP
  step), and fits every comparison at that dispersion, as
  ``dispersion_scope="global"`` (the default) already did without
  covariates; before, the option was ignored and every comparison re-ran the
  dispersion search. About twice as fast on the tutorial's pseudo-bulk, and
  slightly closer to the authors' DESeq2 estimates. The dispersion is stored
  once in ``var``. ``dispersion_scope="per_comparison"`` keeps the old fit.
* **Fixed: ``nb_glm_test(dispersion=...)``** was accepted but never used; a
  fixed dispersion now applies to every gene and comparison.
* **Removed: ``nb_glm_test(share_dispersion=...)``**, which was accepted but
  never used; ``dispersion_scope="global"`` is that option.
* **Faster: streaming a sparse matrix along its fast axis** slices
  uncompressed files directly instead of through anndata (about twice as
  fast), and ``write_filtered_subset`` / ``concat_h5ad`` no longer read row
  chunks they do not keep or read rows just to count them.
  ``iter_matrix_chunks`` takes ``selected`` (skip chunks without a selected
  row) and ``read_ahead`` (read the next chunk on a background thread).

Version 0.1.9
-------------

*Released 2026-10-11.*

Follow-ups from using crispyx in the causarray preprocessing. Several
changes alter what results contain, so DE and pseudo-bulk results written by
0.1.8 are recomputed once.

* **Changed: each DE effect quantity is named for what it is.**
  ``effect_size`` is gone from ``RankGenesGroupsResult`` and from the
  per-group ``result[label]``. It meant ``AUC - 0.5`` for ``wilcoxon_test``,
  the difference of means for ``t_test`` and the log fold change for
  ``nb_glm_test``, and ``to_rank_genes_groups_dict()`` exported all three as
  ``"auc"``. Now ``logfoldchanges`` is on every result (including
  ``result[label].logfoldchanges``); ``auc`` (``wilcoxon_test``) and
  ``mean_difference`` (``t_test``) are ``None`` for the other methods, and
  ``"auc"`` is written to the Scanpy structure only for Wilcoxon.
* **Changed: a Wilcoxon result's** ``X`` **is the AUC**, ``U / (n_pert *
  n_ctrl)`` on ``[0, 1]`` with 0.5 for no shift, instead of ``AUC - 0.5``.
  Layer names are unchanged.
* **Changed: ``aggregate_pseudobulk`` requires** ``method``, names each
  profile by its group key (``"donor1|L2/3 IT"`` instead of
  ``pseudobulk_00000000``) and carries every ``obs`` column that is constant
  within a profile (donor sex, age, disease, ...); columns that vary within
  one are listed at ``verbose=1``. ``method="sum"`` is what count models
  need; the old ``"mean_log1p"`` default was silently wrong for them. The
  usage guide shows how ``n_cells`` forms the donor x cell-type composition
  table, a check before pooling cell types.
* **New: ``cx.pp.concat_h5ad``** concatenates h5ad files on disk and keeps
  one copy of the cells they share (checked to be the same cell by their
  expression; differing ``obs`` values are warned about), or refuses any
  overlap with ``shared_cells="error"``. It keeps the genes every file has,
  ``X`` and the common layers, the union of ``obs``, the common ``obsm``
  entries (arrays, sparse matrices and DataFrames), and can label each
  cell's source file. Each file's matrix is read twice: once to size the
  output and check the shared cells, once to write it.
* **Fixed: a DE run interrupted under an older result layout restarts**
  instead of resuming, so a 0.1.8 Wilcoxon checkpoint is not finished with
  0.1.9's AUC values next to its ``AUC - 0.5`` ones.
* **New: ``wilcoxon_test`` accepts the labels as a ``pd.Series``** indexed by
  cell, so a label-permutation null needs no relabelled copy of the data.
  Each Series (its values and name) is its own call: a fixed ``output_path``
  is recomputed for a new shuffle and reused for the same one.
* **New: ``obs["n_genes_tested"]``** in every DE result counts the genes each
  perturbation was tested on after the low-expression filters, and the
  ``wilcoxon_test`` docstring explains how those filters differ from
  ``scanpy.tl.rank_genes_groups``.
* **Fixed: ``PseudobulkMethod`` listed** ``"log_mean"``, which
  ``aggregate_pseudobulk`` rejects (it belongs to
  ``compute_normalized_effects``). Each function now validates against its
  own ``Literal``.
* An in-memory ``AnnData`` is rejected with the line that fixes it
  (``adata.write_h5ad(...)``), and the ``data`` docstrings say a backed one.
* ``shrink_lfc(method="full")`` no longer strips a ``/workspace/`` prefix or
  searches ``.cache`` directories for a missing original dataset; it says
  which path is missing.

Version 0.1.8
-------------

*Released 2026-10-05.*

* **Changed: tied genes are untested in ``wilcoxon_test``.** A gene whose
  values are all tied across control and perturbation (e.g. zero in every
  cell, which only reaches the test with the low-expression filters off)
  used to be reported as ``score=0``, ``pvalue=1``, ``logfoldchanges=0``, as
  in Scanpy. The rank test is undefined there, so it is now ``NaN`` in every
  derived column, matching ``t_test``, ``nb_glm_test`` and every other
  untested gene -- with or without ``tie_correct`` and with or without
  ``batch_column`` (where a gene tied within every batch is untested). DE
  results written by 0.1.7 are therefore recomputed once.
* **Removed: ``min_pct_both``** from ``t_test``, ``wilcoxon_test`` and
  ``nb_glm_test``. It was deprecated in favour of ``min_pct_ctrl`` and
  ``min_pct_pert``, but the DE functions accepted it without a warning.
  Pass ``min_pct_ctrl`` and ``min_pct_pert`` instead.
* **Fixed: ``shrink_lfc(method="full")`` ignored ``memory_limit_gb``.** The
  parameter was documented to cap the parallel workers but was never read.
  The worker count is now capped so the workers fit in the budget (or in the
  memory available, when it is ``None``); ``method="stats"`` is unaffected.
* An argument left at ``None`` no longer counts when deciding whether an
  existing result or a checkpoint can be reused, so adding or removing an
  optional parameter in a release does not force every cached result to be
  recomputed or throw away an interrupted run. A ``resume=True`` call that
  cannot use the checkpoint it finds now warns and says why before starting
  over.
* **Fixed: aliases and** ``cx.tl.rank_genes_groups`` **did not share
  results.** ``groupby``/``reference`` and ``perturbation_column``/
  ``control_label`` now name the same call, and ``rank_genes_groups`` reuses
  (and is reused by) a direct DE call on the same file. ``wilcoxon_test``
  also checks for a reusable result before reading the input's labels.
* **Fixed: ``wilcoxon_test(scanpy_format=True)``** skipped
  ``uns["rank_genes_groups"]`` when the result was too large to load, while
  recording that it had been written. It is now written from the file, a
  block of groups at a time, at any size. The result's provenance is also
  written with it, so a run killed while the result is read back is reused
  rather than recomputed.
* ``nb_glm_test`` records whether it fitted with frozen control statistics
  in ``uns["frozen_control"]``. With ``freeze_control=None`` (chosen from the
  memory available), an existing result is reused whichever mode it used.
* ``shrink_lfc(method="full")`` also budgets the batches joblib queues for
  each worker, and sizes the workers once rather than per perturbation.

Version 0.1.7
-------------

*Released 2026-10-03.*

* **Fixed: stale results.** The DE functions, ``batch_process``, the
  pseudo-bulk functions, ``sort_by_perturbation`` and ``standardize_dataset``
  could return an existing output after its input or a result-shaping
  argument (e.g. ``perturbations``) had changed. Each output now records
  the input file, every result-shaping argument, the crispyx version and its
  layout in ``uns["crispyx"]``, and is reused only when these match; the
  notice says why one is recomputed. Execution-only arguments
  (``chunk_size``, ``memory_limit_gb``, ``n_jobs``, ...) do not invalidate a
  result.
* Outputs from 0.1.6 or earlier carry no such record and are recomputed once
  after upgrading.

Version 0.1.6
-------------

*Released 2026-09-26.*

This release makes crispyx outputs smaller without changing any computed
value: every number the DE functions stored before is stored now, bit for
bit, just not more than once. The on-disk layout of the DE results changes,
so code that reads the layers below directly needs updating; results loaded
through crispyx (``RankGenesGroupsResult``, ``cx.pl``, ``shrink_lfc``) are
unaffected.

* **New: ``cx.compress_h5ad(src, dst)``** losslessly re-encodes any
  ``.h5ad`` with HDF5's built-in gzip + shuffle filters, verifies it byte for
  byte, and only then puts it in place (``dst=src`` replaces the source).
  Chunks are compressed on every core, memory stays bounded, and the result
  opens in any HDF5 reader without plugins. Single-cell matrices shrink to
  ~15-30 % of their size and dense DE matrices to ~55-75 %. See
  :ref:`compress-h5ad`.
* **Compressed inputs stream several times faster.** When a backed matrix is
  stored with gzip (with or without shuffle) and streamed along its fast axis,
  crispyx now reads the compressed chunks directly and inflates them on a
  thread pool instead of on the calling thread: 8-10x faster than before on
  the files measured, with identical blocks. Uncompressed files take the same
  path as before.
* **``nb_glm_test`` output is ~40 % smaller** (about half with
  ``lfc_base="ln"``), with a matching drop in write time and peak memory:

  - ``layers['logfoldchanges']`` is gone -- it was always identical to
    ``X``. ``layers['logfoldchange_raw']`` is written only when ``X`` is
    shrunk (``lfc_shrinkage_type="apeglm"`` or after ``shrink_lfc``);
    otherwise the MLE fold change *is* ``X``.
  - With ``lfc_base="ln"``, ``logfoldchange_raw_ln`` / ``standard_error_ln``
    are no longer written, since the main layers are already on the ln
    scale.
  - A global dispersion (the default ``dispersion_scope="global"``) is one
    value per gene, now stored once as ``var['dispersion']`` and
    ``var['dispersion_trend']`` instead of three identical-row matrices
    (``dispersion_raw`` was a third copy). A per-comparison dispersion keeps
    its three layers.
  - ``converged`` is stored as ``bool`` and ``iterations`` as the narrowest
    integer type that holds its largest value (both were float32).
* **``pts_rest`` is a ``var`` column in every DE output.** It is the control
  arm's detection rate -- one value per gene -- and was repeated for every
  perturbation in ``t_test``, ``wilcoxon_test`` (all three paths) and
  ``nb_glm_test``. ``RankGenesGroupsResult.pts_rest`` still has one row per
  perturbation, and ``scanpy_format=True`` still writes scanpy's per-group
  array.
* **``shrink_lfc``** no longer duplicates ``X`` as ``layers['logfoldchanges']``.
  It writes the MLE fold change to ``logfoldchange_raw`` (and, with an ln
  base, the MLE standard error to ``standard_error_ln``) before replacing
  ``X`` and ``standard_error`` with their shrunk values.
* **``batch_process`` with ``channels``** no longer writes the first channel
  twice: it is ``X``, and later channels are ``layers[name]``.
* **``resume=True`` now resumes interrupted DE runs, in every method and
  path.** Partial results used to live in a temporary directory that an
  interrupted run took with it, so ``nb_glm_test`` returned an all-``NaN``
  row for every perturbation its checkpoint marked as done (``t_test`` the
  same with zeros), and ``wilcoxon_test`` refused to resume at all. The
  partial arrays now live beside the output in a hidden
  ``.<output name>.resume`` directory, removed on success, and the
  checkpoint records a fingerprint of the call (input file, perturbations and
  result-affecting parameters). A run resumes only from its own checkpoint --
  anything else is discarded and the run starts over -- and a resumed run's
  output is bit-identical to an uninterrupted one. Memory and chunking
  settings (``memory_limit_gb``, ``max_dense_fraction``, ``cell_chunk_size``,
  ``chunk_size``, ``irls_batch_size``) are not part of the fingerprint, so a
  run killed for memory resumes with a lower limit; a different chunk size
  sums cells in a different order, so the rows fitted after such a resume
  can differ in the last bits. Every ``wilcoxon_test`` path writes its
  file under a partial name and renames it into place, so a file at the
  output path is always complete: an interrupted run used to leave a
  half-filled output that the next call loaded as a finished result. Disk estimates for
  the DE functions are now reported under ``"output"`` instead of
  ``"tempdir"``.
* **``t_test`` reports ``NaN`` for a perturbation it could not test** (e.g.
  one named in ``perturbations`` with no cells), as ``nb_glm_test`` does. It
  used to report ``pvalue = 0`` for every gene of it. ``t_test`` also
  rejects an unknown ``corr_method`` before reading the data, not after a
  full pass over it.
* **``n_jobs`` follows joblib in ``t_test`` and ``nb_glm_test``**, capped at
  the CPUs the process may use (affinity mask and cgroup quota): ``None``
  and ``-1`` use them all, ``-2`` all but one, and ``0`` is an error.
* **``cx.tl.rank_genes_groups(method="t-test")`` returns ``t_test``'s own
  result**, like the other methods. It used to rebuild the result, storing
  the mean difference as ``logfoldchanges``, zeroing ``pts`` and
  ``pts_rest``, and rewriting the file with per-group arrays several times
  its size. ``t_test`` gains ``corr_method`` (``"benjamini-hochberg"`` or
  ``"bonferroni"``) for it to pass through.
* **Reloading an existing ``t_test`` result keeps its effect size.** The
  reloaded ``effect_size`` was the fold change.
* **Wilcoxon results store their ``uns`` metadata where anndata reads it.**
  It was written as HDF5 attributes, so ``adata.uns`` came back empty and
  ``cx.pl.materialize_rank_genes_groups`` lost the method, groupby and
  reference.
* **``t_test`` no longer writes result matrices it then throws away.** It
  wrote eight ``uns/rank_genes_groups/full`` matrices during the run that the
  final write truncated.
* **Streaming ``wilcoxon_test``** no longer builds an
  ``n_perturbations x n_genes`` placeholder matrix in memory before writing
  its output.
* **Temporary files no longer leak on failure.** The QC gene-filter cache in
  ``$TMPDIR`` (about the size of the kept cells' matrix) was removed only
  after gene filtering succeeded, and ``sort_by_perturbation`` left its
  ``.meta.h5ad`` file behind on error.

Version 0.1.5
-------------

This release reworks the GLM solver. Estimates change for lowly-expressed
genes -- toward, not away from, the reference implementations -- so results
from 0.1.4 on covariate-adjusted or low-count genes will not reproduce
exactly. ``nb_glm_test`` keeps its own ``min_mu`` default of 0.5, so the
DESeq2-compatible path is unchanged.

* **A filtered gene is now reported the same way everywhere: untested.**
  ``nb_glm_test`` already reported an excluded (gene, perturbation) pair as
  ``NaN`` throughout; ``t_test`` and ``wilcoxon_test`` did not. Depending on
  which of the three Wilcoxon paths ran, a gene dropped by
  ``min_cells_expressed`` came back with ``p = 1.0`` and a fold change of
  exactly 0.0 -- "tested, no change", asserted about a gene nobody tested --
  or with a ``NaN`` p-value beside that same 0.0, and ``t_test`` returned a
  ``NaN`` p-value beside a finite fold change and effect size. Worse, a gene
  excluded for one perturbation but tested for another picked up that other
  perturbation's default ``p = 1.0``, so a result row depended on which
  perturbations happened to be in the same run. All three Wilcoxon paths and
  ``t_test`` now write ``NaN`` to every column derived from the comparison --
  ``score``, ``pvalue``, ``pvalue_adj``, ``u_statistic``, ``logfoldchanges``,
  ``effect_size`` -- for every gene they did not test, and nothing is spelled
  as 0.0 or 1.0 to mean "untested". Because a ``NaN`` p-value does not enter
  the Benjamini-Hochberg denominator, adjusted p-values on the genes that
  *were* tested get slightly smaller; that is the correct denominator.
  ``nb_glm_test`` had one such number left of its own: with
  ``lfc_shrinkage_type="apeglm"``, an untested gene's standard error was
  reported as the literal 1.0 that the per-gene shrinkage falls back to. It is
  ``NaN`` there now as well.
* **``pts`` and ``pts_rest`` survive the filters.** The Wilcoxon paths zeroed
  the fractions of expressing cells for excluded genes, which is exactly the
  information needed to see that a gene absent from one arm was excluded at
  all. They are descriptions of the data, not of the comparison, and are now
  reported for every gene, as ``t_test`` and ``nb_glm_test`` already did.
* **``t_test`` no longer folds untested cells into the last perturbation.**
  With an explicit ``perturbations=`` subset, every cell belonging to a
  perturbation outside that subset was accumulated into the last tested
  group's sums, means, variances and cell count, because the unselected
  labels code to ``-1`` and ``-1`` indexes the last row. Testing one
  perturbation out of three therefore compared control against that
  perturbation *plus both untested ones*. Only ``t_test`` was affected;
  ``wilcoxon_test`` and ``nb_glm_test`` select cells by label.
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
  observation remains visible in ``pts`` and ``pts_rest``. Shrinkage does not
  reach them either -- an excluded pair is dropped before any fit, so it stays
  ``NaN`` under ``lfc_shrinkage_type="apeglm"`` too. Controlled by the new
  ``nb_glm_test`` parameters ``min_cells_ctrl`` and ``min_cells_pert``, both
  defaulting to 1 -- symmetric, and the exact boundary between an effect that
  exists and one that does not. They are separate because the informative
  direction depends on the screen;
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
