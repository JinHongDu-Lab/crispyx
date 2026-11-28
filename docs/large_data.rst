Working with large data
=======================

Storage formats, memory, disk space, messages, and re-running or resuming long jobs. Most of crispyx streams from disk; these pages say what that costs and how to control it.


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

.. _csc-disk-note:

.. note::

   For small datasets (files < ~12.5 GB), the CSC conversion overhead is
   negligible (< 1 s on fast NVMe) and the total wilcoxon time is dominated
   by process startup (2–3 s). For a large screen the conversion itself
   takes seconds per GB on a local SSD (300,000 Feng-ts cells, a 12 GB file
   of 1.5 billion non-zeros: 15 s) and eliminates the ~18× repeated
   full-file scans that CSR imposes.

   ``memory_limit_gb`` bounds the conversion's output buffers at half of it.
   An output larger than that is built in bands of columns. In a Slurm or
   cgroup job, the source is read once and the later bands are spilled to
   the temporary directory (``$TMPDIR``; about 12 bytes per non-zero for
   float32 data), so put ``$TMPDIR`` on node-local disk. A RAM-backed
   ``$TMPDIR`` (``tmpfs``) is skipped, as the spill would count against the
   job's memory. Elsewhere, each band reads the source again from page cache.

   Because the source and destination coexist during conversion, this
   temporarily requires roughly 2× the source file's size in free disk space
   (e.g. ~54 GB for a 27 GB screen). crispyx warns automatically if the
   output filesystem looks too tight; check up front with
   ``cx.estimate_disk_usage("convert_to_csc", path)`` (see :ref:`disk-space`).

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
read in row-chunks (axis=0).

Sorting by perturbation
-----------------------

A CSR file reads a perturbation's cells quickly only when they sit together.
When they are scattered through the file, each row is its own read, and a
compressed file decompresses every HDF5 chunk a row touches, so a fit that
needs a few gigabytes can read the whole file many times over. For that
reason ``nb_glm_test`` and ``nb_offset_test`` check the input before reading
it and, when it is large (360,000 cells and 100 perturbations or more) and
scattered (under half of a perturbation's cells contiguous), write a copy
sorted by perturbation next to the input (``<name>_sorted.h5ad``, the
controls first) in one streaming pass, and run on that. The step is printed
when ``verbose`` is on; the copy is reused by later runs on the same input,
and passing it as the input skips the check. Sort once up front with
``cx.data.sort_by_perturbation`` for a file under those thresholds that is
still slow (a smaller screen on a network filesystem), or for a script of
your own that reads the cells of each perturbation. When the file goes
through QC first, ``cx.pp.qc_summary(..., sort_by_perturbation=True)`` writes
the filtered file in that order directly, from the same passes over the
source, so no separate sort is needed.

NB-GLM then streams the CSR file:

* Size factors are computed in blocks of ``chunk_size`` rows (2,048 by
  default), never holding the matrix. ``"deseq2"`` holds only a
  perturbations × reference-genes count table.
* Without covariates, the control cells are summarised in one pass, 4,096 at
  a time (each gene's count histogram and size-factor-binned moments), and
  every comparison is fitted against that summary. Peak memory is
  O(chunk_size × n_genes) plus the largest perturbation's cells, not
  O(n_control × n_genes).
* With covariates, the control cells are held in memory.

After each streaming phase,
``drop_file_cache()`` evicts file data from the kernel page cache so that
cgroup-limited environments (e.g. SLURM) do not count cached pages toward the
memory limit. Streaming reads, and files being written -- a DE result as it is
filled, QC's chunk cache and its filtered output, sorted and converted copies
-- are evicted the same way every gigabyte or checkpoint. This happens only
when page cache counts against a limit: a cgroup memory limit is visible, or
the process runs in a Slurm job. On a personal machine the kernel manages
the cache on its own and nothing is evicted, so a second pass over a file
still reads from memory.

.. note::

   If ``nb_glm_test()`` detects CSC storage, it emits a ``UserWarning``
   advising conversion to CSR. In the benchmark pipeline, CSR conversion
   is handled automatically via the ``crispyx_standardize_csr`` step.

.. _memory-model:

Memory model
------------

crispyx aims for memory you can predict before a run: each step holds a chunk
of cells or genes rather than the matrix, and where it takes
``memory_limit_gb`` (detected from the machine or the job when not given; on
a Slurm job set it to ``--mem``) it sizes that chunk from it. The limit is the
whole process's: what importing crispyx added (crispyx and the libraries it
imports, 0.3-0.4 GB) is taken off it, measured once so that a budget does not
move with the session. Data you load yourself is not counted: leave room for
it. Sizes that identify a result (PCA's method and chunk, a
resumable run's chunks) take the limit as given. Some terms still
grow with the screen. The table says, for each streaming function, what its
peak grows with beyond the chunk, so a run can be sized from the file's shape.

Notation: n cells, g genes, c cells (or genes) per chunk, P perturbations,
B batches or strata, k principal components.

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Function
     - Peak beyond the chunk
     - Sized from ``memory_limit_gb``
   * - ``qc_summary``
     - Reads the whole file into memory when its estimate is below
       ``min(0.6 × budget, 50 GB)``; streams otherwise. On CSC, chunks are
       n × c.
     - In-memory switch, chunk size
   * - ``filter_cells`` / ``filter_genes``, ``downsample_counts``
     - O(n) per-cell arrays. A CSC file is re-read whole for every row
       chunk; convert it to CSR first.
     - No
   * - ``normalize_total_log1p``, ``collapse_genes``
     - O(n) per-cell arrays.
     - No (a CSC conversion uses half the detected memory)
   * - ``highly_variable_genes``
     - Two g-length accumulators; on CSC, chunks are n × c.
     - No
   * - ``sort_by_perturbation``, ``convert_to_csc`` / ``convert_to_csr``
     - Output bands of at most half the budget; O(n) indices.
     - Band size
   * - ``read_csv`` / ``read_text``
     - One parsed block; O(n) row names.
     - Block size
   * - ``read_mtx`` / ``read_10x_mtx``
     - One parsed block of entries, then one band of cells, held twice as
       12-byte records; O(n + g) names. The non-zeros are also written once
       to scratch files on disk.
     - Block size, band size
   * - ``scrublet``
     - One sample's cells, run through :func:`scanpy.pp.scrublet` in memory,
       which copies them and scales a dense matrix for PCA: Scanpy's peak for
       that sample, whatever the number of samples. About 0.5 GB + 60 bytes
       per non-zero + 40 bytes per entry of the selected genes for the cells
       and their simulated doublets (fitted on 2,000-20,000 cells, within
       25%: 4.5 GB at 10,000 cells and 1,900 selected genes).
     - The check: a sample over the budget raises ``MemoryError`` before
       Scanpy runs
   * - ``score_genes``, ``score_genes_cell_cycle``
     - O(n + g) arrays.
     - No
   * - ``infercnv``
     - A chunk of cells dense over the placed genes (about 32 bytes per
       cell and gene); a sparse genes x windows weight matrix. On disk, the
       profiles before the threshold (4 bytes per cell and window).
     - Chunk size
   * - ``pl.dotplot``, ``pl.violin``
     - The plotted genes for every cell, sparse; the ``obs`` columns drawn.
     - No
   * - ``cnv_summary``
     - 2,048 cells of the profiles at a time; groups x windows sums.
     - No
   * - ``pl.chromosome_heatmap``
     - The plotted cells x windows, dense (Scanpy's heatmap).
     - ``max_cells``, ``batch``
   * - ``aggregate_pseudobulk``
     - P × g sums in a memory-mapped file on disk.
     - Chunk size
   * - ``compute_normalized_effects``
     - Up to five P × g float64 result arrays in memory, written at the end.
     - Chunk size, pair sums in memory or on disk
   * - ``t_test``
     - P × g sums, sums of squares and counts (20 bytes per entry:
       12.6 GB at 18,000 perturbations × 35,000 genes).
     - Cell chunk size
   * - ``wilcoxon_test``
     - A dense n × c block of genes and two copies; c is at least 32, so at
       least 32 × n × 12 bytes whatever the budget.
     - Gene chunk width
   * - ``nb_glm_test``
     - Without covariates: a summary of the controls plus the largest
       perturbation's cells. With covariates, apeGLM or
       ``targets_per_fit > 1``: the control cells, sparse, and a dense
       block of at least 64 genes per fit.
     - Worker count, gene block
   * - ``nb_offset_test``
     - Without covariates: B × g baselines. With covariates: the control
       cells, sparse, and an n × covariates design.
     - Gene and cell blocks
   * - ``residual_effects``
     - Three P × g float64 arrays; raises ``MemoryError`` up front when they
       do not fit.
     - Chunk size, the check
   * - ``shrink_lfc``
     - ``method="stats"``: P × g estimates, cached when they fit in a
       quarter of the budget. ``method="full"``: every control cell, dense
       (controls × g).
     - ``"stats"``: the cache; ``"full"``: worker count only
   * - ``batch_process``
     - One reducer state per (group, batch) pair for a c-gene chunk.
     - Gene chunk width (pair states not counted)
   * - ``pca``
     - ``method="covariance"``: one g × g float64 matrix, ``8 g²`` bytes
       (32 MB at 2,000 genes, 8.7 GB at 33,000). ``"incremental"``: none.
       The scores go to the file a chunk at a time on a crispyx backed file;
       for any other object they are held in memory (``4 n k`` bytes). With
       a gene subset (HVGs), about three full-width input rows per cell of
       the chunk. A CSR file's row index, 8 bytes per cell. A CSC file is
       re-read whole for every row chunk; convert it to CSR first.
     - Method and chunk size; ``"covariance"`` raises ``MemoryError`` when
       its peak at the smallest chunk cannot fit
   * - ``neighbors``, ``umap``
     - The embedding and the k-NN graph, in memory: about 6-8 KB per cell
       for ``neighbors`` and 6 KB for ``umap`` (50 components, 15
       neighbors; measured).
     - No
   * - Any function with ``copy=True``
     - The whole file, as Scanpy's ``copy`` does.
     - No

PCA's peak, measured on a Linux server (synthetic log-normalised data,
10% non-zero, peak resident memory above the process's size before the call,
16 BLAS threads), against the formula ``pca`` sizes itself with
(``8 g² + 16 c g`` bytes for covariance, ``50 (c + k) g + 32 (c + k)²`` for
incremental, each plus 64 MB of fixed overhead; k = 50):

.. list-table::
   :header-rows: 1

   * - Genes
     - Method
     - Chunk
     - 50,000 cells
     - 200,000 cells
     - Formula
   * - 2,000
     - covariance
     - 512
     - 91 MB
     - 96 MB
     - 112 MB
   * - 2,000
     - covariance
     - 2,048
     - 146 MB
     - 157 MB
     - 162 MB
   * - 2,000
     - incremental
     - 512
     - 90 MB
     - 105 MB
     - 130 MB
   * - 2,000
     - incremental
     - 2,048
     - 403 MB
     - 439 MB
     - 415 MB
   * - 5,000
     - covariance
     - 512
     - 289 MB
     - 298 MB
     - 305 MB
   * - 5,000
     - covariance
     - 2,048
     - 372 MB
     - 395 MB
     - 428 MB
   * - 5,000
     - incremental
     - 512
     - 197 MB
     - 215 MB
     - 215 MB
   * - 5,000
     - incremental
     - 2,048
     - 664 MB
     - 676 MB
     - 729 MB
   * - 10,000
     - covariance
     - 512
     - 891 MB
     - 921 MB
     - 946 MB
   * - 10,000
     - covariance
     - 2,048
     - 969 MB
     - 1,053 MB
     - 1,192 MB
   * - 10,000
     - incremental
     - 512
     - 323 MB
     - 343 MB
     - 355 MB
   * - 10,000
     - incremental
     - 2,048
     - 1,165 MB
     - 1,197 MB
     - 1,254 MB

The peak does not grow with the cells: 50,000 and 200,000 cells agree within
a few percent. The formula is at most 6% below a measurement and up to 45%
above it, most at 2,000 genes with 512-cell chunks, where the fixed overhead
dominates. These runs use every gene; with HVGs, the full-width rows read
before the subset add the term above, which predicts 0.69 GB for the
2-million-cell screen in :ref:`the QC page <quality-control-pca>`
(measured 0.71 GB).
``neighbors`` measured 0.41 GB at 50,000 cells and 1.26 GB at 200,000;
``umap`` 0.27 GB and 1.17 GB.

The peaks above are what crispyx allocates. The C allocator keeps memory a
process frees for reuse, so resident memory can stay above them;
``scrublet`` and the MatrixMarket readers hand it back to the system after
each sample or band, since each allocates arrays of a new size. On macOS,
pandas' parser buffers (about 0.2 GB) stay resident until the system is
short of memory, so a run there can exceed a small limit. Peak resident
memory of the whole process converting 40 million non-zeros:

.. list-table::
   :header-rows: 1

   * - ``memory_limit_gb``
     - Linux server (``read_mtx``, 0.2.10)
     - Mac (``read_10x_mtx``)
   * - 1
     - 0.82 GB
     - 1.01-1.10 GB
   * - 1.5
     - --
     - 1.39-1.48 GB
   * - 2
     - --
     - 1.38-1.50 GB

.. _disk-space:

Disk space
----------

crispyx's memory savings come from streaming, but several operations trade
RAM for **disk**: the per-``(perturbation, batch)`` accumulator described
above, the result file of ``cx.tl.t_test`` / ``cx.tl.wilcoxon_test`` /
``cx.tl.nb_glm_test`` as it is filled, QC's chunk cache, and whole-file
CSR↔CSC conversions. The DE functions fill their result file beside the output
(in a hidden ``.<output name>.resume`` directory, moved to the output on
success) rather than in ``$TMPDIR``, so an interrupted run continues where it
stopped; their ``"output"`` estimate covers that file.
``cx.tl.batch_process`` writes results directly into its output file as each
gene chunk finishes, so its ``"output"`` estimate likewise accounts for the
full footprint (see ``resume`` below for what that also enables). Every other
output is written as a hidden partial file beside its final name and renamed
into place when complete, so an interrupted run leaves any previous output
untouched (:doc:`faq`). crispyx warns
automatically -- without
blocking the call -- if free space on the relevant filesystem looks tight or
the write is unusually large. There is no configurable disk budget analogous
to ``memory_limit_gb``: the check is a feasibility heads-up, not a resource
allocator, so it always reads the real filesystem via ``shutil.disk_usage``.

To check disk usage *before* committing to a run, call
``cx.estimate_disk_usage`` (also available as ``cx.tl.estimate_disk_usage``,
same function, for Scanpy-style namespace discovery) with the function you
intend to run, its input file, and the same keyword arguments you plan to
pass:

.. code-block:: python

   import crispyx as cx

   cx.estimate_disk_usage(
       "compute_normalized_effects", "screen.h5ad",
       perturbation_column="guide_target", batch_column="gem_group",
   )
   # {'tempdir': 4.1 GB required, 812.3 GB free at /tmp [OK],
   #  'output': 0.3 GB required, 812.3 GB free at /tmp [OK]}

The result is keyed by filesystem location: ``"tempdir"`` for disk-backed
intermediate accumulators (usually ``$TMPDIR``), ``"output"`` for the final
result file, and ``"scratch"`` for the temporary fast-axis copy that a
converting ``format_mismatch_policy`` writes *beside the output file* (pass
``output_path`` to have it assessed at that location). Under the default
``"auto"`` policy the ``"scratch"`` entry appears when the real call would
convert, which the estimate decides exactly as the call does -- including the
bounded 64 MB probe read that decision needs, the one case where the query
reads ``X`` at all. The measurement is cached per file, so the call that
follows reuses it rather than re-measuring a file the query just pulled into
the page cache. Pass ``chunk_size`` and ``memory_limit_gb`` here if you will
pass them to the call: they set how many chunks it streams, which is what the
convert-or-stream decision turns on. Point
``$TMPDIR`` at a larger volume if the default location is too small; crispyx
respects it since ``tempfile.*`` reads it automatically, with no code change
needed. Whole-file conversions (:func:`crispyx.convert_to_csc`,
:func:`crispyx.convert_to_csr`, and a converting ``format_mismatch_policy`` in
``wilcoxon_test`` or ``batch_process`` on a CSR source) temporarily need
roughly 2× the source file's size, since the source and destination coexist
until the caller deletes the source -- see :ref:`the CSC conversion note below
<csc-disk-note>`.

Combining files that share cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A screen delivered as several files often repeats the same control cells in
each. :func:`crispyx.pp.concat_h5ad` concatenates the files on disk and keeps
one copy of every shared cell:

.. code-block:: python

   cx.pp.concat_h5ad(
       ["pilot_a.h5ad", "pilot_b.h5ad"], output_path="pilot.h5ad", label="source",
   )

Cells are matched by ``obs_names``. A barcode in more than one file must be
the same cell there -- the same expression over the output's genes -- or the
call raises; ``obs`` values that differ between the copies (such as QC
metrics computed per file) are warned about and the first file's are kept.
Pass ``shared_cells="error"`` to refuse any overlap instead of dropping it.
The result keeps the genes every file has (in the first file's order), ``X``
and the layers present in every file, the union of the ``obs`` columns, and
the ``obsm`` entries every file has in the same form (array, sparse or
DataFrame); ``label`` records each cell's source file (its stem, or its path
if two files share a stem).
Cell-cell graphs (``obsp``), ``varm``/``varp`` and ``.raw`` describe one file
and are not carried over; ``verbose=1`` lists what was dropped.

Some pipelines list only the genes detected in each sample. ``join="outer"``
keeps every file's genes instead, as zeros in the files that lack them (in
the order first seen, each with the ``var`` values of the first file that has
it); it costs no extra pass, since zeros are not stored.

.. _compress-h5ad:

Compressing results
~~~~~~~~~~~~~~~~~~~

crispyx writes every ``.h5ad`` uncompressed, so the next step can stream it
at full speed. To shrink a file you are keeping -- for archiving, sharing, or
a data deposit -- re-encode it losslessly with :func:`crispyx.compress_h5ad`:

.. code-block:: python

   cx.compress_h5ad("screen_normalized.h5ad", "screen_normalized.gz.h5ad")
   cx.compress_h5ad(path, path, overwrite=True)  # in place, after verification

Every large numeric dataset (``X`` in any layout, layers, obsm/varm/obsp,
dataframe columns, ``uns`` arrays) is rewritten with HDF5's built-in gzip and
shuffle filters; groups, dtypes, shapes, sparse layout and attributes are
unchanged, and ``verify=True`` (the default) compares every byte before the
destination is put in place. Any HDF5 reader opens the result without plugins
(anndata/scanpy, R, ``h5ls``). The copy is independent of AnnData, so it
applies to any ``.h5ad``, including ones not written by crispyx. Chunks are
compressed on ``n_jobs`` threads (every core by default), and memory stays
bounded regardless of file size.

Typical ratios are ~0.15-0.3 for single-cell matrices (counts or
log-normalized) and ~0.55-0.75 for dense effect and DE matrices, whose
mantissas are close to random. crispyx streams a compressed input with its
chunks inflated on a thread pool, several times faster than h5py's own
single-threaded decoding, but still slower than an uncompressed read from
the page cache: keep the files you are about to stream repeatedly
uncompressed on fast local disks, and compress what you archive or ship. On
HDD or network storage the smaller file often reads faster.

.. _messaging-and-verbosity:

Messaging and verbosity
-----------------------

``verbose`` defaults to ``True`` (equivalent to ``1``) across crispyx, so a
first-time call already reports what it did without any extra argument.
Pass ``verbose=False`` (or ``0``) to silence it completely. A handful of
differential-expression functions (``cx.tl.t_test``, ``cx.tl.nb_glm_test``,
``cx.tl.wilcoxon_test``) additionally accept ``verbose=2`` for a per-
perturbation progress line; every other function currently stops at level 1.

crispyx separates feedback into three channels with different visibility
rules:

- **Prints** (``[cx] {name}: {message}``) -- gated on ``verbose``, this is
  what most calls produce: what file is being read, what was inferred
  (control label, an auto-picked chunk size, the QC in-memory/streaming
  strategy), how many rows survived filtering, and what was written.
- **Warnings** (``warnings.warn``, prefixed ``"{context}: "``) -- always
  visible regardless of ``verbose``, since they flag something worth a
  second look: low disk space, an unusually large write, filtering that
  dropped most of the data, missing grouping/batch values, or groups with
  no usable statistics. Silence them the standard Python way
  (``warnings.filterwarnings``) if needed, not via ``verbose``.
- **Logger messages** (``logger.debug``/``logger.info``) -- invisible
  unless you configure logging yourself; this is where lower-level,
  developer-oriented detail goes (e.g. the exact QC strategy heuristic
  inputs), duplicating what the ``verbose`` print already told you in
  plainer language.

A run with the defaults touches most of this in one pass, for example
:func:`crispyx.pb.aggregate`:

.. code-block:: text

   [cx] pb.aggregate: chunk_size=4096 (auto)
   [cx] pb.aggregate accumulator: estimated disk usage: 0.8 GB (812.3 GB free)
   [cx] pb.aggregate: estimated disk usage: 0.8 GB (812.3 GB free)
   [cx] pb.aggregate: 18 perturbations × 2000 genes
   [cx] pb.aggregate: Saving → screen_pseudobulk_mean_log1p.h5ad

Coverage added beyond the basic Reading/Saving/Done triad:

.. list-table::
   :header-rows: 1

   * - Feedback
     - Where
   * - Filtering counts + aggressive-filter warning (kept/total, warns
       below 50% retained)
     - :func:`crispyx.pp.filter_cells`, :func:`crispyx.pp.filter_genes`,
       :func:`crispyx.pp.filter_perturbations`, :func:`crispyx.pp.qc_summary`
   * - Progress bars (``tqdm``, degrade to a no-op without it)
     - CSC/CSR conversion, ``cx.pb.aggregate``, ``cx.tl.batch_process``, QC
       streaming passes, plus the pre-existing DE loops
   * - Auto-picked chunk size / strategy
     - Every function that calls ``calculate_optimal_chunk_size`` (and
       siblings) internally
   * - Disk-usage confirmation alongside the unconditional warning
     - Every ``warn_if_disk_space_low`` call site with a ``verbose``
       parameter in scope

.. _auto-reload:

Auto-reload and re-run control
------------------------------

The DE functions, ``batch_process`` and the pseudo-bulk functions write their
results to an ``.h5ad`` file. A later call that would compute the same result
**loads and returns the saved file instead of rerunning the analysis**, making
notebook and script reruns instant:

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
   # [cx] Loading existing result: data/crispyx_wilcoxon.h5ad
   # [cx] Pass force=True to rerun the analysis.

Each output records in ``uns['crispyx']`` the input file (path, size and
modification time) and every argument that shapes the result. The file is
reused only when these match the call; otherwise it is recomputed and the
notice says why. Arguments that only change how a run is carried out
(``chunk_size``, ``memory_limit_gb``, ``n_jobs``, ``verbose``, ...) do not
invalidate it. Pass ``force=True`` to recompute regardless.

Resume and checkpointing
------------------------

``cx.tl.t_test``, ``cx.tl.wilcoxon_test``, ``cx.tl.nb_glm_test`` and
``cx.tl.nb_offset_test`` resume an interrupted run by default: rerun the same call and the perturbations (or gene
chunks) it had finished are skipped. A checkpoint is used only when it comes
from the same crispyx version, input file and result-affecting arguments;
``n_jobs``, ``memory_limit_gb`` and other operational arguments may change, so a
run killed for memory can restart with a lower limit or more workers. Any other
checkpoint is discarded with a warning. ``resume=False`` or ``force=True``
starts over.

.. code-block:: python

   result = cx.tl.nb_glm_test(
       "screen.h5ad",
       perturbation_column="perturbation",
       checkpoint_interval=10,  # save progress every 10 perturbations
   )

The checkpoint file ``<output>.progress.json`` is written atomically to prevent
corruption.

``cx.tl.batch_process`` takes the same ``resume``/``checkpoint_interval``
parameters, with ``resume=False`` by default. Its unit of resumable progress is a *gene chunk* rather than a
perturbation, since a chunk's result depends on every group and batch having
been swept once, not on one group alone; if the checkpoint file is itself
missing or corrupted, it falls back to scanning the (already partially
written) output file to detect the last completed chunk.
