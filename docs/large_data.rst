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
read in row-chunks (axis=0). For large datasets, the streaming control
statistics function automatically activates — fitting the intercept-only
model in chunks of 4,096 control cells instead of densifying the full
control matrix. This keeps peak memory at O(chunk_size × n_genes) rather
than O(n_control × n_genes). When ``freeze_control`` is auto-enabled,
streaming is used unconditionally. DESeq2-style size factors
(``size_factor_method="deseq2"``) also stream automatically when the
intermediate counts array would exceed 4 GB, computing geometric means
and per-cell median ratios in chunks. After each streaming phase,
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

.. _disk-space:

Disk space
----------

crispyx's memory savings come from streaming, but several operations trade
RAM for **disk**: the per-``(perturbation, batch)`` accumulator described
above, the result file of ``cx.tl.t_test`` / ``cx.tl.wilcoxon_test`` /
``cx.tl.nb_glm_test`` as it is filled, QC's chunk cache, and whole-file
CSR↔CSC conversions. The DE functions fill their result file beside the output
(in a hidden ``.<output name>.resume`` directory, moved to the output on
success) rather than in ``$TMPDIR``, so an interrupted run can continue with
``resume=True``; their ``"output"`` estimate covers that file.
``cx.tl.batch_process`` writes results directly into its output file as each
gene chunk finishes, so its ``"output"`` estimate likewise accounts for the
full footprint (see ``resume`` below for what that also enables). crispyx warns
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
the same cell there -- the same expression over the shared genes -- or the
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

Long-running differential expression analyses can be resumed after interruption
using the ``resume`` and ``checkpoint_interval`` parameters:

.. code-block:: python

   # Enable checkpointing during long runs
   result = cx.nb_glm_test(
       adata_ro,
       perturbation_column="perturbation",
       resume=True,
       checkpoint_interval=10,  # Save progress every 10 perturbations
   )

If interrupted, simply re-run the same command - completed perturbations will
be skipped automatically. The checkpoint file ``<output>.progress.json`` is
written atomically to prevent corruption.

``cx.tl.batch_process`` supports the same ``resume``/``checkpoint_interval``
parameters. Its unit of resumable progress is a *gene chunk* rather than a
perturbation, since a chunk's result depends on every group and batch having
been swept once, not on one group alone; if the checkpoint file is itself
missing or corrupted, it falls back to scanning the (already partially
written) output file to detect the last completed chunk.
