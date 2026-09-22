FAQ & Troubleshooting
=====================

Common issues
-------------

``MemoryError`` or out-of-memory kills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``memory_limit_gb`` to match your available RAM:

.. code-block:: python

   result = cx.tl.rank_genes_groups(
       adata,
       perturbation_column="perturbation",
       method="nb_glm",
       memory_limit_gb=32,  # set to your SLURM --mem value
   )

For very large datasets, consider:

* Converting to CSC before Wilcoxon or ``batch_process`` (see :doc:`usage`).
* Converting to CSR before NB-GLM.
* Using ``freeze_control=True`` for datasets with >100K control cells.

When should I use CSC vs CSR format, and how do I control the streaming order?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There's no separate "streaming order" setting to pick independently -- each
function reads the matrix along one fixed axis (cells or genes), determined
by what it computes, and that axis is cheap only on the matching sparse
format. The choice that actually matters is **which format the file is
stored in**:

* **CSC** (Compressed Sparse Column) -- fast for gene-(column-)major
  functions: :func:`crispyx.wilcoxon_test` and :func:`crispyx.batch_process`
  (including custom ``BatchReducer`` callbacks). Convert with
  :func:`crispyx.convert_to_csc`.
* **CSR** (Compressed Sparse Row) -- fast for cell-(row-)major functions:
  :func:`crispyx.t_test`, :func:`crispyx.nb_glm_test`, size factors, quality
  control, and :func:`crispyx.normalize_total_log1p`. Convert with
  :func:`crispyx.convert_to_csr`.

Which axis a function needs follows directly from what it has to hold in
memory at once. ``batch_process`` keeps one accumulator per
``(group, batch)`` pair, finalized only once every cell for that pair has
been seen; holding those accumulators for every gene simultaneously would
need ``O(n_groups × n_batches × n_genes)`` memory, unbounded for a
genome-wide screen with many perturbations. So it chunks by **genes**
instead, bounding memory to ``O(n_groups × n_batches × chunk_size)`` --
which means its actual disk access, repeated once per chunk, is "these few
columns, across every row." That's cheap on CSC (column ranges are
contiguous in the underlying arrays) and ``O(total_nnz)`` on CSR (every
row's full nonzero list has to be scanned and filtered, regardless of how
narrow the column range is) -- exactly the ~100x penalty
``wilcoxon_test`` already has for the identical reason (it also chunks by
gene, ranking cells within each gene across groups). ``t_test``/
``nb_glm_test`` instead chunk by **cells** to accumulate simple per-gene
running sums, so their access is row-slices -- cheap on CSR, and why they
prefer the opposite format.

Running a function against the wrong format still produces correct results,
just far slower (see below). Two ways to fix that:

1. **Convert the file once**, up front, if it will be reused across several
   steps that want the same format.
2. **Let ``format_mismatch_policy`` handle it** for a one-off call, on the
   functions that support it (:func:`crispyx.wilcoxon_test`,
   :func:`crispyx.batch_process` and :func:`crispyx.normalize_total_log1p`).
   Its default, ``"auto"``, measures the source and converts only when that
   is cheaper than streaming off the fast axis -- see the next section.

QC, normalisation, DE, or batch_process is extremely slow on a mismatched file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cell-(row-)streaming operations — quality control and
:func:`crispyx.normalize_total_log1p` — read the matrix one block of cells at a
time. On a **CSC** file a row slice must scan the column pointers across every
gene, making each chunk ``O(total_nnz)`` and the whole pass up to ~100x slower
than the equivalent CSR streaming. Gene-(column-)streaming operations —
:func:`crispyx.wilcoxon_test` and :func:`crispyx.batch_process` — are
naturally fast on CSC and just as slow on a **CSR** file, and CSR is what
every crispyx writer produces. The mechanism there is worse than a scan:
anndata serves a column slice of a backed CSR matrix by reading the *entire*
``data``/``indices`` arrays into memory and filtering them, so every gene
chunk re-reads the whole file and transiently needs the matrix's full size in
RAM. For a multi-million-cell screen on a network filesystem that is tens of
minutes per gene chunk, times ~40 chunks.

crispyx mitigates this for you:

* **Quality control** automatically dispatches CSC inputs to a
  column-oriented path (including the masks-only ``output_dir=None`` call), so
  no action is needed.
* :func:`crispyx.wilcoxon_test`, :func:`crispyx.batch_process` and
  :func:`crispyx.normalize_total_log1p` all default to
  ``format_mismatch_policy="auto"``, which **measures** the source before
  deciding. Converting costs one full read plus one full write; streaming off
  the fast axis costs one full read *per chunk*. Which is cheaper is a
  property of the filesystem, not of the data: a 500 MB file in page cache
  re-reads in ~0.1 s, so even 47 chunks beat the conversion's write, while a
  40 GB screen on Lustre costs minutes per chunk and the conversion repays
  itself almost immediately. ``"auto"`` times a bounded (64 MB) prefix read of
  the source, projects ``(n_chunks - 1) x`` the resulting full-read time, and
  converts only when that projection exceeds 60 s **and** there are at least 4
  chunks (below that, one conversion cannot repay its own write however slow
  the filesystem). Runs with ``verbose>=1`` print the decision and the numbers
  behind it.
* When ``"auto"`` -- or an explicit ``"convert"`` -- does convert, the source
  is converted once to a temporary copy **beside the output file** (bounded
  memory, honouring ``memory_limit_gb``), streamed from there, and the copy is
  removed before returning. This temporarily needs ~2x the source file's size
  in free disk space at the output location; if that is not available the call
  warns and streams from the source instead (the ``"warn"`` behaviour) rather
  than failing midway, and :func:`crispyx.estimate_disk_usage` reports the need
  under ``"scratch"`` (pass the same ``output_path``/``output_dir``, and the
  same ``chunk_size``/``memory_limit_gb``, you will pass to the real call --
  under ``"auto"`` the ``"scratch"`` entry appears when the real call would
  convert, and the chunk count those arguments set is what that turns on).
* A run killed outright (``SIGKILL``, an out-of-memory kill, a scheduler
  timeout) cannot delete its temporary copy. The next call that reads a
  mismatched source from the same directory removes copies that have gone a
  day untouched and whose owning run is gone, so they do not accumulate; they
  are hidden files named ``.cx_<function>_<pid>-<host>_*`` if you want to
  clear them by hand sooner. The day-long grace period is what makes this
  safe when the output directory is shared by several nodes of a cluster job
  array, where a PID from another node cannot be checked -- a copy still
  being written is never a day old.
* **The copy lasts for one call only.** That is the right trade for a run
  that finishes in one go, and the wrong one for a run that does not: with
  ``resume=True``, every restart rebuilds the whole copy before any new work
  begins. A job under a scheduler walltime shorter than the computation --
  say 7.7 h of gene chunks under a 6 h limit -- is *guaranteed* to restart,
  so it pays the conversion at least twice, and each payment comes out of
  the next window. crispyx warns once when a resumable call is about to
  convert. Convert once yourself instead, and point every step at the
  result (see below); several steps over the same file want this anyway.

.. code-block:: python

   # "auto" is the default: converted when it pays off, streamed when not.
   cx.de.wilcoxon_test(csr_path, ..., memory_limit_gb=128)

   # Always convert, whatever the measurement says.
   cx.tl.batch_process(csr_path, reducer, format_mismatch_policy="convert", ...)

   # Proceed on the mismatched file after one UserWarning that quantifies
   # the cost (matrix size x number of chunks) ...
   cx.tl.batch_process(csr_path, reducer, format_mismatch_policy="warn", ...)

   # ... or silently (you have already accounted for the cost).
   cx.tl.batch_process(csr_path, reducer, format_mismatch_policy="off", ...)

   # normalize_total_log1p takes the same four values for a CSC source.
   cx.pp.normalize_total_log1p(csc_path, out, format_mismatch_policy="convert")

Converting once, for resumable or multi-step work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the same file feeds more than one gene-streaming step -- Wilcoxon DE
*and* ``batch_process`` on the same screen, say -- or when a single step needs
``resume=True``, do the conversion yourself and keep it, so it is not repeated
per call or per restart:

.. code-block:: python

   import crispyx as cx

   half = "screen_half0.h5ad"                       # CSR, as crispyx writes
   csc = "screen_half0_csc.h5ad"
   cx.pp.convert_to_csc(half, output_path=csc, memory_limit_gb=160).close()
   # Peak memory is bounded by memory_limit_gb (larger matrices are converted
   # in several bands, one extra pass over the source each). Also needs ~2x
   # the source file's size in free disk space during conversion; check up
   # front with cx.estimate_disk_usage("convert_to_csc", half).

   # Both steps now stream their fast axis with no per-call copy at all,
   # and a resumed run starts on the first unfinished chunk.
   cx.de.wilcoxon_test(csc, ..., batch_column="batch", memory_limit_gb=160)
   cx.tl.batch_process(csc, reducer, ..., resume=True, memory_limit_gb=160)

The cost is one persistent file of roughly the source's size, against one
conversion per call per restart. On a scheduler, also budget the walltime for
the *whole* computation rather than relying on resume to make up the
difference.

``tomllib`` / ``tomli`` import errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If building docs on Python 3.10, install the backport:

.. code-block:: bash

   pip install tomli

Python 3.11+ includes ``tomllib`` in the standard library.

Control label not detected
~~~~~~~~~~~~~~~~~~~~~~~~~~

crispyx auto-detects control labels (``ctrl``, ``NTC``, ``scramble``, etc.).
If your dataset uses a non-standard label, pass it explicitly:

.. code-block:: python

   adata = cx.pp.qc_summary(
       adata,
       perturbation_column="perturbation",
       control_label="my_control_name",
   )

Or use :func:`crispyx.normalise_perturbation_labels` to canonicalise labels
before analysis.

``UserWarning: CSC storage detected`` during NB-GLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB-GLM requires CSR format. Convert first:

.. code-block:: python

   adata_csr = cx.pp.convert_to_csr(adata, output_dir="results/")
   result = cx.nb_glm_test(adata_csr, perturbation_column="perturbation")

Top hits by ``logfoldchanges`` are genes expressed in a handful of cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``t_test`` and ``wilcoxon_test`` reproduce Scanpy's fold change, which keeps
the ratio finite with a constant::

   log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9))

When a gene has no expressing cells in one arm, that arm's term *is* the
constant, so the ratio becomes ``expm1(mean) / 1e-9``: the magnitude reported
says how small the constant is, not how large the effect is. Its ceiling is
``log2(1 / 1e-9) = 29.9``.

The scale of it: split 500 real control cells at random into two arms of 250,
so that every difference is noise, and roughly 119 of 11,630 genes report
``|log2FC| > 2``, the largest being 24.9.

**This affects the effect-size column only.** On those same null runs the
rejection rate is 4.1-4.9% against a nominal 5% at every level of sparsity,
and no gene comes close to significance -- the smallest ``padj`` is 0.73. The
p-values are sound; it is ranking or thresholding on ``logfoldchanges`` that
promotes these genes above every real effect.

``pts`` and ``pts_rest`` -- the fractions of expressing cells in the perturbed
and control arms -- separate the two cases that look identical in the fold
change. A gene with ``pts = 0.00`` and ``pts_rest = 0.95`` is a real complete
knockdown; one with ``pts = 0.00`` and ``pts_rest = 0.02`` was never
detectable in the first place. To rank on magnitudes that the data actually
determines, keep the genes both arms can see:

.. code-block:: python

   import numpy as np

   res = cx.wilcoxon_test(path, perturbation_column="perturbation")
   row = res.groups.index("TARGET1")
   measurable = (res.pts[row] > 0.05) & (res.pts_rest[row] > 0.05)
   lfc = np.where(measurable, res.logfoldchanges[row], np.nan)
   top = res.genes[np.argsort(-np.abs(lfc))]     # NaN sorts last

On one real Adamson perturbation this takes the largest reported ``|log2FC|``
from 23.6 to 3.0 and brings the targeted gene itself into the top five, at the
cost of setting aside 2,612 of 11,630 genes as not measurable in both arms.

Genes absent from one arm are excluded by that mask by construction, so look
for them in ``pts`` / ``pts_rest`` rather than at the top of the fold-change
ranking. Alternatively, raise ``min_pct_ctrl`` / ``min_pct_pert`` to drop
sparse genes from the analysis entirely, which also removes them from the
multiple-testing correction.

HPC / SLURM tips
~~~~~~~~~~~~~~~~~

* Set ``memory_limit_gb`` to your SLURM ``--mem`` allocation.
* Use ``resume=True`` and ``checkpoint_interval=10`` for long jobs that may
  be preempted.
* ``drop_file_cache()`` is called automatically to prevent cgroup-cached
  pages from counting toward memory limits.

My DE result is loaded instantly on the second call — is that expected?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes.  Since v0.0.3 all three DE functions auto-reload an existing result file
instead of rerunning the analysis.  When ``verbose=True`` a notice is printed:

.. code-block:: text

   [crispyx] Loading existing result: data/crispyx_wilcoxon.h5ad
   [crispyx] Pass force=True to rerun the analysis.

If you changed a parameter (e.g. ``min_pct_ctrl``, ``min_pct_pert``, a covariate list, or
``dispersion_scope``) and want the result to reflect the new settings, pass
``force=True`` to the DE function.  The existing output file will be
overwritten.

Can I pickle / serialise a ``RankGenesGroupsResult``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, since v0.0.3.  The ``RecursionError`` that occurred when calling
``pickle.dumps`` on a result object is fixed.  The on-disk HDF5 handle is
excluded from the pickle payload and reopened lazily after unpickling:

.. code-block:: python

   import pickle
   result = cx.wilcoxon_test("data.h5ad", perturbation_column="perturbation")

   data = pickle.dumps(result)        # no RecursionError
   restored = pickle.loads(data)      # works
   # restored.result is None — no open file handle after unpickling.
   # Access restored["KO1"].pvalue etc. normally.

Note that ``restored.result`` is ``None`` after unpickling.  If you need the
backed AnnData reference (e.g. to call ``result.result_path``), re-open it:

.. code-block:: python

   from crispyx.data import AnnData
   restored.result = AnnData(original_output_path)

Performance tips
----------------

* **Pre-convert matrix formats** before DE: CSC for Wilcoxon and
  ``batch_process``, CSR for NB-GLM. This avoids O(total_nnz × n_chunks)
  scans.
* **Use ``freeze_control=True``** for datasets with >100K control cells to
  reduce per-worker memory from ~32 GB to <1 GB.
* **Increase ``n_jobs``** for multi-core NB-GLM on machines with sufficient
  RAM.
* **Use adaptive chunk sizes** (the default): let crispyx calculate optimal
  chunk sizes based on your ``memory_limit_gb``.

Comparison questions
--------------------

When should I use crispyx instead of Scanpy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use crispyx when your dataset does not fit in RAM, when you are running on an
HPC system with a memory limit, or when you want a streaming on-disk pipeline.
crispyx produces results identical to Scanpy for t-test and Wilcoxon DE
(Pearson *r* > 0.9999). For datasets that fit in RAM and where you need
Scanpy's broader ecosystem, use Scanpy.

Can I use crispyx instead of Pertpy or PyDESeq2 for NB-GLM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. crispyx implements a negative binomial GLM that is approximately 2× faster
than Pertpy/PyDESeq2 and uses far less memory on genome-wide datasets. Results
agree with PyDESeq2 (Pearson *r* > 0.97 for LFC estimates). crispyx does not
implement the full PyDESeq2 feature set (custom design matrices, Cook's
outlier filtering, etc.). For large genome-wide screens where PyDESeq2 runs
out of memory, crispyx is currently the only practical Python option.

Does crispyx replace the full Pertpy workflow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. crispyx focuses on QC, normalization, pseudobulk, and differential
expression for CRISPR screens. Pertpy provides many additional perturbation
analysis methods (Augur, Mixscape, CINEMA-OT, etc.) that are outside the scope
of crispyx. For large screens, you can use crispyx for the memory-intensive
DE steps and Pertpy for downstream perturbation-specific analyses.

See :doc:`comparison` for a full side-by-side comparison.
