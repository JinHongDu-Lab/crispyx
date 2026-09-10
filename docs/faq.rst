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
   :func:`crispyx.batch_process` and :func:`crispyx.normalize_total_log1p`)
   -- see the next section for the exact options and defaults.

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
* :func:`crispyx.wilcoxon_test` and :func:`crispyx.batch_process` default to
  ``format_mismatch_policy="convert"``: a CSR source is converted once to a
  temporary CSC copy **beside the output file** (bounded memory, honouring
  ``memory_limit_gb``), streamed from there, and the copy is removed before
  returning. This temporarily needs ~2x the source file's size in free disk
  space at the output location; if that is not available the call warns and
  streams from the CSR source instead (the ``"warn"`` behaviour) rather than
  failing midway, and :func:`crispyx.estimate_disk_usage` reports the need
  under ``"scratch"`` (pass the same ``output_path``/``output_dir`` you will
  pass to the real call).
* :func:`crispyx.normalize_total_log1p` (the opposite mismatch, a CSC
  source) defaults to ``"warn"`` because crispyx writers never produce CSC.

  .. code-block:: python

     # wilcoxon_test / batch_process on a CSR file: converted for you.
     cx.de.wilcoxon_test(csr_path, ..., memory_limit_gb=128)      # "convert" is the default

     # Proceed on the mismatched file after one UserWarning that quantifies
     # the cost (matrix size x number of chunks) ...
     cx.tl.batch_process(csr_path, reducer, format_mismatch_policy="warn", ...)

     # ... or silently (you have already accounted for the cost).
     cx.tl.batch_process(csr_path, reducer, format_mismatch_policy="off", ...)

     # normalize_total_log1p takes the same three values for a CSC source.
     cx.pp.normalize_total_log1p(csc_path, out, format_mismatch_policy="convert")

For a file you will reuse across several steps that want the same format --
e.g. Wilcoxon DE *and* ``batch_process`` on the same screen -- convert it once
up front instead, so the conversion is not repeated per call:

.. code-block:: python

   cx.pp.convert_to_csc(csr_path, output_path=csc_path, memory_limit_gb=128)
   # Peak memory is bounded by memory_limit_gb (larger matrices are converted
   # in several bands, one extra pass over the source each). Also needs ~2x
   # the source file's size in free disk space during conversion; check up
   # front with cx.estimate_disk_usage("convert_to_csc", csr_path).

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
