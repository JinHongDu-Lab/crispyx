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

* Converting to CSC before Wilcoxon (see :doc:`usage`).
* Converting to CSR before NB-GLM.
* Using ``freeze_control=True`` for datasets with >100K control cells.

When should I use CSC vs CSR format?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **CSC** (Compressed Sparse Column): Use for Wilcoxon rank-sum tests, which
  iterate over gene columns. Convert with :func:`crispyx.convert_to_csc`.
* **CSR** (Compressed Sparse Row): Use for NB-GLM, size factors, and
  operations that iterate over cells (rows). Convert with
  :func:`crispyx.convert_to_csr`.

The benchmark pipeline handles this automatically, but manual workflows
should convert before calling DE functions.

QC or normalisation is extremely slow on a CSC file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cell-(row-)streaming operations — quality control and
:func:`crispyx.normalize_total_log1p` — read the matrix one block of cells at a
time. On a **CSC** file a row slice must scan the column pointers across every
gene, making each chunk ``O(total_nnz)`` and the whole pass up to ~100x slower
than the equivalent CSR streaming. Gene-(column-)streaming operations such as
the Wilcoxon test are naturally fast on CSC and slow on CSR — the penalty is
symmetric.

crispyx mitigates this for you:

* **Quality control** automatically dispatches CSC inputs to a
  column-oriented path (including the masks-only ``output_dir=None`` call), so
  no action is needed.
* :func:`crispyx.normalize_total_log1p` exposes ``format_mismatch_policy``:

  .. code-block:: python

     # Default: proceed but log one actionable warning.
     cx.pp.normalize_total_log1p(csc_path, out, format_mismatch_policy="warn")

     # Transparently stream via a temporary CSR copy (bounded memory);
     # the temp file is removed before returning. This temporarily needs
     # ~2x the source file's size in free disk space (source + temp copy
     # coexist); crispyx warns automatically if that looks tight.
     cx.pp.normalize_total_log1p(csc_path, out, format_mismatch_policy="convert")

     # Proceed silently (you have already accounted for the cost).
     cx.pp.normalize_total_log1p(csc_path, out, format_mismatch_policy="off")

For a file you will reuse across several cell-streaming steps, convert it once
up front instead:

.. code-block:: python

   cx.data.convert_to_csr(csc_path, output_path=csr_path)  # bounded-memory, two-pass
   # Also needs ~2x the source file's size in free disk space during
   # conversion; check up front with cx.estimate_disk_usage("convert_to_csr", csc_path).

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

* **Pre-convert matrix formats** before DE: CSC for Wilcoxon, CSR for NB-GLM.
  This avoids O(total_nnz × n_chunks) scans.
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
