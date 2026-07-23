Benchmarking
============

The benchmarking toolkit automates comparisons between the streaming
implementations in this project. Supply your own data (or generate a synthetic
demo dataset with ``python benchmarking/generate_demo_dataset.py``) and run the
benchmark suite against any compatible ``.h5ad`` file that exposes
``perturbation`` and ``gene_symbols`` columns.

Running benchmarks
------------------

.. code-block:: bash

   cd benchmarking
   ./run_benchmark.sh config/Adamson.yaml          # single dataset
   ./run_benchmark.sh config/*.yaml                 # all datasets

Outputs
-------

* Resource usage measurements stored in ``benchmarking/results/benchmark_results.csv``.
* A GitHub-friendly table at ``benchmarking/results/benchmark_results.md``.
* Intermediate AnnData files written directly to ``benchmarking/results`` (or
  any directory provided via ``--output-dir``).
* Comprehensive differential expression parity metrics: max absolute
  differences, Pearson/Spearman correlations, top-``k`` overlaps (``k=50`` by
  default), and AUROC values when ground-truth labels are present. These are
  captured in both the CSV and Markdown summaries so ranking agreement is easy
  to audit alongside effect size deviations.

The script accepts additional options to benchmark a subset of methods or to
redirect outputs to a different directory. Refer to ``benchmarking/README.md``
for further details.

Available benchmark methods
---------------------------

**crispyx**: ``crispyx_qc_filtered``, ``crispyx_preprocess``,
``crispyx_pb_avg_log``, ``crispyx_pb_pseudobulk``, ``crispyx_de_t_test``,
``crispyx_de_wilcoxon``, ``crispyx_de_nb_glm``

**Reference**: ``scanpy_qc_filtered``, ``scanpy_de_t_test``,
``scanpy_de_wilcoxon``, ``edger_de_glm``, ``pertpy_de_pydeseq2``

The ``crispyx_preprocess`` step normalizes the QC-filtered output with streaming
``normalize_total_log1p()`` and is a prerequisite for all t-test and Wilcoxon DE
methods (both crispyx and Scanpy). The execution order is enforced via
``depends_on``: **QC → preprocess → DE**.

Available NB-GLM benchmark methods
----------------------------------

The benchmark suite includes the following NB-GLM variants:

* ``crispyx_de_nb_glm``: Standard NB-GLM differential expression
* ``crispyx_de_nb_glm_shrunk``: NB-GLM with apeGLM LFC shrinkage (recommended)

The shrinkage variant applies adaptive Cauchy prior shrinkage to log-fold changes,
which improves accuracy by preserving large effects while shrinking uncertain
estimates toward zero.

Results summary
---------------

See `benchmarking/benchmark_summary.md <https://github.com/jinhongdu-lab/crispyx/blob/main/benchmarking/benchmark_summary.md>`_
for a human-readable summary of performance and accuracy results across all 12
benchmark datasets, including runtime tables, memory tables, and accuracy metrics.

For comparison of crispyx against Scanpy, Pertpy/PyDESeq2, and edgeR, see
:doc:`comparison`.
