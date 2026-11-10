# Benchmark result summaries (2026-11-10)

Summary files of the benchmark runs, one folder per dataset (17 datasets). Full outputs (DE tables,
QC-filtered and preprocessed files) are not in git; they live on HKU HPC2021 and the lab server.

- Timing: exclusive AMD EPYC 7542 nodes (32 cores). crispyx 0.2.3 for QC, preprocessing, pseudobulk,
  t-test and Wilcoxon; crispyx 0.2.6 (PyPI) for NB-GLM, on the perturbation-sorted QC-filtered counts.
  Reference methods (Scanpy, PyDESeq2, edgeR) on the full data are the performance rows.
- Accuracy: every method on the same 50 seeded perturbations per dataset. Datasets with more than
  20,000 control cells are scored on 20,000 seeded controls for every method (`_subset_ctrl` runs);
  smaller controls are used in full. NB-GLM references share crispyx's library-size size factors and
  mean floor. Each accuracy row records its control (full / capped) and role (primary / supplementary).

Files per dataset: `benchmark_report.md` (report), `benchmark_performance.csv`, `benchmark_accuracy.csv`,
`benchmark_accuracy_reruns.csv`, overlap heatmaps, `results.md` / `results.csv` / `summary.json`
(run summary), `.accuracy_subset.json` and `.accuracy_perturbations.json` (the accuracy subset).
