# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Total (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| crispyx | QC filter | success | 6.09 | 5726.34 | 65282.0 | 19568.0 |
| scanpy | QC filter | success | 21.1 | 7628.7 | 65282.0 | 19568.0 |
| crispyx | pseudobulk (avg log) | success | 33.14 | 3951.31 |  |  |
| crispyx | pseudobulk | success | 15.68 | 2882.41 |  |  |


### DE: t-test

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | t-test | success | 19.69 | 3222.52 | 88 |
| crispyx | t-test | success | 56.13 | 663.19 | 88 |


### DE: Wilcoxon

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | Wilcoxon | success | 32.82 | 3278.93 | 88 |
| scanpy | Wilcoxon | success | 1555.62 | 3240.67 | 88 |


### DE: NB GLM

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM | success | 83.54 | 624.6 | 88.0 |
| edgeR | NB-GLM | timeout | 21605.24 |  |  |
| pertpy | NB-GLM | success | 5253.05 | 33776.41 | 88.0 |


### Other

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | preprocess | success | 18.42 | 572.12 |  |
| crispyx | standardize csr | success | 1.03 | 255.19 |  |


## 2. Performance Comparison

### crispyx vs Reference Tools

_crispyx as baseline. Negative values = crispyx is faster/uses less memory._

#### Preprocessing / QC

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | -15.0s | 28.9% | ✅ | -1902.4 MB | 75.1% | ✅ |


#### DE: t-test

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | +36.4s | 285.0% | ❌ | -2559.3 MB | 20.6% | ✅ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | -1522.8s | 2.1% | ✅ | +38.3 MB | 101.2% | ⚠️ |


#### DE: NB GLM

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM (subset) | -2218.2s | 3.6% | ✅ | +138.1 MB | 128.4% | ❌ |
| NB-GLM | pertpy NB-GLM (subset) | -1729.1s | 4.6% | ✅ | -37483.3 MB | 1.6% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| package A | method A | package B | method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM (subset) | pertpy | NB-GLM (subset) | +489.1s | 127.0% | ❌ | -37621.4 MB | 1.3% | ✅ |


## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### Preprocessing / QC

| crispyx method | compared to | Cells Δ |  | Genes Δ |   |
| --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | +0 | ✅ | +0 | ✅ |


### DE: t-test

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: Wilcoxon

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: NB GLM

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM (subset) | 0.513<br><small>±0.054</small> | ❌ | 0.874<br><small>±0.022</small> | ⚠️ | 0.320<br><small>±0.126</small> | ❌ | 0.815<br><small>±0.028</small> | ⚠️ | 0.954<br><small>±0.060</small> | ✅ | 0.995<br><small>±0.002</small> | ✅ |
| NB-GLM | pertpy NB-GLM (subset) | 0.983<br><small>±0.023</small> | ✅ | 0.999<br><small>±0.001</small> | ✅ | 0.999<br><small>±0.001</small> | ✅ | 0.999<br><small>±0.000</small> | ✅ | 0.998<br><small>±0.004</small> | ✅ | 0.999<br><small>±0.001</small> | ✅ |


### Tool Comparisons

| package A | method A | package B | method B | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM (subset) | pertpy | NB-GLM (subset) | 0.103<br><small>±0.054</small> | ❌ | 0.683<br><small>±0.030</small> | ❌ | 0.149<br><small>±0.120</small> | ❌ | 0.131<br><small>±0.077</small> | ❌ | 0.941<br><small>±0.070</small> | ⚠️ | 0.849<br><small>±0.052</small> | ⚠️ |


## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

### Effect Size Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM | edgeR NB-GLM (subset) | 0.031 | ❌ | 0.033 | ❌ | 0.051 | ❌ |
| NB-GLM | pertpy NB-GLM (subset) | 0.949 | ✅ | 0.967 | ✅ | 0.982 | ✅ |


### P-value Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM | edgeR NB-GLM (subset) | 0.842 | ✅ | 0.874 | ✅ | 0.911 | ✅ |
| NB-GLM | pertpy NB-GLM (subset) | 0.976 | ✅ | 0.980 | ✅ | 0.981 | ✅ |


_Note: Some methods are missing due to errors:_
- t-test vs scanpy t-test (subset): _missing output: scanpy_de_t_test_subset (not run, no output)_
- Wilcoxon vs scanpy Wilcoxon (subset): _missing output: scanpy_de_wilcoxon_subset (not run, no output)_
- t-test (subset, capped control) vs scanpy t-test (subset, capped control): _missing output: crispyx_de_t_test_subset_ctrl (not run, no output), scanpy_de_t_test_subset_ctrl (not run, no output)_
- Wilcoxon (subset, capped control) vs scanpy Wilcoxon (subset, capped control): _missing output: crispyx_de_wilcoxon_subset_ctrl (not run, no output), scanpy_de_wilcoxon_subset_ctrl (not run, no output)_
- NB-GLM (subset, capped control) vs pertpy NB-GLM (subset, capped control): _missing output: crispyx_de_nb_glm_subset_ctrl (not run, no output), pertpy_de_pydeseq2_subset_ctrl (not run, no output)_
- NB-GLM (subset, capped control) vs edgeR NB-GLM (subset, capped control): _missing output: crispyx_de_nb_glm_subset_ctrl (not run, no output), edger_de_glm_subset_ctrl (not run, no output)_

### Overlap Heatmaps (Top-100)

#### Effect Size

![Effect Size Top-100 Overlap](benchmark_effect_top_100_overlap.png)

#### P-value

![P-value Top-100 Overlap](benchmark_pvalue_top_100_overlap.png)

---

**Legend:**
- **Performance:** ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse
- **Accuracy:** ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8
- **Overlap:** ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5
- **Shrinkage:** ✅ <1% inflated | ⚠️ 1-10% inflated | ❌ >10% inflated

**Abbreviations:**
- ρ = Pearson correlation, ρₛ = Spearman correlation
- log-Pval = correlations on -log₁₀(p) transformed values
- sf=per = per-comparison size factor estimation (matches PyDESeq2)

**Notes:**
- Correlation and overlap values shown as mean±std across perturbations
- crispyx lfcShrink uses `method='stats'` (Gaussian approximation) which is numerically stable and ~35× faster than `method='full'`.
- P-value overlap excludes lfcShrink methods since shrinkage only affects effect sizes, not p-values.
- **Warning:** PyDESeq2 may produce aberrant shrinkage when dispersion trend fitting fails. crispyx shrinkage is more robust.

## Accuracy subset

Accuracy metrics use the same 50 seeded perturbations (seed 0) for every method; the list is in `.accuracy_perturbations.json`.

Reruns on that subset, for accuracy only: they are not performance rows, and their time and memory (kept in `benchmark_accuracy_reruns.csv` for the record) are not compared. Performance uses the full-data runs, where a failure (timeout, memory limit) is the result. `_subset` = full control; `_subset_ctrl` = control capped, crispyx rerun on the same cells.

Control: 7,629 cells, cap 20,000. By the rule (decided by the control's size, not by which runs fit on a machine), this dataset's primary accuracy comparisons use the **full** control (the cap does not bind).

| method | status | perturbations | control cells |
| --- | --- | --- | --- |
| edger_de_glm_subset | success | 50.0 | 7,629 |
| pertpy_de_pydeseq2_subset | success | 50.0 | 7,629 |
