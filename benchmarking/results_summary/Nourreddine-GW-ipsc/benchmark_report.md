# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Total (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| crispyx | QC filter | success | 2526.77 | 2816.08 | 2641058.0 | 36246.0 |
| scanpy | QC filter | memory_limit | 138.47 | 121051.65 |  |  |
| crispyx | pseudobulk (avg log) | success | 2534.98 | 24246.66 |  |  |
| crispyx | pseudobulk | success | 1298.47 | 13242.61 |  |  |


### DE: t-test

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | t-test | memory_limit | 1601.09 | 125309.94 |  |
| crispyx | t-test | success | 1766.01 | 8574.36 | 10922.0 |


### DE: Wilcoxon

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | Wilcoxon | success | 4332.24 | 61061.82 | 10922.0 |
| scanpy | Wilcoxon | memory_limit | 2129.34 | 125310.95 |  |


### DE: NB GLM

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM | success | 6565.97 | 1539.53 | 10922 |


### Other

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | preprocess | success | 1771.49 | 1852.86 |  |
| crispyx | standardize csr | success | 1.19 | 854.45 |  |


## 2. Performance Comparison

### crispyx vs Reference Tools

_crispyx as baseline. Negative values = crispyx is faster/uses less memory._

#### DE: t-test

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test (subset, capped control) | scanpy t-test (subset, capped control) | -10.7s | 41.9% | ✅ | -11.3 MB | 99.4% | ⚠️ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon (subset) | -35140.4s | 11.0% | ✅ | +49705.3 MB | 537.7% | ❌ |
| Wilcoxon (subset, capped control) | scanpy Wilcoxon (subset, capped control) | -4381.2s | 0.8% | ✅ | -163.8 MB | 92.1% | ⚠️ |


#### DE: NB GLM

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM (subset, capped control) | edgeR NB-GLM (subset, capped control) | -48617.4s | 0.2% | ✅ | +73.4 MB | 107.4% | ⚠️ |
| NB-GLM (subset, capped control) | pertpy NB-GLM (subset, capped control) | -11765.2s | 0.8% | ✅ | -130234.9 MB | 0.8% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| package A | method A | package B | method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM (subset, capped control) | pertpy | NB-GLM (subset, capped control) | +36852.2s | 410.8% | ❌ | -130308.2 MB | 0.8% | ✅ |


## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### DE: t-test

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test (subset, capped control) | scanpy t-test (subset, capped control) | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: Wilcoxon

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon (subset) | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |
| Wilcoxon (subset, capped control) | scanpy Wilcoxon (subset, capped control) | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: NB GLM

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM (subset, capped control) | edgeR NB-GLM (subset, capped control) | 0.497<br><small>±0.053</small> | ❌ | 0.836<br><small>±0.028</small> | ⚠️ | 0.279<br><small>±0.017</small> | ❌ | 0.295<br><small>±0.071</small> | ❌ | 0.970<br><small>±0.020</small> | ✅ | 0.993<br><small>±0.001</small> | ✅ |
| NB-GLM (subset, capped control) | pertpy NB-GLM (subset, capped control) | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 0.999<br><small>±0.001</small> | ✅ | 0.999<br><small>±0.000</small> | ✅ |


### Tool Comparisons

| package A | method A | package B | method B | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM (subset, capped control) | pertpy | NB-GLM (subset, capped control) | 0.196<br><small>±0.066</small> | ❌ | 0.622<br><small>±0.038</small> | ❌ | 0.274<br><small>±0.018</small> | ❌ | 0.008<br><small>±0.033</small> | ❌ | 0.913<br><small>±0.034</small> | ⚠️ | 0.795<br><small>±0.018</small> | ❌ |


## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

### Effect Size Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test (subset, capped control) | scanpy t-test (subset, capped control) | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon (subset) | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon (subset, capped control) | scanpy Wilcoxon (subset, capped control) | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM (subset, capped control) | edgeR NB-GLM (subset, capped control) | 0.000 | ❌ | 0.001 | ❌ | 0.008 | ❌ |
| NB-GLM (subset, capped control) | pertpy NB-GLM (subset, capped control) | 0.977 | ✅ | 0.981 | ✅ | 0.993 | ✅ |


### P-value Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test (subset, capped control) | scanpy t-test (subset, capped control) | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon (subset) | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon (subset, capped control) | scanpy Wilcoxon (subset, capped control) | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM (subset, capped control) | edgeR NB-GLM (subset, capped control) | 0.821 | ✅ | 0.818 | ✅ | 0.811 | ✅ |
| NB-GLM (subset, capped control) | pertpy NB-GLM (subset, capped control) | 0.990 | ✅ | 0.988 | ✅ | 0.986 | ✅ |


_Note: Some methods are missing due to errors:_
- t-test vs scanpy t-test: _missing output: scanpy_de_t_test (no output file)_
- Wilcoxon vs scanpy Wilcoxon: _missing output: scanpy_de_wilcoxon (no output file)_
- t-test vs scanpy t-test (subset): _missing output: scanpy_de_t_test_subset (not run, no output)_
- NB-GLM vs pertpy NB-GLM (subset): _missing output: pertpy_de_pydeseq2_subset (not run, no output)_
- NB-GLM vs edgeR NB-GLM (subset): _missing output: edger_de_glm_subset (not run, no output)_

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

Control: 146,747 cells, cap 20,000. By the rule (decided by the control's size, not by which runs fit on a machine), this dataset's primary accuracy comparisons use the **capped** control (`_subset_ctrl` pairs, every method on the same cells); comparisons on the full control are supplementary.

| method | status | perturbations | control cells |
| --- | --- | --- | --- |
| crispyx_de_nb_glm_subset_ctrl | success | 50.0 |  |
| crispyx_de_t_test_subset_ctrl | success | 50.0 |  |
| crispyx_de_wilcoxon_subset_ctrl | success | 50.0 |  |
| edger_de_glm_subset_ctrl | success | 50.0 | 20,000 |
| pertpy_de_pydeseq2_subset_ctrl | success | 50.0 | 20,000 |
| scanpy_de_t_test_subset_ctrl | success | 50.0 | 20,000 |
| scanpy_de_wilcoxon_subset | success | 50.0 |  |
| scanpy_de_wilcoxon_subset_ctrl | success | 50.0 | 20,000 |
