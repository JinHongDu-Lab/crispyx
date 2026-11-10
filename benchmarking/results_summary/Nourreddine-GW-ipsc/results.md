## Benchmark summary

- **Methods executed:** 19
- **Succeeded:** 16 (84.2% success rate)
- **Did not succeed:** 3
  - Memory limit exceeded: 3
- **Average runtime:** 6803.703s
- **Notable issues:**
  - Other errors recorded:
    - MemoryError: Unable to allocate 29.1 GiB for an array with shape (7818423035,) and data type float32
    - MemoryError: Unable to allocate 58.7 GiB for an array with shape (7872183461,) and data type int64

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 2534.976 | 24246.656 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 1298.474 | 13242.605 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 2526.767 | 2816.078 | 2641058.0 | 36246.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 138.469 | 121051.648 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 6565.968 | 1539.53 | 10922.0 | 36246.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 91.609 | 1065.31 | 50.0 | 36246.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 1766.013 | 8574.363 | 10922.0 | 36246.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 7.688 | 2019.074 | 50.0 | 36246.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 4332.238 | 61061.816 | 10922.0 | 36246.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 33.559 | 1909.129 | 50.0 | 36246.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm_subset_ctrl | success | 48708.962 | 991.914 |
| pertpy_de_pydeseq2_subset_ctrl | success | 11856.769 | 131300.164 |
| scanpy_de_t_test | memory_limit | 1601.085 | 125309.938 |
| scanpy_de_t_test_subset_ctrl | success | 18.354 | 2030.363 |
| scanpy_de_wilcoxon | memory_limit | 2129.341 | 125310.949 |
| scanpy_de_wilcoxon_subset | success | 39472.636 | 11356.527 |
| scanpy_de_wilcoxon_subset_ctrl | success | 4414.763 | 2072.922 |

