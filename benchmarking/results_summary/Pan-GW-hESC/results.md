## Benchmark summary

- **Methods executed:** 19
- **Succeeded:** 17 (89.5% success rate)
- **Did not succeed:** 2
  - Timeouts: 1
  - Memory limit exceeded: 1
- **Average runtime:** 10817.318s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 86400 seconds
    - MemoryError: Unable to allocate 35.1 GiB for an array with shape (4714522126,) and data type int64

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 932.158 | 7856.773 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 445.333 | 5214.285 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 697.013 | 1935.066 | 1154040.0 | 35649.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 497.223 | 95200.707 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 2895.805 | 1055.77 | 2950.0 | 35649.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 160.711 | 1471.09 | 50.0 | 35649.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 379.458 | 3651.305 | 2950.0 | 35649.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 9.766 | 3482.102 | 50.0 | 35649.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 885.714 | 37143.648 | 2950.0 | 35649.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 52.608 | 3300.125 | 50.0 | 35649.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm_subset_ctrl | success | 47194.319 | 646.676 |
| pertpy_de_pydeseq2_subset_ctrl | success | 12381.545 | 147185.418 |
| scanpy_de_t_test | success | 1641.075 | 80127.668 |
| scanpy_de_t_test_subset_ctrl | success | 20.947 | 3380.656 |
| scanpy_de_wilcoxon | timeout | 86405.179 |  |
| scanpy_de_wilcoxon_subset | success | 45682.092 | 7740.59 |
| scanpy_de_wilcoxon_subset_ctrl | success | 4875.402 | 3379.453 |

