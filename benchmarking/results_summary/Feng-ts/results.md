## Benchmark summary

- **Methods executed:** 21
- **Succeeded:** 17 (81.0% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 4
  - Timeouts: 1
  - Memory limit exceeded: 2
  - Errors: 1
- **Average runtime:** 16084.389s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 259200 seconds
    - MemoryError: Unable to allocate 158. GiB for an array with shape (1161864, 36518) and data type float32
    - MemoryError: Unable to allocate 66.4 GiB for an array with shape (537132, 33165) and data type float32
    - Worker process crashed (exitcode=2)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 1329.817 | 5188.504 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 799.751 | 3775.117 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 1921.468 | 1683.031 | 1161864.0 | 33165.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | recovered | 17.538 | 354.637 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 1793.74 | 930.32 | 444.0 | 33165.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 309.859 | 2462.76 | 50.0 | 33165.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 1538.446 | 1499.848 | 444.0 | 33165.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 17.1 | 8783.207 | 50.0 | 33165.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 4955.032 | 46899.09 | 444.0 | 33165.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 161.262 | 8696.324 | 50.0 | 33165.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | memory_limit | 9.124 |  |
| edger_de_glm_subset | error | 706.109 | 124203.965 |
| edger_de_glm_subset_ctrl | success | 43300.822 | 624.465 |
| pertpy_de_pydeseq2 | memory_limit | 439.368 | 123455.074 |
| pertpy_de_pydeseq2_subset_ctrl | success | 12864.041 | 148166.875 |
| scanpy_de_t_test | success | 1181.284 | 101367.652 |
| scanpy_de_t_test_subset_ctrl | success | 30.067 | 8910.805 |
| scanpy_de_wilcoxon | timeout | 259205.104 |  |
| scanpy_de_wilcoxon_subset_ctrl | success | 5420.033 | 8907.91 |

