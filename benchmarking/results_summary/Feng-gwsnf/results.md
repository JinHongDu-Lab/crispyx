## Benchmark summary

- **Methods executed:** 20
- **Succeeded:** 17 (85.0% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 3
  - Timeouts: 1
  - Memory limit exceeded: 1
  - Errors: 1
- **Average runtime:** 4207.012s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds
    - MemoryError: Unable to allocate 16.8 GiB for an array with shape (2260821654,) and data type int64
    - Worker process crashed (exitcode=2)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 301.55 | 12303.988 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 139.831 | 7253.75 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 274.939 | 1596.043 | 393465.0 | 32373.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | recovered | 422.958 | 110752.676 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 2636.316 | 881.55 | 4516.0 | 32373.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 108.848 | 1131.95 | 50.0 | 32373.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 183.578 | 4851.031 | 4516.0 | 32373.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 8.86 | 2602.812 | 50.0 | 32373.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 495.92 | 20210.328 | 4516.0 | 32373.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 30.652 | 2556.184 | 50.0 | 32373.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 30.761 |  |
| edger_de_glm_subset_ctrl | success | 38474.815 | 592.332 |
| pertpy_de_pydeseq2 | memory_limit | 297.135 | 121987.742 |
| pertpy_de_pydeseq2_subset_ctrl | success | 12451.124 | 142717.09 |
| scanpy_de_t_test | success | 1304.202 | 54165.906 |
| scanpy_de_t_test_subset_ctrl | success | 21.218 | 2650.246 |
| scanpy_de_wilcoxon | timeout | 21605.102 |  |
| scanpy_de_wilcoxon_subset_ctrl | success | 5172.969 | 2729.445 |

