## Benchmark summary

- **Methods executed:** 22
- **Succeeded:** 19 (86.4% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 3
  - Timeouts: 1
  - Memory limit exceeded: 1
  - Errors: 1
- **Average runtime:** 4319.241s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds
    - MemoryError: Unable to allocate 753. MiB for an array with shape (197395868,) and data type float32
    - Worker process crashed (exitcode=2)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 87.111 | 3115.305 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 39.682 | 2285.117 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 15.274 | 17361.766 | 218023.0 | 23710.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 101.886 | 23060.957 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 424.445 | 1046.39 | 239.0 | 23710.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 363.426 | 1388.36 | 50.0 | 23710.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 26.017 | 828.789 | 239.0 | 23710.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 13.44 | 3741.203 | 50.0 | 23710.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 104.236 | 10971.363 | 239.0 | 23710.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 84.617 | 3588.246 | 50.0 | 23710.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 1380.128 |  |
| edger_de_glm_subset_ctrl | success | 29845.634 | 366.988 |
| pertpy_de_pydeseq2 | timeout | 21606.513 |  |
| pertpy_de_pydeseq2_subset | memory_limit | 2169.376 | 241933.914 |
| pertpy_de_pydeseq2_subset_ctrl | success | 6288.654 | 112948.598 |
| scanpy_de_t_test | success | 63.719 | 9674.27 |
| scanpy_de_t_test_subset_ctrl | success | 31.893 | 3765.719 |
| scanpy_de_wilcoxon | recovered | 21605.101 |  |
| scanpy_de_wilcoxon_subset | success | 7224.866 | 5797.684 |
| scanpy_de_wilcoxon_subset_ctrl | success | 3483.561 | 3681.629 |

