## Benchmark summary

- **Methods executed:** 21
- **Succeeded:** 17 (81.0% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 4
  - Timeouts: 1
  - Memory limit exceeded: 1
  - Errors: 2
- **Average runtime:** 4504.375s
- **Notable issues:**
  - Other errors recorded:
    - Error: cannot allocate vector of size 52.4 Gb

    - Exceeded time limit of 32400 seconds
    - MemoryError: Unable to allocate 4.90 GiB for an array with shape (1315451992,) and data type int32
    - Worker process crashed (exitcode=2)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 231.665 | 7351.469 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 107.95 | 4768.523 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 223.098 | 1429.176 | 320706.0 | 31772.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | recovered | 163.336 | 90329.16 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 1219.268 | 751.43 | 2031.0 | 31772.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 155.172 | 1130.89 | 50.0 | 31772.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 98.386 | 2389.539 | 2031.0 | 31772.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 9.44 | 2898.375 | 50.0 | 31772.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 368.112 | 18136.91 | 2031.0 | 31772.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 36.26 | 2868.004 | 50.0 | 31772.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 22.417 |  |
| edger_de_glm_subset | error | 3044.489 | 230258.285 |
| edger_de_glm_subset_ctrl | success | 38982.814 | 472.156 |
| pertpy_de_pydeseq2 | memory_limit | 276.738 | 124415.375 |
| pertpy_de_pydeseq2_subset_ctrl | success | 11234.13 | 140820.984 |
| scanpy_de_t_test | success | 590.322 | 26949.324 |
| scanpy_de_t_test_subset_ctrl | success | 37.851 | 2946.773 |
| scanpy_de_wilcoxon | timeout | 32405.101 |  |
| scanpy_de_wilcoxon_subset_ctrl | success | 5250.688 | 2983.426 |

