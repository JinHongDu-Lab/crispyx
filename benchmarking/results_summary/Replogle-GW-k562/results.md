## Benchmark summary

- **Methods executed:** 20
- **Succeeded:** 18 (90.0% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 2
  - Timeouts: 1
  - Errors: 1
- **Average runtime:** 6400.192s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 86400 seconds
    - Worker process crashed (exitcode=2)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 717.026 | 7179.035 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 394.693 | 4610.738 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 1146.379 | 2865.914 | 1971608.0 | 8248.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | recovered | 402.397 | 126400.973 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 3704.193 | 1519.0 | 9326.0 | 8248.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 161.584 | 955.96 | 50.0 | 8248.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 1900.94 | 2789.355 | 9326.0 | 8248.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 7.075 | 1936.035 | 50.0 | 8248.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 4364.265 | 49393.605 | 9326.0 | 8248.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 35.836 | 2932.387 | 50.0 | 8248.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 219.135 |  |
| edger_de_glm_subset_ctrl | success | 3019.34 | 335.824 |
| pertpy_de_pydeseq2_subset_ctrl | success | 3094.155 | 39434.25 |
| scanpy_de_t_test | success | 1279.114 | 119350.0 |
| scanpy_de_t_test_subset_ctrl | success | 10.72 | 2131.312 |
| scanpy_de_wilcoxon | timeout | 86405.101 |  |
| scanpy_de_wilcoxon_subset | success | 18635.235 | 5474.047 |
| scanpy_de_wilcoxon_subset_ctrl | success | 1409.135 | 2220.199 |

