## Benchmark summary

- **Methods executed:** 21
- **Succeeded:** 17 (81.0% success rate)
- **Did not succeed:** 4
  - Memory limit exceeded: 3
  - Errors: 1
- **Average runtime:** 20342.522s
- **Notable issues:**
  - Other errors recorded:
    - MemoryError: Unable to allocate 215. GiB for an array with shape (28904027252,) and data type int64
    - MemoryError: Unable to allocate 217. GiB for an array with shape (29136391388,) and data type int64
    - Worker process crashed (exitcode=-9)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 6410.82 | 35584.035 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 3933.778 | 18669.148 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 10017.427 | 4727.102 | 4498965.0 | 37465.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 914.451 | 112443.93 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 24149.445 | 2463.04 | 17094.0 | 37465.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 437.87 | 1610.43 | 50.0 | 37465.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 7627.776 | 14210.633 | 17094.0 | 37465.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 10.942 | 4206.461 | 50.0 | 37465.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 27868.471 | 75227.0 | 17094.0 | 37465.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 64.505 | 4323.223 | 50.0 | 37465.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm_subset_ctrl | success | 52279.66 | 1475.32 |
| pertpy_de_pydeseq2_subset | error | 3827.551 | 481777.633 |
| pertpy_de_pydeseq2_subset_ctrl | success | 14424.731 | 154426.414 |
| scanpy_de_t_test | memory_limit | 2740.337 | 111039.195 |
| scanpy_de_t_test_subset | success | 3485.531 | 35605.961 |
| scanpy_de_t_test_subset_ctrl | success | 23.689 | 4349.816 |
| scanpy_de_wilcoxon | memory_limit | 2394.904 | 111026.387 |
| scanpy_de_wilcoxon_subset | success | 255970.183 | 35555.125 |
| scanpy_de_wilcoxon_subset_ctrl | success | 6003.694 | 4255.902 |

