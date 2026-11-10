## Benchmark summary

- **Methods executed:** 22
- **Succeeded:** 17 (77.3% success rate)
- **Did not succeed:** 5
  - Memory limit exceeded: 3
  - Errors: 2
- **Average runtime:** 11941.199s
- **Notable issues:**
  - Other errors recorded:
    - Error: cannot allocate vector of size 40.3 Gb

    - MemoryError: Unable to allocate 128. GiB for an array with shape (17152914017,) and data type int64
    - MemoryError: Unable to allocate 130. GiB for an array with shape (17414414157,) and data type int64
    - PyDESeq2 failed for all 50 perturbations; first error: ABCB9: MemoryError: Unable to allocate 43.4 GiB for an array with shape (165942, 35117) and data type float64

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 4485.617 | 36495.336 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 2432.728 | 19233.801 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 5079.441 | 3472.383 | 3359635.0 | 36326.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 1292.232 | 67301.801 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 16972.922 | 2094.7 | 16482.0 | 36326.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 171.121 | 1352.5 | 50.0 | 36326.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 5633.616 | 12917.836 | 16482.0 | 36326.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 9.419 | 3114.602 | 50.0 | 36326.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 15882.407 | 74009.207 | 16482.0 | 36326.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 44.438 | 3192.527 | 50.0 | 36326.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm_subset | error | 3385.04 | 217660.973 |
| edger_de_glm_subset_ctrl | success | 76614.829 | 1117.699 |
| pertpy_de_pydeseq2_subset | error | 36621.653 | 473265.941 |
| pertpy_de_pydeseq2_subset_ctrl | success | 10343.238 | 147133.398 |
| scanpy_de_t_test | memory_limit | 1466.943 | 66043.449 |
| scanpy_de_t_test_subset | success | 3677.889 | 21562.742 |
| scanpy_de_t_test_subset_ctrl | success | 20.737 | 3200.137 |
| scanpy_de_wilcoxon | memory_limit | 474.276 | 66045.664 |
| scanpy_de_wilcoxon_subset | success | 70574.233 | 21382.883 |
| scanpy_de_wilcoxon_subset_ctrl | success | 5286.545 | 3161.957 |

