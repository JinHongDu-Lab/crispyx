## Benchmark summary

- **Methods executed:** 21
- **Succeeded:** 16 (76.2% success rate)
- **Did not succeed:** 5
  - Timeouts: 1
  - Memory limit exceeded: 2
  - Errors: 2
- **Average runtime:** 12432.761s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 86400 seconds
    - MemoryError: Unable to allocate 3.79 GiB for an array with shape (1017527740,) and data type float32
    - MemoryError: Unable to allocate 32.0 GiB for an array with shape (4288967098,) and data type int64
    - PyDESeq2 failed for all 298 perturbations; first error: CHMP3: MemoryError: Unable to allocate 15.4 GiB for an array with shape (115694, 17851) and data type float64
    - Worker process crashed (exitcode=-6)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 272.081 | 2528.492 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 152.81 | 1881.551 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 616.471 | 1976.777 | 490966.0 | 18055.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 58.953 | 119308.574 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 1103.04 | 663.25 | 298.0 | 18055.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_subset_ctrl | NB-GLM base fitting (no shrinkage) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 574.599 | 3297.23 | 50.0 | 18055.0 | de/crispyx_de_nb_glm_subset_ctrl.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 146.011 | 1378.996 | 298.0 | 18055.0 | de/de_cx_t_test.h5ad |
| crispyx_de_t_test_subset_ctrl | t-test differential expression test [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 25.428 | 14037.66 | 50.0 | 18055.0 | de/de_t_test_subset_ctrl_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 631.062 | 34283.73 | 298.0 | 18055.0 | de/de_cx_wilcoxon.h5ad |
| crispyx_de_wilcoxon_subset_ctrl | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) [accuracy rerun: 50 seeded perturbations, 20,000 control cells] | success | 218.783 | 14251.109 | 50.0 | 18055.0 | de/de_wilcoxon_subset_ctrl_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 54.274 | 83733.77 |
| edger_de_glm_subset_ctrl | success | 21980.253 | 346.93 |
| pertpy_de_pydeseq2 | error | 80862.469 | 123122.023 |
| pertpy_de_pydeseq2_subset | memory_limit | 3214.889 | 240061.871 |
| pertpy_de_pydeseq2_subset_ctrl | success | 8034.406 | 93705.582 |
| scanpy_de_t_test | success | 1777.934 | 70040.902 |
| scanpy_de_t_test_subset_ctrl | success | 39.678 | 14187.695 |
| scanpy_de_wilcoxon | timeout | 86405.222 |  |
| scanpy_de_wilcoxon_subset | success | 50274.969 | 36026.426 |
| scanpy_de_wilcoxon_subset_ctrl | success | 4302.996 | 14015.98 |

