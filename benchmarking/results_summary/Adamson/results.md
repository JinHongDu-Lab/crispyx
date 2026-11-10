## Benchmark summary

- **Methods executed:** 14
- **Succeeded:** 13 (92.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 2224.982s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 33.139 | 3951.309 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 15.684 | 2882.41 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 6.091 | 5726.344 | 65282.0 | 19568.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 21.104 | 7628.695 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 719.095 | 595.9 | 88.0 | 19568.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 56.132 | 663.188 | 88.0 | 19568.0 | de/de_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 32.819 | 3278.93 | 88.0 | 19568.0 | de/de_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.245 |  |
| pertpy_de_pydeseq2 | success | 5253.051 | 33776.414 |
| pertpy_de_pydeseq2_subset | success | 1812.622 | 38107.945 |
| scanpy_de_t_test | success | 19.693 | 3222.523 |
| scanpy_de_wilcoxon | success | 1555.62 | 3240.668 |

