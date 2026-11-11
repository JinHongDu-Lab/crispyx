## Benchmark summary

- **Methods executed:** 14
- **Succeeded:** 13 (92.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 4742.861s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 43.821 | 3138.531 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 22.843 | 2005.73 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 12.898 | 13931.496 | 142281.0 | 9624.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 31.293 | 21350.453 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 4602.815 | 773.96 | 2160.0 | 9624.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 43.182 | 1047.227 | 2160.0 | 9624.0 | de/de_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 89.664 | 9766.047 | 2160.0 | 9624.0 | de/de_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.109 |  |
| pertpy_de_pydeseq2 | success | 27145.575 | 16331.957 |
| pertpy_de_pydeseq2_subset | success | 965.793 | 12025.531 |
| scanpy_de_t_test | success | 198.161 | 8618.281 |
| scanpy_de_wilcoxon | success | 11588.148 | 8753.84 |

