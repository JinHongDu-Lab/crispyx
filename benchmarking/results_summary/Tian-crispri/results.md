## Benchmark summary

- **Methods executed:** 14
- **Succeeded:** 13 (92.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 516.150s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 3600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 17.764 | 4128.719 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 9.253 | 2971.152 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 4.116 | 3536.105 | 32255.0 | 23451.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 16.846 | 4663.523 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 949.632 | 576.24 | 183.0 | 23451.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 8.916 | 733.863 | 183.0 | 23451.0 | de/de_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 22.698 | 1889.891 | 183.0 | 23451.0 | de/de_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 3605.109 |  |
| pertpy_de_pydeseq2 | success | 1862.688 | 4631.348 |
| pertpy_de_pydeseq2_subset | success | 499.979 | 4109.695 |
| scanpy_de_t_test | success | 28.641 | 2246.453 |
| scanpy_de_wilcoxon | success | 187.596 | 2238.117 |

