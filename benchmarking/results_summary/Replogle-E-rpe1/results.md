## Benchmark summary

- **Methods executed:** 15
- **Succeeded:** 13 (86.7% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 2
  - Timeouts: 2
- **Average runtime:** 5437.257s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 64.635 | 3004.43 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 32.312 | 1972.129 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 17.487 | 20137.82 | 244412.0 | 8749.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 85.562 | 33264.031 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 13721.302 | 4888.62 | 2393.0 | 8749.0 | de/de_cx_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 57.639 | 1017.871 | 2155.0 | 8749.0 | de/de_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 135.693 | 15501.246 | 2155.0 | 8749.0 | de/de_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.11 |  |
| pertpy_de_pydeseq2 | timeout | 21605.861 |  |
| pertpy_de_pydeseq2_subset | success | 1668.72 | 24312.562 |
| scanpy_de_t_test | success | 148.822 | 11475.078 |
| scanpy_de_wilcoxon | recovered | 21605.101 |  |
| scanpy_de_wilcoxon_subset | success | 735.512 | 1513.992 |

