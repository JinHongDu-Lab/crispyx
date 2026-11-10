## Benchmark summary

- **Methods executed:** 15
- **Succeeded:** 13 (86.7% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 2
  - Timeouts: 2
- **Average runtime:** 5405.045s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 81.727 | 2791.164 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 39.498 | 1890.68 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 22.883 | 25854.727 | 304114.0 | 8563.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 115.239 | 40464.887 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 13430.721 | 4459.62 | 2057.0 | 8563.0 | de/de_cx_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 67.077 | 974.848 | 1853.0 | 8563.0 | de/de_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 157.278 | 19291.203 | 1853.0 | 8563.0 | de/de_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.122 |  |
| pertpy_de_pydeseq2 | timeout | 21606.588 |  |
| pertpy_de_pydeseq2_subset | success | 1440.51 | 22523.797 |
| scanpy_de_t_test | success | 131.821 | 14471.715 |
| scanpy_de_wilcoxon | recovered | 21605.101 |  |
| scanpy_de_wilcoxon_subset | success | 677.255 | 1654.363 |

