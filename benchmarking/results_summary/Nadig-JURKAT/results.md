## Benchmark summary

- **Methods executed:** 15
- **Succeeded:** 13 (86.7% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 2
  - Timeouts: 2
- **Average runtime:** 4729.644s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 70.032 | 2734.613 |  |  | preprocessing/pb_avg_log_cx_normalized_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 34.759 | 1743.961 |  |  | preprocessing/pb_pseudobulk_cx_normalized_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 18.865 | 21210.66 | 259086.0 | 8882.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 76.867 | 35680.27 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 3109.719 | 852.38 | 2168.0 | 8882.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 61.239 | 996.656 | 2168.0 | 8882.0 | de/de_cx_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 143.715 | 16188.582 | 2168.0 | 8882.0 | de/de_cx_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.167 |  |
| pertpy_de_pydeseq2 | timeout | 21606.125 |  |
| pertpy_de_pydeseq2_subset | success | 1634.851 | 26216.391 |
| scanpy_de_t_test | success | 156.552 | 12024.758 |
| scanpy_de_wilcoxon | recovered | 21605.072 |  |
| scanpy_de_wilcoxon_subset | success | 744.107 | 1689.445 |

