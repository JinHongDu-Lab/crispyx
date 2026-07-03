## Benchmark summary

- **Methods executed:** 13
- **Succeeded:** 9 (69.2% success rate)
- **Did not succeed:** 4
  - Timeouts: 1
  - Memory limit exceeded: 2
  - Errors: 1
- **Average runtime:** 9042.711s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 86400 seconds
    - MemoryError: Unable to allocate 46.4 GiB for an array with shape (6233255629,) and data type int64
    - MemoryError: Unable to allocate 61.1 GiB for an array with shape (1989578, 8248) and data type float32
    - Worker process crashed (exitcode=2)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 830.399 | 4690.449 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 760.677 | 4502.727 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 2645.87 | 50227.141 | 1971608.0 | 8248.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 402.397 | 126400.973 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 15809.574 | 21837.68 | 9866.0 | 8248.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 805.471 | 5953.652 | 9326.0 | 8248.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon DE with CSR→CSC conversion (conversion + DE timed together) | success | 7629.281 | 49747.863 | 9326.0 | 8248.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 219.135 |  |
| pertpy_de_pydeseq2 | memory_limit | 585.331 | 122314.895 |
| scanpy_de_t_test | success | 1279.114 | 119350.0 |
| scanpy_de_wilcoxon | timeout | 86405.101 |  |

