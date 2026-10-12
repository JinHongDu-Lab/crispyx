#!/bin/bash
#SBATCH --partition=amd
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=140G
#
# Re-time the crispyx methods of one dataset on an exclusive node.
#
# Shared-node contention changed runtimes by up to ~4x (Feng-gwsf NB-GLM:
# 19,274 s shared vs ~4,500 s exclusive), so the runtime figures use these
# timings. Results go to benchmarking/results_timing/<dataset>; the real
# results directory is not touched. The standardized inputs (and NB-GLM's
# sorted copy, made by slurm_presort.sh) are symlinked in, so nothing large
# is recomputed outside the timed methods. Large outputs are deleted at the
# end; results.csv and .benchmark_cache are kept.
#
# Usage:
#   CRISPYX_SRC=$PROJECT_ROOT/crispyx-0.1.8/src \
#     sbatch --export=ALL,DATASET=VCC --job-name=retime-slot1 \
#            --dependency=singleton slurm_retime_crispyx.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"
BENCH="$PROJECT_ROOT/benchmarking"
D="${DATASET:?set DATASET}"
SRC_DIR="$(readlink -f "$BENCH/results/$D")"
T="$BENCH/results_timing/$D"
METHODS="crispyx_qc_filtered crispyx_preprocess crispyx_pb_avg_log crispyx_pb_pseudobulk crispyx_de_t_test crispyx_de_wilcoxon crispyx_de_nb_glm"

echo "Re-timing crispyx on $D (job $SLURM_JOB_ID on $SLURMD_NODENAME, exclusive)"
lscpu | grep -E "Model name|^CPU\(s\)"

mkdir -p "$T/.cache" "$T/preprocessing" "$BENCH/config/timing"
for f in "$SRC_DIR"/.cache/standardized_"$D".h5ad "$SRC_DIR"/.cache/standardized_"$D"_sorted.h5ad \
         "$SRC_DIR"/preprocessing/standardized_csr_"$D".h5ad "$SRC_DIR"/preprocessing/standardized_csr_"$D"_sorted.h5ad; do
    # Relative links: the container mounts the project at /workspace, so
    # absolute /lustre1 targets would dangle in there.
    [ -e "$f" ] && ln -sfnr "$f" "$T/${f#"$SRC_DIR"/}"
done
# Same config, results redirected to the timing directory.
sed "s#^output_dir: .*#output_dir: \"benchmarking/results_timing/$D\"#" \
    "$BENCH/config/$D.yaml" > "$BENCH/config/timing/$D.yaml"

bash "$BENCH/singularity/slurm_benchmark.sh" "timing/$D.yaml" 128 "--methods $METHODS"

# Keep the timings, drop the large outputs (real files only; links go too).
find "$T" -name "*.h5ad" \( -type f -o -type l \) -delete
find "$T" -name "*.csv" -path "*/de/*" -delete
rm -f "$BENCH/config/timing/$D.yaml"
echo "Kept: $(ls -A "$T")"
