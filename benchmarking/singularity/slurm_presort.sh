#!/bin/bash
#SBATCH --job-name=crispyx-presort
#SBATCH --partition=amd
#
# Pre-sort a dataset by perturbation for NB-GLM (see tools/presort_nbglm.py),
# so the timed crispyx_de_nb_glm run reuses the sorted copy and measures
# fitting only. Chain the benchmark job after it, e.g.:
#   S=$(sbatch --parsable --time=5-00:00:00 --mem=140G --cpus-per-task=8 \
#         slurm_presort.sh VCC.yaml)
#   sbatch --dependency=afterok:$S ... slurm_benchmark.sh VCC.yaml 128 \
#         '--methods crispyx_de_nb_glm --force --skip-existing'
# Set CRISPYX_SRC to the same crispyx source as the benchmark job.
#
# TOOL=prestandardize runs tools/prestandardize.py instead: the same
# standardize_dataset call as the benchmark, with a size-verified copy.

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"
SIF_FILE="${CRISPYX_SIF:-$PROJECT_ROOT/benchmarking/singularity/crispyx-benchmark.sif}"
ORIGIN_DATA_DIR="${ORIGIN_DATA_DIR:-$PROJECT_ROOT/data/origin}"
CONFIG_FILE="$1"
TOOL="${TOOL:-presort_nbglm}"

BINDS="--bind $PROJECT_ROOT:/workspace:rw"
[ -d "$ORIGIN_DATA_DIR" ] && BINDS="$BINDS --bind $ORIGIN_DATA_DIR:/data/origin:ro"
[ -n "${CRISPYX_SRC:-}" ] && BINDS="$BINDS --bind $CRISPYX_SRC:/workspace/src"

echo "$TOOL: $CONFIG_FILE (job $SLURM_JOB_ID on $SLURMD_NODENAME)"
echo "crispyx source: ${CRISPYX_SRC:-$PROJECT_ROOT/src}"
echo "Start: $(date)"

# shellcheck disable=SC2086  # BINDS intentionally unquoted for word splitting
singularity exec $BINDS \
    --pwd /workspace \
    --env "PYTHONPATH=/workspace/src:/workspace" \
    --env "PYTHONUNBUFFERED=1" \
    "$SIF_FILE" \
    python -m "benchmarking.tools.$TOOL" "benchmarking/config/$CONFIG_FILE"

echo "End: $(date)"
