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

# STAGE_LOCAL=1: run crispyx's own sort with its input read from the node's
# local SSD. The sort reads rows in perturbation order -- scattered reads that
# crawl on Lustre (~1 GB/h for Huang-HCT116) but are cheap on a local disk.
# The copy keeps size and mtime and is bind-mounted at the file's real path,
# so the sorted copy's provenance matches the Lustre file and NB-GLM reuses
# it. The sorted copy itself is written to Lustre as usual.
if [ "${STAGE_LOCAL:-0}" = 1 ]; then
    D="${CONFIG_FILE%.yaml}"
    SRC="$(readlink -f "$PROJECT_ROOT/benchmarking/results/$D/.cache/standardized_$D.h5ad")"
    # NBGLM_INPUT=qc: the NB-GLM input is the QC-filtered counts (crispyx >= 0.2.3).
    [ "${NBGLM_INPUT:-}" = qc ] && SRC="$(readlink -f "$PROJECT_ROOT/benchmarking/results/$D/preprocessing/crispyx_qc_filtered.h5ad")"
    # crispyx >= 0.2 writes a small standardized file whose matrices are HDF5
    # external links to the raw file (config dataset_path, repo-relative);
    # stage that raw file instead -- it is what the sort reads.
    if [ "${NBGLM_INPUT:-}" != qc ] && [ "$(stat -c %s "$SRC")" -lt 2000000000 ]; then
        RAW="$(sed -n 's/^dataset_path: *"\{0,1\}\([^"# ]*\).*/\1/p' "$PROJECT_ROOT/benchmarking/config/$CONFIG_FILE")"
        case "$RAW" in /*) ;; *) RAW="$PROJECT_ROOT/$RAW" ;; esac
        SRC="$(readlink -f "$RAW")"
        echo "Standardized file links its matrix; staging the raw file $SRC"
    fi
    LOCAL="${TMPDIR:-/tmp}/$(basename "$SRC")"
    NEED=$(stat -c %s "$SRC")
    AVAIL=$(df --output=avail -B1 "$(dirname "$LOCAL")" | tail -1)
    if [ "$AVAIL" -le "$NEED" ]; then
        echo "ERROR: local disk has $AVAIL bytes free, need $NEED"; exit 1
    fi
    trap 'rm -f "$LOCAL"' EXIT
    T0=$(date +%s)
    # Direct I/O on both ends: a plain cp leaves source and copy in the page
    # cache, which Slurm charges to the job (~2x the file size). dd drops
    # O_DIRECT itself for a short final block.
    dd if="$SRC" of="$LOCAL" bs=64M iflag=direct oflag=direct status=none
    touch -r "$SRC" "$LOCAL"
    echo "Staged $SRC -> $LOCAL ($NEED bytes) in $(( $(date +%s) - T0 )) s"
    if [ "$(stat -c '%s %Y' "$SRC")" != "$(stat -c '%s %Y' "$LOCAL")" ]; then
        echo "ERROR: staged copy differs in size or mtime"; exit 1
    fi
    BINDS="$BINDS --bind $LOCAL:/workspace/${SRC#"$PROJECT_ROOT"/}"
fi

echo "$TOOL: $CONFIG_FILE (job $SLURM_JOB_ID on $SLURMD_NODENAME)"
echo "crispyx source: ${CRISPYX_SRC:-$PROJECT_ROOT/src}"
echo "Start: $(date)"

# shellcheck disable=SC2086  # BINDS intentionally unquoted for word splitting
singularity exec $BINDS \
    --pwd /workspace \
    --env "PYTHONPATH=/workspace/src:/workspace" \
    --env "PYTHONUNBUFFERED=1" \
    --env "NBGLM_INPUT=${NBGLM_INPUT:-}" \
    "$SIF_FILE" \
    python -m "benchmarking.tools.$TOOL" "benchmarking/config/$CONFIG_FILE"

echo "End: $(date)"
