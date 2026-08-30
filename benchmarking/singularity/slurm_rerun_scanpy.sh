#!/bin/bash
#SBATCH --job-name=rerun-scanpy
#SBATCH --partition=amd
#
# SLURM job script for rerunning Scanpy methods with Singularity
#
# IMPORTANT: Use submit_rerun_scanpy.sh to auto-configure resources from config:
#   ./submit_rerun_scanpy.sh Adamson_subset.yaml
#   ./submit_rerun_scanpy.sh Replogle-GW-k562.yaml
#
# Log files are named: rerun_scanpy_{dataset}_{YYYYMMDD_HHMMSS}.out/.err

set -e

# ============================================================================
# IMPORTANT: Set PROJECT_ROOT to the absolute path of your project on HPC
# ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"

# Derived paths from PROJECT_ROOT
SINGULARITY_DIR="$PROJECT_ROOT/benchmarking/singularity"

# Configuration
CONFIG_FILE="${1:-Adamson_subset.yaml}"
EXTRA_ARGS="${2:-}"
SIF_FILE="${CRISPYX_SIF:-$SINGULARITY_DIR/crispyx-benchmark.sif}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
ORIGIN_DATA_DIR="${ORIGIN_DATA_DIR:-$PROJECT_ROOT/data/origin}"  # large-dataset .h5ad files; override on HPC
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmarking/results}"
LOGS_DIR="${LOGS_DIR:-$PROJECT_ROOT/benchmarking/logs}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/benchmarking/config}"

# Use SLURM-allocated resources (default 32 cores, same as benchmark)
N_CORES="${SLURM_CPUS_PER_TASK:-32}"

# Create output directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Print job info
echo "========================================"
echo "Rerun Scanpy - SLURM Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $N_CORES"
echo "Time limit: ${PRESET_TIME:-unknown}"
echo "Memory limit: ${PRESET_MEM:-unknown}"
echo "Config: $CONFIG_FILE"
[ -n "$EXTRA_ARGS" ] && echo "Extra args: $EXTRA_ARGS"
echo "Start time: $(date)"
echo "========================================"

# Validate required files
if [ ! -f "$SIF_FILE" ]; then
    echo "ERROR: Singularity image not found: $SIF_FILE"
    exit 1
fi

# Set thread limits
export OMP_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OPENBLAS_NUM_THREADS=$N_CORES
export NUMBA_NUM_THREADS=$N_CORES

# Build bind args for /data/origin (large-dataset absolute paths in YAML configs)
ORIGIN_BIND=""
if [ -d "$ORIGIN_DATA_DIR" ]; then
    ORIGIN_BIND="--bind $ORIGIN_DATA_DIR:/data/origin:ro"
else
    echo "WARNING: ORIGIN_DATA_DIR '$ORIGIN_DATA_DIR' not found — large-dataset configs will fail."
fi

# Build command with optional extra args
CMD="python -m benchmarking.tools.rerun_scanpy --config benchmarking/config/$CONFIG_FILE"
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
# shellcheck disable=SC2086  # ORIGIN_BIND intentionally unquoted for word splitting
singularity exec \
    --bind "$PROJECT_ROOT:/workspace:rw" \
    $ORIGIN_BIND \
    --pwd /workspace \
    --env "OMP_NUM_THREADS=$N_CORES" \
    --env "MKL_NUM_THREADS=$N_CORES" \
    --env "OPENBLAS_NUM_THREADS=$N_CORES" \
    --env "NUMBA_NUM_THREADS=$N_CORES" \
    --env "PYTHONUNBUFFERED=1" \
    "$SIF_FILE" \
    bash -c "$CMD"

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
