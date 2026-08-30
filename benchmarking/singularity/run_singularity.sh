#!/bin/bash
# Run crispyx benchmark using Singularity on HPC
#
# Usage:
#   ./run_singularity.sh [CONFIG_FILE] [OPTIONS]
#
# Examples:
#   ./run_singularity.sh config/Adamson.yaml
#   ./run_singularity.sh config/Adamson.yaml --tools crispyx scanpy
#   ./run_singularity.sh config/Adamson.yaml --dry-run
#
# Environment variables:
#   CRISPYX_SIF: Path to Singularity image (default: ./crispyx-benchmark.sif)
#   DATA_DIR: Path to data directory (default: ../data)
#   RESULTS_DIR: Path to results directory (default: ./results)
#   N_CORES: Number of CPU cores to use (default: 8)
#
# SLURM submission example:
#   sbatch --cpus-per-task=8 --mem=64G --time=4:00:00 run_singularity.sh config/Adamson.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration with defaults
SIF_FILE="${CRISPYX_SIF:-$SCRIPT_DIR/crispyx-benchmark.sif}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../data}"
ORIGIN_DATA_DIR="${ORIGIN_DATA_DIR:-$SCRIPT_DIR/../../data/origin}"  # large-dataset .h5ad files; override on HPC
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
LOGS_DIR="${LOGS_DIR:-$SCRIPT_DIR/logs}"
CONFIG_DIR="${CONFIG_DIR:-$SCRIPT_DIR/config}"
N_CORES="${N_CORES:-8}"

# Parse arguments
CONFIG_FILE="${1:-config/Adamson_subset.yaml}"
shift || true  # Remove first argument, rest are passed to the benchmark script

# Resolve absolute paths
DATA_DIR="$(cd "$DATA_DIR" 2>/dev/null && pwd)"
RESULTS_DIR="$(mkdir -p "$RESULTS_DIR" && cd "$RESULTS_DIR" && pwd)"
LOGS_DIR="$(mkdir -p "$LOGS_DIR" && cd "$LOGS_DIR" && pwd)"
CONFIG_DIR="$(cd "$CONFIG_DIR" 2>/dev/null && pwd)"

# Validate SIF file exists
if [ ! -f "$SIF_FILE" ]; then
    echo "ERROR: Singularity image not found: $SIF_FILE"
    echo "Set CRISPYX_SIF environment variable or place crispyx-benchmark.sif in $SCRIPT_DIR"
    exit 1
fi

echo "=== Running CRISPYx Benchmark with Singularity ==="
echo "SIF file: $SIF_FILE"
echo "Config: $CONFIG_FILE"
echo "Data dir: $DATA_DIR"
echo "Origin data dir: $ORIGIN_DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Cores: $N_CORES"
echo ""

# Set thread environment variables
export OMP_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OPENBLAS_NUM_THREADS=$N_CORES
export NUMBA_NUM_THREADS=$N_CORES

# Run Singularity container
# --bind mounts directories into the container
# --pwd sets the working directory
# --cleanenv prevents host environment from leaking in (except explicitly passed vars)
# Build bind args for /data/origin (large-dataset absolute paths in YAML configs)
ORIGIN_BIND=""
if [ -d "$ORIGIN_DATA_DIR" ]; then
    ORIGIN_BIND="--bind $ORIGIN_DATA_DIR:/data/origin:ro"
else
    echo "WARNING: ORIGIN_DATA_DIR '$ORIGIN_DATA_DIR' not found — large-dataset configs will fail."
fi

# shellcheck disable=SC2086  # ORIGIN_BIND intentionally unquoted for word splitting
singularity exec \
    --bind "$DATA_DIR:/workspace/data:ro" \
    --bind "$RESULTS_DIR:/workspace/benchmarking/results:rw" \
    --bind "$LOGS_DIR:/workspace/benchmarking/logs:rw" \
    --bind "$CONFIG_DIR:/workspace/benchmarking/config:ro" \
    $ORIGIN_BIND \
    --pwd /workspace \
    --env "OMP_NUM_THREADS=$N_CORES" \
    --env "MKL_NUM_THREADS=$N_CORES" \
    --env "OPENBLAS_NUM_THREADS=$N_CORES" \
    --env "NUMBA_NUM_THREADS=$N_CORES" \
    --env "PYTHONUNBUFFERED=1" \
    "$SIF_FILE" \
    python -m benchmarking.tools.run_benchmarks --config "benchmarking/$CONFIG_FILE" "$@"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $RESULTS_DIR"
