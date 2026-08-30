#!/bin/bash
#SBATCH --job-name=crispyx-bench
#SBATCH --partition=intel
#
# SLURM job script for running CRISPYx benchmarks with Singularity
#
# IMPORTANT: Use submit_benchmark.sh to auto-configure resources from config:
#   ./submit_benchmark.sh Adamson_subset.yaml
#   ./submit_benchmark.sh Replogle-GW-k562.yaml
#
# Log files are named: {dataset}_{YYYYMMDD_HHMMSS}.out/.err
#
# Or override manually:
#   sbatch --time=24:00:00 --mem=256G --cpus-per-task=32 slurm_benchmark.sh config.yaml

set -e

# ============================================================================
# IMPORTANT: Set PROJECT_ROOT to the absolute path of your project on HPC
# This is required because SLURM copies scripts to a spool directory,
# breaking relative path resolution.
# ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"

# Derived paths from PROJECT_ROOT
SINGULARITY_DIR="$PROJECT_ROOT/benchmarking/singularity"

# Configuration
CONFIG_FILE="${1:-Adamson_subset.yaml}"
MEMORY_LIMIT_GB="${2:-128}"  # Memory limit in GB for Singularity (from submit_benchmark.sh)
BENCHMARK_ARGS="${3:-}"      # Additional benchmark args (e.g., --force --clean)
SIF_FILE="${CRISPYX_SIF:-$SINGULARITY_DIR/crispyx-benchmark.sif}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
ORIGIN_DATA_DIR="${ORIGIN_DATA_DIR:-$PROJECT_ROOT/data/origin}"  # large-dataset .h5ad files; override on HPC
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmarking/results}"
LOGS_DIR="${LOGS_DIR:-$PROJECT_ROOT/benchmarking/logs}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/benchmarking/config}"

# Use SLURM-allocated resources
N_CORES="${SLURM_CPUS_PER_TASK:-8}"

# Create output directories (using absolute paths)
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Print job info
echo "========================================"
echo "CRISPYx Benchmark - SLURM Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $N_CORES"
echo "SLURM Memory: $((SLURM_MEM_PER_NODE / 1024))G"
echo "Container Memory Limit: ${MEMORY_LIMIT_GB}G"
echo "Config: $CONFIG_FILE"
[ -n "$BENCHMARK_ARGS" ] && echo "Benchmark args: $BENCHMARK_ARGS"
echo "Origin data dir: $ORIGIN_DATA_DIR"
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

# Build bind args: project root always mounted; /data/origin for large-dataset configs
# (all benchmark configs outside Adamson_subset use /data/origin/*.h5ad absolute paths)
# On HPC, set ORIGIN_DATA_DIR to wherever that data lives, e.g.
#   ORIGIN_DATA_DIR=/data/origin ./submit_benchmark.sh Feng-gwsf.yaml
ORIGIN_BIND=""
if [ -d "$ORIGIN_DATA_DIR" ]; then
    ORIGIN_BIND="--bind $ORIGIN_DATA_DIR:/data/origin:ro"
else
    echo "WARNING: ORIGIN_DATA_DIR '$ORIGIN_DATA_DIR' not found — large-dataset configs will fail."
fi

# Note: PROJECT_ROOT is bound to /workspace to preserve all relative paths.
# Memory is enforced by SLURM (--mem). Singularity --memory requires cgroups v2
# unified mode, which most HPC clusters do not support.
CMD="python -m benchmarking.tools.run_benchmarks --config benchmarking/config/$CONFIG_FILE"
if [ -n "$BENCHMARK_ARGS" ]; then
    CMD="$CMD $BENCHMARK_ARGS"
fi

# Pre-compile Numba kernels before the benchmark.
# New kernels (v0.7.4+) must be JIT-compiled on first use; doing this with a
# single thread keeps LLVM memory usage low and avoids OOM on large-CPU nodes.
# Skip compilation if the index file for the batch-perts kernel already exists
# in the workspace __pycache__ (i.e. warmup was done on a previous run).
NUMBA_CACHE_SENTINEL="$PROJECT_ROOT/src/crispyx/__pycache__/_kernels._wilcoxon_batch_perts_presorted_numba-1784.py311.nbi"
if [ -f "$NUMBA_CACHE_SENTINEL" ]; then
    echo "Numba kernel cache found — skipping warmup."
else
    echo "Pre-compiling Numba kernels (single-threaded)..."
    singularity exec \
        --bind "$PROJECT_ROOT:/workspace:rw" \
        $ORIGIN_BIND \
        --pwd /workspace \
        --env "PYTHONPATH=/workspace/src:/workspace" \
        --env "NUMBA_NUM_THREADS=1" \
        --env "PYTHONUNBUFFERED=1" \
        --env "R_HOME=/usr/lib/R" \
        "$SIF_FILE" \
        python3 /workspace/benchmarking/tools/numba_warmup.py
    echo "Numba warmup complete."
fi

echo "Running: $CMD"
# shellcheck disable=SC2086  # ORIGIN_BIND intentionally unquoted for word splitting
singularity exec \
    --bind "$PROJECT_ROOT:/workspace:rw" \
    $ORIGIN_BIND \
    --pwd /workspace \
    --env "PYTHONPATH=/workspace/src:/workspace" \
    --env "OMP_NUM_THREADS=$N_CORES" \
    --env "MKL_NUM_THREADS=$N_CORES" \
    --env "OPENBLAS_NUM_THREADS=$N_CORES" \
    --env "NUMBA_NUM_THREADS=$N_CORES" \
    --env "PYTHONUNBUFFERED=1" \
    --env "R_HOME=/usr/lib/R" \
    "$SIF_FILE" \
    bash -c "$CMD"

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
