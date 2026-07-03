#!/bin/bash
#
# Wrapper script to submit SLURM jobs for rerunning Scanpy methods
#
# Usage:
#   ./submit_rerun_scanpy.sh                              # Submit all default configs
#   ./submit_rerun_scanpy.sh Adamson_subset.yaml          # Submit single config
#   ./submit_rerun_scanpy.sh Replogle-GW-k562.yaml --force  # Submit with force flag
#
# Resources: 32 cores (same as benchmark), no memory limit, 3-day time limit.
# Run this AFTER benchmarks to get Scanpy outputs for datasets where Scanpy
# timed out or ran out of memory.

set -e

# Project root (modify for your HPC)
PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/crispyx}"
CONFIG_DIR="$PROJECT_ROOT/benchmarking/config"
SLURM_SCRIPT="$PROJECT_ROOT/benchmarking/singularity/slurm_rerun_scanpy.sh"

# Default config files - only large datasets where Scanpy typically fails
# Small datasets usually complete in benchmark, so skip by default
DEFAULT_CONFIGS=(
    # "Feng-ts.yaml"
    # "Feng-gwsf.yaml"
    # "Feng-gwsnf.yaml"
    "Replogle-GW-k562.yaml"
    # "Replogle-E-k562.yaml"
    # "Replogle-E-rpe1.yaml"
    # "Huang-HCT116.yaml"
    # "Huang-HEK293T.yaml"
)

# Script-specific flags that should be forwarded to Python script
SCRIPT_FLAGS=("--force" "--no-report" "--methods" "--quiet")

# If no arguments provided, use default configs
if [ $# -lt 1 ]; then
    echo "No config specified, using default configs (large datasets):"
    for cfg in "${DEFAULT_CONFIGS[@]}"; do
        echo "  - $cfg"
    done
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
    SBATCH_ARGS=()
    SCRIPT_ARGS=()
else
    # Collect config files, sbatch options, and script options
    CONFIGS=()
    SBATCH_ARGS=()      # Options for sbatch (e.g., --partition, --qos)
    SCRIPT_ARGS=()      # Options for Python script (e.g., --force, --methods)
    
    SKIP_NEXT=false
    for arg in "$@"; do
        if [ "$SKIP_NEXT" = true ]; then
            # This is a value for a script flag (e.g., methods list)
            SCRIPT_ARGS+=("$arg")
            SKIP_NEXT=false
            continue
        fi
        
        if [[ "$arg" == *.yaml ]]; then
            CONFIGS+=("$arg")
        elif [[ "$arg" == --* ]]; then
            # Check if this is a script flag
            IS_SCRIPT_FLAG=false
            for sf in "${SCRIPT_FLAGS[@]}"; do
                if [[ "$arg" == "$sf" || "$arg" == "$sf="* ]]; then
                    IS_SCRIPT_FLAG=true
                    break
                fi
            done
            
            if [ "$IS_SCRIPT_FLAG" = true ]; then
                SCRIPT_ARGS+=("$arg")
                # Check if this flag expects a value
                if [[ "$arg" == "--methods" ]]; then
                    SKIP_NEXT=true
                fi
            else
                SBATCH_ARGS+=("$arg")
            fi
        else
            SBATCH_ARGS+=("$arg")
        fi
    done
    
    # If no yaml files found in args, use default configs
    if [ ${#CONFIGS[@]} -eq 0 ]; then
        echo "No config specified, using default configs with flags: ${SCRIPT_ARGS[*]}"
        CONFIGS=("${DEFAULT_CONFIGS[@]}")
    fi
fi

# Fixed resource allocation for rerun (same parallelism as benchmark)
N_CORES=32       # Same as benchmark
SLURM_MEMORY="500G"  # Large memory limit to avoid OOM on big datasets

# Process each config file
for CONFIG_FILE in "${CONFIGS[@]}"; do
    # Resolve config path
    if [ -f "$CONFIG_FILE" ]; then
        CONFIG_PATH="$CONFIG_FILE"
    elif [ -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
        CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"
    else
        echo "ERROR: Config file not found: $CONFIG_FILE"
        continue
    fi

    # Extract dataset name for job name
    DATASET_NAME=$(basename "$CONFIG_FILE" .yaml)

    # Set job time limit - Scanpy can be slow, allow up to 5 days
    JOB_TIME_LIMIT="5-00:00:00"

    # Generate timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOGS_DIR="$PROJECT_ROOT/benchmarking/logs"
    mkdir -p "$LOGS_DIR"

    # Log file names
    LOG_OUT="$LOGS_DIR/rerun_scanpy_${DATASET_NAME}_${TIMESTAMP}.out"
    LOG_ERR="$LOGS_DIR/rerun_scanpy_${DATASET_NAME}_${TIMESTAMP}.err"

    echo "========================================"
    echo "Submitting Rerun Scanpy job"
    echo "========================================"
    echo "Config: $CONFIG_FILE"
    echo "Job time limit: $JOB_TIME_LIMIT"
    echo "SLURM Memory: ${SLURM_MEMORY}"
    echo "CPUs: $N_CORES"
    echo "Log: rerun_scanpy_${DATASET_NAME}_${TIMESTAMP}.out/err"
    [ ${#SCRIPT_ARGS[@]} -gt 0 ] && echo "Script args: ${SCRIPT_ARGS[*]}"
    echo "========================================"

    # Submit job (no time/memory limits, 32 cores same as benchmark)
    SCRIPT_ARGS_STR="${SCRIPT_ARGS[*]}"
    sbatch \
        --job-name="rs-$DATASET_NAME" \
        --time="$JOB_TIME_LIMIT" \
        --mem="$SLURM_MEMORY" \
        --cpus-per-task="$N_CORES" \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        --export=ALL,PRESET_TIME="$JOB_TIME_LIMIT",PRESET_MEM="$SLURM_MEMORY" \
        "${SBATCH_ARGS[@]}" \
        "$SLURM_SCRIPT" "$CONFIG_FILE" "$SCRIPT_ARGS_STR"
    
    echo ""
done
