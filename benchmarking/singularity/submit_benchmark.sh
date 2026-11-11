#!/bin/bash
#
# Wrapper script to submit SLURM benchmark jobs with auto-configured resources
#
# Usage:
#   ./submit_benchmark.sh                         # Submit all default configs
#   ./submit_benchmark.sh Adamson_subset.yaml     # Submit single config
#   ./submit_benchmark.sh Adamson.yaml Replogle-GW-k562.yaml  # Submit multiple configs
#
# Resources (time, memory, cores) are automatically read from the config file.

set -e

# Project root (modify for your HPC)
PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"
CONFIG_DIR="$PROJECT_ROOT/benchmarking/config"
SLURM_SCRIPT="$PROJECT_ROOT/benchmarking/singularity/slurm_benchmark.sh"

# Default config files to run when none specified
DEFAULT_CONFIGS=(
    # "Adamson_subset.yaml"
    # "Adamson.yaml"
    # "Feng-gwsf.yaml"
    # "Feng-gwsnf.yaml"
    "Feng-ts.yaml"
    # "Frangieh.yaml"
    # "Nadig-HEPG2.yaml"
    # "Nadig-JURKAT.yaml"
    # "Replogle-E-k562.yaml"
    # "Replogle-E-rpe1.yaml"
    "Replogle-GW-k562.yaml"
    # "Tian-crispra.yaml"
    # "Tian-crispri.yaml"
    # "Huang-HCT116.yaml"
    # "Huang-HEK293T.yaml"
)

# Benchmark-specific flags that should be forwarded to Python script
# (not to sbatch)
BENCHMARK_FLAGS=("--force" "--clean" "--methods")

# If no arguments provided, use default configs
if [ $# -lt 1 ]; then
    echo "No config specified, using default configs: ${DEFAULT_CONFIGS[*]}"
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
    SBATCH_ARGS=()
    BENCHMARK_ARGS=()
else
    # Collect config files, sbatch options, and benchmark options
    CONFIGS=()
    SBATCH_ARGS=()      # Options for sbatch (e.g., --partition, --qos)
    BENCHMARK_ARGS=()   # Options for Python script (e.g., --force, --clean)
    
    SKIP_NEXT=false
    PREV_ARG=""
    for arg in "$@"; do
        if [ "$SKIP_NEXT" = true ]; then
            # This is a value for a benchmark flag (e.g., methods list)
            BENCHMARK_ARGS+=("$arg")
            SKIP_NEXT=false
            continue
        fi
        
        if [[ "$arg" == *.yaml ]]; then
            CONFIGS+=("$arg")
        elif [[ "$arg" == --* ]]; then
            # Check if this is a benchmark flag
            IS_BENCHMARK_FLAG=false
            for bf in "${BENCHMARK_FLAGS[@]}"; do
                if [[ "$arg" == "$bf" || "$arg" == "$bf="* ]]; then
                    IS_BENCHMARK_FLAG=true
                    break
                fi
            done
            
            if [ "$IS_BENCHMARK_FLAG" = true ]; then
                BENCHMARK_ARGS+=("$arg")
                # Check if this flag expects a value (--methods)
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
    
    # If no yaml files found in args, use default configs (allows --force without specifying configs)
    if [ ${#CONFIGS[@]} -eq 0 ]; then
        echo "No config specified, using default configs with flags: ${BENCHMARK_ARGS[*]}"
        CONFIGS=("${DEFAULT_CONFIGS[@]}")
    fi
fi

# Function to extract resource limits from YAML config using Python
read_config() {
    local config_path="$1"
    python3 -c "
import yaml
import sys

with open('$config_path', 'r') as f:
    config = yaml.safe_load(f)

limits = config.get('resource_limits', {})
parallel = config.get('parallel_config', {})

time_limit = limits.get('time_limit', 3600)  # default 1 hour
memory_limit = limits.get('memory_limit', 64)  # default 64 GB
n_cores = parallel.get('n_cores', 8)  # default 8 cores

# Convert time_limit (seconds) to SLURM format (HH:MM:SS)
hours = int(time_limit // 3600)
minutes = int((time_limit % 3600) // 60)
seconds = int(time_limit % 60)
time_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

# Memory in GB (return raw value for SLURM buffer calculation)
print(f'{time_str} {int(memory_limit)} {n_cores}')
"
}

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

    # Parse config
    CONFIG_VALUES=$(read_config "$CONFIG_PATH")
    TIME_LIMIT=$(echo $CONFIG_VALUES | cut -d' ' -f1)
    MEMORY_GB=$(echo $CONFIG_VALUES | cut -d' ' -f2)
    N_CORES=$(echo $CONFIG_VALUES | cut -d' ' -f3)
    
    # Extract dataset name for job name (needed before memory check)
    DATASET_NAME=$(basename "$CONFIG_FILE" .yaml)
    
    # Large datasets need fixed 180GB SLURM allocation
    # Other datasets use config memory + 10% buffer
    # LARGE_DATASETS="Feng-ts Replogle-GW-k562"
    # if [[ " $LARGE_DATASETS " =~ " $DATASET_NAME " ]]; then
    #     SLURM_MEMORY_GB=180
    #     SLURM_MEMORY="180G"
    # else
    #     SLURM_MEMORY_GB=$((MEMORY_GB + MEMORY_GB / 10))
    #     SLURM_MEMORY="${SLURM_MEMORY_GB}G"
    # fi
    SLURM_MEMORY_GB=$((MEMORY_GB + MEMORY_GB / 10))
    SLURM_MEMORY="${SLURM_MEMORY_GB}G"

    # Set job time limit to 1 week (max) - config time_limit is per-method, not total job
    # Multiple methods run sequentially, so we use a fixed max
    JOB_TIME_LIMIT="7-00:00:00"  # 7 days in SLURM format (D-HH:MM:SS)

    # Generate timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOGS_DIR="$PROJECT_ROOT/benchmarking/logs"
    mkdir -p "$LOGS_DIR"

    # Log file names: dataset_YYYYMMDD_HHMMSS.out/.err
    LOG_OUT="$LOGS_DIR/${DATASET_NAME}_${TIMESTAMP}.out"
    LOG_ERR="$LOGS_DIR/${DATASET_NAME}_${TIMESTAMP}.err"

    echo "========================================"
    echo "Submitting benchmark job"
    echo "========================================"
    echo "Config: $CONFIG_FILE"
    echo "Job time limit: $JOB_TIME_LIMIT (per-method limit from config: $TIME_LIMIT)"
    echo "SLURM Memory: ${SLURM_MEMORY}"
    echo "Container Memory Limit: ${MEMORY_GB}G (enforced by Singularity)"
    echo "CPUs: $N_CORES"
    echo "Log: ${DATASET_NAME}_${TIMESTAMP}.out/err"
    [ ${#BENCHMARK_ARGS[@]} -gt 0 ] && echo "Benchmark args: ${BENCHMARK_ARGS[*]}"
    echo "========================================"

    # Submit job with auto-configured resources
    # Pass MEMORY_GB as second argument for Singularity --memory enforcement
    # Pass benchmark args as third argument (quoted to preserve as single arg)
    BENCHMARK_ARGS_STR="${BENCHMARK_ARGS[*]}"
    sbatch \
        --job-name="cx-$DATASET_NAME" \
        --time="$JOB_TIME_LIMIT" \
        --mem="$SLURM_MEMORY" \
        --cpus-per-task="$N_CORES" \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        "${SBATCH_ARGS[@]}" \
        "$SLURM_SCRIPT" "$CONFIG_FILE" "$MEMORY_GB" "$BENCHMARK_ARGS_STR"
    
    echo ""
done
