#!/bin/bash
# Rerun Scanpy Methods Script
# Runs Scanpy QC, t-test, and Wilcoxon without time/memory limits
#
# Use this script AFTER running benchmarks to extract Scanpy outputs
# for datasets where Scanpy timed out or ran out of memory.
# Reports are automatically regenerated with updated accuracy comparisons.
#
# Key features:
# - No time limits, no memory limits
# - Outputs to same locations as regular benchmarks (de/, preprocessing/)
# - Does NOT affect .benchmark_cache (preserves benchmark integrity)
# - Automatically regenerates benchmark reports
#
# Usage:
#   ./run_rerun_scanpy.sh config/Adamson.yaml
#   ./run_rerun_scanpy.sh --methods scanpy_de_t_test config/Adamson.yaml
#   ./run_rerun_scanpy.sh config/*.yaml

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_ENV="${PYTHON_ENV:-$(command -v python)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] config.yaml [config2.yaml ...]"
    echo ""
    echo "Rerun Scanpy methods without resource limits for accuracy comparison."
    echo "Run this AFTER benchmarks complete to get Scanpy outputs for datasets"
    echo "where Scanpy timed out or ran out of memory."
    echo ""
    echo "Options:"
    echo "  --methods METHOD [METHOD ...]  Methods to run (default: all Scanpy methods)"
    echo "  --force                        Force re-run even if output exists"
    echo "  --no-report                    Skip regenerating benchmark reports"
    echo "  --quiet                        Suppress progress output"
    echo "  -h, --help                     Show this help message"
    echo ""
    echo "Available methods:"
    echo "  scanpy_qc_filtered    - Scanpy QC filtering"
    echo "  scanpy_de_t_test      - Scanpy t-test differential expression"
    echo "  scanpy_de_wilcoxon    - Scanpy Wilcoxon differential expression"
    echo ""
    echo "Examples:"
    echo "  $0 config/Adamson.yaml                          # Run all Scanpy methods"
    echo "  $0 --methods scanpy_de_t_test config/Adamson.yaml  # Run specific method"
    echo "  $0 --force config/Adamson.yaml                  # Force re-run"
    echo "  $0 --no-report config/Adamson.yaml              # Skip report regeneration"
    echo "  $0 config/*.yaml                                # Process multiple configs"
    echo ""
    echo "Note: This script does NOT affect the benchmark cache."
    echo "      Outputs are saved to the same locations as regular benchmarks."
    echo "      Reports are automatically regenerated unless --no-report is used."
}

# Parse arguments
CONFIGS=()
EXTRA_ARGS=()
METHODS=()
IN_METHODS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            IN_METHODS=true
            shift
            ;;
        --force)
            EXTRA_ARGS+=("--force")
            IN_METHODS=false
            shift
            ;;
        --no-report)
            EXTRA_ARGS+=("--no-report")
            IN_METHODS=false
            shift
            ;;
        --quiet)
            EXTRA_ARGS+=("--quiet")
            IN_METHODS=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            log "ERROR: Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [ "$IN_METHODS" = true ]; then
                # Check if this looks like a method name (no path separators, no .yaml)
                if [[ "$1" != *"/"* && "$1" != *".yaml"* && "$1" != *".yml"* ]]; then
                    METHODS+=("$1")
                else
                    # This is a config file, not a method
                    IN_METHODS=false
                    CONFIGS+=("$1")
                fi
            else
                CONFIGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Add methods to extra args if specified
if [ ${#METHODS[@]} -gt 0 ]; then
    EXTRA_ARGS+=("--methods" "${METHODS[@]}")
fi

# Check if config files provided
if [ ${#CONFIGS[@]} -eq 0 ]; then
    log "ERROR: No config files provided"
    show_usage
    exit 1
fi

# Start logging
log "=========================================="
log "Rerun Scanpy Methods Started"
log "=========================================="
log "Script directory: $SCRIPT_DIR"
log "Project root: $PROJECT_ROOT"
log "Number of datasets: ${#CONFIGS[@]}"
if [ ${#METHODS[@]} -gt 0 ]; then
    log "Methods to run: ${METHODS[*]}"
else
    log "Methods to run: all Scanpy methods"
fi
log ""

# Convert relative paths to absolute paths
ABS_CONFIGS=()
for arg in "${CONFIGS[@]}"; do
    if [[ "$arg" = /* ]]; then
        ABS_CONFIGS+=("$arg")
    else
        ABS_CONFIGS+=("$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")")
    fi
done

# Change to project root
cd "$PROJECT_ROOT"

# Track results
TOTAL_CONFIGS=${#ABS_CONFIGS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_DATASETS=()

# Process each config file
CURRENT=0
for CONFIG_FILE in "${ABS_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    log ""
    log "=========================================="
    log "Dataset $CURRENT/$TOTAL_CONFIGS: $(basename "$CONFIG_FILE")"
    log "=========================================="
    
    # Check if config exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log "ERROR: Config file not found: $CONFIG_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_DATASETS+=("$(basename "$CONFIG_FILE") - File not found")
        continue
    fi
    
    # Create dataset-specific log
    DATASET_NAME=$(basename "$CONFIG_FILE" .yaml)
    DATASET_LOG="${LOG_DIR}/rerun_scanpy_${DATASET_NAME}_${TIMESTAMP}.log"
    
    log "Config: $CONFIG_FILE"
    log "Log: $DATASET_LOG"
    
    # Run rerun_scanpy
    if "$PYTHON_ENV" -m benchmarking.tools.rerun_scanpy \
        --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$DATASET_LOG"; then
        log "✓ Rerun completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        log "ERROR: Rerun failed with exit code: $EXIT_CODE"
        log "ERROR: Check detailed log: $DATASET_LOG"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_DATASETS+=("$(basename "$CONFIG_FILE") - Exit code $EXIT_CODE")
    fi
done

# Final summary
log ""
log "=========================================="
log "Rerun Scanpy Completed"
log "=========================================="
log "Total datasets: $TOTAL_CONFIGS"
log "Successful: $SUCCESS_COUNT"
log "Failed: $FAILED_COUNT"

if [ $FAILED_COUNT -gt 0 ]; then
    log ""
    log "Failed datasets:"
    for failed in "${FAILED_DATASETS[@]}"; do
        log "  ✗ $failed"
    done
fi

log ""
log "Scanpy outputs: benchmarking/results/<dataset>/de/ and preprocessing/"
log "Reports: benchmarking/results/<dataset>/benchmark_report.md"
log "Logs: ${LOG_DIR}/rerun_scanpy_*.log"
log ""
log "=========================================="

# Exit with error if any failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
