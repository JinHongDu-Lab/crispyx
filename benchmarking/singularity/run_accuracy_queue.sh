#!/bin/bash
# Run accuracy-only reruns on this machine (e.g. hydra) in the benchmark container.
# Reads lines "<dataset> <method> [<method> ...]" from a queue file; datasets run in
# parallel (PARALLEL, default 3), a dataset's methods one after another.
#   PARALLEL=3 bash benchmarking/singularity/run_accuracy_queue.sh queue.txt
# Inputs: benchmarking/accuracy_inputs/<D>/ (tools/extract_accuracy_inputs.py on HPC).
# Outputs: $OUT/<D>/ (default /data/projects/Streamlining-CRISPR-Screen-Analysis/accuracy_hydra),
# synced back into HPC results/<D>/ afterwards.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-/data/projects/Streamlining-CRISPR-Screen-Analysis/accuracy_hydra}"
mkdir -p "$OUT/logs"
run_line() {
    set -- $1
    local d=$1; shift
    singularity exec --bind "$REPO:/workspace" --bind /data/projects:/data/projects --pwd /workspace \
        --env PYTHONPATH=/workspace/src:/workspace --env PYTHONUNBUFFERED=1 \
        "$REPO/benchmarking/singularity/crispyx-benchmark.sif" \
        python -m benchmarking.tools.run_accuracy_reruns "benchmarking/config/$d.yaml" --out "$OUT" --methods "$@" \
        > "$OUT/logs/${d}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    echo "$(date '+%F %T') $d done (exit $?): $*"
}
export -f run_line; export REPO OUT
grep -v '^\s*#' "$1" | grep . | xargs -P "${PARALLEL:-3}" -I{} bash -c 'run_line "{}"'
