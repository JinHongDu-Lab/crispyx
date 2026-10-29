#!/bin/bash
# Run accuracy-only reruns on this machine (e.g. hydra) in the benchmark container.
# Reads lines "<dataset> <method> [<method> ...]" from a queue file; lines run one after
# another (PARALLEL, default 1: large datasets and edgeR need it to stay clear of OOM),
# a dataset's methods one after another. Optional MEMORY_MAX (e.g. 240G) runs the queue in
# a systemd user scope with that hard memory limit.
#   MEMORY_MAX=240G bash benchmarking/singularity/run_accuracy_queue.sh queue.txt
# Inputs: benchmarking/accuracy_inputs/<D>/ (tools/extract_accuracy_inputs.py on HPC).
# Outputs: $OUT/<D>/ (default /data/projects/Streamlining-CRISPR-Screen-Analysis/accuracy_hydra),
# synced back into HPC results/<D>/ afterwards.
set -u
if [ -n "${MEMORY_MAX:-}" ] && [ -z "${IN_MEMORY_SCOPE:-}" ]; then
    exec systemd-run --user --scope -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
        env IN_MEMORY_SCOPE=1 PARALLEL="${PARALLEL:-1}" OUT="${OUT:-}" bash "$0" "$@"
fi
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
grep -v '^\s*#' "$1" | grep . | xargs -P "${PARALLEL:-1}" -I{} bash -c 'run_line "{}"'
