#!/bin/bash
#SBATCH --partition=amd
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=140G
#
# Re-time the crispyx methods of one dataset on an exclusive node.
#
# Shared-node contention changed runtimes by up to ~4x (Feng-gwsf NB-GLM:
# 19,274 s shared vs ~4,500 s exclusive), so the runtime figures use these
# timings. Results go to benchmarking/results_timing/<dataset>; the real
# results directory is not touched. The standardized inputs (and NB-GLM's
# sorted copy, made by slurm_presort.sh) are symlinked in, so nothing large
# is recomputed outside the timed methods. Large outputs are deleted at the
# end; results.csv and .benchmark_cache are kept.
#
# Usage:
#   CRISPYX_SRC=$PROJECT_ROOT/crispyx-0.1.8/src \
#     sbatch --export=ALL,DATASET=VCC --job-name=retime-slot1 \
#            --dependency=singleton slurm_retime_crispyx.sh
# METHODS=<space-separated list> re-times a subset (default: all crispyx steps).
# STAGE_LOCAL=1 runs on the node-local SSD where the dataset fits (see below).
# KEEP_OUTPUTS=1 keeps the method outputs in results_timing/<dataset>.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"
BENCH="$PROJECT_ROOT/benchmarking"
D="${DATASET:?set DATASET}"
SRC_DIR="$(readlink -f "$BENCH/results/$D")"
T="$BENCH/results_timing/$D"
METHODS="${METHODS:-crispyx_qc_filtered crispyx_preprocess crispyx_pb_avg_log crispyx_pb_pseudobulk crispyx_de_t_test crispyx_de_wilcoxon crispyx_de_nb_glm}"

echo "Re-timing crispyx on $D (job $SLURM_JOB_ID on $SLURMD_NODENAME, exclusive)"
lscpu | grep -E "Model name|^CPU\(s\)"

mkdir -p "$T/.cache" "$T/preprocessing" "$BENCH/config/timing"
INPUTS=()
for f in "$SRC_DIR"/.cache/standardized_"$D".h5ad "$SRC_DIR"/.cache/standardized_"$D"_sorted.h5ad \
         "$SRC_DIR"/preprocessing/standardized_csr_"$D".h5ad "$SRC_DIR"/preprocessing/standardized_csr_"$D"_sorted.h5ad; do
    # Relative links: the container mounts the project at /workspace, so
    # absolute /lustre1 targets would dangle in there.
    [ -e "$f" ] && ln -sfnr "$f" "$T/${f#"$SRC_DIR"/}" && INPUTS+=("$(readlink -f "$f")")
done
# NB-GLM reads the QC-filtered counts (and their sorted copy, made by
# NBGLM_INPUT=qc slurm_presort.sh). Linked only when QC itself is not re-timed:
# crispyx_qc_filtered would otherwise write through the link into results/.
case " $METHODS " in
    *" crispyx_qc_filtered "*) ;;
    *) for f in "$SRC_DIR"/preprocessing/crispyx_qc_filtered.h5ad "$SRC_DIR"/preprocessing/crispyx_qc_filtered_sorted.h5ad; do
           [ -e "$f" ] && ln -sfnr "$f" "$T/${f#"$SRC_DIR"/}" && INPUTS+=("$(readlink -f "$f")")
       done ;;
esac
# Same config, results redirected to the timing directory.
sed "s#^output_dir: .*#output_dir: \"benchmarking/results_timing/$D\"#" \
    "$BENCH/config/$D.yaml" > "$BENCH/config/timing/$D.yaml"

# STAGE_LOCAL=1: run with inputs and outputs on the node-local SSD ($TMPDIR),
# so I/O-bound steps are timed against a local disk rather than shared
# Lustre. Inputs are copied with direct I/O (no page cache charged to the
# job), keep size and mtime, and are bind-mounted at their real paths, so
# provenance checks (e.g. NB-GLM's sorted copy) still match. The timing
# directory itself is a local directory bind-mounted over $T; only the small
# files are copied back. Datasets that do not fit run on Lustre as before.
LAYOUT=lustre
if [ "${STAGE_LOCAL:-0}" = 1 ]; then
    # crispyx >= 0.2 standardized files link the raw matrix (a repo-relative
    # config dataset_path, e.g. Huang) instead of copying it; stage that file
    # too. Absolute /data/origin paths are older full-copy standardizations.
    RAW="$(sed -n 's/^dataset_path: *"\{0,1\}\([^"# ]*\).*/\1/p' "$BENCH/config/$D.yaml")"
    case "$RAW" in /*) ;; *) INPUTS+=("$(readlink -f "$PROJECT_ROOT/$RAW")") ;; esac
    IN_BYTES=0; for f in "${INPUTS[@]}"; do IN_BYTES=$(( IN_BYTES + $(stat -c %s "$f") )); done
    # Outputs: the sizes of the same crispyx outputs in the real results.
    OUT_GLOB=(\( -name 'crispyx_*.h5ad' -o -name 'preprocessed_*.h5ad' -o -name 'pb_*.h5ad' -o -name 'de_cx_*' \))
    [ "$METHODS" = crispyx_de_nb_glm ] && OUT_GLOB=(-name 'crispyx_de_nb_glm.h5ad')
    OUT_BYTES=$(find -L "$SRC_DIR/preprocessing" "$SRC_DIR/de" -maxdepth 1 -type f "${OUT_GLOB[@]}" \
        -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    NEED=$(( IN_BYTES + OUT_BYTES * 12 / 10 + 10 * 1024**3 ))
    AVAIL=$(df --output=avail -B1 "${TMPDIR:-/tmp}" | tail -1)
    echo "Local staging: inputs $((IN_BYTES/1024**3)) GB + outputs ~$((OUT_BYTES/1024**3)) GB; need $((NEED/1024**3)) GB, $((AVAIL/1024**3)) GB free"
    if [ "$NEED" -le "$AVAIL" ]; then
        LAYOUT=local
        L="${TMPDIR:-/tmp}/retime_$D"
        trap 'rm -rf "$L"' EXIT
        mkdir -p "$L/in" "$L/out"
        EXTRA_BINDS=""
        T0=$(date +%s)
        for f in "${INPUTS[@]}"; do
            c="$L/in/$(basename "$f")"
            dd if="$f" of="$c" bs=64M iflag=direct oflag=direct status=none
            touch -r "$f" "$c"
            if [ "$(stat -c '%s %Y' "$f")" != "$(stat -c '%s %Y' "$c")" ]; then
                echo "ERROR: staged copy of $f differs in size or mtime"; exit 1
            fi
            EXTRA_BINDS="$EXTRA_BINDS --bind $c:/workspace/${f#"$PROJECT_ROOT"/}:ro"
        done
        echo "Staged ${#INPUTS[@]} input(s) in $(( $(date +%s) - T0 )) s"
        # The same links, in the local timing directory.
        (cd "$T" && find . -type l) | while read -r rel; do
            mkdir -p "$L/out/$(dirname "$rel")"
            ln -sfn "$(readlink "$T/$rel")" "$L/out/$rel"
        done
        export EXTRA_BINDS="$EXTRA_BINDS --bind $L/out:/workspace/benchmarking/results_timing/$D"
    else
        echo "Does not fit on local disk; running on Lustre"
    fi
fi
echo "layout=$LAYOUT" > "$T/staging.txt"

bash "$BENCH/singularity/slurm_benchmark.sh" "timing/$D.yaml" 128 "--methods $METHODS"

if [ "${KEEP_OUTPUTS:-0}" = 1 ]; then
    # Keep the outputs too (to check them against earlier results); the
    # input links are kept as well.
    [ "$LAYOUT" = local ] && rsync -rt "$L/out/" "$T/"
else
    if [ "$LAYOUT" = local ]; then
        # Bring back the timings (regular files only; outputs stay behind).
        rsync -rt --exclude '*.h5ad' --exclude 'de/*.csv' "$L/out/" "$T/"
    fi
    # Keep the timings, drop the large outputs (real files only; links go too).
    find "$T" -name "*.h5ad" \( -type f -o -type l \) -delete
    find "$T" -name "*.csv" -path "*/de/*" -delete
fi
rm -f "$BENCH/config/timing/$D.yaml"
echo "Kept: $(ls -A "$T")"
