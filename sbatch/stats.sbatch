#!/bin/bash
#SBATCH --job-name=gbm_stats
#SBATCH --output=/home/afatehi/logs/gbm/stats.log
#SBATCH --error=/home/afatehi/logs/gbm/stats.log
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Parallel statistical analysis.
#
# Usage:  sbatch sbatch/stats.sbatch <experiment-name> <inference-tag> [--clipping]
#
# The per-sample stats stage (load + mask + outlier removal + per-sample
# plots + censored-MLE) is ~50 s/sample and embarrassingly parallel, so
# this parent submits a SLURM ARRAY — one task per sample — exactly like
# morph.sbatch. After the array finishes, a single REDUCE job aggregates
# the per-sample sidecars into the cohort outputs + publication figures.
# 15 samples drop from ~14 min serial to ~1 min wall-clock.

mkdir -p /home/afatehi/logs/gbm/

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Parent phase ---
    EXP_NAME="$1"
    TAG="$2"
    CLIP="$3"   # optional "--clipping"
    if [ -z "$EXP_NAME" ] || [ -z "$TAG" ]; then
        echo "Usage: sbatch $0 <experiment-name> <inference-tag> [--clipping]"
        exit 1
    fi

    ROOT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('./configs/template.yaml'))['experiments']['root'])")
    FULL_PATH="${ROOT_PATH}/${EXP_NAME}/results-infer/${TAG}/"
    if [ ! -d "$FULL_PATH" ]; then
        echo "Error: Directory $FULL_PATH does not exist"
        exit 1
    fi

    # Collect sample subdirectory names (null-delimited, sorted).
    DIR_NAMES=()
    while IFS= read -r -d '' dir; do
        DIR_NAMES+=("$(basename "$dir")")
    done < <(find "${FULL_PATH}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    N=${#DIR_NAMES[@]}
    if [ "$N" -eq 0 ]; then
        echo "Error: No sample subdirectories found in $FULL_PATH"
        exit 1
    fi
    echo "Found $N samples to process in parallel."

    DIR_NAMES_STR=$(printf "%s|" "${DIR_NAMES[@]}")
    DIR_NAMES_STR=${DIR_NAMES_STR%|}

    mkdir -p /home/afatehi/logs/gbm/stats

    # Per-sample array. Each task needs enough RAM for one full-volume
    # thickness array + the COL4 channel of one prediction.tif.
    ARRAY_JID=$(sbatch --parsable \
        -p cpu -c 4 --mem=64G --time=06:00:00 \
        --array=1-${N} \
        --output=/home/afatehi/logs/gbm/stats/job_%A_%a.log \
        --error=/home/afatehi/logs/gbm/stats/job_%A_%a.log \
        --export=ALL,DIR_NAMES="$DIR_NAMES_STR",STATS_TAG="$TAG",STATS_EXP="$EXP_NAME",STATS_CLIP="$CLIP" \
        "$0" "$@")
    echo "Submitted stats array job $ARRAY_JID"

    # Block until the array completes, then run the reduce step.
    source sbatch/lib_array_wait.sh
    wait_for_array "$ARRAY_JID"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "Stats array did not finish cleanly (rc=$rc); skipping reduce."
        exit $rc
    fi

    echo "Array complete — running reduce."
    source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
    ./gbm.py stats-reduce "$EXP_NAME" -it "$TAG" $CLIP
    exit $?
fi

# --- Child array task phase ---
IFS='|' read -r -a DIR_NAMES <<< "$DIR_NAMES"
if [ "$SLURM_ARRAY_TASK_ID" -gt "${#DIR_NAMES[@]}" ]; then
    echo "Error: array task $SLURM_ARRAY_TASK_ID exceeds ${#DIR_NAMES[@]} samples"
    exit 1
fi
SUBDIR="${DIR_NAMES[$((SLURM_ARRAY_TASK_ID-1))]}"
if [ -z "$SUBDIR" ]; then
    echo "Error: empty SUBDIR for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Per-sample stats: $SUBDIR"
source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py stats "$STATS_EXP" -it "$STATS_TAG" -sn "$SUBDIR" $STATS_CLIP
