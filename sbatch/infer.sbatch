#!/bin/bash
#SBATCH --job-name=gbm_infer
#SBATCH --output=/home/afatehi/logs/gbm/infer.log
#SBATCH --error=/home/afatehi/logs/gbm/infer.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=train
# The parent blocks until its array finishes (see lib_array_wait.sh), so
# its walltime must comfortably exceed the array children's 1-day limit.

# Per-volume parallel inference. One SLURM array task = one test volume =
# one GPU, so the volumes infer concurrently instead of one process
# looping them serially. Mirrors morph.sbatch's parent->array pattern.
#
# Args: $1 exp  $2 snapshot  $3 batch  $4 sample_dim  $5 stride
#       $6 scale_factor  $7 interpolation (true/false)

mkdir -p /home/afatehi/logs/gbm/
source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Parent phase: enumerate the volumes, submit the array ---
    # venv + cd already done above so the python yaml read below works.
    ROOT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('./configs/template.yaml'))['experiments']['root'])")
    DS_DIR="${ROOT_PATH}/$1/datasets/ds_test_unlabeled"
    if [ ! -d "$DS_DIR" ]; then
        echo "Error: inference set $DS_DIR does not exist"
        exit 1
    fi

    VOLS=()
    while IFS= read -r -d '' f; do
        VOLS+=("$(basename "$f")")
    done < <(find "$DS_DIR" -maxdepth 1 -type f \
                  \( -name '*.tiff' -o -name '*.tif' \) -print0 | sort -z)

    N=${#VOLS[@]}
    if [ "$N" -eq 0 ]; then
        echo "Error: no .tif/.tiff volumes in $DS_DIR"
        exit 1
    fi
    echo "Submitting inference array for $N volume(s):"
    printf '  %s\n' "${VOLS[@]}"

    # Pack the volume list into one env var (| delimiter, safe for spaces).
    VOLS_STR=$(printf "%s|" "${VOLS[@]}")
    VOLS_STR=${VOLS_STR%|}

    mkdir -p /home/afatehi/logs/gbm/infer

    # One A100 GPU per task. --array=1-N runs all volumes concurrently
    # (SLURM queues them as GPUs free up).
    ARRAY_JID=$(sbatch --parsable \
           --gres=gpu:A100:1 --cpus-per-task=16 --mem=120G \
           --time=1-00:00:00 --partition=train \
           --array=1-"${N}" \
           --output=/home/afatehi/logs/gbm/infer/job_%A_%a.log \
           --error=/home/afatehi/logs/gbm/infer/job_%A_%a.log \
           --export=ALL,VOLS="$VOLS_STR" "$0" "$@")
    echo "Submitted inference array job $ARRAY_JID"

    # Block until the array finishes so an afterok dependent (run_pipeline.sh)
    # fires only after every volume is actually inferred.
    source sbatch/lib_array_wait.sh
    wait_for_array "$ARRAY_JID"
    exit $?
fi

# --- Child array phase: infer exactly one volume ---
IFS='|' read -r -a VOLS <<< "$VOLS"
VOL="${VOLS[$((SLURM_ARRAY_TASK_ID - 1))]}"

if [ -z "$VOL" ]; then
    echo "Error: no volume for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Array task $SLURM_ARRAY_TASK_ID inferring volume: $VOL"
# -sn picks the single volume; -f makes infer non-interactive. $7 must be
# `true` when the model trained on Z-upsampled data so InferenceDataset
# upsamples to match — see ds_infer.py:43 (needs scale_factor>1 AND interp).
./gbm.py infer "$1" -s "$2" -bs "$3" -sd "$4" -st "$5" -sf "$6" \
    -in "$7" -sn "$VOL" -f
