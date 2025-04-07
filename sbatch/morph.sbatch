#!/bin/bash
#SBATCH --job-name=gbm_morph
#SBATCH --output=/home/afatehi/gbm_morph.log
#SBATCH --error=/home/afatehi/gbm_morph.err
# (No GPU request here so the parent job does not get a GPU)

# Check if this is the parent job (no SLURM_ARRAY_TASK_ID defined)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Parent Phase ---
    # Parse the YAML file to extract the root path (requires PyYAML)
    ROOT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('../configs/template.yaml'))['experiments']['root'])")

    # Combine the root path with the argument passed to the script.
    FULL_PATH="${ROOT_PATH}/$1/results-infer/$2/"

    # List subdirectories under FULL_PATH.
    mapfile -t DIRS < <(find "${FULL_PATH}" -mindepth 1 -maxdepth 1 -type d)

    # Extract directory names from paths
    mapfile -t DIR_NAMES < <(basename -a "${DIRS[@]}")

    # Get the number of directories found.
    N=${#DIRS[@]}

    # Convert DIR_NAMES into a single string with a safe delimiter (e.g., `|`)
    DIR_NAMES_STR=$(printf "%s|" "${DIR_NAMES[@]}")
    DIR_NAMES_STR=${DIR_NAMES_STR%|}  # Remove trailing separator

    # Submit a job array with DIR_NAMES properly passed
    sbatch -p train -c 8 --mem=32G --gres=gpu:1 --time=7-00:00:00 \
        --array=1-${N} --export=ALL,DIR_NAMES="$DIR_NAMES_STR" "$0" "$@"

    exit
fi

# --- Child Array Job Phase ---
# Redirect output to /dev/null for child jobs.
# exec > /dev/null 2>&1

# Convert the exported DIR_NAMES string back into an array (split by `|`)
IFS='|' read -r -a DIR_NAMES <<< "$DIR_NAMES"

# Get the directory corresponding to the current array task.
SUBDIR="${DIR_NAMES[$((SLURM_ARRAY_TASK_ID-1))]}"

# Debugging: Print to check value (can be removed)
echo "DEBUG: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "DEBUG: SUBDIR=$SUBDIR"

# Ensure SUBDIR is not empty before running the command
if [ -n "$SUBDIR" ]; then
    echo "Processing directory: $SUBDIR"
    source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
    ./gbm.py morph "$1" -it "$2" -sn "$SUBDIR"
else
    echo "Error: SUBDIR is empty!"
    exit 1
fi
