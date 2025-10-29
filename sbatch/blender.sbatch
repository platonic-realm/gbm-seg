#!/bin/bash
#SBATCH --job-name=gbm_blender
#SBATCH --output=/home/afatehi/gbm_blender.log
#SBATCH --error=/home/afatehi/gbm_blender.err
# (No GPU request here so the parent job does not get a GPU)

# Check if this is the parent job (no SLURM_ARRAY_TASK_ID defined)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Parent Phase ---
    # Parse the YAML file to extract the root path (requires PyYAML)
    ROOT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('./configs/template.yaml'))['experiments']['root'])")

    # Combine the root path with the argument passed to the script.
    FULL_PATH="${ROOT_PATH}/$1/results-infer/$2/"

    # Check if path exists
    if [ ! -d "$FULL_PATH" ]; then
        echo "Error: Directory $FULL_PATH does not exist"
        exit 1
    fi

    # Build DIR_NAMES array more reliably using null-delimited input
    DIR_NAMES=()
    while IFS= read -r -d '' dir; do
        DIR_NAMES+=("$(basename "$dir")")
    done < <(find "${FULL_PATH}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    # Get the number of directories found.
    N=${#DIR_NAMES[@]}
    
    if [ "$N" -eq 0 ]; then
        echo "Error: No subdirectories found in $FULL_PATH"
        exit 1
    fi

    echo "Found $N directories to process:"
    printf '%s\n' "${DIR_NAMES[@]}"

    # Convert DIR_NAMES into a single string with a safe delimiter (e.g., `|`)
    DIR_NAMES_STR=$(printf "%s|" "${DIR_NAMES[@]}")
    DIR_NAMES_STR=${DIR_NAMES_STR%|}  # Remove trailing separator

    # Submit a job array with DIR_NAMES properly passed
    sbatch -p cpu -c 2 --mem=16G --time=1-00:00:00 \
        --array=1-${N} --export=ALL,DIR_NAMES="$DIR_NAMES_STR" "$0" "$@"

    exit
fi

# --- Child Array Job Phase ---
# Redirect output to /dev/null for child jobs.
# exec > /dev/null 2>&1

# Convert the exported DIR_NAMES string back into an array (split by `|`)
IFS='|' read -r -a DIR_NAMES <<< "$DIR_NAMES"

# Validate array bounds
if [ "$SLURM_ARRAY_TASK_ID" -gt "${#DIR_NAMES[@]}" ]; then
    echo "Error: Array task ID $SLURM_ARRAY_TASK_ID exceeds number of directories ${#DIR_NAMES[@]}"
    exit 1
fi

# Get the directory corresponding to the current array task.
SUBDIR="${DIR_NAMES[$((SLURM_ARRAY_TASK_ID-1))]}"

# Debugging: Print to check value (can be removed)
echo "DEBUG: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "DEBUG: SUBDIR=$SUBDIR"
echo "DEBUG: Total directories in array: ${#DIR_NAMES[@]}"

# Ensure SUBDIR is not empty before running the command
if [ -n "$SUBDIR" ]; then
    echo "Processing directory: $SUBDIR"
    source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
    ./gbm.py blender "$1" -it "$2" -sn "$SUBDIR"
else
    echo "Error: SUBDIR is empty for task ID $SLURM_ARRAY_TASK_ID!"
    exit 1
fi
