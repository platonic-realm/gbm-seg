#!/bin/bash
#SBATCH --job-name=gbm_continuity
#SBATCH --output=/home/afatehi/logs/gbm/continuity.log
#SBATCH --error=/home/afatehi/logs/gbm/continuity.log
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Z-continuity analysis of inference predictions — quantifies the along-Z
# jaggedness that the validation Dice can't see (the axis ContLoss targets).
# Single-process (cheap numpy Z-diffs over each binary volume); no array.
#
# Usage:  sbatch sbatch/continuity.sbatch <experiment-name> <inference-tag>
#
# Writes <exp>/results-infer/<tag>_continuity/continuity_result.yaml.
EXP_NAME="$1"
TAG="$2"

if [ -z "$EXP_NAME" ] || [ -z "$TAG" ]; then
    echo "Usage: sbatch $0 <experiment-name> <inference-tag>"
    exit 1
fi

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

python gbm.py continuity "$EXP_NAME" -it "$TAG"
