#!/bin/bash
#SBATCH --job-name=gbm_aggregate_cv
#SBATCH --output=/home/afatehi/logs/gbm/aggregate_cv_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/aggregate_cv_%x_%A.log
#SBATCH --time=00:30:00
#SBATCH --partition=infer
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Args: $1 = experiment name. Run after all per-fold train jobs finish to
# consolidate <exp>/results-train/fold_*/best_metrics.yaml into
# <exp>/cv_results.{yaml,npz} and post a W&B cv_summary run.
EXP_NAME="$1"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
python gbm.py aggregate-cv "$EXP_NAME"
