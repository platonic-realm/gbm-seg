#!/bin/bash
#SBATCH --job-name=gbm_train
#SBATCH --output=/home/afatehi/logs/gbm/train_%x_%A_fold%a.log
#SBATCH --error=/home/afatehi/logs/gbm/train_%x_%A_fold%a.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=train
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=64

# Args: $1 = experiment name, $2 = fold index (0..k-1)
EXP_NAME="$1"
FOLD="$2"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
python gbm.py train "$EXP_NAME" --fold "$FOLD"
