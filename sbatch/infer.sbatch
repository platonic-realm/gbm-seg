#!/bin/bash
#SBATCH --job-name=gbm_infer
#SBATCH --output=/home/afatehi/gbm_log/infer.log
#SBATCH --error=/home/afatehi/gbm_log/infer.log
#SBATCH --time=7-00:00:00   # Adjust time as needed
#SBATCH --partition=train   # Adjust partition based on available resources
#SBATCH --gres=gpu:A100:4  # Request 8 GPU
#SBATCH --mem=320G          # Adjust memory as needed
#SBATCH --cpus-per-task=64  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/gbm_log

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py infer $1 -s $2 -bs $3 -sd "$4" -st "$5" -sf $6

