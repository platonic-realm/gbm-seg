#!/bin/bash
#SBATCH --job-name=gbm_render
#SBATCH --output=/home/afatehi/gbm_log/render.log
#SBATCH --error=/home/afatehi/gbm_log/render.log
#SBATCH --time=7-00:00:00   # Adjust time as needed
#SBATCH --partition=train   # Adjust partition based on available resources
#SBATCH --gres=gpu:V100:4        # Request 4 GPU
#SBATCH --mem=320GB         # Adjust memory as needed
#SBATCH --cpus-per-task=64  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/gbm_log

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py render $1 -it $2
