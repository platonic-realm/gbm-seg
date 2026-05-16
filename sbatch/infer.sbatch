#!/bin/bash
#SBATCH --job-name=gbm_infer
#SBATCH --output=/home/afatehi/logs/gbm/infer.log
#SBATCH --error=/home/afatehi/logs/gbm/infer.log
#SBATCH --time=7-00:00:00   # Adjust time as needed
#SBATCH --partition=train   # Adjust partition based on available resources
#SBATCH --gres=gpu:A100:4  # Request 8 GPU
#SBATCH --mem=320G          # Adjust memory as needed
#SBATCH --cpus-per-task=64  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
# Args: $1 exp  $2 snapshot  $3 batch  $4 sample_dim  $5 stride
#       $6 scale_factor  $7 interpolation (true/false)
# -f makes infer non-interactive (SLURM-safe). $7 must be `true` when the
# model was trained on Z-upsampled data so InferenceDataset upsamples to
# match — see ds_infer.py:43 (needs scale_factor>1 AND interpolate).
./gbm.py infer $1 -s $2 -bs $3 -sd "$4" -st "$5" -sf $6 -in "$7" -f

