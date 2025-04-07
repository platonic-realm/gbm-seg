#!/bin/bash
#SBATCH --job-name=gbm_train
#SBATCH --output=/home/afatehi/gbm_train.log
#SBATCH --error=/home/afatehi/gbm_train.err
#SBATCH --time=7-00:00:00   # Adjust time as needed
#SBATCH --partition=train   # Adjust partition based on available resources
#SBATCH --gres=gpu:A100+:8  # Request 8 GPU
#SBATCH --mem=320G          # Adjust memory as needed
#SBATCH --cpus-per-task=64  # Adjust CPU cores as needed

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py infer $1 -s $2 -bs 168 -sd "12, 256, 256" -st "1, 128, 128" -sf 6

