#!/bin/bash
#SBATCH --job-name=gbm_export
#SBATCH --output=/home/afatehi/logs/gbm/export.log
#SBATCH --error=/home/afatehi/logs/gbm/export.log
#SBATCH --time=7-00:00:00   # Adjust time as needed
#SBATCH --partition=cpu     # Adjust partition based on available resources
#SBATCH --mem=32GB         # Adjust memory as needed
#SBATCH --cpus-per-task=32  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py export $1 -it $2
