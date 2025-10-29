#!/bin/bash
#SBATCH --job-name=gbm_export
#SBATCH --output=/home/afatehi/gbm_export.log
#SBATCH --error=/home/afatehi/gbm_export.err
#SBATCH --time=7-00:00:00   # Adjust time as needed
#SBATCH --partition=cpu     # Adjust partition based on available resources
#SBATCH --mem=32GB         # Adjust memory as needed
#SBATCH --cpus-per-task=32  # Adjust CPU cores as needed

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py export $1 -it $2
