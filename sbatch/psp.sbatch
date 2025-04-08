#!/bin/bash
#SBATCH --job-name=gbm_psp
#SBATCH --output=/home/afatehi/gbm_psp.log
#SBATCH --error=/home/afatehi/gbm_psp.err
#SBATCH --time=1-00:00:00   # Adjust time as needed
#SBATCH --partition=train   # Adjust partition based on available resources
#SBATCH --mem=160GB         # Adjust memory as needed
#SBATCH --cpus-per-task=64  # Adjust CPU cores as needed

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py psp $1 -it $2 -mc 64
