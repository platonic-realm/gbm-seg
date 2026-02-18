#!/bin/bash
#SBATCH --job-name=gbm_stats
#SBATCH --output=/home/afatehi/gbm_log/stats.log
#SBATCH --error=/home/afatehi/gbm_log/stats.log
#SBATCH --time=1-00:00:00   # Adjust time as needed
#SBATCH --partition=cpu     # Adjust partition based on available resources
#SBATCH --mem=16GB         # Adjust memory as needed
#SBATCH --cpus-per-task=8  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/gbm_log

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py stats $1 -it $2 --source $3