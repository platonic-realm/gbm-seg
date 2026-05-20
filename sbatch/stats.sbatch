#!/bin/bash
#SBATCH --job-name=gbm_stats
#SBATCH --output=/home/afatehi/logs/gbm/stats.log
#SBATCH --error=/home/afatehi/logs/gbm/stats.log
#SBATCH --time=1-00:00:00   # Adjust time as needed
#SBATCH --partition=cpu     # Adjust partition based on available resources
#SBATCH --mem=128GB        # large test volumes load full-volume thickness arrays
#SBATCH --cpus-per-task=8  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg
./gbm.py stats $1 -it $2 $3
