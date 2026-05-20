#!/bin/bash
#SBATCH --job-name=gbm_psp
#SBATCH --output=/home/afatehi/logs/gbm/psp.log
#SBATCH --error=/home/afatehi/logs/gbm/psp.log
#SBATCH --time=1-00:00:00   # Adjust time as needed
#SBATCH --partition=train   # Adjust partition based on available resources
#SBATCH --mem=160GB         # Adjust memory as needed
#SBATCH --cpus-per-task=64  # Adjust CPU cores as needed

# Create log directory if it doesn't exist
mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

# $3 (optional, "true"/"false") routes psp through results-infer-labeled/
# instead of results-infer/. Pair with `gbm.py infer --labeled`.
LABELED_FLAG=""
if [ "${3:-false}" = "true" ]; then
    LABELED_FLAG="--labeled"
fi

./gbm.py psp $1 -it $2 -mc 64 $LABELED_FLAG
