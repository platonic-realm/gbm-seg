#!/bin/bash
#SBATCH --job-name=gbm_train_all_data
#SBATCH --output=/home/afatehi/logs/gbm/train_all_data_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/train_all_data_%x_%A.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
# 400G excludes the 384G boxes (lg7, lg8); see train_ddp.sbatch.

# Stage-B all-data DDP training launcher: trains one final model on every
# ds_train subject with no fold holdout. Spawns one Python process per GPU
# (--nproc-per-node=4) via torchrun; each runs `gbm.py train --all-data`.
#
# Args: $1 = experiment name, $2 = epoch count (optional; overrides
#            configs.trainer.epochs)
EXP_NAME="$1"
EPOCHS="$2"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

# expandable_segments cuts the reserved-but-unallocated fragmentation that
# can tip a tight model into a spurious OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS_ARG=""
if [[ -n "$EPOCHS" ]]; then
    EPOCHS_ARG="--epochs $EPOCHS"
fi

torchrun \
    --standalone \
    --nproc-per-node=4 \
    gbm.py train "$EXP_NAME" --all-data $EPOCHS_ARG
