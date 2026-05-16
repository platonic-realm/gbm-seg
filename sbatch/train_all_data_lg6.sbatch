#!/bin/bash
#SBATCH --job-name=gbm_train_lg6
#SBATCH --output=/home/afatehi/logs/gbm/train_all_data_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/train_all_data_%x_%A.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=train
#SBATCH --nodelist=lyn-gpu-06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --cpus-per-task=64

# All-data DDP training pinned to lyn-gpu-06 — the 8x A100-80GB node.
# Uses all 8 GPUs (torchrun --nproc-per-node=8 → 8 DDP ranks). For the
# standard 4-GPU launchers see train_all_data.sbatch / train_ddp.sbatch.
#
# Args: $1 = experiment name, $2 = epoch count (optional; overrides
#            configs.trainer.epochs)
EXP_NAME="$1"
EPOCHS="$2"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

# expandable_segments cuts reserved-but-unallocated fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS_ARG=""
if [[ -n "$EPOCHS" ]]; then
    EPOCHS_ARG="--epochs $EPOCHS"
fi

torchrun \
    --standalone \
    --nproc-per-node=8 \
    gbm.py train "$EXP_NAME" --all-data $EPOCHS_ARG
