#!/bin/bash
#SBATCH --job-name=gbm_train_ddp
#SBATCH --output=/home/afatehi/logs/gbm/train_ddp_%x_%A_fold%a.log
#SBATCH --error=/home/afatehi/logs/gbm/train_ddp_%x_%A_fold%a.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
# 400G excludes the 384G boxes (lg7, lg8) where fold 4 OOM'd previously,
# leaves headroom on lg3/lg4/lg5 (510G) and is permissive enough that 5
# folds can queue without long resource waits.

# Single-fold DDP training launcher. Spawns one Python process per GPU
# (--nproc-per-node=4) via torchrun; each process runs gbm.py train.
# init_ddp() inside main_train picks up LOCAL_RANK / RANK / WORLD_SIZE
# from the torchrun env and brings up an NCCL process group.
#
# Args: $1 = experiment name, $2 = fold index (0..k-1)
EXP_NAME="$1"
FOLD="$2"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

# expandable_segments cuts the reserved-but-unallocated fragmentation that
# can tip a tight model into a spurious OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# rdzv settings work for single-node multi-GPU; for multi-node, add
# --rdzv-endpoint=<host>:<port> with a coordinator.
# nproc-per-node tracks the actual GPU allocation (SLURM_GPUS_ON_NODE), so
# the GPU count is controlled purely by the sbatch `--gres=gpu:N` flag —
# e.g. submit with `--gres=gpu:2` for a 2-GPU run. Falls back to 4.
torchrun \
    --standalone \
    --nproc-per-node="${SLURM_GPUS_ON_NODE:-4}" \
    gbm.py train "$EXP_NAME" --fold "$FOLD"
