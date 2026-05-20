#!/bin/bash
# Per-node worker for sbatch/train_multi_node.sbatch.
#
# srun launches one copy of this on every allocated node. Each copy
# spawns GPUS_PER_NODE worker processes via torchrun and joins the
# global rendezvous at MASTER_ADDR:MASTER_PORT — so the world_size of
# the resulting DDP training group is NNODES * GPUS_PER_NODE.
#
# Inherits these env vars from the parent sbatch (see the export line in
# train_multi_node.sbatch):
#   MASTER_ADDR, MASTER_PORT — rdzv endpoint (rank 0's host + a job-id-derived port)
#   GPUS_PER_NODE            — per-node worker count (uniform across the pool)
#   MULTI_NODES              — total node count (= SLURM_NNODES)
#   EXP_NAME, EPOCHS         — forwarded to `gbm.py train --all-data`
set -e

source ~/.penvs/venv-gbm/bin/activate
cd ~/.vix/projects/gbm-seg

EPOCHS_ARG=""
[[ -n "${EPOCHS:-}" ]] && EPOCHS_ARG="--epochs $EPOCHS"

torchrun \
    --nnodes="$MULTI_NODES" \
    --node-rank="$SLURM_NODEID" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    gbm.py train "$EXP_NAME" --all-data $EPOCHS_ARG
