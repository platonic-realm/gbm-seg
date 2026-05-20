#!/bin/bash
#SBATCH --job-name=gbm_train_multi
#SBATCH --output=/home/afatehi/logs/gbm/train_multi_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/train_multi_%x_%A.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=train
#SBATCH --ntasks-per-node=1
# --nodes, --nodelist, --gres, --mem, --cpus-per-task are passed at
# submit time by sbatch/launch_multi_arch.sh (they depend on the chosen
# arch pool's actual node set).

# Multi-node DDP training launcher.
#
# Spans every node SLURM allocated to this job. Picks the first
# allocated host as the rendezvous server, derives the rendezvous port
# from the job id so two concurrent multi-node jobs (e.g. A100 and V100
# halves of an ablation pair) don't collide, then launches one torchrun
# per node via srun. Each torchrun spawns GPUS_PER_NODE worker
# processes; world_size = nnodes * gpus_per_node.
#
# Args: $1 = experiment name, $2 = epoch override (optional).

set -e

EXP_NAME="$1"
EPOCHS="${2:-}"

mkdir -p /home/afatehi/logs/gbm/

# Rendezvous: first node of the allocation owns rank-0; SLURM_JOB_ID
# salts the port so concurrent multi-node jobs don't fight for 29500.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
MULTI_NODES=$SLURM_NNODES

echo "Multi-node DDP for '$EXP_NAME'"
echo "  nodes        : $MULTI_NODES"
echo "  nodelist     : $(scontrol show hostnames "$SLURM_NODELIST" | paste -sd, -)"
echo "  master       : $MASTER_ADDR:$MASTER_PORT"
echo "  gpus/node    : $GPUS_PER_NODE"
echo "  world_size   : $((MULTI_NODES * GPUS_PER_NODE))"

# Caching allocator hygiene + NCCL error handling. Multi-node is where
# silent NCCL hangs on a dead peer hurt the most — make them fail loudly.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Uncomment on the first launch to confirm the right inter-node interface:
# export NCCL_DEBUG=INFO

# Hand these to the per-node worker.
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE MULTI_NODES EXP_NAME EPOCHS

# One srun task per node; torchrun inside spawns the local worker pool
# and joins the global rendezvous.
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
     bash sbatch/_train_multi_node_inner.sh
