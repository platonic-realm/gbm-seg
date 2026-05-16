#!/bin/bash
# Submit a full 5-fold CV training run as parallel DDP sbatch jobs.
#
# Usage:  sbatch/submit_cv_ddp.sh EXPERIMENT_NAME [K] [EXTRA_SBATCH_FLAGS]
#
# Each fold is a separate `torchrun` invocation on its own 4-GPU node
# (sbatch/train_ddp.sbatch). The aggregate-cv job depends on all folds
# via afterany so partial results still get summarised if a fold dies.
#
# EXTRA_SBATCH_FLAGS (optional 3rd arg) is passed verbatim to every
# fold's `sbatch` call — e.g. "--gres=gpu:A100:4" to pin SwinUNETR folds
# to A100 nodes (SwinUNETR OOMs on the 32 GB V100s). Not applied to the
# aggregate-cv job, which needs no GPU.
#
# DDP path is opt-in via `trainer.ddp: True` in the experiment yaml.
# For DP runs use sbatch/submit_cv.sh instead.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME [K] [EXTRA_SBATCH_FLAGS]" >&2
    exit 1
fi

EXP_NAME="$1"
K="${2:-5}"
EXTRA_SBATCH="${3:-}"

train_ids=()
for ((fold=0; fold<K; fold++)); do
    # EXTRA_SBATCH is intentionally unquoted so a flag such as
    # --gres=gpu:A100:4 splits into its own argv entry; empty by default.
    out=$(sbatch ${EXTRA_SBATCH} \
                 --job-name="gbm_${EXP_NAME}_ddp_f${fold}" \
                 --parsable \
                 sbatch/train_ddp.sbatch "$EXP_NAME" "$fold")
    echo "Submitted DDP fold $fold as job $out"
    train_ids+=("$out")
done

deps=$(IFS=:; echo "${train_ids[*]}")
agg_id=$(sbatch --job-name="gbm_${EXP_NAME}_cv_agg" \
                --parsable \
                --dependency=afterany:"$deps" \
                sbatch/aggregate_cv.sbatch "$EXP_NAME")
echo "Submitted aggregate-cv as job $agg_id (waits on: $deps)"

echo
echo "Watch progress with:  squeue -u \$USER -j $(IFS=,; echo "${train_ids[*]}"),$agg_id"
