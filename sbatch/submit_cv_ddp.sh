#!/bin/bash
# Submit a full 5-fold CV training run as parallel DDP sbatch jobs.
#
# Usage:  sbatch/submit_cv_ddp.sh EXPERIMENT_NAME [K]
#
# Each fold is a separate `torchrun` invocation on its own 4-GPU node
# (sbatch/train_ddp.sbatch). The aggregate-cv job depends on all folds
# via afterany so partial results still get summarised if a fold dies.
#
# DDP path is opt-in via `trainer.ddp: True` in the experiment yaml.
# For DP runs use sbatch/submit_cv.sh instead.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME [K]" >&2
    exit 1
fi

EXP_NAME="$1"
K="${2:-5}"

train_ids=()
for ((fold=0; fold<K; fold++)); do
    out=$(sbatch --job-name="gbm_${EXP_NAME}_ddp_f${fold}" \
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
