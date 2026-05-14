#!/bin/bash
# Submit a full 5-fold CV training run as parallel sbatch jobs.
#
# Usage:  sbatch/submit_cv.sh EXPERIMENT_NAME [K]
#
# Submits one `gbm.py train EXP --fold N` sbatch job per fold (default K=5),
# then a 6th `gbm.py aggregate-cv EXP` job that waits for all of them via
# --dependency=afterany so the aggregate runs even if a fold partially fails.
#
# The script must be executed from the repo root (the sbatch templates use
# absolute paths so it's location-independent, but the relative sbatch/
# lookup is not).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME [K]" >&2
    exit 1
fi

EXP_NAME="$1"
K="${2:-5}"

train_ids=()
for ((fold=0; fold<K; fold++)); do
    out=$(sbatch --job-name="gbm_${EXP_NAME}_f${fold}" \
                 --parsable \
                 sbatch/train.sbatch "$EXP_NAME" "$fold")
    echo "Submitted fold $fold as job $out"
    train_ids+=("$out")
done

# afterany: aggregate runs regardless of fold-job success so partial
# results still get summarised. Use afterok if you want strict "all-or-nothing".
deps=$(IFS=:; echo "${train_ids[*]}")
agg_id=$(sbatch --job-name="gbm_${EXP_NAME}_cv_agg" \
                --parsable \
                --dependency=afterany:"$deps" \
                sbatch/aggregate_cv.sbatch "$EXP_NAME")
echo "Submitted aggregate-cv as job $agg_id (waits on: $deps)"

echo
echo "Watch progress with:  squeue -u \$USER -j $(IFS=,; echo "${train_ids[*]}"),$agg_id"
