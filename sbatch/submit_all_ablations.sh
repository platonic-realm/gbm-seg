#!/bin/bash
# Launch the ENTIRE loss ablation in one command.
#
#   sbatch/submit_all_ablations.sh [K]        # K = folds, default 5
#
# Does, for BOTH architectures (loss_unet.yaml + loss_swin.yaml specs):
#   1. `gbm.py ablate <spec>` — materialise one experiment dir per loss cell
#      (cells symlink datasets/+code/ from the base experiment, so they share
#      the offline-aug cache; nothing is re-copied).
#   2. For every materialised cell, `submit_cv_ddp.sh <cell> K` — submit its
#      K folds (each a single-H100 DDP job, sbatch/train_ddp.sbatch) plus a
#      dependent aggregate-cv job.
#
# Result: 2 archs x 4 losses x K folds train jobs + 8 aggregates, all queued
# at once. On ramses the `gpu_user_limit` QOS caps you at 8 GPUs, so 8 folds
# train concurrently and the rest queue and drain automatically.
#
# PREREQUISITE: the two base experiments must already exist (build them with
# create -> offline-aug -> reuse-create; see the ablation-paused runbook).
# Run from the repo root.

set -euo pipefail

K="${1:-5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPECS=(ablation_specs/loss_unet.yaml ablation_specs/loss_swin.yaml)

# --- derive the cell experiment names (<study>__<cell>) from the specs ---
mapfile -t CELLS < <(python - "${SPECS[@]}" <<'PY'
import sys, yaml
for f in sys.argv[1:]:
    s = yaml.safe_load(open(f))
    study = s['study']
    for c in s['cells']:
        print(f"{study}__{c['name']}")
PY
)

echo "=== loss ablation: ${#CELLS[@]} cells x ${K} folds = $(( ${#CELLS[@]} * K )) train jobs ==="

# --- 1. materialise every cell from both specs ---
for spec in "${SPECS[@]}"; do
    echo "--- materialising cells from $spec ---"
    python gbm.py ablate "$spec"
done

# --- 2. submit K-fold CV for every cell ---
for cell in "${CELLS[@]}"; do
    echo "=== submitting ${K}-fold CV: $cell ==="
    bash sbatch/submit_cv_ddp.sh "$cell" "$K"
done

echo
echo "=== ALL ${#CELLS[@]} cells submitted (${#CELLS[@]} x ${K} folds + ${#CELLS[@]} aggregates) ==="
echo "Watch:  squeue -u \$USER"
echo "Note: gpu_user_limit QOS caps you at 8 H100 concurrently; folds queue and drain."
