#!/bin/bash
#SBATCH --job-name=gbm_labels_as_pred
#SBATCH --output=/home/afatehi/logs/gbm/labels_as_pred_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/labels_as_pred_%x_%A.log
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Treat the TRAINING labels (ds_train/) as if they were model predictions,
# then run the standard psp → morph → (blender → render → export) + stats
# chain against them. Result is a "perfect-model" ground-truth baseline for
# every morphometric and statistical figure the pipeline produces.
#
# Usage:
#   sbatch sbatch/labels_as_pred.sbatch <experiment-name> [output-tag]
# Example:
#   sbatch sbatch/labels_as_pred.sbatch swin_offline_a100_v6_zconvembed
#
# This job is a LAUNCHER. It only does the (fast, CPU) label-materialisation
# step inline, then submits the heavy stages as dependent SLURM jobs:
#
#   psp  (afterok: none — runs after this launcher's inline materialise)
#     └─ morph  (afterok: psp)  ← self-arrays: ONE array task per sample,
#        │                         so all 15 training volumes morph IN
#        │                         PARALLEL instead of serially. The earlier
#        │                         in-process for-loop morphed sequentially
#        │                         and could not finish full-resolution
#        │                         training volumes within any walltime.
#        ├─ stats   (afterok: morph)   ← critical path
#        └─ blender (afterok: morph) → render → export   ← best-effort viz
#
# stats and the viz chain both hang off morph and run in parallel; a viz
# failure therefore never blocks stats.

mkdir -p /home/afatehi/logs/gbm/

EXP_NAME="$1"
TAG="${2:-labels_train}"
if [ -z "$EXP_NAME" ]; then
    echo "Usage: sbatch $0 <experiment-name> [output-tag]"
    echo "  output-tag defaults to labels_train"
    exit 1
fi

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

set -euo pipefail

# 1. Materialise training labels → prediction.{npz,tif} in the inference
#    output layout. ds_train/ TIFFs are already at upsampled Z, so no
#    extra np.repeat needed (default --z-repeat 1). Fast, CPU-only.
python gbm.py labels-as-pred "$EXP_NAME" --output-tag "$TAG"

# 2. Submit the heavy stages as a dependency chain. morph.sbatch self-arrays
#    (one task per sample) and its parent blocks until the array completes,
#    so afterok:$MORPH fires only after every sample's thickness is done.
PSP=$(sbatch --parsable sbatch/psp.sbatch "$EXP_NAME" "$TAG")
echo "Submitted psp:    $PSP"

MORPH=$(sbatch --parsable --dependency=afterok:"$PSP" \
        sbatch/morph.sbatch "$EXP_NAME" "$TAG")
echo "Submitted morph:  $MORPH  (afterok:$PSP; self-arrays per sample)"

STATS=$(sbatch --parsable --dependency=afterok:"$MORPH" \
        sbatch/stats.sbatch "$EXP_NAME" "$TAG")
echo "Submitted stats:  $STATS  (afterok:$MORPH)"

# Best-effort visualisation chain (parallel to stats). Uses afterok so a
# failed upstream skips downstream, but it never gates stats.
BLEND=$(sbatch --parsable --dependency=afterok:"$MORPH" \
        sbatch/blender.sbatch "$EXP_NAME" "$TAG")
REND=$(sbatch --parsable --dependency=afterok:"$BLEND" \
        sbatch/render.sbatch "$EXP_NAME" "$TAG")
EXPORT=$(sbatch --parsable --dependency=afterok:"$REND" \
        sbatch/export.sbatch "$EXP_NAME" "$TAG")
echo "Submitted viz:    blender=$BLEND render=$REND export=$EXPORT"

echo
echo "labels-as-pred chain launched for $EXP_NAME / tag=$TAG"
echo "Critical path: psp($PSP) -> morph($MORPH) -> stats($STATS)"
echo "Outputs will land under: experiments/$EXP_NAME/results-infer/${TAG}{,_stats,_export}/"
