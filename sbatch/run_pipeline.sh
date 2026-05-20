#!/bin/bash
# gbm-seg inference pipeline orchestrator.
#
# Submits the whole post-training pipeline as one dependency-chained set of
# SLURM jobs and returns immediately — no node is held while it runs.
#
#   infer -> psp -> morph -+-> stats                     (analysis branch)
#                          +-> blender -> render -> export   (visual branch)
#
# Every stage starts only after its predecessor succeeds (--dependency=afterok).
# The array stages (infer/morph/blender) use a parent that blocks until its
# array completes (see lib_array_wait.sh), so afterok on them is honest:
# a dependent fires only once every volume/sample actually finished.
#
# stats depends on morph directly, so the analysis branch is not held up by
# the expensive Blender render branch.
#
# Usage:
#   sbatch/run_pipeline.sh <exp> <snapshot> <bs> <sample_dim> <stride> \
#                          <scale> <interp> [--output-name NAME] [--clipping]
#
#   --output-name NAME  isolate this run under results-infer/NAME/ instead of
#                       the auto-derived tag — lets a fresh run sit alongside
#                       an existing one without colliding.
#   --clipping          pass --clipping through to the stats stage.
# Example:
#   sbatch/run_pipeline.sh debug_swinunetr_alldata_5ep all_data/005-7000.pt \
#       8 "24, 128, 128" "12, 64, 64" 6 true
set -euo pipefail

if [ "$#" -lt 7 ]; then
    echo "Usage: $0 <exp> <snapshot> <bs> <sample_dim> <stride> <scale>" \
         "<interp> [--output-name NAME] [--clipping]"
    exit 1
fi

EXP="$1"; SNAP="$2"; BS="$3"; SD="$4"; ST="$5"; SCALE="$6"; INTERP="$7"
shift 7

OUTPUT_NAME=""
CLIP=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --clipping)    CLIP="--clipping"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd ~/.vix/projects/gbm-seg

# The downstream stages locate the run by its inference tag. With
# --output-name the tag IS that name; otherwise it must match
# infer_experiment()'s derivation exactly:
#   f"{snapshot}_{''.join(sample_dim)}_{''.join(stride)}_{scale}"
# i.e. the dimension lists with every comma and space stripped out.
if [ -n "$OUTPUT_NAME" ]; then
    TAG="$OUTPUT_NAME"
else
    SD_J=$(echo "$SD" | tr -d ', ')
    ST_J=$(echo "$ST" | tr -d ', ')
    TAG="${SNAP}_${SD_J}_${ST_J}_${SCALE}"
fi

# --- Pre-flight checks (fail fast, before queueing anything) ---
ROOT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('./configs/template.yaml'))['experiments']['root'])")
if [ ! -d "${ROOT_PATH}/${EXP}" ]; then
    echo "Error: experiment '${EXP}' not found under ${ROOT_PATH}"
    exit 1
fi
if [ ! -f "${ROOT_PATH}/${EXP}/results-train/snapshots/${SNAP}" ]; then
    echo "Error: snapshot '${SNAP}' not found under" \
         "${ROOT_PATH}/${EXP}/results-train/snapshots/"
    exit 1
fi

echo "Pipeline for experiment : ${EXP}"
echo "Snapshot                : ${SNAP}"
echo "Inference tag           : ${TAG}"
echo

# --- Chain the stages ---
#
# Two parallel inference branches, joined at stats:
#
#   unlabeled branch (whole glomerular volumes, for morphometry):
#     infer ─→ psp ─→ morph ─┬→ blender ─→ render ─→ export
#                            └→ stats
#
#   labeled branch (expert-annotated crops, for expert comparison):
#     infer --labeled ─→ psp --labeled ─→ stats
#
# stats depends on BOTH morph (for the unlabeled-side morph aggregation)
# AND psp_labeled (for the expert comparison the new
# src/infer/expert_comparison module runs against ds_test_labeled).
#
# $8 (OUTPUT_NAME) is forwarded to infer.sbatch; empty => auto-derived tag.
# The 9th positional on infer.sbatch is the labeled flag (true/false).

J_INFER=$(sbatch --parsable sbatch/infer.sbatch \
    "$EXP" "$SNAP" "$BS" "$SD" "$ST" "$SCALE" "$INTERP" "$OUTPUT_NAME" "false")
echo "infer            : $J_INFER"

J_PSP=$(sbatch --parsable --dependency=afterok:"$J_INFER" \
    sbatch/psp.sbatch "$EXP" "$TAG" "false")
echo "psp              : $J_PSP   (afterok $J_INFER)"

J_MORPH=$(sbatch --parsable --dependency=afterok:"$J_PSP" \
    sbatch/morph.sbatch "$EXP" "$TAG")
echo "morph            : $J_MORPH   (afterok $J_PSP)"

# Labeled branch — fires in parallel from the start.
J_INFER_LBL=$(sbatch --parsable sbatch/infer.sbatch \
    "$EXP" "$SNAP" "$BS" "$SD" "$ST" "$SCALE" "$INTERP" "$OUTPUT_NAME" "true")
echo "infer-labeled    : $J_INFER_LBL"

J_PSP_LBL=$(sbatch --parsable --dependency=afterok:"$J_INFER_LBL" \
    sbatch/psp.sbatch "$EXP" "$TAG" "true")
echo "psp-labeled      : $J_PSP_LBL   (afterok $J_INFER_LBL)"

# stats joins the two branches — morph AND psp-labeled must both succeed.
J_STATS=$(sbatch --parsable --dependency=afterok:"$J_MORPH":"$J_PSP_LBL" \
    sbatch/stats.sbatch "$EXP" "$TAG" "$CLIP")
echo "stats            : $J_STATS   (afterok $J_MORPH + $J_PSP_LBL)"

J_BLENDER=$(sbatch --parsable --dependency=afterok:"$J_MORPH" \
    sbatch/blender.sbatch "$EXP" "$TAG")
echo "blender          : $J_BLENDER   (afterok $J_MORPH)"

J_RENDER=$(sbatch --parsable --dependency=afterok:"$J_BLENDER" \
    sbatch/render.sbatch "$EXP" "$TAG")
echo "render           : $J_RENDER   (afterok $J_BLENDER)"

J_EXPORT=$(sbatch --parsable --dependency=afterok:"$J_RENDER" \
    sbatch/export.sbatch "$EXP" "$TAG")
echo "export           : $J_EXPORT   (afterok $J_RENDER)"

echo
echo "Submitted. Monitor with:  squeue -u $USER"
echo "If any stage fails, every afterok dependent is auto-cancelled."
