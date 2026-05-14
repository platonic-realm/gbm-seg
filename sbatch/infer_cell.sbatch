#!/bin/bash
#SBATCH --job-name=gbm_infer_cell
#SBATCH --output=/home/afatehi/logs/gbm/infer_cell_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/infer_cell_%x_%A.log
#SBATCH --time=12:00:00
#SBATCH --partition=train
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# One inference-axis ablation cell. Designed to be the sbatch wrapper
# passed to `gbm.py infer-ablate ... --sbatch sbatch/infer_cell.sbatch`,
# which emits 4 such submissions per stitch mode etc.
#
# Args: $1=base_experiment $2=snapshot $3=output_tag $4=stitching_mode
# (stitching is optional — pass empty string to use the experiment yaml's).

EXP="$1"
SNAP="$2"
TAG="$3"
STITCH="${4:-}"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

EXTRA_ARGS=""
[ -n "$STITCH" ] && EXTRA_ARGS="--stitching $STITCH"

python gbm.py infer "$EXP" \
    -s "$SNAP" \
    -bs 32 \
    -sd "12, 256, 256" \
    -st "1, 128, 128" \
    -sf 6 \
    -in true \
    --output-name "$TAG" \
    -f \
    $EXTRA_ARGS
