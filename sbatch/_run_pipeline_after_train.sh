#!/bin/bash
# Launch the full inference pipeline for <exp> once its all-data training has
# finished — picks the LATEST all_data snapshot and hands it to
# run_pipeline.sh (infer -> psp -> morph -> stats + visual branch).
#
# Meant to be submitted as an afterok-dependent SLURM job:
#   sbatch --dependency=afterok:<train_jid> --partition=cpu \
#          --wrap="bash sbatch/_run_pipeline_after_train.sh <exp>"
#
# Inference geometry MUST match training: sample_dimension [12,256,256],
# z scale 6 (see configs.trainer.data.z_scale).
set -euo pipefail
EXP="$1"

cd ~/.vix/projects/gbm-seg
ROOT=$(python -c "import yaml; print(yaml.safe_load(open('./configs/template.yaml'))['experiments']['root'])")
SNAPDIR="${ROOT}/${EXP}/results-train/snapshots/all_data"

if [ ! -d "$SNAPDIR" ] || [ -z "$(ls -A "$SNAPDIR"/*.pt 2>/dev/null)" ]; then
    echo "Error: no snapshots in $SNAPDIR — training may have failed."
    exit 1
fi

SNAP=$(ls -t "$SNAPDIR"/*.pt | head -1)
echo "Latest snapshot for ${EXP}: $SNAP"

# bs=8, sample_dim must equal training [12,256,256], stride [3,128,128],
# scale 6 (z_scale), interpolate true.
bash sbatch/run_pipeline.sh "$EXP" "all_data/$(basename "$SNAP")" \
    8 "12, 256, 256" "3, 128, 128" 6 true
