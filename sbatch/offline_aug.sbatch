#!/bin/bash
#SBATCH --job-name=gbm_offline_aug
#SBATCH --output=/home/afatehi/logs/gbm/offline_aug_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/offline_aug_%x_%A.log
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32

# Precompute the offline-augmentation cache for an experiment — ONCE,
# single-process, before training. Offline aug (twist/zoom/rotate
# variants of every training volume) is CPU-heavy; doing it lazily in
# the training dataset means every DDP rank recomputes it at the same
# time (N x CPU + N x RAM -> swap death). This builds <ds_train>/cache/
# up front so the DDP ranks only read it.
#
# Run after `gbm.py create`, before any `gbm.py train` whose config has
# train_ds.augmentation.enabled_offline: true.
#
# Args: $1 = experiment name
EXP_NAME="$1"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

python gbm.py offline-aug "$EXP_NAME"
