#!/bin/bash
#SBATCH --job-name=gbm_create
#SBATCH --output=/home/afatehi/logs/gbm/create_%x_%A.log
#SBATCH --error=/home/afatehi/logs/gbm/create_%x_%A.log
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16

# Create a new experiment as a batch job. `gbm.py create` copies the whole
# source tree into <exp>/code/, resizes every default-dataset TIFF to
# default_voxel_size and Z-upsamples it by default_z_scale, snapshots
# pip freeze + git SHA/diff, and renders configs/template.yaml ->
# <exp>/configs.yaml. That TIFF resizing is CPU/RAM-heavy enough to belong
# on a compute node, not the login node.
#
# Run before offline_aug.sbatch and any train launcher.
#
# Args: $1 = experiment name
EXP_NAME="$1"

mkdir -p /home/afatehi/logs/gbm/

source ~/.penvs/venv-gbm/bin/activate && cd ~/.vix/projects/gbm-seg

python gbm.py create "$EXP_NAME"
