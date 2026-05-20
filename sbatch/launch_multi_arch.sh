#!/bin/bash
# Discover every live node of a chosen GPU arch (A100 or V100) in the
# cluster's `train` partition, then submit one multi-node DDP training
# job across all of them via sbatch/train_multi_node.sbatch.
#
# The two arch pools are mutually exclusive — A100 nodes vs V100 nodes —
# so two of these can run concurrently. That's the point: a future
# ablation campaign can compare offline-aug or loss variants on A100
# and V100 in parallel.
#
# Design notes:
#   - Per-node GPU count is fixed at the MINIMUM of the pool (4 for both
#     pools on this cluster) so per-rank work stays uniform. SwinUNETR on
#     lg6 has 8 A100-80GB but contributes only 4 here — heterogeneous
#     per-rank batches break DDP (sync barrier waits on the slowest rank
#     anyway, and the gradient mean becomes rank-weighted, see CLAUDE.md).
#     For an 8-GPU lg6-only run use sbatch/train_all_data_lg6.sbatch.
#   - lg6 (A100+ = the 80GB variant) is matched into the A100 pool via
#     a substring test, and addressed with an UNTYPED `--gres=gpu:N` so
#     it co-exists with lg3/4/5's typed `A100` GRES under one job.
#   - Down/drained nodes are skipped automatically.
#
# Args:
#   $1  arch       A100 | V100 | auto
#   $2  exp_name   experiment directory under experiments.root
#   $3  epochs     optional override of trainer.optimization.epochs
#
# Examples:
#   sbatch/launch_multi_arch.sh A100 swin_offline_ablation
#   sbatch/launch_multi_arch.sh V100 unet_offline_ablation
#   sbatch/launch_multi_arch.sh auto smoke_test         # picks A100 if any live

set -euo pipefail

ARCH="${1:?usage: launch_multi_arch.sh A100|V100|auto <exp_name> [epochs]}"
EXP="${2:?usage: launch_multi_arch.sh A100|V100|auto <exp_name> [epochs]}"
EPOCHS="${3:-}"

cd "$(dirname "$0")/.."

# Arch test: A100+ (80GB) counts as A100, so a substring match on the
# lowered GRES string is the right granularity.
match_arch() {
    local gres="${1,,}" want="$2"
    case "$want" in
        A100) [[ "$gres" =~ a100 ]];;
        V100) [[ "$gres" =~ v100 ]];;
        *) return 1;;
    esac
}

# SLURM uses '*' / '+' suffixes on the state field (e.g. mix*, idle+).
# `live` here means the node is at least reachable and not under
# maintenance. The job sbatch waits in the queue for resources anyway.
is_live() {
    local state="${1,,}"
    [[ ! "$state" =~ (down|drain|fail|maint|invalid|unk|future) ]]
}

# Pull the integer GPU count from a GRES string like "gpu:A100:4" or
# "gpu:A100+:8". `(`-anchored to skip socket pinning suffixes.
gres_count() {
    echo "$1" | grep -oE ':[0-9]+(\(|$)' | grep -oE '[0-9]+' | head -1
}

discover_pool() {
    local want="$1"
    sinfo -h -p train -N -o "%n %G %t" 2>/dev/null \
      | while read -r node gres state; do
          if match_arch "$gres" "$want" && is_live "$state"; then
              echo "$node $(gres_count "$gres")"
          fi
        done
}

case "${ARCH^^}" in
  A100|V100) WANT="${ARCH^^}";;
  AUTO|"")
    a=$(discover_pool A100 | wc -l)
    v=$(discover_pool V100 | wc -l)
    echo "[auto] live A100 nodes=$a  live V100 nodes=$v"
    if [ "$a" -ge 1 ]; then WANT="A100"
    elif [ "$v" -ge 1 ]; then WANT="V100"
    else echo "ERROR: no live A100 or V100 nodes in train partition" >&2; exit 1; fi
    ;;
  *)
    echo "ERROR: unknown arch '$ARCH' — expected A100, V100, or auto" >&2; exit 1;;
esac

mapfile -t POOL < <(discover_pool "$WANT")
if [ "${#POOL[@]}" -eq 0 ]; then
    echo "ERROR: no live $WANT nodes in the train partition" >&2; exit 1
fi

NODELIST=""
MIN_GPUS=999
for entry in "${POOL[@]}"; do
    node=$(echo "$entry" | awk '{print $1}')
    n=$(echo "$entry" | awk '{print $2}')
    [ -z "$n" ] && continue
    [ "$n" -lt "$MIN_GPUS" ] && MIN_GPUS="$n"
    NODELIST="${NODELIST:+$NODELIST,}$node"
done
NNODES=${#POOL[@]}

# Per-pool resource floor: pick the safe minimum across the pool's nodes.
#   V100 (lg7/8 have 384G RealMemory) -> --mem=320G  (room for SLURM overhead)
#   A100 (lg3/4/5 have 510G, lg6 has 1024G) -> --mem=400G (matches train_all_data.sbatch)
# 32 cpus/task fits even the 32-core V100 boxes (lg9/10).
if [ "$WANT" = "V100" ]; then
    MEM="320G"
else
    MEM="400G"
fi
CPT=32

cat <<EOF
Launching $WANT multi-node training:
  experiment   : $EXP
  pool         : $NODELIST
  nodes        : $NNODES
  gpus/node    : $MIN_GPUS  (uniform; an 8-GPU lg6 underutilises if it is in the pool)
  world_size   : $((NNODES * MIN_GPUS))
  cpus-per-task: $CPT
  mem          : $MEM
EOF

JID=$(sbatch --parsable \
  -J "gbm_train_${WANT}_multi" \
  -N "$NNODES" \
  --nodelist="$NODELIST" \
  --ntasks-per-node=1 \
  --gres="gpu:$MIN_GPUS" \
  --mem="$MEM" \
  --cpus-per-task="$CPT" \
  sbatch/train_multi_node.sbatch "$EXP" "$EPOCHS")

echo "Submitted multi-node training: $JID"
