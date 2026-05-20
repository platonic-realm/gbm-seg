#!/bin/bash
# Multi-node DDP training launcher.
#
# Two modes, mutually exclusive:
#
#   Explicit (preferred) — you tell it exactly which nodes and the
#   uniform per-node GPU count:
#
#     sbatch/launch_multi_node.sh <exp> --nodes <csv> --gpus <n> [options]
#
#     Example:
#       sbatch/launch_multi_node.sh test_run --nodes lyn-gpu-07,lyn-gpu-08 --gpus 2
#
#   Auto-arch — discovers every live node of one GPU architecture
#   (A100, V100, future H100, …) in the chosen partition. Per-node GPU
#   count is the minimum capacity across the discovered pool, so an
#   8-GPU lg6 contributes only 4 in a pool with 4-GPU peers (for the
#   full lg6 use sbatch/train_all_data_lg6.sbatch instead).
#
#     sbatch/launch_multi_node.sh <exp> --arch A100|V100|auto [options]
#
# Per-rank work stays uniform across the pool in both modes — heterogeneous
# DDP batches stall on the sync barrier and bias the gradient mean (see
# CLAUDE.md). Arch matching is a substring test, so future arches need no
# code change.
#
# CPU/mem are auto-sized from the smallest FREE-resource node in the pool
# (queried via `scontrol show node`), with a 10% safety margin on memory.
# Explicit `--cpus` / `--mem` override.
#
# Options:
#   --epochs N      override trainer.optimization.epochs at train time
#   --cpus N        per-task CPU count   (default: 8 × gpus-per-node, capped at min free)
#   --mem N[G|M]    memory per node      (default: 90% of min(free mem) across pool)
#   --partition P   SLURM partition      (default: train)
#   --dry-run       print the sbatch command and exit without submitting

set -euo pipefail

EXP="" ARCH="" NODES_CSV="" GPUS_PER_NODE=""
EPOCHS="" CPT="" MEM="" PARTITION="train" DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --arch)      ARCH="$2";          shift 2;;
        --nodes)     NODES_CSV="$2";     shift 2;;
        --gpus)      GPUS_PER_NODE="$2"; shift 2;;
        --epochs)    EPOCHS="$2";        shift 2;;
        --cpus)      CPT="$2";           shift 2;;
        --mem)       MEM="$2";           shift 2;;
        --partition) PARTITION="$2";     shift 2;;
        --dry-run)   DRY_RUN=1;          shift;;
        -h|--help)
            sed -nE 's/^# ?(.*)$/\1/p' "$0" | sed -n '/^Multi-node DDP/,/^$/p'
            exit 0;;
        -*)  echo "Unknown flag: $1" >&2; exit 1;;
        *)
            if [ -z "$EXP" ]; then EXP="$1"; shift
            else echo "Unexpected positional arg '$1'" >&2; exit 1; fi;;
    esac
done

[ -z "$EXP" ] && { echo "ERROR: experiment name required (positional)" >&2; exit 1; }

# Mode selection: explicit takes precedence; --arch is the fallback.
if [ -n "$NODES_CSV" ] || [ -n "$GPUS_PER_NODE" ]; then
    [ -z "$NODES_CSV" ]    && { echo "ERROR: --gpus requires --nodes" >&2; exit 1; }
    [ -z "$GPUS_PER_NODE" ] && { echo "ERROR: --nodes requires --gpus" >&2; exit 1; }
    MODE=explicit
elif [ -n "$ARCH" ]; then
    MODE=arch
else
    echo "ERROR: provide '--nodes <csv> --gpus <n>' or '--arch <A100|V100|auto>'" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

# --- helpers ----------------------------------------------------------

# Pull the int GPU count from a GRES string like "gpu:A100:4" or
# "gpu:V100:4(...)" — last :N before '(' or end-of-line.
gres_count() {
    echo "$1" | grep -oE ':[0-9]+(\(|$)' | grep -oE '[0-9]+' | head -1
}

is_live() {
    local s="${1,,}"
    [[ ! "$s" =~ (down|drain|fail|maint|invalid|unk|future) ]]
}

# Substring match — future arches (H100, MI300, …) work with no code change.
match_arch() {
    local gres="${1,,}" want="${2,,}"
    [[ "$gres" =~ $want ]]
}

# Per-node free CPU / free mem (MB). Prints: <node> <free_cpu> <free_mem_mb>.
node_free() {
    local n="$1"
    scontrol show node "$n" 2>/dev/null | awk -v out="$n" '
        match($0,/CPUTot=[0-9]+/)     { ct = substr($0,RSTART+7,RLENGTH-7)+0 }
        match($0,/CPUAlloc=[0-9]+/)   { ca = substr($0,RSTART+9,RLENGTH-9)+0 }
        match($0,/RealMemory=[0-9]+/) { rm = substr($0,RSTART+11,RLENGTH-11)+0 }
        match($0,/AllocMem=[0-9]+/)   { am = substr($0,RSTART+9,RLENGTH-9)+0 }
        END { print out, (ct-ca), (rm-am) }'
}

# --- build pool -------------------------------------------------------
declare -a POOL=()

if [ "$MODE" = explicit ]; then
    IFS=',' read -ra _NODES <<< "$NODES_CSV"
    for n in "${_NODES[@]}"; do
        info=$(scontrol show node "$n" 2>/dev/null)
        [ -z "$info" ] && { echo "ERROR: node '$n' not found" >&2; exit 1; }
        gres=$(echo "$info" | grep -oE 'Gres=[^ ]+' | head -1 | cut -c6-)
        cap=$(gres_count "$gres")
        if [ "${cap:-0}" -lt "$GPUS_PER_NODE" ]; then
            echo "ERROR: node '$n' has only ${cap:-?} GPU(s) (Gres=$gres); requested $GPUS_PER_NODE per node" >&2
            exit 1
        fi
        POOL+=("$n")
    done
    NODELIST="$NODES_CSV"
else
    # --arch mode: discover live nodes of the arch that currently have
    # >0 free GPUs (a busy/fully-allocated node would just make the
    # whole job queue indefinitely). Per-node GPU = min(free) across
    # the resulting pool, again for DDP uniformity.
    declare -A FREE_GPU=()

    # Parse free GPUs from `scontrol show node` (capacity - allocated).
    # AllocTRES has `gres/gpu=N`; absent means 0 allocated.
    node_free_gpus() {
        local n="$1" info gres cap alloc
        info=$(scontrol show node "$n" 2>/dev/null) || return
        gres=$(echo "$info" | grep -oE 'Gres=[^ ]+' | head -1 | cut -c6-)
        cap=$(gres_count "$gres")
        alloc=$(echo "$info" | grep -oE 'AllocTRES=[^ ]*' | head -1 \
                | grep -oE 'gres/gpu=[0-9]+' | cut -d= -f2)
        echo $(( ${cap:-0} - ${alloc:-0} ))
    }

    if [ "${ARCH^^}" = AUTO ]; then
        # Pick whichever arch has more total free GPUs right now.
        best=""; best_n=-1
        for try in A100 V100; do
            tot=0
            while IFS= read -r line; do
                n=$(awk '{print $1}' <<< "$line")
                g=$(awk '{print $2}' <<< "$line")
                s=$(awk '{print $3}' <<< "$line")
                if match_arch "$g" "$try" && is_live "$s"; then
                    tot=$(( tot + $(node_free_gpus "$n") ))
                fi
            done < <(sinfo -h -p "$PARTITION" -N -o "%n %G %t" 2>/dev/null)
            if [ "$tot" -gt "$best_n" ]; then best="$try"; best_n="$tot"; fi
        done
        [ -z "$best" ] || [ "$best_n" -lt 1 ] && {
            echo "ERROR: no free A100/V100 GPUs in '$PARTITION'" >&2; exit 1; }
        ARCH="$best"
        echo "[auto] selected arch=$ARCH  (free GPUs in this arch = $best_n)"
    fi

    while IFS= read -r line; do
        n=$(awk '{print $1}' <<< "$line")
        g=$(awk '{print $2}' <<< "$line")
        s=$(awk '{print $3}' <<< "$line")
        if match_arch "$g" "$ARCH" && is_live "$s"; then
            f=$(node_free_gpus "$n")
            if [ "$f" -ge 1 ]; then
                POOL+=("$n"); FREE_GPU["$n"]="$f"
            fi
        fi
    done < <(sinfo -h -p "$PARTITION" -N -o "%n %G %t" 2>/dev/null)
    [ "${#POOL[@]}" -eq 0 ] && {
        echo "ERROR: no $ARCH nodes in '$PARTITION' with free GPUs" >&2; exit 1; }

    MIN_FREE=999
    for n in "${POOL[@]}"; do
        [ "${FREE_GPU[$n]}" -lt "$MIN_FREE" ] && MIN_FREE="${FREE_GPU[$n]}"
    done
    GPUS_PER_NODE="$MIN_FREE"
    NODELIST=$(IFS=','; echo "${POOL[*]}")
fi

# --- auto-size CPU / mem from free resources --------------------------
MIN_FREE_CPU=999999
MIN_FREE_MEM_MB=999999999
for n in "${POOL[@]}"; do
    read _ fc fmm <<< "$(node_free "$n")"
    [ "${fc:-0}"  -lt "$MIN_FREE_CPU"    ] && MIN_FREE_CPU="$fc"
    [ "${fmm:-0}" -lt "$MIN_FREE_MEM_MB" ] && MIN_FREE_MEM_MB="$fmm"
done

if [ -z "$CPT" ]; then
    # Default: 8 cpus per GPU (matches train_all_data*.sbatch's 32/4-GPU and
    # 64/8-GPU convention — enough for the DataLoader workers + Python
    # overhead, no more). Capped at min-free so we don't oversubscribe.
    CPT=$(( GPUS_PER_NODE * 8 ))
    [ "$CPT" -gt "$MIN_FREE_CPU" ] && CPT="$MIN_FREE_CPU"
    [ "$CPT" -lt 1 ] && CPT=1
fi
if [ -z "$MEM" ]; then
    MEM_GB=$(( MIN_FREE_MEM_MB * 9 / 10 / 1024 ))
    [ "$MEM_GB" -lt 4 ] && MEM_GB=4
    MEM="${MEM_GB}G"
fi

NNODES="${#POOL[@]}"
WORLD=$(( NNODES * GPUS_PER_NODE ))

cat <<EOF
Multi-node DDP training plan:
  mode         : $MODE${ARCH:+  (arch=$ARCH)}
  partition    : $PARTITION
  experiment   : $EXP
  nodelist     : $NODELIST
  nodes        : $NNODES
  gpus/node    : $GPUS_PER_NODE
  world_size   : $WORLD
  cpus/task    : $CPT  (min free across pool = $MIN_FREE_CPU)
  mem/node     : $MEM  (min free across pool = $(( MIN_FREE_MEM_MB / 1024 ))G)
EOF

JOB_NAME="gbm_train_multi_$( [ "$MODE" = arch ] && echo "$ARCH" || echo "node" )"

SBATCH_ARGS=(
    --parsable
    -J "$JOB_NAME"
    -p "$PARTITION"
    -N "$NNODES"
    --nodelist="$NODELIST"
    --ntasks-per-node=1
    --gres="gpu:$GPUS_PER_NODE"
    --mem="$MEM"
    --cpus-per-task="$CPT"
)

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "[dry-run] would submit:"
    echo "  sbatch ${SBATCH_ARGS[*]} sbatch/train_multi_node.sbatch '$EXP' '$EPOCHS'"
    exit 0
fi

JID=$(sbatch "${SBATCH_ARGS[@]}" sbatch/train_multi_node.sbatch "$EXP" "$EPOCHS")
echo "Submitted: $JID"
