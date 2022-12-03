#!/usr/bin/env bash

set -o errexit  # fail in the case of error
set -o nounset  # fail if there is an unset variable
set -o pipefail # helps to catch pip issues

work_dir=$(dirname "$(readlink --canonicalize-existing "${0}" 2> /dev/null)")

if [ "$#" -ne 2 ]; then
    echo "Default: configs=train, "
    echo "Please provide configuration name and node's rank"
    config_name=train
    no_of_nodes=1
    node_rank=0
else
    config_name=$1
    node_rank=$2
fi

config_path=$work_dir/configs/$config_name.yaml
configs=$(cat $config_path)

script_path=$work_dir/train.py


yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

model_name=$(yaml $config_path "['trainer']['model']['name']")
model_tag=$(yaml $config_path "['trainer']['model']['tag']")
no_of_nodes=$(yaml $config_path "['trainer']['ddp']['no_of_nodes']")
rdzv_backend=$(yaml $config_path "['trainer']['ddp']['rdzv_backend']")
rdzv_endpoint=$(yaml $config_path "['trainer']['ddp']['rdzv_endpoint']")

result_dir=../results/$config_name-$model_name-$model_tag

torchrun \
    --nproc_per_node=gpu \
    --nnodes=$no_of_nodes \
    --node_rank=$node_rank \
    --rdzv_id=643 \
    --rdzv_backend=$rdzv_backend \
    --rdzv_endpoint=$rdzv_endpoint \
    $script_path -c $config_path -rp $result_dir

