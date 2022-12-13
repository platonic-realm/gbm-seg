#!/usr/bin/env bash

set -o errexit  # fail in the case of error
set -o nounset  # fail if there is an unset variable
set -o pipefail # helps to catch pip issues

work_dir=$(dirname "$(readlink --canonicalize-existing "${0}" 2> /dev/null)")
config_name=train
no_of_nodes=1
node_rank=0
debug=false
standalone=false

# For parsing arguments I am using getops, its not a robust way of
# handling it, but it gets the job done and for the time being, 
# I don't need anything fancy...
# https://sookocheff.com/post/bash/parsing-bash-script-arguments-with-shopts/
while getopts ":dsc:n:" opt; do
    case ${opt} in
        d )
            debug=true
            ;;
        s )
            standalone=true
            ;;
        c )
            echo "Setting configuration to $OPTARG"
            config_name=$OPTARG
            ;;
        n )
            echo "Setting node rank: $OPTARG"
            node_rank=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

config_path=$work_dir/configs/$config_name.yaml
configs=$(cat $config_path)

script_path=$work_dir/train.py

# Using python to parse yaml files and extract config values
yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

model_name=$(yaml $config_path "['trainer']['model']['name']")
no_of_nodes=$(yaml $config_path "['trainer']['ddp']['no_of_nodes']")
rdzv_backend=$(yaml $config_path "['trainer']['ddp']['rdzv_backend']")
rdzv_endpoint=$(yaml $config_path "['trainer']['ddp']['rdzv_endpoint']")

if $debug
then
    echo "Runnign train.py in pudb"
    python -m pudb $script_path -c $config_path
elif $standalone
then
    echo "Running torchrun in standalone mode"
    torchrun \
        --standalone \
        --nproc_per_node=gpu \
        $script_path -c $config_path
else
    echo "Running torchrun in distributed mode"
    torchrun \
        --nproc_per_node=gpu \
        --nnodes=$no_of_nodes \
        --node_rank=$node_rank \
        --rdzv_id=643 \
        --rdzv_backend=$rdzv_backend \
        --rdzv_endpoint=$rdzv_endpoint \
        $script_path -c $config_path
fi
