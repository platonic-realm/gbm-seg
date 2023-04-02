"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts
import os
import logging
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
import yaml

# Library Imports
import torch


def parse(_description: str) -> None:
    parser = ArgumentParser(description=_description)
    parser.add_argument("-c", "--config",
                        default="./configs/train.yaml",
                        help="Configuration's path")

    arguments = parser.parse_args()

    with open(arguments.config) as config_file:
        configs = yaml.safe_load(config_file)

    configs = add_filename_to_configs(configs,
                                      arguments.config)

    return sanity_check(configs)


# Check the configurations and changes
# some if needed, for example:
# tqmd and log_std are mutually exclusive
def sanity_check(_configs: dict) -> dict:
    assert not _configs['trainer']['train_ds']['path'] is None, \
           "Please provide path to the training dataset"

    assert not _configs['trainer']['valid_ds']['path'] is None, \
           "Please provide path to the validation dataset"

    if _configs['trainer']['visualization']['enabled']:
        assert not _configs['trainer']['visualization']['path'] is None, \
               "Please provide path to store visualization files"

    if _configs['trainer']['tensorboard']['enabled']:
        assert not _configs['trainer']['tensorboard']['path'] is None, \
               "Please provide path for tensorboard logs"

    if _configs['trainer']['model']['name'] == 'unet_3d_ss':
        _configs['trainer']['mode'] = 'semi_supervised'
    else:
        _configs['trainer']['mode'] = 'supervised'

    if _configs['trainer']['mode'] not in ['supervised', 'semi_supervised']:
        _configs['trainer']['mode'] = 'supervised'

    if _configs['logging']['log_std']:
        _configs['logging']['tqdm'] = False

    if torch.cuda.device_count() == 0:
        _configs['trainer']['device'] = 'cpu'
        _configs['trainer']['mixed_precision'] = False
        _configs['inference']['device'] = 'cpu'

    if _configs['trainer']['device'] == 'cpu':
        _configs['trainer']['cudnn_benchmark'] = False
        _configs['trainer']['nvtx_patching'] = False

    # Checking if script has been run via torchrun
    # and add the environment variables to configs
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        _configs['trainer']['ddp']['local_rank'] = local_rank

        rank = int(os.environ["RANK"])
        _configs['trainer']['ddp']['rank'] = rank

        node = int(os.environ["GROUP_RANK"])
        _configs['trainer']['ddp']['node'] = node

        local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        _configs['trainer']['ddp']['local_size'] = local_size

        world_size = int(os.environ["WORLD_SIZE"])
        _configs['trainer']['ddp']['world_size'] = world_size

        master_address = os.environ["MASTER_ADDR"]
        _configs['trainer']['ddp']['master_address'] = master_address

        master_port = int(os.environ["MASTER_PORT"])
        _configs['trainer']['ddp']['master_port'] = master_port

        ddp = True

    except KeyError:
        ddp = False

    if ddp:
        _configs['trainer']['dp'] = False

    _configs['trainer']['ddp']['enabled'] = ddp

    return _configs


def add_filename_to_configs(_configs: dict,
                            _config_path: str) -> dict:
    file_name = Path(_config_path).stem
    _configs['tag'] = file_name
    return _configs


def summerize(_configs: dict) -> None:
    with StringIO() as configs_dump:
        yaml.dump(_configs,
                  configs_dump,
                  default_flow_style=None,
                  sort_keys=False)
        logging.info("Configurations\n%s******************",
                     configs_dump.getvalue())
