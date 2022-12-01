"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts
import os
import logging
from argparse import ArgumentParser
from io import StringIO
import yaml


def parse(_description: str) -> None:
    parser = ArgumentParser(description=_description)
    parser.add_argument("-c", "--config",
                        default="./configs/train.yaml",
                        help="Configuration's path")

    with open(parser.parse_args().config) as config_file:
        configs = yaml.safe_load(config_file)

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

    if _configs['logging']['log_std']:
        _configs['logging']['tqdm'] = False

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

    _configs['trainer']['ddp']['enabled'] = ddp

    return _configs


def summerize(_configs: dict) -> None:
    with StringIO() as configs_dump:
        yaml.dump(_configs,
                  configs_dump,
                  default_flow_style=None,
                  sort_keys=False)
        logging.info("Configurations\n%s******************",
                     configs_dump.getvalue())
