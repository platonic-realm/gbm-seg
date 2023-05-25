"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts
import os
import sys
import logging
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
import yaml

# Library Imports
import torch


def parse_exper() -> None:
    # Create the top-level parser
    parser = ArgumentParser(description='configuring GBM experiments')

    # Add the debug option
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help='enable debugging mode')

    # Create subparsers for each item in the list
    subparsers = \
        parser.add_subparsers(title='commands', dest='action')

    # Define a subparser for the 'tui' action
    subparsers.add_parser('tui',
                          help='manage experiments in a ncurses ui')

    # Define a subparser for the 'list' action
    list_parser = \
        subparsers.add_parser('list',
                              help='provides a list of created experiments')
    list_parser.add_argument('-r',
                             '--root',
                             action='store_true',
                             help='the root directory of experminets')
    list_parser.add_argument('-s',
                             '--snapshots',
                             action='store',
                             help='list the snapshots of an experiment')

    # Define a subparser for the 'create' action
    create_parser = \
        subparsers.add_parser('create',
                              help='create a new experiment')
    create_parser.add_argument('name',
                               help='name of the experiment to create')

    create_parser.add_argument('-bs',
                               '--batch-size',
                               action='store',
                               default=8,
                               help='set the batch size for training')

    # Define a subparser for the 'delete' action
    delete_parser = \
        subparsers.add_parser('delete',
                              help='deletes the selected experiment')
    delete_parser.add_argument('name',
                               help='name of the experiment to delete')

    # Define a subparser for the 'train' action
    train_parser = \
        subparsers.add_parser('train',
                              help='start/continue training')
    train_parser.add_argument('name',
                              help='name of the experiment.')

    # Define a subparser for the 'infer' action
    infer_parser = \
        subparsers.add_parser('infer',
                              help='create an inference session')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-s',
                              '--snapshot',
                              action='store',
                              required=True,
                              help='select the snapshot for inference')

    infer_parser.add_argument('-bs',
                              '--batch-size',
                              action='store',
                              default=8,
                              help='set the batch size for inference')

    infer_parser.add_argument('-sd',
                              '--sample-dimension',
                              action='store',
                              default='12, 256, 256',
                              help='set sample dimension for inference')

    infer_parser.add_argument('-st',
                              '--stride',
                              action='store',
                              default='1, 64, 64',
                              help='set the stride for inference')

    infer_parser.add_argument('-sf',
                              '--scale-factor',
                              action='store',
                              default=1,
                              help='set the scale for interpolation')

    infer_parser.add_argument('-cm',
                              '--channel-map',
                              action='store',
                              default='0, 1, 2',
                              help='the channel map of inference dataset')

    # Parse the arguments
    args = parser.parse_args()

    with open('./configs/template.yaml', encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return args, configs


def read_configurations(_config_path: str):
    with open(_config_path, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    configs = add_filename_to_configs(configs,
                                      _config_path)

    return sanity_check(configs)


def parse(_description: str):
    parser = ArgumentParser(description=_description)
    parser.add_argument("-c", "--config",
                        default="./configs/train.yaml",
                        help="Configuration's path")

    arguments = parser.parse_args()

    return read_configurations(arguments.config)


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
