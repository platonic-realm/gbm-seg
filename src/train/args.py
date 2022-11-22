"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts
import logging
from argparse import ArgumentParser
from io import StringIO
import yaml


def parse(_description: str) -> None:
    parser = ArgumentParser(description=_description)
    parser.add_argument("-c", "--config",
                        default="./train.yaml",
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

    if _configs['trainer']['visualization']:
        assert not _configs['trainer']['visualization_path'] is None, \
               "Please provide path to the validation dataset"

    if _configs['logging']['log_std']:
        _configs['logging']['tqdm'] = False

    # Adding  

    return _configs


def summerize(_configs: dict) -> None:
    with StringIO() as configs_dump:
        yaml.dump(_configs,
                  configs_dump,
                  default_flow_style=None,
                  sort_keys=False)
        logging.info("Configurations\n%s******************",
                     configs_dump.getvalue())
