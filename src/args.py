"""
Author: Arash Fatehi
Date:   11.11.2022
"""

# Python Imports
import sys
from argparse import ArgumentParser
import yaml


def parse_arguments(_description):
    parser = ArgumentParser(description=_description)

    parser.add_argument("-c", "--config",
                        default="./conf.yaml",
                        help="Configuration's path")

    with open(parser.parse_args().config) as config_file:
        configs = yaml.safe_load(config_file)

    return configs


def summerize_args(_args):
    print("*********************")
    print("Configurations")
    print("*********************")
    yaml.dump(_args, 
              sys.stdout, 
              default_flow_style=None,
              sort_keys=False)
    print("*********************")
