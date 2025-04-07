# Python Imprts
import os
import sys
from argparse import ArgumentParser
import yaml

# Library Imports

# Local Imports
from src.utils.misc import sanity_check


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

    infer_parser.add_argument('-in',
                              '--interpolation',
                              action='store',
                              default=False,
                              help='determines to interpolate or stack Z planes')

    # Define a subparser for the 'post processsing' action
    infer_parser = \
        subparsers.add_parser('psp',
                              help='post processsing to remove noises and artifacts')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot for visualizations')

    infer_parser.add_argument('-mc',
                              '--max-concurrent',
                              action='store',
                              required=True,
                              help='number of worker processes for post processing')

    # Define a subparser for the 'morph' action
    infer_parser = \
        subparsers.add_parser('morph',
                              help='do the morphometric analysis for a sample')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    infer_parser.add_argument('-sn',
                              '--sample-name',
                              action='store',
                              required=True,
                              help='the relative path to the sample')

    # Define a subparser for the 'blender' action
    infer_parser = \
        subparsers.add_parser('blender',
                              help='prepares for blender visualizations')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    infer_parser.add_argument('-sn',
                              '--sample-name',
                              action='store',
                              required=True,
                              help='the relative path to the sample')

    # Define a subparser for the 'render' action
    infer_parser = \
        subparsers.add_parser('render',
                              help='create blender visualizations')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    # Parse the arguments
    args = parser.parse_args()

    with open('./configs/template.yaml', encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if hasattr(args, 'name'):
        configs['root_path'] = os.path.join(configs['experiments']['root'],
                                            args.name)
    return args, configs


def parse_indep(_description: str):
    parser = ArgumentParser(description=_description)
    parser.add_argument("-c", "--config",
                        default="./configs/template.yaml",
                        help="Configuration's path")

    args = parser.parse_args()
    with open(args.config, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    return args, sanity_check(configs)
