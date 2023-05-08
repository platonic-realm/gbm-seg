"""
Author: Arash Fatehi
Date:   03.05.2022
"""


# Python Imports
import os
import logging
import subprocess
import shutil
import yaml

# Library Imports

# Local Imports
from src.utils.misc import create_dirs_recursively, copy_directory
from src.utils.args import read_configurations
from train import main_train


def experiment_exists(_root_path, _name) -> bool:
    result = False
    if os.path.exists(_root_path):
        for item in os.listdir(_root_path):
            if os.path.isdir(
                    os.path.join(_root_path,
                                 item)) and item == _name:
                result = True
    return result


def list_experiments(_root_path):
    print("Experiments:")
    if os.path.exists(_root_path):
        for item in os.listdir(_root_path):
            if os.path.isdir(os.path.join(_root_path,
                                          item)):
                print(f"* {item}")


def infer_experiment(_name: str,
                     _root_path: str,
                     _snapshot: str,
                     _batch_size: int,
                     _sample_dimension: list,
                     _stride: list,
                     _scale: int):
    pass


def train_experiment(_name: str,
                     _root_path: str):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configurations(configs_path)
    main_train(configs)


def delete_experiment(_name: str,
                      _root_path: str):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    answer = input("Are you sure? (y/n) [default=n]: ")
    if answer.lower() == "y":
        logging.info("Removing the experiment: %s", _name)
        shutil.rmtree(os.path.join(_root_path, _name))
        logging.info('Experiment "%s" has been deleted', _name)


def create_new_experiment(_name: str,
                          _root_path: str,
                          _source_path: str,
                          _dataset_path: str,
                          _semi_supervised: bool = False,
                          _configs: str = None):

    destination_path = os.path.join(_root_path, f'{_name}/')
    logging.info("Creating a new experiment in '%s%s/'", _root_path, _name)
    if os.path.exists(destination_path):
        message = f"Experiment already exists: {destination_path}"
        raise FileExistsError(message)

    logging.info("Copying project's source code")
    create_dirs_recursively(os.path.join(destination_path, 'dummy'))

    code_path = os.path.join(destination_path, 'code/')
    create_dirs_recursively(os.path.join(code_path, 'dummy'))
    copy_directory(_source_path,
                   code_path,
                   ['.git', 'tags'])

    logging.info("Copying experiment's datasets")
    new_dataset_path = os.path.join(destination_path, 'datasets')
    os.makedirs(new_dataset_path, exist_ok=True)

    new_ds_train_path = os.path.join(new_dataset_path, 'ds_train')
    create_dirs_recursively(os.path.join(new_ds_train_path, 'dummy'))
    copy_directory(os.path.join(_dataset_path, 'ds_train'),
                   new_ds_train_path, [])

    new_ds_valid_path = os.path.join(new_dataset_path, 'ds_valid')
    create_dirs_recursively(os.path.join(new_ds_valid_path, 'dummy'))
    copy_directory(os.path.join(_dataset_path, 'ds_valid'),
                   new_ds_valid_path, [])

    new_ds_test_path = os.path.join(new_dataset_path, 'ds_test')
    create_dirs_recursively(os.path.join(new_ds_test_path, 'dummy'))
    copy_directory(os.path.join(_dataset_path, 'ds_test'),
                   new_ds_test_path, [])

    if _semi_supervised:
        new_ds_unlabeled_path = os.path.join(new_dataset_path, 'ds_unlabeld')
        create_dirs_recursively(os.path.join(new_ds_unlabeled_path, 'dummy'))
        copy_directory(os.path.join(_dataset_path, 'ds_unlabeled'),
                       new_ds_unlabeled_path, [])

    logging.info("Saving the requirements file to '%s'",
                 destination_path)
    # Run the 'pip freeze' command and capture the output
    output = subprocess.check_output(["pip", "freeze"])
    output_str = output.decode().strip()

    # Write the output to a file named 'requirements.txt'
    with open(os.path.join(destination_path, 'requirements.txt'), "w",
              encoding='UTF-8') as f:
        f.write(output_str)

    logging.info("Saving the configuration file to '%s'",
                 destination_path)

    if _configs is None:
        logging.warning("Don't forget to edit the configurations")

        with open('./configs/template.yaml',
                  encoding='UTF-8') as template_file:
            configs = yaml.safe_load(template_file)

        configs['root_path'] = destination_path

        configs['trainer']['model']['channels'] = \
            configs['experiments']['default_channels']

        configs['trainer']['epochs'] = \
            configs['experiments']['default_epochs']

        configs['trainer']['loss_weights'] = \
            configs['experiments']['default_loss_weights']

        configs['trainer']['report_freq'] = \
            configs['experiments']['default_report_freq']

        configs['trainer']['train_ds']['path'] = \
            f"{configs['experiments']['default_data_path']}ds_train/"

        configs['trainer']['train_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['trainer']['train_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        configs['trainer']['train_ds']['augmentation']['workers'] = \
            configs['experiments']['default_aug_workers']

        configs['trainer']['valid_ds']['path'] = \
            f"{configs['experiments']['default_data_path']}ds_valid/"

        configs['trainer']['valid_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['trainer']['valid_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        configs['trainer']['unlabeled_ds']['path'] = \
            f"{configs['experiments']['default_data_path']}ds_unlabeled/"

        configs['trainer']['unlabeled_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['trainer']['unlabeled_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        configs['inference']['inference_ds']['path'] = \
            f"{configs['experiments']['default_data_path']}ds_test/"

        configs['inference']['inference_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['inference']['inference_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        del configs['experiments']

        with open(os.path.join(destination_path, 'configs.yaml'), 'w',
                  encoding='UTF-8') as config_file:
            yaml.dump(configs,
                      config_file,
                      default_flow_style=None,
                      sort_keys=False)
