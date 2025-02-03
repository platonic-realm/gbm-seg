# Python Imports
import os
import logging
import subprocess
import shutil
import yaml
import math

# Library Imports

# Local Imports
from src.utils.misc import create_dirs_recursively,\
        copy_directory, read_configs, resize_and_copy
from train import main_train
from infer import main_infer


def experiment_exists(_root_path, _name) -> bool:
    result = False
    if os.path.exists(_root_path):
        for item in os.listdir(_root_path):
            if os.path.isdir(
                    os.path.join(_root_path,
                                 item)) and item == _name:
                result = True
                break
    return result


def list_experiments(_root_path):
    print("Experiments:")
    if os.path.exists(_root_path):
        for item in sorted(os.listdir(_root_path)):
            if os.path.isdir(os.path.join(_root_path,
                                          item)):
                print(f"* {item}")


def infer_experiment(_name: str,
                     _root_path: str,
                     _snapshot: str,
                     _batch_size: int,
                     _sample_dimension: list,
                     _stride: list,
                     _scale: int,
                     _interpolate: bool):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
    create_dirs_recursively(os.path.join(inference_root_path, 'dummy'))

    inference_tag =\
        f"{_snapshot}_{''.join(_sample_dimension)}_{''.join(_stride)}_{_scale}"

    _sample_dimension = [int(item) for item in _sample_dimension]
    _stride = [int(item) for item in _stride]
    _scale = int(_scale)
    _batch_size = int(_batch_size)

    inference_result_path = os.path.join(inference_root_path, inference_tag)
    if os.path.exists(inference_result_path):
        answer = input("Inference already exists,"
                       " overwrite? (y/n) [default=n]: ")
        if answer.lower() == "y":
            shutil.rmtree(inference_result_path)
        else:
            return
    create_dirs_recursively(os.path.join(inference_result_path, 'dummy'))

    configs['inference']['snapshot_path'] =\
        os.path.join(_root_path,
                     _name,
                     'results-train/snapshots/',
                     _snapshot)

    configs['inference']['result_dir'] = inference_result_path

    configs['inference']['inference_ds']['path'] =\
        os.path.join(_root_path,
                     _name,
                     'datasets/',
                     'ds_test/')

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['sample_dimension'] =\
        _sample_dimension

    configs['inference']['inference_ds']['pixel_stride'] = _stride

    configs['inference']['inference_ds']['scale_factor'] = _scale

    configs['inference']['inference_ds']['interpolate'] = _interpolate

    configs['inference']['inference_ds']['workers'] =\
        configs['trainer']['train_ds']['workers']

    inference_configs = configs['inference']

    with open(os.path.join(inference_result_path, 'configs.yaml'), 'w',
              encoding='UTF-8') as config_file:
        yaml.dump(inference_configs,
                  config_file,
                  default_flow_style=None,
                  sort_keys=False)

    main_infer(configs)


def train_experiment(_name: str,
                     _root_path: str):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)
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
                          _batch_size: int,
                          _voxel_size: list,
                          _semi_supervised: bool = False):

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
    resize_and_copy(os.path.join(_dataset_path, 'ds_train'),
                    new_ds_train_path,
                    _voxel_size)

    new_ds_test_path = os.path.join(new_dataset_path, 'ds_test')
    create_dirs_recursively(os.path.join(new_ds_test_path, 'dummy'))
    resize_and_copy(os.path.join(_dataset_path, 'ds_test'),
                    new_ds_test_path,
                    _voxel_size)

    logging.info("Saving the requirements file to '%s'",
                 destination_path)
    # Run the 'pip freeze' command and capture the output
    output = subprocess.check_output(["pip", "freeze"])
    output_str = output.decode().strip()

    # Write the output to a file named 'requirements.txt'
    with open(os.path.join(destination_path, 'requirements.txt'), "w",
              encoding='UTF-8') as f:
        f.write(output_str)

    logging.warning("Don't forget to edit the configurations")

    with open('./configs/template.yaml',
              encoding='UTF-8') as template_file:
        configs = yaml.safe_load(template_file)

    batch_ratio = None
    if configs['experiments']['default_batch_size'] != _batch_size\
       and configs['experiments']['scale_lerning_rate_for_batch_size']:
        batch_ratio =\
            _batch_size / configs['experiments']['default_batch_size']

    if batch_ratio is not None:
        configs['trainer']['optim']['lr'] =\
                round(configs['trainer']['optim']['lr'] * math.sqrt(batch_ratio), 5)
        configs['trainer']['report_freq'] =\
            configs['trainer']['report_freq'] / batch_ratio

    configs['root_path'] = destination_path

    configs['trainer']['train_ds']['path'] = \
        f"{new_dataset_path}/ds_train/"

    configs['trainer']['train_ds']['batch_size'] = _batch_size

    configs['trainer']['valid_ds']['path'] = \
        f"{new_dataset_path}/ds_valid/"

    configs['trainer']['valid_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['path'] = \
        f"{new_dataset_path}/ds_test/"

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    del configs['experiments']

    with open(os.path.join(destination_path, 'configs.yaml'), 'w',
              encoding='UTF-8') as config_file:
        yaml.dump(configs,
                  config_file,
                  default_flow_style=None,
                  sort_keys=False)

    logging.info("Configuration file saved to '%s'",
                 destination_path)
