# Python Imports
import logging
import sys
import os
import shutil
from pathlib import Path
from io import StringIO
import subprocess
import shlex

# Library Imports
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

# Local Imports
from src.utils.metrics.memory import GPURunningMetrics
from src.utils.metrics.clfication import Metrics


def basic_logger() -> None:
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level='INFO',
                        format=log_format)


def configure_logger(_configs: dict, _log_to_file: bool = True) -> None:
    LOG_LEVEL = _configs['logging']['log_level']
    root_path = _configs['root_path']
    log_file = os.path.join(root_path,
                            _configs['trainer']['result_path'],
                            _configs['logging']['log_file'])
    log_std = _configs['logging']['log_std']

    handlers = []
    if _log_to_file and log_file is not None:
        log_file = Path(log_file)
        create_dirs_recursively(log_file)
        handlers.append(logging.FileHandler(log_file))
    if log_std:
        handlers.append(logging.StreamHandler(sys.stdout))

    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    logging.basicConfig(level=LOG_LEVEL,
                        format=log_format,
                        handlers=handlers)

    logging.info("Log Level: %s", LOG_LEVEL)


def create_dirs_recursively(_path: str):
    dir_path = os.path.dirname(_path)
    path: Path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


def copy_directory(_source_dir, _dest_dir, _exclude_list: list):
    for item in os.listdir(_source_dir):
        source_path = os.path.join(_source_dir, item)
        dest_path = os.path.join(_dest_dir, item)
        if item not in _exclude_list:
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)


def to_numpy(_gpu_tensor):
    if torch.is_tensor(_gpu_tensor):
        return _gpu_tensor.clone().detach().to('cpu').numpy()

    return _gpu_tensor


def expand_as_one_hot(_input, _c, _ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL,
    where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert _input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    _input = _input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(_input.size())
    shape[1] = _c

    if _ignore_index is not None:
        # create ignore_index mask for the result
        mask = _input.expand(shape) == _ignore_index
        # clone the src tensor and zero out ignore_index in the input
        _input = _input.clone()
        _input[_input == _ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(_input.device).scatter_(1, _input, 1)
        # bring back the ignore_index in the result
        result[mask] = _ignore_index
        return result

    # Else: scatter to get the one-hot tensor
    return torch.zeros(shape).to(_input.device).scatter_(1, _input, 1)


# Check the configurations and changes
# some if needed.
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

    if torch.cuda.device_count() == 0:
        _configs['trainer']['device'] = 'cpu'
        _configs['trainer']['mixed_precision'] = False
        _configs['inference']['device'] = 'cpu'

    if _configs['trainer']['device'] == 'cpu':
        _configs['trainer']['cudnn_benchmark'] = False
        _configs['trainer']['nvtx_patching'] = False

    return _configs


def blind_test(_model: nn.Module,
               _dataloader: DataLoader,
               _loss,
               _device: str,
               _no_of_classes: int,
               _metric_list: list):

    running_metrics = GPURunningMetrics(_device, _metric_list)

    _model.to(_device)

    for index, data in enumerate(_dataloader):

        sample = data['sample'].to(_device)
        labels = data['labels'].to(_device).long()

        with torch.no_grad():

            logits, results = _model(sample)

            metrics = Metrics(_no_of_classes,
                              results,
                              labels)
            loss = _loss(logits, labels)

            running_metrics.add(metrics.reportMetrics(_metric_list, loss))

    return running_metrics.calculate()


def read_configs(_config_path: str):
    with open(_config_path, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    return sanity_check(configs)


def summerize_configs(_configs: dict) -> None:
    with StringIO() as configs_dump:
        yaml.dump(_configs,
                  configs_dump,
                  default_flow_style=None,
                  sort_keys=False)
        logging.info("Configurations\n%s******************",
                     configs_dump.getvalue())


def blender_render(_inference_dir: str) -> None:

    # Generating Blender commands

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    gpu_index = 0
    gpu_capacity = 4

    # List all items in the current directory
    all_files = os.listdir(_inference_dir)

    # Filter out only the directories, excluding '.' and '..'
    directories = [item for item in all_files if os.path.isdir(os.path.join(_inference_dir,
                                                                            item)) and item not in ['.', '..']]
    commands = []
    for directory in directories:
        verts_path = os.path.join(_inference_dir, directory, "blender", "verts_distance.npy")
        faces_path = os.path.join(_inference_dir, directory, "blender", "faces_distance.npy")
        values_path = os.path.join(_inference_dir, directory, "blender", "values_distance.npy")
        result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_distance.blend")
        result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_distance.mp4")
        commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_index}; blender --background --python src/scripts/blender_render.py -- res/blender_template.blend {shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} {shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        gpu_index = (gpu_index + 1) % num_gpus

        verts_path = os.path.join(_inference_dir, directory, "blender", "verts_bumpiness.npy")
        faces_path = os.path.join(_inference_dir, directory, "blender", "faces_bumpiness.npy")
        values_path = os.path.join(_inference_dir, directory, "blender", "values_bumpiness.npy")
        result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_bumpiness.blend")
        result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_bumpiness.mp4")
        commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_index}; blender --background --python src/scripts/blender_render.py -- res/blender_template.blend {shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} {shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        gpu_index = (gpu_index + 1) % num_gpus

    compute_limit = num_gpus * gpu_capacity
    for i in range(int(len(commands)/compute_limit) + 1):
        sub_commands = commands[i*compute_limit:(i+1)*compute_limit]
        processes = [subprocess.Popen(cmd, shell=True, executable='/bin/bash') for cmd in sub_commands]

        # Wait for all processes to complete
        for proc in processes:
            proc.wait()
