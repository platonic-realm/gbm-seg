"""
Author: Arash Fatehi
Date:   01.11.2022
"""

# Python Imports
import logging
import sys
import os
from pathlib import Path

# Library Imports
import torch


class RunningAverage():
    def __init__(self):
        self.value: float = 0
        self.counter: int = 0

    def add(self, _value: any) -> None:
        self.value += float(_value)
        self.counter += 1

    def calcualte(self) -> float:
        result: float = self.value / self.counter
        self.value = 0.0
        self.counter = 0
        return result


def configure_logger(_configs: dict) -> None:
    LOG_LEVEL = _configs['logging']['log_level']
    log_file = os.path.join(_configs['trainer']['result_path'],
                            _configs['logging']['log_file'])
    log_std = _configs['logging']['log_std']
    ddp = _configs['trainer']['ddp']['enabled']

    # Adding node and rank info to log format if dpp is enabled
    try:
        rank = int(os.environ["RANK"])
        node = int(os.environ["GROUP_RANK"])
        ddp = True
    except KeyError:
        ddp = False

    handlers = []
    if log_file is not None:
        log_file = Path(log_file)
        create_dirs_recursively(log_file)
        if ddp:
            log_stem = log_file.stem
            log_file = os.path.dirname(log_file) + f"/{log_stem}-{rank}.log"
        handlers.append(logging.FileHandler(log_file))
    if log_std:
        handlers.append(logging.StreamHandler(sys.stdout))

    if ddp:
        log_format = f"%(asctime)s [%(levelname)s] [{node},{rank}] %(message)s"
    else:
        log_format = "%(asctime)s [%(levelname)s] %(message)s"

    logging.basicConfig(level=LOG_LEVEL,
                        format=log_format,
                        handlers=handlers)

    logging.info("Log Level: %s", LOG_LEVEL)


def create_dirs_recursively(_path: str):
    dir_path = os.path.dirname(_path)
    path: Path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(_gpu_tensor):
    return _gpu_tensor.clone().detach().to('cpu').numpy()


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
