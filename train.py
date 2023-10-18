"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts
import logging

# Library Imports
import torch
from torch.distributed.elastic.multiprocessing.errors import record

# Local Imports
from src.utils import args
from src.train.trainer_unet3d_me import Unet3DMETrainer
from src.train.trainer_unet3d import Unet3DTrainer
from src.train.trainer_unet3d_ss import Unet3DSemiTrainer
from src.train.trainer_swinunetr import SwinUNETRTrainer
from src.utils.misc import configure_logger


@record
def supervised(_configs):
    def label_correction_function(_labels):
        _labels = _labels.astype(int)
        _labels[_labels > 0] = 1
        _labels[_labels == 1] = 2
        _labels[_labels == 0] = 1
        _labels[_labels == 2] = 0
        return _labels

    if _configs['trainer']['model']['name'] == 'unet_3d':
        trainer = Unet3DTrainer(_configs,
                                label_correction_function)
    elif _configs['trainer']['model']['name'] == 'unet_3d_me':
        trainer = Unet3DMETrainer(_configs,
                                  label_correction_function)
    elif _configs['trainer']['model']['name'] == 'swinunetr':
        trainer = SwinUNETRTrainer(_configs,
                                   label_correction_function)
    else:
        assert False, "Please provide a valid model name in the config file"

    trainer.train()


@record
def semi_supervised(_configs):
    if _configs['trainer']['model']['name'] == 'unet_3d_ss':
        def label_correction_function(_labels):
            _labels = _labels.astype(int)
            _labels[_labels > 0] = 1
            _labels[_labels == 1] = 2
            _labels[_labels == 0] = 1
            _labels[_labels == 2] = 0
            return _labels

        trainer = Unet3DSemiTrainer(_configs,
                                    label_correction_function)
    else:
        assert False, "Please provide a valid model name in the config file"

    trainer.train()


def main_train(_configs):
    if _configs['logging']['log_summary']:
        args.summerize(_configs)

    if _configs['trainer']['cudnn_benchmark']:
        torch.backends.cudnn.benchmark = True
        logging.info("Enabling cudnn benchmarking")

    torch.manual_seed(88233474)
    if _configs['trainer']['mode'] == 'supervised':
        supervised(_configs)
    elif _configs['trainer']['mode'] == 'semi_supervised':
        semi_supervised(_configs)


if __name__ == '__main__':
    configs = args.parse("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    main_train(configs)
