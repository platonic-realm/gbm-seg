"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts

# Library Imports
from torch.distributed.elastic.multiprocessing.errors import record

# Local Imports
from src.utils import args
from src.train.trainer_unet3d_me import Unet3DMETrainer
from src.train.trainer_unet3d import Unet3DTrainer
from src.train.trainer_unet3d_ss import Unet3DSelfTrainer
from src.utils.misc import configure_logger


# Training the network in a Supervised manner
@record
def supervised(_configs):
    if _configs['trainer']['model']['name'] == 'unet_3d':
        trainer = Unet3DTrainer(_configs)
    elif _configs['trainer']['model']['name'] == 'unet_3d_me':
        trainer = Unet3DMETrainer(_configs)
    else:
        assert False, "Please provide a valid model name in the config file"

    trainer.train()


# Training the network in a Self Supervised manner
# I am interpreting the stacks of images as frames
# and ask the network to interpolate them
# The output tensor and loss are the only changes
# to the network's architecture
@record
def self_supervised(_configs):
    if _configs['trainer']['model']['name'] == 'unet_3d':
        trainer = Unet3DSelfTrainer(_configs)
    else:
        assert False, "Please provide a valid model name in the config file"

    trainer.train()


# Fine tuning the network trained with Self Supervised method
# for semantic segmentation
@record
def fine_tuninig(_coinfigs):
    pass


if __name__ == '__main__':
    configs = args.parse("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    if configs['logging']['log_summary']:
        args.summerize(configs)
    if configs['trainer']['mode'] == 'supervised':
        supervised(configs)
    elif configs['trainer']['mode'] == 'self_supervised':
        self_supervised(configs)
