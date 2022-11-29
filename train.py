"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts

# Library Imports
from torch.distributed.elastic.multiprocessing.errors import record

# Local Imports
import src.train.args as args
from src.train.unet3d import Unet3DTrainer
from src.utils.misc import configure_logger


@record
def main():
    configs = args.parse("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    args.summerize(configs)

    trainer = Unet3DTrainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
