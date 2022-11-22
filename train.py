"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imprts

# Library Imports

# Local Imports
import src.train.args as args
from src.utils.misc import configure_logger

if __name__ == '__main__':

    configs = args.parse("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    args.summerize(configs)

#    train_undet3d(_args=args)
