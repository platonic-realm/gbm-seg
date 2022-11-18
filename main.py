"""
Author: Arash Fatehi
Date:   20.10.2022
File:   main.py
"""

# Python Imprts
import sys
import logging

# Library Imports
from src.args import parse_arguments, summerize_args
from src.configs import LOG_LEVEL
from src.train.unet3d import train_undet3d

if __name__ == '__main__':


    args = parse_arguments("Training Unet3D for GBM segmentation")

    summerize_args(args)

    logging.basicConfig(
                    level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("testing.log"),
                              logging.StreamHandler(sys.stdout)])

    train_undet3d(_args=args)
