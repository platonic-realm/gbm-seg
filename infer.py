"""
Author: Arash Fatehi
Date:   12.12.2022
"""

# Python Imprts

# Library Imports

# Local Imports
from src.utils.misc import configure_logger
from src.utils import args
from src.infer.inference import Inference


if __name__ == '__main__':
    configs = args.parse("Inferance -> GBM segmentation")
    configure_logger(configs)
    args.summerize(configs)

    inference = Inference(configs)
    inference.infer()