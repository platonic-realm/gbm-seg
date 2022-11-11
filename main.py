"""
Author: Arash Fatehi
Date:   20.10.2022
File:   main.py
"""

# Python Imprts
import sys
import logging

# Library Imports
from src.configs import LOG_LEVEL
from src.train.unet3d import train_undet3d

if __name__ == '__main__':

    logging.basicConfig(
                    level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("testing.log"),
                              logging.StreamHandler(sys.stdout)])

    train_undet3d(_epochs=50,
                  _no_of_channles=3,
                  _feature_maps=(64, 128, 256),
                  _batch_size=16,
                  _sample_dimension=(12, 256, 256),
                  _training_ds_path='/home/afatehi/gbm/data/GBM-Train-DS',
                  _validation_ds_path='/home/afatehi/gbm/data/GBM-Valid-DS',
                  _validation_no_of_batches=200,
                  _validation_visulization=True,
                  _pixel_per_step=(1, 16, 16),
                  _learning_rate=0.01,
                  _data_parallelism=True)
