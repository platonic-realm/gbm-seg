"""
Author: Arash Fatehi
Date:   06.12.2022
"""

# Library Imports
import numpy as np
from numpy import array

# Local Imports
from src.utils.misc import to_numpy


def scaler_to_vector(_input: array) -> array:

    _input = to_numpy(_input)

    background = np.zeros(_input.shape)
    background[_input == 0] = 1

    sure = np.zeros(_input.shape)
    sure[_input == 1] = 1

    possible = np.zeros(_input.shape)
    possible[_input == 2] = 1

    result = np.stack([background, sure, possible])
    return result


def vector_to_scaler(_input: array) -> array:

    _input = to_numpy(_input)

    labels = np.zeros(_input.shape[1:], dtype=np.uint8)
    labels[_input[1] == 1.0] = 1
    labels[_input[2] == 1.0] = 2

    return labels
