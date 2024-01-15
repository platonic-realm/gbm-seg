# Python Imports
from abc import ABC, abstractmethod

# Library Imports
import numpy as np
import tifffile
import imageio

# Local Imports


class Painter3D(ABC):

    @abstractmethod
    def save(self, _input: np.ndarray, _file_path: str):
        pass


class GIFPainter3D(Painter3D):

    def save(self, _input: np.ndarray, _file_path: str):

        image = _input.astype(np.uint8)
        with imageio.get_writer(_file_path, mode='I') as writer:
            for index in range(_input.shape[0]):
                writer.append_data(image[index])


class TIFPainter3D(Painter3D):

    def save(self, _input: np.ndarray, _file_path: str):

        tifffile.imwrite(_file_path,
                         _input,
                         shape=_input.shape,
                         metadata={'axes': 'ZYX', 'fps': 10.0})
