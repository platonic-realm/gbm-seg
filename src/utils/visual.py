"""
Author: Arash Fatehi
Date:   07.11.2022
File:   visual.py
"""
# Python Imports
import os
from abc import ABC, abstractmethod

# Library Imports
import torch
import torch.nn.functional as Fn
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import imageio
from skimage import measure

# Local Imports
from src.utils.misc import to_numpy


class Visualizer(ABC):
    def __init__(self,
                 _generate_tif: bool = True,
                 _generate_gif: bool = True,
                 _generate_mesh: bool = False,
                 _generate_scatter: bool = False):

        self.generate_tif = _generate_tif
        self.generate_gif = _generate_gif
        self.generate_mesh = _generate_mesh
        self.generate_scatter = _generate_scatter

    @abstractmethod
    def draw_channels(self,
                      _input,
                      _output_dir,
                      _multiplier):
        pass

    @abstractmethod
    def draw_labels(self,
                    _input,
                    _output_dir,
                    _multiplier):
        pass

    @abstractmethod
    def draw_predictions(self,
                         _input,
                         _output_dir,
                         _multiplier):
        pass

    def _tensor_2D_to_tif(self,
                          _input,
                          _file_path: str,
                          _imagej: bool = True
                          ):

        if not self.generate_tif:
            return

        assert _file_path is not None, \
            "_file_path is None, where should I save the result?"

        tifffile.imwrite(_file_path,
                         _input,
                         shape=_input.shape,
                         imagej=_imagej,
                         metadata={'axes': 'YX'}
                         )

    def _tensor_3D_to_tif(self,
                          _input,
                          _file_path: str,
                          _imagej: bool = True
                          ):

        if not self.generate_tif:
            return

        if len(_input.shape) == 2:
            self._tensor_2D_to_tif(_input,
                                   _file_path,
                                   _imagej)
            return

        assert _file_path is not None, \
            "_file_path is None, where should I save the result?"

        tifffile.imwrite(_file_path,
                         _input,
                         shape=_input.shape,
                         imagej=_imagej,
                         metadata={'axes': 'ZYX', 'fps': 10.0}
                         )

    def _tensor_3D_to_gif(self,
                          _input,
                          _file_path: str):

        if not self.generate_gif:
            return

        assert _file_path is not None, \
            "_file_path is None, where should I save the result?"

        image = _input.astype(np.uint8)
        with imageio.get_writer(_file_path, mode='I') as writer:
            for index in range(_input.shape[0]):
                writer.append_data(image[index])

    def _tensor_3D_to_mesh(self,
                           _input,
                           _file_path: str):

        if not self.generate_mesh:
            return

        assert _file_path is not None, \
            "_file_path is None, where should I save the result?"

        _input = Fn.pad(torch.from_numpy(_input),
                        (1, 1, 1, 1, 1, 1),
                        "constant",
                        0)
        _input = to_numpy(_input)
        verts, faces, _, _ = measure.marching_cubes(_input, 0.1)
        np.save(f"{_file_path}_verts.npy", verts)
        np.save(f"{_file_path}_faces.npy", faces)

    def _tensor_3D_to_scatter(self,
                              _input,
                              _file_path: str):
        # pylint: disable=invalid-name
        if not self.generate_scatter:
            return

        assert _file_path is not None, \
            "_file_path is None, where should I save the result?"

        image = _input * 255
        image = image.astype(np.uint8)

        figure = plt.figure()
        ax = figure.gca(projection='3d')
        ax.set_aspect('auto')

        ax.voxels(image, edgecolor='k')

        plt.savefig(_file_path)


class VisualizerUnet3D(Visualizer):

    def draw_channels(self,
                      _input,
                      _output_dir: str,
                      _multiplier=255):

        if _input is None:
            return

        assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

        _input = to_numpy(_input)
        _input = _input * _multiplier

        for index, _channel in enumerate(_input):
            self._tensor_3D_to_tif(_channel,
                                   os.path.join(
                                       _output_dir,
                                       f"channel_{index}.tif"))

            self._tensor_3D_to_gif(_channel,
                                   os.path.join(
                                       _output_dir,
                                       f"channel_{index}.gif"))

    def draw_labels(self,
                    _input,
                    _output_dir: str,
                    _multiplier=1):

        if _input is None:
            return

        assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

        _input = to_numpy(_input)
        # To visualy separate classes from each other
        _input = _input * _multiplier
        _input = _input.astype(np.uint8)

        self._tensor_3D_to_tif(_input,
                               os.path.join(
                                   _output_dir,
                                   "labels.tif"))

        self._tensor_3D_to_gif(_input,
                               os.path.join(
                                   _output_dir,
                                   "labels.gif"))

        self._tensor_3D_to_mesh(_input,
                                os.path.join(
                                    _output_dir,
                                    "labels"))

    def draw_predictions(self,
                         _input,
                         _output_dir: str,
                         _multiplier=1):
        if _input is None:
            return

        assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

        _input = to_numpy(_input)
        # _input = vector_to_scaler(_input)

        # To visualy separate classes from each other
        _input = _input * _multiplier
        _input = _input.astype(np.uint8)

        self._tensor_3D_to_tif(_input,
                               os.path.join(
                                   _output_dir,
                                   "prediction.tif"))

        self._tensor_3D_to_gif(_input,
                               os.path.join(
                                   _output_dir,
                                   "prediction.gif"))

        self._tensor_3D_to_mesh(_input,
                                os.path.join(
                                    _output_dir,
                                    "prediction"))
