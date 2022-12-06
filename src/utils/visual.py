"""
Author: Arash Fatehi
Date:   07.11.2022
File:   visual.py
"""
# Python Imports
import os

# Library Imports
import torch
import torch.nn.functional as Fn
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import imageio
from skimage import measure

# Local Imports
from src.utils.misc import to_numpy
from src.data.labels_cat import vector_to_scaler


def visualize_predictions(_inputs,
                          _labels,
                          _predictions,
                          _output_dir,
                          _produce_tif_files=True,
                          _produce_gif_files=True,
                          _produce_3d_model=True):

    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

    _inputs = to_numpy(_inputs)
    _labels = to_numpy(_labels)
    _predictions = to_numpy(_predictions)

    if _produce_tif_files:
        prediction_to_tif(_inputs=_inputs,
                          _label=_labels,
                          _prediction=_predictions,
                          _output_dir=_output_dir)

    if _produce_gif_files:
        prediction_to_gif(_inputs=_inputs,
                          _label=_labels,
                          _prediction=_predictions,
                          _output_dir=_output_dir)

    if _produce_3d_model:
        prediction_to_verticies(_prediction=_predictions,
                                _output_dir=_output_dir)


def visualize_scaler_predictions(_input: array,
                                 _output_dir: str,
                                 _produce_tif_files: bool = True,
                                 _produce_gif_files: bool = True,
                                 _produce_3d_model: bool = True) -> None:

    _input = to_numpy(_input)

    _input = _input * 127

    tifffile.imwrite(_output_dir,
                     _input,
                     shape=_input.shape,
                     imagej=True,
                     metadata={'axes': 'ZYX', 'fps': 10.0}
                     )


def visualize_vector_predictions(_input: array,
                                 _output_dir: str,
                                 _produce_tif_files: bool = True,
                                 _produce_gif_files: bool = True,
                                 _produce_3d_model: bool = True) -> None:

    _input = vector_to_scaler(_input)
    visualize_scaler_predictions(_input,
                                 _output_dir,
                                 _produce_tif_files,
                                 _produce_gif_files,
                                 _produce_3d_model)


def prediction_to_tif(_inputs=None,
                      _label=None,
                      _prediction=None,
                      _output_dir=None):
    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

    if _inputs is not None:
        for index, _input in enumerate(_inputs):
            tile_3d_to_tif(_input,
                           os.path.join(_output_dir,
                                        f"channel_{index}.tif"))

    if _prediction is not None:
        tile_3d_to_tif(_prediction[0],
                       os.path.join(_output_dir,
                                    "prediction.tif"))

    if _label is not None:
        tile_3d_to_tif(_label[0],
                       os.path.join(_output_dir,
                                    "label.tif"))


def prediction_to_gif(_inputs=None,
                      _label=None,
                      _prediction=None,
                      _output_dir=None):
    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

    if _inputs is not None:
        for index, _input in enumerate(_inputs):
            tile_3d_to_gif(_input,
                           os.path.join(_output_dir,
                                        f"channel_{index}.gif"))

    if _prediction is not None:
        tile_3d_to_gif(_prediction[0],
                       os.path.join(_output_dir,
                                    "prediction.gif"))

    if _label is not None:
        tile_3d_to_gif(_label[0],
                       os.path.join(_output_dir,
                                    "label.gif"))


def prediction_to_scatter_plot(_inputs=None,
                               _label=None,
                               _prediction=None,
                               _output_dir=None):

    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

    if _inputs is not None:
        for index, _input in enumerate(_inputs):
            tile_3d_to_scatter_plot(_input,
                                    os.path.join(_output_dir,
                                                 f"channel_{index}.png"))

    if _prediction is not None:
        tile_3d_to_scatter_plot(_prediction[0],
                                os.path.join(_output_dir,
                                             "prediction.png"))

    if _label is not None:
        tile_3d_to_scatter_plot(_label[0],
                                os.path.join(_output_dir,
                                             "label.png"))


def prediction_to_verticies(_inputs=None,
                            _label=None,
                            _prediction=None,
                            _output_dir=None):
    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

    if _inputs is not None:
        for index, _input in enumerate(_inputs):
            tile_3d_to_verticies(_input, os.path.join(_output_dir,
                                                      f"channel_{index}"))

    if _prediction is not None:
        tile_3d_to_verticies(_prediction[0], os.path.join(_output_dir,
                                                          "prediction"))

    if _label is not None:
        tile_3d_to_verticies(_label[0], os.path.join(_output_dir,
                                                     "label"))


def tile_3d_to_tif(_input, _output_file_path):
    tifffile.imwrite(_output_file_path,
                     _input,
                     shape=_input.shape,
                     imagej=True,
                     metadata={'spacing': _input.shape[0],
                               'unit': 'um',
                               'axes': 'TYX',
                               })


def tile_3d_to_gif(_input, _output_file_path):
    image = _input * 255
    image = image.astype(np.uint8)
    with imageio.get_writer(_output_file_path, mode='I') as writer:
        for index in range(_input.shape[0]):
            writer.append_data(image[index])


def tile_3d_to_scatter_plot(_input, _output_file_name):
    # pylint: disable=invalid-name
    image = _input * 255
    image = image.astype(np.uint8)

    figure = plt.figure()
    ax = figure.gca(projection='3d')
    ax.set_aspect('auto')

    ax.voxels(image, edgecolor='k')

    plt.savefig(_output_file_name)


def tile_3d_to_verticies(_input, _output_file_name):
    _input = Fn.pad(torch.from_numpy(_input),
                    (1, 1, 1, 1, 1, 1),
                    "constant",
                    0)
    _input = to_numpy(_input)
    verts, faces, _, _ = measure.marching_cubes(_input, 0.1)
    np.save(f"{_output_file_name}_verts.npy", verts)
    np.save(f"{_output_file_name}_faces.npy", faces)
