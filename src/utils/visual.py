"""
Author: Arash Fatehi
Date:   07.11.2022
File:   visual.py
"""
# Python Imports
import os

# Library Imports
import numpy as np
import tifffile
import imageio
from skimage import measure


def visualize_predictions(_inputs,
                          _labels,
                          _predictions,
                          _output_dir,
                          _produce_tif_files=True,
                          _produce_gif_files=True,
                          _produce_3d_model=True):

    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

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
        prediction_to_verticies(_inputs=_inputs,
                                _label=_labels,
                                _prediction=_predictions,
                                _output_dir=_output_dir)



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


def prediction_to_verticies(_inputs=None,
                            _label=None,
                            _prediction=None,
                            _output_dir=None):
    assert _output_dir is not None, \
            "Output directory is None, where should I save the result?"

    if _inputs is not None:
        for index, _input in enumerate(_inputs):
            tile_3d_to_verticies(_input, f"channel_{index}")

    if _prediction is not None:
        tile_3d_to_verticies(_prediction[0], "prediction")

    if _label is not None:
        tile_3d_to_verticies(_label[0], "label")

def tile_3d_to_tif(_input, _output_file_path):
    tifffile.imwrite(_output_file_path,
                     _input,
                     shape=_input.shape,
                     imagej=True,
                     metadata={'spacing': _input.shape[0],
                               'unit': 'um',
                               'axes': 'ZYX',
                               })

def tile_3d_to_gif(_input, _output_file_path):
    image = _input.astype(np.uint8) * 255
    with imageio.get_writer(_output_file_path, mode='I') as writer:
        for index in range(_input.shape[0]):
            writer.append_data(image[index])

def tile_3d_to_verticies(_input, _output_file_name):
    return
    verts, faces, _, _ = measure.marching_cubes(_input, 0.1)
    np.save(f"{_output_file_name}_verts.npy", verts)
    np.save(f"{_output_file_name}_faces.npy", faces)
