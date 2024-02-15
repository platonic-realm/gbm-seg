# Python Imports
import os
import random
from pathlib import Path

# Library Imports
import numpy as np
from torch import Tensor

# Local Imports
from src.utils.misc import to_numpy
from src.utils.visual.painter import GIFPainter3D, TIFPainter3D


class TrainVisualizer():

    def __init__(self,
                 _enabled: bool,
                 _chance: float,
                 _gif_painter: GIFPainter3D,
                 _tif_painter: TIFPainter3D,
                 _path: str,
                 _channels_multiplier: int = 255,
                 _labels_multiplier: int = 255,
                 _predictions_multiplier: int = 255):

        self.enabled = _enabled
        self.chance = _chance

        self.gif_painter = _gif_painter
        self.tif_painter = _tif_painter
        self.path = _path

        self.channels_multiplier = _channels_multiplier
        self.labels_multiplier = _labels_multiplier
        self.predictions_multiplier = _predictions_multiplier

    def draw(self,
             _channels: Tensor,
             _labels: Tensor,
             _predictions: Tensor,
             _epoch_id: int,
             _batch_id: int):

        if not self.enabled:
            return

        dice = random.random()
        if dice > self.chance:
            return

        # Randomly selecting one of the samples in the batch
        batch_size = len(_channels)
        sample_id = random.randint(0, batch_size-1)
        _channels = _channels[sample_id]
        _labels = _labels[sample_id]
        _predictions = _predictions[sample_id]

        path: Path = Path(self.path + f"/epoch-{_epoch_id}/batch-{_batch_id}/")
        path.mkdir(parents=True, exist_ok=True)
        output_dir: str = path.resolve()

        self.draw_channels(_channels, output_dir)
        self.draw_labels(_labels, output_dir)
        self.draw_predictions(_predictions, output_dir)

    def save_gif(self,
                 _input: np.ndarray,
                 _path: str):

        if self.gif_painter is None:
            return

        self.gif_painter.save(_input, _path)

    def save_tif(self,
                 _input: np.ndarray,
                 _path: str):

        if self.tif_painter is None:
            return

        self.tif_painter.save(_input, _path)

    def draw_channels(self,
                      _input,
                      _output_dir: str):

        _input = to_numpy(_input)
        _input = _input * self.channels_multiplier

        for index, _channel in enumerate(_input):
            self.save_tif(_channel,
                          os.path.join(
                              _output_dir,
                              f"channel_{index}.tif"))

            self.save_gif(_channel,
                          os.path.join(
                              _output_dir,
                              f"channel_{index}.gif"))

    def draw_labels(self,
                    _input,
                    _output_dir: str):

        _input = to_numpy(_input)
        # To visualy separate classes from each other
        _input = _input * self.labels_multiplier
        _input = _input.astype(np.uint8)

        self.save_tif(_input,
                      os.path.join(
                          _output_dir,
                          "labels.tif"))

        self.save_gif(_input,
                      os.path.join(
                          _output_dir,
                          "labels.gif"))

    def draw_predictions(self,
                         _input,
                         _output_dir: str):

        _input = to_numpy(_input)
        # _input = vector_to_scaler(_input)

        # To visualy separate classes from each other
        _input = _input * self.predictions_multiplier
        _input = _input.astype(np.uint8)

        self.save_tif(_input,
                      os.path.join(
                          _output_dir,
                          "prediction.tif"))

        self.save_gif(_input,
                      os.path.join(
                          _output_dir,
                          "prediction.gif"))
