"""
Author: Arash Fatehi
Date:   12.12.2022
"""

# Python Imports
import logging
import os
import re

# Library Imports
import numpy as np
from numpy import array
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import tifffile
import imageio

# Local Imports
from src.data.ds_infer import InferenceDataset
from src.models.unet3d.unet3d import Unet3D
from src.utils.misc import to_numpy, create_dirs_recursively


class Inference():
    def __init__(self,
                 _configs: dict):
        self.base_configs = _configs
        self.configs = _configs['inference']

        self.freq: int = self.configs['report_freq']
        self.model_name: str = self.configs['model']['name']
        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.number_class: int = self.configs['model']['number_class']
        self.source_path: str = self.configs['inference_ds']['path']
        self.sample_dimension: list = \
            self.configs['inference_ds']['sample_dimension']
        self.pixel_stride: list = \
            self.configs['inference_ds']['pixel_stride']
        self.batch_size: int = self.configs['inference_ds']['batch_size']

    def infer(self):

        directory_content = os.listdir(self.source_path)
        directory_content = list(filter(lambda _x: re.match(r'(.+).(tiff|tif)',
                                        _x),
                                        directory_content))

        for file_name in directory_content:
            file_path = os.path.join(self.source_path, file_name)
            dataset = InferenceDataset(
                    _file_path=file_path,
                    _sample_dimension=self.sample_dimension,
                    _pixel_per_step=self.pixel_stride)
            dataset = InferenceDataset(
                    _file_path=file_path,
                    _sample_dimension=self.sample_dimension,
                    _pixel_per_step=self.pixel_stride)

            data_loader = DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)

            result_shape: list = []
            list(dataset.image_shape)
            result_shape.append(3)
            result_shape.append(dataset.image_shape[0])
            result_shape.append(dataset.image_shape[2])
            result_shape.append(dataset.image_shape[3])

            if self.model_name == 'unet_3d':
                model = Unet3D(len(self.channels),
                               self.number_class,
                               _feature_maps=self.feature_maps,
                               _inference=True,
                               _result_shape=result_shape,
                               _sample_dimension=self.sample_dimension)

            snapshot_path: str = self.configs['snapshot_path']
            snapshot = torch.load(snapshot_path)
            model.load_state_dict(snapshot['MODEL_STATE'])

            device: str = self.configs['device']
            model.to(device)

            for index, data in enumerate(data_loader):

                nephrin = data['nephrin'].to(device)
                wga = data['wga'].to(device)
                collagen4 = data['collagen4'].to(device)
                offsets = data['offsets'].to(device)

                sample = self.stack_channels(self.configs,
                                             nephrin,
                                             wga,
                                             collagen4)

                with torch.no_grad():
                    _, _ = model(sample, offsets)

                if index % self.freq == 0:
                    logging.info("%s: %d/%d completed ",
                                 file_name,
                                 index*self.batch_size,
                                 len(dataset))

            output_dir = os.path.join(
                    self.configs['result_dir'],
                    self.base_configs['tag'],
                    file_name)

            create_dirs_recursively(
                    os.path.join(output_dir, "dummy"))

            result = model.get_result()
            result = to_numpy(result)
            result = np.argmax(result, axis=0)
            self.save_result(dataset.nephrin,
                             dataset.wga,
                             dataset.collagen4,
                             result,
                             output_dir,
                             dataset.tiff_tags)

    # Channels list in the configuration determine
    # Which channels to be stacked
    @staticmethod
    def stack_channels(_configs,
                       _nephrin: Tensor,
                       _wga: Tensor,
                       _collagen4: Tensor) -> Tensor:

        channels: list = _configs['model']['channels']

        if len(channels) == 3:
            result = torch.cat((_nephrin, _wga, _collagen4), dim=1)
        elif len(channels) == 2:
            if channels in ([0, 1], [1, 0]):
                result = torch.cat((_nephrin, _wga), dim=1)
            if channels in ([0, 2], [2, 0]):
                result = torch.cat((_nephrin, _collagen4), dim=1)
            if channels in ([1, 2], [2, 1]):
                result = torch.cat((_wga, _collagen4), dim=1)
        elif len(channels) == 1:
            # Nephrin -> 0, WGA -> 1, Collagen4 -> 2
            if channels[0] == 0:
                result = _nephrin
            if channels[0] == 1:
                result = _wga
            if channels[0] == 2:
                result = _collagen4
        else:
            assert True, "Wrong channels list configuration"

        return result

    def save_result(self,
                    _nephrin: array,
                    _wga: array,
                    _collagen4: array,
                    _prediction: array,
                    _output_path: str,
                    _tiff_tags: dict,
                    _multiplier: int = 127):

        prediction_tif_path = os.path.join(_output_path, "prediction.tif")
        prediction_gif_path = os.path.join(_output_path, "prediction.gif")

        sure_tif_path = os.path.join(_output_path, "sure.tif")
        possible_tif_path = os.path.join(_output_path, "possible.tiff")

        sure_tensor = np.copy(_prediction)
        sure_tensor[sure_tensor != 2] = 0
        sure_tensor[sure_tensor == 2] = 255
        sure_tensor = sure_tensor.astype(np.uint8)

        possible_tensor = np.copy(_prediction)
        possible_tensor[possible_tensor != 1] = 0
        possible_tensor[possible_tensor == 1] = 255
        possible_tensor = possible_tensor.astype(np.uint8)

        _prediction = _prediction * _multiplier
        _prediction = _prediction.astype(np.uint8)

        possible_tensor = np.stack([_nephrin,
                                    _wga,
                                    _collagen4,
                                    possible_tensor],
                                   axis=1)

        sure_tensor = np.stack([_nephrin,
                                _wga,
                                _collagen4,
                                sure_tensor],
                               axis=1)

        if self.configs['save_npy']:
            np.save(os.path.join(_output_path,
                                 "prediction.npy"),
                    _prediction)
            np.save(os.path.join(_output_path,
                                 "sure.npy"),
                    sure_tensor)
            np.save(os.path.join(_output_path,
                                 "possible.npy"),
                    possible_tensor)

        with imageio.get_writer(prediction_gif_path, mode='I') as writer:
            for index in range(_prediction.shape[0]):
                writer.append_data(_prediction[index])

        tifffile.imwrite(prediction_tif_path,
                         _prediction,
                         shape=_prediction.shape,
                         imagej=True,
                         metadata={'axes': 'ZYX', 'fps': 10.0})

        tifffile.imwrite(possible_tif_path,
                         possible_tensor,
                         shape=possible_tensor.shape,
                         imagej=True,
                         metadata={'axes': 'ZCYX', 'fps': 10.0})

        tifffile.imwrite(sure_tif_path,
                         sure_tensor,
                         shape=sure_tensor.shape,
                         imagej=True,
                         metadata={'axes': 'ZCYX', 'fps': 10.0})
