"""
Author: Arash Fatehi
Date:   12.12.2022
"""

# Python Imports
import os
import re

# Library Imports
import numpy as np
from numpy import array
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP
import tifffile
import imageio
from tqdm import tqdm
from skimage import measure, morphology

# Local Imports
from src.data.ds_infer import InferenceDataset
from src.models.unet3d import Unet3D
from src.models.unet3d_me import Unet3DME
from src.models.unet3d_ss import Unet3DSS

from src.utils.misc import create_dirs_recursively


class Inference():
    def __init__(self,
                 _configs: dict):
        self.root_path = _configs['root_path']
        self.base_configs = _configs
        self.configs = _configs['inference']

        self.device: str = self.configs['device']
        self.model_name: str = self.configs['model']['name']
        self.number_class: int = self.configs['number_class']
        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.input_path: str = self.configs['inference_ds']['path']
        self.sample_dimension: list = \
            self.configs['inference_ds']['sample_dimension']
        self.pixel_stride: list = \
            self.configs['inference_ds']['pixel_stride']
        self.batch_size: int = self.configs['inference_ds']['batch_size']
        self.scale_factor: int = self.configs['inference_ds']['scale_factor']
        self.channel_map: list = self.configs['inference_ds']['channel_map']

    def infer(self):
        directory_path = os.path.join(self.root_path,
                                      self.input_path)
        directory_content = os.listdir(directory_path)
        directory_content = list(filter(lambda _x: re.match(r'(.+).(tiff|tif)',
                                        _x),
                                        directory_content))

        for file_name in directory_content:
            file_path = os.path.join(directory_path, file_name)
            dataset = InferenceDataset(
                    _file_path=file_path,
                    _sample_dimension=self.sample_dimension,
                    _pixel_per_step=self.pixel_stride,
                    _channel_map=self.channel_map,
                    _scale_factor=self.scale_factor)

            data_loader = DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)

            result_shape: list = []
            list(dataset.image_shape)
            result_shape.append(self.number_class)
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
            elif self.model_name == 'unet_3d_me':
                model = Unet3DME(1,
                                 self.number_class,
                                 _feature_maps=self.feature_maps,
                                 _inference=True,
                                 _result_shape=result_shape,
                                 _sample_dimension=self.sample_dimension)
            elif self.model_name == 'unet_3d_ss':
                model = Unet3DSS(len(self.channels),
                                 self.number_class,
                                 _feature_maps=self.feature_maps,
                                 _inference=True,
                                 _result_shape=result_shape,
                                 _sample_dimension=self.sample_dimension)

            snapshot_path: str = self.configs['snapshot_path']
            snapshot_path = os.path.join(self.root_path,
                                         snapshot_path)
            snapshot = torch.load(snapshot_path,
                                  map_location=torch.device(self.device))

            # Check if the state dict was created with data parallelism
            state_dict = snapshot['MODEL_STATE']
            if 'module' in list(state_dict.keys())[0]:
                corrected_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    corrected_state_dict[name] = v
                model.load_state_dict(corrected_state_dict)
            else:
                model.load_state_dict(state_dict)

            device: str = self.configs['device']

            model.to(device)
            model = DP(model)
            model.eval()

            for data in tqdm(data_loader,
                             desc="Prcossing"):

                nephrin = data['nephrin'].to(device)
                wga = data['wga'].to(device)
                collagen4 = data['collagen4'].to(device)
                offsets = data['offsets'].to(device)

                sample = self.stack_channels(self.configs,
                                             nephrin,
                                             wga,
                                             collagen4).to(device)

                with torch.no_grad():
                    if self.model_name == 'unet_3d_me':
                        _ = model(nephrin, wga, collagen4, offsets)
                    else:
                        _ = model(sample, offsets)

            output_dir = os.path.join(
                    self.root_path,
                    self.configs['result_dir'],
                    file_name)

            create_dirs_recursively(
                    os.path.join(output_dir, "dummy"))

            result = model.module.get_result()
            result = torch.argmax(result, dim=0)
            result = self.post_processing(result)

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
                    _multiplier: int = 120):

        prediction_tif_path = os.path.join(_output_path, "prediction.tif")
        prediction_gif_path = os.path.join(_output_path, "prediction.gif")

        _prediction = _prediction * _multiplier
        _prediction = _prediction.astype(np.uint8)

        # if self.configs['save_npy']:
        #     np.save(os.path.join(_output_path,
        #                          "prediction.npy"),
        #             _prediction)

        with imageio.get_writer(prediction_gif_path, mode='I') as writer:
            for index in range(_prediction.shape[0]):
                writer.append_data(_prediction[index])

        _prediction = np.stack([_nephrin,
                                _wga,
                                _collagen4,
                                _prediction],
                               axis=1)

        tifffile.imwrite(prediction_tif_path,
                         _prediction,
                         shape=_prediction.shape,
                         imagej=True,
                         metadata={'axes': 'ZCYX', 'fps': 10.0},
                         compression='lzw')

    def post_processing(self,
                        _prediction: Tensor):
        prediction = _prediction.detach().cpu().numpy()
        if not self.configs['post_processing']['enabled']:
            return prediction

        for i in range(prediction.shape[0]):
            min_size = self.configs['post_processing']['min_size']
            kernel_size = self.configs['post_processing']['kernel_size']

            kernel = morphology.rectangle(kernel_size, kernel_size)
            eroded_image = morphology.erosion(prediction[i, :, :], kernel)
            prediction[i, :, :] = eroded_image

            sample = prediction[i, :, :]
            labels = measure.label(sample, connectivity=1)
            # count pixels in each connected component
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            # remove small connected components
            # print(f"mean: {int(np.mean(label_counts))}, std: {int(np.std(label_counts))}, max: {int(np.max(label_counts))}, min: {int(np.min(label_counts))}")
            for label, count in zip(unique_labels, label_counts):
                if count < min_size and label != 0:
                    prediction[i, :, :][labels == label] = 0

            kernel = morphology.rectangle(kernel_size, kernel_size)
            dilated_image = morphology.dilation(prediction[i, :, :], kernel)
            prediction[i, :, :] = dilated_image

        return prediction
