"""
Author: Arash Fatehi
Date:   07.11.2022
"""
# Python Imports
import os

# Lirary Imports
import torch
from torch.utils.data import DataLoader

# Local Imports
from src.models.unet3d import Unet3D
from src.configs import VISUAL_OUTPUT_PATH
from src.utils.datasets import GBMDataset
from src.utils.visual import visualize_predictions
from src.utils.misc import to_numpy


def test_visualization_3dunet():
    model = Unet3D(3).to('cuda')
    test_visualization_functinos(model,
                                '/home/afatehi/gbm/data/GBM-Valid-DS',
                                (12, 256, 256),
                                (1, 16, 16),
                                4,
                                'cuda')


def test_visualization_functinos(_model,
                                 _validation_ds_path,
                                 _sample_dimension,
                                 _pixel_per_step,
                                 _batch_size,
                                 _device):

    validation_dataset = GBMDataset(
        _source_directory=_validation_ds_path,
        _sample_dimension=_sample_dimension,
        _pixel_per_step=_pixel_per_step)

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=_batch_size,
                                   shuffle=True)
    index = 0
    for _data in validation_loader:
        index += 1
        if index > 54:
            nephrin = _data['nephrin'].to(_device)
            wga = _data['wga'].to(_device)
            collagen4 = _data['collagen4'].to(_device)
            labels = _data['labels'].to(_device)

            sample = torch.cat((nephrin, wga, collagen4),
                               dim=1)

            outputs = to_numpy(_model(sample))
            sample = to_numpy(sample)
            labels = to_numpy(labels)

            ouput_dir = os.path.join(VISUAL_OUTPUT_PATH,"")
            visualize_predictions(_inputs=sample[0],
                                  _labels=labels[0],
                                  _predictions=outputs[0],
                                  _output_dir=ouput_dir)

            break
