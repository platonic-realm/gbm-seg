"""
Author: Arash Fatehi
Date:   10.12.2022
"""

# Python Imports

# Library Imports
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile

# Local Imports


class InferenceDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, _file_path, _sample_dimension, _pixel_per_step):

        self.file_path = _file_path
        self.tiff_tags = {}
        self.image = self.read_file(self.file_path)
        self.image_shape = self.image.shape

        assert (self.image_shape[0] - _sample_dimension[0]) % \
            _pixel_per_step[0], \
            "(Len(Z_Image) - Len(Z_Sample) % Z_Stride should be 0."

        self.nephrin = self.image[:, 0, :, :]
        self.wga = self.image[:, 1, :, :]
        self.collagen4 = self.image[:, 2, :, :]

        self.sample_dimension = _sample_dimension
        self.pixel_per_step_x = _pixel_per_step[2]
        self.pixel_per_step_y = _pixel_per_step[1]
        self.pixel_per_step_z = _pixel_per_step[0]

        sample_per_z = int((self.image_shape[0] - self.sample_dimension[0]) //
                           self.pixel_per_step_z) + 1
        sample_per_x = int((self.image_shape[2] - self.sample_dimension[1]) //
                           self.pixel_per_step_x) + 1
        sample_per_y = int((self.image_shape[3] - self.sample_dimension[2]) //
                           self.pixel_per_step_y) + 1

        self.no_of_samples = sample_per_z * sample_per_x * sample_per_y

    def __len__(self):
        return self.no_of_samples

    def __getitem__(self, index):

        channel_shape = self.nephrin.shape
        sample_per_x = int((channel_shape[1] - self.sample_dimension[1]) //
                           self.pixel_per_step_x) + 1
        sample_per_y = int((channel_shape[2] - self.sample_dimension[2]) //
                           self.pixel_per_step_y) + 1

        _n = index
        z_start = _n // (sample_per_x * sample_per_y)
        _n = _n % (sample_per_x * sample_per_y)

        y_start = _n // sample_per_x

        x_start = _n % sample_per_x

        z_start = z_start * self.pixel_per_step_z
        x_start = x_start * self.pixel_per_step_x
        y_start = y_start * self.pixel_per_step_y

        nephrin = self.nephrin[z_start: z_start + self.sample_dimension[0],
                               x_start: x_start + self.sample_dimension[1],
                               y_start: y_start + self.sample_dimension[2]]

        wga = self.wga[z_start: z_start + self.sample_dimension[0],
                       x_start: x_start + self.sample_dimension[1],
                       y_start: y_start + self.sample_dimension[2]]

        collagen4 = self.collagen4[z_start: z_start + self.sample_dimension[0],
                                   x_start: x_start + self.sample_dimension[1],
                                   y_start: y_start + self.sample_dimension[2]]

        nephrin = np.expand_dims(nephrin, axis=0)
        wga = np.expand_dims(wga, axis=0)
        collagen4 = np.expand_dims(collagen4, axis=0)

        nephrin = nephrin/255
        wga = wga/255
        collagen4 = collagen4/255

        return {
            'nephrin': torch.from_numpy(nephrin),
            'wga': torch.from_numpy(wga),
            'collagen4': torch.from_numpy(collagen4),
            'offsets': torch.from_numpy(
                np.array([index,
                          x_start,
                          y_start,
                          z_start])
            ).int()
        }

    def read_file(self, _file_path):
        with tifffile.TiffFile(_file_path) as tif:
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                self.tiff_tags[name] = value

        image = tifffile.imread(_file_path)
        image = np.array(image)
        image = image.astype(np.float32)

        return image
