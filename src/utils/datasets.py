"""
Author: Arash Fatehi
Date:   27.10.2022
File:   datasets.py
"""

# Python Imports
import os
import re

# Library Imports
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile


class TrainingDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, _source_directory, _sample_dimension, _pixel_per_step):

        self.source_directory = _source_directory
        self.sample_dimension = _sample_dimension
        self.pixel_per_step_x = _pixel_per_step[2]
        self.pixel_per_step_y = _pixel_per_step[1]
        self.pixel_per_step_z = _pixel_per_step[0]

        self.image_list = []
        self.samples_per_image = []
        self.images = {}

        directory_content = os.listdir(self.source_directory)
        directory_content = list(filter(lambda _x: re.match(r'(.+).(tiff|tif)',
                                        _x),
                                        directory_content))

        for file_name in directory_content:
            self.image_list.append(file_name)
            image_shape = tifffile.imread(os.path.join(_source_directory,
                                                       file_name)).shape
            sample_per_z = int((image_shape[0] - self.sample_dimension[0]) //
                               self.pixel_per_step_z) + 1
            sample_per_x = int((image_shape[2] - self.sample_dimension[1]) //
                               self.pixel_per_step_x) + 1
            sample_per_y = int((image_shape[3] - self.sample_dimension[2]) //
                               self.pixel_per_step_y) + 1

            self.samples_per_image.append(sample_per_z *
                                          sample_per_x *
                                          sample_per_y)

        self.per_image_cumsum = np.cumsum(self.samples_per_image)
        self.current_image_id = -1

    def __len__(self):
        return abs(np.sum(self.samples_per_image))

    def __getitem__(self, index):
        file_id = np.min(np.where(self.per_image_cumsum > index)[0])

        _n = index
        if file_id > 0:
            _n = index - self.per_image_cumsum[file_id - 1]

        file_name = self.image_list[file_id]

        if file_name not in self.images:
            self.images[file_name] = self.read_file(file_name)

        nephrin = self.images[file_name][:, 0, :, :]
        wga = self.images[file_name][:, 1, :, :]
        collagen4 = self.images[file_name][:, 2, :, :]
        labels = self.images[file_name][:, 3, :, :]

        image_shape = nephrin.shape
        sample_per_x = int((image_shape[1] - self.sample_dimension[1]) //
                           self.pixel_per_step_x) + 1
        sample_per_y = int((image_shape[2] - self.sample_dimension[2]) //
                           self.pixel_per_step_y) + 1

        z_start = _n // (sample_per_x * sample_per_y)
        _n = _n % (sample_per_x * sample_per_y)

        y_start = _n // sample_per_x

        x_start = _n % sample_per_x

        nephrin = nephrin[z_start: z_start + self.sample_dimension[0],
                          x_start: x_start + self.sample_dimension[1],
                          y_start: y_start + self.sample_dimension[2]
                          ]

        wga = wga[z_start: z_start + self.sample_dimension[0],
                  x_start: x_start + self.sample_dimension[1],
                  y_start: y_start + self.sample_dimension[2]
                  ]

        collagen4 = collagen4[z_start: z_start + self.sample_dimension[0],
                              x_start: x_start + self.sample_dimension[1],
                              y_start: y_start + self.sample_dimension[2]
                              ]

        labels = labels[z_start: z_start + self.sample_dimension[0],
                        x_start: x_start + self.sample_dimension[1],
                        y_start: y_start + self.sample_dimension[2]
                        ]

        nephrin = nephrin.reshape(1,
                                  nephrin.shape[0],
                                  nephrin.shape[1],
                                  nephrin.shape[2])

        wga = wga.reshape(1,
                          wga.shape[0],
                          wga.shape[1],
                          wga.shape[2])

        collagen4 = collagen4.reshape(1,
                                      collagen4.shape[0],
                                      collagen4.shape[1],
                                      collagen4.shape[2])

        labels = labels.reshape(1,
                                labels.shape[0],
                                labels.shape[1],
                                labels.shape[2])

        return {
            'nephrin': torch.from_numpy(nephrin),
            'wga': torch.from_numpy(wga),
            'collagen4': torch.from_numpy(collagen4),
            'labels': torch.from_numpy(labels),
            'offsets': torch.from_numpy(
                np.array([index,
                          file_id,
                          x_start,
                          y_start,
                          z_start,
                          image_shape[0],
                          image_shape[1],
                          image_shape[2]])
            )
        }

    def get_sample_per_image(self, _image_id):
        return self.samples_per_image[_image_id]

    def get_number_of_images(self):
        return len(self.image_list)

    def read_file(self, _file_name):
        image = tifffile.imread(os.path.join(self.source_directory,
                                             _file_name))
        image = np.array(image)
        image = image.astype(np.float32)
        image = image / 255

        return image
