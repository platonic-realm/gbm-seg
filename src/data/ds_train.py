"""
Author: Arash Fatehi
Date:   27.10.2022
"""

# Python Imports
import os
import re

# Library Imports
import torch
import numpy as np
import tifffile

# Local Imports
from src.data.ds_base import DatasetType, BaseDataset


class GBMDataset(BaseDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 _source_directory,
                 _sample_dimension,
                 _pixel_per_step,
                 _channel_map,
                 _scale_facor=1,
                 _dataset_type=DatasetType.Supervised,
                 _ignore_stride_mismatch=False,
                 _label_correction_function=None):

        super().__init__(_sample_dimension,
                         _pixel_per_step,
                         _channel_map,
                         _scale_facor,
                         _dataset_type,
                         _ignore_stride_mismatch,
                         _label_correction_function)

        self.source_directory = _source_directory

        self.image_list = []
        self.samples_per_image = []
        self.images = {}
        self.images_metadata = {}

        directory_content = os.listdir(self.source_directory)
        directory_content = list(filter(lambda _x: re.match(r'(.+).(tiff|tif)',
                                        _x),
                                        directory_content))

        for file_name in directory_content:
            self.image_list.append(file_name)

            image_path = os.path.join(_source_directory, file_name)
            with tifffile.TiffFile(image_path) as tiff:
                image = tiff.asarray()
                self.images_metadata[file_name] = self.get_tiff_tags(tiff)

            image_shape = image.shape

            self.check_image_shape_compatibility(image_shape,
                                                 file_name)

            image = np.array(image)
            image = image.astype(np.float32)

            if self.dataset_type == DatasetType.Supervised:
                image[:, 3, :, :][image[:, 3, :, :] >= 255] = -1
                image[:, 3, :, :][image[:, 3, :, :] >= 0] = 255
                image[:, 3, :, :][image[:, 3, :, :] == -1] = 0

            if self.scale_factor > 1:
                image = self.scale(image)

            if self.label_correction_function is not None:
                image[:, 3, :, :] = \
                    self.label_correction_function(image[:, 3, :, :])

            self.images[file_name] = image

            # Image's dimentionality order is like (Z, C, Y, X)
            # Sample dimentionality order is like (Z, X, Y)
            steps_per_z = int((image_shape[0] - self.sample_dimension[0]) //
                              self.pixel_per_step_z) + 1
            steps_per_y = int((image_shape[2] - self.sample_dimension[2]) //
                              self.pixel_per_step_y) + 1
            steps_per_x = int((image_shape[3] - self.sample_dimension[1]) //
                              self.pixel_per_step_x) + 1

            self.samples_per_image.append(steps_per_z *
                                          steps_per_x *
                                          steps_per_y)

        # Cumulative sum of number of samples per image
        self.cumulative_sum = np.cumsum(self.samples_per_image)

    def __len__(self):
        return abs(np.sum(self.samples_per_image))

    def __getitem__(self, index):
        # Index of the image in image_list
        image_id = np.min(np.where(self.cumulative_sum > index)[0])

        file_name = self.image_list[image_id]

        nephrin = self.images[file_name][:, self.channel_map[0], :, :]
        wga = self.images[file_name][:, self.channel_map[1], :, :]
        collagen4 = self.images[file_name][:, self.channel_map[2], :, :]

        if self.dataset_type == DatasetType.Supervised:
            labels = self.images[file_name][:, 3, :, :]

        image_shape = nephrin.shape
        # Image's dimention is like (Z, Y, X)
        # Sample dimention is like (Z, X, Y)
        steps_per_x = int((image_shape[2] - self.sample_dimension[1]) //
                          self.pixel_per_step_x) + 1
        steps_per_y = int((image_shape[1] - self.sample_dimension[2]) //
                          self.pixel_per_step_y) + 1

        # Till now we calculated what image_id the index correspond to,
        # Next we need to know which part of the image we should crop
        # to provide the next sample

        # When cropping the images, we start from (0, 0, 0) and first move
        # on the x-axis, then y-axis and at last on the z-axis.
        # So, each image corresonds to many samples, and sample_id uniquely
        # maps an integer to an cordinates in the image that we should
        # cropping from.

        # For the first image its the same as the index
        sample_id = index

        # We need to deduct the number of samples in previous
        # images for the following images
        if image_id > 0:
            sample_id = index - self.cumulative_sum[image_id - 1]

        z_start = sample_id // (steps_per_x * steps_per_y)
        z_start = z_start * self.pixel_per_step_z

        # Same as sample_id but in the xy plane instead of the image stack
        xy_id = sample_id % (steps_per_x * steps_per_y)

        y_start = xy_id // steps_per_x
        y_start = y_start * self.pixel_per_step_y

        x_start = xy_id % steps_per_x
        x_start = x_start * self.pixel_per_step_x

        nephrin = nephrin[z_start: z_start + self.sample_dimension[0],
                          x_start: x_start + self.sample_dimension[1],
                          y_start: y_start + self.sample_dimension[2]
                          ]
        nephrin = torch.from_numpy(nephrin)

        wga = wga[z_start: z_start + self.sample_dimension[0],
                  x_start: x_start + self.sample_dimension[1],
                  y_start: y_start + self.sample_dimension[2]
                  ]
        wga = torch.from_numpy(wga)

        collagen4 = collagen4[z_start: z_start + self.sample_dimension[0],
                              x_start: x_start + self.sample_dimension[1],
                              y_start: y_start + self.sample_dimension[2]
                              ]
        collagen4 = torch.from_numpy(collagen4)

        if self.dataset_type == DatasetType.Supervised:
            labels = labels[z_start: z_start + self.sample_dimension[0],
                            x_start: x_start + self.sample_dimension[1],
                            y_start: y_start + self.sample_dimension[2]
                            ]
            labels = torch.from_numpy(labels)

        nephrin = np.expand_dims(nephrin, axis=0)
        wga = np.expand_dims(wga, axis=0)
        collagen4 = np.expand_dims(collagen4, axis=0)

        nephrin = nephrin/255
        wga = wga/255
        collagen4 = collagen4/255

        if self.dataset_type == DatasetType.Supervised:
            return {
                'nephrin': nephrin,
                'wga': wga,
                'collagen4': collagen4,
                'labels': labels,
            }

        return {
            'nephrin': nephrin,
            'wga': wga,
            'collagen4': collagen4,
        }

    def get_sample_per_image(self, _image_id):
        return self.samples_per_image[_image_id]

    def get_number_of_images(self):
        return len(self.image_list)

    def get_number_of_classes(self):
        labels = self.images[self.image_list[0]][:, 3, :, :]
        unique_labels = np.unique(labels)
        return len(unique_labels)
