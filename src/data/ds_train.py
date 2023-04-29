"""
Author: Arash Fatehi
Date:   27.10.2022
"""

# Python Imports
import os
import re
import random

# Library Imports
import torch
import torch.multiprocessing as mp
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
                 _label_correction_function=None,
                 _augmentation=None,
                 _augmentation_workers=8):

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

        self._prepare_images(directory_content)

        self.cache_directory = os.path.join(self.source_directory,
                                            "cache/")
        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)
        self.cache_content = os.listdir(self.cache_directory)

        self.aug_workers = _augmentation_workers
        if _augmentation is not None:
            for method in _augmentation:
                self._prepare_images(directory_content, method)

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

    def _prepare_images(self,
                        _directory,
                        _method=None):

        for file_name in _directory:
            self.image_list.append(file_name)

            image_path = os.path.join(self.source_directory, file_name)

            if _method is not None:
                file_name = f"{file_name}{_method[0]}_{_method[1]}"

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

            if _method is not None:
                cached_file_path = os.path.join(self.cache_directory,
                                                file_name)
                if file_name not in self.cache_content:
                    image = getattr(GBMDataset, _method[0])(self,
                                                            image,
                                                            _method[1],
                                                            self.aug_workers)
                    tifffile.imwrite(cached_file_path,
                                     image,
                                     shape=image.shape,
                                     imagej=True,
                                     metadata=self.images_metadata[file_name],
                                     compression="lzw")
                else:
                    with tifffile.TiffFile(cached_file_path) as tiff:
                        image = tiff.asarray()

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

    def _twist_clock(self, _image, _step_angle, _workers):
        z_angles = []
        z_length = _image.shape[0]
        _step_angle = float(_step_angle)
        for i in range(z_length):
            angle = (i*_step_angle) - (z_length//2)*_step_angle
            z_angles.append(angle)
        return self._twister(_image, z_angles, _workers)

    def _twist_reverse(self, _image, _step_angle, _workers):
        z_angles = []
        z_length = _image.shape[0]
        _step_angle = float(_step_angle)
        for i in range(z_length):
            angle = (z_length//2)*_step_angle - (i*_step_angle)
            z_angles.append(angle)
        return self._twister(_image, z_angles, _workers)

    def _rotate_linear(self, _image, _step_angle, _workers):
        z_angles = []
        z_length = _image.shape[0]
        _step_angle = float(_step_angle)
        for i in range(z_length):
            angle = i*_step_angle
            z_angles.append(angle)
        return self._twister(_image, z_angles, _workers)

    def _rotate_reverse(self, _image, _step_angle, _workers):
        z_angles = []
        z_length = _image.shape[0]
        _step_angle = float(_step_angle)
        for i in range(z_length):
            angle = -i*_step_angle
            z_angles.append(angle)
        return self._twister(_image, z_angles, _workers)

    def _rotate_random(self, _image, _step_angle, _workers):
        z_angles = []
        z_length = _image.shape[0]
        _step_angle = float(_step_angle)
        max_angle = z_length // 2 * _step_angle
        min_angle = -max_angle
        for _ in range(z_length):
            z_angles.append(random.uniform(min_angle, max_angle))
        return self._twister(_image, z_angles, _workers)

    def _twister(self, _image, _z_angles, _workers):
        result_image = np.copy(_image)
        with mp.Pool(_workers) as pool:
            processes = [
                [None for _ in range(
                    _image.shape[1])] for _ in range(_image.shape[0])]
            results = [
                [None for _ in range(
                    _image.shape[1])] for _ in range(_image.shape[0])]

            for z in range(_image.shape[0]):
                for c in range(_image.shape[1]):
                    processes[z][c] = \
                        pool.apply_async(self._rotate_plane,
                                         args=(_image[z, c, :, :],
                                               _z_angles[z]))

            for z in range(_image.shape[0]):
                for c in range(_image.shape[1]):
                    results[z][c] = processes[z][c].get()

            for z in range(_image.shape[0]):
                for c in range(_image.shape[1]):
                    result_image[z, c, :, :] = results[z][c]

        return result_image

    @staticmethod
    def _rotate_plane(_plane, _angle):
        shape = _plane.shape
        center = np.array([shape[0] / 2, shape[1] / 2])
        result_plane = np.zeros(shape)

        rotation_matrix = np.array([[np.cos(np.radians(_angle)),
                                     -np.sin(np.radians(_angle))],
                                    [np.sin(np.radians(_angle)),
                                     np.cos(np.radians(_angle))]])

        # loop through each pixel in the plane
        for i in range(shape[0]):
            for j in range(shape[1]):
                # get the coordinates of this pixel
                coordinates = np.array([i, j])
                # translate the coordinates to the center of the plane
                translated_coordinates = coordinates - center
                # rotate the coordinates using the rotation matrix
                rotated_coordinates = np.dot(rotation_matrix,
                                             translated_coordinates)
                # translate the rotated coordinates back to the original
                final_coordinates = rotated_coordinates + center
                # round the final coordinates to the nearest integer
                final_coordinates = np.round(final_coordinates).astype(int)
                # check if the final coordinates are within the bounderies
                if (final_coordinates[0] >= 0 and
                        final_coordinates[0] < shape[0] and
                        final_coordinates[1] >= 0 and
                        final_coordinates[1] < shape[1]):
                    result_plane[i][j] = \
                        _plane[final_coordinates[0]][final_coordinates[1]]

        return result_plane
