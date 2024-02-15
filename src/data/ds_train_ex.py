# Python Imports
import os
import re
import random

# Library Imports
import torch
import torch.multiprocessing as mp
import numpy as np
import tifffile
import cv2
from scipy.ndimage import zoom

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
        self.images_padded = {}

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
        # Sample dimention is like (Z, X, Y)
        steps_per_x = int((image_shape[1] - self.sample_dimension[1]) //
                          self.pixel_per_step_x) + 1
        steps_per_y = int((image_shape[2] - self.sample_dimension[2]) //
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

        padding_values = ((self.sample_dimension[0], self.sample_dimension[0]),
                          (self.sample_dimension[1], self.sample_dimension[1]),
                          (self.sample_dimension[2], self.sample_dimension[2]))

        if file_name in self.images_padded:
            padded_nephrin = self.images_padded[file_name][0, :, :, :]
            padded_wga = self.images_padded[file_name][1, :, :, :]
            padded_collagen4 = self.images_padded[file_name][2, :, :, :]
        else:
            padded_nephrin = np.pad(nephrin, pad_width=padding_values)
            padded_wga = np.pad(wga, pad_width=padding_values)
            padded_collagen4 = np.pad(collagen4, pad_width=padding_values)

            padded_sample = np.stack((padded_nephrin,
                                      padded_wga,
                                      padded_collagen4), axis=0)
            self.images_padded[file_name] = padded_sample

        lm_z_start = z_start
        lm_z_end = z_start + self.sample_dimension[0] * 3
        lm_x_start = x_start
        lm_x_end = x_start + self.sample_dimension[1] * 3
        lm_y_start = y_start
        lm_y_end = y_start + self.sample_dimension[2] * 3

        lm_nephrin = padded_nephrin[lm_z_start: lm_z_end,
                                    lm_x_start: lm_x_end,
                                    lm_y_start: lm_y_end]

        lm_nephrin = zoom(lm_nephrin, zoom=1/3, order=1)

        lm_nephrin = torch.from_numpy(lm_nephrin)

        lm_wga = padded_wga[lm_z_start: lm_z_end,
                            lm_x_start: lm_x_end,
                            lm_y_start: lm_y_end]

        lm_wga = zoom(lm_wga, zoom=1/3, order=1)
        lm_wga = torch.from_numpy(lm_wga)

        lm_collagen4 = padded_collagen4[lm_z_start: lm_z_end,
                                        lm_x_start: lm_x_end,
                                        lm_y_start: lm_y_end]

        lm_collagen4 = zoom(lm_collagen4, zoom=1/3, order=1)
        lm_collagen4 = torch.from_numpy(lm_collagen4)

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

        lm_nephrin = np.expand_dims(lm_nephrin, axis=0)
        lm_wga = np.expand_dims(lm_wga, axis=0)
        lm_collagen4 = np.expand_dims(lm_collagen4, axis=0)

        nephrin = self._intensity_shift(nephrin/255)
        wga = self._intensity_shift(wga/255)
        collagen4 = self._intensity_shift(collagen4/255)

        lm_nephrin = self._intensity_shift(lm_nephrin/255)
        lm_wga = self._intensity_shift(lm_wga/255)
        lm_collagen4 = self._intensity_shift(lm_collagen4/255)

        sample = torch.cat((nephrin, collagen4, wga,
                            lm_nephrin, lm_collagen4, lm_wga), dim=0)

        return {
            'sample': sample,
            'labels': labels,
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
                file_name = f"{file_name}{_method[0]}_{_method[1]}.tiff"

            with tifffile.TiffFile(image_path) as tiff:
                image = tiff.asarray()
                self.images_metadata[file_name] = self.get_tiff_tags(tiff)

            image_shape = image.shape

            self.check_image_shape_compatibility(image_shape,
                                                 file_name)

            image = np.array(image)
            image = image.astype(np.float32)

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

    def _intensity_shift(self, _sample):
        return _sample
        # random_shift = (torch.rand(1) * 0.2) - 0.1
        # random_scale = (torch.rand(1) * 0.2) + 0.9
        # return random_scale * _sample + random_shift

    def _zoom(self, _image, _factor, _workers):
        image_shape = _image.shape
        zoomed_image = None

        for c in range(image_shape[0] - 1):
            for z in range(image_shape[1]):
                resized = cv2.resize(_image[c][z],
                                     None,
                                     fx=_factor,
                                     fy=_factor)
                if zoomed_image is None:
                    zoomed_image = np.zeros((image_shape[0],
                                             image_shape[1],
                                             resized.shape[0],
                                             resized.shape[1]), dtype=np.int16)
                zoomed_image[c][z] = resized

        for z in range(image_shape[1]):
            zoomed_image[3][z] = cv2.resize(_image[3][z],
                                            None,
                                            fx=_factor,
                                            fy=_factor,
                                            interpolation=cv2.INTER_NEAREST)

        return zoomed_image

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
        h = shape[0]
        w = shape[1]
        center = (h/2, w/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, _angle, 1.0)
        result_plane = cv2.warpAffine(_plane, rotation_matrix, (h, w))
        return result_plane
