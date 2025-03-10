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
from scipy.ndimage import zoom, gaussian_filter, rotate
import torchvision.transforms.functional as TF

# Local Imports
from src.data.ds_base import DatasetType, BaseDataset


class GBMDataset(BaseDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 _source_directory,
                 _sample_dimension,
                 _pixel_per_step,
                 _scale_facor=1,
                 _dataset_type=DatasetType.Supervised,
                 _ignore_stride_mismatch=False,
                 _label_correction_function=None,
                 _augmentation_offline=None,
                 _augmentation_online=None,
                 _augmentation_workers=8,
                 _is_valid=False):

        super().__init__(_sample_dimension,
                         _pixel_per_step,
                         _scale_facor,
                         _dataset_type,
                         _ignore_stride_mismatch,
                         _label_correction_function)

        self.source_directory = _source_directory

        self.image_list = []
        self.samples_per_image = []
        self.images = {}
        self.images_metadata = {}
        self.augmentation_online = _augmentation_online
        self.augmentation_offline = _augmentation_offline
        self.is_valid = _is_valid

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
        if _augmentation_offline is not None:
            for method in _augmentation_offline:
                self._prepare_images(directory_content, method)

        # Cumulative sum of number of samples per image
        self.cumulative_sum = np.cumsum(self.samples_per_image)

    def __len__(self):
        return abs(np.sum(self.samples_per_image))

    def __getitem__(self, index):

        # Index of the image in image_list
        image_id = np.min(np.where(self.cumulative_sum > index)[0])

        file_name = self.image_list[image_id]

        nephrin = self.images[file_name][:, 0, :, :]
        collagen4 = self.images[file_name][:, 1, :, :]
        wga = self.images[file_name][:, 2, :, :]

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

        z_end = z_start + self.sample_dimension[0]
        x_end = x_start + self.sample_dimension[1]
        y_end = y_start + self.sample_dimension[2]

        # print(f"filename = {file_name}\nz_s:{z_start}, z_e:{z_end}\nx_s:{x_start}, x_e:{x_end}\ny_s:{y_start}, y_e:{y_end}")

        nephrin = nephrin[z_start: z_end,
                          x_start: x_end,
                          y_start: y_end]

        wga = wga[z_start: z_end,
                  x_start: x_end,
                  y_start: y_end]

        collagen4 = collagen4[z_start: z_end,
                              x_start: x_end,
                              y_start: y_end]

        labels = labels[z_start: z_end,
                        x_start: x_end,
                        y_start: y_end]

        # Online Augmentations
        if self.augmentation_online and not self.is_valid:

            # Blurring
            sigma = 2
            blurring_chance = self.augmentation_online['blur']
            if np.random.rand() < blurring_chance:
                nephrin = gaussian_filter(nephrin, sigma=sigma)

            if np.random.rand() < blurring_chance:
                collagen4 = gaussian_filter(collagen4, sigma=sigma)

            if np.random.rand() < blurring_chance:
                wga = gaussian_filter(wga, sigma=sigma)

            # Rotation
            rotation_chance = self.augmentation_online['rotate']
            if np.random.rand() < rotation_chance:
                nephrin, collagen4, wga, labels = \
                        self._rotate_channels(nephrin,
                                              collagen4,
                                              wga,
                                              labels)

            # Cropping
            crop_chance = self.augmentation_online['crop']
            if np.random.rand() < crop_chance:
                nephrin = self._crop_channels(nephrin)
                collagen4 = self._crop_channels(collagen4)
                wga = self._crop_channels(wga)

            # Channel drop
            drop_chance = self.augmentation_online['channel_drop']
            if np.random.rand() < drop_chance:
                channel_id = random.randint(0, 2)
                if channel_id == 0:
                    nephrin = np.zeros_like(nephrin)
                if channel_id == 1:
                    collagen4 = np.zeros_like(collagen4)
                if channel_id == 2:
                    wga = np.zeros_like(wga)

        nephrin = np.expand_dims(nephrin, axis=0)
        wga = np.expand_dims(wga, axis=0)
        collagen4 = np.expand_dims(collagen4, axis=0)

        nephrin = torch.from_numpy(nephrin/255)
        wga = torch.from_numpy(wga/255)
        collagen4 = torch.from_numpy(collagen4/255)
        labels = torch.from_numpy(labels)

        sample = torch.cat((nephrin, collagen4, wga), dim=0)

        return {
            'sample': sample,
            'labels': labels,
        }

    def get_sample_per_image(self, _image_id):
        return self.samples_per_image[_image_id]

    def get_number_of_images(self):
        return len(self.image_list)

    def getNumberOfClasses(self):
        labels = self.images[self.image_list[0]][:, 3, :, :]
        unique_labels = np.unique(labels)
        return len(unique_labels)

    def getNumberOfChannels(self):
        # Channels are the second axis, -1 is for the labels
        return self.images[self.image_list[0]].shape[1] - 1

    def setIsValid(self, _is_valid: bool):
        self.is_valid = _is_valid

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

            def label_correction_function(_labels):
                _labels = _labels.astype(int)
                _labels[_labels > 0] = 1
                # _labels[_labels == 1] = 2
                # _labels[_labels == 0] = 1
                # _labels[_labels == 2] = 0
                return _labels

            image[:, 3, :, :] = \
                label_correction_function(image[:, 3, :, :])

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
        random_shift = (torch.rand(1) * 0.1)
        random_scale = (torch.rand(1) * 0.1) + 0.95
        return random_scale * _sample + random_shift

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
        for z in range(_image.shape[0]):
            for c in range(_image.shape[1]):
                result_image[z, c, :, :] = self._rotate_plane(_image[z, c, :, :],
                                                              _z_angles[z])

        return result_image

    def _rotate_plane(self, img, angle):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = torch.from_numpy(img).to(device)

        # Add a batch dimension: [C, H, W] -> [1, C, H, W]
        img_batch = img.unsqueeze(0)

        # Rotate the image; the underlying transform expects a 4D tensor.
        rotated_batch = TF.rotate(img_batch, angle=angle, expand=False, fill=0)

        # Remove the batch dimension: [1, C, H, W] -> [C, H, W]
        rotated_img = rotated_batch.squeeze(0).detach().cpu().numpy()

        return rotated_img

    @staticmethod
    def _rotate_plane_(_plane, _angle):
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

    @staticmethod
    def _crop_channels(_channel):
        z_patch_length = np.random.randint(2, 7)
        x_patch_length = np.random.randint(30, 51)
        y_patch_length = np.random.randint(30, 51)

        z_patch_start = np.random.randint(0, _channel.shape[0] - 7)
        x_patch_start = np.random.randint(0, _channel.shape[1] - 51)
        y_patch_start = np.random.randint(0, _channel.shape[2] - 51)

        _channel[z_patch_start:z_patch_length,
                 x_patch_start:x_patch_length,
                 y_patch_start:y_patch_length] = 0

        return _channel

    @staticmethod
    def _rotate_channels(nephrin, collagen4, wga, label):
        # Define the rotation angles
        angle_x = np.random.randint(0, 20)  # rotation around the x-axis
        angle_y = np.random.randint(0, 20)  # rotation around the y-axis
        angle_z = np.random.randint(0, 20)  # rotation around the z-axis

        def rotate_voxel_space(voxel_space):
            # Rotate around each axis
            rotated_voxel_space = rotate(voxel_space, angle=angle_x, axes=(1, 2), reshape=False, order=1)
            rotated_voxel_space = rotate(rotated_voxel_space, angle=angle_y, axes=(0, 2), reshape=False, order=1)
            rotated_voxel_space = rotate(rotated_voxel_space, angle=angle_z, axes=(0, 1), reshape=False, order=1)

            return rotated_voxel_space

        return (rotate_voxel_space(nephrin),
                rotate_voxel_space(collagen4),
                rotate_voxel_space(wga),
                rotate_voxel_space(label))
