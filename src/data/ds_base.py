# Python Imports
from enum import Enum
from abc import ABC, abstractmethod

# Library Imports
import torch
from torch.utils.data import Dataset
# from scipy.ndimage import zoom
import torch.nn.functional as Fn
import numpy as np

# Local Imports


class DatasetType(Enum):
    Supervised = 1
    Unsupervised = 2
    Inference = 3


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 _sample_dimension,
                 _pixel_per_step,
                 _scale_facor=1,
                 _dataset_type=DatasetType.Supervised,
                 _ignore_stride_mismatch=False,
                 _label_correction_function=None):

        self.sample_dimension = _sample_dimension

        self.scale_factor = _scale_facor

        self.dataset_type = _dataset_type

        self.ignore_stride_mismatch = _ignore_stride_mismatch

        self.label_correction_function = _label_correction_function

        self.sample_dimension = _sample_dimension

        self.pixel_per_step_x = _pixel_per_step[2]
        self.pixel_per_step_y = _pixel_per_step[1]
        self.pixel_per_step_z = _pixel_per_step[0]

    @abstractmethod
    def getNumberOfClasses(self):
        pass

    @abstractmethod
    def getNumberOfChannels(self):
        pass

    @staticmethod
    def interpolate_channel(_channel: np.ndarray,
                            _scale_facor) -> np.ndarray:
        channel = torch.unsqueeze(torch.from_numpy(_channel), dim=0)
        channel = torch.unsqueeze(channel, dim=0)
        zoomed = Fn.interpolate(channel,
                                scale_factor=(_scale_facor, 1, 1),
                                mode='trilinear')

        return zoomed

    def scale(self, image):
        # The shape is Depth, Channel, Height, Width
        image = np.array(image)
        image = image.astype(np.float32)

        scaled_image = np.empty(((image.shape[0])*self.scale_factor,
                                 image.shape[1],
                                 image.shape[2],
                                 image.shape[3]),
                                dtype=np.float32)

        scaled_image[:, 0, :, :] = self.interpolate_channel(image[:, 0, :, :],
                                                            self.scale_factor)
        scaled_image[:, 1, :, :] = self.interpolate_channel(image[:, 1, :, :],
                                                            self.scale_factor)
        scaled_image[:, 2, :, :] = self.interpolate_channel(image[:, 2, :, :],
                                                            self.scale_factor)
        return scaled_image

    def scale_2(self, image):
        # The shape is Depth, Channel, Height, Width
        scaled_image = np.empty(((image.shape[0]-1)*self.scale_factor,
                                 image.shape[1],
                                 image.shape[2],
                                 image.shape[3]),
                                dtype=np.float32)

        # To prevent out of bound exception in image[index+1],
        # copy the last layer in the beginning.
        scaled_image[-1] = image[-1]
        for i in range(scaled_image.shape[0]-1):
            index = int(i/self.scale_factor)
            i_mod = i % self.scale_factor
            if i_mod == 0:
                scaled_image[i] = image[index]
            else:
                scaled_image[i] = image[index] +\
                    (image[index+1] - image[index]) /\
                    self.scale_factor * i_mod

        return scaled_image

    @staticmethod
    def get_tiff_tags(_tiff):
        tiff_tags = {}
        for tag in _tiff.pages[0].tags.values():
            name, value = tag.name, tag.value
            tiff_tags[name] = value
        return tiff_tags

    def check_image_shape_compatibility(self,
                                        _shape,
                                        _file_name):

        if self.ignore_stride_mismatch:
            return

        # Checking image's z-axis
        criterion = (_shape[0] - self.sample_dimension[0]) % \
            self.pixel_per_step_z
        assert criterion == 0, \
            (f'Dimension of file "{_file_name}" on z-axis'
             f'={_shape[0]} and is not compatible with'
             f' stride={self.pixel_per_step_z} and'
             f' sample size={self.sample_dimension[0]}.\n'
             '(Image Dimension - Sample Dimension) % Stride should be 0.\n'
             f'For z-axis: ({_shape[0]} - {self.sample_dimension[0]}) %'
             f' {self.pixel_per_step_z} = {criterion}')

        # Checking image's y-axis
        criterion = (_shape[2] - self.sample_dimension[2]) % \
            self.pixel_per_step_y
        assert criterion == 0, \
            (f'Dimension of file "{_file_name}" on y-axis'
             f'={_shape[2]} and is not compatible with'
             f' stride={self.pixel_per_step_y} and'
             f' sample size={self.sample_dimension[2]}.\n'
             '(Image Dimension - Sample Dimension) % Stride should be 0.\n'
             f'For y-axis: ({_shape[2]} - {self.sample_dimension[2]}) %'
             f' {self.pixel_per_step_y} = {criterion}')

        # Checking image's x-axis
        criterion = (_shape[3] - self.sample_dimension[1]) % \
            self.pixel_per_step_x
        assert criterion == 0, \
            (f'Dimension of file "{_file_name}" on x-axis'
             f'={_shape[3]} and is not compatible with'
             f' stride={self.pixel_per_step_x} and'
             f' sample size={self.sample_dimension[1]}.\n'
             '(Image Dimension - Sample Dimension) % Stride should be 0.\n'
             f'For x-axis: ({_shape[3]} - {self.sample_dimension[1]}) %'
             f' {self.pixel_per_step_x} = {criterion}')
