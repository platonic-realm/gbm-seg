"""
Author: Arash Fatehi
Date:   06.12.2022
"""

import tifffile

image = tifffile.imread(
        './Merged.tif')
image_shape = image.shape

training = image[:, :, :720, :]
validation = image[:, :, 720:, :]

tifffile.imwrite('training.tiff',
                 training,
                 shape=training.shape)

tifffile.imwrite('validation.tiff',
                 validation,
                 shape=validation.shape)
