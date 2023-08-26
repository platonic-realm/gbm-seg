"""
Author: Arash Fatehi
Date:   06.12.2022
"""

import tifffile

image = tifffile.imread(
        '/data/afatehi/gbm/experiments/VA-Big-Mouse/datasets/COL4-mutation.Nephrin1.WGA2.COLIV3.Mask4.tif')
image_shape = image.shape

training = image[:, :, :1440, :]
validation = image[:, :, 1440:, :]

tifffile.imwrite('training.tiff',
                 training,
                 shape=training.shape)

tifffile.imwrite('validation.tiff',
                 validation,
                 shape=validation.shape)
