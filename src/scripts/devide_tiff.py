"""
Author: Arash Fatehi
Date:   06.12.2022
"""

import tifffile
import numpy as np
import random

source_list = [
        '/data/afatehi/gbm/data/ds_train/CKM104.Series2.Annotated.Nephrin1.WGA2.COLIV3.Mask4.tif',
        '/data/afatehi/gbm/data/ds_train/CKM104.Series005.Annotated.Nephrin1.WGA2.COLIV3.Mask4.tif',
        '/data/afatehi/gbm/data/ds_train/COL4-mutation.Nephrin1.WGA2.COLIV3.Mask4.tif',
        '/data/afatehi/gbm/data/ds_train/Control-mouse.Nephrin1.WGA2.COLIV3.Mask4.tif',
        ]

size = 3
start = 5

for source_file in source_list:
    image = tifffile.imread(source_file)
    image_shape = image.shape

    valid_start_index = (start + random.randint(1, 3))
    valid_end_index = valid_start_index + size

    valid_start = valid_start_index * 128
    valid_end = valid_end_index * 128

    train1 = image[:, :, :valid_start, :]
    valid = image[:, :, valid_start:valid_end, :]
    train2 = image[:, :, valid_end:, :]

    tifffile.imwrite(f"{source_file}_train1.tif",
                     train1,
                     shape=train1.shape)

    tifffile.imwrite(f"{source_file}_train2.tif",
                     train2,
                     shape=train2.shape)

    tifffile.imwrite(f"{source_file}_valid.tif",
                     valid,
                     shape=valid.shape)
