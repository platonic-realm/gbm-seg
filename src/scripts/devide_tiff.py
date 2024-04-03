"""
Author: Arash Fatehi
Date:   06.12.2022
"""

import tifffile
import numpy as np
import random

source_list = [
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.COL4-mutation.tif',
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.Control-mouse.01.tif',
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.Pod-mutation.BDP669.Series005.tif',
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.Pod-mutation.BDP669.Series011.tif',
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.Pod-mutation.BDP672.Series004.tif',
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.COL4-mutation.CKM104.Series2.Annotated.tif',
        '/data/afatehi/gbm/experiments/pod-full/datasets/ds_raw/NCWM.COL4-mutation.CKM104.Series005.Annotated.tif',
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
