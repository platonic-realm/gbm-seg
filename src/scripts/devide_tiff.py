"""
Author: Arash Fatehi
Date:   06.12.2022
"""

import tifffile
import numpy as np
import random

source_list = [
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.CKM104.Series002.Annotated-mutation.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.CKM104.Series005.Annotated-mutation.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.AUY380.Series008.Control.20240723.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.AUY381.Series002.Control-mouse.20240805.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.BDP669.Series005.Pod-mutation.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.BDP669.Series011.Pod-mutation.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.BDP672.Series004.Pod.Done.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.BDP675.Series008.Pod-mutation.20240722.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.CKM103.Series004.Control-mouse.tif',
        '/data/afatehi/gbm/data/ds_mouse/ds_train/raw/NCWM.CKM104.Series003.COL4-mutation.tif',
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
