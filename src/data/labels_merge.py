"""
Author: Arash Fatehi
Date:   06.12.2022
"""

import json
import numpy as np
import tifffile

true_value = 1
possible_value = 2

gbm_true = tifffile.imread(
        '/data/afatehi/gbm/data/raw/Batch 3 - Annotated/GBM-True.tif')
gbm_possible = tifffile.imread(
        '/data/afatehi/gbm/data/raw/Batch 3 - Annotated/GBM-Possible.tif')

nephrin = gbm_true[:, 0, :, :]
wga = gbm_true[:, 1, :, :]
collagen4 = gbm_true[:, 2, :, :]

true_label = gbm_true[:, 3, :, :]
possible_lable = gbm_possible[:, 3, :, :]

labels = np.zeros(true_label.shape, dtype=np.uint8)
labels[possible_lable == 0] = possible_value
labels[true_label == 0] = true_value

nephrin = np.expand_dims(nephrin, axis=1)
wga = np.expand_dims(wga, axis=1)
collagen4 = np.expand_dims(collagen4, axis=1)
labels = np.expand_dims(labels, axis=1)

result = np.concatenate([nephrin, wga, collagen4, labels], axis=1)

with tifffile.TiffFile('/data/afatehi/gbm/data/raw/Batch 3 - Annotated/GBM-True.tif') as tif:
    tif_tags = {}
    for tag in tif.pages[0].tags.values():
        name, value = tag.name, tag.value
        tif_tags[name] = value

    with open('tags.json', 'w') as fp:
        json.dump(tif_tags, fp)

tifffile.imwrite('./result.tif',
                 result,
                 shape=result.shape,
                 imagej=True,
                 metadata=tif_tags)
