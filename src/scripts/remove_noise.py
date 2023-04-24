import tifffile
import numpy as np
from skimage import measure, morphology


use_morphology = False
min_size = 100
# source_file = '/data/afatehi/gbm/data/gbm_valid_ds/Series38.All.Possible.GBM.2023-01-17.tif'
# output_file = '/data/afatehi/validation_erosion_ccl_dilation.tif'
source_file = '/data/afatehi/gbm/data/gbm_train_ds/Series33.All.Possible.GBM.20230126.tif'
output_file = '/data/afatehi/training_erosion_ccl_dilation.tif'

image = tifffile.imread(source_file)
image = image.astype(np.float32)

image[:, 3, :, :][image[:, 3, :, :] >= 255] = -1
image[:, 3, :, :][image[:, 3, :, :] >= 0] = 255
image[:, 3, :, :][image[:, 3, :, :] == -1] = 0


for i in range(image.shape[0]):

    if use_morphology:
        kernel_size = 3
        kernel = morphology.rectangle(kernel_size, kernel_size)
        eroded_image = morphology.erosion(image[i, 3, :, :], kernel)
        image[i, 3, :, :] = eroded_image

    sample = image[i, 3, :, :]
    labels = measure.label(sample, connectivity=1)
    # count pixels in each connected component
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # remove small connected components
    for label, count in zip(unique_labels, label_counts):
        if count < min_size and label != 0:
            image[i, 3, :, :][labels == label] = 0

    if use_morphology:
        kernel_size = 3
        kernel = morphology.rectangle(kernel_size, kernel_size)
        dilated_image = morphology.dilation(image[i, 3, :, :], kernel)
        image[i, 3, :, :] = dilated_image

with tifffile.TiffFile(source_file) as tif:
    tif_tags = {}
    for tag in tif.pages[0].tags.values():
        name, value = tag.name, tag.value
        tif_tags[name] = value

    tifffile.imwrite(output_file,
                     image,
                     shape=image.shape,
                     imagej=True,
                     metadata=tif_tags,
                     compression="lzw")
