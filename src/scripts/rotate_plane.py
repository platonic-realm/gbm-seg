"""
Author: Arash Fatehi
Date:   28.04.2022
"""

import tifffile
import numpy as np
import torch.multiprocessing as mp


def rotate_plane(_plane, _angle):
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
            # translate the rotated coordinates back to the original position
            final_coordinates = rotated_coordinates + center
            # round the final coordinates to the nearest integer
            final_coordinates = np.round(final_coordinates).astype(int)
            # check if the final coordinates are within the bounds of the plane
            if (final_coordinates[0] >= 0 and
                    final_coordinates[0] < shape[0] and
                    final_coordinates[1] >= 0 and
                    final_coordinates[1] < shape[1]):
                result_plane[i][j] = \
                    _plane[final_coordinates[0]][final_coordinates[1]]
    return result_plane

source_file = '/data/afatehi/gbm/data/gbm_train_ds/Series33.All.Possible.GBM.20230126.tif'
output_file = '/data/afatehi/training_augmented.tif'

image = tifffile.imread(source_file)
image = image.astype(np.float32)

image[:, 3, :, :][image[:, 3, :, :] >= 255] = -1
image[:, 3, :, :][image[:, 3, :, :] >= 0] = 255
image[:, 3, :, :][image[:, 3, :, :] == -1] = 0

with mp.Pool(mp.cpu_count()) as pool:
    processes = [
        [None for _ in range(image.shape[1])] for _ in range(image.shape[0])]
    results = [
        [None for _ in range(image.shape[1])] for _ in range(image.shape[0])]

    for z in range(image.shape[0]):
        for c in range(image.shape[1]):
            processes[z][c] = pool.apply_async(rotate_plane,
                                               args=(image[z, c, :, :], z/3))

    for z in range(image.shape[0]):
        for c in range(image.shape[1]):
            results[z][c] = processes[z][c].get()

    for z in range(image.shape[0]):
        for c in range(image.shape[1]):
            image[z, c, :, :] = results[z][c]


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
