"""
Author: Arash Fatehi
Date:   24.05.2023
"""

# Python Imports

# Library Imports
from torch import Tensor
import torch
import tifffile

# Local Imports

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_voxel_space(_path: str) -> Tensor:
    with tifffile.TiffFile(_path) as tiff:
        image = tiff.asarray()
        return torch.tensor(image, device=device, requires_grad=False)


def morph():
    voxel_space = load_voxel_space('/data/afatehi/prediction.tif')

    voxel_space[voxel_space == 255] = 1


if __name__ == '__main__':
    morph()
