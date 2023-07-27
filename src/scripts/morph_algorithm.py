"""
Author: Arash Fatehi
Date:   24.05.2023
"""

# Python Imports

# Library Imports
from torch import Tensor
import torch
import torch.nn.functional as F
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
    voxel_space = torch.unsqueeze(voxel_space, dim=0)
    voxel_space = torch.unsqueeze(voxel_space, dim=0)

    kernel = torch.tensor([
                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]],
                          [[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]],
                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]],
                          ])
    kernel = torch.unsqueeze(kernel, dim=0)
    kernel = torch.unsqueeze(kernel, dim=0)

    surface_voxels = F.conv3d(voxel_space,
                              kernel,
                              stride=1,
                              padding='same')
    print(torch.unique(surface_voxels))


if __name__ == '__main__':
    morph()
