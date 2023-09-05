"""
Author: Arash Fatehi
Date:   10.08.2023
"""

# Python Imports

# Library Imports
import imageio
import torch
import numpy as np

vs_size = (100, 100, 100)

big_circle_radius = 50
small_circlue_radius = 25


def draw(_file_path, _input):
    _input = _input.astype(np.uint8)
    with imageio.get_writer(_file_path, mode='I') as writer:
        for index in range(_input.shape[0]):
            writer.append_data(_input[index])


def cube():
    grid = torch.meshgrid(torch.arange(100),
                          torch.arange(100),
                          torch.arange(100),
                          indexing='ij')

    mask0 = torch.logical_and(grid[0] > 25, grid[0] < 75)
    mask1 = torch.logical_and(grid[1] > 25, grid[1] < 75)
    mask2 = torch.logical_and(grid[2] > 25, grid[2] < 75)

    mask = torch.logical_and(mask0, mask1)
    mask = torch.logical_and(mask, mask2).int()

    mask = mask.cpu().numpy()

    mask[mask == 1] = 255

    return mask


def circle():
    pass


def cylinder():
    pass


def torus():
    pass


def circle_circle():
    grid = torch.meshgrid(torch.arange(100),
                          torch.arange(100),
                          torch.arange(100),
                          indexing='ij')

    center = torch.tensor([50, 50, 50], dtype=torch.float32)
    distances = torch.sqrt(
                (grid[0] - center[0])**2 +
                (grid[1] - center[1])**2 +
                (grid[2] - center[2])**2
            )

    mask = (distances <= big_circle_radius).int()

    center = torch.tensor([75, 75, 75], dtype=torch.float32)
    distances = torch.sqrt(
                (grid[0] - center[0])**2 +
                (grid[1] - center[1])**2 +
                (grid[2] - center[2])**2
            )

    small_mask = (distances <= small_circlue_radius)

    mask[small_mask] = 0
    mask = mask.cpu().numpy()

    mask[mask == 1] = 255

    return mask


def circle_cylinder():
    pass


def cube_circle():
    pass


def cube_cylinder():
    pass


if __name__ == "__main__":
    cube = circle_circle()
    np.save("/data/afatehi/gbm/cube.npy", cube)
