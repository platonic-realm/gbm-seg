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


if __name__ == "__main__":

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
    draw("spheres.gif", mask)
    np.save("spheres.npy", mask)
