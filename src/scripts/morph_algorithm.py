"""
Author: Arash Fatehi
Date:   24.05.2023
"""

# Python Imports
import itertools

# Library Imports
from torch import Tensor
import torch
import torch.nn.functional as F
import tifffile
import imageio
import numpy as np

# Local Imports

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prime_no = 29
ave_kernel_size = 5

norm1 = 1
norm2 = 0.7071
norm3 = 0.5773

# device = 'cpu'

direction_vectors = torch.Tensor([[norm1, 0, 0],   # 1. Up
                                  [-norm1, 0, 0],  # 2. Down
                                  [0, norm1, 0],   # 3. Right
                                  [0, -norm1, 0],  # 4. Left
                                  [0, 0, norm1],   # 5. Front
                                  [0, 0, -norm1],  # 6. Back
                                  [norm2, norm2, 0],    # 7. Up-Right
                                  [norm2, -norm2, 0],   # 8. Up-Left
                                  [-norm2, norm2, 0],   # 9. Down-Right
                                  [-norm2, -norm2, 0],  # 10. Down-Left
                                  [norm2, 0, norm2],    # 11. Up-Front
                                  [norm2, 0, -norm2],   # 12. Up-Back
                                  [-norm2, 0, norm2],   # 13. Down-Front
                                  [-norm2, 0, -norm2],  # 14. Down-Back
                                  [0, norm2, norm2],    # 15. Right-Front
                                  [0, norm2, -norm2],   # 16. Right-Back
                                  [0, -norm2, norm2],   # 17. Left-Front
                                  [0, -norm2, -norm2],  # 18. Left-Back
                                  [norm3, norm3, norm3],     # 19. Up-Right-Front
                                  [norm3, norm3, -norm3],    # 20. Up-Right-Back
                                  [norm3, -norm3, norm3],    # 21. Up-Left-Front
                                  [-norm3, norm3, norm3],    # 22. Down-Right-Front
                                  [-norm3, -norm3, norm3],   # 23. Down-Left-Front
                                  [-norm3, norm3, -norm3],   # 24. Down-Right-Back
                                  [norm3, -norm3, -norm3],   # 25. Up-Left-Back
                                  [-norm3, -norm3, -norm3],  # 26. Down-Left-Back
                                  ]).to(device)


displacement_vectors = torch.Tensor([[1.0, 0.5, 0.5],  # 1. Up
                                     [0.0, 0.5, 0.5],  # 2. Down
                                     [0.5, 1.0, 0.5],  # 3. Right
                                     [0.5, 0.0, 0.5],  # 4. Left
                                     [0.5, 0.5, 1.0],  # 5. Front
                                     [0.5, 0.5, 0.0],  # 6. Back
                                     [1.0, 1.0, 0.5],  # 7. Up-Right
                                     [1.0, 0.0, 0.5],  # 8. Up-Left
                                     [0.0, 1.0, 0.5],  # 9. Down-Right
                                     [0.0, 0.0, 0.5],  # 10. Down-Left
                                     [1.0, 0.5, 1.0],  # 11. Up-Front
                                     [1.0, 0.5, 0.0],  # 12. Up-Back
                                     [0.0, 0.5, 1.0],  # 13. Down-Front
                                     [0.0, 0.5, 0.0],  # 14. Down-Back
                                     [0.5, 1.0, 1.0],  # 15. Right-Front
                                     [0.5, 1.0, 0.0],  # 16. Right-Back
                                     [0.5, 0.0, 1.0],  # 17. Left-Front
                                     [0.5, 0.0, 0.0],  # 18. Left-Back
                                     [1.0, 1.0, 1.0],  # 19. Up-Right-Front
                                     [1.0, 1.0, 0.0],  # 20. Up-Right-Back
                                     [1.0, 0.0, 1.0],  # 21. Up-Left-Front
                                     [0.0, 1.0, 1.0],  # 22. Down-Right-Front
                                     [0.0, 0.0, 1.0],  # 23. Down-Left-Front
                                     [0.0, 1.0, 0.0],  # 24. Down-Right-Back
                                     [1.0, 0.0, 0.0],  # 25. Up-Left-Back
                                     [0.0, 0.0, 0.0],  # 26. Down-Left-Back
                                     ]).to(device)


def load_voxel_space(_path: str) -> Tensor:
    with tifffile.TiffFile(_path) as tiff:
        image = tiff.asarray()
        # image = image[0:10, 400:500, 400:500]
        return torch.tensor(image, device=device, requires_grad=False)


def draw(_file_path, _input):
    _input = torch.squeeze(_input)
    _input = torch.squeeze(_input)
    _input = _input.cpu().numpy().astype(np.uint8)
    with imageio.get_writer(_file_path, mode='I') as writer:
        for index in range(_input.shape[0]):
            writer.append_data(_input[index])


def morph():
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_grad_enabled(False)

    voxel_space = load_voxel_space('/data/afatehi/prediction.tif')

    voxel_space[voxel_space == 255] = 1
    voxel_space = voxel_space.view(tuple(itertools.chain((1, 1),
                                                         voxel_space.shape)))

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
                          ]).to(device)
    kernel = kernel.view(1, 1, 3, 3, 3)

    surface_voxels = F.conv3d(voxel_space,
                              kernel,
                              stride=1,
                              padding='same')

    surface_voxels = surface_voxels * voxel_space

    surface_voxels[surface_voxels == 0] = 128.0
    surface_voxels[surface_voxels < 26] = 1.0
    surface_voxels[surface_voxels == 26] = prime_no
    surface_voxels[surface_voxels == 128] = 0.0

    surface_mask = (surface_voxels == 1.0).int().float()

    # draw('surface.gif', surface_mask)

    pos = torch.ones(3, 3).to(device)
    zero = torch.zeros(3, 3).to(device)
    neg = pos * -1

    kernel_y_slope = torch.stack((neg, zero, pos), dim=2).to(device)
    kernel_y_slope = kernel_y_slope.view(1, 1, 3, 3, 3)

    kernel_z_slope = torch.stack((neg, zero, pos), dim=0).to(device)
    kernel_z_slope = kernel_z_slope.view(1, 1, 3, 3, 3)

    kernel_x_slope = torch.stack((neg, zero, pos), dim=1).to(device)
    kernel_x_slope = kernel_x_slope.view(1, 1, 3, 3, 3)

    field_x_slope = F.conv3d(surface_voxels,
                             kernel_x_slope,
                             stride=1,
                             padding='same')
    field_x_slope = field_x_slope * surface_mask

    field_y_slope = F.conv3d(surface_voxels,
                             kernel_y_slope,
                             stride=1,
                             padding='same')
    field_y_slope = field_y_slope * surface_mask

    field_z_slope = F.conv3d(surface_voxels,
                             kernel_z_slope,
                             stride=1,
                             padding='same')
    field_z_slope = field_z_slope * surface_mask

    field_slope = torch.stack((field_z_slope,
                               field_x_slope,
                               field_y_slope), dim=len(field_x_slope.shape))

    field_slope = field_slope / prime_no

    trunced_slope = torch.trunc(field_slope)
    field_slope = field_slope - trunced_slope

    field_slope = field_slope * prime_no
    field_slope = field_slope + trunced_slope

    # draw('./x_slope.gif', field_slope[:, :, :, :, :, 1])
    # draw('./y_slope.gif', field_slope[:, :, :, :, :, 2])
    # draw('./z_slope.gif', field_slope[:, :, :, :, :, 0])

    del voxel_space
    del field_x_slope
    del field_y_slope
    del field_z_slope
    torch.cuda.empty_cache()

    kernel_average = torch.ones(ave_kernel_size, ave_kernel_size, ave_kernel_size).to(device)
    kernel_average = kernel_average.view(1, 1, ave_kernel_size, ave_kernel_size, ave_kernel_size)

    z_slope = field_slope[:, :, :, :, :, 0]
    x_slope = field_slope[:, :, :, :, :, 1]
    y_slope = field_slope[:, :, :, :, :, 2]

    z_div = (z_slope > 0).int().float()
    x_div = (x_slope > 0).int().float()
    y_div = (y_slope > 0).int().float()

    z_div = F.conv3d(z_div,
                     kernel_average,
                     stride=1,
                     padding='same')
    z_div[z_div == 0] = 1

    field_slope[:, :, :, :, :, 0] = F.conv3d(z_slope,
                                             kernel_average,
                                             stride=1,
                                             padding='same') / z_div

    x_div = F.conv3d(x_div,
                     kernel_average,
                     stride=1,
                     padding='same')
    x_div[x_div == 0] = 1

    field_slope[:, :, :, :, :, 1] = F.conv3d(x_slope,
                                             kernel_average,
                                             stride=1,
                                             padding='same') / x_div

    y_div = F.conv3d(y_div,
                     kernel_average,
                     stride=1,
                     padding='same')
    y_div[y_div == 0] = 1

    field_slope[:, :, :, :, :, 2] = F.conv3d(y_slope,
                                             kernel_average,
                                             stride=1,
                                             padding='same') / y_div

    directions = torch.tensordot(field_slope, direction_vectors, dims=([5], [1]))
    directions = torch.argmax(directions, dim=5)

    points_tensor = displacement_vectors[directions]
    index_z, index_x, index_y = torch.meshgrid(torch.arange(points_tensor.shape[2]),
                                               torch.arange(points_tensor.shape[3]),
                                               torch.arange(points_tensor.shape[4]),
                                               indexing='ij')
    index_tensor = torch.stack((index_z, index_x, index_y), dim=len(index_z.shape)).to(device)
    index_tensor = index_tensor.view(tuple(itertools.chain((1, 1), index_tensor.shape)))

    points_tensor += index_tensor

    del z_slope
    del x_slope
    del y_slope
    del z_div
    del x_div
    del y_div
    del index_z
    del index_x
    del index_y
    del index_tensor
    del directions
    torch.cuda.empty_cache()

    distance_tesnor = torch.zeros(surface_mask.shape).to(device)

    size_z = surface_mask.shape[2]
    size_x = surface_mask.shape[3]
    size_y = surface_mask.shape[4]

    for z, matrix in enumerate(surface_mask[0][0]):
        for x, column in enumerate(matrix):
            for y, voxel in enumerate(column):
                if voxel != 0:
                    print("=================================")
                    print(f"Finding the distance for voxel on: {z}, {x}, {y}")
                    point = points_tensor[0][0][z][x][y]
                    slope = field_slope[0][0][z][x][y]
                    # print(f"The slope is: {slope}")
                    # print(f"The point is: {point}")

                    shortest_distance = float('inf')

                    range_z = torch.arange(0, size_z).to(device)
                    t_z = (range_z - point[0]) / slope[0]
                    xx = slope[1] * t_z + point[1]
                    yy = slope[2] * t_z + point[2]

                    intersection_z_planes = torch.stack((range_z, xx, yy), dim=1)
                    truncated = intersection_z_planes.trunc().int().unique(dim=0)
                    for target in truncated:
                        surface_view = surface_mask.view(size_z, size_x, size_y)
                        index = target.detach().cpu().numpy()
                        in_range = index[0] > 0 and index[0] < size_z
                        in_range = in_range and index[1] > 0 and index[1] < size_x
                        in_range = in_range and index[2] > 0 and index[2] < size_y
                        if in_range and surface_view[index[0]][index[1]][index[2]]:
                            distance = target - point
                            is_valid = torch.dot(distance, slope).detach().cpu().numpy().item() >= 0
                            distance = torch.dot(distance, distance).detach().cpu().numpy().item()
                            if is_valid:
                                if distance < shortest_distance:
                                    shortest_distance = distance

                    range_x = torch.arange(0, size_x).to(device)
                    t_x = (range_x - point[1]) / slope[1]
                    zz = slope[0] * t_x + point[0]
                    yy = slope[2] * t_x + point[2]

                    intersection_x_planes = torch.stack((zz, range_x, yy), dim=1)
                    truncated = intersection_x_planes.trunc().int().unique(dim=0)
                    for target in truncated:
                        surface_view = surface_mask.view(size_z, size_x, size_y)
                        index = target.detach().cpu().numpy()
                        in_range = index[0] > 0 and index[0] < size_z
                        in_range = in_range and index[1] > 0 and index[1] < size_x
                        in_range = in_range and index[2] > 0 and index[2] < size_y
                        if in_range and surface_view[index[0]][index[1]][index[2]]:
                            distance = target - point
                            is_valid = torch.dot(distance, slope).detach().cpu().numpy().item() >= 0
                            distance = torch.dot(distance, distance).detach().cpu().numpy().item()
                            if is_valid:
                                if distance < shortest_distance:
                                    shortest_distance = distance

                    range_y = torch.arange(0, size_y).to(device)
                    t_y = (range_y - point[2]) / slope[2]
                    zz = slope[0] * t_y + point[0]
                    xx = slope[1] * t_y + point[1]

                    intersection_y_planes = torch.stack((zz, xx, range_y), dim=1)
                    truncated = intersection_y_planes.trunc().int().unique(dim=0)
                    for target in truncated:
                        surface_view = surface_mask.view(size_z, size_x, size_y)
                        index = target.detach().cpu().numpy()
                        in_range = index[0] > 0 and index[0] < size_z
                        in_range = in_range and index[1] > 0 and index[1] < size_x
                        in_range = in_range and index[2] > 0 and index[2] < size_y
                        if in_range and surface_view[index[0]][index[1]][index[2]]:
                            distance = target - point
                            is_valid = torch.dot(distance, slope).detach().cpu().numpy().item() >= 0
                            distance = torch.dot(distance, distance).detach().cpu().numpy().item()
                            if is_valid:
                                if distance < shortest_distance:
                                    shortest_distance = distance

                    distance_tesnor[0][0][z][x][y] = shortest_distance

    draw("distance.gif", distance_tesnor)
    with open("result.npy", 'wb') as f:
        np.save(f, distance_tesnor.detach().cpu().numpy())


if __name__ == '__main__':
    morph()
