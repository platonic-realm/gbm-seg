# Python Imports
import itertools
import logging

import imageio
import numpy as np

# Library Imports
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

# Local Imports

# Constants
norm1 = 1
norm2 = 0.7071
norm3 = 0.5773

# PSF values
PSFLateral = 149  # 149nm
PSFAxial = 434  # 434nm
# Thickness = ( Measured^2 - ((PSFLateral * cos(a))^2 + (PSFAxial * sin(a))^2) ) ^ 0.5

# Minimum ray length (in voxels) before a surface hit counts as the
# opposite wall — a hit must clear the source voxel's own 3x3x3 block.
# sqrt(3) is one voxel-diagonal; it is comfortably below the thinnest
# real GBM (the membrane spans many voxels after the x6 Z-upsampling),
# so genuinely thin membrane is still measured.
MIN_RAY_VOXELS = 3 ** 0.5


def draw(_file_path, _input):
    _input = torch.squeeze(_input)
    _input = torch.squeeze(_input)
    _input[_input > 0] = 255
    _input = _input.cpu().numpy().astype(np.uint8)
    with imageio.get_writer(_file_path, mode='I') as writer:
        for index in range(_input.shape[0]):
            writer.append_data(_input[index])


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
                                  ], device='cpu')

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
                                     ], device='cpu')


class Morph(nn.Module):
    """3D morphometric analysis: surface detection + per-voxel slope vectors +
    distance map, returning thickness and PSF-corrected thickness arrays."""

    def __init__(self,
                 _device: str,
                 _calc_displayments: bool = True,
                 _calc_displayments_on_cpu: bool = True,
                 _ave_kernel_size: int = 5,
                 _voxel_size: float = 50):

        global direction_vectors
        global displacement_vectors

        super().__init__()

        self.device = _device
        self.calc_displacements = _calc_displayments
        self.ave_kernel_size = _ave_kernel_size
        self.calc_displacements_on_cpu = _calc_displayments_on_cpu
        self.voxel_size = _voxel_size  # in nano meter

        # Taking care of tensor's location
        if self.calc_displacements and not _calc_displayments_on_cpu:
            displacement_vectors = displacement_vectors.to(self.device)
            direction_vectors = direction_vectors.to(self.device)

        # Using this kernel we count and sum the number of voxels in the neighbourhood
        # of the center voxel, the value of background is zero, and forground is one
        # therefore any voxel with less than 26 neighbours will be considered
        # a surface voxel
        self.surface_kernel = torch.tensor([
                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]],
                          [[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]],
                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]],
                                            ], device=self.device)
        # Reshaping the kernel to be compatible with conv3d operation
        self.surface_kernel = self.surface_kernel.view(1, 1, 3, 3, 3)

        ########################################
        # In the forward pass we are calculating a vector pointing inside the shape
        # for each of the voxels, we do this using the relative position of neighbours
        # for example for each of the neighbours, if they are above the voxel, we will add
        # one to z element of the slope vector of that voxel, we do this for all surface voxels
        # and for Z, X, Y axis. To vectorize the operaiton, we define and use below kernels
        # Combined with a conv3d operation, each do the described operation for one the axis for all
        # the voxels in surface_voxels tensor

        pos = torch.ones(3, 3)
        zero = torch.zeros(3, 3)
        neg = pos * -1

        self.y_slope_kernel = torch.stack((neg, zero, pos), dim=2).to(self.device)
        # Reshaping the kernel to be compatible with conv3d operation
        self.y_slope_kernel = self.y_slope_kernel.view(1, 1, 3, 3, 3)

        self.z_slope_kernel = torch.stack((neg, zero, pos), dim=0).to(self.device)
        # Reshaping the kernel to be compatible with conv3d operation
        self.z_slope_kernel = self.z_slope_kernel.view(1, 1, 3, 3, 3)

        self.x_slope_kernel = torch.stack((neg, zero, pos), dim=1).to(self.device)
        # Reshaping the kernel to be compatible with conv3d operation
        self.x_slope_kernel = self.x_slope_kernel.view(1, 1, 3, 3, 3)
        ########################################

        self.averaging_kernel = torch.ones(size=(self.ave_kernel_size,
                                                 self.ave_kernel_size,
                                                 self.ave_kernel_size),
                                           device=self.device)
        self.averaging_kernel = self.averaging_kernel.view(1, 1,
                                                           self.ave_kernel_size,
                                                           self.ave_kernel_size,
                                                           self.ave_kernel_size)

    def forward(self,
                _voxel_space):
        with torch.no_grad():
            if _voxel_space.device != self.device:
                voxel_space = _voxel_space.to(self.device)
            else:
                voxel_space = _voxel_space.clone()
            # Adding two dimenstions to the voxel_space, so it would be compatiple
            # with conv3d operation
            voxel_space = voxel_space.view(tuple(itertools.chain((1, 1),
                                                 voxel_space.shape)))

            # Finding the surface voxels using a 3D convultion operation and surface_kernel
            surface_voxels = F.conv3d(voxel_space,
                                      self.surface_kernel,
                                      stride=1,
                                      padding='same')

            # The conv3d operation calcualte all the forground voxels in the neighbourhood,
            # it does this for all voxels, even for the background voxels, therefore
            # the next step is to filter out background voxels by performing an element-wise
            # multiplication between surface_voxels and voxel_space
            surface_voxels = surface_voxels * voxel_space

            # At this point, all the needed information is stored in surface_voxels tensor
            # so we don't need the voxel_space anymore, we delete it and empty the cuda cache
            # in case, we were performing the operation on GPU
            del voxel_space
            del _voxel_space
            self.empty_cache()

            # We dont want background voxels to be included while checking surface_voxels < 26
            surface_voxels[surface_voxels == 0] = 128.0

            # All the surface voxels will be 1.0 from now on
            surface_voxels[surface_voxels < 26] = 1.0

            # Set the background back to 0.0
            surface_voxels[surface_voxels == 128] = 0.0

            # The surface_mask only consists of surface voxels
            surface_mask = (surface_voxels == 1.0).int().float()

            # Now that we have the surface_mask, we can set the value of
            # all voxels to 1.0 to calculate the slopes
            surface_voxels[surface_voxels == 26] = 1.0

            # Calculating the X element of slope vectors for all the surface voxels
            x_slope = F.conv3d(surface_voxels,
                               self.x_slope_kernel,
                               stride=1,
                               padding='same')
            # We are only interested in surface voxels
            x_slope = x_slope * surface_mask

            # Calculating the Y element of slope vectors for all the surface voxels
            y_slope = F.conv3d(surface_voxels,
                               self.y_slope_kernel,
                               stride=1,
                               padding='same')
            # We are only interested in surface voxels
            y_slope = y_slope * surface_mask

            # Calculating the Z element of slope vectors for all the surface voxels
            z_slope = F.conv3d(surface_voxels,
                               self.z_slope_kernel,
                               stride=1,
                               padding='same')
            # We are only interested in surface voxels
            z_slope = z_slope * surface_mask

            slope_tensor = torch.stack((z_slope,
                                        x_slope,
                                        y_slope), dim=len(x_slope.shape))

            del x_slope
            del y_slope
            del z_slope
            self.empty_cache()

            z_slope = slope_tensor[:, :, :, :, :, 0]
            x_slope = slope_tensor[:, :, :, :, :, 1]
            y_slope = slope_tensor[:, :, :, :, :, 2]

            # Check if it should be slope != 0 instead of > 0
            z_div = (z_slope != 0).int().float()
            x_div = (x_slope != 0).int().float()
            y_div = (y_slope != 0).int().float()

            z_div = F.conv3d(z_div,
                             self.averaging_kernel,
                             stride=1,
                             padding='same')
            z_div[z_div == 0] = 1

            slope_tensor[:, :, :, :, :, 0] = F.conv3d(z_slope,
                                                      self.averaging_kernel,
                                                      stride=1,
                                                      padding='same') / z_div

            del z_div
            self.empty_cache()

            x_div = F.conv3d(x_div,
                             self.averaging_kernel,
                             stride=1,
                             padding='same')
            x_div[x_div == 0] = 1

            slope_tensor[:, :, :, :, :, 1] = F.conv3d(x_slope,
                                                      self.averaging_kernel,
                                                      stride=1,
                                                      padding='same') / x_div
            del x_div
            self.empty_cache()

            y_div = F.conv3d(y_div,
                             self.averaging_kernel,
                             stride=1,
                             padding='same')
            y_div[y_div == 0] = 1

            slope_tensor[:, :, :, :, :, 2] = F.conv3d(y_slope,
                                                      self.averaging_kernel,
                                                      stride=1,
                                                      padding='same') / y_div

            z_slope = z_slope.to('cpu')
            x_slope = x_slope.to('cpu')
            y_slope = y_slope.to('cpu')
            del y_div
            self.empty_cache()

            displacement_device = 'cpu' if self.calc_displacements_on_cpu else self.device
            slope_tensor = slope_tensor.to(displacement_device)

            self.empty_cache()

            # In order to chose a point on the voxel to calculate the distance from
            # we are predefining 26 unit vector directions and their correspondent displacement on
            # a voxel, and first we calculate the dot product of the slope of the vector to each
            # of those unit vectors and select the index of the vector that resulted in the max value
            directions = torch.tensordot(slope_tensor,
                                         direction_vectors,
                                         dims=([5], [1]))
            directions = torch.argmax(directions, dim=5)

            directions.to(displacement_device)
            # Here we use that indexes to select a displacement out of the 26 available for each voxel
            points_tensor = displacement_vectors[directions]
            index_z, index_x, index_y = torch.meshgrid(torch.arange(points_tensor.shape[2]),
                                                       torch.arange(points_tensor.shape[3]),
                                                       torch.arange(points_tensor.shape[4]),
                                                       indexing='ij')

            index_tensor = torch.stack((index_z, index_x, index_y),
                                       dim=len(index_z.shape)).to(displacement_device)
            index_tensor = index_tensor.view(tuple(itertools.chain((1, 1),
                                                                   index_tensor.shape)))

            points_tensor += index_tensor

            del index_z
            del index_x
            del index_y
            del index_tensor
            del directions
            self.empty_cache()

            distance_tesnor = torch.zeros(surface_mask.shape, device=self.device)

            size_z = surface_mask.shape[2]
            size_x = surface_mask.shape[3]
            size_y = surface_mask.shape[4]
            for z, matrix in tqdm(enumerate(surface_mask[0][0]),
                                  desc="Morph Analysis"):
                for x, column in enumerate(matrix):
                    logging.debug("######################")
                    logging.debug("Calculating the distances for column Z=%d and X=%d", z, x)

                    condition = column == 1
                    if condition.int().sum() == 0:
                        logging.debug("No surface voxel on this cloumn, skipping...")
                        continue

                    points = points_tensor[0][0][z][x].to(self.device)
                    slopes = slope_tensor[0][0][z][x].to(self.device)
                    shortest_distance = torch.full_like(column, float('inf'))

                    for axis in range(3):
                        shortest_distance = self.calculate_intersections(axis,
                                                                         column,
                                                                         shortest_distance,
                                                                         points,
                                                                         slopes,
                                                                         surface_mask,
                                                                         [size_z, size_x, size_y])

                    distance_tesnor[0][0][z][x][:] = shortest_distance
                    self.empty_cache()

            # points_tensor is only used inside the loop above — free it (a
            # full (..., 3) volume) before the memory-heavy PSF correction.
            del points_tensor
            self.empty_cache()

            distance_tesnor[distance_tesnor.isinf()] = 0
            distance_tesnor[surface_mask <= 0] = 0
            distance_tesnor = distance_tesnor[0][0]
            distance_tesnor = torch.sqrt(distance_tesnor) * self.voxel_size

            # Calculate angles for PSF correction
            corrected_thickness, clamp_info = self.calculate_thickness_correction(
                slope_tensor=slope_tensor.squeeze(0).squeeze(0),
                surface_mask=surface_mask.squeeze(0).squeeze(0),
                distance_tensor=distance_tesnor)
            del slope_tensor  # not used past the PSF step
            self.empty_cache()

            x_slope_std = self.calculate_patched_std(x_slope, surface_mask)
            self.empty_cache()
            y_slope_std = self.calculate_patched_std(y_slope, surface_mask)
            self.empty_cache()
            z_slope_std = self.calculate_patched_std(z_slope, surface_mask)
            self.empty_cache()

            bumpiness_tensor = x_slope_std**2 + y_slope_std**2 + z_slope_std**2
            bumpiness_tensor = bumpiness_tensor.squeeze(-1).squeeze(-1).squeeze(-1)
            bumpiness_tensor = F.conv3d(bumpiness_tensor,
                                        self.averaging_kernel,
                                        stride=1,
                                        padding='same')
            bumpiness_tensor[surface_mask == 0] = 0
            bumpiness_tensor = bumpiness_tensor.squeeze()

            return distance_tesnor, corrected_thickness, bumpiness_tensor, clamp_info

    def calculate_patched_std(self,
                              _slope: Tensor,
                              _surface_mask: Tensor,
                              _kernel_size: int = 5):

        slope = _slope.to(self.device)

        padding = _kernel_size // 2
        padded = F.pad(slope,
                       (padding, padding, padding, padding, padding, padding),
                       mode='reflect')

        # Use unfold to create patches
        patches = padded.unfold(2,
                                _kernel_size,
                                1).unfold(3,
                                          _kernel_size,
                                          1).unfold(4,
                                                    _kernel_size,
                                                    1)
        # Calculate mean across the last three dimensions
        std = patches.std(dim=(-1, -2, -3), keepdim=True)
        std[_surface_mask == 0] = 0

        return std

    def calculate_intersections(self,
                                _axis: int,
                                _column: Tensor,
                                _shortest_distance: Tensor,
                                _points: Tensor,
                                _slopes: Tensor,
                                _surface_mask: Tensor,
                                _size: list):
        axis_list = [0, 1, 2]
        del axis_list[_axis]

        range_size = _size[_axis]
        del _size[_axis]

        # Create values between 0 to the size of the dimension
        range = torch.arange(0, range_size).to(self.device)
        range = range.unsqueeze(1)
        # The parametric equation of a line in 3D space is:
        # z = Mz*t + z0
        # x = Mx*t + x0
        # y = My*t + y0
        # Here we are calculating t values in case that the chosen axis = 0 ... range_size
        t = (range[:] - _points[:, _axis]) / _slopes[:, _axis]
        # Then we calculate the values for the other two dimensions, lets say we are
        # doing this for the Z axis, after knowing the value of t for Z = 0 ... size_z
        # we calculate the value for X and Y of intercestions. So we can find out the
        # coordinate of intercestion of the lines with all planes Z = 0 ... size_t
        other_axis_1 = _slopes[:, axis_list[0]] * t + _points[:, axis_list[0]]
        other_axis_2 = _slopes[:, axis_list[1]] * t + _points[:, axis_list[1]]

        range = range.expand(range.shape[0], len(_column))

        if _axis == 0:
            intersection_planes = torch.stack((range,
                                               other_axis_1,
                                               other_axis_2), dim=2)
        elif _axis == 1:
            intersection_planes = torch.stack((other_axis_1,
                                               range,
                                               other_axis_2), dim=2)
        elif _axis == 2:
            intersection_planes = torch.stack((other_axis_1,
                                               other_axis_2,
                                               range), dim=2)

        # `truncated` snaps each intersection to the integer base corner of
        # the voxel it lands in — used only for the integer surface_mask
        # lookup and the bounds tests below, not for the distance itself.
        truncated = intersection_planes.trunc()
        # We filter out out of bound values, because we are only interested in the
        # intercestions inside the voxel space
        condition = truncated[:, :, axis_list[0]] >= 0
        truncated[~condition] = torch.nan

        condition = truncated[:, :, axis_list[1]] >= 0
        truncated[~condition] = torch.nan

        condition = truncated[:, :, axis_list[0]] < _size[0]
        truncated[~condition] = torch.nan

        condition = truncated[:, :, axis_list[1]] < _size[1]
        truncated[~condition] = torch.nan

        # Another filter to keep only intercestions that are surface voxels.
        # Rows the bounds tests above rejected are all-nan here, and
        # nan.int() is INT_MIN — an out-of-bounds index (IndexError on CPU,
        # a silent out-of-bounds read on CUDA). Map nan/inf to a safe 0
        # index purely for the lookup: those rows are already nan in
        # `truncated` and stay excluded whatever surface_mask[0,0,0] returns.
        safe_index = truncated.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        condition = _surface_mask[0][0][safe_index[:, :, 0].int(),
                                        safe_index[:, :, 1].int(),
                                        safe_index[:, :, 2].int()].bool()
        truncated[~condition] = torch.nan

        # The ray endpoint is the EXACT plane crossing, not the voxel's base
        # corner — this mirrors the sub-voxel source point (_points) on the
        # target side, so the measured distance is the true on-ray length to
        # the far surface. `truncated` is used only for the integer lookups
        # above; every intersection it filtered out is carried over here so
        # an invalid hit stays invalid.
        endpoint = intersection_planes.clone()
        endpoint[truncated.isnan().any(dim=2)] = torch.nan

        distances = endpoint - _points
        validation = (distances * _slopes).sum(dim=2)
        condition = validation > 0
        distances[~condition] = torch.nan

        if distances.numel() > 0:
            distances = (distances * distances).sum(dim=2)

            # `distances` is the SQUARED distance here, so the cutoff is
            # squared too — drop hits closer than MIN_RAY_VOXELS (the
            # source voxel and its immediate neighbours).
            condition = distances > MIN_RAY_VOXELS ** 2
            distances[~condition] = torch.nan
            distances[torch.isnan(distances)] = torch.inf

            distances = torch.min(distances, dim=0)[0]

            condition = distances < _shortest_distance

            _shortest_distance = torch.where(condition=condition,
                                             input=distances,
                                             other=_shortest_distance)

        return _shortest_distance

    def calculate_thickness_correction(self,
                                       slope_tensor,
                                       surface_mask,
                                       distance_tensor):
        """
        Calculate the true thickness using PSF correction based on slope angles.

        Args:
            slope_tensor: Tensor containing slope vectors for each voxel
            surface_mask: Mask indicating surface voxels
            distance_tensor: Tensor containing distances before correction

        Returns:
            corrected_tensor: Distance tensor with PSF corrections
        """

        # PSF correction is a purely element-wise, per-voxel computation, so
        # the GPU buys it nothing. On a large volume the dozen-odd full-volume
        # intermediates below overflow GPU memory (a ~600M-voxel test volume
        # needs ~50 GB). Run it on CPU — the morph node has ample RAM — and
        # free each intermediate as soon as it is consumed to bound the peak.
        slope_tensor = slope_tensor.to('cpu')
        surface_mask = surface_mask.to('cpu')
        distance_tensor = distance_tensor.to('cpu')

        # Extract slope vectors for surface voxels only
        surface_slopes = slope_tensor * surface_mask.unsqueeze(-1)

        # Calculate magnitude of slope vectors
        slope_magnitude = torch.norm(surface_slopes, dim=-1, keepdim=True)

        # Avoid division by zero
        slope_magnitude = torch.where(slope_magnitude == 0,
                                      torch.ones_like(slope_magnitude),
                                      slope_magnitude)

        # Normalize slope vectors to unit vectors
        unit_slopes = surface_slopes / slope_magnitude
        del surface_slopes, slope_magnitude

        # Z-axis unit vector [0, 0, 1] - note: adjusted order to match your coordinate system
        z_unit = torch.tensor([0, 0, 1], device=slope_tensor.device, dtype=slope_tensor.dtype)

        # Calculate cos(beta) = dot product of unit slope vector with Z-axis unit vector
        # This gives us the cosine of the angle between the slope and Z-axis
        cos_beta = torch.sum(unit_slopes * z_unit, dim=-1, keepdim=True)
        del unit_slopes

        # Clamp to avoid numerical issues
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)

        # Calculate sin(beta) from cos(beta)
        sin_beta = torch.sqrt(1 - cos_beta**2)

        # Since alpha = π/2 - beta, we have:
        # cos(alpha) = cos(π/2 - beta) = sin(beta)
        # sin(alpha) = sin(π/2 - beta) = cos(beta)
        # Remove the extra dimension at the same time.
        cos_alpha = sin_beta.squeeze(-1)
        sin_alpha = cos_beta.squeeze(-1)

        # Calculate PSF term: PSF_Lateral^2 * cos^2(alpha) + PSF_Axial^2 * sin^2(alpha)
        psf_term = (PSFLateral**2 * cos_alpha**2 + PSFAxial**2 * sin_alpha**2)
        del cos_alpha, sin_alpha, cos_beta, sin_beta

        # Calculate corrected thickness: sqrt(measured^2 - PSF_term)
        measured_squared = distance_tensor**2

        # Diagnostic: count surface voxels where measured² < PSF², i.e. the
        # measured thickness is below the PSF resolution floor. Without this
        # clamp these voxels would have an imaginary corrected thickness; the
        # clamp silently makes the GBM "invisible" at those voxels, so we log
        # the rate to surface the failure mode.
        surface_bool = (surface_mask > 0)
        clamp_bool = (measured_squared < psf_term) & surface_bool
        clamp_count = int(clamp_bool.sum().item())
        surface_count = int(surface_bool.sum().item())
        del clamp_bool, surface_bool
        clamp_percentage = (100.0 * clamp_count / surface_count) if surface_count > 0 else 0.0
        logging.info(
            "PSF clamp activated on %d/%d surface voxels (%.2f%%) — "
            "corrected thickness is 0 for these (measured² < PSF²)",
            clamp_count, surface_count, clamp_percentage)
        clamp_info = {
            'clamp_count': clamp_count,
            'surface_count': surface_count,
            'clamp_percentage': clamp_percentage,
            'psf_lateral_nm': PSFLateral,
            'psf_axial_nm': PSFAxial,
        }

        # Ensure we don't take square root of negative values
        corrected_squared = torch.clamp(measured_squared - psf_term, min=0)
        del measured_squared, psf_term
        corrected_tensor = torch.sqrt(corrected_squared)
        del corrected_squared

        return corrected_tensor, clamp_info

    def empty_cache(self):
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
