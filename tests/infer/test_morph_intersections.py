"""Ray/surface intersection in `Morph.calculate_intersections`.

Two behaviours are pinned here:

* the ray endpoint is the EXACT plane crossing, not the target voxel's
  truncated base corner — the target side now has the same sub-voxel
  precision as the source point;
* the surface-mask lookup tolerates out-of-bounds intersections on CPU.
  Rows the bounds tests reject are all-nan, and ``nan.int()`` is INT_MIN
  — an out-of-bounds index (IndexError on CPU, a silent out-of-bounds
  read on CUDA) unless guarded.
"""

import torch

from src.infer.morph import Morph


def _run(points_y1, slope_y1, surface_voxels, size):
    """Cast one ray from column position y=1 and return the squared
    shortest-distance tensor.

    ``surface_voxels`` is a list of (z, x, y) voxels flagged as surface
    (the ray targets); ``size`` is (size_z, size_x, size_y).
    """
    size_z, size_x, size_y = size
    surface_mask = torch.zeros(1, 1, size_z, size_x, size_y)
    for (z, x, y) in surface_voxels:
        surface_mask[0, 0, z, x, y] = 1.0

    column = torch.zeros(size_y)          # only its length is read
    points = torch.zeros(size_y, 3)
    slopes = torch.zeros(size_y, 3)
    points[1] = torch.tensor(points_y1)
    slopes[1] = torch.tensor(slope_y1)

    morph = Morph(_device="cpu", _ave_kernel_size=3)
    shortest = torch.full((size_y,), float("inf"))
    for axis in range(3):
        shortest = morph.calculate_intersections(
            axis, column, shortest, points, slopes, surface_mask,
            [size_z, size_x, size_y])
    return shortest


def test_intersection_endpoint_is_exact_crossing():
    """Distance is measured to the exact ray/plane crossing.

    The ray from (2.0, 1.5, 1.5) along (1.0, 0.3, 0.0) crosses plane z=4
    at (4, 2.1, 1.5); the squared distance to that exact point is
    2**2 + 0.6**2 = 4.36. Truncating to the voxel base corner (4, 2, 1) —
    the pre-fix behaviour — would instead give 2**2 + 0.5**2 + 0.5**2 = 4.50.
    """
    shortest = _run(points_y1=[2.0, 1.5, 1.5],
                    slope_y1=[1.0, 0.3, 0.0],
                    surface_voxels=[(4, 2, 1)],
                    size=(8, 6, 3))
    assert abs(shortest[1].item() - 4.36) < 1e-4   # exact crossing, not 4.50
    # y=0 and y=2 carry no ray (zero slope) — they stay unset.
    assert shortest[0].isinf() and shortest[2].isinf()


def test_min_ray_cutoff_excludes_near_hits():
    """A hit inside the source voxel's neighbourhood is rejected.

    The ray from (2, 2, 2) along (1, 1, 0) crosses a surface voxel at
    (3, 3, 2) only sqrt(2) voxels away (squared distance 2.0) — inside
    the MIN_RAY_VOXELS = sqrt(3) gate — and a second at (5, 5, 2),
    sqrt(18) away. The near hit must be dropped and the far one kept, so
    the result is 18.0. The old `> 1.7` squared-cutoff would have kept
    the near hit and returned 2.0.
    """
    shortest = _run(points_y1=[2.0, 2.0, 2.0],
                    slope_y1=[1.0, 1.0, 0.0],
                    surface_voxels=[(3, 3, 2), (5, 5, 2)],
                    size=(8, 8, 4))
    assert abs(shortest[1].item() - 18.0) < 1e-4


def test_calculate_intersections_cpu_safe_with_out_of_bounds_rays():
    """Out-of-bounds intersections must not crash the surface lookup.

    The ray exits the X bound almost immediately, so nearly every plane
    crossing is out of bounds; rejected rows become nan and ``nan.int()``
    is INT_MIN. The lookup must stay in bounds — regression for an
    IndexError on CPU / silent out-of-bounds read on CUDA.
    """
    shortest = _run(points_y1=[1.0, 1.0, 1.0],
                    slope_y1=[1.0, 5.0, 0.0],
                    surface_voxels=[],            # nothing to hit
                    size=(8, 6, 3))
    assert torch.all(shortest.isinf())            # ran cleanly, no hit
