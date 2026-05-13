"""PSF-clamp diagnostic from `Morph.calculate_thickness_correction`.

When the measured thickness falls below the PSF resolution floor
(`measured² < PSF_term`), the quadrature-subtraction correction would
require a square root of a negative number. The code clamps the
difference to 0 (making the GBM "invisible" at those voxels). This
clamp must be (a) accurate — only fires for surface voxels where the
inequality actually holds — and (b) reported through `clamp_info` so
downstream stats can surface it.

Locks in §B3.2 from the DL-review backlog.
"""

import math

import torch

from src.infer.morph import Morph, PSFAxial, PSFLateral


def _build_inputs(z=4, h=8, w=8, measured_nm=100.0, slope_z=1.0):
    """Build slope/surface/distance tensors of matching shape.

    `slope_tensor` has shape `(Z, H, W, 3)` after the squeeze in
    `forward()`; we synthesise it here directly. All voxels are surface
    voxels by default so the clamp count and surface count are easy
    to predict.
    """
    surface_mask = torch.ones(z, h, w)
    distance_tensor = torch.full((z, h, w), float(measured_nm))
    # Slope vector pointing along Z (unit) → cos(beta)=1 → sin(alpha)=cos(beta)=1
    # and cos(alpha)=sin(beta)=0; this drives the PSF term toward PSFAxial².
    slope = torch.zeros(z, h, w, 3)
    slope[..., 2] = float(slope_z)
    return slope, surface_mask, distance_tensor


def _make_morph():
    # We don't run the full forward(); we only invoke calculate_thickness_correction.
    return Morph(_device="cpu", _ave_kernel_size=3)


def test_clamp_fires_when_thickness_below_axial_psf():
    """Measured 100 nm with slope along Z → effective PSF term ≈ PSFAxial² = 434² nm² >> 100²."""
    morph = _make_morph()
    slope, mask, dist = _build_inputs(measured_nm=100.0, slope_z=1.0)
    corrected, info = morph.calculate_thickness_correction(slope, mask, dist)
    # Every surface voxel should clamp (corrected = 0 everywhere).
    assert info['surface_count'] == slope.shape[0] * slope.shape[1] * slope.shape[2]
    assert info['clamp_count'] == info['surface_count']
    assert math.isclose(info['clamp_percentage'], 100.0)
    assert torch.all(corrected == 0)


def test_clamp_quiet_when_thickness_above_psf():
    """Measured 1500 nm with slope along Z → 1500² > 434², no clamping."""
    morph = _make_morph()
    slope, mask, dist = _build_inputs(measured_nm=1500.0, slope_z=1.0)
    corrected, info = morph.calculate_thickness_correction(slope, mask, dist)
    assert info['clamp_count'] == 0
    assert info['clamp_percentage'] == 0.0
    # corrected = sqrt(1500² - 434²) ≈ 1435.84
    expected = math.sqrt(1500.0**2 - PSFAxial**2)
    assert torch.allclose(corrected, torch.full_like(corrected, expected), atol=1e-3)


def test_clamp_ignores_non_surface_voxels():
    """Non-surface voxels (mask=0) must not contribute to the clamp count.

    Even if their measured² < psf_term, they are not surface and should
    not inflate the diagnostic.
    """
    morph = _make_morph()
    slope, mask, dist = _build_inputs(measured_nm=100.0, slope_z=1.0)
    # Zero out half the surface mask
    mask[:, :, :4] = 0
    corrected, info = morph.calculate_thickness_correction(slope, mask, dist)
    expected_surface = (mask > 0).sum().item()
    assert info['surface_count'] == expected_surface
    assert info['clamp_count'] == expected_surface  # all remaining surface voxels clamp


def test_clamp_info_records_psf_constants():
    """The diagnostic should carry the PSF constants for traceability."""
    morph = _make_morph()
    slope, mask, dist = _build_inputs()
    _, info = morph.calculate_thickness_correction(slope, mask, dist)
    assert info['psf_lateral_nm'] == PSFLateral
    assert info['psf_axial_nm'] == PSFAxial


def test_zero_surface_voxels_returns_zero_percentage():
    """Empty surface (all-zero mask) must not divide by zero."""
    morph = _make_morph()
    slope, mask, dist = _build_inputs(measured_nm=100.0)
    mask = torch.zeros_like(mask)
    _, info = morph.calculate_thickness_correction(slope, mask, dist)
    assert info['surface_count'] == 0
    assert info['clamp_count'] == 0
    assert info['clamp_percentage'] == 0.0
