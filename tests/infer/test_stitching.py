"""Sliding-window stitching regression. Locks in §A3.

Synthetic two-patch overlap case: two patches with distinct logits land
at adjacent offsets that share a region. Each stitching mode is tested
for its expected behaviour in the overlap region.

Offset convention follows ``src/data/ds_infer.py``:
``offsets[b]`` is a 4-vector where indices ``[1], [2], [3]`` are
``x_start, y_start, z_start``.

Patches in these tests are ``[4, 4, 4]`` (Z, X, Y) rather than the
real-world ``[12, 256, 256]``. Z=4 keeps tests fast while staying
above the Hann-window-degeneracy threshold (a length-2 non-periodic
Hann window is identically zero).
"""

import pytest
import torch

from src.infer.stitching import VALID_MODES, StitchAccumulator


def _patch(value: float, num_classes: int = 2, Z: int = 4, X: int = 4, Y: int = 4):
    """Build a constant-logits patch where class 1 dominates by ``value``."""
    logits = torch.zeros(num_classes, Z, X, Y)
    logits[1] = value
    return logits


def _two_patch_offsets():
    """Two patches at (z=0,x=0,y=0) and (z=0,x=2,y=0); each 4x4x4."""
    return torch.tensor([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
    ], dtype=torch.long)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown stitching mode"):
        StitchAccumulator('bogus', patch_size=[4, 4, 4], result_shape=[2, 4, 6, 4])


def test_rejects_wrong_dim_shapes():
    with pytest.raises(ValueError):
        StitchAccumulator('gaussian', patch_size=[4, 4], result_shape=[2, 4, 6, 4])
    with pytest.raises(ValueError):
        StitchAccumulator('gaussian', patch_size=[4, 4, 4], result_shape=[2, 4, 6])


@pytest.mark.parametrize("mode", VALID_MODES)
def test_single_patch_finalize_runs(mode):
    """Single non-overlapping patch: every mode produces a finite result."""
    acc = StitchAccumulator(mode, patch_size=[4, 4, 4],
                            result_shape=[2, 4, 4, 4])
    logits = _patch(value=5.0).unsqueeze(0)
    offsets = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    acc.add_batch(logits, offsets)
    result = acc.finalize()
    assert result.shape == (4, 4, 4)
    # Class 1 dominates in the centre. (Edges may differ for Hann since the
    # window is ≈0 there and the weight clamp picks class 0 by argmax tie.)
    assert result[2, 2, 2].item() == 1


def _opposed_patches():
    p0 = torch.zeros(2, 4, 4, 4)
    p0[0] = 10.0  # class-0 vote
    p1 = torch.zeros(2, 4, 4, 4)
    p1[1] = 10.0  # class-1 vote
    return torch.stack([p0, p1], dim=0)


def test_sum_logits_no_division():
    """In legacy ``sum_logits`` mode, the numerator is the raw sum across
    all overlapping patches; no division by count."""
    acc = StitchAccumulator('sum_logits', patch_size=[4, 4, 4],
                            result_shape=[2, 4, 6, 4])
    acc.add_batch(_opposed_patches(), _two_patch_offsets())
    result = acc.finalize()
    # Non-overlap regions: each patch picks its own class outright.
    assert torch.all(result[:, 0:2, :] == 0), "patch 0 region should pick class 0"
    assert torch.all(result[:, 4:6, :] == 1), "patch 1 region should pick class 1"


def test_gaussian_overlap_is_weighted_average():
    """In ``gaussian`` mode, voxels near a patch's centre are dominated by
    that patch's softmax vote; the overlap region transitions smoothly."""
    acc = StitchAccumulator('gaussian', patch_size=[4, 4, 4],
                            result_shape=[2, 4, 6, 4])
    acc.add_batch(_opposed_patches(), _two_patch_offsets())
    result = acc.finalize()
    # Patch 0 centre is ~x=1.5; patch 1 centre is ~x=3.5. So the side closest
    # to patch 0's centre picks class 0; the side closest to patch 1's centre
    # picks class 1.
    assert result[1, 0, 0].item() == 0, "near patch-0 centre → class 0"
    assert result[1, 5, 0].item() == 1, "near patch-1 centre → class 1"


def test_flat_softmax_overlap_is_equal_average():
    """In ``flat_softmax``, both patches contribute equally in overlap voxels.
    Equal opposing softmax votes → argmax tie → class 0 wins."""
    acc = StitchAccumulator('flat_softmax', patch_size=[4, 4, 4],
                            result_shape=[2, 4, 6, 4])
    acc.add_batch(_opposed_patches(), _two_patch_offsets())
    result = acc.finalize()
    # Non-overlap: each patch dominates its region.
    assert torch.all(result[:, 0:2, :] == 0)
    assert torch.all(result[:, 4:6, :] == 1)
    # Overlap (x=2:4): equal votes → argmax tie → class 0.
    assert torch.all(result[:, 2:4, :] == 0)


def test_hann_window_peaks_at_centre():
    """Hann window must be zero at edges and positive in the interior.
    For a 4x4x4 separable non-periodic Hann the 1D peak is 0.75
    (at n=1 and n=2), so the 3D centre is 0.75**3 ≈ 0.42."""
    acc = StitchAccumulator('hann', patch_size=[4, 4, 4],
                            result_shape=[2, 4, 4, 4])
    # Edges of a length-4 non-periodic Hann are exactly 0.
    assert acc.window[0, 0, 0].item() == 0.0
    # Interior voxels are strictly positive; the 3D peak is at the four
    # nearest-to-centre voxels of an even-sized cube.
    assert 0.4 < acc.window[1, 1, 1].item() < 0.5
    # And the centre exceeds any edge voxel.
    assert acc.window[1, 1, 1].item() > acc.window[0, 1, 1].item()


def test_gaussian_window_peaks_at_centre():
    """Gaussian window must peak at the centre (normalised to 1)."""
    acc = StitchAccumulator('gaussian', patch_size=[4, 4, 4],
                            result_shape=[2, 4, 4, 4])
    # Centre voxels are near-1; corner voxels are smaller.
    centre = acc.window[1, 1, 1].item()
    corner = acc.window[0, 0, 0].item()
    assert centre > corner
    assert acc.window.max().item() == 1.0  # normalised


def test_weight_zero_voxels_default_to_class_zero():
    """If a voxel was never touched (weight=0), finalize must not NaN —
    the eps-clamp guard ensures the division stays finite, and the resulting
    all-zero numerator argmaxes to class 0."""
    # result_shape Z=8 but patch Z=4 → z=4:8 stays untouched.
    acc = StitchAccumulator('gaussian', patch_size=[4, 4, 4],
                            result_shape=[2, 8, 4, 4])
    p0 = _patch(value=5.0).unsqueeze(0)
    offsets = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    acc.add_batch(p0, offsets)
    result = acc.finalize()
    assert torch.isfinite(result).all()
    # Untouched Z slice (z=4:8): numerator is 0 → argmax picks class 0.
    assert torch.all(result[4:, :, :] == 0)
