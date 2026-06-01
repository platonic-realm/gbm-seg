# Diagnosis — why SwinUNETR memorises the stacked-label period

The Z-jaggedness in SwinUNETR's output comes from the **interaction**
of three system-level facts. Each on its own is fine. Together they
produce the staircase.

## Fact 1: training labels have a 6-slice periodic structure

From [Z anisotropy and label upsampling](../02-data/z-anisotropy-and-upsampling.md):
`gbm.py create` upsamples the training labels by **`np.repeat`** —
every original Z slice is duplicated 6 times in a row. The intensity
channels are trilinearly interpolated (smooth), the labels are
stacked (every 6th slice is a transition; the 5 in between are
identical).

If you scroll Z on the label channel of an upsampled training TIFF
you see flat plateaus of 6 identical slices separated by sharp
transitions. **The training target has a period-6 structure.**

## Fact 2: SwinUNETR has explicit machinery to encode position-specific Z patterns

From [window attention](../03-models/window-attention.md): SwinUNETR's
attention has a **learned relative position bias** indexed by
`(Δz, Δh, Δw)`. The Z component has shape `2*win_z - 1` per head.

At stage 0 with `window_size = (12, 7, 7)` (12 = full input Z), the
Z bias has **23 entries per head per attention layer**. The model can
learn separate bias values for "tokens at offset Δz=6 attend to each
other this much" vs "tokens at Δz=3 attend to each other that much".

This is the mechanism by which the model can encode "labels jump every
6 Z positions" as a learned attention pattern. **No conv layer has
this property** — every conv kernel applies the same weights at every
voxel position by construction.

## Fact 3: full-Z attention windows let the model see the entire period

From [SwinUNETR](../03-models/swinunetr.md): at every encoder stage,
`window_z == stage_z` (the full Z dimension of the input at that
stage). At stage 0 that's the full input Z=12; at the bottleneck
it's stage_z (typically 6).

The first encoder block's attention sees **all 12 Z positions** in
one window. Combined with the relative position bias from fact 2, it
can directly encode "the label transitions are at Z=0, 6, 12, ..."
without needing to compose this information through stacking.

## What U-Net can't do (which Swin can)

U-Net has small 3×3×3 conv kernels with weight sharing across all
three spatial axes. The function it computes at Z=k is the same as
what it computes at Z=k+1 (modulo what fits in 3 Z voxels). It has
**no mechanism** to learn "Z=6 is special". It must predict in a
translation-equivariant way — which means broadly smooth Z, by
construction.

If U-Net wanted to reproduce the stacked-label pattern at inference,
it would need translation-equivariant detectors for period-6 patterns.
That's *possible* (you can build a periodic-pattern detector from
translation-equivariant conv stacks), but **the conv inductive bias
makes it harder to learn than smooth prediction**, and the model
defaults to the easier option.

## Why this isn't a Dice problem

Dice doesn't penalise "Z step pattern vs smooth Z". Both predictions
can produce roughly the same Dice because:

- The **rendered TIFFs** show the staircase but Dice is computed
  per-voxel against the (also-stacked) training target. SwinUNETR's
  per-voxel agreement is sharp where the target's transitions are
  sharp; it loses nothing by being staircased.
- For expert comparison, the labels are at **native** Z; we
  [subsample the prediction](../06-inference/expert-comparison.md)
  to the label's Z grid. Subsampling effectively *picks one slice
  per stair-step* — the staircase is invisible at the comparison
  resolution.

So SwinUNETR's Dice numbers look good. The staircase only shows up
when you view the full-resolution prediction in a 3D viewer.

## The role of rotation augmentation — a subtle red herring

A common (and wrong) framing: "Rotation augmentation makes U-Net's
output smooth because the model learns rotation-invariant features."
That's part of the story, but not the whole story. Rotation aug
**strips absolute-XY as a learnable cue** — every patch is rotated by
a different angle, so the model can't lean on "the GBM is at fixed
`(x, y)`".

Without absolute-XY, the model has to fall back on something else:

- **U-Net** falls back on **Z context** — its receptive field for
  predicting voxel `(z, y, x)` includes neighbouring Z slices, and
  those slices contain the same membrane shifted slightly. The conv's
  translation-equivariance in Z then produces smooth output.
- **SwinUNETR** has **two fallbacks**:
  1. Relative XY within the window — `window_size_xy=7` over a 256-
     wide patch spans most of the field of view; the model can identify
     the membrane from the rotated XY pattern at each Z slice.
  2. Relative Z within the window — but per fact 1, the target has a
     period-6 pattern, so this fallback also leads to memorising the
     period.

The Swin **prefers** the XY fallback because identifying the membrane
from XY is easier than reasoning about Z context. So even after
rotation aug forces the model to drop absolute-XY, the Swin can still
predict each Z slice from its own (rotated) XY content
**independently**. **No Z context required, no Z smoothness emerges,
and the period-6 memorisation kicks in for whatever cross-Z signal
the bias table picks up.**

This is the deeper diagnosis. The interventions in
[v4](v2-v3-v4-experiments.md) target the two fallbacks:

- **`use_relative_pos_bias=False`** kills the mechanism that
  encodes the period-6 pattern (fact 2). The model can still attend
  across the window — it just loses the position-specific scalar.
- **`window_size_xy=3`** starves the XY fallback (cuts the per-Z-slice
  XY context by ~5×), forcing the model to lean on Z context like
  U-Net does.

## The inference-patch-must-match-train constraint (related)

While we're on Swin internals: both U-Net and SwinUNETR bake their
training patch shape into the model in different ways, and **inference
must use the same patch geometry as training**.

- **U-Net3D**: `LayerNorm.normalized_shape = [C, Z, H, W]` is set at
  init time. Inference at a different shape raises:
  ```
  RuntimeError: Given normalized_shape=[3, 12, 256, 256],
                expected input with shape [*, 3, 12, 256, 256],
                but got input of size [8, 3, 24, 128, 128]
  ```
- **SwinUNETR**: `window_partition` asserts `Z == win_z`. Different
  inference Z raises:
  ```
  ValueError: Window Z (12) must equal input Z (24); ...
  ```

Memorialised in the project memory `reference_inference_patch_must_match_train.md`
and the v4 ablation plan. The practical rule: **use the same
`sample_dimension` at infer as at train**. Stride and overlap
density can change freely; the patch size can't.

## Why not just trilinear-interpolate the labels?

That sounds like the obvious fix. We tried it as a thought experiment
and proved it doesn't work — see [Z anisotropy and
upsampling](../02-data/z-anisotropy-and-upsampling.md) and the demo
script `src/scripts/compare_label_upsampling.py`. Trilinear
interpolation of binary `{0, 255}` labels followed by thresholding at
half is mathematically equivalent to `np.repeat` (nearest-neighbour).
The transitions move to different Z positions if you shift the
threshold, but they're still stair-steps.

The only data-side fix is **soft labels** (probability targets in
`[0, 1]`, which requires changing the loss to accept probabilities)
or **signed-distance-transform labels** (compute SDF on the binary
mask, trilinear-interpolate the SDF, threshold at 0). Both are
discussed in [future directions](future-directions.md).
