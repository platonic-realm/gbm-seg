# 3D U-Net

A faithful 3D implementation of the original U-Net (Ronneberger et al.,
2015) generalised to volumetric input. Used as the convolutional
baseline in head-to-head comparisons against [SwinUNETR](swinunetr.md).

## Architecture in one paragraph

A symmetric encoder/decoder. The encoder is a stack of "encoder blocks",
each one a pair of `Conv3D(3×3×3) → InstanceNorm → ReLU` operations
followed by a `MaxPool3D` downsample. The decoder mirrors it:
`ConvTranspose3D` upsample, concatenate with the matching encoder
output (the "skip connection"), then two more conv blocks. The final
1×1×1 conv produces per-class logits.

Config controls in `configs.trainer.model.unet_3d`:

```yaml
unet_3d:
  feature_maps: [64, 128, 256, 512]   # channels at each stage
  encoder_kernel: [3, 3, 3]
  encoder_padding: same
  decoder_kernel: [3, 3, 3]
  decoder_padding: same
  z_deduction_per_stage: auto         # downsample Z by 2 per stage
```

The `z_deduction_per_stage` knob lets us **break the cube assumption
when the input Z is small**. With the default Z=12 input patch and
4 stages, naïve `MaxPool3D(2,2,2)` would yield Z dimensions
12 → 6 → 3 → 1.5 (illegal). The "auto" setting computes a Z stride
that keeps the bottleneck Z ≥ 1 — usually a Z stride of 1 (no Z
downsampling) at the deepest stages.

## What makes the U-Net's behaviour distinctive

Two architectural facts shape almost everything about how this model
behaves on the GBM data:

1. **Conv weight sharing in all three spatial axes.** Every 3×3×3
   kernel applies the same weights at every voxel position. This is
   the standard convolution inductive bias — the function the model
   computes at voxel `(z, y, x)` is identical to what it computes at
   `(z+1, y, x)`, modulo whatever fits in the local 3-voxel
   neighbourhood. **There is no mechanism for "treat position Z=k
   specially" — it would have to be learned in a translation-equivariant
   way, which is harder.**

2. **Small per-layer receptive field.** Each conv layer sees a 3×3×3
   neighbourhood; the receptive field grows hierarchically with depth.
   Even at the bottleneck stage, the receptive field is limited
   compared with the full-attention windows in SwinUNETR.

These two together mean: **the U-Net produces broadly smooth output in
Z**, even when the training target has the stacked-label artifact (see
[Z anisotropy and upsampling](../02-data/z-anisotropy-and-upsampling.md)).
The model lacks the mechanism to memorise "labels live at Z mod 6 == 0",
so it falls back on the only thing it has — translation-equivariant
local-receptive-field predictions — and that gives smooth Z.

See [inductive biases](inductive-biases.md) for the conv-vs-attention
comparison spelled out, and the [Z-jaggedness case
study](../07-case-study-z-jaggedness/README.md) for what this implies
in practice.

## Gradient checkpointing

`src/models/unet3d/unet3d.py:67-74` documents the gradient checkpointing
hook. When enabled (`runtime.gradient_checkpointing: on` or `auto` on
40 GB GPUs), each encoder/decoder block's forward activations are NOT
saved for backward — they're recomputed during backprop. Trades ~30%
more compute for ~50% less memory.

This is what lets us train the U-Net at `[24, 256, 256]` patches on
32 GB V100s; without it we'd OOM. The flag is plumbed via the runtime
config, defaults to `'auto'` which decides based on `torch.cuda.get_device_properties(0).total_memory < 60 GiB`.

## Tests

`tests/models/test_unet3d.py` has the regression suite (≈88% line
coverage). It locks down:

- Forward shape contract (output shape == input spatial shape).
- The `z_deduction` Z trajectory (12 → ... → bottleneck > 0).
- Deep-supervision aux-head outputs.
- Gradient flow end-to-end.
- Compatibility with the post-A3 `forward(x) -> (logits, outputs)`
  interface.

## When to pick the U-Net vs SwinUNETR

This project doesn't choose; it compares. The expert-comparison
numbers as of late May 2026 show SwinUNETR ahead on Dice (~0.71 vs
~0.65 model-vs-Robin), but the U-Net produces visibly cleaner Z output
in the rendered TIFFs/GIFs/MP4s (see [Z-jaggedness
saga](../07-case-study-z-jaggedness/v2-v3-v4-experiments.md)). Which
one "wins" depends on whether you care about Dice (sharper boundaries
in 3D) or downstream morphometry (smooth Z surface).
