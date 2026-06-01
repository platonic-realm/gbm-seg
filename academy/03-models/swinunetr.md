# Custom SwinUNETR

A from-scratch implementation of a 3D Swin-Transformer encoder paired
with a convolutional decoder, tuned for the **shallow, anisotropic
patches** of this dataset. This is **NOT MONAI's SwinUNETR** — it's a
custom implementation chosen so we can ablate specific knobs (window
sizes, position-bias mechanism, deep supervision aux heads) that
MONAI's variant doesn't expose as config options.

## Architecture at a glance

```
Input (B, 3, Z, H, W)
  → PatchEmbed3D                (halves XY, keeps Z, projects to feature_size)
  → Encoder stage 0             (BasicLayer3D + PatchMerging3D)
  → Encoder stage 1
  → Encoder stage 2
  → Encoder stage 3 (bottleneck — no further merge)
  → Decoder: PatchExpand3D + Conv fusion, mirroring the encoder
  → Final upsample + 1×1×1 head + softmax
Output (B, num_classes, Z, H, W)
```

The encoder is the *transformer* part; the decoder is **conv-based**
(`ConvTranspose3D` + `Conv3D` fusion blocks). This asymmetry is
deliberate — Swin's strength is the encoder's long-range modelling,
not decoding back to per-voxel labels.

Default config from `configs/template.yaml`:

```yaml
swin_unetr:
  feature_size: 24                  # embedding dim at stage 0
  depths: [2, 2, 2, 2]              # blocks per stage
  num_heads: [3, 6, 12, 24]         # attention heads per stage
  window_size_xy: 7                 # XY span of each attention window
  mlp_ratio: 4.0
  qkv_bias: true
  drop_rate: 0.0
  attn_drop_rate: 0.0
  z_deduction_per_stage: auto
  use_relative_pos_bias: true       # NEW (commit 03bacc5) — see below
```

## How the encoder works

Each encoder stage is a [BasicLayer3D](window-attention.md): a stack of
`depth[k]` paired regular/shifted Swin blocks, followed by an optional
`PatchMerging3D` downsample. The merge halves XY (kernel `(z_kernel,
2, 2)`, stride `(1, 2, 2)`) and reduces Z by `z_deduction` (typically 2).

So for the default Z=12 input, 4 stages, `z_deduction=2`:
- Stage 0: Z=12, XY = (H/2, W/2)
- Stage 1: Z=10, XY = (H/4, W/4)
- Stage 2: Z= 8, XY = (H/8, W/8)
- Stage 3: Z= 6, XY = (H/16, W/16)

## The crucial detail — window_z is always the FULL stage Z

Look at `src/models/swin_unetr/swin_unetr.py:146`:

```python
window = (stage_z[k], self.window_size_xy, self.window_size_xy)
```

The Z component of the attention window equals `stage_z[k]` — the
entire Z dimension at that stage. There is NO `window_size_z`
configuration knob (as of commit `03bacc5`), and at every stage the
attention sees the entire Z extent in one go.

Why this matters: it gives every encoder block direct access to the
full Z structure of the patch. The model can learn long-range Z
relationships in one attention layer rather than needing to compose
them through stacking. **This is exactly what enables the model to
memorise the 6-slice stacked-label periodicity** described in
[Z anisotropy and upsampling](../02-data/z-anisotropy-and-upsampling.md).

See [the Z-jaggedness diagnosis](../07-case-study-z-jaggedness/diagnosis.md)
for what this implies and what we can do about it.

## The `use_relative_pos_bias` ablation (commit 03bacc5)

Added in May 2026 as part of the [Z-jaggedness
intervention](../07-case-study-z-jaggedness/v2-v3-v4-experiments.md).
Setting `model.swin_unetr.use_relative_pos_bias: false` disables the
learned per-(Δz, Δh, Δw) bias inside every window attention. The
attention itself still runs — it just loses the explicit
position-specific scalar.

Details: see [window attention](window-attention.md), specifically the
"Killing the bias" section.

## Z-shift is hard-coded to 0

Standard Swin Transformers alternate **regular** and **shifted-window**
blocks (the shift breaks the boundary effect of partitioning windows).
In our 3D version, the shift is set to `(0, win_h//2, win_w//2)` —
**Z never shifts** — because `window_z == stage_z` so there's no
meaningful Z direction to shift in. See
`src/models/swin_unetr/blocks.py:417`.

If we ever cap `window_z` below `stage_z` (deferred future work), we'll
also enable Z shifting.

## Auto z_deduction — adapts to patch depth

`z_deduction_per_stage: auto` (default) computes:

```python
z_deduction = max(1, round(sample_z / (2 * (num_stages - 1))))
```

So a Z=12 patch over 4 stages → `z_deduction = 2`. A Z=24 patch (used
in some earlier debug runs) → `z_deduction = 4`. A Z=6 patch → 1.

The formula reduces Z by roughly half across all merges combined. This
keeps the bottleneck reasonable while preserving enough Z resolution
for the decoder to reconstruct.

`src/models/swin_unetr/swin_unetr.py:115-121`.

## Validation guard against too-shallow Z

`swin_unetr.py:136` checks `sample_z - (num_stages-1)*z_deduction > 0`.
If you try to build a 4-stage Swin on a Z=3 input with z_deduction=2,
it raises before allocating any tensors. Test in
`tests/models/test_swin_unetr.py:test_swin_unetr_rejects_too_shallow_z_for_depth`.

## Why custom and not MONAI

The project's design notes call out the rationale: a custom SwinUNETR
gives us per-knob control for the paper's ablations
(window-Z size, relative-position bias, deep supervision aux heads) —
choices MONAI's SwinUNETR doesn't expose as config knobs. The custom
implementation is documented inline (every constant has a comment), so
we can modify it confidently. The repo also commits to **local +
W&B-only logging** (no TensorBoard or SQLite back-ends) and the custom
implementation fits cleanly into that.

## What can go wrong at inference time

Two things bind the inference patch shape to the training patch shape:

1. **Window-Z assertion.** `window_partition` enforces `Z == win_z` at
   runtime (`blocks.py:82`). Inference at a different `Z` than the
   model was built for raises:
   ```
   ValueError: Window Z (12) must equal input Z (24); ...
   ```
2. **PatchMerging Z kernel.** `PatchMerging3D(z_kernel=z_deduction+1)`
   requires the input Z to be `≥ z_kernel`.

Together they enforce **inference patch Z must equal training patch Z**.
See [inference patch must match train](../07-case-study-z-jaggedness/diagnosis.md)
in the case study for the parallel constraint on U-Net (LayerNorm
shape baked at training time, slightly different mechanism but same
practical effect).

## Memory wall

The window-attention scores tensor `(B*nW, heads, N, N)` is the
dominant memory cost. At our `[12, 256, 256]` patch with batch=8 on a
40 GB A100, it OOMs (~11 GiB just for the scores tensor).
**Per-rank batch cap is 4** on 40 GB A100s (`maybe 6 with care`); 80 GB
A100s fit batch=8 cleanly.

Empirically established during the May 2026 training campaign — the
40 GB A100 OOM at batch 8 is reliably reproducible with the v2 swin
config; batch 4 trains cleanly with a comfortable ~15 GiB margin.

## Tests

`tests/models/test_swin_unetr.py` has 33 tests (after the `03bacc5`
commit added 5 for the position-bias flag). Key ones:

- `test_swin_unetr_forward_output_matches_input_shape` — the shape
  contract.
- `test_window_partition_rejects_mismatched_z` — the
  `Z == win_z` guard.
- `test_swin_unetr_auto_z_deduction_scales_with_patch_depth` — the
  auto formula.
- `test_window_attention_sdpa_matches_reference_no_mask` /
  `…_with_mask` — the SDPA fast-path is numerically equivalent to the
  reference plain-softmax implementation.
- `test_window_attention_no_bias_*` — the bias-flag ablation.
