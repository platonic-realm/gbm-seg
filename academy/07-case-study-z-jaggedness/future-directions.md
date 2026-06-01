# Future directions

If the [v4 ablations](v2-v3-v4-experiments.md) don't fully smooth
SwinUNETR's Z output, here's what's on the table next. Ranked by
estimated effort.

## Cheap and architectural

### Cap `window_size_z` independently of `stage_z`

Currently `window_z = stage_z` at every encoder stage (see
[swinunetr](../03-models/swinunetr.md)). This is what gives the attention
direct access to the full 6-slice period in one window.

The intervention: add a `window_size_z` config knob that **caps** the
Z window at some smaller value (e.g. 4 or 6, less than the 6-slice
period). Then:

- The attention sees less than one full period in any single window.
- Memorising the period now requires composing information across
  multiple shifted-window passes — a much weaker inductive bias.

What it takes to implement: this is **not just a config change**. The
current `window_partition` function in
`src/models/swin_unetr/blocks.py:82` enforces `Z == win_z` via a hard
assertion. To cap `window_size_z`, we'd need to:

1. Remove that assertion.
2. Actually partition Z into multiple windows (with padding if Z isn't
   a multiple of win_z).
3. Add Z **shift** for alternating blocks (currently `shift_z = 0` is
   hard-coded in `BasicLayer3D:417` because the full-Z window made
   shifting meaningless).
4. Update `window_reverse` to undo the new Z partitioning.
5. The `relative_position_bias_table` already scales correctly with
   any `win_z` — no change needed there.

Maybe 50-100 LOC plus tests. Deferred from the v4 plan because it
hadn't been tried yet at the time and the simpler ablations were
worth exhausting first.

### Reduce model capacity

If `use_relative_pos_bias=False` cripples the model too broadly,
intermediate options:

- `attn_drop_rate: 0.1` — apply dropout to attention scores.
  Discourages sharp, memorisation-style attention patterns. Soft
  regulariser. Config-only.
- `drop_rate: 0.1` — apply dropout to MLP / projection outputs.
  Generic regulariser. Config-only.
- Smaller `feature_size` (24 → 16) or `depths` ([2,2,2,2] → [1,1,1,1])
  — reduces overall capacity. Risk: hurts Dice broadly.

These don't directly target the jaggedness mechanism, but they
sometimes help with memorisation-style failures.

## Medium-effort — data side

### Distance-transform-based labels (SDF)

**Idea**: don't represent the GBM as a binary mask. Represent it as a
**signed distance function** — for each voxel, store the distance to
the nearest membrane boundary, signed (positive inside the membrane,
negative outside).

**Upsampling**: SDFs are smooth by construction. Trilinear-
interpolating an SDF along Z gives a smooth, monotone interpolation.
**No staircase**.

**Re-binarising**: at training time, threshold the SDF at 0 to get
back to a binary mask, OR train the loss on the SDF directly.

Implementation sketch:

```python
import scipy.ndimage as ndi

def label_to_sdf(binary_label):
    """Signed Euclidean distance: positive inside membrane, negative outside."""
    inside = ndi.distance_transform_edt(binary_label)
    outside = ndi.distance_transform_edt(1 - binary_label)
    return inside - outside

# Z-upsample the SDF smoothly, then threshold:
sdf_upsampled = F.interpolate(sdf, scale_factor=(z_scale, 1, 1), mode='trilinear')
label_upsampled = (sdf_upsampled >= 0).float()
```

The thresholded result has **smooth transitions in Z** (the membrane
boundary moves continuously) rather than stair-step plateaus.

Cost: code change in `_z_interpolate_and_stack` in
`src/utils/misc.py`. Plus we'd want to verify that the validation-mask
logic in `src/train/factory.py:372-379` still makes sense (probably
disable it — every Z slice is now a meaningful label).

Risk: the SDF is computed on the source (Z=6 or so) binary mask, so
its resolution is bounded by that. Thin parts of the membrane that
were 1 voxel thick in the source data still get interpolated at
sub-voxel precision, which may or may not match the true biology.

### Soft (probability) labels

Same idea but more general: train against `[0, 1]` probability targets
instead of `{0, 1}` binary. Smooth-in-Z interpolation gives
soft labels directly. Loss must accept probability targets — that
means changing from `CrossEntropyLoss` (which expects class indices)
to a soft-target equivalent (KL-div, soft-CE, BCE, or Dice loss
which already supports soft targets).

The `Cont` loss can be adapted; the structure stays the same.

Conceptually clean but requires more loss / metric plumbing changes
than the SDF route.

## Big-effort — different formulation

### Mesh-based label generation

The user suggested: convert each labelled volume to a 3D mesh via
marching cubes, scale the mesh 6× in Z, and re-voxelise at the
upsampled grid.

**Pros**: respects the GBM's true topology (a 2D surface in 3D).
Transitions land at geometrically correct positions rather than every
6th slice.

**Cons**: marching cubes adds its own staircase artifacts when the
input is coarse (the source data is Z=13 — coarse). Per-volume mesh
ops are slower than tensor ops. Choosing the threshold for the
re-voxelisation has its own bias.

Probably more effort than the SDF route, with similar end behaviour.
Worth trying if SDF doesn't work because the mesh approach preserves
*topology* explicitly whereas SDF only preserves *geometry*.

### Use a different architecture

Convolution-only architectures (U-Net, V-Net, even Swin-without-attention)
don't have this problem. If SwinUNETR can't be coaxed into smooth Z
output, the pragmatic answer for the paper might be: **use U-Net for
the final morphometry, SwinUNETR for the Dice headline**. The two
architectures have different strengths.

A more elegant variant: a hybrid model where the **encoder** is Swin
(for its long-range modelling power) but the **decoder** includes
heavy Z smoothing — explicit Gaussian Z-pooling at the final layer, or
a TV-regularised refinement step.

## Better metrics

Independent of the architecture changes, we should add quantitative
**smoothness metrics** to the stats stage so we don't have to rely on
visual inspection:

- **Z-difference variance**: `(mask[1:] - mask[:-1]).var()` over the
  full volume. A staircase has high variance at every 6th Z slice
  boundary; a smooth output has low variance everywhere.
- **Z-frequency spectrum**: FFT along Z of the mask, look for peaks
  at the period-6 frequency.

Adding either of these as a per-volume metric in `gbm.py stats` would
make the v4 comparison quantitative rather than qualitative.

## A concise next-move ranking

1. **Wait for v4 ablations** (in progress). They're the simplest test.
2. If neither works: **add quantitative smoothness metrics** to stats.
   Then we can actually measure jaggedness.
3. Then **window_size_z capping** (next-best architectural lever).
4. Then **SDF labels** (data-side fix that addresses the root
   target structure).
5. If all of the above fail: **mesh-based labelling** or **hybrid
   architecture**.

The Academy doc on this case study will be updated as the results
come in — this file currently represents the state at the end of the
2026-05-30 session.
