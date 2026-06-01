# Inductive biases — conv vs attention

The most useful thing to internalise about these two architectures is
that they have **different assumptions about how the world is
structured**, and those assumptions decide what they can learn easily
versus what they have to work hard to learn.

This isn't an architectural-comparison cheat sheet; it's specifically
the lens through which the [Z-jaggedness
problem](../07-case-study-z-jaggedness/README.md) becomes legible.

## What an "inductive bias" is

An inductive bias is a built-in assumption that biases what a model
prefers to learn. Two examples that already shape this project:

- **Conv layers are translation-equivariant.** Applying the same 3×3×3
  kernel at every voxel position bakes in the assumption that "the
  same pattern at a different location means the same thing". You
  can't easily learn "the membrane lives at voxel (z=10, y=5, x=80)"
  with a conv layer — only "the membrane looks like X locally,
  wherever X appears".
- **Attention is permutation-equivariant by default.** The output of
  `attention(Q, K, V)` is unchanged if you permute the tokens. Spatial
  structure has to be *added back in* via position embeddings or biases
  (Swin uses the [relative position bias](window-attention.md)).

These properties aren't accidents — they're chosen by the architect to
match the data's structure. Mismatch the inductive bias to the data
and you get suboptimal models. **Match it perfectly** and you get
remarkably efficient learning.

## The two architectures on this dataset

| Property | [U-Net (conv)](unet3d.md) | [SwinUNETR (attention)](swinunetr.md) |
|---|---|---|
| Receptive field per layer | 3×3×3 (small, local) | full window (e.g. 12×7×7 = 588 tokens at stage 0) |
| Spatial weight sharing | Yes, all 3 axes | No — position-specific via bias table |
| Position information | Implicit in the conv structure | Explicit via learned `relative_position_bias_table` |
| Translation equivariance in Z | Yes (by construction) | No — bias table has Z component |

## What this means for our specific data

Recall from [Z anisotropy and label
upsampling](../02-data/z-anisotropy-and-upsampling.md): the training
target has a 6-slice **stacked-label** structure in Z. The model's job
is to predict per-voxel labels. There are two strategies:

1. **Predict broadly smoothly in Z**, and accept some loss at the
   sharp boundaries between stacking plateaus. This is what the
   conv U-Net does — it has no machinery to do anything else.

2. **Memorise the period-6 stacking and reproduce it**. The model
   learns "5 identical labels, sharp transition, 5 identical labels,
   …". This is only possible if the model has a way to *encode
   absolute-or-relative Z position* — which Swin does via its
   relative position bias.

The first strategy gives smooth output. The second gives slightly
better training-set Dice (because the prediction matches the target's
sharp boundaries) but produces visibly jagged segmentation in the
rendered TIFFs/GIFs/MP4s.

## Why rotation augmentation magnifies the gap

The interaction with [online rotation
augmentation](../02-data/augmentation.md) is the subtle bit. Rotation
strips **absolute-XY** as a learnable cue (because every patch is
rotated by a different angle, "the membrane is at fixed `(x, y)`" no
longer holds). Both models lose that cue. What's left:

- The **U-Net** has only **Z context** (local 3×3×3) as a fallback.
  Z context naturally pulls predictions toward smoothness because the
  physical membrane is a continuous surface — adjacent Z slices
  contain the same membrane shifted slightly.

- The **SwinUNETR** has two fallbacks:
  1. **Relative XY within the window** — window_xy=7 spans a large
     fraction of a 256-wide patch, so the attention can still
     "see" the membrane's full local layout at each Z slice.
  2. **Relative Z within the window** — same mechanism, but for the Z
     dimension.

  When XY information is enough to identify the membrane per Z slice
  (which it is, because the membrane is well-resolved in XY), the Swin
  picks the easy path: predict each Z slice's mask from its own XY
  content. **No Z context required, no Z smoothness emerges**.

That's the punch line. **Rotation aug is a red herring on its own** —
it's the *combination* of rotation aug + the architecture's available
fallbacks that decides Z smoothness.

The [Z-jaggedness diagnosis](../07-case-study-z-jaggedness/diagnosis.md)
spells this out in more detail, with the experimental sequence that led
to the current v4 ablations (`use_relative_pos_bias=False` to kill the
bias mechanism; `window_size_xy=3` to starve the XY fallback).

## The deeper question (for the paper)

If we want **smooth Z output AND high Dice**, the question is whether
the conv inductive bias is doing something irreplaceable, or whether
attention can match it given the right ablations. The v4 experiments
are designed to answer that.

If the answer is "no, even ablated SwinUNETR can't match U-Net Z
smoothness", the next move is **soft labels** or the **mesh-based
relabelling** discussed in the
[future directions](../07-case-study-z-jaggedness/future-directions.md) —
both of which change the *target* the model learns, removing the
period-6 artifact at source.

## A useful frame for further design choices

When tweaking any model architecture decision, ask:

1. What's the inductive bias?
2. What does the data structure favour?
3. Where do the two not match, and which side will give?

For our data the answers are usually:

- "Spatial continuity in 3D" (conv-friendly).
- "Long-range XY structure of a curved 2D surface" (attention-friendly).
- "Discrete period in Z introduced by `np.repeat` upsampling" (attention-can-memorise,
  conv-can't).

The right architecture is whichever bias matches the *real* structure
you want to learn — not whichever can fit the training target's
*artifacts*.
