# Experiments v2 → v3 → v4

A timeline of what we tried to fix the [Z-jaggedness](README.md), in
order. Each experiment isolates a single hypothesis.

The naming convention is `swin_offline_a100_<version>[_variant]` —
historical at this point (the `_a100` suffix is from when only A100s
were considered, even though some have run on V100s).

## v2 (baseline) — `swin_offline_a100_v2`

**Settings (all-data)**:
- Train stride: `[3, 128, 128]` (75% Z overlap, 50% XY overlap)
- Offline aug methods: `[zoom 0.7, zoom 1.5, twist_clock]`
- Online rotate: 0.4 probability, ±45°
- Effective batch: 32 (4 GPUs × per-rank 8 on V100s); later runs at 48
  on A100s
- Patch: `[12, 256, 256]`

**Result**:
- Training Dice peaked at ~0.74 (snapshot `001-11500.pt`).
- Expert comparison (model vs Chris/David/Robin): 0.702 / 0.617 /
  0.740.
- Inside the inter-rater band on Chris and Robin.
- **Rendered output: visibly jagged Z transitions**. 6-slice staircase
  pattern matching the stacked-label periodicity from
  [Z anisotropy](../02-data/z-anisotropy-and-upsampling.md).

This was the first time the jaggedness was clearly diagnosed. The user
flagged it during a render review.

## A side test — dense Z stride at inference

Before resorting to retraining, we tested whether inference-time
**dense Z stride** could TTA-average the artifact away.

The hypothesis: the model produces predictions with period-6
transitions at specific Z phases. If we re-run inference with stride
`[1, 128, 128]` (each output voxel hit by 12 patches at different Z
phases), maybe the phase-cancelling averaging smooths the artifact.

**Result**: ~1 pp Dice drop on every annotator (expected — averaging
across phases costs some sharpness), but **only modest visual
smoothing**. The artifact comes from the model's intrinsic behaviour
inside each patch, not from the patch boundaries — averaging across
patch phases can't fully erase what each patch's prediction looks
like internally.

## v3 — `swin_offline_a100_v3`, train stride `[2, 128, 128]`

**Hypothesis**: if the model sees patches at more **stacking phases**
during training, it can't memorise "transitions live at relative Z =
±6 within a patch". This forces phase invariance, which should
reduce jaggedness.

Phase coverage math:
- Stride 12: patches start at Z=0, 12, 24, ... → all `mod 6 == 0`. One
  phase visited.
- Stride 3 (v2): Z=0, 3, 6, 9, ... → `mod 6` cycles through {0, 3}. Two
  phases.
- **Stride 2 (v3): Z=0, 2, 4, ... → `mod 6` cycles through {0, 2, 4}.
  Three phases**.
- Stride 1: all six. Full coverage but 3× the training data per epoch.

We picked stride 2 as the cost-effective middle ground.

**Other settings**: same as v2 (offline aug, online rotate, effective
batch 48).

**Result**:
- Training Dice peaked at ~0.71 — **~3 pp lower than v2** (the model
  has more data variation per epoch and converges slower / to a
  slightly lower training-set fit).
- Expert comparison: 0.688 / 0.588 / 0.736. Slightly lower than v2
  across the board. Higher TPR, lower PPV — the model is more
  aggressive (more recall) at the cost of precision.
- **Visually: still jagged**. The user inspected the GIFs / MP4s and
  reported the staircase pattern was still present.

So **train-stride changes alone don't fix the jaggedness**, because
they don't touch the [bias mechanism](diagnosis.md#fact-2-swinunetr-has-explicit-machinery-to-encode-position-specific-z-patterns)
that enables the memorisation. They just change what cross-Z phases
the model sees, and the model adapts to whichever phases it's given.

This was the key insight that motivated v4 (architecture-level
interventions).

## v4 — architectural ablations (in progress)

The next move had to attack the **mechanism**, not the data
distribution. Two parallel variants test the two hypotheses from
[diagnosis](diagnosis.md):

### v4_nobias — kill the bias mechanism

**Change**: `model.swin_unetr.use_relative_pos_bias: false`.

**What it does**: disables the learned relative-position-bias table
inside every `WindowAttention3D` layer. The model still attends across
the full window — it just loses the explicit per-`(Δz, Δh, Δw)`
scalar that was the mechanism enabling the period-6 memorisation.

**Code**: see [window attention — Killing the
bias](../03-models/window-attention.md#killing-the-bias-the-v4-ablation).
The flag threads through:
`__init__.py` → `SwinUNETR3D.__init__` → `BasicLayer3D` →
`SwinTransformerBlock3D` → `WindowAttention3D`. Commit `03bacc5`.

**Hardware**: lg6 (8 × A100 80 GB). Per-rank batch 6, world 8 →
**effective batch 48** (matches v3 / v2 on A100).

**Why batch 6 on lg6**: the 80 GB headroom means the SwinUNETR scores
tensor at `[12, 256, 256]` fits per-rank batch 6 with margin (the
40 GB A100 cap was 4). Effective batch is matched to v2/v3 for clean
comparison.

### v4_winxy3 — starve the XY fallback

**Change**: `model.swin_unetr.window_size_xy: 3` (was 7).

**What it does**: shrinks the XY attention window from 7×7 (49 tokens
per Z slice) to 3×3 (9 tokens). This **cuts the long-range XY context
by ~5×**, forcing the model to lean on Z context the way U-Net does
(small local XY receptive field per layer).

No code change — `window_size_xy` was already a config knob.

**Hardware**: lg3 + lg4 + lg5 (12 × A100 40 GB). Per-rank batch 4,
world 12 → effective batch 48.

### What v4 tells us

By running both in parallel, we directly isolate **which mechanism**
is responsible:

- If **v4_nobias** smooths but v4_winxy3 doesn't → the bias table is
  the dominant cause. Useful for the paper, and the fix is a config
  toggle for future swin runs.
- If **v4_winxy3** smooths but v4_nobias doesn't → the long-range XY
  fallback is the dominant cause. Suggests smaller `window_size_xy` is
  generally better for thin-membrane data.
- If **both smooth** → both interventions help; either is acceptable.
- If **neither** → the next move is the deeper architectural change
  (window-Z capping; see [future directions](future-directions.md)) or
  the data-side fix (soft labels / SDF labels).

## What "still jagged" means quantitatively

Pure Dice isn't sensitive enough to distinguish jagged from smooth
(see [diagnosis](diagnosis.md) for why). A better quantitative check
would be:

```python
# Z-direction smoothness penalty, per voxel
z_jaggedness = abs(mask[1:] - mask[:-1]).mean(axis=(1, 2))
```

A staircase pattern with transitions every 6 Z slices has a periodic
jaggedness signature (peaks at Z mod 6 boundaries). A smooth model has
flat low jaggedness across Z. This metric hasn't been added yet —
it would be a clean follow-up to make the comparison reproducible
without visual inspection.

## What we learned about methodology

A few process lessons from the saga:

1. **Visual inspection caught what Dice missed.** The user's
   willingness to render and look at MP4s was the only reason this
   ever surfaced. Lesson: always render at least one volume per
   architecture during paper-quality runs.
2. **Train-time stride is a weaker lever than expected.** Coarse data-
   side interventions (stride, offline aug variants) don't break
   mechanisms that the architecture has explicit machinery for. To
   force a specific behaviour, intervene where the behaviour lives —
   in the model.
3. **Phase coverage is a useful conceptual tool.** Thinking about
   "how many distinct alignments of the artifact does the model see
   per gradient window" gave us the stride-2 hypothesis. It was wrong
   for SwinUNETR, but the framework still applies — see
   [future directions](future-directions.md) for stride-1 + something.
4. **Isolated ablations beat all-in-one changes.** The v4 plan
   deliberately tests two single-variable changes in parallel rather
   than combining them. If one works, we get a clean attribution. If
   we'd combined them and one worked, we wouldn't know which.
