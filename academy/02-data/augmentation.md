# Augmentation — offline cache + online sampling

With only ~15 training volumes (see [domain](../01-domain.md)), data
augmentation matters a lot. The pipeline runs **two different
augmentation regimes** that solve different problems and live in
different places.

## Offline augmentation — the cache

**What it is.** A set of deterministic image-level transforms applied
**once per experiment** to every training volume. The outputs are
written to disk under `<exp>/datasets/ds_train/cache/`. During
training, the dataloader treats each cache entry as just another
training file.

**Why offline.** Each transform (zoom, twist) is CPU-heavy on full
~5 GB resized volumes. If we ran them lazily inside the dataloader,
every DDP rank would recompute the same variants on every epoch, and
the workers would burn CPU + RAM at the same time and swap-die. By
precomputing once and writing to disk, every rank just reads.

**The methods.** Configured in the experiment's
`configs.yaml: trainer.data.train_ds.augmentation.methods_offline`
as a list of `[method_name, arg]` pairs. The current swin variants use:

```yaml
methods_offline:
- [_zoom, 0.7]            # shrunk variant
- [_zoom, 1.5]            # zoomed-in variant
- [_twist_clock, '0.5']   # clockwise twist around Z
```

Each pair produces one new TIFF per original training file. With 15
training volumes × 3 methods = **45 augmented variants**. The cache
filename encodes the method, e.g.
`NCWM.AUY380.Series008.Control.20240723.tiff_zoom_0.7.tiff`.

(Historical: earlier experiments used 5 methods. The v2/v3/v4 swin
runs trimmed to 3; see commit messages and
[the Z-jaggedness saga](../07-case-study-z-jaggedness/v2-v3-v4-experiments.md).)

**How it's built.** `sbatch sbatch/offline_aug.sbatch <exp>` submits a
**single-process** CPU job that runs `gbm.py offline-aug <exp>`. Single
process by design — see comment in `sbatch/offline_aug.sbatch`. Takes
~30 minutes for 15 volumes × 3 methods.

**Why it's a separate step.** Until commit `8b1d034`, this was the
slowest part of the pipeline. The script is intentionally separate
from `gbm.py create` (which builds the resized base dataset) because:
- `create` is a one-time setup; offline-aug can change.
- If you tweak the offline methods, you re-run only `offline-aug`, not
  the full create.
- DDP training expects the cache to exist before the first batch; see
  [the v3 launch failure](../07-case-study-z-jaggedness/v2-v3-v4-experiments.md)
  for an instance where this step was skipped and training crashed at
  step 0 on every rank.

**Caveat.** `enabled_offline: true` in the config makes the dataloader
*require* the cache files to exist. If the experiment was created but
offline-aug never ran, training fails at step 0 with
`FileNotFoundError: Offline-augmentation cache missing for …`. This
caught us once during v3 (commit log shows the surgical fix).

## Online augmentation — the dataloader random transforms

**What it is.** Stochastic, per-patch transforms applied **inside the
dataset's `__getitem__`** on every read. They never touch disk.

**The methods.** Configured under
`trainer.data.train_ds.augmentation.methods_online`:

```yaml
methods_online:
  scale: 6                # Z scale (matched to default_z_scale)
  rotate: 0.4             # probability of XY rotation
  rotate_degrees: 45      # max rotation magnitude (±45°)
  blur: 0.2               # probability of Gaussian blur
  crop: 0.2               # probability of random crop / zoom
  channel_drop: 0.3       # probability of dropping one input channel
```

The probabilities decide *how often* a patch gets each transform —
they're applied independently per patch.

**Why rotation only in XY.** Rotation around the Z axis (i.e. in the
XY plane) preserves Z structure but spins the in-plane orientation. We
don't rotate in planes that include Z because Z is axially
different from XY (PSF asymmetry); a YZ rotation would mix axial and
lateral resolution and produce physically nonsensical patches. See
[domain](../01-domain.md) for the source of the anisotropy.

**Why `channel_drop` exists.** With 3 fluorescent channels, the model
could over-rely on one (typically collagen-4, which most directly
marks the GBM). Random channel dropout forces it to also learn from
nephrin and WGA, which improves generalisation to crops where one
stain may be weaker.

## Online rotation matters more than the obvious "augmentation for free"

There's a subtler effect of XY rotation that bridges into the
[case study](../07-case-study-z-jaggedness/README.md): rotation
randomises absolute `(x, y)` coordinates within each patch frame, so the
model can't rely on "the GBM lives at absolute pixel `(x, y)`". This
forces it to use either local image features (helps the conv U-Net) or
relative within-window attention (helps SwinUNETR). The two
architectures fall back on different cues when absolute-XY is taken
away — that's the leverage we exploited in the v3/v4 experiments.

## Key code paths

- `src/data/ds_train.py:GBMDataset.__init__` — registers cache files
  alongside original files via `_register_file(_method=...)`.
- `src/data/ds_train.py:__getitem__` — lazy-loads, applies the online
  transforms.
- `src/scripts/generate_voxels.py` (if you ever need to look at the
  raw offline transforms) — historical/debug helper, not on the active
  path.
- `sbatch/offline_aug.sbatch` — the SBATCH wrapper.

See also the [lazy-load sampler](lazy-loading-sampler.md) for how the
augmented variants get scheduled into batches.
