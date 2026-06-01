# Post-processing (PSP)

The model's raw prediction is a per-voxel binary mask, but it's noisy:
isolated false-positive voxels, small disconnected blobs, and
sometimes thin spurious segments running into the background. The
**PSP step** cleans these up via classical morphological + connected-
component operations, producing a tidier mask for downstream
morphometry.

PSP is run by `gbm.py psp <exp> -it <tag>` (after `infer`, before
`morph`). Implementation lives in `src/infer/psp.py`.

## The pipeline

For each volume's `prediction.npz`, PSP does:

1. **3D connected-component filter** — removes any 3D blob below
   `min_3d_size` voxels.
2. **Per-Z opening by reconstruction**, which combines:
   a. 2D erosion (per Z slice).
   b. 2D connected-component filter (per Z slice) below `min_2d_size`.
   c. 2D geodesic dilation back to the pre-erosion extent.
3. **Optional 2D / 3D hole-filling** below `max_2d_hole_size` /
   `max_3d_hole_size`.
4. **Reconnection** — bridges small gaps within `reconnect_radius`
   voxels, keeping only the top `keep_fraction` of components.

Writes `prediction_psp.npz` per volume.

## Why "opening by reconstruction" rather than plain opening

A naïve **open** (erode then dilate with the same kernel) on a thin
membrane structure would *thin the membrane* — every erosion eats one
voxel off each side, and the dilation only restores up to the
kernel's extent, not to the membrane's actual extent. This biases
the downstream thickness measurement.

**Opening by reconstruction** uses geodesic dilation: starting from the
eroded mask, dilate iteratively but only within the original
(pre-erosion) extent. The result has the same boundary as the original
where the eroded marker was non-empty, and zero where it was — so it
**removes small blobs without thinning the survivors**.

Concretely:

```python
# Pseudocode for the per-Z opening by reconstruction
eroded = erode_2d(mask, kernel_size)
filtered = remove_small_components_2d(eroded, min_2d_size)
restored = geodesic_dilate_2d(filtered, mask, until_convergence=True)
# `restored` ⊆ `mask`, same boundary where filtered was non-empty.
```

This was the design decision baked into `psp.py`. The historical
alternative — a plain erode-then-dilate open — was explicitly rejected
because it biased the downstream thickness measurement (the membrane
that `morph` measures thickness on would have been thinned at every
PSP pass).

## Parallelism — multiprocessing.Pool over volumes

Each test volume is independent, so PSP processes them with a CPU
worker pool (`multiprocessing.Pool`). The sbatch (`sbatch/psp.sbatch`)
allocates one node with many CPUs and `gbm.py psp` fans out across
them.

A historical bug worth knowing about: the **pre-fix code re-ran every
sample serially after the pool**, doubling the work. The Phase-1
cleanup removed that re-run. Coverage in `tests/infer/test_psp.py` is
~79% — the regression tests for the duplication bug are in there.

## Default settings

From `configs/template.yaml: inference.post_processing`:

```yaml
post_processing:
  enabled: true
  min_2d_size: 500
  min_3d_size: 5000
  kernel_size: 3
  max_2d_hole_size: 64
  max_3d_hole_size: 0       # 3D holes left alone by default
  reconnect_radius: 8
  keep_fraction: 0.2
```

`max_3d_hole_size: 0` deliberately disables 3D hole-filling. The GBM
is a topological sheet (no genuine voids inside it), but the
predictions sometimes have legitimate small voids — filling them
risks losing real structure. 2D hole-filling (per-slice) is safer
because the sheet's projection to a single Z slice is approximately
2D and can have hole artefacts from sampling.

## When PSP cleans up too much

PSP can be too aggressive. If `min_3d_size: 5000` is bigger than a
small but real GBM fragment, that fragment is deleted. The current
settings were tuned empirically on the source dataset; if you change
the input data, retune.

The user once observed unwanted regions on inference outputs and
asked for tighter PSP — see commit history around early May 2026,
the "Can you check the results yourself and come up with post-
processing pipeline that removes the unwanted regions" thread.

## What's downstream

`morph` reads `prediction_psp.npz`. So whatever PSP outputs is what
gets measured for thickness. A more aggressive PSP → fewer false
positives in the thickness histograms → cleaner statistics, at the
cost of possibly missing real thin parts.

The visualisations in `stats` (and the rendered MP4s) also use
`prediction_psp.npz` — they show the cleaned mask, not the raw
prediction. So if a render looks "wrong", first check whether PSP
ate something it shouldn't have.

## Tests

`tests/infer/test_psp.py` — 79% coverage. Locks in:

- The component filter sizes don't accidentally include 1-voxel blobs.
- Opening by reconstruction preserves the boundary where the marker
  is non-empty.
- The serial re-run bug is gone (regression test).
- The hole-filling behaviour matches the configured thresholds.
