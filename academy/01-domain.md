# Domain background — biology, microscope, task

You can train a segmentation model without knowing what you're segmenting,
but you can't make good design decisions that way. This doc gives you
just enough biology and instrumentation context to understand why the
pipeline looks the way it does.

## The biological structure: the glomerular basement membrane

A kidney filters blood through ~1 million tiny units called **glomeruli**.
Each glomerulus is a tuft of capillaries wrapped in a specialised
filtration barrier. The middle layer of that barrier is the **glomerular
basement membrane (GBM)** — a sheet of extracellular matrix, ~200–400 nm
thick, that the blood plasma has to pass through before reaching the
urine side. Proteins like nephrin (on the urine-facing podocytes),
collagen-IV (the GBM's structural scaffold), and WGA-stainable
glycoproteins flank or sit inside the membrane.

If the GBM thickens unevenly or becomes bumpy, filtration fails. That's
the clinical signal in diseases like Alport syndrome and diabetic
nephropathy. **Measuring 3D GBM thickness and bumpiness at scale** is
the downstream goal of this whole pipeline — segmentation is means, not
end.

## How the data is acquired: confocal microscopy

The dataset comes from a confocal fluorescence microscope. Three
fluorescent stains light up three different molecules — call them
channels 0/1/2 (nephrin, collagen-4, WGA). The microscope takes a stack
of optical sections (Z slices), each one a 2D image at a fixed depth
inside a fixed glomerulus.

Three things about confocal data matter for the model design:

1. **The voxels are not cubes.** Lateral (XY) resolution is set by the
   diffraction limit and the objective NA, and is typically much better
   than axial (Z) resolution. In this dataset the target voxel size is
   `[0.050, 0.050, 0.300]` µm (X, Y, Z) — Z is **6× coarser** than XY.
   See `configs/template.yaml: experiments.default_voxel_size`.
2. **The membrane is thinner than the Z voxel.** A 300 nm GBM seen at
   300 nm Z spacing fits in one Z slice. So we **upsample Z by 6×**
   before training so the membrane is resolved across multiple slices
   and the 3D convolution kernels (3×3×3) can actually see structure in
   Z. See [Z anisotropy and label upsampling](02-data/z-anisotropy-and-upsampling.md).
3. **Labels are scarce.** Each labelled volume took an expert
   nephrologist tens of minutes per Z slice to annotate; ~15 training
   volumes is what's available. Three annotators each labelled three
   crops for the test set so we can measure inter-rater agreement (the
   "human ceiling"). See [expert comparison](06-inference/expert-comparison.md).

## What the model has to do

Per-voxel binary classification: each voxel of the upsampled 3D volume
is either GBM (1) or background (0). The model sees the three image
channels stacked along the channel axis; ground-truth labels are a 4th
channel in the TIFF (see [TIFF and channels](02-data/tiff-and-channels.md)).

Because the GBM is *very* thin (most voxels are background), this is a
**severely class-imbalanced segmentation problem**. Two design choices
respond to that:
- The [Cont loss](04-loss-cont.md) uses class weights `[3.0, 7.0]` — the
  GBM class is 7/3 ≈ 2.3× more expensive per pixel.
- Dice/IoU rather than accuracy are the headline metrics, both during
  training and in [expert comparison](06-inference/expert-comparison.md).

## Why two architectures

Two segmentation architectures get the same treatment in this repo:

- A **3D U-Net** (`unet_3d`) — the conventional baseline. Encoder/decoder
  with 3×3×3 conv blocks and skip connections. See [unet3d](03-models/unet3d.md).
- A **custom SwinUNETR** (`swin_unetr`) — Swin Transformer encoder
  followed by a (convolutional) decoder. See [swinunetr](03-models/swinunetr.md).

The interesting science is in [comparing their behaviour](03-models/inductive-biases.md)
— particularly in the Z direction, where the conv U-Net behaves quite
differently from the attention-based Swin. That comparison is the
backbone of the [Z-jaggedness case study](07-case-study-z-jaggedness/README.md).

## Where things live

Outside the repo:
- **Dataset** — `/projects/ag-bozek/afatehi/gbm/dataset/ds_mouse/`
  (cluster path; see `configs/template.yaml: experiments.default_data_path`).
- **Experiments** — `/projects/ag-bozek/afatehi/gbm/experiments/<name>/`,
  each with its own copied source tree, datasets, and config. See
  [08-experiment-isolation](08-experiment-isolation.md).

Inside the repo, the headline entry point is `gbm.py` — a single CLI
that dispatches every pipeline stage. See [the pipeline DAG](05-training/pipeline.md).
