# gbm-seg Academy

A guided tour of the codebase for someone who knows deep learning basics
(gradients, conv layers, attention, training loops) but not the
biomedical-imaging domain or the specific design decisions that shaped
this project.

Read top to bottom for a coherent picture, or use the index below to
jump to whatever you actually need. Almost every concept here is linked
to a deeper file — follow the links when something isn't obvious.

## Why this exists

This repository trains 3D segmentation models to detect the **glomerular
basement membrane (GBM)** — a thin, curved sheet of extracellular matrix
inside kidney filtration units — from confocal microscopy z-stacks. The
downstream science is GBM thickness and topology as a marker of kidney
disease.

The repo is research-grade: experiments run on a SLURM cluster, two
architectures are compared, and most of the design decisions only make
sense when you understand both the **domain quirks** (anisotropic
voxels, thin membrane, expensive expert labels) and the **engineering
constraints** (small dataset, ~5 GB per Z-upsampled volume, multi-node
DDP). This Academy explains both.

## Table of contents

### 1. Orientation
- [Domain background — GBM biology, confocal microscopy, the segmentation
  task](01-domain.md)

### 2. Data
- [Data overview](02-data/README.md)
- [TIFF layout and channels](02-data/tiff-and-channels.md)
- [Z anisotropy and label upsampling](02-data/z-anisotropy-and-upsampling.md)
- [Augmentation (offline + online)](02-data/augmentation.md)
- [Lazy loading and the block-random sampler](02-data/lazy-loading-sampler.md)

### 3. Models
- [Why two architectures](03-models/README.md)
- [3D U-Net](03-models/unet3d.md)
- [Custom SwinUNETR](03-models/swinunetr.md)
- [Window attention and relative position bias](03-models/window-attention.md)
- [Inductive biases — conv vs attention](03-models/inductive-biases.md)

### 4. Loss functions
- [The Cont loss — weighted CE + Z-continuity term](04-loss-cont.md)

### 5. Training
- [DDP and the multi-node launcher](05-training/ddp-and-multinode.md)
- [Samples-based reporting cadence](05-training/samples-based-reporting.md)
- [Snapper and resume-from-snapshot](05-training/snapper-and-resume.md)
- [The full pipeline DAG: create → train → infer → psp → morph → … → stats](05-training/pipeline.md)

### 6. Inference and downstream
- [Patch stitching with Gaussian accumulation](06-inference/patch-stitching.md)
- [Post-processing (PSP)](06-inference/post-processing.md)
- [Morphometry — thickness via distance transform](06-inference/morphometry.md)
- [Expert comparison — Dice vs three annotators](06-inference/expert-comparison.md)

### 7. Case study — the Z-jaggedness saga
- [The problem](07-case-study-z-jaggedness/README.md)
- [Diagnosis: why SwinUNETR memorises the stacked-label period](07-case-study-z-jaggedness/diagnosis.md)
- [Experiments v2 → v3 → v4](07-case-study-z-jaggedness/v2-v3-v4-experiments.md)
- [Future directions](07-case-study-z-jaggedness/future-directions.md)

### 8. Engineering practices
- [Experiment isolation and reproducibility](08-experiment-isolation.md)

## How to read this

You don't have to read in order, but if you've never touched the project
the linear path is:

```
01 (domain)
 → 02 (data)
 → 03 (models)
 → 04 (loss)
 → 05 (training)
 → 06 (inference)
 → 07 (the most interesting open problem)
 → 08 (how reproducibility is enforced)
```

If you're picking up an existing problem, [07 (Z-jaggedness)](07-case-study-z-jaggedness/README.md)
is the richest entry — it pulls in concepts from every other section.

## Conventions used in this Academy

- `path/to/file.py:LINE` — clickable in editors. Cited line numbers can
  drift; if the cited code looks different, search the file for the
  function name.
- "the user" inside code (`_name`, `_root_path`) — the repo uses a
  **leading-underscore parameter convention** by design. Don't
  "fix" it; it's how the project style is set throughout `gbm.py`
  handlers and `src/utils/exper.py`.
- "experiment" = one materialised training run with its own copy of the
  code, datasets, and config (see [08](08-experiment-isolation.md)).
- "stage" can mean either an architecture stage (Swin encoder stage) or
  a pipeline stage (`infer`, `psp`, etc.) — context disambiguates.
