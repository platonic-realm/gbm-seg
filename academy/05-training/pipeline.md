# The full pipeline DAG

End-to-end, a paper-quality experiment runs through this DAG:

```
create
  ↓
offline-aug
  ↓
train ────┐
          ↓
         (after training completes)
          ↓
    ┌── infer (unlabeled — 15 test volumes) ──┐
    │                                          ↓
    │                                         psp
    │                                          ↓
    │                                         morph ─┬─→ stats ─┐
    │                                                │           │
    │                                                ↓           │
    │                                              blender → render → export
    │                                                                    │
    └── infer-labeled (3 expert-annotated crops) ─→ psp-labeled ─────────┤
                                                                          │
                                                              expert_comparison
                                                              (inside stats)
```

Each box is one `gbm.py` subcommand, dispatched from
`gbm.py:HANDLERS` (`gbm.py:204`). All stages are SLURM-able as
individual sbatch jobs, and `sbatch/run_pipeline.sh` wires them
together with `--dependency=afterok:...` so the chain auto-fires when
training finishes.

## Stage by stage

### `create` — set up the experiment

`gbm.py create <name>` is the one-time setup. It:

1. **Copies the project source** into `<exp>/code/`. This freezes the
   code as it was when the experiment started. The training and
   inference processes don't actually USE this copy — they use the
   working dir — but it's an audit trail.
2. **Resizes every TIFF** in the default dataset (XY zoom +
   Z-upsample) and writes them into `<exp>/datasets/{ds_train,
   ds_test_unlabeled, ds_test_labeled}/`. With the parallelised
   resize (commit `8b1d034`), this takes ~3 min for ~40 TIFFs.
3. **Snapshots reproducibility metadata** —
   `<exp>/git_sha.txt`, `<exp>/git_diff.patch` (if working tree is
   dirty), `<exp>/requirements.txt` (pip freeze).
4. **Renders `configs/template.yaml` → `<exp>/configs.yaml`**, scaling
   LR by `sqrt(batch_ratio)` if `batch_size` differs from
   `experiments.default_batch_size`. This is the experiment's
   editable config; subsequent training reads it, not the template.

See [experiment isolation](../08-experiment-isolation.md) for the
philosophy.

### `offline-aug` — precompute augmented variants

`gbm.py offline-aug <name>` applies the configured offline
augmentation methods (zoom, twist) to every training volume and
writes the results to `<exp>/datasets/ds_train/cache/`. See
[augmentation](../02-data/augmentation.md). Single-process by design;
takes ~30 min for 15 volumes × 3 methods.

### `train` — the actual training

`gbm.py train <name>` runs the trainer (either all-data or k-fold CV).
Reads `<exp>/configs.yaml`, builds the model + loss + optimizer +
scheduler via `src/train/factory.py`, runs the loop. Writes
snapshots to `<exp>/results-train/snapshots/{all_data,fold-N}/`.
See [DDP](ddp-and-multinode.md), [samples-based
reporting](samples-based-reporting.md), [snapshot
resume](snapper-and-resume.md).

Multi-node launch: `sbatch/launch_multi_node.sh <exp> --nodes <csv>
--gpus <n>` (the user-facing wrapper). Single-node: `sbatch
sbatch/train_all_data.sbatch <exp>`.

### `infer` — predict on a test set

`gbm.py infer <exp> -s <snapshot> -bs <batch> -sd <sample_dim>
-st <stride> -sf <scale> -in <interpolate>` runs the trained model on
each volume in `ds_test_unlabeled/` (or `ds_test_labeled/<annotator>/`
when `--labeled`). Writes per-volume outputs to
`<exp>/results-infer/<tag>/<volume_name>/prediction.npz` (the raw
argmax mask).

The **inference tag** is derived from the CLI args:
`<snapshot>_<sample_dim_joined>_<stride_joined>_<scale>`. So a run
with patch `[12, 256, 256]`, stride `[3, 128, 128]`, scale 6 from
snapshot `all_data/001-11500.pt` has tag
`all_data/001-11500.pt_12256256_3128128_6`. Every downstream stage
(`psp`, `morph`, …) takes `-it <tag>` to locate that directory.

The infer parent (sbatch job) enumerates volumes in `ds_test_*`,
**submits a SLURM array** with one task per volume, and blocks until
every array task finishes. See `sbatch/infer.sbatch`.

### `psp` — post-processing

`gbm.py psp <exp> -it <tag>` reads `prediction.npz` from each volume,
applies the [PSP pipeline](../06-inference/post-processing.md) (3D
component filter → per-Z opening by reconstruction), writes
`prediction_psp.npz`. Multiprocessing parallelised across volumes.

### `morph` — thickness measurement

`gbm.py morph <exp> -it <tag>` reads `prediction_psp.npz`, applies a 3D
distance transform inside the mask to measure local thickness, and
applies a PSF correction. Writes `distance_result.npz` and a PSF-
corrected variant. See [morphometry](../06-inference/morphometry.md).

The morph parent (sbatch job) also submits an array — one task per
volume — because morphometry on a 5 GB volume is single-thread CPU-
heavy.

### `stats` — aggregate stats + expert comparison

`gbm.py stats <exp> -it <tag>` aggregates per-volume thickness arrays
into summary statistics, histograms, polar plots, top-down views, and
a comparative box plot. It also **automatically runs the
[expert comparison](../06-inference/expert-comparison.md)** if both
`results-infer/<tag>/` and `results-infer-labeled/<tag>/` exist on
disk (the labeled branch fires in parallel from `run_pipeline.sh`).

Writes to `<exp>/results-infer/<tag>_stats/`:
- `summary_statistics.npz`, `comparative_box_plot.png`
- Per-volume histograms, polar plots, top-down views
- `expert_comparison.yaml`, `expert_comparison_summary.yaml`, `expert_comparison.png`

### `blender`, `render`, `export` — the visual branch

These three produce 3D rendered MP4s of the segmented GBM:

- `gbm.py blender <exp> -it <tag>` — marching cubes on
  `prediction_psp.npz` → 3D mesh (`mesh.obj` or `.ply`). Per-volume
  array.
- `gbm.py render <exp> -it <tag>` — spawns Blender (the 3D-graphics
  package, via subprocess) to render each mesh into a series of
  PNG frames. The slow stage — ~3 hours on 15 volumes.
- `gbm.py export <exp> -it <tag>` — assembles the PNG frames into MP4
  videos and the segmented volumes into BigTIFF stacks for Fiji.

The visual branch is independent of `stats`; both fan out from
`morph`.

## `run_pipeline.sh` — the orchestrator

`sbatch/run_pipeline.sh <exp> <snapshot> <bs> <sample_dim> <stride>
<scale> <interp>` submits the whole post-training pipeline as a chain
of SLURM jobs with `--dependency=afterok:<prev>` links. It also forks
a **labeled inference branch** for the expert comparison
(commit `2b29419 run_pipeline.sh: fork parallel labeled inference
branch`).

Visually:

```
infer ─→ psp ─→ morph ─┬→ blender → render → export
                       └→ stats
                          ↑
                          (also needs psp-labeled, see below)

infer-labeled ─→ psp-labeled ─┘
```

`stats` depends on both `morph` AND `psp-labeled`, so it fires when
both branches have finished. If only the unlabeled branch is wanted
(no expert comparison), pass appropriate flags — see
`run_pipeline.sh` directly for the current arg list.

## CLI catalog

| Subcommand | Purpose | Sbatch wrapper |
|---|---|---|
| `gbm.py create <exp>` | One-time setup | `sbatch/create.sbatch` |
| `gbm.py offline-aug <exp>` | Precompute aug cache | `sbatch/offline_aug.sbatch` |
| `gbm.py train <exp>` | DDP training | `sbatch/train_multi_node.sbatch` (via launcher) |
| `gbm.py infer <exp>` | Predict on test set | `sbatch/infer.sbatch` (with `--labeled` to switch sets) |
| `gbm.py psp <exp>` | Post-processing | `sbatch/psp.sbatch` |
| `gbm.py morph <exp>` | Thickness measurement | `sbatch/morph.sbatch` (array per volume) |
| `gbm.py blender <exp>` | Marching cubes → mesh | `sbatch/blender.sbatch` |
| `gbm.py render <exp>` | Blender PNG render | `sbatch/render.sbatch` |
| `gbm.py export <exp>` | Frames → MP4/TIFF | `sbatch/export.sbatch` |
| `gbm.py stats <exp>` | Aggregated stats + expert comparison | `sbatch/stats.sbatch` |
| `gbm.py delete <exp>` | Remove experiment | (direct, no sbatch needed) |
| `gbm.py ablate <spec>` | Materialise an ablation study | (specialised path) |

Each takes `-it <tag>` to locate the inference results, except
`create`, `offline-aug`, `train`, `delete`, `ablate` which work at the
experiment level.

`gbm.py infer` and `gbm.py delete` accept `-f/--force` to be
non-interactive — critical for SLURM-driven runs (no TTY to confirm).
