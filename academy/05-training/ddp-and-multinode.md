# DDP and the multi-node launcher

Training runs across multiple GPUs (and multiple nodes) via PyTorch's
`DistributedDataParallel` (DDP). This doc explains the launcher that
sets it all up, why we picked DDP over DataParallel, and the
non-obvious details that bite when something goes wrong.

## What DDP is, in one paragraph

DDP runs **one Python process per GPU**. Each process loads its own
batch slice (so the **effective batch** is per-rank batch × world
size), runs forward and backward, and the framework all-reduces the
gradients across all processes before each `optimizer.step()`. After
the all-reduce, every rank has identical model parameters — they
diverged momentarily during the backward and converged again before
the step.

Compared with `nn.DataParallel` (DP), DDP:
- Scales to multiple nodes. DP is single-process and can't.
- Has lower communication overhead (no master/worker pattern; all
  collectives are symmetric).
- Requires more careful handling (one process per rank; spawn
  semantics; rank 0 does the logging / W&B init).

Run the trainer with `runtime.ddp: true, runtime.dp: false` in the
config. That's the default for every experiment in this project.

## The multi-node launcher

`sbatch/launch_multi_node.sh` is the user-facing wrapper. It accepts
either an *explicit* nodelist or an *auto-arch* mode that discovers
the largest pool of one GPU architecture in the partition.

```bash
# Explicit (preferred)
sbatch/launch_multi_node.sh <exp> --nodes lyn-gpu-03,lyn-gpu-04 --gpus 4

# Auto
sbatch/launch_multi_node.sh <exp> --arch A100
```

What it does:

1. Resolves the nodelist + per-node GPU count.
2. Computes per-task CPU and memory budgets from `scontrol show node`
   — defaults to `8 × gpus_per_node` CPUs and 90% of the smallest
   "FreeMem" across the pool (so the job is sized to fit the
   tightest box).
3. Submits `sbatch/train_multi_node.sbatch` with the nodelist,
   resources, and the experiment name. The sbatch script then runs
   `srun bash sbatch/_train_multi_node_inner.sh` on each node, which
   in turn runs `torchrun --rdzv-endpoint=... gbm.py train ...`.

The inner script is the only place that knows about `torchrun`
specifically. Everything above is "submit and forget".

## Why explicit nodes are preferred

The auto-arch mode is convenient but two issues:
- It scans `scontrol show node` for liveness, which can mismatch
  reality if a node is degraded.
- Per-node GPU count is set to the minimum across the discovered
  pool — so an 8-GPU lg6 contributes only 4 if there's a 4-GPU peer
  in the pool. To use lg6's full 8 GPUs you'd use a dedicated
  `sbatch/train_all_data_lg6.sbatch` or explicit launch on lg6 alone.

For reproducible launches, **explicit nodes** are best practice. Every
experiment in this session was launched explicitly.

## Heterogeneous DDP is bad — don't mix nodes with different GPUs

DDP all-reduces gradients at every step. If one rank is slower than
the others, every rank waits. Mixing 80 GB A100s on lg6 with 40 GB
A100s on lg3/4/5 in one DDP job sounds tempting (more GPUs!) but in
practice:

- Per-rank work must be uniform — heterogeneous DDP batches stall on
  the sync barrier and bias the gradient mean (the slower-but-larger
  batch contributes less per unit time).
- Memory headroom differs across nodes, so the per-rank batch is
  capped at the smallest node's limit anyway.

The right pattern: **one DDP job per homogeneous pool**. Run swin v4
variants in parallel — one on lg6 (8 × 80 GB) by itself, one on
lg3+lg4+lg5 (12 × 40 GB).

## Effective batch and the LR scaling rule

The **effective batch** is `per_rank_batch × world_size`. With per-rank
4 and world 12, effective is 48. With per-rank 6 and world 8, also 48
— that's why we matched the two parallel v4 variants on effective
batch.

In principle, larger effective batches need larger learning rates to
maintain the same per-sample gradient signal (linear or
square-root scaling). In practice, `gbm.py create` scales the LR by
`sqrt(batch_ratio)` if the experiment's `batch_size` differs from the
template default, *at create time*. After-the-fact patching the
`batch_size` in `configs.yaml` (as we do post-create) does NOT re-scale
LR. We just accept the same LR; it's been adequate.

See `src/utils/exper.py:scale_learning_rate_for_batch_size` and the
template's `experiments.scale_learning_rate_for_batch_size` flag.

## Memory budget walls

Per-rank batch can't exceed what the GPU can hold. The big numbers:

| Model | Patch | 40 GB A100 max batch | 80 GB A100 max batch |
|---|---|---|---|
| Unet3D | `[12, 256, 256]` | ~8 | ~16 |
| SwinUNETR | `[12, 256, 256]` | **4 (comfortable), 6 (borderline)** | 8 cleanly |
| SwinUNETR | `[24, 128, 128]` | OOM (deferred work) | borderline |

The Swin memory wall is dominated by the **window-attention scores
tensor** `(B*nW, heads, N, N)`, not by gradient activations.
Gradient checkpointing helps a little but not a lot for this term.
See [the swin model doc](../03-models/swinunetr.md#memory-wall) for
the detailed breakdown.

## Sbatch templates

Three relevant templates in `sbatch/`:

| File | When to use |
|---|---|
| `train_multi_node.sbatch` | Multi-node DDP (driven by `launch_multi_node.sh`) |
| `train_all_data.sbatch` | Single-node 4 GPU, all-data training |
| `train_all_data_lg6.sbatch` | Single-node lg6 (8 × 80 GB A100) |

The `train_all_data.sbatch` script is the simplest case: 4 GPUs on one
node, `torchrun --standalone --nproc-per-node=4`. Used for the original
unet runs.

## Resume / fault tolerance

DDP doesn't checkpoint by itself; the [Snapper](snapper-and-resume.md)
does. The trainer writes snapshots periodically, and if a job is
re-launched the Snapper picks up the latest from `<snapshot_dir>/continue/`
(the resume slot — separate from the regular snapshot history). See
the [snapper doc](snapper-and-resume.md) for the directory layout and
the once-bitten lesson (resume requires explicitly *moving* a snapshot
into the `continue/` slot — auto-discovery doesn't pick from the
history).

## Tests and verification

- `tests/train/test_distributed.py` — DDP utility tests (rank0,
  world_size, is_distributed); CPU-only, doesn't actually launch DDP.
- Submitting + watching a multi-node job: see the launcher signature
  above, monitor with `squeue -u $USER`, tail
  `/home/afatehi/logs/gbm/train_multi_<jobname>_<jobid>.log`.

The first cadence line in the log is the simplest sanity check that
DDP is alive:

```
[INFO] Reporting cadence: report_freq=2000 (at reference batch 8)
       → effective batch 48 → step interval 333 ...
```

One such line per rank → world size × this line. Mismatch means
something's off (a rank failed to join the rendezvous).
