# Snapper and resume-from-snapshot

A 2-day SLURM walltime is shorter than most of our training runs. The
[Snapper](#) component periodically writes everything needed to
**resume** training to a `.pt` file, and the trainer can pick up
mid-run from one of those files.

## The save artefact

`Snapper.save()` (in `src/train/snapper.py`) writes a single `.pt`
file containing:

| Key | What |
|---|---|
| `MODEL_STATE` | Model weights (the only thing inference needs) |
| `OPTIMIZER_STATE` | Adam's `m` and `v` moments + step counter |
| `SCHEDULER_STATE` | `poly_decay` (or whatever) state |
| `STEPPER_STATE` | Mixed-precision scaler state |
| `EPOCH`, `STEP` | Where we were |
| `BEST_VALID_*` | The best-validation tracker (None for all-data runs) |
| `WANDB_RUN_ID` | So a resumed run reattaches to the same W&B run |
| `RNG_PYTHON`, `RNG_NUMPY`, `RNG_TORCH`, `RNG_TORCH_CUDA` | Reproducibility |
| `MODEL_CARD` | Architecture hyperparameters for cross-checking |

Filename convention: `{epoch:03d}-{step}.pt`. E.g.
`001-11500.pt` is epoch 1, step 11500.

## File size — why it's not just the weights

A typical unet checkpoint:

- MODEL_STATE: ~97 MB
- OPTIMIZER_STATE (Adam m + v): ~190 MB
- Other state: small
- **Total: ~290 MB**

Adam keeps two moments per parameter (first and second), so its state
is roughly 2× the model size. That's most of the file. If you only
want inference snapshots (no resume), you can strip everything except
`MODEL_STATE` and save ~⅔ the disk — see the user's question in the
session log ("Why the model checkpoints are getting bigger than
before"). Not currently implemented as a flag; would be a one-line
Snapper option.

For SwinUNETR which has much smaller `feature_size`, a checkpoint is
~48 MB — most of that is still the optimizer state.

## The save schedule

The trainer writes a snapshot at every report (~ every 16000 samples;
see [samples-based reporting](samples-based-reporting.md)). So
training produces a snapshot every ~30 min of wall time. Historical
snapshots accumulate in
`<exp>/results-train/snapshots/all_data/{000-1000.pt, 000-2000.pt, …}`.

The directory pattern is `<snapshot_path>/<fold_tag>/` where
`fold_tag` is `all_data` for all-data training or `fold-<k>` for CV
training (see `src/utils/exper.py:357` and `train.py:185`).

## Resume — the `continue/` slot

This is the bit that bit us once. **Resume is opt-in**, not automatic.
Look at `Snapper.load()` (`src/train/snapper.py:208`):

```python
def load(self, ...):
    # Auto-discovery mode (_path is None): looks in
    # <snapshot_path>/continue/ for the most-recent .pt.
    # Files in <snapshot_path>/ itself (historical snapshots) are
    # intentionally ignored.
    ...
```

To resume from a snapshot, you have to **move (or copy) it into a
`continue/` subdirectory**:

```
<exp>/results-train/snapshots/all_data/
├── 000-1000.pt
├── 000-2000.pt
├── …
├── 001-10000.pt
└── continue/
    └── 001-10000.pt    # <-- only this one is auto-loaded on next train
```

Why this design instead of "load the most recent": because the
history accumulates dozens of snapshots over a 2-day run, and the
"most recent" might not be the one you want to resume from (e.g. you
might want to roll back to a pre-divergence snapshot). Explicit
placement is safer.

When `Snapper.load()` finds a `continue/*.pt`, it loads everything,
sets `starting_epoch = saved_epoch + 1`, and the trainer's epoch loop
picks up there. `train.py:288`:

```python
starting_epoch = (resume['epoch'] + 1) if resume else 0
```

The `train_multi_node.sbatch` does NOT skip this — every launch tries
to resume; if there's no `continue/` dir, `Snapper.load()` returns
`None` and the trainer starts fresh.

## W&B run continuity

Snapper stores the `WANDB_RUN_ID`. On resume, `maybe_init_wandb` reads
it and calls `wandb.init(resume="must", id=...)` (`train.py:74`)
rather than starting a fresh run. So the W&B chart continues from
where it left off — no separate "resumed" run.

## A quirky failure mode

If the resumed code is meaningfully different from the code that wrote
the snapshot (e.g. you changed the model architecture), the `MODEL_STATE`
loader will fail with a shape mismatch. The `MODEL_CARD` key is meant
to be a sanity check ("this snapshot was for `Unet3D(feature_maps=[64,
128, 256, 512])`"), but it's only logged, not enforced. Be careful
not to resume into an experiment whose code has drifted.

## Tests

`tests/train/test_snapper.py` — ~89% line coverage. Locks in:

- Save/load roundtrip for every component (weights, optimizer state,
  scheduler state, RNG).
- Symmetric handling of `DataParallel`-wrapped vs bare models (the
  trainer used to use DP; the symmetry survives even though we're on
  DDP now).
- Auto-discovery from `continue/` (the resume slot).
- `weights_only=True` is used when loading, to avoid arbitrary code
  execution from a malicious snapshot.

## Practical recipe — resuming a TIMEOUT'd training

The unet's first training run (`42925`, May 2026) hit the SLURM 2-day
walltime. To resume:

```bash
SD=/projects/ag-bozek/afatehi/gbm/experiments/unet_offline_v100/results-train/snapshots/all_data
mkdir -p "$SD/continue"
cp "$SD/001-10000.pt" "$SD/continue/"   # pick the snapshot to resume from
sbatch --time=7-00:00:00 --gres=gpu:A100:4 sbatch/train_all_data.sbatch unet_offline_v100
```

The walltime is bumped to 7 days, the train job auto-loads
`001-10000.pt` from `continue/`, and the W&B chart picks up from
samples ~320k.

The "from where" is the user's choice — the latest snapshot for
maximum continuation, or an earlier snapshot if you want to roll back
to before a regression.
