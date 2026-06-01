# Experiment isolation and reproducibility

This repo treats each experiment as a **self-contained snapshot** of
the code, the data, and the configuration — not just a config file in
a shared workspace. This doc explains why and how.

## What `gbm.py create <name>` actually does

When you create an experiment, **four things** end up under
`<experiments_root>/<name>/`:

```
<exp>/
├── code/                  # full copy of the project source at create time
│   ├── src/
│   ├── train.py
│   └── ...
├── datasets/              # resized + Z-upsampled copies of the data
│   ├── ds_train/
│   ├── ds_test_unlabeled/
│   └── ds_test_labeled/<Chris,David,Robin>/
├── configs.yaml           # rendered template, editable per experiment
├── git_sha.txt            # `git rev-parse HEAD` at create time
├── git_diff.patch         # working-tree diff at create time (if dirty)
└── requirements.txt       # `pip freeze` at create time
```

The four key effects:

1. **Editing the working-dir source doesn't affect a running experiment.**
   At training/inference time, the launchers do `cd ~/.vix/projects/gbm-seg`,
   which is the **working dir, not the experiment's code/**. So when you
   edit the working-dir source mid-run, the *next* training step picks
   up the change.

   This is actually critical for fast iteration (you don't have to
   re-create an experiment to change the trainer). It's also a footgun
   (you can change the code mid-run and confuse yourself).

2. **The experiment's `code/` is for an audit trail.** When you write
   the paper and want to know "what code produced this snapshot", look
   at `<exp>/code/`. Combined with `git_sha.txt` and `git_diff.patch`,
   it lets you reconstruct the exact code that started the experiment
   — even if you've made many edits to the working tree since.

3. **The datasets are also per-experiment.** When `default_z_scale`
   changes or you switch source datasets, a fresh `gbm.py create`
   re-processes from `default_data_path`. The old experiment's
   datasets stay intact.

4. **`configs.yaml` is the per-experiment editable config.** It starts
   as a render of `configs/template.yaml` but can be edited freely
   afterwards. The template is **only** read at create time. Editing
   the template after that doesn't propagate to existing experiments.

## Why this isolation matters

A common pattern in ML research: someone tweaks a config, runs a
training, observes something interesting, tweaks again, runs another
training. Three months later, no one remembers what *exactly* was in
the config for the original interesting run. Without isolation, the
experiment's results are tied to ephemeral state.

With this pattern, every interesting run is anchored to:
- A specific code state (`git_sha.txt` + `git_diff.patch` + the
  `code/` snapshot).
- A specific dataset state (the resized TIFFs in `datasets/`).
- A specific config state (`configs.yaml`).

All three are stored on disk for the lifetime of the experiment.
Reproducing a result is `gbm.py create <new_name>` from the same
template + the same `git_sha` of the source, or by re-running from
the existing experiment's preserved state.

## How `gbm.py delete` works

`gbm.py delete <name> -f` removes the experiment directory in full.
With `-w` it also deletes the experiment's W&B runs (matched by
`group = experiment_name` in W&B's API). Both flags are required for
non-interactive batch jobs — there's no TTY to confirm.

The `-w` flag is best-effort: a W&B API failure won't block the local
delete, but you'll see a warning.

## How LR scaling works at create time

Inside `gbm.py create`, the template's LR is scaled by
`sqrt(batch_ratio)`:

```python
batch_ratio = configs.trainer.batch_size / experiments.default_batch_size
configs.trainer.lr *= sqrt(batch_ratio)
```

So if the template default is `lr=1e-4` at `batch=4`, and you create
with `--batch-size 8`, the rendered config has `lr=1e-4 * sqrt(2) ≈
1.41e-4`.

This is the square-root scaling rule — between the linear scaling
rule (Goyal et al., 2017) and not scaling at all. Empirical compromise.

The flag `experiments.scale_learning_rate_for_batch_size: False` in
the template disables this — it's set in the current template, so
**LR is NOT scaled** in current experiments. Setting it explicitly
in the config gives you full control.

Whatever you do, **post-create config edits don't re-scale LR**. If
you patch `batch_size` after create, the LR stays the same.

## Snapshotting `git_sha.txt` and `git_diff.patch`

At create time, `gbm.py create` runs:

```bash
git rev-parse HEAD                  > git_sha.txt
git diff > git_diff.patch           # if working tree is dirty
```

This captures the exact state of the code, including any uncommitted
changes. Combined with the working-dir `code/` snapshot, you have:

- `git_sha`: which commit the create was based on.
- `git_diff.patch`: what additional changes were on top.
- `code/`: the literal file contents.

The three are redundant, by design. If your workflow ever loses one
(e.g. `code/` gets accidentally deleted), the git sha + diff can
reconstruct.

## When to NOT use experiment isolation

Some operations are simpler without it:

- **Bug-fix iteration**: don't create a new experiment for every code
  fix. Edit the working dir, run tests, commit, re-run the trainer.
  The running training picks up the next step from the new code (well,
  sort of — most code changes only matter at process startup, but the
  dataloader's `__getitem__` is read on every batch).
- **Throwaway experiments**: a 1-hour smoke test doesn't need an
  audit trail. Use a meaningful name like `smoke_test_<date>` and
  delete it afterwards (`gbm.py delete smoke_test_<date> -f`).

For paper-quality runs, experiment isolation is **always** worth it.

## Where experiments live

`configs/template.yaml: experiments.root` points to the experiments
root. The cluster path
`/projects/ag-bozek/afatehi/gbm/experiments/` is currently active;
home-directory paths are commented out (for local testing if needed).

Every experiment is a sub-directory of that root. The current set
(end-of-session 2026-05-30):

| Experiment | Purpose | Status |
|---|---|---|
| `unet_offline_v100` | U-Net baseline | trained, inferred |
| `swin_offline_a100_v2` | SwinUNETR baseline (stride 3) | trained, inferred |
| `swin_offline_a100_v3` | SwinUNETR with stride 2 | trained, inferred |
| `swin_offline_a100_v4_nobias` | bias-table ablation | in progress |
| `swin_offline_a100_v4_winxy3` | small XY window ablation | in progress |

## A reproducibility audit checklist

If you need to reproduce a result from this repo:

1. `cat <exp>/git_sha.txt` → check out that commit.
2. `git apply <exp>/git_diff.patch` if non-empty.
3. `pip install -r <exp>/requirements.txt` in a fresh venv.
4. Re-create the experiment via `gbm.py create <new_name>` (with the
   same template settings as the original — visible in the original's
   `configs.yaml`).
5. Re-run `offline-aug` and the training pipeline.

The dataset is in `default_data_path` which lives **outside the repo**
(`configs/template.yaml: experiments.default_data_path` points to a
cluster path). That's a separate version-control concern — you need
access to the same source TIFFs. The repo doesn't (and shouldn't)
contain copies of training data.
