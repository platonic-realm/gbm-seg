# Samples-based reporting cadence

A small but consequential design decision: training metrics and W&B
charts are plotted against a **samples-processed axis**, not a
training-step axis. This doc explains why and how.

## The problem this solves

If two training runs use **different per-rank batch sizes** (say
batch=4 on 12 GPUs vs batch=6 on 8 GPUs — both effective batch 48),
their training curves *can't* be directly compared on a step axis.
Run A at step 1000 has processed `1000 × 48 = 48000` samples; run B
at step 1000 has processed `1000 × 48 = 48000` samples too (lucky
because we matched effective batch). But if run C has effective batch
32, step 1000 means only 32000 samples — different points in
"training progress".

The fix: report against **samples processed** rather than steps. Then
runs with different batch configurations land on the same x-axis.

## How it's wired

The trainer tracks total `samples_processed = stepper.getSteps() ×
effective_batch_size`. Both the local console log and W&B `wandb.log()`
calls pass this as the **step metric** (a W&B feature for custom
x-axes).

`src/utils/metrics/log/metric_logger.py` and
`src/utils/metrics/log/metric_wandb.py` both renamed their internal
`_step` parameter to `_samples` and emit a `samples` field in every
log payload. W&B's `wandb.define_metric()` declares `samples` as the
custom x-axis:

```python
# train.py:maybe_init_wandb (after wandb.init)
wandb.define_metric("samples")
wandb.define_metric("train/*", step_metric="samples")
wandb.define_metric("valid/*", step_metric="samples")
```

So every `train/loss` or `valid/dice` chart on the W&B UI plots
against the samples axis automatically.

## The configurable cadence

The trainer needs to decide how often to log. The natural unit is
**samples between reports**, but the config knob is in *steps* (so it
maps to what the trainer's inner loop counts).

Reference batch is `8` (constant in `src/train/factory.py:54`,
`REFERENCE_EFFECTIVE_BATCH`). The config knob `report_freq` is the
desired report cadence **at the reference batch**. At the actual
effective batch, the step interval is scaled:

```python
def _scaled_freq_steps(report_freq, effective_batch_size):
    samples_per_report = REFERENCE_EFFECTIVE_BATCH * report_freq
    return max(1, samples_per_report // effective_batch_size)
```

So `report_freq=2000` (the template default) means
`8 × 2000 = 16000 samples between reports`. At effective batch 48,
that's `16000 / 48 ≈ 333 steps`. At effective batch 32, it's
`500 steps`.

The trainer's first console line confirms the cadence so you can
sanity-check at launch time:

```
[INFO] Reporting cadence: report_freq=2000 (at reference batch 8)
       → effective batch 48 → step interval 333
       (samples between reports: 15984)
```

(The "15984" is the rounded version — `333 × 48` — because we round
the step interval to an integer.)

## Why this matters in practice

The Z-jaggedness case study compared three swin variants (v2, v3,
v4_nobias, v4_winxy3) all at effective batch 48 — but the previous
swin_offline_a100 was at effective batch 32. With samples-based
reporting, every run's training curve sits on the same axis, and we
can read off "by 200k samples, run X has Dice 0.5, run Y has Dice 0.4"
without doing arithmetic. With step-based reporting, the comparison
would be ambiguous.

## What about the unet's old runs

The unet_offline_v100 experiment was created **before** this change.
Its console log uses `Step:` instead of `Samples:`. When we resumed
unet_offline_v100 in late May 2026 (commit log), the resumed run's
log used the new `Samples:` format because the working-dir code (which
both training and inference use) had moved on, even though the
experiment's `code/` snapshot was older.

This is a feature of [experiment
isolation](../08-experiment-isolation.md): code in the experiment dir
is for *reproducibility audit*, not for execution. Execution uses the
working dir.

## The trade-off — fewer points

Doubling `report_freq` halves the resolution of the W&B chart but
each point averages more samples (smoother). The user once asked to
double it from 8000 to 16000 specifically because the jagged curves
were hard to read — see the commit history (`9c253ca config: report
every 16000 samples (was 8000)`). That's the current default.

## Tests

`tests/utils/metrics/test_metric_logger.py` and
`test_metric_wandb.py` lock in the `samples` payload format and the
`define_metric` calls. Both around 100% coverage.

## What the config looks like in practice

`configs/template.yaml`:

```yaml
trainer:
  logging:
    report_freq: 2000   # samples-between-reports = 8 × 2000 = 16000
```

When you create an experiment, the rendered `configs.yaml` inherits
`report_freq` from the template. Patches to the experiment's config
override; the value at run time is what matters.
