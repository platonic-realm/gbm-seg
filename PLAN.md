# gbm-seg — improvement plan history

A condensed record of the two improvement passes done on this codebase, the rationale behind each accepted change, and what's still planned. Useful when reviewing commit history months later, or when an external collaborator wants to know "why did you do X".

The master plan / discussion ledger (with full per-item rationale and pending work) lives outside the repo — this file is the version that ships.

---

## Two passes, two purposes

The work splits cleanly into two passes:

1. **Pass 1 — Code-quality overhaul (Phases 1–5)**: correctness bug fixes, dead-code purge, architectural cleanup, test suite, lint baseline. Single commit `9b543ff`. Did not change methodology or behaviour by design — only fixed bugs and reorganised.
2. **Pass 2 — Methodology review (Phases 6+)**: acted on a thorough scientific review by a DL-research-scientist agent. Items selected on the basis of impact, evidence in the literature, and compatibility with the project's specific contribution (label-efficient 3D segmentation at the upsampled isotropic grid).

---

## Hard constraints carried throughout

These shape what was accepted and rejected. New work must respect them.

### Selling point of the project

**Label-efficient 3D segmentation at the upsampled isotropic grid.** Training input is trilinear-interpolated 6× in Z; training labels are *replicated* 6× (each manually-drawn slice tiled to fill the interpolated z-positions). The model — a standard 3D U-Net with uniform `(3,3,3)` kernels throughout — is then supervised at the isotropic grid against these replicated labels and is expected to learn the true continuous Z structure despite the coarse label signal. Inference's `scale_factor=6` Z-interpolation is the methodology being demonstrated, **not** an accident. Both training and inference operate at the upsampled isotropic grid.

This rules out: anisotropic kernels in shallow layers (B1), nnU-Net's auto-detected 2D/3D-cascade pipelines (C1), and "stop Z-interpolation at inference" recommendations. They directly contradict the contribution.

### Continuity loss is intentional

ContLoss was added because **Podocin-mutation labels have inter-slice gaps**; the regulariser forces continuous predictions despite discontinuous ground truth. clDice / boundary loss are the *wrong* remedies here — both preserve label topology, which would re-introduce the gaps. The agent's clDice recommendation is a category error in this context. Whether ContLoss is still earning its place after the patch-level rotation + label-stacking augmentation is unknown; that's what the A2 ablation will measure.

### Eventual removal targets (informing design)

These will be removed in a future round; do not add new dependencies on them:

- `src/train/profiler.py`
- `src/utils/metrics/log/metric_tboard.py` (TensorBoard)
- `src/utils/metrics/log/metric_sql.py` (SQLite)

Existing usage is intact until explicit removal. New modules / tests / config fields must not lean on these.

### Architectural intent: ablation-friendly

The codebase should make it trivial to swap *model × loss × stitching × optimiser × fold* via config. Each axis has its own registry-style dispatch (see post-Phase-6 architecture). Future model additions (custom SwinUNETR for shallow Z-stacks; multi-encoder / self-supervised variants) plug in by writing a `build()` callable and registering it. No factory changes required.

MONAI's `SwinUNETR` is not used — its shifted-window mechanism needs Z-patches deeper than the window size, and `sample_dimension[0]=12` is too thin. A custom Swin variant will go in `src/models/swin_unetr/` when written.

---

## Pass 1: Code-quality overhaul — `9b543ff`

Single commit, +1695 / −3374 lines, 61 files. Five phases:

### Phase 1 — Correctness bugs

Eight confirmed bugs fixed. All had been silently producing wrong outputs.

- **`Pervalence`** (`src/utils/metrics/clfication.py:130`) — `return AP / AP + AN` parsed as `(AP/AP) + AN = 1 + AN`. Added parentheses.
- **PSP double-run** (`src/infer/psp.py:47-50`) — after `pool.starmap(...)` finished, a serial loop re-ran every task. PSP took ~2× as long as needed. Removed the redundant loop.
- **`Snapper.load`** (`src/train/snapper.py:73-92`) — `_model.module.load_state_dict(...)` unconditionally; crashed for non-DP models. Now handles both with `isinstance` + uses `weights_only=True`.
- **Crop augmentation** (`src/data/ds_train.py:438`) — `_channel[z_start:z_length, ...]` used patch-length as slice endpoint. Now `start:start+length`.
- **ContLoss axes** (`src/train/losses/loss_cont.py:7-42`) — softmax was over `dim=-1` (W axis); the depth-adjacency diff sliced along the *class* axis. Fixed both: softmax over class, diff along depth.
- **Morph draw** (`src/infer/morph.py:31`) — `_input[_input > 0] == 255` was a discarded comparison instead of assignment.
- **MetricSQL `resume` column** (`src/utils/metrics/log/metric_sql.py:42`) — INSERT referenced a column the CREATE TABLE didn't define.
- **Confusion matrix** (`src/utils/metrics/clfication.py:45-84`) — replaced the buggy class-0-remap trick + nested Python loops with a single vectorised `scatter_add_`. Matrix indexed `[true, pred]` consistently.

### Phase 2 — Dead-code purge + SLURM reliability

- Deleted 12 files: `loss_cont_.py`, `ds_train_ex.py`, three `morph_algorithm{2,3,31}.py` variants, five single-use scripts, root `blender.py`, empty `src/models/sunetr/`.
- Removed dead "Cylindrical removal" block in `psp.py`.
- Replaced interactive `input()` overwrite/delete prompts with `-f/--force` flags (SLURM-safe; would have hung under batch submission).
- Fixed missing `roi.sbatch` reference in `gbm_inference.sh:119` (ROI was removed in commit `dd5895c` but the pipeline still tried to submit the job).
- `assert False` → `NotImplementedError` so it survives `python -O`.

### Phase 3 — Architectural cleanup

- **`src/utils/misc.py` split** from 49 KB / 1196 lines down to ~10 KB. Moved stats + visualisation helpers to new `src/infer/stats.py`. Moved blender + export helpers to new `src/infer/blender_io.py`. Removed an exact-duplicate ~35-line block inside `save_top_down_view_aspect_ratio`. Removed dead `_recursive_roi_analysis` (ROI feature was removed).
- **PyTorch modernisation**: `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler('cuda')`. Removed deprecated `ReduceLROnPlateau(verbose=True)`.
- **Reproducibility seeding**: `train.py` now seeds Python `random`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and per-DataLoader-worker via `_worker_init_fn`. `random_split` takes a seeded generator. The experiment-create step persists `git_sha.txt` + `git_diff.patch` alongside `requirements.txt`.
- **Config-ified magic numbers**: `1400` thickness clip → `configs.inference.morph.thickness_clip_max`; the hardcoded `scale_factor=6` override in `factory.createInferer` (which was shadowing the config) now reads the config; the `_multipier`/`_multiplier` typo fixed.

### Phase 4 — Test suite

Set up pytest, added 31 regression tests pinning every Phase 1 fix. Coverage on priority modules: `loss_dice` 100%, `metric_sql` 100%, `loss_cont` 97%, `loss_iou` 92%, `snapper` 89%, `unet3d` 88%, `psp` 79%, `clfication` 71%. CI workflow deferred (no existing CI infrastructure to integrate with).

### Phase 5 — Quality of life

- `gbm.py` refactored from an 11-branch `if`-chain into a small `HANDLERS` dispatch table.
- Class-level docstrings added to Factory, Trainer, Snapper, Inference, Morph, PSP, dataset classes.
- `ruff` adopted as the linter (replaces `.pylintrc`). Config in `pyproject.toml`. Initial sweep applied 32 safe autofixes (import sorting, redundant `class Foo()` → `class Foo`, `Tuple` → `tuple`). `E241` suppressed for hand-aligned dict literals (per user preference; `gbm.py:HANDLERS` is the canonical case).
- Kept the leading-underscore parameter convention (`def foo(_name)`) — repo policy, not changed.

### Deferred from Pass 1

- **DDP scaffolding** — adding optional Distributed Data Parallel touches snapper, inference, factory, sbatch launch. Not core to the methodology questions; deferred. Single-GPU `DataParallel` works.

---

## Pass 2: Methodology review — DL-research-scientist agent

A second agent reviewed the methodology (architecture choices, training protocol, inference strategy, loss design, PSF physics, reproducibility). 25 items catalogued (A1–G1); each was discussed conversationally with the user before deciding. Below is the decision matrix.

### Decisions

| ID | Item | Decision | Rationale |
|---|---|---|---|
| **A1** | Subject-wise stratified 5-fold CV | **ACCEPT** | `random_split(95/5)` on patches leaks ~92% of voxels between train/val; reported Dice is optimistic. Stratify by `detect_group_type()` (Control/Podocin/Collagen). |
| **A2** | ContLoss ablation (5 configs) | **INVESTIGATE** | Gated on A1. Test current ContLoss vs Dice-only vs Dice+CE vs fixed-weight continuity vs TV-on-hard-mask. Podocin subgroup reported separately. |
| **A2 precursor** | Drop learned α/β | **ACCEPT** | Independent of ablation outcome. Sigmoid-renormalise-and-floor is opaque; replace with fixed weight. |
| **A3** | Config-driven inference stitching (4 modes) | **ACCEPT** | Default `gaussian` (nnU-Net convention); also `hann`, `sum_logits` (legacy), `flat_softmax` (diagnostic). |
| **B1** | Anisotropic kernels + stop Z-interp | **REJECT** | Directly conflicts with the selling point (model learns Z patterns at the upsampled grid). |
| **B1.2** | Model-dispatch architecture | **ACCEPT** | Registry pattern for future custom SwinUNETR / variants. |
| **B2** | Class-imbalance loss (focal/Tversky/Lovász) | **DEFER** | Largely subsumed by A2's `Dice-only` arm; revisit if needed after results. |
| **B3.1** | Verify PSF constants empirically | **NO-ACTION** | User confirmed 149/434 nm were measured with sub-resolution beads on the actual microscope. |
| **B3.2** | Log PSF clamp activation rate | **ACCEPT** | Cheap diagnostic; thin GBMs were silently clamped to 0. |
| **B4** | Richardson-Lucy deconvolution preprocessing | **REJECT** | User does not want to pivot the methodology pipeline. |
| **C1** | Adopt nnU-Net wholesale | **REJECT** | Conflicts with the selling point (forces anisotropic strategies). |
| **C1.1** | SGD+momentum+poly-decay schedule | **ACCEPT** | Cherry-picked nnU-Net default; doesn't conflict with the contribution. |
| **C1.2** | Deep supervision auxiliary losses | **ACCEPT** | Aux heads at decoder mid-levels; improves training stability. |
| **C1.3** | Test-time augmentation (mirror flips) | **DEFER** | 8× inference cost for ~0.5–1 Dice; not core. |
| **C1.4** | Foreground-balanced patch sampling | **DEFER** | Couples with A2; revisit after A2 results. |
| **D1** | Channel-input ablation | **DEFER** | Biological argument suffices; revisit if a reviewer challenges. |
| **D2** | Clamp logging | (subsumed by B3.2) | — |
| **E1** | Deterministic algorithms + CUBLAS workspace | **ACCEPT** | Bit-exact reproducibility; ~5–10% speed cost accepted. |
| **E2** | W&B experiment tracking (free tier) | **ACCEPT** | Primary cross-run comparison. Local layout (yaml + snapshots) preserved. |
| **E3** | OME-Zarr | **REJECT** | Current TIFF format works; conversion cost not worth it. |
| **E4** | Per-snapshot model cards | **ACCEPT** | Sibling `.yaml` next to every `.pt`. |
| **F1** | Keep 3-channel input | **NO-CHANGE** | Biologically defensible (nephrin / col4 / WGA sandwich). |
| **F2** | Keep PSF formula structure | **NO-CHANGE** | Correct first-order physics; only constants need verification (B3.1 ✓). |
| **F3** | Keep current seeding | **NO-CHANGE** | E1 layers on top. |
| **G1** | DDP scaffolding | **REJECT** | DP works; complexity not justified. |

### Phase 6 — Pass 2 plumbing (done)

Five commits on top of `9b543ff`, each with its own regression-test suite.

| Commit | Item | What |
|---|---|---|
| `67a03bc` | B3.2 | PSF clamp activation logged; sibling `psf_clamp_stats.yaml` per sample; aggregate section in `metadata.txt`. |
| `9b66365` | E1 | `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`; cudnn_benchmark warns and skips when deterministic mode is on. |
| `b563ea1` | E4 | `Snapper.save` writes a sibling `.yaml` per snapshot capturing torch/CUDA/python versions, GPU name, git SHA, experiment name. |
| `51aa1f4` | A3 | Sliding-window accumulator extracted to `src/infer/stitching.py:StitchAccumulator`; 4 configurable modes via `inference.stitching`. Unet3D loses inference-mode state. `main_infer` builds the model once (no longer per-volume). |
| `32b1c14` | B1.2 | `src/models/__init__.py:MODEL_REGISTRY` + `build_model(name, configs, in_ch, num_classes)`. Each model in its own subdir with a `build()` callable. Trainer is now fully model-agnostic. |

### Phase 7 — Training-methodology changes (done)

Five commits, each gated on its own regression-test suite. Full test suite reaches 106 at end of Phase 7.

| Commit | Item | What |
|---|---|---|
| `3393322` | C1.1 | SGD+momentum=0.99+poly-decay added alongside Adam+ReduceLROnPlateau. Trainer dispatches `.step()` by scheduler type. Default backwards-compat to ReduceLROnPlateau when the new `trainer.scheduler` block is missing. |
| `b3bb5b2` | A2-precursor | ContLoss drops the learned-α/β sigmoid-renormalise-with-floors machinery. Replaced with fixed `cont_alpha` / `cont_beta` HP fields (defaults 0.7 / 0.3, approximating the prior init-time equilibrium). |
| `b915b58` | C1.2 | Deep supervision auxiliary heads at decoder mid-levels in `Unet3D`; new `DeepSupervisionLoss` wrapper sums per-level losses against downsampled labels with `0.5^k` weights. Wrapper degrades transparently when the model produces a single tensor. |
| `e123eb0` | A1 | Subject-wise stratified k-fold CV. New `src/data/folds.py` with hand-rolled stratified round-robin (no sklearn dep). `--fold 0..4` CLI flag; `fold_assignments.yaml` persisted per experiment; fail-loud guard for unknown sample-name prefixes. Replaces the leaky patch-level `random_split`. |
| `ce2170f` | E2 | W&B integration. New `MetricWandb` backend alongside the existing tensorboard/SQL ones. `wandb.init(config=configs)` at training start so every ablation axis is filterable on the UI; snapshot `.pt` files uploaded as artifacts via `wandb.save`. Lazy import; missing wandb logs a warning rather than crashing. |

### Phase 8 — Ablation orchestrator (done)

Two commits; tests reach 134 at end of Phase 8.

| Commit | Item | What |
|---|---|---|
| `e5ab318` | Orchestrator | `gbm.py ablate <spec.yaml>` reads a YAML spec, materialises one experiment per `(cell, fold)` cross-product under `experiments.root`, and prints the exact training commands to submit. Cells symlink `datasets/` + `code/` from the base experiment (no GB-scale copies) and inherit `fold_assignments.yaml`; only the rewritten `configs.yaml` carries the per-cell overrides + a unique `wandb.run_name`. Idempotent: re-running over an existing cell directory leaves it alone. Three example specs in `ablation_specs/` (loss / optim / DS pilots). |
| `ab0ef8c` | CompoundLoss | Weighted sum of base losses (`Dice + CrossEntropy`, `Dice + ContLoss`, …) so the catalog's literal A2 ablation configs (b)–(d) can be expressed in YAML. `_build_single_loss` extracted as the factory's name→class helper so adding a new sub-loss class touches one place. Existing `DeepSupervisionLoss` wrapper composes on top transparently. |

---

## What's still open

- **A2 (e) TV-on-argmaxed-mask loss** — the 5th A2 configuration needs a new loss class operating on `argmax(softmax(logits))` rather than softmax probabilities. ~1 hour of work; not strictly required to start running the (a)–(d) ablation.
- **Custom SwinUNETR for shallow Z-stacks** — B1.2's model registry has a slot for it. The actual implementation is an independent piece of research work.
- **DDP scaffolding** — DataParallel works today; revisit if single-experiment throughput becomes a bottleneck on 4× A100.
- **Remove profiler / TensorBoard / SQLite** — slated for removal once the W&B logging path is in active use. New code already avoids depending on them.
- **A2 ablation results will dictate** whether to revisit B2 (focal/Tversky), C1.3 (TTA), C1.4 (foreground-balanced sampling), or D1 (channel ablation). All four are parked in the DEFER bucket until the loss ablation lands a winner.

## How to run an ablation

```bash
# 1. Create a baseline experiment with the current `gbm.py create`.
python gbm.py create my_baseline --batch-size 8

# 2. Verify fold_assignments.yaml landed in <experiments.root>/my_baseline/.
# 3. Pick an ablation study, e.g. the loss pilot:
python gbm.py ablate ablation_specs/loss_pilot.yaml

# 4. The orchestrator prints commands like:
#    python gbm.py train loss_pilot__dice__fold0 --fold 0
#    ...
# 5. Submit each command (interactively, or wrap in sbatch yourself).
# 6. Compare runs on W&B (filtering by `cell` or `fold`) once results land.
```
