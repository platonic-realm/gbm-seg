# Python Imprts
from __future__ import annotations

import copy
import logging
import os
import random
import re
import warnings
from pathlib import Path

# Must be set before `import torch` (cuBLAS reads it once at first CUDA init).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Library Imports
import numpy as np
import torch
import yaml

from src.data.folds import read_assignments
from src.train.distributed import (
    cleanup_ddp,
    ddp_launchable,
    ddp_requested,
    init_ddp,
    is_distributed,
    is_main_process,
)
from src.train.factory import Factory

# Local Imports
from src.utils import args
from src.utils.misc import configure_logger, summerize_configs

SEED = 88233474


def seed_everything(_seed: int):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    # Force deterministic CUDA / cuDNN kernels. Small speed regression
    # (~5-10%) in exchange for bit-exact reproducibility across runs.
    # `warn_only=True` so ops without a deterministic implementation log
    # a warning rather than raise — flip to False for strict mode.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Silence the routine "does not have a deterministic implementation"
    # warnings that fire on every loss / pooling step (nll_loss2d,
    # max_pool3d_backward, memory-efficient attention, …). We keep
    # `warn_only=True` above so a *new* non-deterministic op still fails
    # loudly in strict mode if we ever flip it; this filter only suppresses
    # the chat from the already-known ones.
    warnings.filterwarnings(
        "ignore",
        message=r".*does not have a deterministic implementation.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*defaults to a non-deterministic algorithm.*",
        category=UserWarning,
    )


def maybe_init_wandb(_configs, _fold: int = 0,
                     _resume_run_id: str | None = None,
                     _all_data: bool = False) -> bool:
    """Initialise a W&B run when ``configs.trainer.logging.wandb.enabled`` is true.

    The full ``configs`` dict is logged as ``wandb.config`` so every axis
    of the eventual A2 ablation (model.name × loss × inference.stitching
    × optim.name × fold) is filterable on the W&B UI.

    When ``_resume_run_id`` is provided (returned by ``Snapper.load`` on
    a resumed run), the W&B run is reattached via
    ``wandb.init(resume="must", id=...)`` instead of starting a fresh
    one — so the metric charts continue from where they left off rather
    than appearing as a separate run.

    ``_all_data=True`` marks an all-data (no-fold) final training: the run
    is named ``<exp>-all_data`` and tagged ``all_data`` instead of
    ``fold-N``, so Stage-B finals are visually distinct from the
    Stage-A CV folds even within the same W&B project.

    Returns True iff a run was successfully initialised; False otherwise
    (wandb missing, disabled, or init failed). Failure is logged but
    does not abort training.
    """
    wandb_cfg = _configs['trainer'].get('logging', {}).get('wandb', {})
    if not wandb_cfg.get('enabled', False):
        return False
    # Only rank-0 opens a W&B run under DDP. The other ranks would
    # otherwise create 4 separate runs with the same name and overwrite
    # each other's history.
    if not is_main_process():
        return False
    try:
        import wandb
    except ImportError:
        logging.warning(
            "trainer.wandb.enabled=True but wandb is not installed; "
            "skipping W&B logging. `pip install wandb` to enable.")
        return False

    # Default the W&B `group` to the experiment-directory name so single-fold
    # sbatch invocations end up under the same dashboard group the all-folds
    # orchestrator uses. Without this, each per-fold sbatch run gets a
    # whimsical wandb-auto name like "avid-energy-3" with no shared group.
    exp_name = Path(_configs.get('root_path', '.') or '.').name or 'experiment'
    # Strip the static `__foldN` suffix the ablation runner bakes into cell
    # dir names, so the W&B group is the cell itself (all 5 folds land in one
    # group) rather than carrying a misleading fold-0 label.
    exp_name = re.sub(r'[-_]+fold\d+$', '', exp_name)
    if not wandb_cfg.get('group'):
        wandb_cfg['group'] = exp_name

    # Cluster name (lyn / ramses / …) so runs from different clusters are
    # distinguishable on the shared W&B project. SLURM_CLUSTER_NAME is the
    # natural source, but some clusters export it literally as "(null)", so
    # fall back to the alphabetic prefix of the node name (lyn-gpu-06 → lyn,
    # ramses16301 → ramses) — works regardless of the node naming scheme.
    cluster = os.environ.get('SLURM_CLUSTER_NAME', '') or ''
    if cluster in ('', '(null)', 'N/A'):
        node = os.environ.get('SLURMD_NODENAME') or os.uname().nodename
        m = re.match(r'[A-Za-z]+', node)
        cluster = m.group(0).lower() if m else 'local'

    run_config = {**_configs,
                  'fold': ('all_data' if _all_data else int(_fold)),
                  'cluster': cluster}
    explicit_name = wandb_cfg.get('run_name')

    # Surface compile state in the run name and as a tag so torch.compile vs
    # eager runs are visually distinguishable on the W&B dashboard. Other
    # axes (model.name, loss, stitching, optimiser) are already filterable
    # via wandb.config — only these two surface as first-class UI labels.
    compile_on = bool(
        _configs['trainer'].get('runtime', {}).get('compile', False))
    compile_suffix = '-compile' if compile_on else '-eager'
    split_tag = 'all_data' if _all_data else f"fold-{int(_fold)}"
    tags = ['compile' if compile_on else 'eager', split_tag, cluster]

    # Run name = <cluster>-<base>-<fold-N><compile>. `base` is the explicit
    # run_name (with any static cell fold suffix stripped) or the group — so
    # the name ALWAYS carries the cluster and the ACTUAL fold, even when the
    # ablation runner pins run_name to the cell dir name (which ends in
    # `_fold0`). Previously an explicit run_name bypassed the fold suffix
    # entirely, mislabelling every fold as fold0.
    base = explicit_name if explicit_name else wandb_cfg['group']
    base = re.sub(r'[-_]+fold\d+$', '', base)
    auto_name = f"{cluster}-{base}-{split_tag}{compile_suffix}"
    init_kwargs = dict(
        project=wandb_cfg.get('project', 'gbm-seg'),
        entity=wandb_cfg.get('entity'),
        name=auto_name,
        group=wandb_cfg.get('group'),
        job_type='train',
        tags=tags,
        config=run_config,
    )
    if _resume_run_id:
        # `resume="must"` raises if the run id can't be found, which is
        # the right failure mode — silently starting a fresh run would
        # split metric history across two W&B runs.
        init_kwargs['id'] = _resume_run_id
        init_kwargs['resume'] = 'must'

    try:
        wandb.init(**init_kwargs)
        # Tell W&B that the chart x-axis for train/* and valid/* metrics
        # is `samples` (single-sample data units processed by the model),
        # NOT W&B's default internal step. The trainer logs samples
        # = optimiser_step × effective_batch_size so curves from runs
        # with different batch sizes line up directly on the UI.
        wandb.define_metric("samples")
        wandb.define_metric("train/*", step_metric="samples")
        wandb.define_metric("valid/*", step_metric="samples")
        if _resume_run_id:
            logging.info("W&B run resumed: id=%s", _resume_run_id)
        return True
    except Exception as exc:  # pragma: no cover — wandb init is networked
        logging.warning("W&B init failed (%s); continuing without it.", exc)
        return False


def main_train(_configs, _fold: int = 0):
    """Run one training fold and return its best validation metrics.

    A1: training and validation are partitioned subject-wise by
    ``<experiment>/fold_assignments.yaml``. ``_fold`` (0..k-1) selects
    which fold to train. The held-out test cohort (``ds_test/``) is
    untouched.

    Returns a dict with the best-validation metric snapshot (Dice and
    every other metric tracked, plus the epoch/step that achieved it),
    or ``None`` if no validation cycle ran. Persists the same dict to
    ``<exp>/results-train/fold_{N}/best_metrics.yaml``. Also logs the
    best as W&B run summary if W&B is enabled.
    """
    # Namespace per-fold outputs so 5-fold CV trains five independent models
    # without overwriting each other. Snapper.load() would otherwise pick up
    # fold N-1's latest snapshot at fold N's start. Visualization output and
    # log files get the same treatment so artefacts stay separable.
    fold_tag = f"fold_{_fold}"
    trainer_logging = _configs['trainer']['logging']
    trainer_logging['snapshot_path'] = os.path.join(
        trainer_logging['snapshot_path'].rstrip('/'), fold_tag)
    trainer_logging['visualization']['path'] = os.path.join(
        trainer_logging['visualization']['path'].rstrip('/'), fold_tag)
    _configs['logging']['log_file'] = _configs['logging']['log_file'].replace(
        '.log', f'.{fold_tag}.log')

    if _configs['logging']['log_summary'] and is_main_process():
        summerize_configs(_configs)

    # DDP bring-up. No-ops in single-process mode; required *before* model
    # construction so each rank pins its own cuda:LOCAL_RANK device.
    if ddp_requested(_configs) and not ddp_launchable():
        raise RuntimeError(
            "trainer.ddp=True requires the torchrun env vars "
            "(LOCAL_RANK/RANK/WORLD_SIZE). Submit via "
            "`sbatch/train_ddp.sbatch` (which wraps `torchrun "
            "--nproc-per-node=N gbm.py train …`).")
    init_ddp(_configs)

    seed_everything(SEED)

    if _configs['trainer']['runtime']['cudnn_benchmark']:
        # Benchmark mode picks the fastest cuDNN kernel based on input shapes
        # — non-deterministic but a real throughput win when sample_dim is
        # fixed (it is, in our patch-based training). Under DDP we accept
        # the determinism trade-off in exchange for speed. Outside DDP the
        # determinism guarantee from seed_everything wins.
        if is_distributed():
            torch.backends.cudnn.benchmark = True
            # Disable strict determinism so the benchmark kernels are usable.
            torch.use_deterministic_algorithms(False)
            if is_main_process():
                logging.info("DDP: cudnn.benchmark enabled (non-deterministic)")
        else:
            logging.warning(
                "trainer.cudnn_benchmark=True is incompatible with "
                "deterministic algorithms in single-process mode; skipping "
                "cuDNN benchmarking.")

    factory = Factory(_configs)

    # A1: subject-wise stratified 5-fold CV. Resolve which TIFFs belong to
    # the chosen fold's train and validation sets.
    fold_assignments = read_assignments(_configs['root_path'])
    if not 0 <= _fold < len(fold_assignments):
        raise ValueError(
            f"--fold {_fold} out of range; experiment has "
            f"{len(fold_assignments)} folds.")
    fold = fold_assignments[_fold]
    logging.info(
        "Fold %d: %d training subjects, %d validation subjects "
        "(validation groups: %s).",
        _fold, len(fold['train']), len(fold['valid']),
        fold.get('groups_in_valid', '?'))

    train_dataset = factory.createTrainDataset(file_filter=fold['train'])
    valid_dataset = factory.createValidDataset(file_filter=fold['valid'])

    valid_dataloader = factory.createValidDataLoader(valid_dataset)
    train_dataloader = factory.createTrainDataLoader(train_dataset)

    model = factory.createModel(train_dataset.getNumberOfChannels(),
                                train_dataset.getNumberOfClasses())
    loss_function = factory.createLoss()
    optimizer = factory.createOptimizer(model, loss_function)
    lr_scheduler = factory.createScheduler(optimizer)

    stepper = factory.createStepper(model, optimizer, loss_function)
    snapper = factory.createSnapper()

    # Resume from snapshot if `<snapshot_path>/continue/*.pt` is present.
    # Restores model + optimiser + scheduler + stepper (incl. AMP scaler)
    # + RNGs in-place. Returns the resume info (epoch/step/best/wandb_id)
    # or None for a fresh run. snapper.load is fold-isolated because
    # snapshot_path is already mutated to include `fold_N/` above.
    device = _configs['trainer']['runtime']['device']
    resume = snapper.load(model,
                          _device=device,
                          _stepper=stepper,
                          _optimizer=optimizer,
                          _scheduler=lr_scheduler)
    if resume is not None:
        logging.info(
            "Resuming fold %d from snapshot: epoch=%d step=%d "
            "best_dice=%s wandb_id=%s",
            _fold, resume['epoch'], resume['step'],
            (None if resume['best_metrics'] is None
             else round(float(resume['best_metrics'].get('Dice', 0)), 4)),
            resume['wandb_run_id'])

    # W&B comes AFTER snapper.load so we can reattach to the original
    # run via resume="must" when a run id was persisted in the snapshot.
    wandb_active = maybe_init_wandb(
        _configs, _fold=_fold,
        _resume_run_id=(resume['wandb_run_id'] if resume else None))

    visualizer = factory.createVisualizer()
    metric_logger = factory.createMetricLogger()

    # Saved EPOCH is the epoch that was *in progress* (or that just
    # finished) when the snapshot was written. To avoid re-iterating
    # work already done, the resumed run starts at saved_epoch + 1.
    starting_epoch = (resume['epoch'] + 1) if resume else 0

    trainer = factory.createTrainer(
        model,
        loss_function,
        stepper,
        snapper,
        visualizer,
        metric_logger,
        lr_scheduler,
        train_dataloader,
        valid_dataloader,
        train_dataset.getNumberOfClasses(),
        _starting_epoch=starting_epoch,
        _starting_best_metrics=(resume['best_metrics'] if resume else None),
        _starting_best_epoch=(resume['best_epoch'] if resume else None),
        _starting_best_step=(resume['best_step'] if resume else None),
    )

    try:
        trainer.train()
    finally:
        # Always tear the process group down so a follow-up job doesn't
        # inherit a dangling NCCL state.
        cleanup_ddp()

    best = trainer.best_metrics()

    # Persist per-fold best + close W&B from rank-0 only. The other ranks
    # have the same `best` (they all-reduced metrics inside the trainer)
    # but only one process should touch the filesystem and the W&B run.
    if is_main_process():
        if best is not None:
            out_dir = (Path(_configs['root_path']) / 'results-train'
                       / f'fold_{_fold}')
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / 'best_metrics.yaml', 'w', encoding='UTF-8') as f:
                yaml.safe_dump(best, f, sort_keys=False)
            logging.info("Fold %d best validation metrics: %s", _fold, best)

        # E2: log best to W&B run summary, then close cleanly.
        if wandb_active:
            try:
                import wandb
                if best is not None:
                    for k, v in best.items():
                        if isinstance(v, (int, float)):
                            wandb.run.summary[f'best/{k}'] = v
                wandb.finish()
            except ImportError:
                pass

    return best


def main_train_all_data(_configs, _epochs: int | None = None):
    """Train one model on ALL ``ds_train`` subjects — no fold holdout.

    Stage-B final-model training. There is no validation set, hence no
    "best" snapshot: the run trains for a fixed number of epochs and the
    *last* snapshot is the deliverable. ``_epochs`` overrides
    ``configs.trainer.epochs`` when given.

    Outputs are namespaced under ``results-train/{snapshots,…}/all_data/``
    so they never collide with the per-fold CV artefacts. An interrupted
    run resumes from ``…/all_data/continue/*.pt`` like any fold run.
    """
    split_tag = 'all_data'
    trainer_logging = _configs['trainer']['logging']
    trainer_logging['snapshot_path'] = os.path.join(
        trainer_logging['snapshot_path'].rstrip('/'), split_tag)
    trainer_logging['visualization']['path'] = os.path.join(
        trainer_logging['visualization']['path'].rstrip('/'), split_tag)
    _configs['logging']['log_file'] = _configs['logging']['log_file'].replace(
        '.log', f'.{split_tag}.log')

    if _epochs is not None:
        _configs['trainer']['optimization']['epochs'] = int(_epochs)

    if _configs['logging']['log_summary'] and is_main_process():
        summerize_configs(_configs)

    # DDP bring-up — same contract as main_train.
    if ddp_requested(_configs) and not ddp_launchable():
        raise RuntimeError(
            "trainer.ddp=True requires the torchrun env vars "
            "(LOCAL_RANK/RANK/WORLD_SIZE). Submit via "
            "`sbatch/train_ddp.sbatch` (which wraps `torchrun "
            "--nproc-per-node=N gbm.py train …`).")
    init_ddp(_configs)

    seed_everything(SEED)

    if _configs['trainer']['runtime']['cudnn_benchmark']:
        if is_distributed():
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)
            if is_main_process():
                logging.info(
                    "DDP: cudnn.benchmark enabled (non-deterministic)")
        else:
            logging.warning(
                "trainer.cudnn_benchmark=True is incompatible with "
                "deterministic algorithms in single-process mode; skipping "
                "cuDNN benchmarking.")

    factory = Factory(_configs)

    # All-data: every ds_train subject trains. file_filter=None → "all".
    train_dataset = factory.createTrainDataset(file_filter=None)
    train_dataloader = factory.createTrainDataLoader(train_dataset)
    n_subjects = len(getattr(train_dataset, 'image_list', []) or [])
    logging.info("All-data training: %d training subjects, no holdout.",
                 n_subjects)

    model = factory.createModel(train_dataset.getNumberOfChannels(),
                                train_dataset.getNumberOfClasses())
    loss_function = factory.createLoss()
    optimizer = factory.createOptimizer(model, loss_function)
    lr_scheduler = factory.createScheduler(optimizer)

    stepper = factory.createStepper(model, optimizer, loss_function)
    snapper = factory.createSnapper()

    device = _configs['trainer']['runtime']['device']
    resume = snapper.load(model,
                          _device=device,
                          _stepper=stepper,
                          _optimizer=optimizer,
                          _scheduler=lr_scheduler)
    if resume is not None:
        logging.info(
            "Resuming all-data training from snapshot: epoch=%d step=%d "
            "wandb_id=%s",
            resume['epoch'], resume['step'], resume['wandb_run_id'])

    wandb_active = maybe_init_wandb(
        _configs,
        _resume_run_id=(resume['wandb_run_id'] if resume else None),
        _all_data=True)

    visualizer = factory.createVisualizer()
    metric_logger = factory.createMetricLogger()

    starting_epoch = (resume['epoch'] + 1) if resume else 0

    # validation_loader=None → the trainer skips the validation cycle and
    # steps the epoch-driven scheduler (poly_decay) once per epoch instead.
    trainer = factory.createTrainer(
        model,
        loss_function,
        stepper,
        snapper,
        visualizer,
        metric_logger,
        lr_scheduler,
        train_dataloader,
        None,
        train_dataset.getNumberOfClasses(),
        _starting_epoch=starting_epoch,
    )

    try:
        trainer.train()
    finally:
        cleanup_ddp()

    if is_main_process():
        logging.info(
            "All-data training complete (%d epochs). The final snapshot is "
            "the deliverable — there is no best-validation selection.",
            _configs['trainer']['optimization']['epochs'])
        if wandb_active:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass


def _aggregate_cv(per_fold: list) -> dict:
    """Mean/std/min/max/n across folds for every numeric metric."""
    if not per_fold:
        return {}
    skip = {'fold', 'best_epoch', 'best_step'}
    metric_names = [k for k in per_fold[0]
                    if k not in skip
                    and isinstance(per_fold[0][k], (int, float))]
    out = {}
    for name in metric_names:
        vals = np.array([d[name] for d in per_fold if d.get(name) is not None],
                        dtype=float)
        if vals.size == 0:
            continue
        out[name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n': int(vals.size),
        }
    return out


def _read_per_fold_best(_root_path: str, _k: int) -> list:
    """Load every fold's ``best_metrics.yaml`` from disk into a list of
    dicts (one per fold, sorted by fold id). Missing files become a
    placeholder ``{'fold': N}`` so the aggregator can still produce a
    summary with the available folds — useful when an sbatch job died
    mid-fold and the user wants to inspect the survivors.
    """
    out = []
    root = Path(_root_path)
    for fold_id in range(_k):
        path = root / 'results-train' / f'fold_{fold_id}' / 'best_metrics.yaml'
        entry = {'fold': fold_id}
        if path.exists():
            try:
                with open(path, encoding='UTF-8') as f:
                    entry.update(yaml.safe_load(f) or {})
            except Exception as exc:
                logging.warning("Could not read %s (%s); skipping.", path, exc)
        else:
            logging.warning("Missing %s; fold %d will be excluded from "
                            "the aggregate.", path, fold_id)
        out.append(entry)
    return out


def consolidate_cv_results(_per_fold: list,
                           _configs: dict,
                           _exp_name: str = None) -> dict:
    """Persist per-fold + aggregate metrics to ``cv_results.{yaml,npz}``
    and optionally open a W&B ``cv_summary`` run.

    Shared between the in-process orchestrator (``main_train_all_folds``)
    and the standalone ``gbm.py aggregate-cv`` command, so a partially-
    completed sbatch fan-out can still be summarised after the fact.
    """
    root_path = _configs['root_path']
    if _exp_name is None:
        _exp_name = Path(root_path).name or 'experiment'
    aggregate = _aggregate_cv(_per_fold)
    out = {
        'k': len(_per_fold),
        'experiment': _exp_name,
        'per_fold': _per_fold,
        'aggregate': aggregate,
    }
    yaml_path = Path(root_path) / 'cv_results.yaml'
    with open(yaml_path, 'w', encoding='UTF-8') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    logging.info("CV results written to %s", yaml_path)

    # Raw arrays for plotting; one key per numeric metric.
    if _per_fold:
        skip = {'fold', 'best_epoch', 'best_step'}
        arrays = {}
        for name in _per_fold[0]:
            if name in skip or not isinstance(_per_fold[0][name],
                                              (int, float)):
                continue
            arrays[name] = np.array(
                [d.get(name, np.nan) for d in _per_fold], dtype=float)
        if arrays:
            npz_path = Path(root_path) / 'cv_results.npz'
            np.savez_compressed(npz_path, **arrays)
            logging.info("CV per-fold arrays written to %s", npz_path)

    wandb_cfg = _configs['trainer'].get('logging', {}).get('wandb', {}) or {}
    if wandb_cfg.get('enabled', False):
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get('project', 'gbm-seg'),
                entity=wandb_cfg.get('entity'),
                name=f"{_exp_name}-cv-summary",
                group=wandb_cfg.get('group', _exp_name),
                job_type='cv_summary',
                config={'k': out['k'], 'experiment': _exp_name},
            )
            for metric, stats in aggregate.items():
                for stat_name, val in stats.items():
                    wandb.run.summary[f'cv/{metric}/{stat_name}'] = val
            try:
                wandb.save(str(yaml_path), policy='now')
            except Exception:  # pragma: no cover — wandb sync edge cases
                pass
            wandb.finish()
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover — networked
            logging.warning("Could not log CV summary to W&B: %s", exc)

    return out


def main_train_all_folds(_configs):
    """Run every fold in ``fold_assignments.yaml`` sequentially.

    Each fold is launched via :func:`main_train` (which already namespaces
    snapshots/visuals/logs by fold). Best-validation metrics from every
    fold are collected, aggregated (mean/std/min/max), and persisted:

    * ``<exp>/results-train/fold_{N}/best_metrics.yaml`` — per fold.
    * ``<exp>/cv_results.yaml`` — per-fold + aggregate, human-readable.
    * ``<exp>/cv_results.npz`` — raw per-fold arrays for plotting.

    When W&B is enabled, each fold runs as its own W&B run with
    ``group=<exp_name>``; an additional ``job_type='cv_summary'`` run is
    opened at the end with the aggregate stats so the dashboard has a
    single row for the whole CV.

    Returns the same dict that gets written to ``cv_results.yaml``.
    """
    if ddp_requested(_configs):
        raise RuntimeError(
            "trainer.ddp=True is incompatible with the in-process all-folds "
            "orchestrator: each fold must be its own torchrun launch. "
            "Submit per-fold via `sbatch sbatch/train_ddp.sbatch <exp> <fold>` "
            "(or use sbatch/submit_cv_ddp.sh), then aggregate with "
            "`gbm.py aggregate-cv <exp>` once all folds finish.")
    root_path = _configs['root_path']
    fold_assignments = read_assignments(root_path)
    k = len(fold_assignments)
    exp_name = Path(root_path).name or 'experiment'

    # Set a stable W&B group so per-fold runs show up together.
    trainer_logging = _configs['trainer'].setdefault('logging', {})
    wandb_cfg = trainer_logging.get('wandb', {}) or {}
    wandb_cfg.setdefault('group', exp_name)
    trainer_logging['wandb'] = wandb_cfg

    per_fold = []
    for fold_id in range(k):
        logging.info("===== Fold %d / %d starting =====", fold_id, k - 1)
        cfg = copy.deepcopy(_configs)
        best = main_train(cfg, _fold=fold_id)
        per_fold.append({'fold': fold_id, **(best or {})})
        logging.info("===== Fold %d / %d done =====", fold_id, k - 1)

    return consolidate_cv_results(per_fold, _configs, exp_name)


def aggregate_cv_from_disk(_configs):
    """Standalone aggregation: read each fold's best_metrics.yaml from
    disk and produce the consolidated cv_results.{yaml,npz} + W&B summary.

    Used by ``gbm.py aggregate-cv``. Sbatch fan-out submits 5 per-fold
    ``gbm.py train EXP --fold N`` jobs and a 6th dependent
    ``gbm.py aggregate-cv EXP`` job that calls this.
    """
    root_path = _configs['root_path']
    fold_assignments = read_assignments(root_path)
    k = len(fold_assignments)
    exp_name = Path(root_path).name or 'experiment'

    trainer_logging = _configs['trainer'].setdefault('logging', {})
    wandb_cfg = trainer_logging.get('wandb', {}) or {}
    wandb_cfg.setdefault('group', exp_name)
    trainer_logging['wandb'] = wandb_cfg

    per_fold = _read_per_fold_best(root_path, k)
    return consolidate_cv_results(per_fold, _configs, exp_name)


if __name__ == '__main__':
    _, configs = args.parse_indep("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    main_train(configs)
