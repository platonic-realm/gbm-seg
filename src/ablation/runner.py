"""Materialise ablation cells as experiment directories.

Per (cell, fold) materialisation:
  - mkdir ``<output_root>/<experiment_name>/``
  - symlink ``datasets/`` and ``code/`` from the base experiment
    (avoids copying GB-scale data 25 times for a 25-cell study)
  - copy ``fold_assignments.yaml``, ``requirements.txt``, and
    ``git_sha.txt`` from the base for provenance
  - write a fresh ``configs.yaml`` = base ⨁ cell overrides, with
    ``trainer.logging.wandb.run_name`` set to the experiment name so each
    cell shows up as a separate W&B run

Idempotent: if the cell experiment dir already exists, it's left alone
(we still emit its training command so reruns are explicit).
"""

import logging
import os
import shutil
from pathlib import Path

import yaml

from src.ablation.spec import CellRun, Spec, apply_overrides


def materialise(spec: Spec, experiments_root: Path) -> list[Path]:
    """Create cell directories for every ``(cell, fold)`` in ``spec``.

    ``experiments_root`` is the project's ``experiments.root`` config.
    Both the base experiment and the cells live under it (cells are placed
    directly under ``experiments_root`` so ``gbm.py train`` discovers them
    via its non-recursive name lookup).

    Returns the list of cell experiment paths in submission order.
    """
    experiments_root = Path(experiments_root)
    base_path = experiments_root / spec.base_experiment
    if not base_path.is_dir():
        raise FileNotFoundError(
            f"Base experiment not found: {base_path}. "
            "Create it first with `gbm.py create <name>` before ablating.")

    base_configs_path = base_path / 'configs.yaml'
    if not base_configs_path.is_file():
        raise FileNotFoundError(
            f"Base experiment missing configs.yaml: {base_configs_path}")

    base_configs = yaml.safe_load(base_configs_path.read_text(encoding="UTF-8"))

    paths: list[Path] = []
    for run in spec.expand():
        cell_path = experiments_root / run.experiment_name
        if cell_path.exists():
            logging.info("Cell already exists, leaving as-is: %s", cell_path)
        else:
            _materialise_one(base_path, base_configs, run, cell_path)
        paths.append(cell_path)
    return paths


def _materialise_one(base_path: Path, base_configs: dict,
                     run: CellRun, cell_path: Path) -> None:
    """Create one cell experiment directory."""
    cell_path.mkdir(parents=True)

    # Symlinks for the bulk content. Use absolute paths so the symlinks
    # survive if the cell dir is later moved.
    for name in ('datasets', 'code'):
        src = (base_path / name).resolve()
        if src.exists():
            os.symlink(src, cell_path / name)

    # Copies for the small, per-experiment provenance files.
    for name in ('fold_assignments.yaml', 'requirements.txt', 'git_sha.txt'):
        src = base_path / name
        if src.exists():
            shutil.copy2(src, cell_path / name)

    # Build the cell's configs.yaml.
    cell_configs = apply_overrides(base_configs, run.cell.overrides)
    # Re-root the cell at its OWN directory. base_configs carries the base
    # experiment's root_path, so without this every cell writes its
    # results-train/snapshots into the BASE dir — and sibling cells (e.g.
    # swin__cont vs swin__crossentropy) collide there, overwriting each
    # other's fold_N/best_metrics.yaml. All other path keys are relative to
    # root_path, so this is the only one to rewrite. Trailing os.sep matches
    # create_new_experiment's format.
    cell_configs['root_path'] = str(cell_path) + os.sep
    # Inject the W&B run name so each cell shows up separately on the UI.
    # The logging/wandb blocks may not exist on older base experiments;
    # build defensively.
    logging_block = cell_configs['trainer'].setdefault('logging', {})
    wandb_block = logging_block.setdefault('wandb', {})
    wandb_block.setdefault('enabled', False)
    wandb_block['run_name'] = run.experiment_name
    # Record the parent study + cell + fold for sanity at runtime.
    cell_configs['ablation'] = {
        'study': cell_path.name.split('__')[0],
        'cell': run.cell.name,
        'fold': int(run.fold),
        'overrides': dict(run.cell.overrides),
    }

    (cell_path / 'configs.yaml').write_text(
        yaml.safe_dump(cell_configs, sort_keys=False),
        encoding="UTF-8")

    logging.info("Materialised cell: %s", cell_path)


def emit_commands(spec: Spec, cell_paths: list[Path],
                  *, sbatch_wrapper: str = None) -> list[str]:
    """Build the ``python gbm.py train`` (optionally sbatch-wrapped) commands
    to submit each cell.

    If ``sbatch_wrapper`` is provided, each command is wrapped as
    ``sbatch <wrapper> ...``. If None, plain ``python gbm.py train ...``
    commands are emitted and the caller wraps them in their own SLURM
    submission scheme.
    """
    runs = spec.expand()
    if len(runs) != len(cell_paths):
        raise RuntimeError(
            "Internal: emit_commands received mismatched run vs path counts.")

    commands: list[str] = []
    for run, _cell_path in zip(runs, cell_paths):
        # Cell experiment_name is what `gbm.py train <name>` expects when
        # `experiments.root` points at spec.output_root. Since cells live
        # under spec.output_root rather than the original experiments.root,
        # the user typically overrides experiments.root when submitting (or
        # the cell's configs.yaml carries the right paths already — see
        # the W&B run_name + paths in the materialised configs).
        cmd = f"python gbm.py train {run.experiment_name} --fold {run.fold}"
        if sbatch_wrapper:
            cmd = f"sbatch {sbatch_wrapper} {run.experiment_name} {run.fold}"
        commands.append(cmd)
    return commands
