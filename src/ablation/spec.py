"""Ablation spec parsing and cell expansion.

A spec YAML describes one study:

    study: loss_pilot
    base_experiment: my_baseline
    cells:
      - name: dice
        overrides: {trainer.optimization.loss.name: Dice}
      - name: cont
        overrides:
          trainer.optimization.loss.name: Cont
          trainer.optimization.loss.cont_alpha: 0.7
    folds: [0]                     # or [0, 1, 2, 3, 4] for the final pass

Each ``(cell, fold)`` pair becomes one experiment named
``<study>__<cell.name>__fold<n>`` placed directly under the project's
``experiments.root`` (resolved from ``configs/template.yaml`` at orchestrator
launch time, so each cell is discoverable by ``gbm.py train <cell_name>``).
``overrides`` keys use dotted-paths into the configs dict
(``trainer.optimization.loss.name`` ≡
``configs['trainer']['optimization']['loss']['name']``); values are written
verbatim.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Cell:
    """One row of the ablation: a name + a flat dict of dotted-path overrides."""
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class CellRun:
    """One materialised experiment to run: a Cell × a fold."""
    cell: Cell
    fold: int
    experiment_name: str

    @property
    def wandb_run_name(self) -> str:
        return self.experiment_name


@dataclass(frozen=True)
class Spec:
    study: str
    base_experiment: str
    cells: list[Cell]
    folds: list[int]

    def expand(self) -> list[CellRun]:
        """Full list of (cell, fold) materialisations in submission order."""
        runs: list[CellRun] = []
        for cell in self.cells:
            for fold in self.folds:
                name = f"{self.study}__{cell.name}__fold{fold}"
                runs.append(CellRun(cell=cell, fold=fold, experiment_name=name))
        return runs


_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')


def _validate_name(value: str, what: str) -> None:
    if not value:
        raise ValueError(f"{what} must be non-empty.")
    if not _NAME_RE.match(value):
        raise ValueError(
            f"{what}={value!r} contains characters other than [A-Za-z0-9._-].")


def parse_spec(spec_path) -> Spec:
    """Read and validate an ablation spec YAML.

    Required top-level keys: study, base_experiment, output_root, cells, folds.
    Each cell needs a name and an overrides dict (overrides may be empty).
    """
    raw = yaml.safe_load(Path(spec_path).read_text(encoding="UTF-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{spec_path}: top-level must be a mapping.")

    required = ('study', 'base_experiment', 'cells', 'folds')
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{spec_path}: missing required keys: {missing}")

    study = raw['study']
    _validate_name(study, "study")
    _validate_name(raw['base_experiment'], "base_experiment")

    cells_raw = raw['cells']
    if not isinstance(cells_raw, list) or not cells_raw:
        raise ValueError(f"{spec_path}: 'cells' must be a non-empty list.")
    cells: list[Cell] = []
    seen_names: set[str] = set()
    for i, c in enumerate(cells_raw):
        if not isinstance(c, dict) or 'name' not in c:
            raise ValueError(f"{spec_path}: cells[{i}] must be a dict with a 'name'.")
        name = c['name']
        _validate_name(name, f"cells[{i}].name")
        if name in seen_names:
            raise ValueError(f"{spec_path}: duplicate cell name {name!r}.")
        seen_names.add(name)
        overrides = c.get('overrides', {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"{spec_path}: cells[{i}].overrides must be a dict.")
        cells.append(Cell(name=name, overrides=dict(overrides)))

    folds = raw['folds']
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{spec_path}: 'folds' must be a non-empty list of ints.")
    if not all(isinstance(f, int) and f >= 0 for f in folds):
        raise ValueError(f"{spec_path}: every fold must be a non-negative int.")

    return Spec(
        study=study,
        base_experiment=raw['base_experiment'],
        cells=cells,
        folds=folds,
    )


def apply_overrides(configs: dict, overrides: dict[str, Any]) -> dict:
    """Return a deep-copied ``configs`` with the dotted-path overrides applied.

    ``trainer.optimization.loss.name`` writes
    ``configs['trainer']['optimization']['loss']['name']``; missing
    intermediate dicts are created on the way. Existing leaf values are
    overwritten. Returns a fresh dict — the input is not mutated.
    """
    import copy
    out = copy.deepcopy(configs)
    for dotted_key, value in overrides.items():
        if not dotted_key:
            raise ValueError("override key must be non-empty")
        parts = dotted_key.split('.')
        cur = out
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
    return out
