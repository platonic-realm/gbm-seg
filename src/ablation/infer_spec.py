"""Inference-axis ablation spec parsing and command emission.

A spec YAML describes one inference study:

    study: stitch_pilot
    base_experiment: full_train       # source of snapshots + ds_test
    snapshot: 000-0500.pt             # one trained checkpoint, evaluated N ways
    # Common inference CLI args (shared across cells):
    batch_size: 32
    sample_dimension: '12, 256, 256'
    stride: '1, 128, 128'
    scale_factor: 6
    interpolation: true
    cells:
      - name: gaussian
        overrides: {stitching: gaussian}
      - name: hann
        overrides: {stitching: hann}
      - name: sum_logits
        overrides: {stitching: sum_logits}

Each cell becomes one `gbm.py infer <base_experiment> -s <snapshot> ...
--output-name <study>__<cell> --stitching <mode>` command. Results land
under ``<base_experiment>/results-infer/<study>__<cell>/`` so the four
runs are visually separated. Downstream `gbm.py psp / morph / stats` can
then be run per cell using the same ``--inference-tag <study>__<cell>``.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class InferCell:
    """One inference variant: a name + flat dict of override knobs."""
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class InferSpec:
    """Inference-axis study definition."""
    study: str
    base_experiment: str
    snapshot: str
    cells: list[InferCell]
    # Defaults applied to every cell (override on the gbm.py infer CLI):
    batch_size: int
    sample_dimension: str
    stride: str
    scale_factor: int
    interpolation: bool


# Supported override keys per cell. Currently only `stitching` rewires
# through configs.inference.stitching via the new `--stitching` CLI flag.
# Adding a new axis is a one-liner here + a flag in args.py.
_SUPPORTED_OVERRIDES = {'stitching'}


def parse_infer_spec(spec_path) -> InferSpec:
    raw = yaml.safe_load(Path(spec_path).read_text(encoding="UTF-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{spec_path}: top-level must be a mapping.")

    for key in ('study', 'base_experiment', 'snapshot', 'cells'):
        if key not in raw:
            raise ValueError(f"{spec_path}: missing required key '{key}'.")

    cells_raw = raw['cells']
    if not isinstance(cells_raw, list) or not cells_raw:
        raise ValueError(f"{spec_path}: 'cells' must be a non-empty list.")
    cells: list[InferCell] = []
    seen: set[str] = set()
    for i, c in enumerate(cells_raw):
        if not isinstance(c, dict) or 'name' not in c:
            raise ValueError(
                f"{spec_path}: cells[{i}] must be a dict with a 'name'.")
        if c['name'] in seen:
            raise ValueError(
                f"{spec_path}: duplicate cell name {c['name']!r}.")
        seen.add(c['name'])
        overrides = c.get('overrides', {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"{spec_path}: cells[{i}].overrides must be a dict.")
        unknown = set(overrides) - _SUPPORTED_OVERRIDES
        if unknown:
            raise ValueError(
                f"{spec_path}: cells[{i}] has unsupported override keys "
                f"{sorted(unknown)}. Supported: {sorted(_SUPPORTED_OVERRIDES)}.")
        cells.append(InferCell(name=c['name'], overrides=dict(overrides)))

    return InferSpec(
        study=raw['study'],
        base_experiment=raw['base_experiment'],
        snapshot=raw['snapshot'],
        cells=cells,
        batch_size=int(raw.get('batch_size', 8)),
        sample_dimension=str(raw.get('sample_dimension', '12, 256, 256')),
        stride=str(raw.get('stride', '1, 64, 64')),
        scale_factor=int(raw.get('scale_factor', 6)),
        interpolation=bool(raw.get('interpolation', True)),
    )


def emit_infer_commands(spec: InferSpec,
                        *, sbatch_wrapper: Optional[str] = None) -> list[str]:
    """Build ``gbm.py infer`` commands (optionally sbatch-wrapped), one per
    cell. Each cell uses --output-name=<study>__<cell> so the cells'
    results-infer subdirs stay separated."""
    commands: list[str] = []
    for cell in spec.cells:
        tag = f"{spec.study}__{cell.name}"
        # Common args.
        parts = [
            'python', 'gbm.py', 'infer',
            spec.base_experiment,
            '-s', spec.snapshot,
            '-bs', str(spec.batch_size),
            '-sd', shlex.quote(spec.sample_dimension),
            '-st', shlex.quote(spec.stride),
            '-sf', str(spec.scale_factor),
            '-in', str(spec.interpolation).lower(),
            '--output-name', tag,
            '-f',
        ]
        if 'stitching' in cell.overrides:
            parts += ['--stitching', cell.overrides['stitching']]
        cmd = ' '.join(parts)
        if sbatch_wrapper:
            cmd = (f"sbatch {sbatch_wrapper} {spec.base_experiment} "
                   f"{spec.snapshot} {tag} "
                   f"{cell.overrides.get('stitching', '')}").rstrip()
        commands.append(cmd)
    return commands
