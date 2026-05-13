"""Subject-wise stratified k-fold cross-validation. Locks in §A1.

The pre-A1 training loop split the *patch* list 95/5 via
``torch.utils.data.random_split``. With ``pixel_stride=[1, 64, 64]`` and
``sample_dimension=[12, 256, 256]`` adjacent patches share ~92% of their
voxels, so a random patch-level split causes severe information leakage
between train and validation. Every Dice reported under that protocol
is optimistic.

This module produces a *subject-wise* (file-level) split:

  1. Enumerate the TIFFs in ``ds_train/``.
  2. Classify each by its sample-name prefix (Control / Podocin / Collagen)
     — the same mapping ``src/infer/stats.py:detect_group_type`` uses for
     downstream statistics. Unknown prefixes raise ``UnknownSampleGroupError``
     so the failure mode is loud.
  3. Generate ``k=5`` deterministic folds via stratified round-robin
     within each group, seeded by the project SEED so the same dataset
     always yields the same fold partition.
  4. Persist the partition to ``<experiment>/fold_assignments.yaml`` next
     to ``requirements.txt`` / ``git_sha.txt``.

Held-out cohort: ``ds_test/`` is already structurally segregated and
never touched by ``train.py``. No change there.

The split is hand-rolled (no scikit-learn dep) to keep the package's
dependency surface small. For sparse groups (fewer members than ``k``),
the round-robin distributes members deterministically, leaving some
folds without that group represented in their validation set — the best
that can be done without duplicating examples.
"""

import random
import re
from pathlib import Path
from typing import Optional

import yaml

# Project-wide reproducibility seed. Kept here as well as in ``train.py``
# so a fold partition is deterministic across runs even if the calling
# code somehow passes a different seed.
DEFAULT_SEED = 88233474
DEFAULT_K = 5

# Identical to ``src/infer/stats.py:detect_group_type``'s mapping. Repeated
# here so that file (which is downstream stats, not training data) can be
# evolved independently. Keys are uppercased prefix tuples.
SAMPLE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "Control": ("NCW.AUY381", "NCW.AUY380", "CKM103", "CKM110"),
    "Podocin": ("NCW.BDP669", "NCW.BDP672", "NCW.BDP675"),
    "Collagen": ("NCW.CKM105", "NCW.CKM104"),
}


class UnknownSampleGroupError(ValueError):
    """Raised when one or more filenames don't match any known prefix."""


def classify(sample_name: str) -> Optional[str]:
    """Return the sample's group label, or ``None`` if unknown.

    Strict: no default group. Callers decide whether to raise on ``None``.
    """
    cleaned = sample_name.strip().upper()
    for group, prefixes in SAMPLE_GROUP_PREFIXES.items():
        if cleaned.startswith(prefixes):
            return group
    return None


def assign_folds(file_paths,
                 k: int = DEFAULT_K,
                 seed: int = DEFAULT_SEED) -> list[dict]:
    """Build ``k`` stratified folds over ``file_paths``.

    Returns a list of length ``k``, each entry::

        {'fold_id': int, 'group': str, 'train': [str, ...], 'valid': [str, ...]}

    ``train`` and ``valid`` are sorted file *basenames* (not full paths).
    ``group`` is a comma-separated summary of the validation groups in
    this fold (for sanity at experiment-create time).

    Raises ``UnknownSampleGroupError`` if any file's prefix is not in
    ``SAMPLE_GROUP_PREFIXES``. The error message lists every offending
    name so the user can fix the prefixes (or update the mapping) in one
    pass — failing one-at-a-time is a footgun.
    """
    names = sorted(Path(p).name for p in file_paths)
    if not names:
        raise ValueError("assign_folds received an empty file list.")

    labels = []
    unknown = []
    for name in names:
        # Strip the extension before classification.
        stem = re.sub(r'\.(tif|tiff)$', '', name, flags=re.IGNORECASE)
        g = classify(stem)
        if g is None:
            unknown.append(name)
        labels.append(g)

    if unknown:
        raise UnknownSampleGroupError(
            f"{len(unknown)} sample name(s) don't match any known prefix in "
            f"SAMPLE_GROUP_PREFIXES. Either fix the filenames or extend the "
            f"mapping in src/data/folds.py. Offending names: {unknown}")

    if len(names) < k:
        raise ValueError(
            f"Cannot build {k} folds from {len(names)} files. "
            f"Either reduce k or add more training subjects.")

    # Group files by label.
    by_group: dict[str, list[str]] = {g: [] for g in SAMPLE_GROUP_PREFIXES}
    for name, lab in zip(names, labels):
        by_group[lab].append(name)

    # Within each group: deterministically shuffle. Then assign to folds via
    # a *continuous* round-robin counter across groups (rather than restarting
    # at fold 0 for every group). This keeps fold sizes balanced even when
    # groups have fewer members than ``k`` — restarting per group would leave
    # later folds empty when groups are small.
    rng = random.Random(seed)
    shuffled_per_group: dict[str, list[str]] = {}
    for group_name in sorted(by_group):
        shuffled = list(by_group[group_name])
        rng.shuffle(shuffled)
        shuffled_per_group[group_name] = shuffled

    valid_per_fold: list[list[str]] = [[] for _ in range(k)]
    counter = 0
    for group_name in sorted(shuffled_per_group):
        for fname in shuffled_per_group[group_name]:
            valid_per_fold[counter % k].append(fname)
            counter += 1

    folds: list[dict] = []
    for fold_id in range(k):
        valid_files = sorted(valid_per_fold[fold_id])
        valid_set = set(valid_files)
        train_files = sorted(n for n in names if n not in valid_set)
        # Sorted list of distinct group labels present in this fold's validation set.
        valid_groups = sorted({
            classify(re.sub(r'\.(tif|tiff)$', '', n, flags=re.IGNORECASE))
            for n in valid_files
        })
        folds.append({
            'fold_id': fold_id,
            'groups_in_valid': ','.join(valid_groups),
            'train': train_files,
            'valid': valid_files,
        })
    return folds


def write_assignments(experiment_dir, assignments: list[dict]) -> Path:
    """Write the fold layout to ``<experiment_dir>/fold_assignments.yaml``."""
    path = Path(experiment_dir) / "fold_assignments.yaml"
    path.write_text(yaml.safe_dump({
        'k': len(assignments),
        'seed': DEFAULT_SEED,
        'folds': assignments,
    }, sort_keys=False), encoding="UTF-8")
    return path


def read_assignments(experiment_dir) -> list[dict]:
    """Load the fold layout. Raises ``FileNotFoundError`` if missing.

    Older experiments (pre-A1) won't have this file; the caller is
    expected to surface a clear error pointing the user at
    ``gbm.py create`` (which now generates this file).
    """
    path = Path(experiment_dir) / "fold_assignments.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. This experiment was created before A1; "
            "re-create it with the current `gbm.py create` to generate fold "
            "assignments, or copy a fold_assignments.yaml from another "
            "experiment that uses the same ds_train file list.")
    return yaml.safe_load(path.read_text(encoding="UTF-8"))['folds']
