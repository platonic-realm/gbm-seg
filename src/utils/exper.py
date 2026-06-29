# Python Imports
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Library Imports
import numpy as np
import yaml

from infer import main_infer
from src.data.folds import assign_folds, write_assignments
from src.infer.blender_io import blender_prepare, blender_render, export_results
from src.infer.continuity import calculate_continuity, compare_continuity
from src.infer.stats import (
    calculate_stats,
    calculate_stats_one_sample,
    calculate_stats_reduce,
)

# Local Imports
from src.train.factory import Factory
from src.utils.misc import (
    copy_directory,
    create_dirs_recursively,
    hardlink_directory,
    morph_analysis,
    read_configs,
    resize_and_copy,
)
from train import (
    aggregate_cv_from_disk,
    main_train,
    main_train_all_data,
    main_train_all_folds,
)


def _persist_git_provenance(_destination_path: str, _source_path: str):
    """Write git SHA and uncommitted diff next to requirements.txt.

    `pip freeze` alone doesn't capture the working-tree state; this makes the
    experiment dir genuinely reproducible.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_source_path, stderr=subprocess.DEVNULL).decode().strip()
        with open(os.path.join(_destination_path, "git_sha.txt"), "w",
                  encoding="UTF-8") as f:
            f.write(sha + "\n")

        diff = subprocess.check_output(
            ["git", "diff", "HEAD"],
            cwd=_source_path, stderr=subprocess.DEVNULL).decode()
        if diff:
            with open(os.path.join(_destination_path, "git_diff.patch"), "w",
                      encoding="UTF-8") as f:
                f.write(diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Could not capture git provenance for %s", _source_path)


def experiment_exists(_root_path, _name) -> bool:
    result = False
    if os.path.exists(_root_path):
        for item in os.listdir(_root_path):
            if os.path.isdir(
                    os.path.join(_root_path,
                                 item)) and item == _name:
                result = True
                break

    return result

def list_experiments(_root_path):
    print("Experiments:")
    if os.path.exists(_root_path):
        for item in sorted(os.listdir(_root_path)):
            if os.path.isdir(os.path.join(_root_path,
                                          item)):
                print(f"* {item}")

def visualize_results(_name: str,
                      _root_path: str,
                      _inference_tag: str,
                      _sample_name: str):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    # configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    # configs = read_configs(configs_path)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
    sample_path = os.path.join(inference_root_path, _inference_tag, _sample_name)
    if not os.path.exists(sample_path):
        raise FileNotFoundError("Incorrect path: {inference_result_path}")

    blender_prepare(sample_path)


def post_processing(_name: str,
                    _root_path: str,
                    _inference_tag: str,
                    _max_concurrent: int,
                    _labeled: bool = False):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)

    # `_labeled` swaps the output root from results-infer/ to
    # results-infer-labeled/ — pair with `gbm.py infer --labeled`. The
    # two streams are kept separate on disk so morph (unlabeled volumes)
    # and expert comparison (labeled crops) never collide.
    sub = 'results-infer-labeled' if _labeled else 'results-infer'
    inference_root_path = os.path.join(_root_path, _name, sub)
    inference_result_path = os.path.join(inference_root_path, _inference_tag)

    if not os.path.exists(inference_result_path):
        raise FileNotFoundError("Incorrect path: {inference_result_path}")

    factory = Factory(configs)
    psp = factory.createPSPer()
    psp.parallel_post_processing(inference_result_path,
                                 _max_concurrent)


def render_results(_name: str,
                   _root_path: str,
                   _inference_tag: str):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
    inference_result_path = os.path.join(inference_root_path, _inference_tag)

    if not os.path.exists(inference_result_path):
        raise FileNotFoundError("Incorrect path: {inference_result_path}")

    blender_render(inference_result_path)


def clipping_high_values(sample_path):
    logging.info(f"Starting aggressive analysis for sample: {sample_path}")
    thickness_map_path = os.path.join(sample_path, 'distance_result.npz')

    if not os.path.exists(thickness_map_path):
        raise FileNotFoundError(f"Thickness map not found: {thickness_map_path}")

    logging.info(f"Loading thickness map from: {thickness_map_path}")
    thickness_map = np.load(thickness_map_path)['arr']

    original_non_zero_voxels = np.count_nonzero(thickness_map)

    thickness_map[thickness_map > 1400] = 0

    final_non_zero_voxels = np.count_nonzero(thickness_map)
    altered_voxels = original_non_zero_voxels - final_non_zero_voxels
    if original_non_zero_voxels > 0:
        percentage_altered = (altered_voxels / original_non_zero_voxels) * 100
        logging.info(f"Altered {altered_voxels} voxels, which is {percentage_altered:.2f}% of the original non-zero data.")
    else:
        logging.info("No non-zero voxels to alter.")

    save_path = os.path.join(sample_path, 'distance_aggressive_removed.npz')
    logging.info(f"Saving aggressively removed thickness map to: {save_path}")
    np.savez_compressed(save_path, arr=thickness_map)
    logging.info("Aggressive analysis completed.")


def clipping(_name: str,
             _root_path: str,
             _inference_tag: str,
             _sample_name: str = None):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
    inference_result_path = os.path.join(inference_root_path, _inference_tag)

    if not os.path.exists(inference_result_path):
        raise FileNotFoundError("Incorrect path: {inference_result_path}")

    if _sample_name:
        sample_path = os.path.join(inference_result_path, _sample_name)
        if not os.path.exists(sample_path):
            raise FileNotFoundError("Incorrect path: {sample_path}")
        clipping_high_values(sample_path)


def _resolve_stats_paths(_name, _root_path, _inference_tag):
    """Shared path resolution for the stats entry points. Returns
    (inference_result_path, stats_dir, max_thickness).

    `max_thickness` is the single stats knob: it bounds every figure axis /
    colour-scale AND is the --clipping drop threshold (one value, not two).
    Read from inference.stats.max_thickness, falling back to the legacy
    inference.morph.thickness_clip_max, then 1200."""
    if not experiment_exists(_root_path, _name):
        raise FileNotFoundError(f"Experiment '{_name}' doesn't exist")

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)
    max_thickness = (configs['inference'].get('stats') or {}).get('max_thickness')
    if max_thickness is None:
        max_thickness = configs['inference'].get('morph', {}).get(
            'thickness_clip_max', 1200)

    inference_root_path = Path(os.path.join(_root_path, _name, 'results-infer'))
    inference_result_path = inference_root_path / _inference_tag
    if not inference_result_path.exists():
        raise FileNotFoundError(f"Incorrect path: {inference_result_path}")

    stats_dir = inference_root_path / f"{_inference_tag}_stats"
    return inference_result_path, stats_dir, max_thickness


def _maybe_expert_comparison(_name, _root_path, _inference_tag, _stats_dir):
    """Run the optional expert comparison (model-vs-annotators) if the
    labeled-inference output + annotator dirs exist. Shared by stats() and
    stats_reduce()."""
    labeled_inference_path = os.path.join(
        _root_path, _name, 'results-infer-labeled', _inference_tag)
    ds_test_labeled_path = os.path.join(
        _root_path, _name, 'datasets', 'ds_test_labeled')
    if os.path.isdir(labeled_inference_path) and os.path.isdir(ds_test_labeled_path):
        from src.infer.expert_comparison import calculate_expert_comparison
        calculate_expert_comparison(Path(labeled_inference_path),
                                    Path(ds_test_labeled_path),
                                    _stats_dir)
    else:
        logging.info(
            "Skipping expert comparison: no labeled-inference output at "
            "%s, or no ds_test_labeled at %s.",
            labeled_inference_path, ds_test_labeled_path)


def stats(_name: str,
          _root_path: str,
          _inference_tag: str,
          _clipping: bool,
          _sample_name: str = None):
    """Statistical analysis.

    With ``_sample_name`` set, runs ONLY that sample's per-sample stage
    (the SLURM-array task entry point) — no aggregation, no expert
    comparison; the reduce step does those. Without it, runs the full
    single-process pipeline (per-sample + reduce + expert comparison).
    """
    inference_result_path, stats_dir, max_thickness = \
        _resolve_stats_paths(_name, _root_path, _inference_tag)

    if _sample_name is not None:
        calculate_stats_one_sample(inference_result_path, stats_dir,
                                   _sample_name, _clipping, max_thickness)
        return

    calculate_stats(inference_result_path,
                    stats_dir,
                    _clipping,
                    max_thickness)

    # Opportunistically run the expert comparison: if a labeled-inference
    # output exists at the same tag (results-infer-labeled/<tag>/) and
    # the experiment has annotator subdirs under datasets/ds_test_labeled/,
    # compare model predictions to each annotator + measure inter-rater
    # agreement. Outputs go into the same stats_dir but in their own
    # files so the morph stats and the comparison report stay independent.
    _maybe_expert_comparison(_name, _root_path, _inference_tag, stats_dir)


def stats_reduce(_name: str,
                 _root_path: str,
                 _inference_tag: str,
                 _clipping: bool):
    """Reduce step of the parallel stats path: aggregate the per-sample
    sidecars written by the array tasks, then run the expert comparison.
    ``_clipping`` is accepted for CLI symmetry (the per-sample stage
    already applied it)."""
    inference_result_path, stats_dir, max_thickness = \
        _resolve_stats_paths(_name, _root_path, _inference_tag)
    calculate_stats_reduce(inference_result_path, stats_dir, max_thickness)
    _maybe_expert_comparison(_name, _root_path, _inference_tag, stats_dir)


def continuity(_name: str,
               _root_path: str,
               _inference_tag: str):
    """Z-continuity analysis — quantifies the along-Z jaggedness of the
    predictions that the validation Dice can't see (the axis ContLoss
    targets). Dataset-agnostic: point it at any inference tag (validation or
    test predictions). Writes <tag>_continuity/continuity_result.yaml."""
    if not experiment_exists(_root_path, _name):
        raise FileNotFoundError(f"Experiment '{_name}' doesn't exist")

    inference_root = Path(os.path.join(_root_path, _name, 'results-infer'))
    inference_result_path = inference_root / _inference_tag
    if not inference_result_path.exists():
        raise FileNotFoundError(f"Incorrect path: {inference_result_path}")

    output_dir = inference_root / f"{_inference_tag}_continuity"
    calculate_continuity(inference_result_path, output_dir)


def continuity_compare(_root_path: str,
                       _specs: list):
    """Tabulate the continuity aggregate of several runs side by side
    (e.g. a Cont vs CrossEntropy comparison). Each spec is an
    ``EXPERIMENT:INFERENCE_TAG`` string; the experiment name is the column
    label. Runs may live in different experiments (the ablation cells do)."""
    runs = {}
    for spec in _specs:
        if ':' not in spec:
            raise ValueError(
                f"continuity-compare spec must be EXPERIMENT:TAG, got '{spec}'")
        exp, tag = spec.split(':', 1)
        runs[exp] = (Path(_root_path) / exp / 'results-infer'
                     / f"{tag}_continuity" / "continuity_result.yaml")
    compare_continuity(runs)


def export(_name: str,
           _root_path: str,
           _inference_tag: str):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')

    inference_result_path = os.path.join(inference_root_path, _inference_tag)
    if not os.path.exists(inference_result_path):
        raise FileNotFoundError("Incorrect path: {inference_result_path}")

    inference_root_path = Path(inference_root_path)
    inference_result_path = Path(inference_result_path)
    # Anchor the export dir next to the tag dir (.parent), not at the bare
    # results-infer root: an inference tag carries a fold_N/ or all_data/
    # prefix, and `.name` alone would drop it — colliding the exports of
    # different folds (or a fold vs all-data) that share a snapshot stem.
    inference_export_path = (inference_result_path.parent
                             / f"{inference_result_path.name}_export")

    if os.path.exists(inference_export_path):
        raise FileExistsError(f"The path already exists: {inference_export_path}")
        # answer = input("Export directory for this inference already exists,"
        #                " overwrite? (y/n) [default=n]: ")
        # if answer.lower() == "y":
        #     shutil.rmtree(inference_export_path)
        # else:
        #     return

    create_dirs_recursively(inference_export_path / "dummy")

    export_results(inference_result_path,
                   inference_export_path)


def analyze_morphometrics(_name: str,
                          _root_path: str,
                          _inference_tag: str,
                          _sample_name: str):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')

    sample_path = os.path.join(inference_root_path, _inference_tag, _sample_name)
    if not os.path.exists(sample_path):
        raise FileNotFoundError("Incorrect path: {sample_path}")

    factory = Factory(configs)
    morph = factory.createMorphModule()

    morph_analysis(sample_path, morph)


def infer_experiment(_name: str,
                     _root_path: str,
                     _snapshot: str,
                     _batch_size: int,
                     _sample_dimension: list,
                     _stride: list,
                     _scale: int,
                     _interpolate: bool,
                     _force: bool = False,
                     _stitching: str = None,
                     _output_name: str = None,
                     _sample_name: str = None,
                     _labeled: bool = False):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)

    # `_labeled` swaps the output root + the source dataset:
    # results-infer-labeled/ <- ds_test_labeled/<first annotator>/.
    # The crops are identical across annotators (only channel-3 labels
    # differ), so we read whichever annotator subdir sorts first
    # alphabetically and the per-expert labels are picked up at
    # stats-comparison time. results-infer-labeled stays separate from
    # results-infer so the morph (full-volume) and expert-comparison
    # (crop) streams never collide.
    sub = 'results-infer-labeled' if _labeled else 'results-infer'
    inference_root_path = os.path.join(_root_path, _name, sub)
    create_dirs_recursively(os.path.join(inference_root_path, 'dummy'))

    # `--output-name` overrides the auto-derived tag so inference-axis
    # ablation cells (same snapshot + sample_dim + stride + scale, different
    # stitching mode) end up in distinct directories.
    if _output_name is not None:
        inference_tag = _output_name
    else:
        inference_tag = (f"{_snapshot}_{''.join(_sample_dimension)}_"
                         f"{''.join(_stride)}_{_scale}")

    _sample_dimension = [int(item) for item in _sample_dimension]
    _stride = [int(item) for item in _stride]
    _scale = int(_scale)
    _batch_size = int(_batch_size)

    inference_result_path = os.path.join(inference_root_path, inference_tag)
    if _sample_name is None:
        # Whole-directory mode: --force wipes the entire tag directory.
        if os.path.exists(inference_result_path):
            if not _force:
                raise FileExistsError(
                    f"Inference already exists at {inference_result_path}. "
                    "Pass --force to overwrite.")
            logging.info("--force given; removing existing inference at %s",
                         inference_result_path)
            shutil.rmtree(inference_result_path)
        create_dirs_recursively(os.path.join(inference_result_path, 'dummy'))
    else:
        # Per-volume mode (SLURM array): the tag directory is SHARED across
        # the array — every task writes its own <tag>/<volume>/ subdir into
        # it, so the shared dir must never be wiped. makedirs(exist_ok) is
        # race-safe across the concurrently-starting array tasks.
        create_dirs_recursively(os.path.join(inference_result_path, 'dummy'))

    # The experiment directory, not the bare experiments root — a bare
    # root makes the factory build trainer paths against it, and
    # Snapper.__init__ then mkdir's a stray results-train/snapshots/ there.
    configs['root_path'] = os.path.join(_root_path, _name)

    configs['inference']['snapshot_path'] =\
        os.path.join(_root_path,
                     _name,
                     'results-train/snapshots/',
                     _snapshot)

    configs['inference']['result_path'] = inference_result_path

    if _labeled:
        # ds_test_labeled/ has per-annotator subdirs (e.g. Chris, David,
        # Robin) with the SAME crops; the model only reads channels 0-2
        # (the image), so any annotator works — pick the first alphabetic.
        labeled_root = os.path.join(_root_path, _name,
                                    'datasets', 'ds_test_labeled')
        annotators = sorted(
            d for d in os.listdir(labeled_root)
            if os.path.isdir(os.path.join(labeled_root, d)))
        if not annotators:
            raise FileNotFoundError(
                f"No annotator subdirs under {labeled_root}. Expected "
                f"e.g. Chris/, David/, Robin/.")
        configs['inference']['inference_ds']['path'] = (
            os.path.join(labeled_root, annotators[0]) + '/')
        logging.info("Labeled inference reading crops from annotator "
                     "subdir '%s' (channel-3 labels from every annotator "
                     "are picked up at stats time).", annotators[0])
    else:
        configs['inference']['inference_ds']['path'] = \
            os.path.join(_root_path, _name,
                         'datasets', 'ds_test_unlabeled') + '/'

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['sample_dimension'] =\
        _sample_dimension

    configs['inference']['inference_ds']['pixel_stride'] = _stride

    configs['inference']['inference_ds']['scale_factor'] = _scale

    configs['inference']['inference_ds']['interpolate'] = _interpolate

    configs['inference']['inference_ds']['workers'] =\
        configs['trainer']['data']['train_ds']['workers']

    if _stitching is not None:
        configs['inference']['stitching'] = _stitching

    inference_configs = configs['inference']

    # In per-volume mode every array task would write this; the content is
    # identical (sample_name is not persisted), so a write-if-absent guard
    # avoids the concurrent-write race on the shared tag directory.
    cfg_yaml = os.path.join(inference_result_path, 'configs.yaml')
    if _sample_name is None or not os.path.exists(cfg_yaml):
        with open(cfg_yaml, 'w', encoding='UTF-8') as config_file:
            yaml.dump(inference_configs,
                      config_file,
                      default_flow_style=None,
                      sort_keys=False)

    main_infer(configs, _sample_name=_sample_name)


def train_experiment(_name: str,
                     _root_path: str,
                     _fold=None,
                     _all_data: bool = False,
                     _epochs=None):
    """Dispatch to all-data, single-fold, or all-folds training.

    ``_all_data=True`` runs a Stage-B final model on every ds_train
    subject (no holdout) via :func:`main_train_all_data`; ``_epochs``
    optionally overrides ``configs.trainer.optimization.epochs``.
    ``_fold=None`` (the default for `gbm.py train EXP` without --fold)
    runs the full CV via :func:`main_train_all_folds`. ``_fold=<int>``
    runs a single fold via :func:`main_train`.
    """
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)
    if _all_data:
        main_train_all_data(configs,
                            _epochs=(None if _epochs is None else int(_epochs)))
    elif _fold is None:
        main_train_all_folds(configs)
    else:
        main_train(configs, _fold=int(_fold))


def offline_augment_experiment(_name: str, _root_path: str):
    """Precompute the offline-augmentation cache for an experiment.

    Offline augmentation (twist/zoom/rotate variants of every training
    volume, written to ``<ds_train>/cache/``) is CPU-heavy. Doing it
    lazily inside the training dataset means every DDP rank recomputes
    the same volumes simultaneously — N x CPU and N x RAM, which swap-
    kills the node. This command builds the cache ONCE, single-process,
    ahead of training; the DDP ranks then only read it.

    Run it after `gbm.py create` and before any `gbm.py train` whose
    config has ``train_ds.augmentation.enabled_offline: true``.
    """
    if not experiment_exists(_root_path, _name):
        raise FileNotFoundError(f"Experiment '{_name}' doesn't exist")
    configs = read_configs(os.path.join(_root_path, _name, 'configs.yaml'))
    methods = (configs['trainer']['data']['train_ds']['augmentation']
               .get('methods_offline'))
    if not methods:
        logging.warning(
            "Experiment '%s' has no "
            "trainer.data.train_ds.augmentation.methods_offline; "
            "nothing to precompute.", _name)
        return
    logging.info("Precomputing offline-augmentation cache for '%s' "
                 "(methods: %s)", _name, methods)
    factory = Factory(configs)
    # Building the dataset in precompute mode populates <ds_train>/cache/
    # as a side effect of GBMDataset.__init__; the object is then dropped.
    factory.createTrainDataset(file_filter=None, _offline_precompute=True)
    logging.info("Offline-augmentation cache built for '%s'.", _name)


def aggregate_cv_experiment(_name: str, _root_path: str):
    """Standalone CV aggregation. Reads each fold's best_metrics.yaml from
    disk and writes <exp>/cv_results.{yaml,npz} plus a W&B cv_summary run
    (if enabled). Designed to run after a parallel sbatch fan-out of
    per-fold training jobs (see ``sbatch/submit_cv.sh``).
    """
    if not experiment_exists(_root_path, _name):
        raise FileNotFoundError(f"Experiment '{_name}' doesn't exist")
    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)
    aggregate_cv_from_disk(configs)


def _delete_wandb_runs(_name: str, _configs_path: str) -> None:
    """Best-effort deletion of an experiment's Weights & Biases runs.

    The W&B project/entity are read from the experiment's ``configs.yaml``;
    runs are matched by their W&B ``group``, which ``maybe_init_wandb``
    sets to the experiment-directory name. This NEVER raises — a W&B
    problem (not installed, offline, project gone) must not block the
    local experiment deletion. It just logs and returns.
    """
    if not os.path.isfile(_configs_path):
        logging.warning(
            "No configs.yaml for '%s'; cannot resolve its W&B project — "
            "skipping W&B cleanup.", _name)
        return
    try:
        with open(_configs_path, encoding='UTF-8') as f:
            cfg = yaml.safe_load(f) or {}
        trainer_cfg = cfg.get('trainer', {}) or {}
        logging_cfg = trainer_cfg.get('logging', {}) or {}
        wandb_cfg = logging_cfg.get('wandb', {}) or {}
        entity = wandb_cfg.get('entity')
        project = wandb_cfg.get('project')
    except Exception as exc:  # pragma: no cover — defensive
        logging.warning("Could not read W&B config for '%s': %s", _name, exc)
        return
    if not project:
        logging.warning(
            "No trainer.logging.wandb.project in '%s' config; "
            "skipping W&B cleanup.", _name)
        return
    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed; skipping W&B cleanup.")
        return
    api_path = f"{entity}/{project}" if entity else project
    try:
        runs = list(wandb.Api().runs(api_path))
    except Exception as exc:  # pragma: no cover — networked
        logging.warning(
            "Could not list W&B runs at '%s' for '%s': %s — skipping.",
            api_path, _name, exc)
        return
    deleted = 0
    for run in runs:
        if run.group == _name:
            try:
                run.delete()
                deleted += 1
            except Exception as exc:  # pragma: no cover — networked
                logging.warning("Failed to delete W&B run %s: %s",
                                run.id, exc)
    logging.info("Deleted %d W&B run(s) grouped under '%s' (project '%s').",
                 deleted, _name, project)


def delete_experiment(_name: str,
                      _root_path: str,
                      _force: bool = False,
                      _remove_wandb: bool = False):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    if not _force:
        raise RuntimeError(
            f"Refusing to delete experiment '{_name}' without --force.")
    exp_path = os.path.join(_root_path, _name)
    if _remove_wandb:
        # Resolve + delete the W&B runs *before* rmtree — the W&B
        # project/entity are read from the experiment's configs.yaml,
        # which is about to be removed.
        _delete_wandb_runs(_name, os.path.join(exp_path, 'configs.yaml'))
    logging.info("Removing the experiment: %s", _name)
    shutil.rmtree(exp_path)
    logging.info('Experiment "%s" has been deleted', _name)


def create_new_experiment(_name: str,
                          _root_path: str,
                          _source_path: str,
                          _dataset_path: str,
                          _batch_size: int,
                          _voxel_size: list,
                          _z_scale_factor: int = 1,
                          _semi_supervised: bool = False,
                          _reuse_dataset_from: Optional[str] = None,
                          _ds_train_subdir: str = 'ds_train',
                          _ds_test_labeled_subdir: str = 'ds_test_labeled',
                          _ds_test_unlabeled_subdir: str = 'ds_test_unlabeled'):

    destination_path = os.path.join(_root_path, f'{_name}/')
    logging.info("Creating a new experiment in '%s%s/'", _root_path, _name)
    if os.path.exists(destination_path):
        message = f"Experiment already exists: {destination_path}"
        raise FileExistsError(message)

    logging.info("Copying project's source code")
    create_dirs_recursively(os.path.join(destination_path, 'dummy'))

    code_path = os.path.join(destination_path, 'code/')
    create_dirs_recursively(os.path.join(code_path, 'dummy'))
    # Exclude runtime artefacts from the code/ snapshot: .git (history),
    # tags (ctags index), and wandb/ — the latter accumulates ~100 stale
    # run directories and carries dangling snapshot symlinks that would
    # otherwise abort the copy.
    copy_directory(_source_path,
                   code_path,
                   ['.git', 'tags', 'wandb'])

    new_dataset_path = os.path.join(destination_path, 'datasets')
    os.makedirs(new_dataset_path, exist_ok=True)

    if _reuse_dataset_from:
        # Skip the ~30-min TIFF resize and hardlink another experiment's
        # already-prepared datasets/ subtree. Same filesystem only. If the
        # source had run `gbm.py offline-aug`, the augmented variants come
        # along for free.
        src_dataset_path = os.path.join(_root_path, _reuse_dataset_from,
                                        'datasets')
        if not os.path.isdir(src_dataset_path):
            raise FileNotFoundError(
                f"--reuse-dataset source has no datasets/ directory: "
                f"{src_dataset_path}")
        logging.info("Hardlinking dataset from '%s' (skipping TIFF resize)",
                     src_dataset_path)
        count = hardlink_directory(src_dataset_path, new_dataset_path)
        logging.info("Hardlinked %d files from '%s'", count, _reuse_dataset_from)
        new_ds_train_path = os.path.join(new_dataset_path, 'ds_train')
    else:
        logging.info("Copying experiment's datasets")

        new_ds_train_path = os.path.join(new_dataset_path, 'ds_train')
        create_dirs_recursively(os.path.join(new_ds_train_path, 'dummy'))
        resize_and_copy(os.path.join(_dataset_path, _ds_train_subdir),
                        new_ds_train_path,
                        _voxel_size,
                        _z_scale_factor=_z_scale_factor)

        # Both test sets stay at native Z resolution; `gbm.py infer` performs the
        # Z upsampling on-the-fly via InferenceDataset when --interpolation true.
        # Pre-upsampling here would double the inflation factor (and also fight
        # with inference.inference_ds.scale_factor=6).

        # ds_test_unlabeled: flat directory of whole-glomerulus 3-channel volumes.
        new_ds_test_unlabeled_path = os.path.join(new_dataset_path,
                                                  'ds_test_unlabeled')
        create_dirs_recursively(os.path.join(new_ds_test_unlabeled_path, 'dummy'))
        resize_and_copy(os.path.join(_dataset_path, _ds_test_unlabeled_subdir),
                        new_ds_test_unlabeled_path,
                        _voxel_size)

        # ds_test_labeled: one sub-directory per annotator (e.g. Chris/David/Robin),
        # each holding the same crops as 4-channel TIFFs (channel 3 = that expert's
        # label). Copy each expert into a matching sub-directory so the 3-expert
        # structure is preserved for Stage-C scoring.
        new_ds_test_labeled_path = os.path.join(new_dataset_path, 'ds_test_labeled')
        create_dirs_recursively(os.path.join(new_ds_test_labeled_path, 'dummy'))
        src_labeled = Path(_dataset_path) / _ds_test_labeled_subdir
        expert_dirs = (sorted(p for p in src_labeled.iterdir() if p.is_dir())
                       if src_labeled.is_dir() else [])
        if not expert_dirs:
            logging.warning(
                "ds_test_labeled source '%s' has no annotator sub-directories; "
                "Stage-C evaluation will be unavailable for this experiment.",
                src_labeled)
        for expert_dir in expert_dirs:
            dest_expert = os.path.join(new_ds_test_labeled_path, expert_dir.name)
            create_dirs_recursively(os.path.join(dest_expert, 'dummy'))
            resize_and_copy(str(expert_dir), dest_expert, _voxel_size)
            logging.info("Copied labeled test crops for annotator '%s'",
                         expert_dir.name)

    logging.info("Saving the requirements file to '%s'",
                 destination_path)
    output = subprocess.check_output(["pip", "freeze"])
    with open(os.path.join(destination_path, 'requirements.txt'), "w",
              encoding='UTF-8') as f:
        f.write(output.decode().strip())

    _persist_git_provenance(destination_path, _source_path)

    # A1: generate fold_assignments.yaml from the just-copied training TIFFs
    # so subject-wise CV is available out of the box.
    train_files = sorted(p.name for p in Path(new_ds_train_path).glob('*.tif*'))
    if train_files:
        fold_assignments = assign_folds(train_files)
        path = write_assignments(destination_path, fold_assignments)
        logging.info("Wrote %d-fold assignments to %s", len(fold_assignments), path)
    else:
        logging.warning(
            "ds_train/ contains no TIFFs; skipping fold assignment. "
            "Training will fail until ds_train/ is populated and "
            "`gbm.py create` is re-run for this experiment.")

    logging.warning("Don't forget to edit the configurations")

    with open('./configs/template.yaml',
              encoding='UTF-8') as template_file:
        configs = yaml.safe_load(template_file)

    batch_ratio = None
    if configs['experiments']['default_batch_size'] != _batch_size\
       and configs['experiments']['scale_learning_rate_for_batch_size']:
        batch_ratio =\
            _batch_size / configs['experiments']['default_batch_size']

    if batch_ratio is not None:
        optim_cfg = configs['trainer']['optimization']['optim']
        optim_cfg['lr'] = round(optim_cfg['lr'] * math.sqrt(batch_ratio), 5)
        configs['trainer']['logging']['report_freq'] =\
            configs['trainer']['logging']['report_freq'] / batch_ratio

    configs['root_path'] = destination_path

    configs['trainer']['data']['train_ds']['path'] = \
        f"{new_dataset_path}/ds_train/"

    configs['trainer']['data']['train_ds']['batch_size'] = _batch_size

    configs['trainer']['data']['valid_ds']['path'] = \
        f"{new_dataset_path}/ds_valid/"

    configs['trainer']['data']['valid_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['path'] = \
        f"{new_dataset_path}/ds_test_unlabeled/"

    configs['inference']['labeled_test_ds']['path'] = \
        f"{new_dataset_path}/ds_test_labeled/"

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    # Persist the Z-scale used at create time. ``Factory.createTrainer``
    # reads this to mask validation metrics to original-label slices only
    # (positions where Z % z_scale == 0); without it the same label voxel
    # would be counted z_scale times. The key is injected here because it
    # is not present in configs/template.yaml.
    configs['trainer']['data']['z_scale'] = int(_z_scale_factor)

    del configs['experiments']

    with open(os.path.join(destination_path, 'configs.yaml'), 'w',
              encoding='UTF-8') as config_file:
        yaml.dump(configs,
                  config_file,
                  default_flow_style=None,
                  sort_keys=False)

    logging.info("Configuration file saved to '%s'",
                 destination_path)
