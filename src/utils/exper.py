# Python Imports
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path

# Library Imports
import numpy as np
import yaml

from infer import main_infer
from src.data.folds import assign_folds, write_assignments
from src.infer.blender_io import blender_prepare, blender_render, export_results
from src.infer.stats import calculate_stats

# Local Imports
from src.train.factory import Factory
from src.utils.misc import (
    copy_directory,
    create_dirs_recursively,
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
                    _max_concurrent: int):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
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


def stats(_name: str,
          _root_path: str,
          _inference_tag: str,
          _clipping: bool):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)
    thickness_clip_max = configs['inference']['morph'].get('thickness_clip_max', 1400)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
    inference_result_path = os.path.join(inference_root_path, _inference_tag)
    if not os.path.exists(inference_result_path):
        raise FileNotFoundError("Incorrect path: {inference_result_path}")

    inference_root_path = Path(inference_root_path)
    inference_result_path = Path(inference_result_path)

    stats_dir = inference_root_path / f"{_inference_tag}_stats"

    calculate_stats(inference_result_path,
                    stats_dir,
                    _clipping,
                    thickness_clip_max)


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
    inference_export_path = inference_root_path / f"{inference_result_path.name}_export"

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
                     _output_name: str = None):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configs(configs_path)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
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
    if os.path.exists(inference_result_path):
        if not _force:
            raise FileExistsError(
                f"Inference already exists at {inference_result_path}. "
                "Pass --force to overwrite.")
        logging.info("--force given; removing existing inference at %s",
                     inference_result_path)
        shutil.rmtree(inference_result_path)
    create_dirs_recursively(os.path.join(inference_result_path, 'dummy'))

    configs['root_path'] = _root_path

    configs['inference']['snapshot_path'] =\
        os.path.join(_root_path,
                     _name,
                     'results-train/snapshots/',
                     _snapshot)

    configs['inference']['result_dir'] = inference_result_path

    configs['inference']['inference_ds']['path'] =\
        os.path.join(_root_path,
                     _name,
                     'datasets/',
                     'ds_test_unlabeled/')

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['sample_dimension'] =\
        _sample_dimension

    configs['inference']['inference_ds']['pixel_stride'] = _stride

    configs['inference']['inference_ds']['scale_factor'] = _scale

    configs['inference']['inference_ds']['interpolate'] = _interpolate

    configs['inference']['inference_ds']['workers'] =\
        configs['trainer']['train_ds']['workers']

    if _stitching is not None:
        configs['inference']['stitching'] = _stitching

    inference_configs = configs['inference']

    with open(os.path.join(inference_result_path, 'configs.yaml'), 'w',
              encoding='UTF-8') as config_file:
        yaml.dump(inference_configs,
                  config_file,
                  default_flow_style=None,
                  sort_keys=False)

    main_infer(configs)


def train_experiment(_name: str,
                     _root_path: str,
                     _fold=None,
                     _all_data: bool = False,
                     _epochs=None):
    """Dispatch to all-data, single-fold, or all-folds training.

    ``_all_data=True`` runs a Stage-B final model on every ds_train
    subject (no holdout) via :func:`main_train_all_data`; ``_epochs``
    optionally overrides ``configs.trainer.epochs``.
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


def delete_experiment(_name: str,
                      _root_path: str,
                      _force: bool = False):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    if not _force:
        raise RuntimeError(
            f"Refusing to delete experiment '{_name}' without --force.")
    logging.info("Removing the experiment: %s", _name)
    shutil.rmtree(os.path.join(_root_path, _name))
    logging.info('Experiment "%s" has been deleted', _name)


def create_new_experiment(_name: str,
                          _root_path: str,
                          _source_path: str,
                          _dataset_path: str,
                          _batch_size: int,
                          _voxel_size: list,
                          _z_scale_factor: int = 1,
                          _semi_supervised: bool = False,
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
    copy_directory(_source_path,
                   code_path,
                   ['.git', 'tags'])

    logging.info("Copying experiment's datasets")
    new_dataset_path = os.path.join(destination_path, 'datasets')
    os.makedirs(new_dataset_path, exist_ok=True)

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
       and configs['experiments']['scale_lerning_rate_for_batch_size']:
        batch_ratio =\
            _batch_size / configs['experiments']['default_batch_size']

    if batch_ratio is not None:
        configs['trainer']['optim']['lr'] =\
                round(configs['trainer']['optim']['lr'] * math.sqrt(batch_ratio), 5)
        configs['trainer']['report_freq'] =\
            configs['trainer']['report_freq'] / batch_ratio

    configs['root_path'] = destination_path

    configs['trainer']['train_ds']['path'] = \
        f"{new_dataset_path}/ds_train/"

    configs['trainer']['train_ds']['batch_size'] = _batch_size

    configs['trainer']['valid_ds']['path'] = \
        f"{new_dataset_path}/ds_valid/"

    configs['trainer']['valid_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['path'] = \
        f"{new_dataset_path}/ds_test_unlabeled/"

    configs['inference']['labeled_test_ds']['path'] = \
        f"{new_dataset_path}/ds_test_labeled/"

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    # Persist the Z-scale used at create time. ``Factory.createTrainer``
    # reads this to mask validation metrics to original-label slices only
    # (positions where Z % z_scale == 0); without it the same label voxel
    # would be counted z_scale times.
    configs['trainer']['z_scale'] = int(_z_scale_factor)

    del configs['experiments']

    with open(os.path.join(destination_path, 'configs.yaml'), 'w',
              encoding='UTF-8') as config_file:
        yaml.dump(configs,
                  config_file,
                  default_flow_style=None,
                  sort_keys=False)

    logging.info("Configuration file saved to '%s'",
                 destination_path)
