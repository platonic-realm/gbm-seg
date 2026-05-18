# General utilities: logging, TIFF/dir IO, config handling, tensor helpers.
# Stats/visualization moved to src/infer/stats.py.
# Blender helpers + export_results moved to src/infer/blender_io.py.

# Python Imports
import logging
import os
import shutil
import sys
from io import StringIO
from pathlib import Path

# Library Imports
import numpy as np
import tifffile
import torch
import torch.nn.functional as Fn
import yaml
from scipy.ndimage import zoom

# Local Imports


def basic_logger() -> None:
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level='INFO', format=log_format)


def configure_logger(_configs: dict, _log_to_file: bool = True) -> None:
    LOG_LEVEL = _configs['logging']['log_level']
    root_path = _configs['root_path']
    log_file = os.path.join(root_path,
                            _configs['trainer']['logging']['result_path'],
                            _configs['logging']['log_file'])
    log_std = _configs['logging']['log_std']

    handlers = []
    if _log_to_file and log_file is not None:
        log_file = Path(log_file)
        create_dirs_recursively(log_file)
        handlers.append(logging.FileHandler(log_file))
    if log_std:
        handlers.append(logging.StreamHandler(sys.stdout))

    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.getLogger("pudb").setLevel(logging.INFO)
    logging.basicConfig(level=LOG_LEVEL, format=log_format, handlers=handlers)

    logging.info("Log Level: %s", LOG_LEVEL)


def create_dirs_recursively(_path: str):
    dir_path = os.path.dirname(_path)
    path: Path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


# https://forum.image.sc/t/reading-pixel-size-from-image-file-with-python/74798/2
def get_voxel_size(_tiff, _path=None, _default=None):
    """Read [x, y, z] voxel size (µm/pixel) from a tifffile-opened TIFF.

    The pre-fix silent fallback to ``1.0`` combined with the
    ``default_voxel_size=[0.05, 0.05, 0.3]`` target produced a ~5% zoom
    that corrupted resize_and_copy outputs to 102x102 from 2048x2048
    sources. Two files in the cluster mouse dataset
    (NCWM.BDP669.Series011.Pod-mutation, NCWM.CKM104.Series006.New.COL4-mutation)
    are known to be affected.

    Behaviour:
    * ``_default=None`` — raise ``ValueError`` on missing X/Y resolution
      tags (safe default; surfaces bad sources at ``gbm.py create`` time).
    * ``_default=[x, y, z]`` — soft fallback: log a warning naming the
      file and the missing tag, return the supplied default so callers
      compute a zoom factor of 1.0 (no shrink). The TODO marker below
      tracks files that need their metadata investigated.

    Z falls back to ``1.0`` when ImageJ metadata lacks ``spacing`` —
    ``resize_and_copy`` doesn't resize on Z so this is harmless. (If a
    future caller resizes Z, route it through ``_default[2]`` too.)
    """
    # TODO(arash): The two cluster-mouse files mentioned above carry no
    # X/YResolution metadata — confirm whether they were acquired at the
    # default voxel size (0.05 µm/pixel) and either re-export with the
    # right tags or move them out of ds_train/. Tracking via the soft
    # fallback below; the warning log fires at every gbm.py create.
    def _xy_voxel_size(tags, key, fallback):
        assert key in ['XResolution', 'YResolution']
        if key not in tags:
            if fallback is None:
                raise ValueError(
                    f"TIFF{' ' + str(_path) if _path else ''} is missing the "
                    f"{key} tag. resize_and_copy needs X/Y pixel size to compute "
                    "zoom_factors against default_voxel_size; without it the "
                    "silent fallback to 1.0 µm/pixel produces a ~5% zoom that "
                    "corrupts the output (e.g. 2048→102). Fix the source TIFF "
                    "(ImageJ: Image → Properties; tifffile.imwrite: pass "
                    "resolution=(px_per_unit, px_per_unit)), drop the file, "
                    "or call get_voxel_size with a non-None _default.")
            logging.warning(
                "TIFF %s lacks %s; assuming voxel_size=%s µm/pixel (no zoom). "
                "Investigate this file's metadata.",
                _path, key, fallback)
            return fallback
        num_pixels, units = tags[key].value
        return units / num_pixels

    image_metadata = _tiff.imagej_metadata
    if image_metadata is not None:
        z = image_metadata.get('spacing', 1.)
    else:
        z = 1.

    tags = _tiff.pages[0].tags
    fallback_x = _default[0] if _default is not None else None
    fallback_y = _default[1] if _default is not None else None
    y = _xy_voxel_size(tags, 'YResolution', fallback_y)
    x = _xy_voxel_size(tags, 'XResolution', fallback_x)
    return [x, y, z]


def _z_interpolate_and_stack(_image: np.ndarray,
                             _scale_factor: int,
                             _has_labels: bool) -> np.ndarray:
    """Z-upsample channels and stack labels — restores the deleted
    ``src/scripts/interpolate.py`` logic so it now lives inside
    ``resize_and_copy``.

    * Channels 0..2 (nephrin, collagen-4, WGA): trilinear interpolation
      along Z via ``F.interpolate(scale_factor=(N, 1, 1), mode='trilinear')``.
    * Channel 3 (labels): ``np.repeat(..., N, axis=0)`` so each original
      label slice is replicated N times along Z. The "real" labels in the
      output are at Z positions ``i * N`` for the source slice ``i``; the
      N-1 positions after each are exact copies. ``factory.createTrainer``
      reads ``trainer.data.z_scale`` so the validation step
      (``_select_real_z``) can mask metric computation to those originals.

    Returns a new array of shape ``(Z * N, C, H, W)`` with the same dtype
    as the input. ``N <= 1`` is a no-op pass-through.
    """
    if _scale_factor <= 1:
        return _image

    z_in, channels, height, width = _image.shape
    out_z = z_in * _scale_factor
    out = np.empty((out_z, channels, height, width), dtype=_image.dtype)

    intensity_count = 3 if _has_labels else channels
    for c in range(intensity_count):
        ch = _image[:, c, :, :]  # (Z, H, W)
        # F.interpolate wants (N, C_in, D, H, W).
        tensor = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0)
        scaled = Fn.interpolate(tensor,
                                scale_factor=(_scale_factor, 1, 1),
                                mode='trilinear',
                                align_corners=False)
        out[:, c, :, :] = (scaled.squeeze(0).squeeze(0).numpy()
                           .astype(_image.dtype))

    if _has_labels:
        labels = _image[:, 3, :, :]  # (Z, H, W)
        out[:, 3, :, :] = np.repeat(labels, _scale_factor, axis=0)

    return out


def resize_and_copy(_source_dir, _dest_dir, _target_size,
                    _z_scale_factor: int = 1):
    """Copy every TIFF from ``_source_dir`` to ``_dest_dir``.

    * XY axes are zoomed to match ``_target_size[0..1]`` µm/pixel (per-slice
      via ``scipy.ndimage.zoom``).
    * When ``_z_scale_factor > 1``, the assembled volume is also
      Z-upsampled (``trilinear`` on channels, ``np.repeat`` on labels) —
      restoring the deleted offline ``interpolate.py`` step as part of
      ``gbm.py create``. The label-stacking sets up the
      ``trainer.data.z_scale``-driven validation mask in ``Unet3DTrainer``.
    """
    source_path = Path(_source_dir)
    tiff_files = list(source_path.glob('*.tif')) + list(source_path.glob('*.tiff'))
    for file in tiff_files:
        file_name = file.stem
        with tifffile.TiffFile(file) as tiff:
            voxel_space = tiff.asarray()
            # Soft fallback for TIFFs missing X/YResolution metadata: assume
            # the source is already at the target voxel size (zoom_factor=1.0)
            # rather than silently shrinking 95%. The TODO in get_voxel_size
            # tracks the affected files.
            voxel_size = get_voxel_size(tiff, _path=file, _default=_target_size)
            metadata = tiff.imagej_metadata or tiff.metadata

            has_labels = voxel_space.shape[1] == 4

            zoom_factors = (_target_size[0] / voxel_size[0],
                            _target_size[1] / voxel_size[1])
            nephrin_stack = voxel_space[:, 0, :, :]
            collagen4_stack = voxel_space[:, 1, :, :]
            wga_stack = voxel_space[:, 2, :, :]

            if has_labels:
                labels_stack = voxel_space[:, 3, :, :]

            resized_nephrin_stack = None
            resized_collagen4_stack = None
            resized_wga_stack = None
            resized_labels_stack = None

            for i in range(voxel_space.shape[0]):
                nephrin = nephrin_stack[i, :, :]
                collagen4 = collagen4_stack[i, :, :]
                wga = wga_stack[i, :, :]

                if has_labels:
                    labels = labels_stack[i, :, :]

                nephrin = zoom(nephrin, zoom_factors, order=1, prefilter=False)
                collagen4 = zoom(collagen4, zoom_factors, order=1, prefilter=False)
                wga = zoom(wga, zoom_factors, order=1, prefilter=False)

                if has_labels:
                    labels = zoom(labels, zoom_factors, order=1, prefilter=False)
                    threshold = 0.5
                    labels[labels >= threshold] = 255
                    labels[labels < threshold] = 0

                if resized_nephrin_stack is None:
                    resized_shape = (voxel_space.shape[0],
                                     nephrin.shape[0],
                                     nephrin.shape[1])
                    resized_nephrin_stack = np.zeros(resized_shape, dtype=np.float32)
                    resized_collagen4_stack = np.zeros(resized_shape, dtype=np.float32)
                    resized_wga_stack = np.zeros(resized_shape, dtype=np.float32)
                    if has_labels:
                        resized_labels_stack = np.zeros(resized_shape, dtype=np.float32)

                resized_nephrin_stack[i, :, :] = nephrin
                resized_collagen4_stack[i, :, :] = collagen4
                resized_wga_stack[i, :, :] = wga
                if has_labels:
                    resized_labels_stack[i, :, :] = labels

            if has_labels:
                image_data = np.stack([resized_nephrin_stack,
                                       resized_collagen4_stack,
                                       resized_wga_stack,
                                       resized_labels_stack], axis=1)
            else:
                image_data = np.stack([resized_nephrin_stack,
                                       resized_collagen4_stack,
                                       resized_wga_stack], axis=1)

            if _z_scale_factor > 1:
                image_data = _z_interpolate_and_stack(
                    image_data, _z_scale_factor, has_labels)
                # Update ImageJ spacing so downstream tools (and an eventual
                # re-load by `get_voxel_size`) see the new, finer Z step.
                if isinstance(metadata, dict) and metadata.get('spacing'):
                    try:
                        metadata['spacing'] = (float(metadata['spacing'])
                                               / _z_scale_factor)
                    except (TypeError, ValueError):
                        pass

        file_path = Path(_dest_dir) / f"{file_name}.tiff"
        tifffile.imwrite(file_path,
                         image_data,
                         shape=image_data.shape,
                         imagej=True,
                         metadata=metadata,
                         compression='lzw')


def copy_directory(_source_dir, _dest_dir, _exclude_list: list):
    # Skip Python bytecode / test caches at every nesting level — they only
    # bloat the experiment's code/ snapshot. `ignore_dangling_symlinks`
    # makes copytree skip (rather than raise on) broken symlinks: wandb
    # writes `policy='live'` symlinks to snapshot files that may already be
    # gone, and following one of those would abort the whole copy.
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    for item in os.listdir(_source_dir):
        source_path = os.path.join(_source_dir, item)
        dest_path = os.path.join(_dest_dir, item)
        if item not in _exclude_list:
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path,
                                ignore=ignore,
                                ignore_dangling_symlinks=True)
            else:
                shutil.copy2(source_path, dest_path)


def to_numpy(_gpu_tensor):
    if torch.is_tensor(_gpu_tensor):
        return _gpu_tensor.clone().detach().to('cpu').numpy()
    return _gpu_tensor


def sanity_check(_configs: dict) -> dict:
    assert _configs['trainer']['data']['train_ds']['path'] is not None, \
        "Please provide path to the training dataset"

    if _configs['trainer']['logging']['visualization']['enabled']:
        assert (_configs['trainer']['logging']['visualization']['path']
                is not None), \
            "Please provide path to store visualization files"

    if torch.cuda.device_count() == 0:
        _configs['trainer']['runtime']['device'] = 'cpu'
        _configs['trainer']['runtime']['mixed_precision'] = False
        _configs['inference']['device'] = 'cpu'

    if _configs['trainer']['runtime']['device'] == 'cpu':
        _configs['trainer']['runtime']['cudnn_benchmark'] = False

    return _configs


REMOVED_CONFIG_KEYS: list[str] = [
    "experiments.model_sizes",
    "experiments.optimizers",
    "experiments.metrics",
    "experiments.train_same_sample_size",
    "experiments.train_same_batch_size",
    "experiments.train_same_stride",
    "logging.log_levels",
    "trainer.tensorboard",
    "trainer.sqlite",
    "trainer.profiling",
    "trainer.logging.visualization.blender",
]


def _has_dotted_key(configs: dict, dotted: str) -> bool:
    node = configs
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _check_removed_keys(configs: dict) -> None:
    """Raise on stale config fields that no longer exist in the schema.

    Old experiment YAMLs may carry keys whose code paths have been deleted
    (tensorboard, sqlite, profiling, etc.). Letting them load silently risks
    a future caller accidentally reading them again; surface them loudly.
    """
    found = [k for k in REMOVED_CONFIG_KEYS if _has_dotted_key(configs, k)]
    if found:
        raise ValueError(
            "Config contains removed keys: " + ", ".join(found) +
            ". Strip them or update from configs/template.yaml.")


def read_configs(_config_path: str):
    with open(_config_path, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)
    _check_removed_keys(configs)
    return sanity_check(configs)


def summerize_configs(_configs: dict) -> None:
    with StringIO() as configs_dump:
        yaml.dump(_configs, configs_dump,
                  default_flow_style=None, sort_keys=False)
        logging.info("Configurations\n%s******************",
                     configs_dump.getvalue())


def morph_analysis(_sample_path: str, _morph) -> None:
    sample = Path(_sample_path)
    logging.info("Executing morphometric analysis for %s", str(sample))
    input_path = sample / "prediction_psp.npz"
    distance_path = sample / "distance_result.npz"
    psf_path = sample / "psf_result.npz"
    clamp_path = sample / "psf_clamp_stats.yaml"

    result = np.load(input_path)['arr']
    distance_result, psf_result, _, clamp_info = _morph(torch.from_numpy(result).float())

    distance_result = distance_result.detach().cpu().numpy()
    psf_result = psf_result.detach().cpu().numpy()

    np.savez_compressed(distance_path, arr=distance_result)
    np.savez_compressed(psf_path, arr=psf_result)

    with open(clamp_path, "w", encoding="UTF-8") as f:
        yaml.safe_dump(clamp_info, f, sort_keys=False)
