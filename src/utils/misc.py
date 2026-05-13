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
                            _configs['trainer']['result_path'],
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
def get_voxel_size(_tiff):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """
    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        return 1.

    image_metadata = _tiff.imagej_metadata
    if image_metadata is not None:
        z = image_metadata.get('spacing', 1.)
    else:
        z = 1.

    tags = _tiff.pages[0].tags
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    return [x, y, z]


def resize_and_copy(_source_dir, _dest_dir, _target_size):
    source_path = Path(_source_dir)
    tiff_files = list(source_path.glob('*.tif')) + list(source_path.glob('*.tiff'))
    for file in tiff_files:
        file_name = file.stem
        with tifffile.TiffFile(file) as tiff:
            voxel_space = tiff.asarray()
            voxel_size = get_voxel_size(tiff)
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

        file_path = Path(_dest_dir) / f"{file_name}.tiff"
        tifffile.imwrite(file_path,
                         image_data,
                         shape=image_data.shape,
                         imagej=True,
                         metadata=metadata,
                         compression='lzw')


def copy_directory(_source_dir, _dest_dir, _exclude_list: list):
    for item in os.listdir(_source_dir):
        source_path = os.path.join(_source_dir, item)
        dest_path = os.path.join(_dest_dir, item)
        if item not in _exclude_list:
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)


def to_numpy(_gpu_tensor):
    if torch.is_tensor(_gpu_tensor):
        return _gpu_tensor.clone().detach().to('cpu').numpy()
    return _gpu_tensor


def sanity_check(_configs: dict) -> dict:
    assert _configs['trainer']['train_ds']['path'] is not None, \
        "Please provide path to the training dataset"

    if _configs['trainer']['visualization']['enabled']:
        assert _configs['trainer']['visualization']['path'] is not None, \
            "Please provide path to store visualization files"

    if torch.cuda.device_count() == 0:
        _configs['trainer']['device'] = 'cpu'
        _configs['trainer']['mixed_precision'] = False
        _configs['inference']['device'] = 'cpu'

    if _configs['trainer']['device'] == 'cpu':
        _configs['trainer']['cudnn_benchmark'] = False
        _configs['trainer']['nvtx_patching'] = False

    return _configs


def read_configs(_config_path: str):
    with open(_config_path, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)
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
