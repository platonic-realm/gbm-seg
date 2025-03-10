# Python Imports
import logging
import sys
import os
import shutil
from pathlib import Path
from io import StringIO
import subprocess
import shlex

# Library Imports
import torch
import yaml
import tifffile
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from scipy.ndimage import zoom, gaussian_filter

# Local Imports
from src.utils.metrics.memory import GPURunningMetrics
from src.utils.metrics.clfication import Metrics


def basic_logger() -> None:
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level='INFO',
                        format=log_format)


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

    logging.basicConfig(level=LOG_LEVEL,
                        format=log_format,
                        handlers=handlers)

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
        # return default
        return 1.

    image_metadata = _tiff.imagej_metadata
    if image_metadata is not None:
        z = image_metadata.get('spacing', 1.)
    else:
        # default voxel size
        z = 1.

    tags = _tiff.pages[0].tags
    # parse X, Y resolution
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    # return voxel size
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
                                     nephrin.shape[1],)
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
                                       resized_labels_stack],
                                      axis=1)
            else:
                image_data = np.stack([resized_nephrin_stack,
                                       resized_collagen4_stack,
                                       resized_wga_stack],
                                      axis=1)

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


def expand_as_one_hot(_input, _c, _ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL,
    where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert _input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    _input = _input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(_input.size())
    shape[1] = _c

    if _ignore_index is not None:
        # create ignore_index mask for the result
        mask = _input.expand(shape) == _ignore_index
        # clone the src tensor and zero out ignore_index in the input
        _input = _input.clone()
        _input[_input == _ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(_input.device).scatter_(1, _input, 1)
        # bring back the ignore_index in the result
        result[mask] = _ignore_index
        return result

    # Else: scatter to get the one-hot tensor
    return torch.zeros(shape).to(_input.device).scatter_(1, _input, 1)


# Check the configurations and changes
# some if needed.
def sanity_check(_configs: dict) -> dict:
    assert not _configs['trainer']['train_ds']['path'] is None, \
           "Please provide path to the training dataset"

    if _configs['trainer']['visualization']['enabled']:
        assert not _configs['trainer']['visualization']['path'] is None, \
               "Please provide path to store visualization files"

    if _configs['trainer']['tensorboard']['enabled']:
        assert not _configs['trainer']['tensorboard']['path'] is None, \
               "Please provide path for tensorboard logs"

    if torch.cuda.device_count() == 0:
        _configs['trainer']['device'] = 'cpu'
        _configs['trainer']['mixed_precision'] = False
        _configs['inference']['device'] = 'cpu'

    if _configs['trainer']['device'] == 'cpu':
        _configs['trainer']['cudnn_benchmark'] = False
        _configs['trainer']['nvtx_patching'] = False

    return _configs


def blind_test(_model: nn.Module,
               _dataloader: DataLoader,
               _loss,
               _device: str,
               _no_of_classes: int,
               _metric_list: list):

    running_metrics = GPURunningMetrics(_device, _metric_list)

    _model.to(_device)

    counter = 0
    for index, data in enumerate(_dataloader):
        counter += 1
        if counter > 20:
            break

        sample = data['sample'].to(_device)
        labels = data['labels'].to(_device).long()

        with torch.no_grad():

            logits, results = _model(sample)

            metrics = Metrics(_no_of_classes,
                              results,
                              labels)
            loss = _loss(logits, labels)

            running_metrics.add(metrics.reportMetrics(_metric_list, loss))

    return running_metrics.calculate()


def read_configs(_config_path: str):
    with open(_config_path, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    return sanity_check(configs)


def summerize_configs(_configs: dict) -> None:
    with StringIO() as configs_dump:
        yaml.dump(_configs,
                  configs_dump,
                  default_flow_style=None,
                  sort_keys=False)
        logging.info("Configurations\n%s******************",
                     configs_dump.getvalue())


# def morph_analysis(_inference_dir: str) -> None:
#
#         if not self.interpolate:
#             repeated_results = np.repeat(result,
#                                          self.scale_factor,
#                                          axis=0)
#
#             distance_result, fd_result = self.morph(torch.from_numpy(repeated_results).float())
#         else:
#             distance_result, fd_result = self.morph(torch.from_numpy(result).float())
#
#         distance_result = distance_result.detach().cpu().numpy()
#         fd_result = fd_result.detach().cpu().numpy()
#
#     distance_npy_path = os.path.join(_output_path, "distance_result.npy")
#     fd_npy_path = os.path.join(_output_path, "fd_result.npy")
#
#     with open(distance_npy_path, 'wb') as distance_npy_file:
#         np.save(distance_npy_file, _distance_results)
#
#     with open(fd_npy_path, 'wb') as fd_npy_file:
#         np.save(fd_npy_file, _fd_results)
#
#
# def blender_prepare(_inference_dir: str) -> None:
#         self.blender_visualization(_distance_results=distance_result,
#                                    _fd_results=fd_result,
#                                    _output_path=os.path.join(output_dir,
#                                                              "blender"))
#

def blender_render(_inference_dir: str) -> None:

    # Generating Blender commands

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    gpu_index = 0
    gpu_capacity = 12

    # List all items in the current directory
    all_files = os.listdir(_inference_dir)

    # Filter out only the directories, excluding '.' and '..'
    directories = [item for item in all_files if os.path.isdir(os.path.join(_inference_dir,
                                                                            item)) and item not in ['.', '..']]
    commands = []
    for directory in directories:
        verts_path = os.path.join(_inference_dir, directory, "blender", "verts_distance.npy")
        faces_path = os.path.join(_inference_dir, directory, "blender", "faces_distance.npy")
        values_path = os.path.join(_inference_dir, directory, "blender", "values_distance.npy")
        result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_distance.blend")
        result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_distance.mp4")
        commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_index}; blender --background --python src/scripts/blender_render.py -- res/blender_template.blend {shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} {shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        gpu_index = (gpu_index + 1) % num_gpus

        verts_path = os.path.join(_inference_dir, directory, "blender", "verts_bumpiness.npy")
        faces_path = os.path.join(_inference_dir, directory, "blender", "faces_bumpiness.npy")
        values_path = os.path.join(_inference_dir, directory, "blender", "values_bumpiness.npy")
        result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_bumpiness.blend")
        result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_bumpiness.mp4")
        commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_index}; blender --background --python src/scripts/blender_render.py -- res/blender_template.blend {shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} {shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        gpu_index = (gpu_index + 1) % num_gpus

    compute_limit = num_gpus * gpu_capacity
    for i in range(int(len(commands)/compute_limit) + 1):
        sub_commands = commands[i*compute_limit:(i+1)*compute_limit]
        processes = [subprocess.Popen(cmd, shell=True, executable='/bin/bash') for cmd in sub_commands]

        # Wait for all processes to complete
        for proc in processes:
            proc.wait()


def analyze_dataset(_dataset):
    def draw(voxels, path, name):
        import matplotlib.pyplot as plt
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(voxels.flatten(), bins=256, edgecolor='black')
        plt.title(f'Histogram of {name} Intensities')
        plt.xlabel('Voxel Intensity')
        plt.ylabel(f'Frequency ({name})')
        plt.savefig(path / f"{name}.png")
        plt.close()

        import imageio
        with imageio.get_writer(path / f"{name}.gif") as writer:
            for index in range(voxels.shape[0]):
                writer.append_data(voxels[index])

    def histogram(_data, _label, _path):
        nephrin = _data[0].cpu().detach().numpy()
        wga = _data[1].cpu().detach().numpy()
        col4 = _data[2].cpu().detach().numpy()
        label = _label.cpu().detach().numpy()
        draw(nephrin, _path, "nephrin")
        draw(wga, _path, "wga")
        draw(col4, _path, "col4")
        draw(label, _path, "label")

    root_path = Path("/data/afatehi/") / "gbm/dataset_analysis/"
    root_path.mkdir(parents=True, exist_ok=True)
    for idx, data in enumerate(_dataset):
        print(f"Item {idx}:")
        path = root_path / f"{idx:04d}"
        path.mkdir(parents=True, exist_ok=True)
        histogram(data['sample'], data['labels'], path)
