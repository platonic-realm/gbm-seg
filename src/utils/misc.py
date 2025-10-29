# Python Imports
import logging
import sys
import os
import shutil
import subprocess
import shlex
import re
from pathlib import Path
from io import StringIO

# Library Imports
import torch
import yaml
import tifffile
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from torch import nn
from scipy.ndimage import zoom
from skimage import measure
from numpy import array
from skimage.transform import resize

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


def morph_analysis(_sample_path: str,
                   _morph) -> None:

    sample = Path(_sample_path)

    logging.info("Executing morphometric analysis for %s",
                 str(sample))
    input_path = sample / "prediction_psp.npy"
    distance_path = sample / "distance_result.npy"
    fd_path = sample / "fd_result.npy"

    result = np.load(input_path)

    distance_result, fd_result = _morph(torch.from_numpy(result).float())

    distance_result = distance_result.detach().cpu().numpy()
    fd_result = fd_result.detach().cpu().numpy()

    with open(distance_path, 'wb') as distance_file:
        np.save(distance_file, distance_result)

    with open(fd_path, 'wb') as fd_file:
        np.save(fd_file, fd_result)


def blender_visualization(_distance_results: array,
                          _fd_results: array,
                          _output_path: str):

    create_dirs_recursively(os.path.join(_output_path, "dummy"))

    mean = np.mean(_distance_results[_distance_results != 0])
    first_layer = _distance_results[0]
    first_layer[first_layer != 0] = mean/2
    _distance_results[0] = first_layer

    last_layer = _distance_results[_distance_results.shape[0]-1]
    last_layer[last_layer != 0] = mean/2
    _distance_results[_distance_results.shape[0]-1] = last_layer

    pad_width = ((1, 1), (1, 1), (1, 1))

    _distance_results = np.pad(_distance_results, pad_width, mode="constant", constant_values=0)
    verts, faces, normals, values = measure.marching_cubes(volume=_distance_results,
                                                           level=0.1,
                                                           step_size=1.1,
                                                           allow_degenerate=False)

    np.save(os.path.join(_output_path, "verts_distance.npy"), verts)
    np.save(os.path.join(_output_path, "faces_distance.npy"), faces)
    np.save(os.path.join(_output_path, "values_distance.npy"), values)

    mean = np.mean(_fd_results[_fd_results != 0])
    first_layer = _fd_results[0]
    first_layer[first_layer != 0] = mean/2
    _fd_results[0] = first_layer

    last_layer = _fd_results[_fd_results.shape[0]-1]
    last_layer[last_layer != 0] = mean/2
    _fd_results[_fd_results.shape[0]-1] = last_layer

    _fd_results = np.pad(_fd_results, pad_width, mode="constant", constant_values=0)
    verts, faces, normals, values = measure.marching_cubes(volume=_fd_results,
                                                           level=0.1,
                                                           step_size=1.1,
                                                           allow_degenerate=False)

    np.save(os.path.join(_output_path, "verts_bumpiness.npy"), verts)
    np.save(os.path.join(_output_path, "faces_bumpiness.npy"), faces)
    np.save(os.path.join(_output_path, "values_bumpiness.npy"), values)


def blender_prepare(_sample_dir: str) -> None:
    sample = Path(_sample_dir)

    distance_result = np.load(sample / "distance_result.npy")
    fd_result = np.load(sample / "fd_result.npy")
    blender_visualization(_distance_results=distance_result,
                          _fd_results=fd_result,
                          _output_path=sample / "blender/")


def remove_outliers_iqr(arr, k=1.5):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return arr[(arr >= lower_bound) & (arr <= upper_bound)]


def save_histogram(_array,
                   _title,
                   _path):
    # Create histogram
    fig = go.Figure(data=[go.Histogram(
        x=_array,
        nbinsx=100,
        marker=dict(
            line=dict(color='black', width=1)
        )
    )])

    # Update layout
    fig.update_layout(
        title=_title,
        xaxis_title='Value',
        yaxis_title='Frequency'
    )

    # Save the plot
    fig.write_image(_path)  # requires kaleido package


def calculate_stats(_inference_result_path: Path,
                    _inference_export_path: Path):

    stats_path = _inference_export_path / "stats"
    hists_path = stats_path / "histograms"
    value_path = stats_path / "values"
    create_dirs_recursively(hists_path / "dummy")

    # Specific patterns for each category based on your list
    patterns = {
        'Control': r'CKM103|AUY381|AUY380|CKM110',
        'Col Mutation': r'CKM105|CKM104',
        'Pod Mutation': r'BDP675|BDP672|BDP669',
    }

    control_thickness = np.empty(0, dtype=np.float32)
    control_bumpiness = np.empty(0, dtype=np.float32)

    colmut_thickness = np.empty(0, dtype=np.float32)
    colmut_bumpiness = np.empty(0, dtype=np.float32)

    podmut_thickness = np.empty(0, dtype=np.float32)
    podmut_bumpiness = np.empty(0, dtype=np.float32)

    for dir in _inference_result_path.iterdir():
        if not dir.is_dir():
            continue

        for category, pattern in patterns.items():
            if re.search(pattern, dir.name):
                group_name = category

        if not group_name:
            raise RuntimeError("Unkown mice category!")

        sample_hists_path = hists_path / group_name
        sample_value_path = value_path / group_name
        create_dirs_recursively(sample_hists_path / "dummy")
        create_dirs_recursively(sample_value_path / "dummy")

        thickness = np.load(dir / "distance_result.npy")
        bumpiness = np.load(dir / "fd_result.npy")

        # Setting the first 5 (averaging kernel size) layers to zero because of sudden gradient shift in the last layers
        bumpiness[:5, :, :] = 0
        bumpiness[-5:, :, :] = 0

        # Remove zeros
        thickness = thickness[thickness != 0]
        bumpiness = bumpiness[bumpiness != 0]

        thickness = remove_outliers_iqr(thickness)
        bumpiness = remove_outliers_iqr(bumpiness)

        if group_name == 'Control':
            control_thickness = np.concatenate((control_thickness, thickness))
            control_bumpiness = np.concatenate((control_bumpiness, bumpiness))
        elif group_name == 'Col Mutation':
            colmut_thickness = np.concatenate((colmut_thickness, thickness))
            colmut_bumpiness = np.concatenate((colmut_bumpiness, bumpiness))
        elif group_name == 'Pod Mutation':
            podmut_thickness = np.concatenate((podmut_thickness, thickness))
            podmut_bumpiness = np.concatenate((podmut_bumpiness, bumpiness))

        save_histogram(thickness,
                       'Histogram of thickness',
                       sample_hists_path / f"thickness_{dir.name}.png")

        save_histogram(bumpiness,
                       'Histogram of bumpiness',
                       sample_hists_path / f"bumpiness_{dir.name}.png")

        np.save(sample_value_path / f"thickness_{dir.name}.npy", thickness)
        np.save(sample_value_path / f"bumpiness_{dir.name}.npy", bumpiness)

        np.savetxt(sample_value_path / f"thickness_{dir.name}.csv",
                   thickness,
                   delimiter=',',
                   fmt='%.4f')
        np.savetxt(sample_value_path / f"bumpiness_{dir.name}.csv",
                   bumpiness,
                   delimiter=',',
                   fmt='%.4f')

    remove_outliers_iqr(control_thickness)
    remove_outliers_iqr(control_bumpiness)
    remove_outliers_iqr(podmut_thickness)
    remove_outliers_iqr(podmut_bumpiness)
    remove_outliers_iqr(colmut_thickness)
    remove_outliers_iqr(colmut_bumpiness)

    save_histogram(control_thickness,
                   'Histogram of thickness',
                   hists_path / "thickness_control.png")

    save_histogram(control_bumpiness,
                   'Histogram of bumpiness',
                   hists_path / "bumpiness_control.png")

    save_histogram(podmut_thickness,
                   'Histogram of thickness',
                   hists_path / "thickness_podmut.png")

    save_histogram(podmut_bumpiness,
                   'Histogram of bumpiness',
                   hists_path / "bumpiness_podmut.png")

    save_histogram(colmut_thickness,
                   'Histogram of thickness',
                   hists_path / "thickness_colmut.png")

    save_histogram(colmut_bumpiness,
                   'Histogram of bumpiness',
                   hists_path / "bumpiness_colmut.png")

    np.save(value_path / "thickness_control.npy", control_thickness)
    np.save(value_path / "bumpiness_control.npy", control_bumpiness)
    np.save(value_path / "thickness_podmut.npy", podmut_thickness)
    np.save(value_path / "bumpiness_podmut.npy", podmut_bumpiness)
    np.save(value_path / "thickness_colmut.npy", colmut_thickness)
    np.save(value_path / "bumpiness_colmut.npy", colmut_bumpiness)

    np.savetxt(value_path / "thickness_control.csv",
               control_thickness, delimiter=',', fmt='%.4f')

    np.savetxt(sample_value_path / "bumpiness_control.csv",
               control_bumpiness, delimiter=',', fmt='%.4f')

    np.savetxt(value_path / "thickness_podmut.csv",
               podmut_thickness, delimiter=',', fmt='%.4f')

    np.savetxt(value_path / "bumpiness_podmut.csv",
               podmut_bumpiness, delimiter=',', fmt='%.4f')

    np.savetxt(value_path / "thickness_colmut.csv",
               colmut_thickness, delimiter=',', fmt='%.4f')

    np.savetxt(value_path / "bumpiness_colmut.csv",
               colmut_bumpiness, delimiter=',', fmt='%.4f')


def blender_render(_inference_dir: str) -> None:

    # Generating Blender commands

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    gpu_index = 0
    gpu_capacity = 5

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


def export_results(_inference_result_path: Path,
                   _inference_export_path: Path):

    scale_factor = 1/3

    for dir in _inference_result_path.iterdir():
        if not dir.is_dir():
            continue

        name = dir.name

        tiff_file = _inference_export_path / "tiff" / name
        thickness_blend = _inference_export_path / "blend" / f"{name} _thickness.blend"
        thickness_mp4 = _inference_export_path / "mp4" / f"{name}_thickness.mp4"
        bumpiness_blend = _inference_export_path / "blend" / f"{name}_bumpiness.blend"
        bumpiness_mp4 = _inference_export_path / "mp4" / f"{name}_bumpiness.mp4"

        create_dirs_recursively(tiff_file)
        create_dirs_recursively(thickness_mp4)
        create_dirs_recursively(thickness_blend)

        shutil.copy(dir / "blender/result_distance.blend", thickness_blend)
        shutil.copy(dir / "blender/result_distance.mp4", thickness_mp4)

        shutil.copy(dir / "blender/result_bumpiness.blend", bumpiness_blend)
        shutil.copy(dir / "blender/result_bumpiness.mp4", bumpiness_mp4)

        labels = np.load(dir / "prediction_psp.npy")
        labels[labels != 0] = 128

        with tifffile.TiffFile(dir / "prediction.tif") as tif:
            data = tif.asarray()

        data[:, 3, :, :] = labels

        # Assuming data shape is (z, channels, y, x)
        if data.ndim != 4:
            raise ValueError("Expected a 4D TIFF file with shape (channels, z, y, x)")

        z, channels, y, x = data.shape
        new_y, new_x = int(y * scale_factor), int(x * scale_factor)

        # Downsample each image
        downsampled = np.empty((z, channels, new_y, new_x), dtype=data.dtype)
        for i in range(z):
            for c in range(channels):
                downsampled[i, c] = resize(data[i, c],
                                           (new_y, new_x),
                                           preserve_range=True,
                                           anti_aliasing=True).astype(data.dtype)

        tifffile.imwrite(tiff_file,
                         downsampled,
                         shape=downsampled.shape,
                         imagej=True,
                         metadata={'axes': 'ZCYX', 'fps': 10.0},
                         compression='lzw')


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
