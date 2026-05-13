# Python Imports
import logging
import sys
import os
import shutil
import subprocess
import shlex
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
from scipy.ndimage import binary_dilation, binary_erosion
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
    logging.getLogger("pudb").setLevel(logging.INFO)
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
    input_path = sample / "prediction_psp.npz"
    distance_path = sample / "distance_result.npz"
    psf_path = sample / "psf_result.npz"
    # fd_path = sample / "fd_result.npy"

    result = np.load(input_path)['arr']

    distance_result, psf_result, _ = _morph(torch.from_numpy(result).float())

    distance_result = distance_result.detach().cpu().numpy()
    psf_result = psf_result.detach().cpu().numpy()

    np.savez_compressed(distance_path, arr=distance_result)

    np.savez_compressed(psf_path, arr=psf_result)

def blender_visualization(_distance_results: array,
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
                                                           # step_size=1.1,
                                                           allow_degenerate=False)

    np.savez_compressed(os.path.join(_output_path, "verts_distance.npz"), arr=verts)
    np.savez_compressed(os.path.join(_output_path, "faces_distance.npz"), arr=faces)
    np.savez_compressed(os.path.join(_output_path, "values_distance.npz"), arr=values)

    # mean = np.mean(_psf_results[_psf_results != 0])
    # first_layer = _psf_results[0]
    # first_layer[first_layer != 0] = mean/2
    # _psf_results[0] = first_layer

    # last_layer = _psf_results[_psf_results.shape[0]-1]
    # last_layer[last_layer != 0] = mean/2
    # _psf_results[_psf_results.shape[0]-1] = last_layer

    # _psf_results = np.pad(_psf_results, pad_width, mode="constant", constant_values=0)
    # verts, faces, normals, values = measure.marching_cubes(volume=_psf_results,
    #                                                        level=0.1,
    #                                                        step_size=1.1,
    #                                                        allow_degenerate=False)

    # np.save(os.path.join(_output_path, "verts_bumpiness.npy"), verts)
    # np.save(os.path.join(_output_path, "faces_bumpiness.npy"), faces)
    # np.save(os.path.join(_output_path, "values_distance.npy"), values)

def replace_outliers_iqr(arr, k=1.5, lower_p=5, upper_p=95, lower_p_zero_iqr=2, upper_p_zero_iqr=98):
    original_size = arr.size
    q1 = np.percentile(arr, lower_p)
    q3 = np.percentile(arr, upper_p)
    iqr = q3 - q1
    if iqr == 0:
        logging.info("IQR is 0, using percentiles for outlier detection.")
        lower_bound = np.percentile(arr, lower_p_zero_iqr)
        upper_bound = np.percentile(arr, upper_p_zero_iqr)
    else:
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
    logging.info(f"Outlier replacement: q1={q1}, q3={q3}, iqr={iqr}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    arr_copy = arr.copy()
    outliers_upper = arr_copy > upper_bound
    replaced_count = np.sum(outliers_upper)
    percentage_replaced = (replaced_count / original_size) * 100 if original_size > 0 else 0
    arr_copy[outliers_upper] = upper_bound
    logging.info(f"Replaced {replaced_count} outliers out of {original_size} values ({percentage_replaced:.2f}%).")
    return arr_copy

def remove_outliers_iqr(arr, k=1.5):
    original_size = len(arr)
    if original_size == 0:
        return arr
    q1 = np.percentile(arr, 5)
    q3 = np.percentile(arr, 95)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    logging.info(f"Outlier removal: q1={q1}, q3={q3}, iqr={iqr}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    clean_arr = arr[(arr <= upper_bound)]
    removed_count = original_size - len(clean_arr)
    percentage_removed = (removed_count / original_size) * 100
    logging.info(f"Removed {removed_count} outliers out of {original_size} values ({percentage_removed:.2f}%).")
    return clean_arr

def blender_prepare(_sample_dir: str) -> None:
    sample = Path(_sample_dir)

    distance_result = np.load(sample / "distance_result.npz")['arr']
    # psf_result = np.load(sample / "psf_result.npy")
    blender_visualization(_distance_results=distance_result,
                          _output_path=sample / "blender/")


def save_histogram(_array,
                   _title,
                   _path,
                   _bins):

    if _array.size < 2 or np.max(_array) == np.min(_array):
        # Not enough data or range to create bins, use nbinsx as a fallback
        bin_config = dict(nbinsx=_bins)
    else:
        # To force the number of bins, we calculate the bin size
        bin_size = (np.max(_array) - np.min(_array)) / _bins
        bin_config = dict(xbins=dict(
            start=np.min(_array),
            end=np.max(_array),
            size=bin_size
        ))

    # Create histogram
    fig = go.Figure(data=[go.Histogram(
        x=_array,
        **bin_config,
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
    logging.info(f"Saving histogram: {_path}")
    fig.write_image(_path, width=1400, height=1000)  # requires kaleido package


def save_polar_plot(_angles, _thickness_values, _title, _path):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=_thickness_values,
        theta=_angles,
        mode='lines',
        line_color='blue',
        line_width=2,
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))

    fig.update_layout(
        title=_title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Average Thickness (nm)",
                tickangle=0,
                dtick=400
            ),
            angularaxis=dict(
                visible=True,
                direction="clockwise",
                period=360,
                dtick=45,  # Show ticks every 45 degrees
                rotation=0  # Set 0 degrees to the right
            )
        )
    )
    logging.info(f"Saving polar plot: {_path}")
    fig.write_image(_path, width=1400, height=1000)

def calculate_cylindrical_analysis(_data, _alpha_step, _radius):
    z_dim, y_dim, x_dim = _data.shape
    center_y, center_x = y_dim // 2, x_dim // 2

    # 1. Create angle bins
    angle_bins = np.arange(0, 360 + _alpha_step, _alpha_step)
    # The actual angles for plotting will be the midpoint of each bin
    angles_for_plot = (angle_bins[:-1] + angle_bins[1:]) / 2

    # 2. Create a list to hold thickness values for each bin
    binned_thickness_values = [[] for _ in range(len(angle_bins) - 1)]

    # 3. Create coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(x_dim), np.arange(y_dim))

    # 4. Vectorized calculation of radius and angle
    dx = x_coords - center_x
    dy = y_coords - center_y
    radius_map = np.sqrt(dx**2 + dy**2)
    angle_map_rad = np.arctan2(dy, dx)  # Range is -pi to pi
    angle_map_deg = np.rad2deg(angle_map_rad)
    angle_map_deg[angle_map_deg < 0] += 360  # Convert to 0-360 range

    # 5. Create a mask for pixels within the specified radius
    radius_mask = radius_map <= _radius

    # 6. Digitize angles into bins
    angle_bin_indices = np.digitize(angle_map_deg, bins=angle_bins) - 1
    # Handle the edge case for 360 degrees, which should be in the last bin
    angle_bin_indices[angle_bin_indices == len(angle_bins) - 1] = len(angle_bins) - 2

    # 7. Collect all valid points across all Z-slices
    all_valid_thicknesses = []
    all_valid_angle_bins = []
    for z in range(z_dim):
        z_slice = _data[z, :, :]
        valid_mask = (z_slice > 0) & radius_mask

        if np.any(valid_mask):
            all_valid_thicknesses.append(z_slice[valid_mask])
            all_valid_angle_bins.append(angle_bin_indices[valid_mask])

    total_points_found = 0
    if all_valid_thicknesses:
        all_valid_thicknesses = np.concatenate(all_valid_thicknesses)
        all_valid_angle_bins = np.concatenate(all_valid_angle_bins)
        total_points_found = len(all_valid_thicknesses)

        total_points_binned_incremental = 0
        # Group all collected points into their respective bins
        for i in range(len(binned_thickness_values)):
            bin_mask = all_valid_angle_bins == i
            if np.any(bin_mask):
                points_to_add = all_valid_thicknesses[bin_mask]
                binned_thickness_values[i].extend(points_to_add)
                total_points_binned_incremental += len(points_to_add)  # Increment counter

    # Use the incremental count for validation
    if total_points_found == total_points_binned_incremental:
        logging.debug(
            f"Point count validation successful: "
            f"Total points found ({total_points_found}) = Total points incrementally binned ({total_points_binned_incremental})."
        )
    else:
        logging.warning(
            f"Point count mismatch: "
            f"Total points found ({total_points_found}) != Total points incrementally binned ({total_points_binned_incremental}). "
            f"Some points were lost during binning."
        )

    # 8. Calculate average for each bin
    avg_thickness_per_angle = []
    for i, values in enumerate(binned_thickness_values):
        if values:
            num_points = len(values)
            avg_thickness = np.mean(values)
            std_thickness = np.std(values)

            log_angle_start = angle_bins[i]
            log_angle_end = angle_bins[i+1]

            logging.debug(
                f"Cylindrical slice @ {log_angle_start}-{log_angle_end} deg: "
                f"Points={num_points}, "
                f"Avg={avg_thickness:.2f}, "
                f"Std={std_thickness:.2f}"
            )
            avg_thickness_per_angle.append(avg_thickness)
        else:
            log_angle_start = angle_bins[i]
            log_angle_end = angle_bins[i+1]
            logging.debug(f"Cylindrical slice @ {log_angle_start}-{log_angle_end} deg: Points=0")
            avg_thickness_per_angle.append(0)

    # Close the loop for plotting
    if avg_thickness_per_angle:
        angles_for_plot = np.append(angles_for_plot, angles_for_plot[0])
        avg_thickness_per_angle.append(avg_thickness_per_angle[0])

    return angles_for_plot, avg_thickness_per_angle

def save_top_down_view_aspect_ratio(_data, _title, _path, _lower_percentile_iqr, _upper_percentile_iqr, _lower_percentile_iqr_zero, _upper_percentile_iqr_zero):
    # Project the 3D data onto a 2D plane by taking the max along the Z-axis
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(top_down_data, lower_p=_lower_percentile_iqr, upper_p=_upper_percentile_iqr, lower_p_zero_iqr=_lower_percentile_iqr_zero, upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)  # Vertical flip (up-down)

    fig = go.Figure(data=go.Heatmap(
        z=top_down_data,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title=_title,
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    logging.info(f"Saving top-down view plot with aspect ratio: {_path}")
    fig.write_image(_path, width=1000, height=1000)

    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(top_down_data,
                                         lower_p=_lower_percentile_iqr,
                                         upper_p=_upper_percentile_iqr,
                                         lower_p_zero_iqr=_lower_percentile_iqr_zero,
                                         upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)  # Vertical flip (up-down)

    fig = go.Figure(data=go.Heatmap(
        z=top_down_data,
        colorscale='Viridis'
    ))

    fig.update_layout(

        title=_title,

        xaxis_title='X-axis',

        yaxis_title='Y-axis',

        yaxis_scaleanchor="x",

        yaxis_scaleratio=1,

        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),

        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),

        margin=dict(l=0, r=0, t=40, b=0)

    )

    logging.info(f"Saving top-down view plot with aspect ratio: {_path}")
    fig.write_image(_path, width=1000, height=1000)


def save_combined_view(_data,
                       _title,
                       _path,
                       _angles,
                       _radius,
                       _avg_thickness_per_angle,
                       _lower_percentile_iqr,
                       _upper_percentile_iqr,
                       _lower_percentile_iqr_zero,
                       _upper_percentile_iqr_zero):
    # Project the 3D data onto a 2D plane by taking the max along the Z-axis
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(top_down_data, lower_p=_lower_percentile_iqr, upper_p=_upper_percentile_iqr, lower_p_zero_iqr=_lower_percentile_iqr_zero, upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)  # Vertical flip (up-down)

    fig = go.Figure(data=go.Heatmap(
        z=top_down_data,
        colorscale='Viridis'
    ))

    # Add cylindrical analysis overlay
    center_y, center_x = top_down_data.shape[0] // 2, top_down_data.shape[1] // 2

    # Add circle for the radius
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=center_x - _radius, y0=center_y - _radius,
        x1=center_x + _radius, y1=center_y + _radius,
        line_color="rgba(255,0,0,0.5)",
        line_width=2
    )

    # Add lines for the angles (single color, transparent)
    for i, angle in enumerate(_angles):
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=center_x, y0=center_y,
            x1=center_x + _radius * np.cos(np.deg2rad(-angle)),
            y1=center_y + _radius * np.sin(np.deg2rad(-angle)),
            line_color="rgba(128,128,128,0.3)",  # Light gray with transparency
            line_width=1
        )

    # Add the thickness data as a line
    # Normalize the thickness values to fit within the radius
    max_thickness = np.max(_avg_thickness_per_angle)
    normalized_thickness = (_avg_thickness_per_angle / max_thickness) * _radius

    # Convert polar to cartesian coordinates
    x_coords = center_x + normalized_thickness * np.cos(np.deg2rad(-_angles))
    y_coords = center_y + normalized_thickness * np.sin(np.deg2rad(-_angles))

    # Add the line trace
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', line=dict(color='white', width=2)))

    # Add annotations for average thickness
    # We iterate over `_angles[:-1]` to avoid duplicating the label for the 0/360 degree angle
    for i, angle in enumerate(_angles[:-1]):
        avg_thickness = _avg_thickness_per_angle[i]

        # Position the text slightly inside the main radius
        text_radius = _radius * 0.9
        text_x = center_x + text_radius * np.cos(np.deg2rad(-angle))
        text_y = center_y + text_radius * np.sin(np.deg2rad(-angle))

        fig.add_annotation(
            x=text_x,
            y=text_y,
            text=f"<b>{avg_thickness:.0f} nm</b>",
            showarrow=False,
            font=dict(
                size=12,
                color="white"
            ),
            textangle=angle,  # Set text angle to match the line angle exactly
            xanchor="center",
            yanchor="middle"
        )

    fig.update_layout(
        title=_title,
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    logging.info(f"Saving combined view plot: {_path}")
    fig.write_image(_path, width=1000, height=1000)


def generate_comparative_box_plot(_stats_dir: Path):
    """
    Generates a comparative box plot from pre-calculated summary statistics.
    """
    summary_file = _stats_dir / "summary_statistics.npz"
    if not summary_file.exists():
        logging.error("Summary statistics file not found: %s", summary_file)
        return

    summary_data = np.load(summary_file, allow_pickle=True)['arr']

    fig = go.Figure()

    for sample_stats in summary_data:
        fig.add_trace(go.Box(
            x=[sample_stats['sample_name']],  # Assign x-axis category
            name=sample_stats['sample_name'],
            q1=[sample_stats['q1']],
            median=[sample_stats['median']],
            q3=[sample_stats['q3']],
            lowerfence=[sample_stats['lowerfence']],
            upperfence=[sample_stats['upperfence']],
            mean=[sample_stats['mean']],
            sd=[sample_stats['std']]
        ))

    fig.update_layout(title_text="Comparative Box Plot of Thickness Across Samples",
                      yaxis_title="Thickness (nm)",
                      showlegend=False)

    plot_path = _stats_dir / "comparative_box_plot.png"
    logging.info(f"Saving comparative box plot from summary: {plot_path}")
    fig.write_image(plot_path, width=1400, height=1000)


def detect_group_type(_sample_name):
    """
    Identifies the group type based on string prefixes.
    """
    # Mapping group types to a tuple of possible prefixes
    mapping = {
        "Control": ("NCW.AUY381", "NCW.AUY380", "CKM103", "CKM110"),
        "Podocin": ("NCW.BDP669", "NCW.BDP672", "NCW.BDP675"),
        "Collagen": ("NCW.CKM105", "NCW.CKM104"),
    }

    # Clean the input to ensure minor typos or casing don't break the check
    input_clean = _sample_name.strip().upper()

    for group_type, prefixes in mapping.items():
        if input_clean.startswith(prefixes):
            return group_type

    return "Control"


def calculate_stats(_inference_result_path: Path,
                    _stats_dir: Path,
                    _clipping: bool):

    _alpha_step = 10
    _radius = 1000
    _lower_percentile_iqr = 5
    _upper_percentile_iqr = 95
    _lower_percentile_iqr_zero = 2
    _upper_percentile_iqr_zero = 98

    logging.info("Starting statistical analysis for inference results")
    logging.info("Input path: %s", _inference_result_path)
    logging.info("Output path: %s", _stats_dir)

    # Create stats directory if it doesn't exist
    _stats_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Created stats directory: %s", _stats_dir)

    # Define bin sizes for multi-bin histograms
    bin_sizes = [10, 20, 50, 100]
    logging.info("Using bin sizes: %s", bin_sizes)

    # Count total samples for progress tracking
    sample_dirs = [d for d in _inference_result_path.iterdir() if d.is_dir()]
    total_samples = len(sample_dirs)
    logging.info("Found %d samples to process", total_samples)

    # Initialize lists to store aggregated data and summary stats
    summary_data_list = []
    sample_names_for_violin = []
    all_thickness_data_for_violin = []
    all_thickness_data_before_outliers = []

    processed_samples = 0

    # Process each directory (sample) in the inference results
    for sample_dir in sample_dirs:
        logging.info("Processing sample: %s", sample_dir.name)

        # Create a directory for this sample's histograms
        sample_hist_dir = _stats_dir / sample_dir.name
        sample_hist_dir.mkdir(exist_ok=True)
        logging.debug("Created sample histogram directory: %s", sample_hist_dir)

        distance_file = sample_dir / "psf_result.npz"

        if distance_file.exists():
            logging.debug("Processing thickness file: %s", distance_file)
            try:
                data = np.load(distance_file)['arr']
                # data[data == 13] = 0

                if _clipping:
                    max_clipping_value = 1400
                    logging.info("Clipping is enabled, removing values above %d", max_clipping_value)
                    original_non_zero_voxels = np.count_nonzero(data)
                    data[data > max_clipping_value] = 0

                    final_non_zero_voxels = np.count_nonzero(data)
                    altered_voxels = original_non_zero_voxels - final_non_zero_voxels
                    if original_non_zero_voxels > 0:
                        percentage_altered = (altered_voxels / original_non_zero_voxels) * 100
                        logging.info(f"Altered {altered_voxels:,} voxels, which is {percentage_altered:.2f}% of the original non-zero data.")
                    else:
                        logging.info("No non-zero voxels to alter.")

                original_size = np.count_nonzero(data)
                logging.debug(f"Loaded thickness data with {original_size:,} values")

                # Laoding the input file
                sample = tifffile.imread(_inference_result_path / sample_dir.name / "prediction.tif")
                col4_stack = sample[:, 1, :, :]

                logging.debug("Creating the mask for %s", sample_dir.name)
                # Creating the mask from WGA
                sample_group = detect_group_type(sample_dir.name)
                if sample_group == 'Control':
                    mask_threshold = np.percentile(col4_stack, 91)
                elif sample_group == 'Collagen':
                    mask_threshold = np.percentile(col4_stack, 97)
                elif sample_group == 'Podocin':
                    mask_threshold = np.percentile(col4_stack, 92.5)

                mask = col4_stack > mask_threshold
                mask = mask.astype(int)

                for i in range(mask.shape[0]):
                    mask[i] = binary_erosion(mask[i], structure=np.ones((3, 3)))
                    mask[i] = binary_dilation(mask[i], structure=np.ones((3, 3)))

                logging.debug("Applying mask: %s", sample_dir.name)
                logging.debug("Before mask mean: %.4f, std: %.4f", np.mean(data[data != 0]), np.std(data[data != 0]))
                data[mask > 0] = 0
                logging.debug("After mask mean: %.4f, std: %.4f", np.mean(data[data != 0]), np.std(data[data != 0]))

                # Remove zero values
                data_no_zeros = data[data != 0]
                after_zero_removal = np.count_nonzero(data_no_zeros)
                logging.debug("Removed %d zero values, %d values remaining",
                              original_size - after_zero_removal, after_zero_removal)

                # Append data before outlier removal for aggregation
                if len(data_no_zeros) > 0:
                    all_thickness_data_before_outliers.append(data_no_zeros)

                # Remove outliers using IQR method
                data_clean = remove_outliers_iqr(data_no_zeros)
                after_outlier_removal = len(data_clean)
                logging.debug("Removed %d outliers, %d values remaining for histogram",
                              after_zero_removal - after_outlier_removal, after_outlier_removal)

                # Store data for aggregation and summary calculation
                if len(data_clean) > 0:
                    # For violin plot (still needs raw data, but we can decide to remove it later)
                    all_thickness_data_for_violin.append(data_clean)
                    sample_names_for_violin.append(sample_dir.name)

                    # Calculate summary statistics for the box plot
                    q1 = np.percentile(data_clean, _lower_percentile_iqr)
                    median = np.median(data_clean)
                    q3 = np.percentile(data_clean, _upper_percentile_iqr)
                    iqr = q3 - q1
                    lowerfence = max(np.min(data_clean), q1 - 1.5 * iqr)
                    upperfence = min(np.max(data_clean), q3 + 1.5 * iqr)
                    mean = np.mean(data_clean)
                    std = np.std(data_clean)

                    summary_data_list.append({
                        'sample_name': sample_dir.name,
                        'q1': q1, 'median': median, 'q3': q3,
                        'lowerfence': lowerfence, 'upperfence': upperfence,
                        'mean': mean, 'std': std
                    })

                # Create multi-bin histograms for thickness
                for bins in bin_sizes:
                    save_histogram(data_clean,
                                   f'Thickness Histogram with {bins} Bins - {sample_dir.name}',
                                   sample_hist_dir / f'thickness_{sample_dir.name}_{bins}_bins.png',
                                   bins)
                    logging.debug("Created thickness histogram with %d bins", bins)
                # Cylindrical analysis
                angles, avg_thickness = calculate_cylindrical_analysis(data, _alpha_step, _radius)
                save_polar_plot(angles, avg_thickness, f'Cylindrical Analysis - {sample_dir.name}', sample_hist_dir / f'cylindrical_analysis_{sample_dir.name}.png')
                logging.debug("Created cylindrical analysis plot")

                # Top-down view with aspect ratio
                save_top_down_view_aspect_ratio(data,
                                                f'Top-Down View (Aspect Ratio) - {sample_dir.name}',
                                                sample_hist_dir / f'top_down_view_aspect_ratio_{sample_dir.name}.png', _lower_percentile_iqr, _upper_percentile_iqr, _lower_percentile_iqr_zero, _upper_percentile_iqr_zero)
                logging.info("Created top-down view plot with aspect ratio")

                # Combined view
                save_combined_view(data, f'Combined View - {sample_dir.name}', sample_hist_dir / f'combined_view_{sample_dir.name}.png', angles, _radius, avg_thickness, _lower_percentile_iqr, _upper_percentile_iqr, _lower_percentile_iqr_zero, _upper_percentile_iqr_zero)
                logging.debug("Created combined view plot")

            except Exception as e:
                logging.error("Error processing thickness file %s: %s", distance_file, e)
        else:
            logging.warning("Thickness file not found: %s", distance_file)

        processed_samples += 1
        logging.info("Completed processing sample %s (%d/%d)",
                     sample_dir.name, processed_samples, total_samples)

    # Save summary statistics to a .npy file
    if summary_data_list:
        # Define the data type for the structured array
        dtype = [('sample_name', 'U100'), ('q1', 'f8'), ('median', 'f8'), ('q3', 'f8'),
                 ('lowerfence', 'f8'), ('upperfence', 'f8'), ('mean', 'f8'), ('std', 'f8')]

        # Convert list of dicts to list of tuples
        records = [tuple(d.values()) for d in summary_data_list]

        summary_array = np.array(records, dtype=dtype)

        summary_file = _stats_dir / "summary_statistics.npz"
        np.savez_compressed(summary_file, arr=summary_array)
        logging.info("Saved summary statistics to %s", summary_file)

    # Save aggregated raw data points (before outlier removal)
    if all_thickness_data_before_outliers:
        aggregated_thickness_raw = np.concatenate(all_thickness_data_before_outliers)
        logging.info(
            "Aggregated %d raw thickness values from %d samples (before outlier removal)",
            len(aggregated_thickness_raw), len(all_thickness_data_before_outliers)
        )
        raw_thickness_file = _stats_dir / "aggregated_thickness_data.npz"
        np.savez_compressed(raw_thickness_file, arr=aggregated_thickness_raw)
        logging.info("Saved aggregated raw thickness data to %s", raw_thickness_file)

    # Generate the fast comparative box plot from the summary file
    generate_comparative_box_plot(_stats_dir)

    # Save metadata file with sample information
    metadata_file = _stats_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Statistical Analysis Metadata\n")
        f.write("============================\n")
        f.write(f"Input directory: {_inference_result_path}\n")
        f.write(f"Output directory: {_stats_dir}\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Samples with valid data: {len(summary_data_list)}\n")
        f.write(f"Bin sizes used: {bin_sizes}\n")
        f.write(f"Alpha step for cylindrical analysis: {_alpha_step}\n")
        f.write(f"Radius for cylindrical analysis: {_radius}\n")
        f.write("\nSample names:\n")
        for item in summary_data_list:
            f.write(f"  - {item['sample_name']}\n")
        f.write("\nGenerated files:\n")
        if summary_data_list:
            f.write("  - summary_statistics.npy (quartiles, fences, mean, std for each sample)\n")
            f.write("  - comparative_box_plot.png (box plot generated from summary)\n")
        if all_thickness_data_before_outliers:
            f.write("  - aggregated_thickness_data.npy (all thickness values before outlier removal)\n")
        if all_thickness_data_for_violin:
            f.write("  - thickness_violin_plot.png (violin plot of all samples)\n")

    logging.info("Saved metadata to %s", metadata_file)
    logging.info("Statistical analysis completed successfully")
    logging.info("Multi-bin histograms and aggregated thickness data saved in '%s' directory", _stats_dir)

def _recursive_roi_analysis(data,
                            divisions,
                            min_region_size,
                            anomaly_threshold,
                            depth,
                            origin=(0, 0, 0)):
    logging.debug(f"Depth {depth}: Analyzing region at {origin} of size {data.shape}")

    if np.isclose(np.mean(data), 0.00):
        logging.debug(f"Depth {depth}: Mean of region at {origin} is approximately 0.00. Ignoring this region.")
        return []

    # Divide the region into sub-regions
    sub_regions = []
    shape = data.shape
    z_div, y_div, x_div = divisions
    for i in range(z_div):
        for j in range(y_div):
            for k in range(x_div):
                z_start, z_end = shape[0] * i // z_div, shape[0] * (i + 1) // z_div
                y_start, y_end = shape[1] * j // y_div, shape[1] * (j + 1) // y_div
                x_start, x_end = shape[2] * k // x_div, shape[2] * (k + 1) // x_div
                sub_regions.append(data[z_start:z_end, y_start:y_end, x_start:x_end])

    # Calculate stats for the sub-regions
    sub_region_means = [np.mean(sub_region) for sub_region in sub_regions]
    median_of_means = np.median(sub_region_means)
    std_of_means = np.std(sub_region_means)
    logging.debug(f"Depth {depth}: Sub-region stats at {origin}: median_of_means={median_of_means:.4f}, std_of_means={std_of_means:.4f}")

    anomalous_regions_to_remove = []
    # Analyze sub-regions
    for i, sub_region in enumerate(sub_regions):
        sub_mean = sub_region_means[i]
        logging.debug(f"Depth {depth}: Sub-region {i} at {origin}: mean={sub_mean:.4f}")

        if abs(sub_mean - median_of_means) > anomaly_threshold * std_of_means and sub_mean > 1000:
            logging.warning(f"Depth {depth}: Anomaly detected in sub-region {i} at {origin} (mean={sub_mean:.4f} > 1000).")

            if sub_region.size < min_region_size:
                logging.warning(f"Depth {depth}: Region at {origin} is smaller than {min_region_size} and will be removed.")
                # This is a leaf node anomaly, return its coordinates for removal
                z_start, z_end = shape[0] * (i // (y_div * x_div)) // z_div, shape[0] * ((i // (y_div * x_div)) + 1) // z_div
                y_start, y_end = shape[1] * ((i % (y_div * x_div)) // x_div) // y_div, shape[1] * (((i % (y_div * x_div)) // x_div) + 1) // y_div
                x_start, x_end = shape[2] * (i % x_div) // x_div, shape[2] * ((i % x_div) + 1) // x_div
                anomalous_regions_to_remove.append((slice(origin[0] + z_start, origin[0] + z_end),
                                                    slice(origin[1] + y_start, origin[1] + y_end),
                                                    slice(origin[2] + x_start, origin[2] + x_end)))
            else:
                # This is a branch node anomaly, recurse
                z_start, z_end = shape[0] * (i // (y_div * x_div)) // z_div, shape[0] * ((i // (y_div * x_div)) + 1) // z_div
                y_start, y_end = shape[1] * ((i % (y_div * x_div)) // x_div) // y_div, shape[1] * (((i % (y_div * x_div)) // x_div) + 1) // y_div
                x_start, x_end = shape[2] * (i % x_div) // x_div, shape[2] * ((i % x_div) + 1) // x_div
                new_origin = (origin[0] + z_start, origin[1] + y_start, origin[2] + x_start)
                anomalous_regions_to_remove.extend(_recursive_roi_analysis(sub_region,
                                                                           divisions,
                                                                           min_region_size,
                                                                           anomaly_threshold + 0.02,
                                                                           depth + 1,
                                                                           new_origin))
        else:
            logging.debug(f"Depth {depth}: Sub-region {i} at {origin} is normal.")

    return anomalous_regions_to_remove

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
        verts_path = os.path.join(_inference_dir, directory, "blender", "verts_distance.npz")
        faces_path = os.path.join(_inference_dir, directory, "blender", "faces_distance.npz")
        values_path = os.path.join(_inference_dir, directory, "blender", "values_distance.npz")
        result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_distance.blend")
        result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_distance.mp4")
        commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_index}; blender --background --python src/scripts/blender_render.py -- res/blender_template.blend {shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} {shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        gpu_index = (gpu_index + 1) % num_gpus

        # verts_path = os.path.join(_inference_dir, directory, "blender", "verts_bumpiness.npy")
        # faces_path = os.path.join(_inference_dir, directory, "blender", "faces_bumpiness.npy")
        # values_path = os.path.join(_inference_dir, directory, "blender", "values_bumpiness.npy")
        # result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_bumpiness.blend")
        # result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_bumpiness.mp4")
        # commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_index}; blender --background --python src/scripts/blender_render.py -- res/blender_template.blend {shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} {shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        # gpu_index = (gpu_index + 1) % num_gpus

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

        labels = np.load(dir / "prediction_psp.npz")['arr']
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
