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

def replace_outliers_iqr(arr, k=1.5):
    original_size = arr.size
    q1 = np.percentile(arr, 5)
    q3 = np.percentile(arr, 95)
    iqr = q3 - q1
    if iqr == 0:
        logging.info("IQR is 0, using percentiles for outlier detection.")
        lower_bound = np.percentile(arr, 2)
        upper_bound = np.percentile(arr, 98)
    else:
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
    logging.info(f"Outlier replacement: q1={q1}, q3={q3}, iqr={iqr}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    arr_copy = arr.copy()
    outliers_upper = arr_copy > upper_bound
    replaced_count = np.sum(outliers_upper)
    arr_copy[outliers_upper] = upper_bound
    logging.info(f"Replaced {replaced_count} outliers out of {original_size} values.")
    return arr_copy

def remove_outliers_iqr(arr, k=1.5):
    original_size = len(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    logging.info(f"Outlier removal: q1={q1}, q3={q3}, iqr={iqr}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    clean_arr = arr[(arr <= upper_bound)]
    removed_count = original_size - len(clean_arr)
    logging.info(f"Removed {removed_count} outliers out of {original_size} values.")
    return clean_arr

def blender_prepare(_sample_dir: str) -> None:
    sample = Path(_sample_dir)

    distance_result = np.load(sample / "distance_result.npy")
    fd_result = np.load(sample / "fd_result.npy")
    blender_visualization(_distance_results=distance_result,
                          _fd_results=fd_result,
                          _output_path=sample / "blender/")


def save_histogram(_array,
                   _title,
                   _path,
                   _bins):
    # Create histogram
    fig = go.Figure(data=[go.Histogram(
        x=_array,
        nbinsx=_bins,
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
                rotation=0 # Set 0 degrees to the right
            )
        )
    )
    logging.info(f"Saving polar plot: {_path}")
    fig.write_image(_path, width=1400, height=1000)

def calculate_cylindrical_analysis(_data, _alpha_step, _radius):
    z_dim, y_dim, x_dim = _data.shape
    center_y, center_x = y_dim // 2, x_dim // 2

    angles = np.arange(0, 360, _alpha_step)
    avg_thickness_per_angle = []

    for angle in angles:
        thickness_values = []
        for r in range(1, int(_radius) + 1):
            for z in range(z_dim):
                y = int(center_y + r * np.sin(np.deg2rad(angle)))
                x = int(center_x + r * np.cos(np.deg2rad(angle)))

                if 0 <= y < y_dim and 0 <= x < x_dim:
                    thickness = _data[z, y, x]
                    if thickness > 0:
                        thickness_values.append(thickness)

        if thickness_values:
            avg_thickness_per_angle.append(np.mean(thickness_values))
        else:
            avg_thickness_per_angle.append(0)

    # Close the loop by appending the first value to the end
    if avg_thickness_per_angle:
        angles = np.append(angles, angles[0])
        avg_thickness_per_angle.append(avg_thickness_per_angle[0])

    return angles, avg_thickness_per_angle

def save_top_down_view_aspect_ratio(_data, _title, _path):
    # Project the 3D data onto a 2D plane by taking the max along the Z-axis
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(top_down_data)

    fig = go.Figure(data=go.Heatmap(
        z=top_down_data,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title=_title,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(l=0, r=0, t=40, b=0) # Adjust margins to remove empty space
    )

    logging.info(f"Saving top-down view plot with aspect ratio: {_path}")
    fig.write_image(_path, width=1400, height=1000)




def save_combined_view(_data, _title, _path, _angles, _radius, _avg_thickness_per_angle):
    # Project the 3D data onto a 2D plane by taking the max along the Z-axis
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(top_down_data)

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


def save_comparative_box_plot(_all_thickness_data, _sample_names, _title, _path):
    fig = go.Figure()

    for i, data in enumerate(_all_thickness_data):
        # Remove outliers for this plot only
        data_clean = remove_outliers_iqr(data)
        fig.add_trace(go.Box(y=data_clean, name=_sample_names[i]))

    fig.update_layout(title_text=_title,
                      yaxis_title="Thickness (nm)")
    logging.info(f"Saving comparative box plot: {_path}")
    fig.write_image(_path, width=1400, height=1000)

def calculate_stats(_inference_result_path: Path,
                    _stats_dir: Path):
    _alpha_step = 10
    _radius = 1000

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

    # Initialize arrays to store aggregated data
    all_thickness_data = []
    sample_names = []

    processed_samples = 0

    # Process each directory (sample) in the inference results
    for sample_dir in sample_dirs:
        logging.info("Processing sample: %s", sample_dir.name)

        # Create a directory for this sample's histograms
        sample_hist_dir = _stats_dir / sample_dir.name
        sample_hist_dir.mkdir(exist_ok=True)
        logging.debug("Created sample histogram directory: %s", sample_hist_dir)

        # Process distance_result.npy (thickness)
        distance_file = sample_dir / "distance_result.npy"
        if distance_file.exists():
            logging.debug("Processing thickness file: %s", distance_file)
            try:
                data = np.load(distance_file)
                original_size = len(data)
                logging.debug("Loaded thickness data with %d values", original_size)

                # Remove zero values
                data_no_zeros = data[data != 0]
                after_zero_removal = len(data_no_zeros)
                logging.debug("Removed %d zero values, %d values remaining",
                              original_size - after_zero_removal, after_zero_removal)

                # Remove outliers using IQR method
                data_clean = remove_outliers_iqr(data_no_zeros)
                after_outlier_removal = len(data_clean)
                logging.debug("Removed %d outliers, %d values remaining for histogram",
                              after_zero_removal - after_outlier_removal, after_outlier_removal)

                # Store data for aggregation
                if len(data_clean) > 0:
                    all_thickness_data.append(data_clean)
                    sample_names.append(sample_dir.name)

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
                save_top_down_view_aspect_ratio(data, f'Top-Down View (Aspect Ratio) - {sample_dir.name}', sample_hist_dir / f'top_down_view_aspect_ratio_{sample_dir.name}.png')
                logging.info("Created top-down view plot with aspect ratio")

                # Combined view
                save_combined_view(data, f'Combined View - {sample_dir.name}', sample_hist_dir / f'combined_view_{sample_dir.name}.png', angles, _radius, avg_thickness)
                logging.debug("Created combined view plot")

            except Exception as e:
                logging.error("Error processing thickness file %s: %s", distance_file, e)
        else:
            logging.warning("Thickness file not found: %s", distance_file)

        processed_samples += 1
        logging.info("Completed processing sample %s (%d/%d)",
                     sample_dir.name, processed_samples, total_samples)

    # Aggregate and save data for box plots
    logging.info("Aggregating thickness data for analysis")

    # Save aggregated thickness data
    if all_thickness_data:
        # Concatenate all thickness data into single array of datapoints
        aggregated_thickness = np.concatenate(all_thickness_data)
        logging.info("Aggregated %d thickness values from %d samples",
                     len(aggregated_thickness), len(all_thickness_data))

        # Save as numpy array only - just the aggregated datapoints
        thickness_np_file = _stats_dir / "aggregated_thickness_data.npy"
        np.save(thickness_np_file, aggregated_thickness)
        logging.info("Saved aggregated thickness data to %s", thickness_np_file)

        # Create and save comparative box plot for all samples
        save_comparative_box_plot(all_thickness_data,
                                  sample_names,
                                  "Comparative Box Plot of Thickness Across Samples",
                                  _stats_dir / "comparative_box_plot.png")
        logging.info("Saved comparative box plot of all samples to %s", _stats_dir / "comparative_box_plot.png")

    # Save metadata file with sample information
    metadata_file = _stats_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Statistical Analysis Metadata\n")
        f.write("============================\n")
        f.write(f"Input directory: {_inference_result_path}\n")
        f.write(f"Output directory: {_stats_dir}\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Samples with valid data: {len(sample_names)}\n")
        f.write(f"Bin sizes used: {bin_sizes}\n")
        f.write(f"Alpha step for cylindrical analysis: {_alpha_step}\n")
        f.write(f"Radius for cylindrical analysis: {_radius}\n")
        f.write("\nSample names:\n")
        for name in sample_names:
            f.write(f"  - {name}\n")
        f.write("\nGenerated files:\n")
        if all_thickness_data:
            f.write("  - aggregated_thickness_data.npy (all thickness values)\n")

    logging.info("Saved metadata to %s", metadata_file)
    logging.info("Statistical analysis completed successfully")
    logging.info("Multi-bin histograms and aggregated thickness data saved in '%s' directory", _stats_dir)

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
        # Plot histogram
        import matplotlib.pyplot as plt
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
