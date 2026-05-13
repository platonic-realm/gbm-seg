# Statistics + visualization for inference results.
# Moved out of src/utils/misc.py during the Phase 3 split.

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion


def replace_outliers_iqr(arr, k=1.5, lower_p=5, upper_p=95,
                         lower_p_zero_iqr=2, upper_p_zero_iqr=98):
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
    logging.info(f"Outlier replacement: q1={q1}, q3={q3}, iqr={iqr}, "
                 f"lower_bound={lower_bound}, upper_bound={upper_bound}")
    arr_copy = arr.copy()
    outliers_upper = arr_copy > upper_bound
    replaced_count = np.sum(outliers_upper)
    percentage_replaced = (replaced_count / original_size) * 100 if original_size > 0 else 0
    arr_copy[outliers_upper] = upper_bound
    logging.info(f"Replaced {replaced_count} outliers out of {original_size} "
                 f"values ({percentage_replaced:.2f}%).")
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
    logging.info(f"Outlier removal: q1={q1}, q3={q3}, iqr={iqr}, "
                 f"lower_bound={lower_bound}, upper_bound={upper_bound}")
    clean_arr = arr[(arr <= upper_bound)]
    removed_count = original_size - len(clean_arr)
    percentage_removed = (removed_count / original_size) * 100
    logging.info(f"Removed {removed_count} outliers out of {original_size} "
                 f"values ({percentage_removed:.2f}%).")
    return clean_arr


def save_histogram(_array, _title, _path, _bins):
    if _array.size < 2 or np.max(_array) == np.min(_array):
        bin_config = dict(nbinsx=_bins)
    else:
        bin_size = (np.max(_array) - np.min(_array)) / _bins
        bin_config = dict(xbins=dict(
            start=np.min(_array), end=np.max(_array), size=bin_size))

    fig = go.Figure(data=[go.Histogram(
        x=_array, **bin_config,
        marker=dict(line=dict(color='black', width=1))
    )])
    fig.update_layout(title=_title, xaxis_title='Value', yaxis_title='Frequency')
    logging.info(f"Saving histogram: {_path}")
    fig.write_image(_path, width=1400, height=1000)  # requires kaleido


def save_polar_plot(_angles, _thickness_values, _title, _path):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=_thickness_values, theta=_angles, mode='lines',
        line_color='blue', line_width=2, fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)'))
    fig.update_layout(
        title=_title,
        polar=dict(
            radialaxis=dict(visible=True, title="Average Thickness (nm)",
                            tickangle=0, dtick=400),
            angularaxis=dict(visible=True, direction="clockwise",
                             period=360, dtick=45, rotation=0)))
    logging.info(f"Saving polar plot: {_path}")
    fig.write_image(_path, width=1400, height=1000)


def calculate_cylindrical_analysis(_data, _alpha_step, _radius):
    z_dim, y_dim, x_dim = _data.shape
    center_y, center_x = y_dim // 2, x_dim // 2

    angle_bins = np.arange(0, 360 + _alpha_step, _alpha_step)
    angles_for_plot = (angle_bins[:-1] + angle_bins[1:]) / 2
    binned_thickness_values = [[] for _ in range(len(angle_bins) - 1)]

    x_coords, y_coords = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    dx = x_coords - center_x
    dy = y_coords - center_y
    radius_map = np.sqrt(dx**2 + dy**2)
    angle_map_rad = np.arctan2(dy, dx)
    angle_map_deg = np.rad2deg(angle_map_rad)
    angle_map_deg[angle_map_deg < 0] += 360

    radius_mask = radius_map <= _radius

    angle_bin_indices = np.digitize(angle_map_deg, bins=angle_bins) - 1
    angle_bin_indices[angle_bin_indices == len(angle_bins) - 1] = len(angle_bins) - 2

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
        for i in range(len(binned_thickness_values)):
            bin_mask = all_valid_angle_bins == i
            if np.any(bin_mask):
                points_to_add = all_valid_thicknesses[bin_mask]
                binned_thickness_values[i].extend(points_to_add)
                total_points_binned_incremental += len(points_to_add)

        if total_points_found == total_points_binned_incremental:
            logging.debug(
                f"Point count validation successful: "
                f"Total points found ({total_points_found}) = "
                f"Total points incrementally binned ({total_points_binned_incremental}).")
        else:
            logging.warning(
                f"Point count mismatch: "
                f"Total points found ({total_points_found}) != "
                f"Total points incrementally binned ({total_points_binned_incremental}). "
                f"Some points were lost during binning.")

    avg_thickness_per_angle = []
    for i, values in enumerate(binned_thickness_values):
        if values:
            num_points = len(values)
            avg_thickness = np.mean(values)
            std_thickness = np.std(values)
            logging.debug(
                f"Cylindrical slice @ {angle_bins[i]}-{angle_bins[i+1]} deg: "
                f"Points={num_points}, Avg={avg_thickness:.2f}, Std={std_thickness:.2f}")
            avg_thickness_per_angle.append(avg_thickness)
        else:
            logging.debug(f"Cylindrical slice @ {angle_bins[i]}-{angle_bins[i+1]} deg: Points=0")
            avg_thickness_per_angle.append(0)

    if avg_thickness_per_angle:
        angles_for_plot = np.append(angles_for_plot, angles_for_plot[0])
        avg_thickness_per_angle.append(avg_thickness_per_angle[0])

    return angles_for_plot, avg_thickness_per_angle


def save_top_down_view_aspect_ratio(_data, _title, _path,
                                    _lower_percentile_iqr, _upper_percentile_iqr,
                                    _lower_percentile_iqr_zero, _upper_percentile_iqr_zero):
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(
        top_down_data,
        lower_p=_lower_percentile_iqr, upper_p=_upper_percentile_iqr,
        lower_p_zero_iqr=_lower_percentile_iqr_zero,
        upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)

    fig = go.Figure(data=go.Heatmap(z=top_down_data, colorscale='Viridis'))
    fig.update_layout(
        title=_title,
        xaxis_title='X-axis', yaxis_title='Y-axis',
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=40, b=0))

    logging.info(f"Saving top-down view plot with aspect ratio: {_path}")
    fig.write_image(_path, width=1000, height=1000)


def save_combined_view(_data, _title, _path, _angles, _radius, _avg_thickness_per_angle,
                       _lower_percentile_iqr, _upper_percentile_iqr,
                       _lower_percentile_iqr_zero, _upper_percentile_iqr_zero):
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(
        top_down_data,
        lower_p=_lower_percentile_iqr, upper_p=_upper_percentile_iqr,
        lower_p_zero_iqr=_lower_percentile_iqr_zero,
        upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)

    fig = go.Figure(data=go.Heatmap(z=top_down_data, colorscale='Viridis'))

    center_y, center_x = top_down_data.shape[0] // 2, top_down_data.shape[1] // 2

    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=center_x - _radius, y0=center_y - _radius,
                  x1=center_x + _radius, y1=center_y + _radius,
                  line_color="rgba(255,0,0,0.5)", line_width=2)

    for angle in _angles:
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=center_x, y0=center_y,
            x1=center_x + _radius * np.cos(np.deg2rad(-angle)),
            y1=center_y + _radius * np.sin(np.deg2rad(-angle)),
            line_color="rgba(128,128,128,0.3)", line_width=1)

    max_thickness = np.max(_avg_thickness_per_angle)
    normalized_thickness = (_avg_thickness_per_angle / max_thickness) * _radius
    x_coords = center_x + normalized_thickness * np.cos(np.deg2rad(-_angles))
    y_coords = center_y + normalized_thickness * np.sin(np.deg2rad(-_angles))
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines',
                             line=dict(color='white', width=2)))

    for i, angle in enumerate(_angles[:-1]):
        avg_thickness = _avg_thickness_per_angle[i]
        text_radius = _radius * 0.9
        text_x = center_x + text_radius * np.cos(np.deg2rad(-angle))
        text_y = center_y + text_radius * np.sin(np.deg2rad(-angle))
        fig.add_annotation(
            x=text_x, y=text_y,
            text=f"<b>{avg_thickness:.0f} nm</b>",
            showarrow=False,
            font=dict(size=12, color="white"),
            textangle=angle, xanchor="center", yanchor="middle")

    fig.update_layout(
        title=_title, xaxis_title='X-axis', yaxis_title='Y-axis',
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))

    logging.info(f"Saving combined view plot: {_path}")
    fig.write_image(_path, width=1000, height=1000)


def generate_comparative_box_plot(_stats_dir: Path):
    """Generate a comparative box plot from pre-calculated summary statistics."""
    summary_file = _stats_dir / "summary_statistics.npz"
    if not summary_file.exists():
        logging.error("Summary statistics file not found: %s", summary_file)
        return

    summary_data = np.load(summary_file, allow_pickle=True)['arr']

    fig = go.Figure()
    for sample_stats in summary_data:
        fig.add_trace(go.Box(
            x=[sample_stats['sample_name']],
            name=sample_stats['sample_name'],
            q1=[sample_stats['q1']],
            median=[sample_stats['median']],
            q3=[sample_stats['q3']],
            lowerfence=[sample_stats['lowerfence']],
            upperfence=[sample_stats['upperfence']],
            mean=[sample_stats['mean']],
            sd=[sample_stats['std']]))

    fig.update_layout(title_text="Comparative Box Plot of Thickness Across Samples",
                      yaxis_title="Thickness (nm)", showlegend=False)
    plot_path = _stats_dir / "comparative_box_plot.png"
    logging.info(f"Saving comparative box plot from summary: {plot_path}")
    fig.write_image(plot_path, width=1400, height=1000)


def detect_group_type(_sample_name):
    """Identify the group type based on string prefixes."""
    mapping = {
        "Control": ("NCW.AUY381", "NCW.AUY380", "CKM103", "CKM110"),
        "Podocin": ("NCW.BDP669", "NCW.BDP672", "NCW.BDP675"),
        "Collagen": ("NCW.CKM105", "NCW.CKM104"),
    }
    input_clean = _sample_name.strip().upper()
    for group_type, prefixes in mapping.items():
        if input_clean.startswith(prefixes):
            return group_type
    return "Control"


def calculate_stats(_inference_result_path: Path,
                    _stats_dir: Path,
                    _clipping: bool,
                    _thickness_clip_max: int = 1400):
    _alpha_step = 10
    _radius = 1000
    _lower_percentile_iqr = 5
    _upper_percentile_iqr = 95
    _lower_percentile_iqr_zero = 2
    _upper_percentile_iqr_zero = 98

    logging.info("Starting statistical analysis for inference results")
    logging.info("Input path: %s", _inference_result_path)
    logging.info("Output path: %s", _stats_dir)

    _stats_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Created stats directory: %s", _stats_dir)

    bin_sizes = [10, 20, 50, 100]
    logging.info("Using bin sizes: %s", bin_sizes)

    sample_dirs = [d for d in _inference_result_path.iterdir() if d.is_dir()]
    total_samples = len(sample_dirs)
    logging.info("Found %d samples to process", total_samples)

    summary_data_list = []
    sample_names_for_violin = []
    all_thickness_data_for_violin = []
    all_thickness_data_before_outliers = []

    processed_samples = 0

    for sample_dir in sample_dirs:
        logging.info("Processing sample: %s", sample_dir.name)

        sample_hist_dir = _stats_dir / sample_dir.name
        sample_hist_dir.mkdir(exist_ok=True)
        logging.debug("Created sample histogram directory: %s", sample_hist_dir)

        distance_file = sample_dir / "psf_result.npz"

        if distance_file.exists():
            logging.debug("Processing thickness file: %s", distance_file)
            try:
                data = np.load(distance_file)['arr']

                if _clipping:
                    max_clipping_value = _thickness_clip_max
                    logging.info("Clipping is enabled, removing values above %d",
                                 max_clipping_value)
                    original_non_zero_voxels = np.count_nonzero(data)
                    data[data > max_clipping_value] = 0

                    final_non_zero_voxels = np.count_nonzero(data)
                    altered_voxels = original_non_zero_voxels - final_non_zero_voxels
                    if original_non_zero_voxels > 0:
                        percentage_altered = (altered_voxels / original_non_zero_voxels) * 100
                        logging.info(f"Altered {altered_voxels:,} voxels, which is "
                                     f"{percentage_altered:.2f}% of the original non-zero data.")
                    else:
                        logging.info("No non-zero voxels to alter.")

                original_size = np.count_nonzero(data)
                logging.debug(f"Loaded thickness data with {original_size:,} values")

                sample = tifffile.imread(
                    _inference_result_path / sample_dir.name / "prediction.tif")
                col4_stack = sample[:, 1, :, :]

                logging.debug("Creating the mask for %s", sample_dir.name)
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
                logging.debug("Before mask mean: %.4f, std: %.4f",
                              np.mean(data[data != 0]), np.std(data[data != 0]))
                data[mask > 0] = 0
                logging.debug("After mask mean: %.4f, std: %.4f",
                              np.mean(data[data != 0]), np.std(data[data != 0]))

                data_no_zeros = data[data != 0]
                after_zero_removal = np.count_nonzero(data_no_zeros)
                logging.debug("Removed %d zero values, %d values remaining",
                              original_size - after_zero_removal, after_zero_removal)

                if len(data_no_zeros) > 0:
                    all_thickness_data_before_outliers.append(data_no_zeros)

                data_clean = remove_outliers_iqr(data_no_zeros)
                after_outlier_removal = len(data_clean)
                logging.debug("Removed %d outliers, %d values remaining for histogram",
                              after_zero_removal - after_outlier_removal, after_outlier_removal)

                if len(data_clean) > 0:
                    all_thickness_data_for_violin.append(data_clean)
                    sample_names_for_violin.append(sample_dir.name)

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

                for bins in bin_sizes:
                    save_histogram(
                        data_clean,
                        f'Thickness Histogram with {bins} Bins - {sample_dir.name}',
                        sample_hist_dir / f'thickness_{sample_dir.name}_{bins}_bins.png',
                        bins)
                    logging.debug("Created thickness histogram with %d bins", bins)

                angles, avg_thickness = calculate_cylindrical_analysis(
                    data, _alpha_step, _radius)
                save_polar_plot(
                    angles, avg_thickness,
                    f'Cylindrical Analysis - {sample_dir.name}',
                    sample_hist_dir / f'cylindrical_analysis_{sample_dir.name}.png')
                logging.debug("Created cylindrical analysis plot")

                save_top_down_view_aspect_ratio(
                    data,
                    f'Top-Down View (Aspect Ratio) - {sample_dir.name}',
                    sample_hist_dir / f'top_down_view_aspect_ratio_{sample_dir.name}.png',
                    _lower_percentile_iqr, _upper_percentile_iqr,
                    _lower_percentile_iqr_zero, _upper_percentile_iqr_zero)
                logging.info("Created top-down view plot with aspect ratio")

                save_combined_view(
                    data, f'Combined View - {sample_dir.name}',
                    sample_hist_dir / f'combined_view_{sample_dir.name}.png',
                    angles, _radius, avg_thickness,
                    _lower_percentile_iqr, _upper_percentile_iqr,
                    _lower_percentile_iqr_zero, _upper_percentile_iqr_zero)
                logging.debug("Created combined view plot")

            except Exception as e:
                logging.error("Error processing thickness file %s: %s", distance_file, e)
        else:
            logging.warning("Thickness file not found: %s", distance_file)

        processed_samples += 1
        logging.info("Completed processing sample %s (%d/%d)",
                     sample_dir.name, processed_samples, total_samples)

    if summary_data_list:
        dtype = [('sample_name', 'U100'), ('q1', 'f8'), ('median', 'f8'), ('q3', 'f8'),
                 ('lowerfence', 'f8'), ('upperfence', 'f8'),
                 ('mean', 'f8'), ('std', 'f8')]
        records = [tuple(d.values()) for d in summary_data_list]
        summary_array = np.array(records, dtype=dtype)
        summary_file = _stats_dir / "summary_statistics.npz"
        np.savez_compressed(summary_file, arr=summary_array)
        logging.info("Saved summary statistics to %s", summary_file)

    if all_thickness_data_before_outliers:
        aggregated_thickness_raw = np.concatenate(all_thickness_data_before_outliers)
        logging.info(
            "Aggregated %d raw thickness values from %d samples (before outlier removal)",
            len(aggregated_thickness_raw), len(all_thickness_data_before_outliers))
        raw_thickness_file = _stats_dir / "aggregated_thickness_data.npz"
        np.savez_compressed(raw_thickness_file, arr=aggregated_thickness_raw)
        logging.info("Saved aggregated raw thickness data to %s", raw_thickness_file)

    generate_comparative_box_plot(_stats_dir)

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
            f.write("  - summary_statistics.npz (quartiles, fences, mean, std for each sample)\n")
            f.write("  - comparative_box_plot.png (box plot generated from summary)\n")
        if all_thickness_data_before_outliers:
            f.write("  - aggregated_thickness_data.npz (all thickness values before outlier removal)\n")
        if all_thickness_data_for_violin:
            f.write("  - thickness_violin_plot.png (violin plot of all samples)\n")

    logging.info("Saved metadata to %s", metadata_file)
    logging.info("Statistical analysis completed successfully")
    logging.info("Multi-bin histograms and aggregated thickness data saved in '%s' directory",
                 _stats_dir)
