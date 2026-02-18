#!/usr/bin/env python3

import os
import numpy as np
import argparse
from pathlib import Path
import logging
import plotly.graph_objects as go

# --- Configurable Values ---
# This value (k) is used to determine the range for outlier removal.
# A common value is 1.5. Data points outside of Q1 - k*IQR and Q3 + k*IQR are considered outliers.
K_VALUE_IQR = 1.5

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def remove_outliers_iqr(arr, k=1.5, group_name="Unknown"):
    """Remove outliers from an array using the IQR method."""
    if len(arr) == 0:
        return arr
    original_size = len(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    clean_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    removed_count = len(arr) - len(clean_arr)
    percentage_removed = (removed_count / original_size) * 100 if original_size > 0 else 0
    logging.info(f"Group '{group_name}': Removed {removed_count} outliers ({percentage_removed:.2f}%) (k={k}).")
    return clean_arr

def find_stats_directories(root_path):
    """Find all directories ending with '_stats' in the given root path."""
    root_path = Path(root_path)
    stats_dirs = []
    logging.info(f"Searching for stats directories in: {root_path}")
    for item in root_path.iterdir():
        if item.is_dir() and item.name.endswith('_stats'):
            stats_dirs.append(item)
            logging.info(f"Found stats directory: {item}")
    if not stats_dirs:
        logging.warning("No stats directories found!")
    return stats_dirs

def load_aggregated_data(stats_dir):
    """Load aggregated thickness data from a stats directory."""
    aggregated_file = stats_dir / "aggregated_thickness_data.npy"
    if aggregated_file.exists():
        try:
            data = np.load(aggregated_file)
            data = data[~np.isnan(data)]  # Remove any NaN values
            if len(data) > 0:
                group_name = stats_dir.name.replace('_stats', '')
                logging.info(f"Loaded {len(data)} thickness values for group '{group_name}'")
                return data, group_name
            else:
                logging.warning(f"No valid data found in {aggregated_file}")
                return None, None
        except Exception as e:
            logging.error(f"Error loading {aggregated_file}: {e}")
            return None, None
    else:
        logging.warning(f"Aggregated data file not found: {aggregated_file}")
        return None, None

def create_box_plot(data_dict, output_path, k_value):
    """Create a box plot using Plotly, with outlier removal."""
    if not data_dict:
        logging.error("No data provided for box plot!")
        return

    fig = go.Figure()
    logging.info(f"Creating box plot for {len(data_dict)} groups.")

    for group_name, data in data_dict.items():
        logging.info(f"Processing group '{group_name}' for box plot.")
        # Remove outliers before plotting
        clean_data = remove_outliers_iqr(data, k=k_value, group_name=group_name)

        if len(clean_data) == 0:
            logging.warning(f"Skipping group '{group_name}' in box plot (no data after outlier removal).")
            continue

        # Downsample if data is too large
        if len(clean_data) > 500000:
            logging.info(f"Downsampling data for group '{group_name}' from {len(clean_data)} to 500000 points.")
            plot_data = np.random.choice(clean_data, 500000, replace=False)
        else:
            plot_data = clean_data

        # Calculate summary statistics
        q1 = np.percentile(clean_data, 25)
        median_val = np.median(clean_data)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        lowerfence = np.max([np.min(clean_data), q1 - 1.5 * iqr])
        upperfence = np.min([np.max(clean_data), q3 + 1.5 * iqr])
        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)

        fig.add_trace(go.Box(
            x=[group_name],
            name=group_name,
            q1=[q1],
            median=[median_val],
            q3=[q3],
            lowerfence=[lowerfence],
            upperfence=[upperfence],
            mean=[mean_val],
            sd=[std_val],
            boxpoints=False  # Do not show points on top of the box
        ))

        # Add annotations for median and std
        median_val = np.median(clean_data)
        std_val = np.std(clean_data)
        fig.add_annotation(
            x=group_name,
            y=0,  # Place at the bottom of the y-axis
            text=f"Median: {median_val:.2f}<br>Mean: {mean_val:.2f}<br>Std: {std_val:.2f}",
            showarrow=False,  # No arrow needed if placed at the bottom
            yshift=-30  # Shift down to avoid overlapping with x-axis labels
        )

    fig.update_layout(
        title_text='Thickness Distribution Comparison Across Groups',
        yaxis_title="Thickness (nm)",
        # xaxis_title="Groups",
        showlegend=False,
        boxmode='group'
    )

    logging.info(f"Saving box plot to: {output_path}")

    fig.write_image(output_path, width=1400, height=1000, scale=2)

def create_raincloud_plot(data_dict, output_path, k_value):
    """Create a raincloud plot using Plotly."""
    if not data_dict:
        logging.error("No data provided for raincloud plot!")
        return

    fig = go.Figure()
    logging.info(f"Creating raincloud plot for {len(data_dict)} groups.")

    for group_name, data in data_dict.items():
        logging.info(f"Processing group '{group_name}' for raincloud plot.")
        # For raincloud plots, we use the cleaned data for the distribution
        clean_data = remove_outliers_iqr(data, k=k_value, group_name=group_name)

        if len(clean_data) == 0:
            logging.warning(f"Skipping group '{group_name}' in raincloud plot (no data after outlier removal).")
            continue

        # Downsample if data is too large
        if len(clean_data) > 50000:
            logging.info(f"Downsampling data for group '{group_name}' from {len(clean_data)} to 50000 points.")
            plot_data = np.random.choice(clean_data, 50000, replace=False)
        else:
            plot_data = clean_data

        fig.add_trace(go.Violin(
            y=plot_data,
            x0=group_name,
            name=group_name,
            box_visible=True,
            meanline_visible=True,
            points='outliers',  # This creates the "rain"
            jitter=0.3,
            pointpos=-1.8,  # Position points to the left
            side='positive'  # Show only one side of the violin
        ))

        # Add annotations for median and std
        median_val = np.median(clean_data)
        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)
        fig.add_annotation(
            x=group_name,
            y=0,  # Place at the bottom of the y-axis
            text=f"Median: {median_val:.2f}<br>Mean: {mean_val:.2f}<br>Std: {std_val:.2f}",
            showarrow=False,  # No arrow needed if placed at the bottom
            yshift=-30  # Shift down to avoid overlapping with x-axis labels
        )

    fig.update_layout(
        title_text='Raincloud Plot of Thickness Distribution Across Groups',
        yaxis_title="Thickness (nm)",
        xaxis_title="Groups",
        showlegend=False,
        violingap=0,
        violinmode='overlay'
    )

    logging.info(f"Saving raincloud plot to: {output_path}")
    fig.write_image(output_path, width=1400, height=1000, scale=2)

def main():
    """Main function to run plot generation."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description='Generate comparative plots from aggregated thickness data in stats directories.'
    )
    parser.add_argument(
        'directory',
        help='Root directory to search for stats directories.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.directory):
        logging.error(f"Directory does not exist: {args.directory}")
        return

    stats_dirs = find_stats_directories(args.directory)

    if not stats_dirs:
        return

    data_dict = {}
    for stats_dir in stats_dirs:
        data, group_name = load_aggregated_data(stats_dir)
        if data is not None and group_name is not None:
            data_dict[group_name] = data

    if not data_dict:
        logging.error("No valid data loaded from any stats directory. Exiting.")
        return

    logging.info(f"Successfully loaded data for {len(data_dict)} groups.")

    # --- Generate Box Plot ---
    box_plot_output_path = Path(args.directory) / "comparison_box_plot.png"
    create_box_plot(data_dict, box_plot_output_path, K_VALUE_IQR)

    # --- Generate Raincloud Plot ---
    raincloud_output_path = Path(args.directory) / "comparison_raincloud_plot.png"
    create_raincloud_plot(data_dict, raincloud_output_path, K_VALUE_IQR)

    logging.info("Plot generation completed successfully.")


if __name__ == "__main__":
    main()
