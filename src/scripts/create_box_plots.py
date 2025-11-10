#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_stats_directories(root_path):
    """
    Find all directories ending with '_stats' in the given root path.
    
    Args:
        root_path (str): Root directory to search
        
    Returns:
        list: List of Path objects for stats directories
    """
    root_path = Path(root_path)
    stats_dirs = []
    
    logging.info(f"Searching for stats directories in: {root_path}")
    
    # Search for directories ending with '_stats'
    for item in root_path.iterdir():
        if item.is_dir() and item.name.endswith('_stats'):
            stats_dirs.append(item)
            logging.info(f"Found stats directory: {item}")
    
    if not stats_dirs:
        logging.warning("No stats directories found!")
    
    return stats_dirs

def load_aggregated_data(stats_dir):
    """
    Load aggregated thickness data from a stats directory.
    
    Args:
        stats_dir (Path): Path to stats directory
        
    Returns:
        tuple: (data, group_name) or (None, None) if no data found
    """
    aggregated_file = stats_dir / "aggregated_thickness_data.npy"
    
    if aggregated_file.exists():
        try:
            data = np.load(aggregated_file)
            # Remove any NaN values
            data = data[~np.isnan(data)]
            
            if len(data) > 0:
                # Extract group name (remove '_stats' suffix)
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

def create_box_plot(data_dict, output_path):
    """
    Create a box plot comparing thickness data across different groups.
    
    Args:
        data_dict (dict): Dictionary with group names as keys and data arrays as values
        output_path (str): Path to save plot.
    """
    if not data_dict:
        logging.error("No data to plot!")
        return
    
    # Prepare data for box plot
    groups = list(data_dict.keys())
    data_values = list(data_dict.values())
    
    logging.info(f"Creating box plot for {len(groups)} groups")
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    box_plot = plt.boxplot(data_values, labels=groups, patch_artist=True)
    
    # Customize box plot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize plot
    plt.title('Thickness Distribution Comparison Across Groups', fontsize=16, fontweight='bold')
    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Thickness (nm)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = []
    for i, (group, data) in enumerate(data_dict.items()):
        if len(data) > 0:
            median = np.median(data)
            mean = np.mean(data)
            std = np.std(data)
            n = len(data)
            stats_text.append(f'{group}: n={n}, μ={mean:.1f}, σ={std:.1f}')
    
    # Add statistics as text below the plot
    plt.figtext(0.5, 0.02, '\n'.join(stats_text), 
                ha='center', va='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for statistics text
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Box plot saved to: {output_path}")
    
    plt.close()

def main():
    """Main function to run box plot generation"""
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate box plots from aggregated thickness data in stats directories'
    )
    parser.add_argument(
        'directory',
        help='Root directory to search for stats directories'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not os.path.exists(args.directory):
        logging.error(f"Directory does not exist: {args.directory}")
        return
    
    # Find all stats directories
    stats_dirs = find_stats_directories(args.directory)
    
    if not stats_dirs:
        logging.error("No stats directories found. Exiting.")
        return
    
    # Load data from all stats directories
    data_dict = {}
    
    for stats_dir in stats_dirs:
        data, group_name = load_aggregated_data(stats_dir)
        if data is not None and group_name is not None:
            data_dict[group_name] = data
    
    if not data_dict:
        logging.error("No valid data loaded from any stats directory. Exiting.")
        return
    
    logging.info(f"Successfully loaded data for {len(data_dict)} groups")
    
    # Generate default output path in the same directory
    output_path = os.path.join(args.directory, "thickness_comparison_box_plot.png")
    
    # Create box plot
    create_box_plot(data_dict, output_path)
    
    logging.info("Box plot generation completed successfully")

if __name__ == "__main__":
    main()