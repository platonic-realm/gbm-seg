#!/usr/bin/env python3
"""
Description: This script will convert GBM training data to a isometric voxel spaces
by interplating the input channels and stacking the labels.
Author: Arash Fatehi
"""

import sys
import logging
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.nn import functional as Fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Script description here.")
    parser.add_argument("-i",
                        "--input",
                        action='store',
                        required=True,
                        help="Path to the input directory")
    parser.add_argument("-o",
                        "--output",
                        action='store',
                        required=True,
                        help="Path to the output directory")
    parser.add_argument("-sc",
                        "--scale-factor",
                        action='store',
                        default=6,
                        help="The multiplier for interplating along the z-axis")
    return parser.parse_args()


def interpolate_channel(_channel: np.ndarray,
                        _scale_facor) -> np.ndarray:
    channel = torch.unsqueeze(torch.from_numpy(_channel), dim=0)
    channel = torch.unsqueeze(channel, dim=0)
    zoomed = Fn.interpolate(channel,
                            scale_factor=(_scale_facor, 1, 1),
                            mode='trilinear')

    return zoomed


def main():
    """Main script logic."""
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)
    scale_factor = args.scale_factor

    input_files = list(input_path.glob("*.tiff")) + list(input_path.glob("*.tif"))
    for image_path in input_files:
        with tifffile.TiffFile(image_path) as tiff:
            image = tiff.asarray()
            ij_meta = tiff.imagej_metadata
            page = tiff.pages[0]

        z_voxel_size = ij_meta.get('spacing', 1.0) / scale_factor

        x_res_tag = page.tags.get('XResolution')
        if x_res_tag is not None:
            x_num, x_den = x_res_tag.value
            x_voxel_size = x_den / x_num  # in the tag’s native unit (often inch)
        else:
            x_voxel_size = 1.0

        y_res_tag = page.tags.get('YResolution')
        if y_res_tag is not None:
            y_num, y_den = y_res_tag.value
            y_voxel_size = y_den / y_num
        else:
            y_voxel_size = 1.0

        ij_meta['spacing'] = z_voxel_size
        ij_meta['pixelWidth'] = x_voxel_size
        ij_meta['pixelHeight'] = y_voxel_size

        image = np.array(image)
        image = image.astype(np.float32)

        scaled_image = np.empty(((image.shape[0])*scale_factor,
                                 image.shape[1],
                                 image.shape[2],
                                 image.shape[3]),
                                dtype=np.float32)

        scaled_image[:, 0, :, :] = interpolate_channel(image[:, 0, :, :],
                                                       scale_factor)
        scaled_image[:, 1, :, :] = interpolate_channel(image[:, 1, :, :],
                                                       scale_factor)
        scaled_image[:, 2, :, :] = interpolate_channel(image[:, 2, :, :],
                                                       scale_factor)
        scaled_image[:, 3, :, :] = np.repeat(image[:, 3, :, :],
                                             scale_factor,
                                             axis=0)

        if x_res_tag is not None and y_res_tag is not None:
            resolution = (x_res_tag.value, y_res_tag.value)
        else:
            # Otherwise, assume 1 pixel per unit
            resolution = ((1, 1), (1, 1))

        filename = image_path.name
        output = output_path / filename
        tifffile.imwrite(output,
                         scaled_image,
                         shape=scaled_image.shape,
                         imagej=True,
                         metadata=ij_meta,
                         compression="lzw",
                         resolution=resolution)
        logging.info("Saving file to: " + str(output))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

