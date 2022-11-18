"""
Author: Arash Fatehi
Date:   11.11.2022
"""

# Python Imports
import ast
from argparse import ArgumentParser

def parse_arguments(_description):
    parser = ArgumentParser(description=_description)

    parser.add_argument("-tds",
                        help="The path to the training dataset directory")

    parser.add_argument("-vds",
                        help="The path to the validation dataset directory")

    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="Increase output verbosity")

    parser.add_argument("-e", "--epochs", default=50, type=int,
                        help="Number of epochs")

    parser.add_argument("-ch", "--channels", type=ast.literal_eval, default=(0, 1, 2),
                        help="Selects channels to be included")

    parser.add_argument("-fm", "--feature-maps", type=ast.literal_eval, default=(64, 128, 256),
                        help="Determines 3DUnet's feature map")

    parser.add_argument("-bs", "--batch-size", type=int, default=4,
                        help="Sets the batch size")

    parser.add_argument("-sd", "--sample-dimension", type=ast.literal_eval, default=(12, 256, 2),
                        help="Determines the dimenion of input 3D tiles")

    parser.add_argument("-ps", "--pixel-stride", type=ast.literal_eval, default=(1, 16, 16),
                        help="Determines the stride for cropping 3D tiles from input smaples")

    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01,
                        help="Learning rate of optimization algorithm")

    parser.add_argument("-Vv", "--validation-visualization", action="store_true",
                        help="Stores results as gif, tif, and npy files during the validation")

    parser.add_argument("-Vp", "--validation-visualization-path", default="/tmp/gbm-vv/",
                        help="Path for the validation visualizations")

    parser.add_argument("-Vn", "--validation-batch-no",
                        help="No of validation batches")

    parser.add_argument("-Vn", "--validation-batch-no",
                        help="No of validation batches")

    parallelism_group = parser.add_mutually_exclusive_group()

    parallelism_group.add_argument("-dp",
                                   "--data-parallelism",
                                   action="store_true",
                                   help="Use data parallelism")

    parallelism_group.add_argument("-ddp",
                                   "--distributed-data-parallelism",
                                   action="store_true",
                                   help="Use distributed data parallelism")

    return parser.parse_args()


def summerize_args(_args):
    assert _args.tds is not None, "Please provide the training dataset path."
    assert _args.vds is not None, "Please provide the validation dataset path."
    print(f"Training dataset path: {_args.tds}")
    print(f"Validation dataset path: {_args.vds}")
    print(f"Number of epochs: {_args.epochs}")
    print(f"Selected channels: {_args.channels}")
    print(f"Feature maps: {_args.feature_maps}")
    print(f"Batch size: {_args.batch_size}")
    print(f"Sample dimension: {_args.sample_dimension}")
    print(f"Pixel stride: {_args.pixel_stride}")
    print(f"Learning rate: {_args.learning_rate}")
    print(f"Pixel stride: {_args.pixel_stride}")
    print(f"Validation Visualization: {_args.validation_visualization}")
    print(f"Visualization path: {_args.validation_visualization_path}")
    print(f"Validation batch no: {_args.validation_batch_no}")
    print(f"Data parallelism: {_args.data_parallelism}")
    print(f"Distributed: {_args.distributed_data_parallelism}")
