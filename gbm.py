#!/usr/bin/env python3

# Python Imprts
import os

# Library Imports
from pudb import set_trace

# Local Imports
from src.utils.misc import configure_logger, basic_logger
from src.utils import args
from src.utils import exper


if __name__ == '__main__':
    args, configs = args.parse_exper()

    if args.debug:
        set_trace()

    if args.action == 'create':
        basic_logger()
        name = args.name
        exper.create_new_experiment(
                _name=name,
                _root_path=configs['experiments']['root'],
                _source_path=os.getcwd(),
                _dataset_path=configs['experiments']['default_data_path'],
                _batch_size=int(args.batch_size),
                _voxel_size=configs['experiments']['default_voxel_size'])

    if args.action == 'list':
        root = args.root
        if not root:
            root = configs['experiments']['root']
        if args.snapshots is None:
            exper.list_experiments(root)
        else:
            exper.list_snapshots(_name=args.snapshots,
                                 _root_path=root)

    if args.action == 'train':
        configure_logger(configs, _log_to_file=True)
        root = configs['experiments']['root']
        name = args.name
        exper.train_experiment(_name=name,
                               _root_path=root)

    if args.action == 'delete':
        basic_logger()
        root = configs['experiments']['root']
        name = args.name
        exper.delete_experiment(_name=name,
                                _root_path=root)

    if args.action == 'infer':
        configure_logger(configs, _log_to_file=False)
        name = args.name
        root = configs['experiments']['root']
        snapshot = args.snapshot
        batch_size = args.batch_size
        interpolate = args.interpolation
        sample_dimension = [item.strip() for
                            item in args.sample_dimension.split(",")]
        stride = [item.strip() for
                  item in args.stride.split(",")]
        scale = args.scale_factor

        exper.infer_experiment(_name=name,
                               _root_path=root,
                               _snapshot=snapshot,
                               _batch_size=batch_size,
                               _sample_dimension=sample_dimension,
                               _stride=stride,
                               _scale=scale,
                               _interpolate=interpolate)

    if args.action == 'psp':
        configure_logger(configs, _log_to_file=False)
        name = args.name
        inference_tag = args.inference_tag
        max_concurrent = int(args.max_concurrent)
        root = configs['experiments']['root']
        exper.post_processing(name,
                              root,
                              inference_tag,
                              max_concurrent)

    if args.action == 'morph':
        configure_logger(configs, _log_to_file=False)
        name = args.name
        inference_tag = args.inference_tag
        sample_name = args.sample_name
        root = configs['experiments']['root']
        exper.analyze_morphometrics(name,
                                    root,
                                    inference_tag,
                                    sample_name)

    if args.action == 'blender':
        configure_logger(configs, _log_to_file=False)
        name = args.name
        inference_tag = args.inference_tag
        sample_name = args.sample_name
        root = configs['experiments']['root']
        exper.visualize_results(name,
                                root,
                                inference_tag,
                                sample_name)

    if args.action == 'render':
        configure_logger(configs, _log_to_file=False)
        name = args.name
        inference_tag = args.inference_tag
        root = configs['experiments']['root']
        exper.render_results(name,
                             root,
                             inference_tag)

    if args.action == 'export':
        configure_logger(configs, _log_to_file=False)
        name = args.name
        inference_tag = args.inference_tag
        root = configs['experiments']['root']
        exper.export(name,
                     root,
                     inference_tag)
