#!/usr/bin/env python3

"""
Author: Arash Fatehi
Date:   30.04.2022
"""

# Python Imprts
import os

# Library Imports
from pudb import set_trace

# Local Imports
from src.utils.misc import configure_logger, basic_logger
from src.utils import args
from src.utils import exper
from src.tui.main import MainWindow

if __name__ == '__main__':
    args, configs = args.parse_exper()

    if args.debug:
        set_trace()

    if args.action == 'tui':
        main_window = MainWindow(configs)
        main_window.run()

    if args.action == 'create':
        basic_logger()
        name = args.name
        exper.create_new_experiment(
                _name=name,
                _root_path=configs['experiments']['root'],
                _source_path=os.getcwd(),
                _dataset_path=configs['experiments']['default_data_path'])

    if args.action == 'list':
        root = args.root
        if not root:
            root = configs['experiments']['root']
        exper.list_experiments(root)

    if args.action == 'train':
        configure_logger(configs)
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
        configure_logger(configs)
        root = configs['experiments']['root']
        name = args.name
        snapshot = args.snapshot
        batch_size = args.batch_size
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
                               _scale=scale)
