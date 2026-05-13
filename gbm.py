#!/usr/bin/env python3

import os

# Must be set before any CUDA op (i.e., before `import torch` anywhere in the
# import chain). cuBLAS reads this once at first CUDA init; setting it later
# is a no-op. Required for bit-exact reproducibility under
# torch.use_deterministic_algorithms(True). See PyTorch docs:
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pudb import set_trace

from src.utils import args as args_module
from src.utils import exper
from src.utils.misc import basic_logger, configure_logger


def _do_create(args, configs):
    basic_logger()
    exper.create_new_experiment(
        _name=args.name,
        _root_path=configs['experiments']['root'],
        _source_path=os.getcwd(),
        _dataset_path=configs['experiments']['default_data_path'],
        _batch_size=int(args.batch_size),
        _voxel_size=configs['experiments']['default_voxel_size'])


def _do_list(args, configs):
    root = args.root or configs['experiments']['root']
    if args.snapshots is None:
        exper.list_experiments(root)
    else:
        exper.list_snapshots(_name=args.snapshots, _root_path=root)


def _do_train(args, configs):
    configure_logger(configs, _log_to_file=True)
    exper.train_experiment(_name=args.name,
                           _root_path=configs['experiments']['root'])


def _do_delete(args, configs):
    basic_logger()
    exper.delete_experiment(_name=args.name,
                            _root_path=configs['experiments']['root'],
                            _force=args.force)


def _do_infer(args, configs):
    configure_logger(configs, _log_to_file=False)
    sample_dimension = [item.strip() for item in args.sample_dimension.split(",")]
    stride = [item.strip() for item in args.stride.split(",")]
    exper.infer_experiment(_name=args.name,
                           _root_path=configs['experiments']['root'],
                           _snapshot=args.snapshot,
                           _batch_size=args.batch_size,
                           _sample_dimension=sample_dimension,
                           _stride=stride,
                           _scale=args.scale_factor,
                           _interpolate=args.interpolation,
                           _force=args.force)


def _do_psp(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.post_processing(args.name,
                          configs['experiments']['root'],
                          args.inference_tag,
                          int(args.max_concurrent))


def _do_morph(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.analyze_morphometrics(args.name,
                                configs['experiments']['root'],
                                args.inference_tag,
                                args.sample_name)


def _do_blender(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.visualize_results(args.name,
                            configs['experiments']['root'],
                            args.inference_tag,
                            args.sample_name)


def _do_render(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.render_results(args.name,
                         configs['experiments']['root'],
                         args.inference_tag)


def _do_export(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.export(args.name,
                 configs['experiments']['root'],
                 args.inference_tag)


def _do_stats(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.stats(args.name,
                configs['experiments']['root'],
                args.inference_tag,
                args.clipping)


# Aligned by hand for readability; suppress pycodestyle's "multiple spaces after ':'".
HANDLERS = {
    'create':  _do_create,
    'list':    _do_list,
    'train':   _do_train,
    'delete':  _do_delete,
    'infer':   _do_infer,
    'psp':     _do_psp,
    'morph':   _do_morph,
    'blender': _do_blender,
    'render':  _do_render,
    'export':  _do_export,
    'stats':   _do_stats,
}


if __name__ == '__main__':
    args, configs = args_module.parse_exper()

    if args.debug:
        set_trace()

    handler = HANDLERS.get(args.action)
    if handler is None:
        raise SystemExit(f"Unknown action: {args.action!r}")
    handler(args, configs)
