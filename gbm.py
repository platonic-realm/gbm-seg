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
    z_scale = (int(args.z_scale)
               if args.z_scale is not None
               else int(configs['experiments'].get('default_z_scale', 1)))
    exp_cfg = configs['experiments']
    exper.create_new_experiment(
        _name=args.name,
        _root_path=exp_cfg['root'],
        _source_path=os.getcwd(),
        _dataset_path=exp_cfg['default_data_path'],
        _batch_size=int(args.batch_size),
        _voxel_size=exp_cfg['default_voxel_size'],
        _z_scale_factor=z_scale,
        _ds_train_subdir=exp_cfg.get('ds_train_subdir', 'ds_train'),
        _ds_test_labeled_subdir=exp_cfg.get('ds_test_labeled_subdir',
                                            'ds_test_labeled'),
        _ds_test_unlabeled_subdir=exp_cfg.get('ds_test_unlabeled_subdir',
                                              'ds_test_unlabeled'))


def _do_list(args, configs):
    root = args.root or configs['experiments']['root']
    if args.snapshots is None:
        exper.list_experiments(root)
    else:
        exper.list_snapshots(_name=args.snapshots, _root_path=root)


def _do_train(args, configs):
    configure_logger(configs, _log_to_file=True)
    exper.train_experiment(_name=args.name,
                           _root_path=configs['experiments']['root'],
                           _fold=args.fold,
                           _all_data=getattr(args, 'all_data', False),
                           _epochs=getattr(args, 'epochs', None))


def _do_aggregate_cv(args, configs):
    configure_logger(configs, _log_to_file=False)
    exper.aggregate_cv_experiment(_name=args.name,
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
                           _force=args.force,
                           _stitching=args.stitching,
                           _output_name=args.output_name)


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


def _do_ablate(args, configs):
    basic_logger()
    from src.ablation.runner import emit_commands, materialise
    from src.ablation.spec import parse_spec

    spec = parse_spec(args.spec_path)
    experiments_root = configs['experiments']['root']
    cell_paths = materialise(spec, experiments_root)
    commands = emit_commands(spec, cell_paths,
                             sbatch_wrapper=args.sbatch_wrapper)
    print(f"# {len(commands)} cells materialised under {experiments_root}")
    print("# Submit the following:")
    for cmd in commands:
        print(cmd)


def _do_infer_ablate(args, configs):
    basic_logger()
    from src.ablation.infer_spec import emit_infer_commands, parse_infer_spec

    spec = parse_infer_spec(args.spec_path)
    commands = emit_infer_commands(spec, sbatch_wrapper=args.sbatch_wrapper)
    print(f"# {len(commands)} inference cells from study '{spec.study}' "
          f"on {spec.base_experiment} / snapshot {spec.snapshot}")
    print("# Each cell writes to <exp>/results-infer/"
          f"{spec.study}__<cell>/. Run psp/morph/stats per cell with "
          "`gbm.py psp <exp> -it {spec.study}__<cell> ...`")
    print("# Submit the following:")
    for cmd in commands:
        print(cmd)


# Aligned by hand for readability; suppress pycodestyle's "multiple spaces after ':'".
HANDLERS = {
    'create':       _do_create,
    'list':         _do_list,
    'train':        _do_train,
    'aggregate-cv': _do_aggregate_cv,
    'delete':       _do_delete,
    'infer':        _do_infer,
    'psp':          _do_psp,
    'morph':        _do_morph,
    'blender':      _do_blender,
    'render':       _do_render,
    'export':       _do_export,
    'stats':        _do_stats,
    'ablate':       _do_ablate,
    'infer-ablate': _do_infer_ablate,
}


if __name__ == '__main__':
    args, configs = args_module.parse_exper()

    if args.debug:
        set_trace()

    handler = HANDLERS.get(args.action)
    if handler is None:
        raise SystemExit(f"Unknown action: {args.action!r}")
    handler(args, configs)
