# Python Imprts
import os
import sys
from argparse import ArgumentParser

import yaml

# Library Imports
# Local Imports
from src.utils.misc import sanity_check


def parse_exper() -> None:
    # Create the top-level parser
    parser = ArgumentParser(description='configuring GBM experiments')

    # Add the debug option
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help='enable debugging mode')

    # Create subparsers for each item in the list
    subparsers = \
        parser.add_subparsers(title='commands', dest='action')

    # Define a subparser for the 'list' action
    list_parser = \
        subparsers.add_parser('list',
                              help='provides a list of created experiments')
    list_parser.add_argument('-r',
                             '--root',
                             action='store_true',
                             help='the root directory of experminets')
    list_parser.add_argument('-s',
                             '--snapshots',
                             action='store',
                             help='list the snapshots of an experiment')

    # Define a subparser for the 'create' action
    create_parser = \
        subparsers.add_parser('create',
                              help='create a new experiment')
    create_parser.add_argument('name',
                               help='name of the experiment to create')

    create_parser.add_argument('-bs',
                               '--batch-size',
                               action='store',
                               default=8,
                               help='set the batch size for training')

    create_parser.add_argument('-zs',
                               '--z-scale',
                               action='store',
                               type=int,
                               default=None,
                               help='Z-axis interpolation factor applied at '
                                    'experiment-creation time. Channels 0-2 '
                                    'are trilinear-upsampled in Z; the label '
                                    'channel is np.repeat\'d so each manual '
                                    'annotation is stacked N times. The '
                                    'persisted value (trainer.data.z_scale) '
                                    'drives the validation-metric mask. '
                                    'Defaults to '
                                    'configs.experiments.default_z_scale '
                                    '(template: 6).')

    create_parser.add_argument('--reuse-dataset',
                               action='store',
                               default=None,
                               metavar='SRC_EXP',
                               help='Hardlink datasets/ from an existing '
                                    'experiment instead of resizing the '
                                    'TIFFs from scratch. Saves ~30 min of '
                                    'TIFF resize and, if the source has '
                                    'already been offline-augmented, '
                                    'inherits its aug cache too. Both '
                                    'experiments must live on the same '
                                    'filesystem. The hardlinks share inodes '
                                    'with the source — safe because the '
                                    'dataset TIFFs are read-only in '
                                    'training.')

    # Define a subparser for the 'delete' action
    delete_parser = \
        subparsers.add_parser('delete',
                              help='deletes the selected experiment')
    delete_parser.add_argument('name',
                               help='name of the experiment to delete')
    delete_parser.add_argument('-f',
                               '--force',
                               action='store_true',
                               help='confirm deletion (required, non-interactive)')
    delete_parser.add_argument('-w',
                               '--wandb',
                               action='store_true',
                               help='also delete the experiment\'s Weights & '
                                    'Biases runs (matched by W&B group = '
                                    'experiment name). Best-effort: a W&B '
                                    'failure does not block the local '
                                    'deletion.')

    # Define a subparser for the 'train' action
    train_parser = \
        subparsers.add_parser('train',
                              help='start/continue training')
    train_parser.add_argument('name',
                              help='name of the experiment.')
    train_parser.add_argument('--fold',
                              type=int,
                              default=None,
                              help='train a single CV fold by index (0..k-1). '
                                   'When omitted, train every fold sequentially '
                                   'and aggregate the validation metrics into '
                                   '<exp>/cv_results.{yaml,npz}. The fold '
                                   'partition lives in '
                                   '<experiment>/fold_assignments.yaml — k=5 '
                                   'by default per gbm.py create.')
    train_parser.add_argument('--all-data',
                              dest='all_data',
                              action='store_true',
                              help='Stage-B final model: train on every '
                                   'ds_train subject with no fold holdout. '
                                   'No validation set, fixed-epoch; the last '
                                   'snapshot is the deliverable. Mutually '
                                   'exclusive with --fold.')
    train_parser.add_argument('--epochs',
                              type=int,
                              default=None,
                              help='override configs.trainer.epochs for this '
                                   'run (typically paired with --all-data).')

    # Define a subparser for the 'aggregate-cv' action — consolidate
    # per-fold best_metrics.yaml files into a single cv_results.{yaml,npz}
    # and optionally post a W&B cv_summary run.
    agg_parser = \
        subparsers.add_parser('aggregate-cv',
                              help='consolidate per-fold metrics into '
                                   '<exp>/cv_results.{yaml,npz} and log a '
                                   'W&B cv_summary run (use after sbatch '
                                   'per-fold training jobs finish)')
    agg_parser.add_argument('name',
                            help='name of the experiment')

    # Define a subparser for the 'offline-aug' action — precompute the
    # offline-augmentation cache once, ahead of (DDP) training.
    offline_aug_parser = \
        subparsers.add_parser('offline-aug',
                              help='precompute the offline-augmentation '
                                   'cache for an experiment (run once after '
                                   'create, before training with '
                                   'enabled_offline: true — keeps DDP ranks '
                                   'from each recomputing it)')
    offline_aug_parser.add_argument('name',
                                    help='name of the experiment')

    # Define a subparser for the 'infer' action
    infer_parser = \
        subparsers.add_parser('infer',
                              help='create an inference session')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-s',
                              '--snapshot',
                              action='store',
                              required=True,
                              help='select the snapshot for inference')

    infer_parser.add_argument('-bs',
                              '--batch-size',
                              action='store',
                              default=8,
                              help='set the batch size for inference')

    infer_parser.add_argument('-sd',
                              '--sample-dimension',
                              action='store',
                              default='12, 256, 256',
                              help='set sample dimension for inference')

    infer_parser.add_argument('-st',
                              '--stride',
                              action='store',
                              default='1, 64, 64',
                              help='set the stride for inference')

    infer_parser.add_argument('-sf',
                              '--scale-factor',
                              action='store',
                              default=1,
                              help='set the scale for interpolation')

    infer_parser.add_argument('-in',
                              '--interpolation',
                              action='store',
                              default=False,
                              help='determines to interpolate or stack Z planes')

    infer_parser.add_argument('-f',
                              '--force',
                              action='store_true',
                              help='overwrite existing inference output (non-interactive)')

    infer_parser.add_argument('-sn',
                              '--sample-name',
                              action='store',
                              default=None,
                              help='infer a SINGLE volume by filename — the '
                                   'per-volume mode used for SLURM-array '
                                   'parallelism (one array task / GPU per '
                                   'volume; see sbatch/infer.sbatch). When '
                                   'omitted, every volume in the inference '
                                   'set is processed sequentially.')

    infer_parser.add_argument('--stitching',
                              action='store',
                              default=None,
                              choices=['gaussian', 'hann',
                                       'flat_softmax', 'sum_logits'],
                              help='override inference.stitching for this run. '
                                   'When omitted, the experiment yaml\'s value '
                                   'is used. Useful for inference-axis ablation.')

    infer_parser.add_argument('--output-name',
                              action='store',
                              default=None,
                              help='override the auto-generated inference tag '
                                   '(<snapshot>_<sample_dim>_<stride>_<scale>) '
                                   'with a custom directory name under '
                                   '<exp>/results-infer/. Required when two '
                                   'inference runs share all CLI params but '
                                   'differ on a non-CLI knob like --stitching.')

    infer_parser.add_argument('--labeled',
                              action='store_true',
                              help='infer on ds_test_labeled (the expert-'
                                   'annotated crops) instead of '
                                   'ds_test_unlabeled (the full glomerular '
                                   'volumes used for morphometry). Output '
                                   'lives in <exp>/results-infer-labeled/ '
                                   'so the two streams never collide. The '
                                   'crops are identical across annotators '
                                   '(only channel 3 differs), so inference '
                                   'reads the first annotator subdir '
                                   'alphabetically and the per-expert labels '
                                   'are picked up at stats time by '
                                   '`gbm.py stats`.')

    # Define a subparser for the 'infer-ablate' action — inference-axis
    # ablation. Reads an inference spec YAML and emits one `gbm.py infer`
    # command per cell (optionally sbatch-wrapped).
    infer_abl_parser = \
        subparsers.add_parser('infer-ablate',
                              help='materialise an inference-axis ablation '
                                   'study from a YAML spec; emits one '
                                   '`gbm.py infer` command per cell.')
    infer_abl_parser.add_argument('spec_path',
                                  help='path to the inference ablation spec '
                                       'YAML (e.g. ablation_specs/'
                                       'stitch_pilot_infer.yaml)')
    infer_abl_parser.add_argument('--sbatch',
                                  dest='sbatch_wrapper',
                                  default=None,
                                  help='if set, emit '
                                       '`sbatch <wrapper> '
                                       '<base_exp> <snapshot> <tag> '
                                       '<stitching>` commands instead of '
                                       'plain `python gbm.py infer ...`')

    # Define a subparser for the 'post processsing' action
    infer_parser = \
        subparsers.add_parser('psp',
                              help='post processsing to remove noises and artifacts')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot for visualizations')

    infer_parser.add_argument('-mc',
                              '--max-concurrent',
                              action='store',
                              required=True,
                              help='number of worker processes for post processing')

    infer_parser.add_argument('--labeled',
                              action='store_true',
                              help='process the labeled-inference output at '
                                   '<exp>/results-infer-labeled/<tag>/ '
                                   'instead of the default '
                                   '<exp>/results-infer/<tag>/. Pair with '
                                   '`gbm.py infer --labeled`.')

    # Define a subparser for the 'morph' action
    infer_parser = \
        subparsers.add_parser('morph',
                              help='do the morphometric analysis for a sample')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    infer_parser.add_argument('-sn',
                              '--sample-name',
                              action='store',
                              required=True,
                              help='the relative path to the sample')

    # Define a subparser for the 'blender' action
    infer_parser = \
        subparsers.add_parser('blender',
                              help='prepares for blender visualizations')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    infer_parser.add_argument('-sn',
                              '--sample-name',
                              action='store',
                              required=True,
                              help='the relative path to the sample')

    # Define a subparser for the 'render' action
    infer_parser = \
        subparsers.add_parser('render',
                              help='create blender visualizations')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    # Define a subparser for the 'export' action
    infer_parser = \
        subparsers.add_parser('export',
                              help='export the results and analysis')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    # Define a subparser for the 'ablate' action
    ablate_parser = \
        subparsers.add_parser('ablate',
                              help='materialise an ablation study from a YAML spec')
    ablate_parser.add_argument('spec_path',
                               help='path to the ablation spec YAML')
    ablate_parser.add_argument('--sbatch',
                               dest='sbatch_wrapper',
                               default=None,
                               help='if set, emit `sbatch <path> <name> <fold>` '
                                    'commands instead of plain python ones')

    # Define a subparser for the 'stats' action
    infer_parser = \
        subparsers.add_parser('stats',
                              help='generate statistics')

    infer_parser.add_argument('name',
                              help='name of the experiment.')

    infer_parser.add_argument('-it',
                              '--inference-tag',
                              action='store',
                              required=True,
                              help='select the snapshot used for inference')

    infer_parser.add_argument('--clipping',
                              action='store_true',
                              help='removes unreal values from the statistics')

    infer_parser.add_argument('-sn',
                              '--sample-name',
                              action='store',
                              default=None,
                              help='process ONLY this sample (per-sample '
                                   'array-task mode); the reduce step '
                                   '(stats-reduce) then aggregates. Omit for '
                                   'the single-process full run.')

    # `stats-reduce` — aggregate the per-sample sidecars written by the
    # parallel `stats -sn <sample>` array tasks into the cohort outputs.
    stats_reduce_parser = subparsers.add_parser(
        'stats-reduce',
        help='aggregate per-sample stats sidecars into the cohort-level '
             'outputs + publication figures (parallel stats reduce step)')
    stats_reduce_parser.add_argument('name', help='name of the experiment.')
    stats_reduce_parser.add_argument('-it', '--inference-tag', required=True,
                                     help='inference tag to reduce')
    stats_reduce_parser.add_argument('--clipping', action='store_true',
                                     help='accepted for CLI symmetry; the '
                                          'per-sample stage already clipped')

    # `labels-as-pred` — copy training labels into the inference output
    # layout so psp/morph/stats can be run against ground truth. See
    # src/infer/labels_as_pred.py.
    lap_parser = subparsers.add_parser(
        'labels-as-pred',
        help='materialise training labels (from <exp>/datasets/ds_train/) '
             'as if they were model predictions, so the downstream pipeline '
             'can compute morph/stats on the ground truth itself')
    lap_parser.add_argument('name', help='name of the experiment.')
    lap_parser.add_argument(
        '--output-tag',
        default=None,
        help="output inference tag (default 'labels_train'). The downstream "
             "pipeline (psp/morph/stats) is then run with -it <this-tag> "
             "exactly as for a model inference.")
    lap_parser.add_argument(
        '--z-repeat',
        type=int,
        default=1,
        help='Extra Z-upsampling factor applied via np.repeat(..., axis=0). '
             'Defaults to 1 (no-op) because ds_train/ TIFFs are already at '
             'the upsampled Z grid produced by `gbm.py create`. Pass 6 only '
             'when feeding native-Z labels (rare).')

    # Parse the arguments
    args = parser.parse_args()

    with open('./configs/template.yaml', encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if hasattr(args, 'name'):
        configs['root_path'] = os.path.join(configs['experiments']['root'],
                                            args.name)
    return args, configs


def parse_indep(_description: str):
    parser = ArgumentParser(description=_description)
    parser.add_argument("-c", "--config",
                        default="./configs/template.yaml",
                        help="Configuration's path")

    args = parser.parse_args()
    with open(args.config, encoding='UTF-8') as config_file:
        configs = yaml.safe_load(config_file)

    return args, sanity_check(configs)
