# Python Imprts
import logging
import os
import random

# Must be set before `import torch` (cuBLAS reads it once at first CUDA init).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Library Imports
import numpy as np
import torch

from src.data.folds import read_assignments
from src.train.factory import Factory

# Local Imports
from src.utils import args
from src.utils.misc import configure_logger, summerize_configs

SEED = 88233474


def seed_everything(_seed: int):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    # Force deterministic CUDA / cuDNN kernels. Small speed regression
    # (~5-10%) in exchange for bit-exact reproducibility across runs.
    # `warn_only=True` so ops without a deterministic implementation log
    # a warning rather than raise — flip to False for strict mode.
    torch.use_deterministic_algorithms(True, warn_only=True)


def main_train(_configs, _fold: int = 0):
    """Run one training fold.

    A1: training and validation are partitioned subject-wise by
    ``<experiment>/fold_assignments.yaml``. ``_fold`` (0..k-1) selects
    which fold to train. The held-out test cohort (``ds_test/``) is
    untouched.
    """
    if _configs['logging']['log_summary']:
        summerize_configs(_configs)

    seed_everything(SEED)

    if _configs['trainer']['cudnn_benchmark']:
        # Benchmark mode picks the fastest cuDNN kernel based on input shapes,
        # which is non-deterministic. seed_everything() above enabled
        # deterministic algorithms; honouring benchmark here would silently
        # break that guarantee. Warn and skip.
        logging.warning(
            "trainer.cudnn_benchmark=True is incompatible with deterministic "
            "algorithms; skipping cuDNN benchmarking.")

    factory = Factory(_configs)

    # A1: subject-wise stratified 5-fold CV. Resolve which TIFFs belong to
    # the chosen fold's train and validation sets.
    fold_assignments = read_assignments(_configs['root_path'])
    if not 0 <= _fold < len(fold_assignments):
        raise ValueError(
            f"--fold {_fold} out of range; experiment has "
            f"{len(fold_assignments)} folds.")
    fold = fold_assignments[_fold]
    logging.info(
        "Fold %d: %d training subjects, %d validation subjects "
        "(validation groups: %s).",
        _fold, len(fold['train']), len(fold['valid']),
        fold.get('groups_in_valid', '?'))

    train_dataset = factory.createTrainDataset(file_filter=fold['train'])
    valid_dataset = factory.createValidDataset(file_filter=fold['valid'])

    valid_dataloader = factory.createValidDataLoader(valid_dataset)
    train_dataloader = factory.createTrainDataLoader(train_dataset)

    model = factory.createModel(train_dataset.getNumberOfChannels(),
                                train_dataset.getNumberOfClasses())
    loss_function = factory.createLoss()
    optimizer = factory.createOptimizer(model, loss_function)
    lr_scheduler = factory.createScheduler(optimizer)

    stepper = factory.createStepper(model, optimizer, loss_function)
    snapper = factory.createSnapper()

    visualizer = factory.createVisualizer()
    profiler = factory.createProfiler()
    metric_logger = factory.createMetricLogger(model,
                                               valid_dataloader,
                                               loss_function,
                                               train_dataset.getNumberOfClasses())

    trainer = factory.createTrainer(model,
                                    loss_function,
                                    stepper,
                                    snapper,
                                    profiler,
                                    visualizer,
                                    metric_logger,
                                    lr_scheduler,
                                    train_dataloader,
                                    valid_dataloader,
                                    train_dataset.getNumberOfClasses())

    trainer.train()


if __name__ == '__main__':
    _, configs = args.parse_indep("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    main_train(configs)
