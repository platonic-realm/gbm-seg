# Python Imprts
import logging
import os
import random

# Must be set before `import torch` (cuBLAS reads it once at first CUDA init).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Library Imports
import numpy as np
import torch
from torch.utils.data import random_split

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


def main_train(_configs):
    if _configs['logging']['log_summary']:
        summerize_configs(_configs)

    seed_everything(SEED)
    split_generator = torch.Generator().manual_seed(SEED)

    if _configs['trainer']['cudnn_benchmark']:
        # Benchmark mode picks the fastest cuDNN kernel based on input shapes,
        # which is non-deterministic. seed_everything() above enabled
        # deterministic algorithms; honouring benchmark here would silently
        # break that guarantee. Warn and skip.
        logging.warning(
            "trainer.cudnn_benchmark=True is incompatible with deterministic "
            "algorithms; skipping cuDNN benchmarking.")

    factory = Factory(_configs)

    train_dataset = factory.createTrainDataset()
    train_ratio = 0.95
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(train_dataset,
                                                [train_size, val_size],
                                                generator=split_generator)

    valid_dataloader = factory.createValidDataLoader(valid_dataset)
    train_dataloader = factory.createTrainDataLoader(train_dataset)

    model = factory.createModel(train_dataset.dataset.getNumberOfChannels(),
                                train_dataset.dataset.getNumberOfClasses())
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
                                               train_dataset.dataset.getNumberOfClasses())

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
                                    train_dataset.dataset.getNumberOfClasses())

    trainer.train()


if __name__ == '__main__':
    _, configs = args.parse_indep("Training Unet3D for GBM segmentation")
    configure_logger(configs)
    main_train(configs)
