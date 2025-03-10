# Python Imprts
import logging

# Library Imports
import torch
from torch.utils.data import random_split

# Local Imports
from src.utils import args
from src.utils.misc import configure_logger
from src.utils.misc import summerize_configs
from src.train.factory import Factory


def main_train(_configs):
    if _configs['logging']['log_summary']:
        summerize_configs(_configs)

    if _configs['trainer']['cudnn_benchmark']:
        torch.backends.cudnn.benchmark = True
        logging.info("Enabling cudnn benchmarking")

    torch.manual_seed(88233474)

    factory = Factory(_configs)

    train_dataset = factory.createTrainDataset()
    # analyze_dataset(train_dataset)
    # Define the split ratio
    train_ratio = 0.95
    # Calculate the lengths of train and validation sets
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(train_dataset,
                                                [train_size, val_size])

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
