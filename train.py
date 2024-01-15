# Python Imprts
import logging

# Library Imports
import torch

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
    train_dataloader = factory.createTrainDataLoader(train_dataset)

    valid_dataset = factory.createValidDataset()
    valid_dataloader = factory.createValidDataLoader(valid_dataset)

    model = factory.createModel(train_dataset.getNumberOfChannels(),
                                train_dataset.getNumberOfClasses())
    loss_function = factory.createLoss()
    optimizer = factory.createOptimizer(model)
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
