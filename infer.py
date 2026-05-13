# Python Imprts

# Library Imports

# Local Imports
from src.train.factory import Factory
from src.utils import args
from src.utils.misc import configure_logger


def main_infer(_configs):
    factory = Factory(_configs)

    data_loaders = factory.createInferenceDataLoaders()
    morph = factory.createMorphModule()
    snapper = factory.createSnapper()

    # A3 refactor: the model is stateless w.r.t. inference (the sliding-window
    # accumulator now lives in Inference). Build it once and re-use it for every
    # volume; each Inference owns its own accumulator sized to the volume.
    first_dataset = data_loaders[0].dataset
    model = factory.createModel(first_dataset.getNumberOfChannels(),
                                first_dataset.getNumberOfClasses())

    for data_loader in data_loaders:
        inferer = factory.createInferer(model, data_loader, morph, snapper)
        inferer.infer()


if __name__ == '__main__':
    _, configs = args.parse_indep("Inferance -> GBM segmentation")
    configure_logger(configs)

    main_infer(configs)
