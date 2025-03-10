# Python Imprts

# Library Imports

# Local Imports
from src.utils.misc import configure_logger
from src.utils import args
from src.utils.misc import blender_render
from src.train.factory import Factory


def main_infer(_configs):
    factory = Factory(_configs)

    data_loaders = factory.createInferenceDataLoaders()

    morph = factory.createMorphModule()

    snapper = factory.createSnapper()

    for data_loader in data_loaders:

        model = factory.createModel(data_loader.dataset.getNumberOfChannels(),
                                    data_loader.dataset.getNumberOfClasses(),
                                    _inference=True,
                                    _result_shape=data_loader.dataset.getResultShape())

        inferer = factory.createInferer(model,
                                        data_loader,
                                        morph,
                                        snapper)

        inferer.infer()

    morph_analysis(_configs['inference']['result_dir'])

    blender_prepare(_configs['inference']['result_dir'])

    blender_render(_configs['inference']['result_dir'])


if __name__ == '__main__':
    _, configs = args.parse_indep("Inferance -> GBM segmentation")
    configure_logger(configs)

    main_infer(configs)
