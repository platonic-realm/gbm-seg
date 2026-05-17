"""3D U-Net package. Exports the model class and a ``build(configs, ...)``
factory function consumed by the model registry in ``src/models/__init__.py``.
"""

from src.models.unet3d.unet3d import Unet3D


def build(configs, input_channels, num_classes):
    """Build a :class:`Unet3D` from ``configs.trainer.model.*`` (and the
    training sample_dimension, which the model uses for diagnostic logging).
    Deep supervision (C1.2) is opt-in via ``configs.trainer.deep_supervision``.
    """
    model_cfg = configs['trainer']['model']
    sample_dimension = configs['trainer']['train_ds']['sample_dimension'].copy()
    ds_cfg = configs['trainer'].get('deep_supervision', {})
    return Unet3D(
        _name=model_cfg['name'],
        _input_channels=input_channels,
        _number_of_classes=num_classes,
        _encoder_kernel_size=model_cfg['encoder_kernel'],
        _decoder_kernel_size=model_cfg['decoder_kernel'],
        _feature_maps=model_cfg['feature_maps'],
        _sample_dimension=sample_dimension,
        _z_deduction_per_stage=model_cfg.get('z_deduction_per_stage', 'auto'),
        _deep_supervision=ds_cfg.get('enabled', False),
        _ds_levels=ds_cfg.get('levels', 2),
    )


__all__ = ['Unet3D', 'build']
