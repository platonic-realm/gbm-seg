"""Swin-UNETR package. Exports the model class and the ``build(configs, ...)``
factory function consumed by the model registry.
"""

from src.models.swin_unetr.swin_unetr import SwinUNETR3D


def build(configs, input_channels, num_classes):
    """Build a :class:`SwinUNETR3D` from ``configs.trainer.model.*``.

    Reads optional SwinUNETR-specific hyperparameters from
    ``configs.trainer.model.swin_unetr.*``; defaults match the original
    paper for a small/medium variant tuned to shallow Z stacks.
    Deep supervision (C1.2) is opt-in via ``configs.trainer.deep_supervision``.
    """
    model_cfg = configs['trainer']['model']
    swin_cfg = model_cfg.get('swin_unetr', {})
    sample_dimension = configs['trainer']['train_ds']['sample_dimension'].copy()
    ds_cfg = configs['trainer'].get('deep_supervision', {})

    return SwinUNETR3D(
        _name=model_cfg['name'],
        _input_channels=input_channels,
        _number_of_classes=num_classes,
        _sample_dimension=sample_dimension,
        _feature_size=int(swin_cfg.get('feature_size', 24)),
        _depths=tuple(swin_cfg.get('depths', (2, 2, 2, 2))),
        _num_heads=tuple(swin_cfg.get('num_heads', (3, 6, 12, 24))),
        _window_size_xy=int(swin_cfg.get('window_size_xy', 7)),
        _mlp_ratio=float(swin_cfg.get('mlp_ratio', 4.0)),
        _qkv_bias=bool(swin_cfg.get('qkv_bias', True)),
        _drop_rate=float(swin_cfg.get('drop_rate', 0.0)),
        _attn_drop_rate=float(swin_cfg.get('attn_drop_rate', 0.0)),
        _z_deduction_per_stage=swin_cfg.get('z_deduction_per_stage', 'auto'),
        _gradient_checkpointing=configs['trainer'].get(
            'gradient_checkpointing', 'auto'),
        _deep_supervision=ds_cfg.get('enabled', False),
        _ds_levels=ds_cfg.get('levels', 2),
    )


__all__ = ['SwinUNETR3D', 'build']
