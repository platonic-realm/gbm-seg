# Python Imports
import logging

# Library Imports
import torch
from torch import nn

# Local Imports
from src.models.unet3d.blocks import create_decoder_layers, create_encoder_layers


class Unet3D(nn.Module):
    """3D U-Net.

    A3 refactor: this module is now stateless w.r.t. inference. It just
    returns ``(logits, outputs)`` per forward pass. Sliding-window
    accumulation across overlapping patches now lives in
    ``src/infer/stitching.py:StitchAccumulator``, driven by
    ``src/infer/inference.py:Inference``.
    """

    def __init__(
            self,
            _name: str,
            _input_channels,
            _number_of_classes,
            _encoder_kernel_size=(3, 3, 3),
            _encoder_padding='same',
            _decoder_kernel_size=(3, 3, 3),
            _decoder_padding='same',
            _feature_maps=(64, 128, 256, 512),
            _conv_layer_type='bcr',
            _sample_dimension=None,
            _z_deduction_per_stage='auto',
            _deep_supervision: bool = False,
            _ds_levels: int = 2):

        super().__init__()

        self.name = _name
        self.sample_dimension = _sample_dimension

        self.input_channels = _input_channels
        self.number_of_classes = _number_of_classes
        self.encoder_kernel_size = _encoder_kernel_size
        self.encoder_padding = _encoder_padding
        self.decoder_kernel_size = _decoder_kernel_size
        self.decoder_padding = _decoder_padding
        self.feature_maps = _feature_maps
        self.conv_layer_type = _conv_layer_type

        # Z deduction per encoder pool. 'auto' derives it from the Z patch
        # depth — the encoder reduces Z to ~half its depth over the
        # (num_stages - 1) pools — the same anisotropic rule as SwinUNETR;
        # an explicit int overrides.
        num_stages = len(_feature_maps)
        if (isinstance(_z_deduction_per_stage, str)
                and _z_deduction_per_stage.lower() == 'auto'):
            sample_z = int(_sample_dimension[0])
            self.z_deduction = max(
                1, round(sample_z / (2 * max(1, num_stages - 1))))
        else:
            self.z_deduction = int(_z_deduction_per_stage)

        logging.debug("Initializing Unet3D: in_ch=%s, features=%s, kernels=%s",
                      _input_channels, _feature_maps, _encoder_kernel_size)

        self.encoder_layers = create_encoder_layers(_input_channels,
                                                    _feature_maps,
                                                    _encoder_kernel_size,
                                                    _encoder_padding,
                                                    _sample_dimension,
                                                    _conv_layer_type,
                                                    self.z_deduction)

        self.decoder_layers = create_decoder_layers(_feature_maps,
                                                    _decoder_kernel_size,
                                                    _decoder_padding,
                                                    _sample_dimension,
                                                    _conv_layer_type,
                                                    self.z_deduction)

        self.last_layer = nn.Conv3d(in_channels=_feature_maps[0],
                                    out_channels=self.number_of_classes,
                                    kernel_size=1)

        self.final_activation = nn.Softmax(dim=1)

        # Deep supervision (C1.2): add `_ds_levels` auxiliary 1x1x1 heads at
        # the deepest decoder mid-levels. Decoder layer i produces
        # `reverse_feature_maps[i+1]`-channel features; we attach heads at
        # decoder indices [0, 1, ..., _ds_levels - 1] which are the deepest
        # _ds_levels decoders (each one above the final / shallowest decoder).
        # Each head projects features → num_classes at the level's spatial
        # resolution.
        self.deep_supervision = _deep_supervision and _ds_levels > 0
        if self.deep_supervision:
            reverse_feature_maps = list(reversed(_feature_maps))
            max_ds_levels = len(self.decoder_layers) - 1  # leave the final head as the main head
            self.ds_levels = min(_ds_levels, max_ds_levels)
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(in_channels=reverse_feature_maps[i + 1],
                          out_channels=self.number_of_classes,
                          kernel_size=1)
                for i in range(self.ds_levels)
            ])
        else:
            self.ds_levels = 0
            self.ds_heads = nn.ModuleList()

    def forward(self, _x):
        encoder_features = []

        for encoder in self.encoder_layers:
            _x = encoder(_x)
            encoder_features.insert(0, _x)

        outputs = encoder_features[0]

        intermediate_features = []
        for i, decoder in enumerate(self.decoder_layers):
            encoder_feature = None if i == 0 else encoder_features[i + 1]
            outputs = decoder(encoder_feature, outputs)
            # Cache intermediate features for DS heads (deepest first).
            if self.deep_supervision and i < self.ds_levels:
                intermediate_features.append(outputs)

        logits = self.last_layer(outputs)

        post = self.final_activation(logits)
        argmax = torch.argmax(post, dim=1)

        if self.deep_supervision:
            # Apply each head to its captured feature (paired by index, so
            # channel counts match), then return in monotonically-decreasing
            # spatial-resolution order so the geometric weight decay
            # 1.0, 0.5, 0.25, ... lines up with resolution.
            # `intermediate_features` are captured deepest-first during the
            # decoder loop, so reversing yields finest-first.
            aux_logits = [head(feat)
                          for head, feat in zip(self.ds_heads, intermediate_features)]
            aux_logits.reverse()
            return [logits, *aux_logits], argmax

        return logits, argmax

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for encoder in self.encoder_layers:
            encoder.to(*args, **kwargs)
        for decoder in self.decoder_layers:
            decoder.to(*args, **kwargs)
        self.last_layer.to(*args, **kwargs)
        for head in self.ds_heads:
            head.to(*args, **kwargs)

        return self
