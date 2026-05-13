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
            _sample_dimension=None):

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

        logging.debug("Initializing Unet3D: in_ch=%s, features=%s, kernels=%s",
                      _input_channels, _feature_maps, _encoder_kernel_size)

        self.encoder_layers = create_encoder_layers(_input_channels,
                                                    _feature_maps,
                                                    _encoder_kernel_size,
                                                    _encoder_padding,
                                                    _sample_dimension,
                                                    _conv_layer_type)

        self.decoder_layers = create_decoder_layers(_feature_maps,
                                                    _decoder_kernel_size,
                                                    _decoder_padding,
                                                    _sample_dimension,
                                                    _conv_layer_type)

        self.last_layer = nn.Conv3d(in_channels=_feature_maps[0],
                                    out_channels=self.number_of_classes,
                                    kernel_size=1)

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, _x):
        encoder_features = []

        for encoder in self.encoder_layers:
            _x = encoder(_x)
            encoder_features.insert(0, _x)

        outputs = encoder_features[0]

        for i, decoder in enumerate(self.decoder_layers):
            encoder_feature = None if i == 0 else encoder_features[i + 1]
            outputs = decoder(encoder_feature, outputs)

        logits = self.last_layer(outputs)

        outputs = self.final_activation(logits)
        outputs = torch.argmax(outputs, dim=1)

        return logits, outputs

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for encoder in self.encoder_layers:
            encoder.to(*args, **kwargs)
        for decoder in self.decoder_layers:
            decoder.to(*args, **kwargs)
        self.last_layer.to(*args, **kwargs)

        return self
