"""
Author: Arash Fatehi
Date:   20.10.2022
file:   undet3d.py
"""

# Python Imports
import logging

# Library Imports
import torch
from torch import nn

# Local Imports
from src.models.unet3d.blocks import \
        create_encoder_layers, create_decoder_layers


class Unet3D(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            _input_channels,
            _kernel_size=(3, 3, 3),
            _feature_maps=(64, 128, 256, 512),
            _conv_layer_type='bcr'):

        super().__init__()

        self.input_channels = _input_channels
        self.kernel_size = _kernel_size
        self.feature_maps = _feature_maps
        self.conv_layer_type = _conv_layer_type

        logging.debug("######################")
        logging.debug("Initializing Unet3D with multiple encoders")
        logging.debug("input channels: %s", _input_channels)
        logging.debug("kernel size: %s", _kernel_size)
        logging.debug("feature maps: %s", _feature_maps)
        logging.debug("convolution layer type: %s", _conv_layer_type)

        logging.debug("Creating encoder for nephrin stain")
        self.encoder_layers = create_encoder_layers(_input_channels,
                                                    _feature_maps,
                                                    _kernel_size,
                                                    _conv_layer_type)

        logging.debug("Creating the decoder layer")
        self.decoder_layers = create_decoder_layers(_feature_maps,
                                                    _kernel_size,
                                                    _conv_layer_type)

        logging.debug("Creating the last layer")
        self.last_layer = nn.Conv3d(in_channels=_feature_maps[0],
                                    out_channels=3,
                                    kernel_size=1)

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, _x):
        encoder_features = []

        for encoder in self.encoder_layers:
            _x = encoder(_x)
            encoder_features.insert(0, _x)

        results = encoder_features[0]

        for i, decoder in enumerate(self.decoder_layers):

            if i == 0:
                encoder_feature = None
            else:
                encoder_feature = encoder_features[i+1]

            results = decoder(encoder_feature, results)

        logits = self.last_layer(results)

        results = self.final_activation(logits)
        results = torch.argmax(results, dim=1)

        return logits, results

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for encoder in self.encoder_layers:
            encoder.to(*args, **kwargs)
        for decoder in self.decoder_layers:
            decoder.to(*args, **kwargs)
        self.last_layer.to(*args, **kwargs)

        return self
