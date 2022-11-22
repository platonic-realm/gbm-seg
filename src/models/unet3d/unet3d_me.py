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
from src.models.blocks import create_encoder_layers, create_decoder_layers_me


class Unet3DME(nn.Module):
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
        self.x_encoder_layers = create_encoder_layers(_input_channels,
                                                      _feature_maps,
                                                      _kernel_size,
                                                      _conv_layer_type)

        logging.debug("Creating encoder for WGA stain")
        self.y_encoder_layers = create_encoder_layers(_input_channels,
                                                      _feature_maps,
                                                      _kernel_size,
                                                      _conv_layer_type)

        logging.debug("Creating encoder for Collagen IV stain")
        self.z_encoder_layers = create_encoder_layers(_input_channels,
                                                      _feature_maps,
                                                      _kernel_size,
                                                      _conv_layer_type)

        logging.debug("Creating the decoder layer")
        self.decoder_layers = create_decoder_layers_me(_feature_maps,
                                                       _kernel_size,
                                                       _conv_layer_type)

        logging.debug("Creating the last layer")
        self.last_layer = nn.Conv3d(_feature_maps[0], 1, 1)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, _x, _y, _z):
        x_encoder_features = []
        y_encoder_features = []
        z_encoder_features = []

        for encoder in self.x_encoder_layers:
            _x = encoder(_x)
            x_encoder_features.insert(0, _x)

        for encoder in self.y_encoder_layers:
            _y = encoder(_y)
            y_encoder_features.insert(0, _y)

        for encoder in self.z_encoder_layers:
            _z = encoder(_z)
            z_encoder_features.insert(0, _z)

        results = torch.cat((
                x_encoder_features[0],
                y_encoder_features[0],
                z_encoder_features[0]), dim=1)

        for i, decoder in enumerate(self.decoder_layers):

            if i == 0:
                encoder_features = None
            else:
                encoder_features = torch.cat((
                    x_encoder_features[i+1],
                    y_encoder_features[i+1],
                    z_encoder_features[i+1]), dim=1)

            results = decoder(encoder_features, results)

        results = self.last_layer(results)
        results = self.final_activation(results)

        return results

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for encoder in self.x_encoder_layers:
            encoder.to(*args, **kwargs)
        for encoder in self.y_encoder_layers:
            encoder.to(*args, **kwargs)
        for encoder in self.z_encoder_layers:
            encoder.to(*args, **kwargs)
        for decoder in self.decoder_layers:
            decoder.to(*args, **kwargs)
        self.last_layer.to(*args, **kwargs)

        return self
