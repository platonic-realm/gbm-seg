"""
Author: Arash Fatehi
Date:   20.10.2022
file:   undet3d.py
"""

# Python Imports
import logging

# Library Imports
import torch
import torch.nn.functional as Fn
from torch import Tensor
from torch import nn

# Local Imports
from src.models.blocks import \
        create_encoder_layers, create_decoder_layers_ss


class Unet3DSS(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
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

        self.sample_dimension = _sample_dimension

        self.input_channels = _input_channels
        self.encoder_kernel_size = _encoder_kernel_size
        self.encoder_padding = _encoder_padding
        self.decoder_kernel_size = _decoder_kernel_size
        self.decoder_padding = _decoder_padding
        self.feature_maps = _feature_maps
        self.conv_layer_type = _conv_layer_type

        logging.debug("######################")
        logging.debug("Initializing Unet3D for Semi-Supervised learning")
        logging.debug("input channels: %s", _input_channels)
        logging.debug("encoder kernel size: %s", _encoder_kernel_size)
        logging.debug("encoder padding: %s", _encoder_padding)
        logging.debug("decoder kernel size: %s", _decoder_kernel_size)
        logging.debug("decoder padding: %s", _decoder_padding)
        logging.debug("feature maps: %s", _feature_maps)
        logging.debug("convolution layer type: %s", _conv_layer_type)

        logging.debug("Creating encoder for nephrin stain")
        self.encoder_layers = create_encoder_layers(_input_channels,
                                                    _feature_maps,
                                                    _encoder_kernel_size,
                                                    _encoder_padding,
                                                    _conv_layer_type)

        logging.debug("Creating the decoder layer")
        self.decoder_layers = create_decoder_layers_ss(_feature_maps,
                                                       _decoder_kernel_size,
                                                       _decoder_padding,
                                                       _conv_layer_type)

        logging.debug("Creating segmentation last layer")
        self.segmentation_last_layer = nn.Conv3d(
                in_channels=_feature_maps[0],
                out_channels=_number_of_classes,
                kernel_size=1)

        self.segmentation_activation = nn.Softmax(dim=1)

        logging.debug("Creating interpolation last layer")
        self.interpolation_last_layer = nn.Conv3d(in_channels=_feature_maps[0],
                                                  out_channels=_input_channels,
                                                  kernel_size=1)

    def forward(self, _x, _offsets=None):
        encoder_features = []

        for encoder in self.encoder_layers:
            _x = encoder(_x)
            encoder_features.insert(0, _x)

        outputs = encoder_features[0]

        # We stored 2 branches of execution in decoder_layers
        # So we should execute these branches separately
        for i in range(len(self.decoder_layers)-2):
            decoder = self.decoder_layers[i]
            if i == 0:
                encoder_feature = None
            else:
                encoder_feature = encoder_features[i+1]

            outputs = decoder(encoder_feature, outputs)

        # Executing the segmentation branch
        decoder = self.decoder_layers[-2]
        segmentation_outputs = decoder(encoder_features[-1], outputs)

        segmentation_logits = \
            self.segmentation_last_layer(segmentation_outputs)

        segmentation_outputs = \
            self.segmentation_activation(segmentation_logits)
        segmentation_outputs = torch.argmax(segmentation_outputs, dim=1)

        # Executing the interpolation branch
        decoder = self.decoder_layers[-1]
        interpolation_ouputs = decoder(encoder_features[-1], outputs)
        interpolation_ouputs = \
            self.interpolation_last_layer(interpolation_ouputs)
        interpolation_ouputs = Fn.normalize(interpolation_ouputs,
                                            p=2,
                                            dim=None)
        interpolation_ouputs = outputs.mul(255).type(torch.uint8)

        return segmentation_logits, segmentation_outputs, interpolation_ouputs

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for encoder in self.encoder_layers:
            encoder.to(*args, **kwargs)
        for decoder in self.decoder_layers:
            decoder.to(*args, **kwargs)
        self.segmentation_last_layer.to(*args, **kwargs)
        self.interpolation_last_layer.to(*args, **kwargs)

        return self

    def get_result(self) -> Tensor:
        if not self.inference:
            return None
        return self.result_tensor
