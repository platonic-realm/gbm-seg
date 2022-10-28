"""
Author: Arash Fatehi
Date:   20.10.2022
File:   blocks.py
"""

# Python Imports
import logging

# Library Imports
import torch
from torch.nn import functional as Fn
from torch import nn

# Local Imports


class ConvLayer(nn.Module):
    def __init__(self,
                 _input_channels,
                 _output_channels,
                 _kernel_size,
                 _conv_layer_type,
                 _padding):

        super().__init__()

        self.convolution = nn.Conv3d(_input_channels,
                                     _output_channels,
                                     _kernel_size,
                                     bias=True,
                                     padding=_padding)

        self.batch_normalization = nn.BatchNorm3d(_output_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, _x):
        _x = self.convolution(_x)
        _x = self.batch_normalization(_x)
        _x = self.activation(_x)
        return _x


class EncoderLayer(nn.Module):
    def __init__(self,
                 _input_channels,
                 _output_channels,
                 _kernel_size,
                 _conv_layer_type,
                 _padding,
                 _pooling,
                 _pool_kernel_size):

        super().__init__()

        self.pooling = _pooling

        self.pooling_layer = nn.MaxPool3d(kernel_size=_pool_kernel_size)

        self.convolution_1 = ConvLayer(_input_channels,
                                       _input_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _padding)

        self.convolution_2 = ConvLayer(_input_channels,
                                       _output_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _padding)

    def forward(self, _x):

        if self.pooling:
            _x = self.pooling_layer(_x)

        _x = self.convolution_1(_x)
        _x = self.convolution_2(_x)

        return _x


class DecoderLayer(nn.Module):
    def __init__(self,
                 _input_channels,
                 _output_channels,
                 _x_channels,
                 _kernel_size,
                 _conv_layer_type,
                 _padding,
                 _upsampling,
                 _scale_factor):

        super().__init__()

        self.scale_factor = _scale_factor
        self.upsampling = _upsampling

#        self.upsampling_layer = nn.ConvTranspose3d(_x_channels,
#                                                   _x_channels,
#                                                   kernel_size=_kernel_size,
#                                                   stride=_scale_factor,
#                                                   padding=_padding)
#
        self.convolution_1 = ConvLayer(_input_channels,
                                       _input_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _padding)

        self.convolution_2 = ConvLayer(_input_channels,
                                       _output_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _padding)

    def forward(self, _encoder_features, _x):

        if self.upsampling:
            if _encoder_features is not None:
                _x = Fn.interpolate(_x,
                                    size=(_encoder_features.shape[2],
                                          _encoder_features.shape[3],
                                          _encoder_features.shape[4]))

                _x = torch.cat((_encoder_features, _x), dim=1)
            else:
                # _x = self.upsampling_layer(_x)
                _x = Fn.interpolate(_x, self.scale_factor)

        _x = self.convolution_1(_x)
        _x = self.convolution_2(_x)

        return _x


def create_encoder_layers(_input_channels,
                          _feature_maps,
                          _kernel_size,
                          _conv_layer_type):
    encoder_layers = []

    logging.debug("Entered into create_encoder_layers")
    logging.debug("Length of feature map: %s", len(_feature_maps))

    # pylint: disable=consider-using-enumerate
    for i in range(len(_feature_maps)):
        if i == 0:
            pooling = False
            input_channels = _input_channels
        else:
            pooling = True
            input_channels = _feature_maps[i-1]

        output_channels = _feature_maps[i]

        logging.debug("Creating layer: %s", i)
        logging.debug("input_channels: %s", input_channels)
        logging.debug("ouput_channels: %s", output_channels)
        logging.debug("pooling: %s", pooling)

        encoder_layers.append(
                EncoderLayer(input_channels,
                             output_channels,
                             _kernel_size,
                             _conv_layer_type,
                             _padding='same',
                             _pooling=pooling,
                             _pool_kernel_size=(2, 2, 2)))

    return encoder_layers


def create_decoder_layers(_feature_maps,
                          _kernel_size,
                          _conv_layer_type):
    decoder_layers = []

    logging.debug("Entered into create_decoder_layers")
    logging.debug("Length of feature map: %s", len(_feature_maps))

    reverse_feature_maps = list(reversed(_feature_maps))

    for i in range(len(reverse_feature_maps) - 1):

        if i == 0:
            input_channels = 3*reverse_feature_maps[i]
            x_channels = input_channels
        else:
            input_channels = reverse_feature_maps[i] + 3*reverse_feature_maps[i+1]
            x_channels = reverse_feature_maps[i]

        output_channels = reverse_feature_maps[i+1]

        logging.debug("Creating layer: %s", i)
        logging.debug("input_channels: %s", input_channels)
        logging.debug("ouput_channels: %s", output_channels)

        decoder_layers.append(
                DecoderLayer(input_channels,
                             output_channels,
                             x_channels,
                             _kernel_size=(2, 2, 2),
                             _conv_layer_type=_conv_layer_type,
                             _upsampling=True,
                             _padding='same',
                             _scale_factor=(2, 2, 2)))

    return decoder_layers
