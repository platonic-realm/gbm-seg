# Python Imports
import itertools
import logging

# Library Imports
import torch
from torch import nn
from torch.nn import functional as Fn

# Local Imports


class ConvLayer(nn.Module):
    def __init__(self,
                 _input_channels,
                 _output_channels,
                 _kernel_size,
                 _conv_layer_type,
                 _normalized_shape,
                 _padding):

        super().__init__()

        self.convolution = nn.Conv3d(_input_channels,
                                     _output_channels,
                                     _kernel_size,
                                     bias=False,
                                     padding=_padding)

        # self.normalization = nn.BatchNorm3d(_output_channels)
        self.normalization = nn.LayerNorm(_normalized_shape,
                                          elementwise_affine=False)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, _x):
        _x = self.convolution(_x)
        _x = self.normalization(_x)
        _x = self.activation(_x)
        return _x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.convolution.to(*args, **kwargs)
        self.normalization.to(*args, **kwargs)


class EncoderLayer(nn.Module):
    def __init__(self,
                 _input_channels,
                 _output_channels,
                 _kernel_size,
                 _conv_layer_type,
                 _normalized_shape_1,
                 _normalized_shape_2,
                 _padding,
                 _pooling,
                 _pool_kernel_size,
                 _pool_stride=None):

        super().__init__()

        self.pooling = _pooling

        # When _pool_stride is unset, MaxPool3d's default is stride == kernel
        # (legacy "halve" behaviour). Setting stride < kernel along an axis
        # produces "deduct" behaviour: output_dim = (input_dim - kernel) / stride + 1.
        self.pooling_layer = nn.MaxPool3d(kernel_size=_pool_kernel_size,
                                          stride=_pool_stride)

        self.convolution_1 = ConvLayer(_input_channels,
                                       _input_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _normalized_shape_1,
                                       _padding)

        self.convolution_2 = ConvLayer(_input_channels,
                                       _output_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _normalized_shape_2,
                                       _padding)

    def forward(self, _x):

        if self.pooling:
            _x = self.pooling_layer(_x)

        _x = self.convolution_1(_x)
        _x = self.convolution_2(_x)

        return _x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.pooling_layer.to(*args, **kwargs)
        self.convolution_1.to(*args, **kwargs)
        self.convolution_2.to(*args, **kwargs)


class DecoderLayer(nn.Module):
    def __init__(self,
                 _input_channels,
                 _output_channels,
                 _x_channels,
                 _kernel_size,
                 _conv_layer_type,
                 _normalized_shape_1,
                 _normalized_shape_2,
                 _padding,
                 _upsampling,
                 _scale_factor):

        super().__init__()

        self.scale_factor = _scale_factor
        self.upsampling = _upsampling

        self.convolution_1 = ConvLayer(_input_channels,
                                       _input_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _normalized_shape_1,
                                       _padding)

        self.convolution_2 = ConvLayer(_input_channels,
                                       _output_channels,
                                       _kernel_size,
                                       _conv_layer_type,
                                       _normalized_shape_2,
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
                _x = Fn.interpolate(_x,
                                    size=(_x.shape[2]*self.scale_factor[0],
                                          _x.shape[3]*self.scale_factor[1],
                                          _x.shape[4]*self.scale_factor[2]))

        _x = self.convolution_1(_x)
        _x = self.convolution_2(_x)

        return _x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.convolution_1.to(*args, **kwargs)
        self.convolution_2.to(*args, **kwargs)


def create_encoder_layers(_input_channels,
                          _feature_maps,
                          _kernel_size,
                          _padding,
                          _sample_shape,
                          _conv_layer_type,
                          _z_deduction_per_stage: int = 2):
    """Build the 4-stage encoder stack.

    Pooling per stage uses ``MaxPool3d(kernel=(z_kernel, 2, 2), stride=(1, 2, 2))``
    where ``z_kernel = z_deduction_per_stage + 1``, giving:

        output_Z = input_Z − z_deduction_per_stage
        output_XY = input_XY / 2

    With the default ``z_deduction_per_stage=2`` and ``sample_dim=[12, 256, 256]``,
    the Z/XY progression through the encoder is:

        L0: (12, 256, 256)  — no pooling
        L1: (10, 128, 128)
        L2: ( 8,  64,  64)
        L3: ( 6,  32,  32)  — bottleneck

    Pass ``_z_deduction_per_stage=0`` to recover the legacy "preserve Z" behaviour.
    """
    encoder_layers = nn.ModuleList([])

    logging.debug("######################")
    logging.debug("Entered into create_encoder_layers")
    logging.debug("Length of feature map: %s", len(_feature_maps))

    z_in = _sample_shape[0]
    z_kernel = _z_deduction_per_stage + 1
    pool_kernel_size = (z_kernel, 2, 2)
    pool_stride = (1, 2, 2)

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

        # Stage 0 has no pooling so it keeps the input Z; stages 1+ each deduct
        # ``_z_deduction_per_stage`` from the previous Z. Validity check: each
        # stage's *input* (the previous stage's output Z) must be ≥ the pool
        # kernel size, since that stage's pool needs that many slices to slide
        # over. Stage 0 has no pool so it's exempt.
        z_at_stage = z_in - i * _z_deduction_per_stage
        if z_at_stage <= 0:
            raise ValueError(
                f"Encoder stage {i} would have Z={z_at_stage} ≤ 0. Reduce "
                f"feature_maps depth or _z_deduction_per_stage, or increase "
                f"sample_dimension[0].")
        if i > 0:
            prev_z = z_in - (i - 1) * _z_deduction_per_stage
            if prev_z < z_kernel:
                raise ValueError(
                    f"Encoder stage {i} pool needs input Z >= {z_kernel} but "
                    f"got {prev_z} from the previous stage.")
        sample_shape = (z_at_stage,
                        int(_sample_shape[1]/(2**i)),
                        int(_sample_shape[2]/(2**i)))
        normalized_shape_1 = tuple(itertools.chain((input_channels, ), sample_shape))
        normalized_shape_2 = tuple(itertools.chain((output_channels, ), sample_shape))

        encoder_layers.append(
                EncoderLayer(input_channels,
                             output_channels,
                             _kernel_size,
                             _conv_layer_type,
                             _padding=_padding,
                             _normalized_shape_1=normalized_shape_1,
                             _normalized_shape_2=normalized_shape_2,
                             _pooling=pooling,
                             _pool_kernel_size=pool_kernel_size,
                             _pool_stride=pool_stride))

    return encoder_layers


def create_decoder_layers(_feature_maps,
                          _kernel_size,
                          _padding,
                          _sample_shape,
                          _conv_layer_type,
                          _z_deduction_per_stage: int = 2):
    """Build the decoder stack mirroring ``create_encoder_layers``.

    Each decoder layer's input tensor (after the upsample to the matching
    encoder skip's shape) determines the LayerNorm ``normalized_shape``.
    Z follows the encoder's deduction pattern in reverse — the deepest
    decoder operates on Z = ``sample_z - (L-1) * deduction``, and Z grows
    back to ``sample_z`` at the final decoder. Mirror of the encoder Z
    progression.
    """
    decoder_layers = nn.ModuleList([])

    logging.debug("######################")
    logging.debug("Entered into create_decoder_layers")
    logging.debug("Length of feature map: %s", len(_feature_maps))

    reverse_feature_maps = list(reversed(_feature_maps))

    feature_length = len(reverse_feature_maps) - 1
    num_stages = len(_feature_maps)
    for i in range(feature_length):

        if i == 0:
            input_channels = reverse_feature_maps[i]
            x_channels = input_channels
        else:
            input_channels = reverse_feature_maps[i] + \
                             reverse_feature_maps[i+1]
            x_channels = reverse_feature_maps[i]

        output_channels = reverse_feature_maps[i+1]

        logging.debug("Creating layer: %s", i)
        logging.debug("input_channels: %s", input_channels)
        logging.debug("ouput_channels: %s", output_channels)

        # Per the Unet3D forward loop: D[i] for i==0 keeps the bottleneck's
        # Z (deepest encoder stage); D[i>=1] upsamples to encoder_features[i+1]
        # which is encoder stage L-2-i, so Z = sample_z − (L-2-i) * deduction.
        if i == 0:
            z_at_stage = _sample_shape[0] - (num_stages - 1) * _z_deduction_per_stage
        else:
            z_at_stage = _sample_shape[0] - (num_stages - 2 - i) * _z_deduction_per_stage

        sample_shape = (z_at_stage,
                        int(_sample_shape[1]/(2**(feature_length - i - 1))),
                        int(_sample_shape[2]/(2**(feature_length - i - 1))))
        normalized_shape_1 = tuple(itertools.chain((input_channels, ), sample_shape))
        normalized_shape_2 = tuple(itertools.chain((output_channels, ), sample_shape))

        decoder_layers.append(
                DecoderLayer(input_channels,
                             output_channels,
                             x_channels,
                             _kernel_size=_kernel_size,
                             _conv_layer_type=_conv_layer_type,
                             _upsampling=True,
                             _padding=_padding,
                             _normalized_shape_1=normalized_shape_1,
                             _normalized_shape_2=normalized_shape_2,
                             _scale_factor=(1, 2, 2)))

    return decoder_layers


