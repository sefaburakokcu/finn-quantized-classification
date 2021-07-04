from torch import nn
from torch.nn import Sequential

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.quant import IntBias

from .common import CommonIntActQuant, CommonUintActQuant
from .common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant


class DwsConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            weight_bit_width,
            act_bit_width,
            pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=activation_scaling_per_channel,
            return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TinyMobileNet(nn.Module):

    def __init__(
            self,
            channels,
            first_stage_stride,
            weight_bit_width, 
            act_bit_width, 
            in_bit_width,
            in_ch=3,
            num_classes=1000):
        super(TinyMobileNet, self).__init__()
        init_block_channels = channels[0][0]

        self.features = Sequential()
        init_block = ConvBlock(
            in_channels=in_ch,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=2,
            weight_bit_width=in_bit_width,
            activation_scaling_per_channel=True,
            act_bit_width=act_bit_width)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    weight_bit_width=weight_bit_width,
                    act_bit_width=act_bit_width,
                    pw_activation_scaling_per_channel=pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = QuantAvgPool2d(kernel_size=7, stride=1, bit_width=weight_bit_width)
        self.output = QuantLinear(
            in_channels, num_classes,
            bias=True,
            bias_quant=IntBias,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=weight_bit_width)

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out


def tinymobilenet_v1(cfg):

    channels = [[32], [64], [128, 128], [256, 256], [512, 512], [512, 512]]
    first_stage_stride = False
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = TinyMobileNet(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        channels=channels,
        first_stage_stride=first_stage_stride,
        num_classes=num_classes,
        in_ch=in_channels)

    return net