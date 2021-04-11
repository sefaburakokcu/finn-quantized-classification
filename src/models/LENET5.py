
import torch
from torch.nn import Module, ModuleList, MaxPool2d

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantHardTanh
from brevitas.core.restrict_val import RestrictValueType
from .tensor_norm import TensorNorm
from .common import CommonWeightQuant, CommonActQuant


CNV_OUT_CH_POOL = [(6, True), (16, True), (120, False)]
INTERMEDIATE_FC_FEATURES = [(120, 84)]
LAST_FC_IN_FEATURES = 84
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 5


class LENET5(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(LENET5, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(QuantConv2d(
                kernel_size=KERNEL_SIZE,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(QuantHardTanh(
                act_quant=CommonActQuant, bit_width=act_bit_width, 
                max_val=1.0, min_val=-1.0, 
                return_quant_tensor=False))
            # self.conv_features.append(QuantIdentity(
            #     act_quant=CommonActQuant,
            #     bit_width=act_bit_width))

            # Original Lenet5 uses AvgPooling. 
            # However, FINN does not support AvgPooling yet.
            if is_pool_enabled:  
                self.conv_features.append(MaxPool2d(kernel_size=2,
                                                    stride=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            self.linear_features.append(QuantHardTanh(
                act_quant=CommonActQuant, bit_width=act_bit_width, 
                max_val=1.0, min_val=-1.0, 
                return_quant_tensor=False))
            # self.linear_features.append(QuantIdentity(
            #     act_quant=CommonActQuant,
            #     bit_width=act_bit_width))

        self.linear_features.append(QuantLinear(
            in_features=LAST_FC_IN_FEATURES,
            out_features=num_classes,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        
        self.name = 'LENET5'
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def lenet5(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    net = LENET5(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels)
    return net
