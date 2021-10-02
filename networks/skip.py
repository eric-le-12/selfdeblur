"""
This file generate the encoder-decoder with skip connection structure
@ Hieu Le Xuan, 2021
"""
import torch.nn as nn
import torch
from code.SelfDeblur.networks.common import act
from utils import layers


class Skip(nn.Module):
    """
    Construct a unet like model with skip connections
    """
    def __init__(
        self,
        num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad="zero",
        upsample_mode="nearest",
        downsample_mode="stride",
        act_fun="LeakyReLU",
        need1x1_up=True,
    ):
        super(Skip, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_channels_down = num_channels_down
        self.num_channels_up = num_channels_up
        self.num_channels_skip = num_channels_skip
        self.filter_size_up = filter_size_up
        self.filter_size_down = filter_size_down
        self.filter_skip_size = filter_skip_size
        self.need_sigmoid = need_sigmoid
        self.need_bias = need_bias
        self.pad = pad
        self.upsample_mode = upsample_mode
        self.downsample_mode = downsample_mode
        self.act_fun = act_fun
        self.need1x1_up = need1x1_up

    def single_skip(self,
                    in_channels,
                    num_channels_skip,
                    filter_skip_size,
                    pad,
                    bias=True):
        """
        implement a skip module:
        receive N,C,H,W output N,Ck,H,W with filter size as filter_skip_size, padding method as pad
        """
        skip = nn.Sequential()
        padder = None
        downsampler = None
        # convolution module to decrease channels
        ## padding size for conv2d
        to_pad = (filter_skip_size - 1) / 2
        ## choosing path method : zero or flection
        if pad == "Reflection":
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0
        ## constructing conv2d
        conv2d = nn.Conv2d(
            in_channels,
            num_channels_skip,
            filter_skip_size,
            1,
            padding=to_pad,
            bias=bias,
        )
        ## leave only non -None layers
        layers = filter(lambda x: x is not None, [padder, conv2d, downsampler])
        ## downsampler if neccessary

        skip.add_module("Skip_conv", nn.Sequential(*layers))

        # adding bn and activation
        skip.add_module("Skip_BN", nn.BatchNorm2d(num_channels_skip))
        skip.add_module("Leaky_RELU", layers.act(act_fun="LeakyReLU"))

        return skip

    def deeper_before_concat(
        self,
        in_channels,
        num_channels_down,
        filter_down_size,
        stride,
        pad,
        downsampler,
        bias,
        act_fun,
    ):
        """
        con_block -> bn -> leaky relu -> nonlocal2D block ->...
        purpose : reduce spatial dimension while increasing depth / num of channels

        """
        before_concat = []
        ## append conv
        before_concat.append(
            layers.conv(
                in_channels,
                num_channels_down,
                filter_down_size,
                stride=stride,
                bias=bias,
                pad=pad,
                downsample_mode=downsampler,
            ))

        before_concat.append(nn.BatchNorm2d(num_channels_down))
        before_concat.append(layers.act(act_fun))
        before_concat.append(
            layers.NONLocalBlock2D(in_channels=num_channels_down))
        before_concat.append(
            layers.conv(
                num_channels_down,
                num_channels_down,
                filter_down_size,
                bias=bias,
                pad=pad,
            ))
        before_concat.append(nn.BatchNorm2d(num_channels_down))
        before_concat.append(layers.act(act_fun))
        return before_concat

    def construct(self):
        # self.model = nn.Sequential()
        # construct a model from input
        depth = len(self.num_channels_down)
        # useful for debug later
        assert len(self.num_channels_up) == len(self.num_channels_down)
        assert len(self.num_channels_down == self.num_channels_skip)
        assert len(self.num_channels_down > 0)
        ### build a unet like module
        encoder = []
        skip = []
        decoder = [None] * depth
        ### create module
        for i in range(depth):
            if (i == 0):
                in_channels = self.num_input_channels
            else:
                in_channels = self.num_channels_down[i - 1]
            encoder[i] = self.deeper_before_concat(in_channels,
                                                   self.num_channels_down[i],
                                                   self.filter_down_size[i],
                                                   self.stride, self.pad,
                                                   self.downsampler, self.bias,
                                                   self.act_fun)
            skip[i] = self.single_skip(in_channels, self.num_channels_skip[i],
                                       self.filter_skip_size[i], self.pad,
                                       self.bias)
            # decoder[i] = nn.Sequential(torch.cat(skip[5-i],))
        return encoder, skip

    def forward(self, input):
        return None

    
demo =  skip(num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad="zero",
        upsample_mode="nearest",
        downsample_mode="stride",
        act_fun="LeakyReLU",
        need1x1_up=True)

e,s = demo.construct
print(len(e))