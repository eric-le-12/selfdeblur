"""
This file generate the encoder-decoder with skip connection structure
@ Hieu Le Xuan, 2021
"""
from utils import layers
from utils import non_local_dot_product as block
import torch.nn as nn
import torch
from utils.act import act_func as act
import numpy as np


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2:diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)


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
        to_pad = int((filter_skip_size - 1) / 2)
        ## choosing path method : zero or flection
        if pad == "reflection":
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
        skip.add_module("Leaky_RELU", act(act_fun="LeakyReLU"))

        return skip

    def deeper_before_concat(self, in_channels, num_channels_down,
                             filter_down_size, stride, pad, downsampler, bias,
                             act_fun, non_local):
        """
        con_block -> bn -> leaky relu -> nonlocal2D block ->...
        purpose : reduce spatial dimension while increasing depth / num of channels

        """
        before_concat = []
        ## append conv : first block stride = 2 to reduce spatial dimensions
        before_concat.append(
            layers.conv(
                in_channels,
                num_channels_down,
                filter_down_size,
                stride=2,
                bias=bias,
                pad=pad,
                downsample_mode=downsampler,
            ))

        before_concat.append(nn.BatchNorm2d(num_channels_down))
        before_concat.append(layers.act(act_fun))
        if (non_local):
            before_concat.append(
                block.NONLocalBlock2D(in_channels=num_channels_down))

        before_concat.append(
            layers.conv(
                num_channels_down,
                num_channels_down,
                filter_down_size,
                bias=bias,
                pad=pad,
            ))
        before_concat.append(nn.BatchNorm2d(num_channels_down))
        before_concat.append(act(act_fun))
        return nn.Sequential(*before_concat)

    def construct(self):
        # self.model = nn.Sequential()
        # construct a model from input
        depth = len(self.num_channels_down)
        # useful for debug later
        assert len(self.num_channels_up) == len(self.num_channels_down)
        assert len(self.num_channels_down) == len(self.num_channels_skip)
        assert len(self.num_channels_down) > 0
        ### build a unet like module
        encoder = [None] * depth
        skip = [None] * depth
        post = [None] * depth
        decoder = [None] * depth
        non_local = False
        ### create module
        for i in range(depth):
            if (i == 0):
                in_channels = self.num_input_channels
                non_local = False
            else:
                in_channels = self.num_channels_down[i - 1]
                if (i > 1):
                    non_local = True
            encoder[i] = self.deeper_before_concat(
                in_channels, self.num_channels_down[i], self.filter_size_down,
                1, self.pad, self.downsample_mode, self.need_bias,
                self.act_fun, non_local)
            skip[i] = self.single_skip(in_channels, self.num_channels_skip[i],
                                       self.filter_skip_size, self.pad,
                                       self.need_bias)
            post[i] = self.post_processing(
                self.num_channels_down[i] + self.num_channels_skip[i],
                self.num_channels_up[i], self.filter_size_up)
            # decoder[i] = nn.Sequential(torch.cat(skip[5-i],))

        return encoder, skip, post

    def post_processing(self, input_channel, output_channel, kernel_size):
        """
        return post_processing layers
        input channel : num of channels for coming input
        output channel: desired number of channels
        """
        bn = []
        bn.append(nn.BatchNorm2d(input_channel))
        bn.append(
            layers.conv(input_channel,
                        output_channel,
                        kernel_size,
                        bias=self.need_bias,
                        pad=self.pad))
        bn.append(nn.BatchNorm2d(output_channel))
        bn.append(act(self.act_fun))
        bn.append(
            layers.conv(output_channel,
                        output_channel,
                        1,
                        bias=self.need_bias,
                        pad=self.pad))
        bn.append(nn.BatchNorm2d(output_channel))
        bn.append(act(self.act_fun))
        return bn

    def model_tail(self):
        """
        construct the tail of model
        """
        tail = []
        input_channel = self.num_channels_up[0]
        output_channel = self.num_output_channels
        kernel_size = 1

        tail.append(
            layers.conv(input_channel,
                        output_channel,
                        kernel_size,
                        bias=self.need_bias,
                        pad=self.pad))
        tail.append(nn.Sigmoid())
        return tail

    def forward(self, input):
        return None
