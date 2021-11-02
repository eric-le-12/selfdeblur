import torch
import torch.nn as nn
import numpy as np

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """
    need paraphrase
    
    """
    # size to pad
    pad_size = int((kernel_size - 1) / 2)
    padder = nn.ReflectionPad2d(pad_size)
    # create a convol layer
    convol = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)
    
    return nn.Sequential(padder,convol)


class NonLocalBlock(nn.Module):
    """
    a block contains convolution and batchnorm layer AND max_pooling
    """
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(in_channels // 2,1)
        conv = nn.Conv2d
        maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        # declar the conv blocks
        self.conv1 = nn.Sequential(conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),maxpool)

    
        self.conv3 = nn.Sequential(
                conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
        
        ## init this layer
        nn.init.constant_(self.conv3[1].weight, 0)
        nn.init.constant_(self.conv3[1].bias, 0)
  

        self.conv2_1 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.conv2_2 = nn.Sequential(conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),maxpool)

    def forward(self, input):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        ## get batch size
        batch_size = input.size(0)
        ### pass through first conv
        conv1 = self.conv1(input)
        conv1 = conv1.view(batch_size, self.inter_channels, -1)
        ### make channel last dim
        conv1 = conv1.permute(0, 2, 1)

        conv2_1 = self.conv2_1(input).view(batch_size, self.inter_channels, -1)
        conv2_1 = conv2_1.permute(0, 2, 1)
        conv2_2 = self.conv2_2(input).view(batch_size, self.inter_channels, -1)
        ## multiplication of conv2_1 and conv2_2
        f = torch.matmul(conv2_1, conv2_2)
        N = f.size(-1)
        # norm by number of channels
        f_normed = f / N

        y = torch.matmul(f_normed, conv1)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *input.size()[2:])
        out = self.conv3(y)
        output = out + input

        return output