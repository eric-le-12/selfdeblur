import torch
import torch.nn as nn


class SFT_Layer(nn.Module):
    def __init__(self, channel, blur_channel):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(blur_channel + channel, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(blur_channel + channel, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add
