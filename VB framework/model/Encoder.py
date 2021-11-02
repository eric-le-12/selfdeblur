import torch.nn as nn
from model.SFTlayer import SFT_Layer
from model.ResBlock import ResBlock


# custom U-NET taken from https://amaarora.github.io/2020/09/13/unet.html#u-net

class EncoderBlock(nn.Module):
    def __init__(self,
                 n_res_block,
                 channel,
                 blur_channel):
        super(EncoderBlock, self).__init__()
        self.res_blocks = nn.ModuleList([ResBlock(channel) for i in range(n_res_block)])
        self.sft_layer = SFT_Layer(channel, blur_channel)

    def forward(self, img, blur_kernel):
        x = img
        for block in self.res_blocks:
            x = block(x)
        x = self.sft_layer(x, blur_kernel)
        return x


# encoder_block = EncoderBlock().to("cuda")
# summary(encoder_block, [(64,8,8), (10,8,8)])

class Encoder(nn.Module):
    def __init__(self,
                 n_res_block,
                 blur_channel,
                 channel,
                 n_scales=5):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList([EncoderBlock(n_res_block,
                                                      channel,
                                                      blur_channel) for i in range(n_scales)])
        self.downsample_layer = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.downsample_kernel_layer = nn.Conv2d(blur_channel, blur_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, img, blur_kernel):
        x = img
        y = blur_kernel
        encoder_blocks = []
        blur_blocks = []
        for block in self.enc_blocks:
            x = block(x, y)
            encoder_blocks.append(x)
            blur_blocks.append(y)
            x = self.downsample_layer(x)
            y = self.downsample_kernel_layer(y)
        return encoder_blocks, blur_blocks

# print("-------Test encoder--------")
# encoder = Encoder().to("cuda")
# enc_list, blur_list = encoder(torch.rand(1,64,64,64, device="cuda"), torch.rand(1,10,64,64, device="cuda"))
# for enc in enc_list:
#     print(enc.size())
# print("---blur---")
# for blur in blur_list:
#     print(blur.size())
# print("---------------------------")
