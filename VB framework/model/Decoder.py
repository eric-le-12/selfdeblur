import torch
import torchvision
import torch.nn as nn
from model.ResBlock import ResBlock
from model.SFTlayer import SFT_Layer


class DecoderBlock(nn.Module):
    def __init__(self,
                 n_res_block,
                 channel,
                 blur_channel):
        super(DecoderBlock, self).__init__()
        self.reduce_channel_conv = nn.Conv2d(channel * 2, channel, kernel_size=1)
        self.res_blocks = nn.ModuleList([ResBlock(channel) for i in range(n_res_block)])
        self.sft_layer = SFT_Layer(channel, blur_channel)

    def forward(self, img, blur_kernel):
        x = self.reduce_channel_conv(img)
        for block in self.res_blocks:
            x = block(x)
        x = self.sft_layer(x, blur_kernel)
        return x


# decoder_block = DecoderBlock().to("cuda")
# summary(decoder_block, [(128,8,8), (10,8,8)])

class Decoder(nn.Module):
    def __init__(self,
                 n_res_block,
                 blur_channel,
                 channel,
                 n_scales=4):
        super(Decoder, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=2)
        self.dec_blocks = nn.ModuleList([DecoderBlock(n_res_block, channel, blur_channel) for i in range(n_scales)])

    def forward(self, encoder_blocks, blur_blocks):
        x = encoder_blocks.pop()
        y = blur_blocks.pop()
        for block in self.dec_blocks:
            x = self.upsample_layer(x)
            z = encoder_blocks.pop()
            y = blur_blocks.pop()
            x = self.crop(x, z)
            x = torch.cat([x, z], dim=1)
            x = block(x, y)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

# print("-------Test decoder--------")
# decoder = Decoder().to("cuda")
# dec = decoder(enc_list, blur_list)
# print(dec.size())
# print("---------------------------")
