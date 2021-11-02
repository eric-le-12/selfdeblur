from model.Encoder import Encoder
from model.Decoder import Decoder
import torch.nn as nn


class DeblurNet(nn.Module):
    def __init__(self,
                 n_res_block,
                 channel,
                 blur_channel):
        super(DeblurNet, self).__init__()
        self.encoder = Encoder(n_res_block, blur_channel, channel)
        self.decoder = Decoder(n_res_block, blur_channel, channel)
        self.conv_in = nn.Conv2d(3, channel, kernel_size=7, padding=3)
        self.conv_mean_out = nn.Conv2d(channel, 3, kernel_size=7, padding=3)
        self.conv_var_out = nn.Conv2d(channel, 3, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
#         self.batchnorm = nn.BatchNorm2d(3)

    def forward(self, img, blur_kernel):
        x = self.conv_in(img)
        enc_list, blur_list = self.encoder(x, blur_kernel)
        x = self.decoder(enc_list, blur_list)

        mean = self.conv_mean_out(x)
        mean = self.sigmoid(mean)

        var = self.conv_var_out(x)
        var = self.sigmoid(var)
        return mean, var

# print("------Test DeblurNet-------")
# deblur = DeblurNet().to("cuda")
# mu, var = deblur(torch.rand(2,3,256,256,device="cuda"), torch.rand(2,10,256,256,device="cuda"))
# print(mu.size())
# print(var.size())