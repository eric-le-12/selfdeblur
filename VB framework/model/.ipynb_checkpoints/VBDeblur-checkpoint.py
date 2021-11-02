import torch
import torch.nn as nn
from model.DeblurNet import DeblurNet
from model.KernelNet import KernelNet
from torchvision.transforms import Resize


class VB_Blur(nn.Module):
    def __init__(self,
                 img_size,
                 kernel_size,
                 n_res_block,
                 channel,
                 blur_channel):
        super(VB_Blur, self).__init__()
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.deblur_net = DeblurNet(n_res_block, channel, blur_channel)
        self.kernel_net = KernelNet(self.kernel_size)
        self.t_conv_layer = nn.ConvTranspose2d(1, blur_channel, kernel_size=3, stride=3)

    def forward(self, blur_img):
        batch_size, _, _, _ = blur_img.size()
        # kernel net predict simplex
        simplex = self.kernel_net(blur_img)
        # make sure simplex is not 0
        simplex = simplex + 1e-20
        simplex = simplex/torch.sum(simplex, dim=[1, 2]).view(-1,1,1)
        # sample from dirichlet distribution blur kernel
        blur_dist = torch.distributions.dirichlet.Dirichlet(simplex.view(batch_size, -1))
        blur_kernel = blur_dist.sample().view(batch_size, self.kernel_size, self.kernel_size)
        # transpose to same as image size
        blur_kernel_norm = self.t_conv_layer(blur_kernel.view(-1, 1, self.kernel_size, self.kernel_size))
        blur_kernel_norm = Resize(self.img_size)(blur_kernel_norm)
        # deblur net predict mu and log_var
        mu, log_var = self.deblur_net(blur_img, blur_kernel_norm)
        # sample from gaussian distribution latent image
        img_dist = torch.distributions.normal.Normal(mu * 255, torch.exp(log_var))
        latent_img = img_dist.rsample()
        # convert to range 0:255
        latent_img = torch.clip(latent_img, 0, 255).to(torch.uint8) / 255
        return mu, log_var, simplex, blur_kernel, latent_img
