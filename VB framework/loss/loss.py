import torch
from torchvision.transforms import Resize
import numpy as np


def gaussian_kl_loss(mu1, log_var1, mu2, epsilon):
    batch_size, channel, image_size, _ = mu1.size()
    var_ratio = torch.exp(log_var1) ** 2 / epsilon ** 2
    temp = 0.5 * (torch.pow(mu1 - mu2, 2) / epsilon ** 2 + var_ratio - 2 * torch.log2(var_ratio) - 1)
    return temp.sum() / batch_size


# print(gaussian_kl_loss(mu, log_var, latent_img))

def dirichlet_kl_loss(simplex, blur_kernel, scaling_term=2e4):
    batch_size, kernel_size, _ = simplex.size()
    blur_kernel = blur_kernel.squeeze()
    blur_kernel = blur_kernel + 1e-20
    blur_kernel = blur_kernel/torch.sum(blur_kernel, dim=[1, 2]).view(-1,1,1)
    blur_kernel = blur_kernel * scaling_term

    sum_simplex = torch.sum(simplex, dim=[1, 2])
    sum_blur = torch.sum(blur_kernel, dim=[1, 2])

    lgamma_sum_simplex = torch.lgamma(sum_simplex).sum()
    lgamma_sum_blur = torch.lgamma(sum_blur).sum()
    lgamma_simplex = torch.lgamma(simplex).sum()
    lgamma_blur = torch.lgamma(blur_kernel).sum()

    diff_digamma = torch.digamma(simplex) - torch.digamma(sum_simplex).view(-1, 1, 1)
    diff_term = simplex - blur_kernel
    diff = (diff_term * diff_digamma).sum()

    return (lgamma_sum_simplex - lgamma_sum_blur - lgamma_simplex + lgamma_blur + diff) / batch_size


# print(diriclet_kl_loss(simplex, blur_kernel))


def filter_gauss(latent_img, blur_kernel):
    c, h, w = latent_img.size()
    k, k = blur_kernel.size()
    blur_kernel = blur_kernel.view(1, 1, k, k)
    channels = []
    for i in range(c):
        blur_channel = latent_img[i].view(1, 1, h, w)
        blur_result = torch.nn.functional.conv2d(blur_channel, blur_kernel, stride=[1, 1])
        channels.append(blur_result)
    return torch.vstack(channels).squeeze()


def gaussian_loss(blur_img, blur_kernel_gen, latent_img_gen, sigma=1e-2):
    loss = torch.tensor(0.0, device="cuda")
    batch_size, channel, image_size, _ = latent_img_gen.size()
    constant = (image_size ** 2) * torch.log2(sigma / torch.sqrt(2 * torch.tensor(np.pi, device="cuda")))
    for i in range(batch_size):
        blur_img_gen = filter_gauss(latent_img_gen[i], blur_kernel_gen[i])
        blur_img_gen = Resize(image_size)(blur_img_gen)
        loss = loss + constant - 0.5 * ((blur_img - blur_img_gen) ** 2 / sigma ** 2).sum()
    return loss / batch_size


# print(gaussian_loss(blur_img ,blur_kernel_gen, latent_img_gen, image_size=256, sigma = 1e-2))

def loss_vb(blur_kernel,
            blur_img,
            latent_img,
            mu,
            log_var,
            simplex,
            blur_kernel_gen,
            latent_img_gen,
            sigma,
            scaling_term,
            epsilon):
    kl_gaussian = gaussian_kl_loss(mu, log_var, latent_img, epsilon)
    kl_dirichlet = dirichlet_kl_loss(simplex, blur_kernel, scaling_term=scaling_term)
    gauss_loss = gaussian_loss(blur_img, blur_kernel_gen, latent_img_gen, sigma=sigma)
    total_loss = kl_gaussian + kl_dirichlet - gauss_loss
    return total_loss / 1e9

# print(loss_vb(blur_kernel, blur_img, latent_img, mu, log_var, simplex, blur_kernel_gen, latent_img_gen))