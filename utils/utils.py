"""
Contain neccessary information
"""
import numpy as np
import torch
import torch.nn.functional as F

## calculate ssim loss


def ssim_loss(x, y, gaussian_window, window_size=11):
    """
    Return the ssim loss given 2 images x and y
    """
    ### define c1 and c2 constant, to help stablize the denominator
    K_1 = 0.001
    K_2 = 0.003
    L = 255  # THE PIXEL RANGE
    const1 = (K_1 * L)**2
    const2 = (K_2 * L)**2
    ### calculate components of ssim loss
    _, img_channel, _, _ = x.shape
    pad = window_size // 2
    ### since we are dueling with 2d image let's get 2d gaussian
    temp = np.random.normal(pad, 1.5, window_size)
    temp = temp / temp.sum()
    temp = temp * temp.T
    ### convert window to 4d tensor
    window = np.expand_dims(temp, 0)
    window = np.expand_dims(window, 0)
    window = torch.Tensor(window)
    mu_x = F.conv2d(x, window, padding=pad, groups=img_channel)
    mu_y = F.conv2d(y, window, padding=pad, groups=img_channel)
    sigma_x_sq = F.conv2d(x * x, window, padding=pad,
                          groups=img_channel) - mu_x * mu_x
    sigma_y_sq = F.conv2d(y * y, window, padding=pad,
                          groups=img_channel) - mu_y * mu_y
    sigma_x_y_sq = F.conv2d(x * y, window, padding=pad,
                            groups=img_channel) - mu_x * mu_y
    ### plug them in and calculate ssim :
    numerator = (2 * mu_x * mu_y + const1) * (2 * sigma_x_y_sq + const2)
    denominator = (mu_x * mu_x + mu_y * mu_y + const1)(sigma_x_sq +
                                                       sigma_x_y_sq + const2)
    ssim = (numerator / denominator).mean()
    return 1 - ssim


def generate_input(shape, scale, type):
    """Generate input tensor of size shape, which is then multiplied by a scale_factor
    type can be either normal or uniform
    """
    if type == "uniform":
        return torch.randn(shape).uniform_() * scale
    else:
        # normal
        return torch.randn(shape) * scale
