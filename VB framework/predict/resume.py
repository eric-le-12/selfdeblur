import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def resume(name, model, optimizer):
    checkpoint_path = './checkpoints/{}.pth'.format(name)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer


# def predict(model, data):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     data = data.to(device)
#     mu, log_var, simplex, blur_kernel, latent_img = model(data)
#     return latent_img, blur_kernel

def swap_axis(img):
    show_img = np.transpose(img.cpu().detach().numpy(), (1,2,0))
    return show_img

def test(model, text, image_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idxes = np.random.randint(0, 1000, 10)
    for idx in idxes:
        blur_kernel, blur_img, clean_img = text[idx]
        plt.imsave("/notebooks/test/text/{}_blur.jpg".format(idx), swap_axis(blur_img), dpi=300)
        plt.imsave("/notebooks/test/text/{}_latent.jpg".format(idx), swap_axis(clean_img), dpi=300)
#         plt.imsave("/notebooks/test/text/{}_kernel.png".format(idx), swap_axis(blur_kernel), cmap=cm.gray)
        mu, log_var, simplex, blur_kernel_gen, latent_img_gen = model(blur_img.view(1,3,image_size,image_size).to(device))
#         plt.imsave("/notebooks/test/text/{}_kernel_gen.png".format(idx), swap_axis(blur_kernel_gen), cmap=cm.gray)
        plt.imsave("/notebooks/test/text/{}_laten_gen.jpg".format(idx), swap_axis(latent_img_gen.squeeze()), dpi=300)
