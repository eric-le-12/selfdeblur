
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from PIL import Image
import warnings
from utils import utils as my_utils
from network import fcn as my_fcn
from trainer import train_one_image
from network import skip as sk


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

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

def my_unet():
    demo =  sk.Skip(num_input_channels=8,
            num_output_channels=1,
            num_channels_down = [128, 128, 128, 128, 128],
            num_channels_up   = [128, 128, 128, 128, 128],
            num_channels_skip = [16, 16, 16, 16, 16],
            filter_size_down=3,
            filter_size_up=3,
            filter_skip_size=1,
            need_sigmoid=True,
            need_bias=True,
            pad="reflection",
            upsample_mode="nearest",
            downsample_mode="stride",
            act_fun="LeakyReLU",
            need1x1_up=True)
    device = torch.device('cuda:0')
    demo = demo.to(device)
    e,s,p = demo.construct()
    e[4].add_module("upsampling_2",nn.Upsample(scale_factor=2, mode="nearest"))

    deeper = nn.Sequential(Concat(1,s[4],e[4]))
    ## add post-process layers
    for i in p[4]:
        deeper.add(i)
    e[3].add_module("deeper_3",deeper)
    ## add upsampling
    e[3].add_module("upsampling_2",nn.Upsample(scale_factor=2, mode="nearest"))

    model = None
    deeper = nn.Sequential(Concat(1,s[3],e[3]))
    ## add post-process layers
    for i in p[3]:
        deeper.add(i)
    e[2].add_module("deeper_3",deeper)
    e[2].add_module("upsampling_2",nn.Upsample(scale_factor=2, mode="nearest"))

    deeper = nn.Sequential(Concat(1,s[2],e[2]))
    ## add post-process layers
    for i in p[2]:
        deeper.add(i)
    e[1].add_module("deeper_3",deeper)
    e[1].add_module("upsampling_2",nn.Upsample(scale_factor=2, mode="nearest"))

    deeper = nn.Sequential(Concat(1,s[1],e[1]))
    ## add post-process layers
    for i in p[1]:
        deeper.add(i)
    e[0].add_module("deeper_2",deeper)
    e[0].add_module("upsampling_1",nn.Upsample(scale_factor=2, mode="nearest"))

    model = nn.Sequential(Concat(1,s[0],e[0]))
    ## add post-process layers
    for i in p[0]:
        model.add(i)

    for i in (demo.model_tail()):
        print(i)
        model.add(i)
    # ref.add_module("deeper_depth_2",deeper)
    return model

def preprocess(path_to_image):
    img = cv2.imread(path_to_image,cv2.IMREAD_UNCHANGED)
    channel = img.shape[-1]
    if (channel==1):
        print('image in grayscale mode')
        return Image.open(path_to_image)
    elif (channel==3):
        print("image in rgb mode")
        x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(x)
        return [y,cr,cb]
    else:
        return None
    
# beginning running...
### get a list of files for levin_data_set
def run(path_to_blur,path_to_save,epochs):
    PATH = path_to_blur
    list_imgs = os.listdir(PATH)
    list_imgs.sort()
    # start #image
    for f in list_imgs:
        if (f==".ipynb_checkpoints"):
            continue
        imgname = f
        new_img_name = imgname.split(".")[0] + "_x."+imgname.split(".")[1]
        
        name_to_save = os.path.join(path_to_save,new_img_name)
        print(name_to_save)
        print(epochs)
        if os.path.isfile(name_to_save):
            print("file exist... move to next")
            continue

        # set predefined kernels
        if imgname.find('kernel1') != -1:
            kernel_size = [17, 17]
        if imgname.find('kernel2') != -1:
            kernel_size = [15, 15]
        if imgname.find('kernel3') != -1:
            kernel_size = [13, 13]
        if imgname.find('kernel4') != -1:
            kernel_size = [27, 27]
        if imgname.find('kernel5') != -1:
            kernel_size = [11, 11]
        if imgname.find('kernel6') != -1:
            kernel_size = [19, 19]
        if imgname.find('kernel7') != -1:
            kernel_size = [21, 21]
        if imgname.find('kernel8') != -1:
            kernel_size = [21, 21]
        if imgname.find('kernel_01') != -1:
            kernel_size = [31, 31]
        if imgname.find('kernel_02') != -1:
            kernel_size = [51, 51]
        if imgname.find('kernel_03') != -1:
            kernel_size = [55, 55]
        if imgname.find('kernel_04') != -1:
            kernel_size = [75, 75]

        # set up device
        # read in every image
        device = torch.device("cuda:0")
        im = preprocess(os.path.join(PATH,f))
        if(isinstance(im, list)):
            im,cr,cb = im
        
        # convert from PIL to numpy
        pic = np.asarray(im)
        # convert from integers to floats
        pic = pic.astype('float32')
        # normalize to the range 0-1
        pic /= 255.0
        # target is the result of blur_kernel * latent_image
        im = torch.from_numpy(pic).unsqueeze(0)
        target = im.to(device)
        target = target.unsqueeze(0)
        img_size = list(im.shape)
        original_size = list(im.shape)
        print(imgname)
        ## estimate padding so that when convolution with a kernel of shape kernel_size
        ## it will result in an image of same initial shape
        padh = kernel_size[0]-1
        padw = kernel_size[1]-1
        ## padding image size calculation
        img_size[0] = img_size[1] + padh
        img_size[1] = img_size[2] + padw

        ## generate noise vectors and corresponding model
        ### generate the unet architecture
        model_x = my_unet()
        model_x = model_x.to(device)
        ### generate kernel neural network estimation
        model_k = my_fcn.kernel_approximation(200, kernel_size[0]*kernel_size[1]).create_model()
        model_k = model_k.to(device)
        ### generate a uniform vector of shape 1,8,img_width,img_height
        data_x = my_utils.generate_input((1,8, img_size[0], img_size[1])).to(device)
        ### generate vector of size 200 for kernel estimation
        data_k = my_utils.generate_input((200)).to(device)
        print("k shape: ",data_k.shape)

        # optimizer and scheduler
        optimizer = torch.optim.Adam([{'params':model_x.parameters()},{'params':model_k.parameters(),'lr':1e-4}], lr=0.01)
        scheduler = MultiStepLR(optimizer, milestones=[700, 800, 900], gamma=0.5)  # learning rates

        # Losses

        mse = torch.nn.MSELoss().to(device)
        criterion = my_utils.ssim_loss

        loss, output_x,output_k = train_one_image(model_x, model_k, kernel_size, epochs, optimizer,scheduler, criterion,
                        target, device, data_x, data_k)

        ## post-processing for saving image
        ### convert to numpy and delete first dimension
        output_x = output_x.detach().cpu().numpy()
        output_x = np.squeeze(output_x,0)
        output_x = np.moveaxis(output_x,0,2)
        output_x = output_x[padh//2:((padh//2)+original_size[1]), padw//2:((padw//2)+original_size[2])]
        if ('cr' in locals()):
            output_x = np.uint8(255 * output_x)
            output_x = cv2.merge([output_x, cr, cb])
            output_x = cv2.cvtColor(output_x, cv2.COLOR_YCrCb2RGB)
            
        imsave(name_to_save,output_x)
        torch.save(model_x.state_dict(),name_to_save+"_model_x.pt")
        torch.save(model_k.state_dict(),name_to_save+"_model_k.pt")
        torch.save(optimizer.state_dict(),name_to_save+"_optim.pt")
        output_k = output_k.squeeze_()
        output_k = output_k.detach().cpu().numpy()
        imsave(name_to_save+'_kernel.png',output_k)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_blur", type=str, default="../levin_set/blur", help='path to folder containing blur images')
    parser.add_argument("--path_to_save", type=str, default="../levin_set/result", help='path to folder for saving')
    parser.add_argument("--epoch", type=int, default=1000, help='number of epochs')
    args = parser.parse_args()
    epoch_folder = "NO_tv_mse_"+str(args.epoch)
    run(args.path_to_blur,os.path.join(args.path_to_save,epoch_folder),args.epoch)
    epoch_folder = "NO_tv_mse_"+str(args.epoch+1000)
    run(args.path_to_blur,os.path.join(args.path_to_save,epoch_folder),args.epoch+1000)
    epoch_folder = "NO_tv_mse_"+str(args.epoch+2000)
    run(args.path_to_blur,os.path.join(args.path_to_save,epoch_folder),args.epoch+2000)
