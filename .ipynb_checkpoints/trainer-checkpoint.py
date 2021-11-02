from tqdm import tqdm
from torch.nn import functional as F
from utils import utils as u
import time
import timeit
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils import utils
import numpy as np

def max_range(img):
    print (np.amax(img.reshape((img.shape[2]*img.shape[3], 1)),axis=0))
    
def train_one_image(model_x, model_k, kernel_size, epochs, optimizer,scheduler, criterion,
                    target, device, data_x, data_k,padh,padw,original_size,name_to_save):
    # training-the-model
    train_loss = 0
#     print('*************WITHout TV****************')
    crit = nn.MSELoss()
    data_loss = []
    for _,epoch in tqdm(enumerate(range(epochs))):
        # varying input by adding a random noise vector every iteration
        data_x_new = data_x.detach().clone() + 0.001*torch.zeros(data_x.shape).type_as(data_x.data).normal_()
        data_x_new = data_x_new.to(device)
        # clear-the-gradients-of-all-optimized-variables
#         scheduler.step(epoch)
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output_x = model_x(data_x_new)
        output_k= model_k(data_k)
        # reshape output_k into 4d tensor
        kernel_w = kernel_size[0]
        kernel_h = kernel_size[1]
        output_k = output_k.view(-1,1,kernel_w,kernel_h)
        # re-constructed deblurred image        
        deblured = F.conv2d(output_x, output_k, padding=0, bias=None)
        # calculate-the-batch-loss
        switch_loss_epoch = 5000
        slope = np.clip(np.array([epoch-switch_loss_epoch]),0,1)[0]
        slope_mse = np.clip(np.array([-epoch+switch_loss_epoch]),0,1)[0]
        loss =  slope*criterion(deblured, target,device) + slope_mse*crit(deblured.to(device), target.to(device)) 
#         + 0.000001*utils.tv_loss(output_x)
        #+ 0.000001*utils.tv_loss(output_x)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item()
    
        if (epoch==20 or epoch==30 or epoch==50 or epoch==100 or epoch%500==0 or epoch==epochs-1):
            from skimage.io import imsave
            output_x_save = output_x.detach().cpu().numpy()
            output_x_save = np.squeeze(output_x_save,0)
            output_x_save = np.moveaxis(output_x_save,0,2)
            output_x_save = output_x_save[padh//2:((padh//2)+original_size[1]), padw//2:((padw//2)+original_size[2])]
            imsave(name_to_save+'_'+str(epoch)+'_deblured.png',output_x_save)
            output_k_save = output_k.squeeze_()
            output_k_save = 255*output_k_save.detach().cpu().numpy()
            output_k_save /= np.max(output_k_save)
            imsave(name_to_save+'_'+str(epoch)+'_kernel.png',output_k_save)
            data_loss.append(loss.item())
    
    import pandas as pd
    dat = pd.DataFrame({'loss_val':data_loss})
    dat.to_csv(name_to_save+'_dat.csv')
    return (
        train_loss,output_x,output_k
    )

