from tqdm import tqdm
from torch.nn import functional as F
from utils import utils as u
import time
import timeit
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR

def train_one_image(model_x, model_k, kernel_size, epochs, optimizer,scheduler, criterion,
                    target, device, data_x, data_k):
    # training-the-model
    train_loss = 0
    print('*****************************')
    crit = nn.MSELoss()
    for _,epoch in tqdm(enumerate(range(epochs))):
        # varying input by adding a random noise vector every iteration
        data_x_new = data_x.detach().clone() + 0.001*torch.zeros(data_x.shape).type_as(data_x.data).normal_()
        data_x_new = data_x_new.to(device)
        # clear-the-gradients-of-all-optimized-variables
        scheduler.step(epoch)
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
        loss =  (max(0,(epoch-1000)))*criterion(deblured, target,device) + crit(deblured.to(device), target.to(device))
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item()
    
    return (
        train_loss,output_x
    )

