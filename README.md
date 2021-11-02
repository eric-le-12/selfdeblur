# Selfdeblur
This repo containing for reconstructing selfdeblur model


# Usage
Kindly run "run.py" for debluring on GrayScale images (Lai dataset)


For RGB images, run "run_color.py" (Lai dataset)


For alternating optimization strategy on Grayscale image, please run "run-alternating_op.py".


All of these 3 files shared the same parameters:

"--path_to_blur", type=str, describing path to folder containing blur images'

"--path_to_save", type=str, describing path to folder for saving'

"--epoch", type=int - number of epochs for deblurring EACH file.

Example: python run.py --epoch 5000 --path_to_blur ../dataset/levin/blur --path_to_save ../dataset/levin/result

### note: you have to create the folder you mentioned in --path_to_save, if not, results wont be saved 


## model reconstruction:


#### including the unet-like architecture for generating Gx prior 
##### implemented in network/skip.py file
This model contains 3 submodule: The encoder module which downsampling the spatial dimension, the skip connection which maintains spatial dimension but reducing number of channels, the post processing modules which including layers to pre-process feature map before upsampling.

#### including the neural network for capturing Gk prior
##### implemented in network/fcn.py

#### other functions, i.e. generate noise, total variation loss please found in utils/utils.py

# Evaluation

For evaluation, we used the same code of the author to allow consistency. The Matlab code for evaluation was written by the authors, source : https://github.com/csdwren/SelfDeblur

For evaluation on Lai and Levin dataset please run the corresponding Matlab file in statistics folder 


For levin dataset: remember to edit path to groundtruth files on line 5, path to deblured results on line 13, and path to other works on line 4

For lai dataset: edit path to ground-truth 4,5,21 to update path to other works, path to gt and path to results

## Useful link


Author repo : https://github.com/csdwren/SelfDeblur


# What is different? (Between our implementation and author's implementation)

- Loss function : We provided TV loss regularization The author did not implement Total variation loss

- Modularity: The architecture of the Encoder-Decoder model of our implementation is splited into 3 modules for every layer at depth i: skip block, encoder block and the post-processing block. While the author simple implemeted them in a very complicated loop - what we found uncomfortable

- We provided the code for alternating optimization strategy (the author's repo did not)

- We provided code for using bayesian instead of MAP (for more details, visit our report)