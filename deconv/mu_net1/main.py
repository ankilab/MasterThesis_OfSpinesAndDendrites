# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:09:56 2018
Mu-net: Multi-scale U-net for Two-photon Microscopy Image Denoising and Restoration
@author: Sehyung Lee, Makiko Negishi, Hidetoshi Urakubo, Haruo Kasai, and Shin Ishii
The demo code in this package includes the proposed 2PM image denoising method. 
For the review, we provide some example images used in the experiments that can be found in sub-folders
When testing/using for other data, please see input image's type such as uint8/uint16/float. 
Default setting is as max_value = 5000 and uint16 image. 
After publication, all source codes and pretrained model will be made available to public. 
We used Python 3.6.3 and Tensorflow 1.14.0
"""

from denoiser import *
import tifffile as tiff

denoiser = Denoiser()
small_or_large = 1

if small_or_large == 0:
    # this is example code for vanilla processing.
    batch_sz = 1
    denoiser.load_model(batch_sz)
    
    # read image
    # dir_path = './example_data/small/'
    # noise_level = 1 # from 1 to 4
    # img_name = '%s/n%d_000001.tif' % (dir_path,noise_level)
    img_name = '../Data/Raw/001.tif'
    img = tiff.imread(img_name)
    
    # denoising process
    L0_pred, L1_pred, L2_pred, denoised_img = denoiser.denoising_patch(img)
    tiff.imsave('L0_pred.tif', L0_pred.astype('uint16') )    
    tiff.imsave('L1_pred.tif', L1_pred.astype('uint16') )    
    tiff.imsave('L2_pred.tif', L2_pred.astype('uint16') )
    tiff.imsave('denoised_img.tif', denoised_img.astype('uint16') )
    
else:
    # this is example code for processing large images based on sliding window.
    batch_sz = 4
    denoiser.load_model(batch_sz)
    
    # read image
    # dir_path = './example_data/large/'
    # img_name = '%s/n1_000001.tif' % (dir_path)
    img_name = '../Data/Raw/001.tif'

    img = tiff.imread(img_name)

    
    # denoising process
    denoised_img = denoiser.denoising_img(img)
    tiff.imsave('denoised_img.tif', denoised_img.astype('uint16') )    
    