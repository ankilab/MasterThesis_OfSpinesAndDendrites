# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:27:53 2018

@author: sehyung
"""
import numpy as np
import pathlib
from data_augmentation import DataAugmenter


def read_patches(batch_sz, size, data_dir):
    # implement image batch read for your dataset
    (sz_z, sz_xy) = size
    args = {}
    args['z_shape'] = sz_z
    args['xy_shape'] = sz_xy
    args['n_patches'] = 50
    da = DataAugmenter(args)
    (augmented_raw, augmented_gt, _), _= da.augment(data_dir, 'Raw', 'GT', care=True)

    return augmented_raw[:,0,:, :,:], augmented_gt[:,0,:, :,:]

def get_resized_patches(patch, max_value):
    patch = patch.astype('float32')
    sc = max_value/2.0
    return patch.astype('float32')/sc-1.0


def get_batch(patch, tensor, x,y,z, x_loc, y_loc, z_loc, count):
    patch_sz = patch.shape
    patch = np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1] )
    tensor[count, 0:patch_sz[0], 0:patch_sz[1], 0:patch_sz[2], 0:1] = patch
    x_loc[count] = x
    y_loc[count] = y
    z_loc[count] = z
    return patch, tensor, x_loc, y_loc, z_loc


def window_sliding(self, img, sampling_step, max_value, patch_sz, batch_sz):
    img_sz = img.shape
    sc = max_value/2.0
    cnn_input_sz = [batch_sz, patch_sz, patch_sz, patch_sz, 1]
    input_tensor = np.zeros(cnn_input_sz, 'float32')
    
    wz = [patch_sz, patch_sz, patch_sz]
    wz = np.int_(wz)    
    
    x_loc = np.zeros(batch_sz, 'int32')
    y_loc = np.zeros(batch_sz, 'int32')
    z_loc = np.zeros(batch_sz, 'int32')
    count = 0
    
    img = img.astype('float32')    
    recon_img = np.zeros(img_sz)
    recon_img = recon_img.astype('float32')
    occupancy = np.zeros(img_sz)
    occupancy = occupancy.astype('float32')

    x_direction = (0,1,0,1,0,1,0,1)
    y_direction = (0,0,1,1,0,0,1,1)
    z_direction = (0,0,0,0,1,1,1,1)
    
    
    for i in range(0, 8):
        print('%g%%'%(i/8*100) )
        x_dir = x_direction[i]
        y_dir = y_direction[i] 
        z_dir = z_direction[i]
        
        if z_dir == 0:
            for z in range(0, img_sz[0]-wz[0], sampling_step[0]):
                if y_dir == 0:
                    for y in range(0, img_sz[1]-wz[1], sampling_step[1]):
                        if x_dir == 0:
                            for x in range(0, img_sz[2]-wz[2], sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                                        
                        elif x_dir == 1:
                            for x in range(img_sz[2]-wz[2], 0, -sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                elif y_dir == 1:
                    for y in range(img_sz[1]-wz[1], 0, -sampling_step[1]): 
                        if x_dir == 0:
                            for x in range(0, img_sz[2]-wz[2], sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                        elif x_dir == 1:
                            for x in range(img_sz[2]-wz[2], 0, -sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
        elif z_dir == 1:
            for z in range(img_sz[0]-wz[0], 0, -sampling_step[0]):
                if y_dir == 0:
                    for y in range(0, img_sz[1]-wz[1], sampling_step[1]):
                        if x_dir == 0:
                            for x in range(0, img_sz[2]-wz[2], sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                        elif x_dir == 1:
                            for x in range(img_sz[2]-wz[2], 0, -sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                elif y_dir == 1:
                    for y in range(img_sz[1]-wz[1], 0, -sampling_step[1]): 
                        if x_dir == 0:
                            for x in range(0, img_sz[2]-wz[2], sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                        elif x_dir == 1:
                            for x in range(img_sz[2]-wz[2], 0, -sampling_step[2]):
                                patch = img[ z:z+wz[0], y:y+wz[1], x:x+wz[2] ]
                                patch = get_resized_patches(patch, max_value)
                                input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor, x, y, z, x_loc, y_loc, z_loc, count)
                                count = count + 1
                                if count % batch_sz == 0 :
                                    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
                                    pred_patch = np.clip(pred_patch, -1, 1)
                                    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
                                    for k in range(0, count):
                                        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
                                        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
                                        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
                                        count = 0
                                        
                                        
    pred_patch = self.sess.run(self.L3_pred, feed_dict = {self.img: input_tensor} )
    pred_patch = np.clip(pred_patch, -1, 1)
    pred_patch = np.reshape((pred_patch+1)*sc, cnn_input_sz[0:4])
    for k in range(0, count):
        prev_patch = recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ]
        recon_img[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = pred_patch[k] + prev_patch
        occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] = occupancy[ z_loc[k]:z_loc[k]+wz[0], y_loc[k]:y_loc[k]+wz[1], x_loc[k]:x_loc[k]+wz[2] ] + 1
        count = 0
    
    recon_img = np.divide(recon_img , occupancy)
    print('done')
    return recon_img