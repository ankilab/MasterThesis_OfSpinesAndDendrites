# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:09:56 2018

@author: sehyung
"""

import tensorflow as tf
from mu_net.cnn_models import *
from mu_net.utils import *
import random
import time
from data_augmentation import DataProvider

tf.compat.v1.disable_eager_execution()


class Denoiser():
    tf.compat.v1.reset_default_graph()

    def __init__(self, args):
        self.args = args
        # basic parameters
        self.batch_sz = 4
        self.sz = 128
        self.sz_z = 32
        self.max_value = 5000.0  # we set maximum value to 5,000 of microscope output

    def model_setup(self):

        self.img = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_sz, self.sz_z, self.sz, self.sz, 1], name = 'img')
        self.label = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_sz, self.sz_z, self.sz, self.sz, 1], name = 'label')
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        
        self.L2_img = tf.compat.v1.nn.conv3d(self.img, filter = tf.constant(1.0,shape=[1,1,1,1,1]), strides = [1,2,2,2,1] , padding = 'SAME')
        self.L1_img = tf.compat.v1.nn.conv3d(self.img, filter = tf.constant(1.0,shape=[1,1,1,1,1]), strides = [1,4,4,4,1] , padding = 'SAME')
        self.L0_img = tf.compat.v1.nn.conv3d(self.img, filter = tf.constant(1.0,shape=[1,1,1,1,1]), strides = [1,8,8,8,1] , padding = 'SAME')

        self.L2_label = tf.compat.v1.nn.conv3d(self.label, filter = tf.constant(1.0,shape=[1,1,1,1,1]), strides = [1,2,2,2,1] , padding = 'SAME')
        self.L1_label = tf.compat.v1.nn.conv3d(self.label, filter = tf.constant(1.0,shape=[1,1,1,1,1]), strides = [1,4,4,4,1] , padding = 'SAME')
        self.L0_label = tf.compat.v1.nn.conv3d(self.label, filter = tf.constant(1.0,shape=[1,1,1,1,1]), strides = [1,8,8,8,1] , padding = 'SAME')
        
        self.real_img = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_sz, self.sz_z, self.sz, self.sz, 1], name = 'real_img')
        self.fake_img = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_sz, self.sz_z, self.sz, self.sz, 1], name = 'fake_img')
        
        with tf.compat.v1.variable_scope("Model") as scope:
            self.L0_pred, self.L1_pred, self.L2_pred, self.L3_pred = munet_cnn(self.L0_img, self.L1_img, self.L2_img, self.img, name = 'gen')
            _, _, _, self.real_rec = discriminator(self.real_img, 'disc')
            
            scope.reuse_variables()
            _, _, _, self.fake_rec = discriminator(self.fake_img, 'disc')
            self.fake_f1, self.fake_f2, self.fake_f3, _ = discriminator(self.L3_pred, 'disc')
            self.real_f1, self.real_f2, self.real_f3, _ = discriminator(self.label, 'disc')
            
    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if(num_fakes < self.pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,self.pool_size-1)
                sampled_fake = fake_pool[random_id]
                fake_pool[random_id] = fake
                return sampled_fake
            else :
                return fake

    def data_setup(self):
        self.data_provider = DataProvider((self.sz_z, self.sz), self.args['source_folder'])

    def loss_setup(self):
        gen_loss = tf.reduce_mean( tf.abs(self.L3_pred - self.label) ) + tf.reduce_mean( tf.abs(self.L2_pred - self.L2_label) ) + tf.reduce_mean( tf.abs(self.L1_pred - self.L1_label) ) + tf.reduce_mean( tf.abs(self.L0_pred - self.L0_label) )
        
        GAN_weight = 0.1
        fm_loss = tf.reduce_mean( tf.abs(self.fake_f1 - self.real_f1) ) + tf.reduce_mean( tf.abs(self.fake_f2 - self.real_f2) ) + tf.reduce_mean( tf.abs(self.fake_f3 - self.real_f3) )
        
        g_loss = gen_loss/3 + fm_loss*GAN_weight
        
#        d_loss = tf.reduce_mean(tf.squared_difference(self.real_rec, 1) ) + tf.reduce_mean(tf.squared_difference(self.fake_rec, 1) )
        d_loss = ( tf.reduce_mean( tf.compat.v1.squared_difference( self.real_rec, 0) ) + tf.reduce_mean( tf.compat.v1.squared_difference( self.fake_rec, random.uniform(0.9, 1.0) ) ) )*0.5

        g_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate*2, beta1=0.5)
        d_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.compat.v1.trainable_variables()
        g_vars = [var for var in self.model_vars if 'gen' in var.name]
        d_vars = [var for var in self.model_vars if 'disc' in var.name]
        
        for var in self.model_vars: print(var.name)        
        self.g_trainer = g_optimizer.minimize(g_loss, var_list=g_vars)
        self.d_trainer = d_optimizer.minimize(d_loss, var_list=d_vars)

        #Summary variables for tensorboard
        self.g_loss = g_loss
        self.d_loss = d_loss 
        
    def train(self):
        self.model_setup()
        self.loss_setup()
        self.data_setup()
        
        saver = tf.compat.v1.train.Saver()
        num_repeat = 2
        weight_initialize = 1
        num_batch_samples = 1000
        training_epochs = 30
        curr_lr = 0.0001
        
        with tf.compat.v1.Session( ) as sess:
            if weight_initialize == 1:
                sess.run(tf.compat.v1.global_variables_initializer())
                tf.compat.v1.get_default_graph().finalize()
            else:
                graph = tf.compat.v1.get_default_graph()
                ckpt_path = saver.restore(sess, 'model')
            
            for repeat in range(0,num_repeat):
                for epoch in range(0,training_epochs):
                    self.data_provider.shuffle()
                    # Dealing with the learning rate as per the epoch number
                    if epoch == 10:
                        curr_lr = curr_lr/2
                    if epoch == 15:
                        curr_lr = curr_lr/2
                    if epoch == 20:    
                        curr_lr = curr_lr/2
                    if epoch == 25:
                        curr_lr = curr_lr/2
                                    
                    for i in range(0,num_batch_samples):
                        t = time.time()                        
                        sample_patch, sample_label = self.data_provider.get(self.batch_sz)
                        sample_patch = sample_patch[:,:,:,:,np.newaxis]
                        sample_label = sample_label[:,:,:,:,np.newaxis]
                        _, pred, g_loss = sess.run([self.g_trainer, self.L3_pred, self.g_loss], feed_dict = {self.img: sample_patch, self.label: sample_label, self.learning_rate: curr_lr} )
                        sampled_fake = self.fake_image_pool(self.num_fake_inputs, pred, self.fake_pool)
                        
                        _, d_loss = sess.run( [self.d_trainer, self.d_loss], feed_dict = {self.real_img: sample_patch, self.fake_img: sampled_fake, self.learning_rate: curr_lr} )
                        
                        self.num_fake_inputs = self.num_fake_inputs + 1                        
                        elapsed = time.time() - t
                        print("epoch %d: step %d, gen_loss %.04g, disc_loss %.04g, time = %g "%(epoch, i, g_loss, d_loss, elapsed))
                        
                    saver.save(sess, './model', write_meta_graph=False)        
            
            
    def load_model(self, batch_sz):
        self.batch_sz = batch_sz
        self.model_setup()
        saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        ckpt_path = saver.restore(self.sess, 'model')
        
    def denoising_patch(self, img):
        img = img.astype('float32')
        sc = self.max_value/2.0
        img = img/sc-1.0
        img = np.clip(img, -1, 1)
                
        [depth,height,width] = img.shape
        input_img = np.reshape(img,[1,depth,height,width,1])
                
        L0_pred, L1_pred, L2_pred, L3_pred = self.sess.run( [self.L0_pred, self.L1_pred, self.L2_pred, self.L3_pred], feed_dict = {self.img: input_img} )
        L3_pred = np.clip(L3_pred[0], -1, 1)
        denoised_img = (L3_pred+1)*self.max_value/2.0
        denoised_img = np.reshape(denoised_img, [depth,height,width] )
                
        L2_pred = np.clip(L2_pred[0], -1, 1)
        L2_pred = (L2_pred+1)*self.max_value/2.0
        L2_pred = np.reshape(L2_pred, [int(depth/2), int(height/2), int(width/2)])
        
        L1_pred = np.clip(L1_pred[0], -1, 1)
        L1_pred = (L1_pred+1)*self.max_value/2.0
        L1_pred = np.reshape(L1_pred, [int(depth/4), int(height/4), int(width/4)])
        
        L0_pred = np.clip(L0_pred[0], -1, 1)
        L0_pred = (L0_pred+1)*self.max_value/2.0
        L0_pred = np.reshape(L0_pred, [int(depth/8), int(height/8), int(width/8)])
        
        return L0_pred, L1_pred, L2_pred, denoised_img
    
    def denoising_img(self, img):
        # TODO: shape
        sliding_step = [4, 128, 128]
        denoised_img = window_sliding(self, img, sliding_step, self.max_value, self.sz, self.batch_sz)
        
        return denoised_img



