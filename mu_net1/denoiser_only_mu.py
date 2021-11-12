# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:09:56 2018

@author: sehyung
"""

import tensorflow as tf
from mu_net1.cnn_models import *
from mu_net1.utils import *
from data_augmentation import DataProvider
import os
from tqdm import tqdm
from csbdeep.internals import nets, train


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

else:
    print("Not enough GPU hardware devices available")


class Denoiser():

    def __init__(self, args):
        self.args = args
        # basic parameters
        self.batch_sz = 1 if 'batch_size' not in args.keys() else args['batch_size']
        self.sz = 128
        self.sz_z = 32
        self.max_value = 8191  # we set maximum value to 5,000 of microscope output
        self.learning_rate = None
        self.model_path = './model'
        self.model_setup()
        print(self.model.summary)
        self.loss_setup()
        self.data_setup()

    def model_setup(self):
        self.img = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz, name='img',
                                         dtype=tf.float32)
        self.label = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz,
                                           name='label', dtype=tf.float32)

        self.L2_img = tf.keras.layers.Conv3D(1, 1, strides=(2, 2, 2), kernel_initializer=tf.keras.initializers.Ones(),
                                             use_bias=False, padding='same')(self.img)
        self.L1_img = tf.keras.layers.Conv3D(1, 1, strides=(4, 4, 4), kernel_initializer=tf.keras.initializers.Ones(),
                                             use_bias=False, padding='same')(self.img)
        self.L0_img = tf.keras.layers.Conv3D(1, 1, strides=(8, 8, 8), kernel_initializer=tf.keras.initializers.Ones(),
                                             use_bias=False, padding='same')(self.img)

        self.L2_label = tf.nn.conv3d(self.label, tf.constant(1.0, shape=(1, 1, 1, 1, 1)), strides=[1, 2, 2, 2, 1],
                                padding='SAME')
        self.L1_label = tf.nn.conv3d(self.label, tf.constant(1.0, shape=(1, 1, 1, 1, 1)), strides=[1, 4, 4, 4, 1],
                                padding='SAME')
        self.L0_label = tf.nn.conv3d(self.label, tf.constant(1.0, shape=(1, 1, 1, 1, 1)), strides=[1, 8, 8, 8, 1],
                                padding='SAME')
        # self.L2_label = tf.keras.layers.Conv3D(1, 1, strides=(2, 2, 2), kernel_initializer=tf.keras.initializers.Ones(),
        #                                      use_bias=False, padding='same')(self.label)
        # self.L1_label = tf.keras.layers.Conv3D(1, 1, strides=(4, 4, 4), kernel_initializer=tf.keras.initializers.Ones(),
        #                                      use_bias=False, padding='same')(self.label)
        # self.L0_label = tf.keras.layers.Conv3D(1, 1, strides=(8, 8, 8), kernel_initializer=tf.keras.initializers.Ones(),
        #                                      use_bias=False, padding='same')(self.label)

        L0_L1, self.L0_pred = munet_cnn_level_0(self.L0_img, name='gen_l0')
        L1_L2, self.L1_pred = munet_cnn_level_1(self.L1_img, L0_L1, name='gen_l1')
        L2_L3, self.L2_pred = munet_cnn_level_2(self.L2_img, L1_L2, name='gen_l2')
        self.L3_pred = munet_cnn_level_3(self.img, L2_L3, name='gen_l3')

        self.model = tf.keras.Model(inputs=self.img, outputs=[self.L0_pred, self.L1_pred, self.L2_pred, self.L3_pred])
        self.gt=tf.keras.Model(inputs=self.label, outputs=[self.L0_label, self.L1_label, self.L2_label])
        self.gt.compile(loss='mse', optimizer='adam')

    def data_setup(self):
        self.data_provider = DataProvider((self.sz_z, self.sz), self.args['data_path'], self.args['source_folder'],
                                          self.args['target_folder'], self.args['n_patches'])

    def loss_setup(self):
        self.loss_func = {'L0_pred': self._gen_loss, 'L1_pred': self._gen_loss,
                     'L2_pred': self._gen_loss, 'L3_pred': self._gen_loss}

    def _gen_loss(self, y_true, y_pred):
        gen_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return gen_loss

    def lr_schedule(self, epoch, curr_lr=0.0001):
        curr_lr = self.learning_rate
        if epoch == 10:
            curr_lr = curr_lr / 2
        if epoch == 15:
            curr_lr = curr_lr / 2
        if epoch == 20:
            curr_lr = curr_lr / 2
        if epoch == 25:
            curr_lr = curr_lr / 2
        self.learning_rate = curr_lr
        return curr_lr

    def train(self, epochs=30):
        num_batch_samples = np.ceil(self.data_provider.size[0]/self.batch_sz).astype(int)
        self.learning_rate = 0.0001

        for epoch in range(0, epochs):
            self.data_provider.shuffle()

            # Set up optimizers
            g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule(epoch) * 2, beta_1=0.5)
            self.model.compile(optimizer=g_optimizer, loss=self.loss_func)

            # ###test
            # training_data=train.DataWrapper(self.data_provider.X, self.data_provider.Y, self.batch_sz,
            #                                 length=epochs*num_batch_samples)
            # self.model.fit(iter(training_data),[L0_label, L1_label, L2_label, sample_label])
            #
            # self.data_provider

            for _ in tqdm(range(0, num_batch_samples)):
                sample_patch, sample_label = self.data_provider.get(self.batch_sz)
                sample_patch = sample_patch[:, :, :, :, np.newaxis]
                sample_label = sample_label[:, :, :, :, np.newaxis]

                L0_label, L1_label, L2_label = self.gt.predict(sample_label)

                # Compile and fit models
                self.model.fit(sample_patch, [L0_label, L1_label, L2_label, sample_label])

            self.model.save_weights(self.model_path)

    def load_model(self, batch_size=4, path='./model'):
        self.model.load_weights(path)

    def denoising_patch(self, img):
        img = img.astype('float32')
        sc = self.max_value / 2.0
        img = img / sc - 1.0
        img = np.clip(img, -1, 1)

        [depth, height, width] = img.shape
        input_img = np.reshape(img, [1, depth, height, width, 1])

        L0_pred, L1_pred, L2_pred, L3_pred = self.model.predict(input_img)
        L3_pred = np.clip(L3_pred[0], -1, 1)
        denoised_img = (L3_pred + 1) * self.max_value / 2.0
        denoised_img = np.reshape(denoised_img, [depth, height, width])

        L2_pred = np.clip(L2_pred[0], -1, 1)
        L2_pred = (L2_pred + 1) * self.max_value / 2.0
        L2_pred = np.reshape(L2_pred, [int(depth / 2), int(height / 2), int(width / 2)])

        L1_pred = np.clip(L1_pred[0], -1, 1)
        L1_pred = (L1_pred + 1) * self.max_value / 2.0
        L1_pred = np.reshape(L1_pred, [int(depth / 4), int(height / 4), int(width / 4)])

        L0_pred = np.clip(L0_pred[0], -1, 1)
        L0_pred = (L0_pred + 1) * self.max_value / 2.0
        L0_pred = np.reshape(L0_pred, [int(depth / 8), int(height / 8), int(width / 8)])

        return L0_pred, L1_pred, L2_pred, denoised_img

    def denoising_img(self, img):
        sliding_step = [32, 128, 128]
        denoised_img = window_sliding(self, img, sliding_step, self.max_value, self.sz, self.batch_sz)

        return denoised_img
