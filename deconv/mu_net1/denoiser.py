# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:09:56 2018

@author: sehyung
"""

import tensorflow as tf
from mu_net1.cnn_models import *
from mu_net1.utils import *
import random
import time
from data_augmentation import DataProvider
from tensorflow.keras.callbacks import LearningRateScheduler


# tf.compat.v1.disable_eager_execution()


class Denoiser():
    # tf.compat.v1.reset_default_graph()

    def __init__(self, args):
        self.args = args
        # basic parameters
        self.batch_sz = 1
        self.sz = 128
        self.sz_z = 32
        self.max_value = 5000.0  # we set maximum value to 5,000 of microscope output
        self.learning_rate = None
        self.model_setup()
        print(self.model.summary)
        self.loss_setup()
        self.data_setup()

    def model_setup(self):
        self.img = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz, name='img',
                                         dtype=tf.float32)
        self.label = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz,
                                           name='label', dtype=tf.float32)

        self.L2_img = tf.nn.conv3d(self.img, filters=tf.constant(1.0, shape=[1, 1, 1, 1, 1]), strides=(1, 2, 2, 2, 1),
                                             padding='SAME')
        self.L1_img = tf.nn.conv3d(self.img, filters=tf.constant(1.0, shape=[1, 1, 1, 1, 1]), strides=(1, 4, 4, 4, 1),
                                             padding='SAME')
        self.L0_img = tf.nn.conv3d(self.img, filters=tf.constant(1.0, shape=[1, 1, 1, 1, 1]), strides=(1, 8, 8, 8, 1),
                                             padding='SAME')

        self.L2_label = tf.nn.conv3d(self.label, filters=tf.constant(1.0, shape=[1, 1, 1, 1, 1]), strides=(1, 2, 2, 2, 1),
                                               padding='SAME')
        self.L1_label = tf.nn.conv3d(self.label, filters=tf.constant(1.0, shape=[1, 1, 1, 1, 1]), strides=(1, 4, 4, 4, 1),
                                               padding='SAME')
        self.L0_label = tf.nn.conv3d(self.label, filters=tf.constant(1.0, shape=[1, 1, 1, 1, 1]), strides=(1, 8, 8, 8, 1),
                                               padding='SAME')

        self.real_img = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz,
                                              name='real_img', dtype=tf.float32)

        self.fake_img = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz,
                                              name='fake_img', dtype=tf.float32)

        L0_L1, self.L0_pred = munet_cnn_level_0(self.L0_img, name='gen_l0')
        L1_L2, self.L1_pred = munet_cnn_level_1(self.L1_img, L0_L1, name='gen_l1')
        L2_L3, self.L2_pred = munet_cnn_level_2(self.L2_img, L1_L2, name='gen_l2')
        self.L3_pred = munet_cnn_level_3(self.img, L2_L3, name='gen_l3')
        self.model = tf.keras.Model(self.img, self.L3_pred)

        _, _, _, self.real_rec = discriminator(self.real_img, 'disc')

        _, _, _, self.fake_rec = discriminator(self.fake_img, 'disc')
        self.fake_f1, self.fake_f2, self.fake_f3, _ = discriminator(self.L3_pred)
        self.real_f1, self.real_f2, self.real_f3, _ = discriminator(self.label)
        self.disc_model = tf.keras.Model(self.img, self.L3_pred)

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if (num_fakes < self.pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                sampled_fake = fake_pool[random_id]
                fake_pool[random_id] = fake
                return sampled_fake
            else:
                return fake

    def data_setup(self):
        self.data_provider = DataProvider((self.sz_z, self.sz), self.args['source_folder'])

    def loss_setup(self):
        self.gen_loss_setup()
        self.disc_loss_setup()
    #     # gen_loss = tf.reduce_mean(tf.abs(self.L3_pred - self.label)) + tf.reduce_mean(
    #     #     tf.abs(self.L2_pred - self.L2_label)) + tf.reduce_mean(
    #     #     tf.abs(self.L1_pred - self.L1_label)) + tf.reduce_mean(tf.abs(self.L0_pred - self.L0_label))
    #     #
    #     # GAN_weight = 0.1
    #     # fm_loss = tf.reduce_mean(tf.abs(self.fake_f1 - self.real_f1)) + tf.reduce_mean(
    #     #     tf.abs(self.fake_f2 - self.real_f2)) + tf.reduce_mean(tf.abs(self.fake_f3 - self.real_f3))
    #     #
    #     # g_loss = gen_loss / 3 + fm_loss * GAN_weight
    #     #
    #     # d_loss = (tf.reduce_mean(tf.compat.v1.squared_difference(self.real_rec, 0)) + tf.reduce_mean(
    #     #     tf.compat.v1.squared_difference(self.fake_rec, random.uniform(0.9, 1.0)))) * 0.5
    #
    #
    #
    #     # # Summary variables for tensorboard
    #     # self.g_loss = g_loss
    #     # self.d_loss = d_loss

    def gen_loss_setup(self, *args):
        gen_loss = tf.reduce_mean(tf.abs(self.L3_pred - self.label)) + tf.reduce_mean(
            tf.abs(self.L2_pred - self.L2_label)) + tf.reduce_mean(
            tf.abs(self.L1_pred - self.L1_label)) + tf.reduce_mean(tf.abs(self.L0_pred - self.L0_label))
        GAN_weight = 0.1
        fm_loss = tf.reduce_mean(tf.abs(self.fake_f1 - self.real_f1)) + tf.reduce_mean(
            tf.abs(self.fake_f2 - self.real_f2)) + tf.reduce_mean(tf.abs(self.fake_f3 - self.real_f3))

        g_loss = gen_loss / 3 + fm_loss * GAN_weight
        self.g_loss = g_loss
        return g_loss

    def disc_loss_setup(self, *args):
        d_loss = (tf.reduce_mean(tf.compat.v1.squared_difference(self.real_rec, 0)) + tf.reduce_mean(
            tf.compat.v1.squared_difference(self.fake_rec, random.uniform(0.9, 1.0)))) * 0.5
        self.d_loss = d_loss
        return d_loss

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

    def train(self):
        num_repeat = 2
        weight_initialize = 1
        num_batch_samples = 1000
        training_epochs = 30
        curr_lr = 0.0001

        for repeat in range(0, num_repeat):
            for epoch in range(0, training_epochs):
                self.data_provider.shuffle()
                # Dealing with the learning rate as per the epoch number

                self.learning_rate = curr_lr

                for i in range(0, num_batch_samples):
                    t = time.time()
                    sample_patch, sample_label = self.data_provider.get(self.batch_sz)
                    sample_patch = sample_patch[:, :, :, :, np.newaxis]
                    sample_label = sample_label[:, :, :, :, np.newaxis]

                    # Set up optimizers
                    g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule(epoch) * 2, beta_1=0.5)
                    d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule(epoch), beta_1=0.5)

                    self.model_vars = tf.compat.v1.trainable_variables()
                    g_vars = [var for var in self.model_vars if 'gen' in var.name]
                    d_vars = [var for var in self.model_vars if 'disc' in var.name]

                    self.g_trainer = g_optimizer.minimize(self.g_loss, var_list=g_vars)
                    self.d_trainer = d_optimizer.minimize(self.d_loss, var_list=d_vars)

                    # Compile and fit models
                    self.model.compile(optimizer=self.g_trainer, loss = self.g_loss)
                    _, pred, g_loss = self.model.fit(sample_patch, sample_label, callbacks=[LearningRateScheduler(self.lr_schedule, verbose=1)])
                    # _, pred, g_loss = sess.run([self.g_trainer, self.L3_pred, self.g_loss],
                    #                            feed_dict={self.img: sample_patch, self.label: sample_label,
                    #                                       self.learning_rate: curr_lr})
                    sampled_fake = self.fake_image_pool(self.num_fake_inputs, pred, self.fake_pool)
                    self.model.compile(optimizer=self.d_trainer, loss=self.d_loss)
                    _, d_loss = self.model.fit(sample_patch, sampled_fake,callbacks=[LearningRateScheduler(self.lr_schedule, verbose=1)])
                    # _, d_loss = sess.run([self.d_trainer, self.d_loss],
                    #                      feed_dict={self.real_img: sample_patch, self.fake_img: sampled_fake,
                    #                                 self.learning_rate: curr_lr})

                    self.num_fake_inputs = self.num_fake_inputs + 1
                    elapsed = time.time() - t
                    print("epoch %d: step %d, gen_loss %.04g, disc_loss %.04g, time = %g " % (
                        epoch, i, g_loss, d_loss, elapsed))

                self.model.save_weights('./model')

    def load_model(self, batch_sz):
        self.batch_sz = batch_sz
        self.model_setup()
        saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        ckpt_path = saver.restore(self.sess, 'model')

    def denoising_patch(self, img):
        img = img.astype('float32')
        sc = self.max_value / 2.0
        img = img / sc - 1.0
        img = np.clip(img, -1, 1)

        [depth, height, width] = img.shape
        input_img = np.reshape(img, [1, depth, height, width, 1])

        L0_pred, L1_pred, L2_pred, L3_pred = self.sess.run([self.L0_pred, self.L1_pred, self.L2_pred, self.L3_pred],
                                                           feed_dict={self.img: input_img})
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
        # TODO: shape
        sliding_step = [4, 128, 128]
        denoised_img = window_sliding(self, img, sliding_step, self.max_value, self.sz, self.batch_sz)

        return denoised_img
