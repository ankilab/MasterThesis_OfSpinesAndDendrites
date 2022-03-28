# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:09:56 2018

@author: sehyung
"""

from deconv.mu_net1.cnn_models import *
from deconv.mu_net1.utils2 import *
from data_augmentation import DataProvider as dp
import gc
from tqdm import tqdm
import os
import pickle

tf.compat.v1.enable_v2_behavior()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


else:
    print("Not enough GPU hardware devices available")


class Denoiser():
    """
    Implements model described in
    "Lee, Sehyung, et al. "Mu-net: Multi-scale U-net for two-photon microscopy image denoising and restoration."
    Neural Networks 125 (2020): 92-103."
    """

    def __init__(self, args):
        self.args = args
        # basic parameters
        self.batch_sz = 1 if 'batch_size' not in args.keys() else args['batch_size']
        self.sz = args.get('xy_shape', 64)
        self.sz_z = args.get('z_shape',16)
        self.max_value = 12870
        self.min_value = -2327
        self.learning_rate = args.get('lr', 0.0001)
        self._train_history_setup()
        self.data_provider = None
        self.n_levels=args['n_levels'] if 'n_levels' in args.keys() else 2
        self.model_path = os.path.join(args['result_path'],'model')
        self.model_setup()
        print(self.model.summary)
        self._loss_setup()
        tx=args.get('train_flag', True)
        if tx:
            self._data_setup()

    def model_setup(self):
        """
        Initialize model based on object parameters.

        """

        self.img = tf.keras.layers.Input(shape=(self.sz_z, self.sz, self.sz, 1), batch_size=self.batch_sz, name='img',
                                         dtype=tf.float32)
        # self.img = tf.keras.layers.Input(shape=(None, None, None, 1), batch_size=self.batch_sz, name='img',
        #                                  dtype=tf.float32)
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
        if self.n_levels>3 or self.n_levels<0:
            print("Specified number of levels is not implemented. Falls back to 3 Levels.")
            self.n_levels=3

        if self.n_levels == 3:
            L0_L1, self.L0_pred = munet_cnn_level_0(self.L0_img, name='gen_l0')
            L1_L2, self.L1_pred = munet_cnn_level_1(self.L1_img, L0_L1, name='gen_l1')
            L2_L3, self.L2_pred = munet_cnn_level_2(self.L2_img, L1_L2, name='gen_l2')
            self.L3_pred = munet_cnn_level_3(self.img, L2_L3, name='gen_l3')
            self.model = tf.keras.Model(inputs=self.img, outputs=[self.L0_pred, self.L1_pred, self.L2_pred, self.L3_pred])
            self.gt=tf.keras.Model(inputs=self.label, outputs=[self.L0_label, self.L1_label, self.L2_label])

        elif self.n_levels == 2:
            L1_L2, self.L1_pred = munet_cnn_level_1(self.L1_img, name='gen_l1')
            L2_L3, self.L2_pred = munet_cnn_level_2(self.L2_img, L1_L2, name='gen_l2')
            self.L3_pred = munet_cnn_level_3(self.img, L2_L3, name='gen_l3')
            self.model = tf.keras.Model(inputs=self.img, outputs=[self.L1_pred, self.L2_pred, self.L3_pred])
            self.gt = tf.keras.Model(inputs=self.label, outputs=[self.L1_label, self.L2_label])
            self.gt.compile(loss='mse', optimizer='adam')

        elif self.n_levels == 1:
            L2_L3, self.L2_pred = munet_cnn_level_2(self.L2_img, name='gen_l2')
            self.L3_pred = munet_cnn_level_3(self.img, L2_L3, name='gen_l3')
            self.model = tf.keras.Model(inputs=self.img, outputs=[self.L2_pred, self.L3_pred])
            self.gt = tf.keras.Model(inputs=self.label, outputs=[self.L2_label])
            self.gt.compile(loss='mse', optimizer='adam')

        elif self.n_levels == 0:
            self.L3_pred = munet_cnn_level_3(self.img, name='gen_l3')
            self.model = tf.keras.Model(inputs=self.img, outputs=[self.L3_pred])
            self.gt = None

    def _train_history_setup(self):
        self.train_hist = []

    def _data_setup(self):
        if self.data_provider is None:
            self.data_provider = dp((self.sz_z, self.sz), self.args['data_path'], self.args['source_folder'],
                                          self.args['target_folder'], self.args['n_patches'])

    def _loss_setup(self):
        l = {'L0_pred': self._gen_loss, 'L1_pred': self._gen_loss,
                      'L2_pred': self._gen_loss, 'L3_pred': self._gen_loss}

        # Extract relevant loss based on amount of levels
        self.loss_func={key: l[key] for key in list(l.keys())[0+(3-self.n_levels):]}

    def _gen_loss(self, y_true, y_pred):
        gen_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return gen_loss

    def train(self, data_provider, epochs=30):
        self.data_provider = data_provider
        num_batch_samples = np.ceil(self.data_provider.size[0]/self.batch_sz).astype(int)
        self._train_history_setup()

        # Set up optimizers
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,
            decay_steps=5*num_batch_samples, decay_rate=0.5, staircase=True)

        g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        self.model.compile(optimizer=g_optimizer, loss=self.loss_func, run_eagerly=True)

        for _ in tqdm(range(0, epochs)):
            self.data_provider.shuffle()

            for _ in range(0, num_batch_samples):
                sample_patch, sample_label = self.data_provider.get(self.batch_sz)
                sample_patch = sample_patch[:, :, :, :, np.newaxis]
                sample_label = sample_label[:, :, :, :, np.newaxis]

                # L0_label, L1_label, L2_label = self.gt.predict(sample_label
                if self.n_levels>0:
                    gt_res = self.gt.predict(sample_label)
                    if not isinstance(gt_res, list):
                        gt_res = [gt_res]
                    gt_res.append(sample_label)
                else:
                    gt_res = sample_label

                # Compile and fit models
                # history = self.model.fit(sample_patch, [L0_label, L1_label, L2_label, sample_label],
                #                          batch_size=self.batch_sz, callbacks=[MyCustomCallback()])
                history = self.model.fit(sample_patch, gt_res,
                                         batch_size=self.batch_sz, callbacks=[MyCustomCallback()])
                self.train_hist.append(history.history)
            self.model.save(self.model_path)
        if not self.n_levels == 0:
            self.gt.save(self.model_path + '/gt')
        self.train_hist = {k: [d.get(k) for d in self.train_hist]
            for k in set().union(*self.train_hist)}
        with open(os.path.join(self.model_path, 'train_history.pkl'), 'wb') as outfile:
            pickle.dump(self.train_hist, outfile, pickle.HIGHEST_PROTOCOL)

        return self.model_path, self.train_hist

    def load_model(self, batch_size=4, path='./model'):
        self.model = tf.keras.models.load_model(path, compile=False)
        if not tf.is_tensor(self.model.output):
            self.n_levels = len(self.model.output)
        else:
            self.n_levels = 1
        # self.gt = tf.keras.models.load_model(path+'/gt', compile=False)

    def denoising_patch(self, img):
        img = img.astype('float32') - self.min_value
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

    def denoising_img(self, img, sliding_step=None):
        if sliding_step is None:
            sliding_step = [16, 32, 32]
        sliding_step = np.array(sliding_step)
        denoised_img = window_sliding(self, img, sliding_step, patch_sz= np.array([self.sz_z, self.sz, self.sz]),
                                      max_value=self.max_value,min_value= self.min_value, batch_sz=self.batch_sz, n_levels=self.n_levels)
        return denoised_img


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate, num_steps_p_epoch):
    self.initial_learning_rate = initial_learning_rate
    self.n_steps = num_steps_p_epoch

  def __call__(self, step):
      if step == 10*self.n_steps:
          self.initial_learning_rate = self.initial_learning_rate / 2
      if step == 15*self.n_steps:
          self.initial_learning_rate = self.initial_learning_rate / 2
      if step == 20*self.n_steps:
          self.initial_learning_rate = self.initial_learning_rate / 2
      if step == 25*self.n_steps:
          self.initial_learning_rate = self.initial_learning_rate / 2
      return self.initial_learning_rate

  def get_config(self):
    config = {
    'initial_learning_rate': self.initial_learning_rate,
    'n_steps': self.n_steps,

     }
    return config


