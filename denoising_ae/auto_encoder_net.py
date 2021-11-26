import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, Concatenate
from tensorflow.keras.models import Model
import numpy as np
from tqdm import tqdm
import gc
import os
import pickle
from mu_net1 import utils2


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

class Autoencoder:
    def __init__(self, args):
        self.z_shape = 32
        self.xy_shape =128
        self.layer = 4
        self.filter_base = 64
        self.train_hist = []
        self.model_path = './model_auto_enc'
        self.create_model()

    def create_model(self):
        input_layer = Input((self.z_shape, self.xy_shape, self.xy_shape, 1))

        x = input_layer
        for i in range(self.layer):
            x = Conv3D(self.filter_base * 2 ** i, (3, 3, 3), activation='relu', padding='same')(x)
            x = Conv3D(self.filter_base * 2 ** i, (3, 3, 3), activation='relu', padding='same')(x)
            x = MaxPool3D()(x)

        x = Conv3D(self.filter_base * 2 ** 4, 3, activation='relu', padding='same', name='bottleneck')(x)

        for i in range(self.layer):
            x = UpSampling3D()(x)
            x = Conv3D(self.filter_base * 2 ** (self.layer - i - 1), (3, 3, 3), activation='relu', padding='same')(x)
            x = Conv3D(self.filter_base * 2 ** (self.layer - i - 1), (3, 3, 3), activation='relu', padding='same')(x)

        # Point convolution (kernel=1) und einem Filter ==> 1 Klasse (bspw. Glottis, Oder Katze)
        x = Conv3D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(x)

        self.model = Model(input_layer, x)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
        #     decay_steps=5*num_batch_samples, decay_rate=0.5, staircase=True)
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
        self.model.compile(optimizer=g_optimizer, loss='mse')

    def train(self, data_provider, epochs=30, batch_size = 4):
        self.data_provider = data_provider
        num_batch_samples = np.ceil(self.data_provider.size[0] / batch_size).astype(int)

        for _ in tqdm(range(0, epochs)):
            self.data_provider.shuffle()

            for _ in range(0, num_batch_samples):
                sample_patch, sample_label = self.data_provider.get(batch_size)
                sample_patch = sample_patch[:, :, :, :, np.newaxis]
                sample_label = sample_label[:, :, :, :, np.newaxis]

                history = self.model.fit(sample_patch, [sample_label],
                                         batch_size=batch_size, callbacks=[MyCustomCallback()])
                self.train_hist.append(history.history['loss'][0])
            self.model.save(self.model_path)
        with open(os.path.join(self.model_path, 'train_history_auto_enc.pkl'), 'wb') as outfile:
            pickle.dump(self.train_hist, outfile, pickle.HIGHEST_PROTOCOL)

        return self.model_path, self.train_hist

    def predict(self,img, sampling_step = np.array([8,32,32])):
        pred_img=utils2.window_sliding(self,img,sampling_step, (self.z_shape, self.xy_shape, self.xy_shape), 8192)
        return pred_img

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
        return self.model


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()