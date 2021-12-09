import numpy as np
import imageio as io
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

class SIREN:
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30., hidden_omega_0=30.):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.model_setup()

    def SineActivation(self, x, omega_0):
        return tf.sin(omega_0 * x)

    def SineLayer(self, x, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        if is_first:
            init = tf.keras.initializers.RandomUniform(-1 / in_features,
                                                       1 / in_features)
        else:
            init = tf.keras.initializers.RandomUniform(-np.sqrt(6 / in_features) / omega_0,
                                                       np.sqrt(6 / in_features) / omega_0)

        x = tf.keras.layers.Dense(out_features, kernel_initializer=init, use_bias=bias)(x)
        x = self.SineActivation(x, omega_0)
        return x

    def model_setup(self):
        in_layer = tf.keras.layers.Input((self.in_features,))
        x = self.SineLayer(in_layer, self.in_features, self.hidden_features, is_first=True, omega_0=self.first_omega_0)

        for i in range(self.hidden_layers):
            x = self.SineLayer(x, self.hidden_features, self.hidden_features, is_first=False, omega_0=self.hidden_omega_0)

        if self.outermost_linear:
            init = tf.keras.initializers.RandomUniform(-np.sqrt(6 / self.in_features) / self.hidden_omega_0,
                                                       np.sqrt(6 / self.in_features) / self.hidden_omega_0)

            final_layer = tf.keras.layers.Dense(self.out_features, kernel_initializer=init, )(x)

        else:
            final_layer = self.SineLayer(x, self.hidden_features, self.out_features, is_first=False,
                                         omega_0=self.hidden_omega_0)

        self.model = tf.keras.models.Model(in_layer, final_layer)
        self.model.compile("adam","mse")


    def get_mgrid3(self, shape, x=1, y=1, z=.1):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors = (np.linspace(-z, z, num=shape[0]), np.linspace(-y, y, num=shape[1]), np.linspace(-x, x, num=shape[2]))
        xx, yy, zz = np.meshgrid(*tensors, indexing='ij')
        M = np.concatenate((xx[..., None], yy[..., None], zz[..., None]), axis=3)
        return M.reshape(-1, len(shape))

    def train(self, steps, X, y, step_to_plot, orig_shape):
        loss = []
        for i in tqdm(range(steps)):
            h=self.model.fit(X, y, batch_size=len(y), epochs=1, verbose=0)
            loss.append(h.history['loss'])

            if i % step_to_plot == 0:
                plt.figure()
                plt.title(f"Step {i}")
                plt.imshow(self.model.predict(X, batch_size=len(y)).reshape(orig_shape).sum(0), cmap='gray')
                plt.show()

        plt.figure()
        plt.plot(loss)
        plt.show()






