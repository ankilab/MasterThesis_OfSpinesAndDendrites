import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class SIREN:
    """
    Network with sinosoidal activation functions, based on:
    "Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in Neural
    Information Processing Systems 33 (2020): 7462-7473."

    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30., hidden_omega_0=30., sine=True):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.data_path = os.path.join('.', "example.tfrecords")
        self.sine= sine
        self.model_setup()

    def SineActivation(self, x, omega_0):
        """
        Sinusoidal activation.

        :param x: input
        :param omega_0: Frequency tuning parameter
        :type omega_0: float
        :return: sinusoidal activated input
        """
        return tf.sin(omega_0 * x)

    def SineLayer(self, X, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        """
        Fully connected layer with sinus activation.

        :param X: Input
        :param in_features: Number of inputs
        :type in_features: int
        :param out_features: Number of outputs
        :type out_features: int
        :param bias: Whether to use bias
        :type bias: bool
        :param is_first: Whether it is the first layer
        :type is_first: bool
        :param omega_0: Frequency tuning parameter
        :type omega_0: float
        :return: Layer output
        """

        if is_first:
            init = tf.keras.initializers.RandomUniform(-1 / in_features,
                                                       1 / in_features)
        else:
            init = tf.keras.initializers.RandomUniform(-np.sqrt(6 / in_features) / omega_0,
                                                       np.sqrt(6 / in_features) / omega_0)

        X = tf.keras.layers.Dense(out_features, kernel_initializer=init, use_bias=bias, dtype=tf.float32)(X)
        X = self.SineActivation(X, omega_0) if self.sine else tf.nn.relu(X)

        return X

    def model_setup(self):
        """
        Initialize and compile network based on object parameters set at initialization.

        """
        in_layer = tf.keras.layers.Input((self.in_features,),dtype=tf.float32)

        # feature_names = ['x1', 'x2', 'x3']
        # columns = [tf.feature_column.numeric_column(header) for header in feature_names]
        # in_layer = tf.keras.layers.DenseFeatures(columns)
        x = self.SineLayer(in_layer, self.in_features, self.hidden_features, is_first=True, omega_0=self.first_omega_0)

        for i in range(self.hidden_layers):
            x = self.SineLayer(x, self.hidden_features, self.hidden_features, is_first=False, omega_0=self.hidden_omega_0)

        if self.outermost_linear:
            init = tf.keras.initializers.RandomUniform(-np.sqrt(6 / self.in_features) / self.hidden_omega_0,
                                                       np.sqrt(6 / self.in_features) / self.hidden_omega_0)

            final_layer = tf.keras.layers.Dense(self.out_features, kernel_initializer=init, dtype=tf.float32)(x)

        else:
            final_layer = self.SineLayer(x, self.hidden_features, self.out_features, is_first=False,
                                         omega_0=self.hidden_omega_0)

        self.model = tf.keras.models.Model(in_layer, final_layer)
        self.model.compile("adam","mse")

    def preprocess(self, X,y):
        """
        Preprocessing: None required.

        :param X: Input data
        :param y: Output data
        """
        #_write_to_file(X,y,self.data_path)
        pass

    def train(self, steps, X, y, batch_size=None):
        """
        Train the network.

        :param steps: Number of training steps
        :type steps: int
        :param X: Input data
        :param y: Output data (ground truth)
        :param batch_size: Batch size
        :type batch_size: int or NoneType
        :return: Loss over training training time
        :rtype: list[float]
        """
        loss = []

        # By using the commented code in this function instead of the fitting not commented, the dataset is saved to
        # file and can be read from file successively. This can enable handling larger amounts of data
        # dataset = _get_dataset(self.data_path)
        # dataset = dataset.shuffle(len(y), reshuffle_each_iteration=True)
        # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # dataset = dataset.batch(batch_size)
        # shuffled = dataset.shuffle(buffer_size=len(y)).batch(batch_size)

        if batch_size is None:
            batch_size=len(y)

        for _ in tqdm(range(steps)):
            p = np.random.permutation(len(y))
            Xs, ys= X[p], y[p]
            h = self.model.fit(Xs,ys, epochs=1, verbose=1, batch_size=batch_size)
            loss.append(h.history['loss'])

        # h=self.model.fit(dataset, epochs=steps, verbose=1, batch_size=batch_size)
        # loss = h.history['loss']

        plt.figure()
        plt.plot(loss)
        plt.show(block=False)
        return loss

    def predict(self, X, batch_size):
        """
        Predict output for input data.

        :param X: Input data
        :param batch_size: Batch size
        :return: Predictions
        """
        return self.model.predict(X, batch_size=batch_size)






