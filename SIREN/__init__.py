import numpy as np
import imageio as io
import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial

class SIREN:
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
        return tf.sin(omega_0 * x)

    def SineLayer(self, x, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        if is_first:
            init = tf.keras.initializers.RandomUniform(-1 / in_features,
                                                       1 / in_features)
        else:
            init = tf.keras.initializers.RandomUniform(-np.sqrt(6 / in_features) / omega_0,
                                                       np.sqrt(6 / in_features) / omega_0)

        x = tf.keras.layers.Dense(out_features, kernel_initializer=init, use_bias=bias, dtype=tf.float32)(x)
        x = self.SineActivation(x, omega_0) if self.sine else tf.nn.relu(x)
        return x

    def model_setup(self):
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
        write_to_file(X,y,self.data_path)

    def train(self, steps, X, y, step_to_plot, orig_shape, batch_size=128):
        loss = []
        dataset = get_dataset(self.data_path)
        dataset = dataset.shuffle(len(y), reshuffle_each_iteration=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        # shuffled = dataset.shuffle(buffer_size=len(y)).batch(batch_size)

        # for i in tqdm(range(steps)):
        #     p = np.random.permutation(len(y))
        #     Xs, ys= X[p], y[p]
        #     h = self.model.fit(Xs,ys, epochs=1, verbose=1, batch_size=batch_size)
        h=self.model.fit(dataset, epochs=steps, verbose=1, batch_size=batch_size)
        loss = h.history['loss']
            # loss.append(h.history['loss'])

            # if step_to_plot!=0 and i % step_to_plot == 0:
            #     plt.figure()
            #     plt.title(f"Step {i}")
            #     plt.imshow(self.model.predict(X, batch_size=batch_size).reshape(orig_shape).sum(0), cmap='gray')
            #     plt.show()

        plt.figure()
        plt.plot(loss)
        plt.show()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def parse_single_item(X, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        'x': _bytes_feature(serialize_array(X)),
        'label': _float_feature(label)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def read_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }

    content = tf.io.parse_single_example(element, data)

    label = content['label']
    X = content['x']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(X, out_type=tf.float32)
    feature = tf.reshape(feature, shape=[3])
    return (feature, label)


def get_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        read_element
    )
    return dataset


def write_to_file(X,y, path):
    with tf.io.TFRecordWriter(path) as file_writer:
        for i in range(len(y)):
            out=parse_single_item(X[i], [y[i]])

            # example = tf.train.Example(features=tf.train.Features(feature={
            #     "x1": tf.train.Feature(float_list=tf.train.FloatList(value=X[:,0])),
            #     "x2": tf.train.Feature(float_list=tf.train.FloatList(value=X[:, 1])),
            #     "x3": tf.train.Feature(float_list=tf.train.FloatList(value=X[:, 2])),
            #     "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            # }))
            file_writer.write(out.SerializeToString())

# # https://keras.io/examples/keras_recipes/tfrecord/
# AUTOTUNE = tf.data.AUTOTUNE
# BATCH_SIZE= 64
# TRAINING_FILENAMES = './example.tfrecords'
#
#
# def read_tfrecord(example, labeled):
#     tfrecord_format = (
#         {
#             "x1": tf.io.FixedLenFeature([], tf.float32),
#             "x2": tf.io.FixedLenFeature([], tf.float32),
#             "x3": tf.io.FixedLenFeature([], tf.float32),
#             "y": tf.io.FixedLenFeature([], tf.float32),
#         }
#         if labeled
#         else {
#             "x1": tf.io.FixedLenFeature([], tf.float32),
#             "x2": tf.io.FixedLenFeature([], tf.float32),
#             "x3": tf.io.FixedLenFeature([], tf.float32),
#         }
#     )
#     example = tf.io.parse_single_example(example, tfrecord_format)
#     return example['x1'],  example['x2'], example['x3'], example['y']
#
#
# def load_dataset(filenames, labeled=True):
#     ignore_order = tf.data.Options()
#     ignore_order.experimental_deterministic = False  # disable order, increase speed
#     dataset = tf.data.TFRecordDataset(
#         [filenames]
#     )  # automatically interleaves reads from multiple files
#     dataset = dataset.with_options(
#         ignore_order
#     )  # uses data as soon as it streams in, rather than in its original order
#     dataset = dataset.map(
#         partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
#     )
#     # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
#     return dataset


# def reshape_example(x1,x2,x3,y):
#     x1, x2, x3 = x1[:,tf.newaxis], x2[:,tf.newaxis], x3[:,tf.newaxis]
#     return (tf.concat([x1,x2,x3], axis=1),y)


# def get_dataset(filenames, labeled=True):
#     dataset = load_dataset(filenames, labeled=labeled)
#     # dataset = dataset.map(reshape_example)
#     dataset = dataset.shuffle(2621440)
#     dataset = dataset.prefetch(buffer_size=AUTOTUNE)
#     dataset = dataset.batch(BATCH_SIZE)
#     return dataset






