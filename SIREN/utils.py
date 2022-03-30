import numpy as np
import tensorflow as tf

def get_mgrid3(shape, x=1.0, y=1.0, z=.1):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    :param shape: Input shape (i.e. image shape)
    :type shape: tuple(int, int, int)
    :param x: Maximum grid value in x-direction
    :type x: float
    :param y: Maximum grid value in y-direction
    :type y: float
    :param z: Maximum grid value in z-direction
    :type z: float
    :return: Rescaled grid
    """

    xx, yy, zz = rescale_indices(shape, x, y, z)
    return flatten_meshgrid(shape, xx, yy, zz)


def rescale_indices(shape, x=1, y=1, z=.1):
    """

    :param shape: Input shape (i.e. image shape)
    :type shape: tuple(int, int, int)
    :param x: Maximum grid value in x-direction
    :type x: float
    :param y: Maximum grid value in y-direction
    :type y: float
    :param z: Maximum grid value in z-direction
    :type z: float
    :return: Meshgrid (x,y,z-direction)
    :rtype: nd.array, nd.array, nd.array
    """
    tensors = (np.linspace(-z, z, num=shape[0]), np.linspace(-y, y, num=shape[1]), np.linspace(-x, x, num=shape[2]))
    xx, yy, zz = np.meshgrid(*tensors, indexing='ij')
    return xx, yy, zz


def flatten_meshgrid(shape, xx, yy, zz):
    """
    Reshape grid such that corresponding x-y-z values are aligned.

    :param shape: Input shape (i.e. image shape)
    :type shape: tuple(int, int, int)
    :param xx: Grid values in x-direction
    :type xx: nd.array
    :param yy: Grid values in y-direction
    :type yy: nd.array
    :param zz: Grid values in z-direction
    :type zz: nd.array
    :return: Reshaped grid
    """
    M = np.concatenate((xx[..., None], yy[..., None], zz[..., None]), axis=3)
    return M.reshape(-1, len(shape))

# Functionality to enable dataset saved to file
# and reading from file successively. This can enable handling larger amounts of data
def _bytes_feature(value):
    """Returns a bytes_list from a str / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def _parse_single_item(X, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        'x': _bytes_feature(_serialize_array(X)),
        'label': _float_feature(label)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def _read_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'x': tf.io.FixedLenFeature([], tf.str),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }

    content = tf.io.parse_single_example(element, data)

    label = content['label']
    X = content['x']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(X, out_type=tf.float32)
    feature = tf.reshape(feature, shape=[3])
    return (feature, label)


def _get_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        _read_element, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def _write_to_file(X, y, path):
    with tf.io.TFRecordWriter(path) as file_writer:
        for i in range(len(y)):
            out=_parse_single_item(X[i], [y[i]])

            # example = tf.train.Example(features=tf.train.Features(feature={
            #     "x1": tf.train.Feature(float_list=tf.train.FloatList(value=X[:,0])),
            #     "x2": tf.train.Feature(float_list=tf.train.FloatList(value=X[:, 1])),
            #     "x3": tf.train.Feature(float_list=tf.train.FloatList(value=X[:, 2])),
            #     "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            # }))
            file_writer.write(out.SerializeTostr())