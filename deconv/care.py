from __future__ import print_function, unicode_literals, absolute_import, division
import timeit

import multiprocessing
from .deconvolver import Deconvolver
from csbdeep.utils import axes_dict, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import pickle
import os
import tifffile as tif
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
import gc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class CAREDeconv(Deconvolver):
    """
    Incorporate CSBDeep deconvolution implementation in our workflow.
    CARE:
    "Weigert, Martin, et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy."
    Nature methods 15.12 (2018): 1090-1097."

    """

    def __init__(self, args):

        super().__init__(args)

    def preprocess(self):
        """
        CARE preprocessing: None required.
        """
        pass

    def train(self, data_dir, validation_split =0.1, epochs =10, batch_size=8, learning_rate = 0.0004, unet_residual=True,
              unet_n_depth=2):
        """
        Train CARE model.

        :param data_dir: Folder/file with training data
        :type data_dir: string
        :param validation_split: Proportion of training data to use as validation data (between 0 and 1), defaults to 0.1
        :type validation_split: float, optional
        :param epochs: Number of training epochs, defaults to 10
        :type epochs: int, optional
        :param batch_size: Training batch size, defaults to 8
        :type batch_size: int, optional
        :param learning_rate: Learning rate for training, defaults to 0.0004
        :type learning_rate: float, optional
        :param unet_residual: Whether to use residual learning, defaults to True
        :type unet_residual: bool, optional
        :param unet_n_depth: U-Net Depth, defaults to 2
        :type unet_n_depth: int, optional
        :return: Model directory, model path
        :rtype: string, string
        """

        # Load training data, easiest obtained using data_augmentation.DataAugmenter.augment()
        (X, Y), (X_val, Y_val), axes = load_training_data(data_dir,
                                                          validation_split=validation_split, verbose=True)

        # Calculate number of steps per epoch such that all data is presented once
        train_steps = X.shape[0]//batch_size
        train_steps= train_steps+1 if X.shape[0]%batch_size!=0 else train_steps

        c = axes_dict(axes)['C']
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

        # Specify training parameters
        config = Config(axes, n_channel_in, n_channel_out, train_batch_size=batch_size,
                        train_steps_per_epoch=train_steps, train_epochs=epochs, train_learning_rate=learning_rate,
                        unet_residual=unet_residual, unet_n_depth=unet_n_depth)
        model_dir = os.path.join(self.res_path,'models')

        # Initialize CARE model
        model = CARE(config, 'my_model', basedir=model_dir)
        model_dir = os.path.join(model_dir, 'my_model')
        print(model.keras_model.summary())

        # Train CARE model
        history = model.train(X, Y, validation_data=(X_val, Y_val))

        # Save training history
        history.history['lr'] = [float(f) for f in history.history['lr']]
        res_dict = dict(history.history, **history.params)

        with open(os.path.join(self.res_path, f'results_care.pkl'), 'wb') \
                as outfile:
            pickle.dump(res_dict, outfile, pickle.HIGHEST_PROTOCOL)

        # Save model
        mdl_path = os.path.join(self.res_path, 'TF_SavedModel.zip')
        model.export_TF(mdl_path)

        # Plot training metrics
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
        with open(os.path.join(self.res_path, 'history_care.pkl'), 'wb') as outfile:
            pickle.dump(history.history, outfile, pickle.HIGHEST_PROTOCOL)

        return model_dir, mdl_path

    def predict(self, data_dir, model_dir, name,  res_folder='./'):
        """
        Deconvolve all images in specified folder using CARE model.

        :param data_dir: Folder containing files that should be deconvolved
        :type data_dir: string
        :param model_dir: Directory where CARE model is stored
        :type model_dir: string
        :param name: Directory where CARE model is stored
        :type name: string
        :param res_folder: Folder where results are to be stored, defaults to './'
        :type res_folder: string, optional
        :return: Time taken to deconvolve each image
        :rtype: list[float]
        """
        p = os.path.join(self.res_path, 'Predictions')
        if not os.path.exists(p):
            os.makedirs(p)

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        t = []
        for f in files:
            X = io.imread(os.path.join(data_dir,f))

            _, tx=self.predict_img(X, model_dir, name, os.path.join(res_folder,f))
            t.append(tx)

            # Attempt to solve memory issues with tensorflow when deconvolving several images
            # p = multiprocessing.Process(target=self.predict_img, args=(X, model_dir, name, os.path.join(res_folder,f)),)
            # p.start()
            # p.join()
            del X
            tf.keras.backend.clear_session()
            gc.collect()
        return t

    def predict_img(self, X, model_dir, name, save_as=None):
        """
        Deconvolve image using CARE model.

        :param X: Input image
        :type X: nd.array
        :param model_dir: Directory where CARE model is stored
        :type model_dir: string
        :param name: Directory where CARE model is stored
        :type name: string
        :param save_as: File name under which to store the deconvolution result. If nothing is specified, it is not stored.
                        Defaults to None
        :type save_as: string, optional
        :return: Deconvolved image, time taken for deconvolution
        :rtype: nd.array, float
        """
        axes = 'ZYX'
        start = timeit.default_timer()

        model = CARE(config=None, name=name, basedir=model_dir)
        restored = model.predict(X, axes)
        t = timeit.default_timer()-start

        # if no name is specified, the result is not saved to file
        if save_as is not None:
            tif.imwrite(save_as, restored)
            print(save_as)
        # Attempt to solve memory issues with tensorflow
        del model
        gc.collect()
        return restored, t

