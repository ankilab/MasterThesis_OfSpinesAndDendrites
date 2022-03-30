import timeit

from .deconvolver import Deconvolver
from .mu_net1 import denoiser_only_mu as den
import tifffile as tif
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import pickle


class Mu_Net(Deconvolver):
    """
    Deconvolve image using Mu-Net.
    Based on:
    "Lee, Sehyung, et al. "Mu-net: Multi-scale U-net for two-photon microscopy image denoising and restoration."
    Neural Networks 125 (2020): 92-103."
    """

    def __init__(self, args):

        super().__init__(args)
        self.train_flag = args.get('train', True)
        self.denoiser = den.Denoiser(args)

    def preprocess(self):
        """
        Mu-Net preprocessing: None required.
        """
        pass

    def train(self, data_provider, epochs=10, batch_size=8):
        """
        Train Mu-Net model.

        :param data_provider: Object handling input data.
        :type data_provider: data_augmentation.DataProvider object
        :param epochs: Number of training epochs, defaults to 10
        :type epochs: int, optional
        :param batch_size: Training batch size, defaults to 8
        :type batch_size: int, optional
        :return: Directory containing model, Training history
        :rtype: str, dict
        """
        model_dir, train_hist= self.denoiser.train(data_provider, epochs=epochs)
        with open(os.path.join(self.res_path, 'history_mu_net.pkl'), 'wb') as outfile:
            pickle.dump(train_hist, outfile, pickle.HIGHEST_PROTOCOL)
        return model_dir, train_hist

    def plot_training(self, train_history):
        """
        Plot training history.

        :param train_history: Object containing training history (output of train())
        :type train_history: dict
        """
        x_max= 0
        for k in train_history.keys():
            y= train_history[k]
            plt.plot(np.arange(len(y)), y)
            x_max = len(y)
        plt.grid()
        plt.legend(train_history.keys())
        plt.xlim(0,x_max)
        plt.xlabel('Number of steps')
        plt.show(block=False)

    def predict(self, data_dir, model_dir):
        """
        Deconvolve all tif-files within specified folder

        :param data_dir: File containing image data
        :type data_dir: str
        :param model_dir: Directory containing trained Mu-Net model
        :type model_dir: str
        :return: Time taken to deconvolve each image
        :rtype: list[float]
        """
        # Find all tif-files in folder
        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        t = []
        for f in files:
            X = np.float32(io.imread(os.path.join(data_dir, f)))
            if f not in os.listdir(self.res_path):
                # Deconvolve image
                _,tx =self.predict_img(X, model_dir, f)

                # Store time required for deconvolution
                t.append(tx)
        return t

    def predict_img(self,X, model_dir, save_as=None):
        """
        Deconvolve image using Mu-Net model.

        :param X: Input image
        :type X: nd.array
        :param model_dir: Directory where Mu-Net model is stored
        :type model_dir: str
        :param save_as: File name under which to store the deconvolution result. If nothing is specified, it is not stored.
                        Defaults to None
        :type save_as: str, optional
        :return: Deconvolved image, time taken for deconvolution
        :rtype: nd.array, float
        """
        batch_sz = 1
        start = timeit.default_timer()
        self.denoiser.load_model(batch_sz, model_dir)

        # denoising process
        denoised_img = self.denoiser.denoising_img(X)
        t = timeit.default_timer()-start
        if save_as is not None:
            tif.imwrite(os.path.join(self.res_path, save_as), denoised_img)
        return denoised_img,t
