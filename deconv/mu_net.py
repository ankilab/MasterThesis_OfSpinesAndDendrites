from .deconvolver import Deconvolver
from mu_net1 import denoiser_only_mu as den
import tifffile as tif
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import pickle


class Mu_Net(Deconvolver):
    def __init__(self, args):

        super().__init__(args)
        self.train_flag = args['train']
        self.denoiser = den.Denoiser(args)

    def preprocess(self):
        pass

    def train(self, data_provider, epochs=10, batch_size=8):
        model_dir, train_hist= self.denoiser.train(data_provider, epochs=epochs)
        with open(os.path.join(self.res_path, 'history_mu_net.pkl'), 'wb') as outfile:
            pickle.dump(train_hist, outfile, pickle.HIGHEST_PROTOCOL)
        return model_dir, train_hist

    def plot_training(self, train_history):
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

        :param X: Input image
        :param model_dir:
        :param save_as: If not None, result is saved as file with the name specified (e.g. 'denoised_img.tif')
        :return:
        """
        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        for f in files:
            X = np.float32(io.imread(os.path.join(data_dir, f)))
            self.predict_img(X, model_dir, f)

    def predict_img(self,X, model_dir, save_as='Mu_Net_res.tif'):
        batch_sz = 1
        self.denoiser.load_model(batch_sz, model_dir)

        # denoising process
        denoised_img = self.denoiser.denoising_img(X)

        if save_as is not None:
            tif.imsave(os.path.join(self.res_path, save_as), denoised_img)
        return denoised_img
