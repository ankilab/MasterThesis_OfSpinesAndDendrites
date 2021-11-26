from .deconvolver import Deconvolver
from denoising_ae.auto_encoder_net import Autoencoder
import data_augmentation as da
import numpy as np
from skimage import io
import os
import tifffile as tif


class AE(Deconvolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = Autoencoder(args)
        self.data_provider = da.DataProvider((args['z_shape'], args['xy_shape']),data_file='data32_128.h5')

    def preprocess(self):
        pass

    def train(self, epochs=10, batch_size=4):
        model_path, train_hist = self.model.train(self.data_provider, epochs, batch_size)
        return model_path, train_hist

    def predict(self, data_dir, model_dir=None):
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

    def predict_img(self, img, sampling_step, file_name=None, model_dir=None):
        if model_dir is not None:
            self.model.load_model(model_dir)

        # denoising process
        pred_img = self.model.predict(img, sampling_step)

        if file_name is not None:
            tif.imsave(os.path.join(self.res_path, file_name), pred_img)
        return pred_img
