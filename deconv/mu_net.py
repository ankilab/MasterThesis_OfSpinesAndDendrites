from .deconvolver import Deconvolver
from mu_net1 import denoiser_only_mu as den
import tifffile as tif

class Mu_Net(Deconvolver):
    def __init__(self, args):

        super().__init__(args)
        self.train_flag = args['train']
        self.denoiser = den.Denoiser(args)

    def preprocess(self):
        pass

    def train(self, epochs=10, batch_size=8):
        self.denoiser.train(epochs=epochs)

    def predict(self, X, model_dir, save_as=None):
        """

        :param X: Input image
        :param model_dir:
        :param save_as: If not None, result is saved as file with the name specified (e.g. 'denoised_img.tif')
        :return:
        """
        batch_sz = 1
        self.denoiser.load_model(batch_sz)

        # denoising process
        denoised_img = self.denoiser.denoising_img(X)

        if save_as is not None:
            tif.imsave(save_as, denoised_img.astype('uint16'))
        return denoised_img