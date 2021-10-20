from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # or 1 or 2 or 3
# from csbdeep.utils import axes_dict, plot_some, plot_history
# from csbdeep.io import load_training_data
# from csbdeep.models import Config, CARE
# from csbdeep.utils import normalize
import json
# import tensorflow as tf
# import fbpconvnet_pytorch as fbp
from scipy.signal import fftconvolve, convolve
from skimage.filters import gaussian
from skimage import io
import timeit
import imagequalitymetrics
import yaml
# import imagej
import tifffile as tif
import pickle
# from numba import jit
# import cupy as cp
# import cupyx.scipy.signal as csig

from mu_net import denoiser as den
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# sess.as_default()


class Deconvolver:
    def __init__(self, args):
        self.data_path = os.path.join(os.getcwd(),args['data_path'])
        dir = os.path.join(os.getcwd(), self.data_path, args['result_path'])
        self.res_path = dir

        if not os.path.exists(dir):
            os.makedirs(dir)

        # TODO: LOgger, log results, intermediate results for Blind RL

    def preprocess(self, **kwargs):
        return NotImplementedError

    def train(self, **kwargs):
        return NotImplementedError

    def predict(self, **kwargs):
        return NotImplementedError

    def predict_img(self, **kwargs):
        return NotImplementedError


class BlindRL(Deconvolver):
    def __init__(self, args):
        super().__init__(args)
        self.psf_dir = args['psf']
        self.last_img = None

    def preprocess(self, img, sigma= 1):
        smoothed = gaussian(img, sigma=sigma)
        return smoothed

    def train(self, **kwargs):
        pass

    def predict(self, data_dir, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=1,
                eval_img_steps=False, save_intermediate_res = False, pixels_padding =20, planes_padding=20):
        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        res_dict = {}
        res_dict['n_iter_outer'] = n_iter_outer
        res_dict['n_iter_image'] = n_iter_image
        res_dict['n_iter_psf'] = n_iter_psf
        res_dict['sigma'] = sigma
        res_dict['Runtime_per_image'] = []

        for (idx, nm) in enumerate(files):
            X = io.imread(os.path.join(data_dir,nm))

            g = self._get_psf(X.shape[1] + pixels_padding * 2, X.shape[0] + planes_padding * 2)
            self.last_img = nm
            start = timeit.default_timer()
            f, psf, metrics = self.predict_img(X, g, n_iter_outer, n_iter_image, n_iter_psf, sigma, plot_frequency,
                                               eval_img_steps, save_intermediate_res=save_intermediate_res,
                                               pixels_padding=pixels_padding, planes_padding=planes_padding)
            stop = timeit.default_timer()

            # Save results
            res_dict['Runtime_per_image'].append(stop-start)
            res_dict[nm[:-4]] = metrics
            name = os.path.join(self.res_path,'iter_' + str(n_iter_outer)+ nm)
            tif.imsave(name, f)
            tif.imsave(name[:-4] + '_psf.tif',psf)

        with open(os.path.join(self.res_path,f'results_{n_iter_outer}_{n_iter_image}_{n_iter_psf}_{sigma}.pkl'), 'wb') \
                as outfile:
            pickle.dump(res_dict, outfile, pickle.HIGHEST_PROTOCOL)

    def predict_img(self, X, psf, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma = 1, plot_frequency =1,
                    eval_img_steps = False, save_intermediate_res = False, pixels_padding = 20, planes_padding=20):
        X, g = self._constraints(X, psf)
        X_padded = self._pad(X, pixels_padding, planes_padding)
        X_smoothed = self.preprocess(X_padded, sigma=sigma)

        # Initial guess for object distribution
        f = np.full(X_smoothed.shape, 0.5)
        psf = np.array(psf)
        epsilon = 1e-9  # Avoid division by 0
        met = imagequalitymetrics.ImageQualityMetrics()
        res = {}
        res['brisque'] = []
        res['snr'] = []
        res['brisque_img_steps'] = []
        res['snr_img_steps'] = []

        # Blind RL iterations
        for k in range(n_iter_outer):

            # Save intermediate result
            if save_intermediate_res:
                f_unpad = self._unpad(f, pixels_padding, planes_padding)
                psf_unpad = self._unpad(psf, pixels_padding, planes_padding)
                name_img = os.path.join(self.res_path,'iter_' + str(k) + self.last_img)
                tif.imsave(name_img, f_unpad)
                name_psf = os.path.join(self.res_path, 'iter_' + str(k) +self.last_img[:-4] +'_psf.tif')
                tif.imsave(name_psf, psf_unpad)

            for i in range(n_iter_psf):  # m RL iterations, refining PSF
                psf = fftconvolve((X_smoothed / (convolve(psf, f, mode='same') + epsilon)), f[::-1, ::-1, ::-1],
                             mode='same') * psf
            for i in range(n_iter_image):  # m RL iterations, refining reconstruction
                f = fftconvolve((X_smoothed / (convolve(f, psf, mode='same') + epsilon)), psf[::-1, ::-1, ::-1],
                             mode='same') * f

                if eval_img_steps:
                    f_1, psf_1 = self._constraints(f, psf)
                    res['brisque_img_steps'].append(met.brisque(f_1))
                    res['snr_img_steps'].append(met.snr(f_1))

            f, psf = self._constraints(f, psf)

            # Evaluate intermediate result
            res['brisque'].append(met.brisque(f))
            res['snr'].append(met.snr(f))

            # Plot intermediate result
            if n_iter_outer % plot_frequency == 0:
                plt.figure()
                plt.imshow(f[11, :, :])
                plt.title(k)
                plt.show()
        f_unpad = self._unpad(f, pixels_padding, planes_padding)
        psf_unpad = self._unpad(psf, pixels_padding, planes_padding)
        return f_unpad, psf_unpad, res

    def _get_psf(self,size_xy, size_z):
        z = 150 if size_z%2 ==0 else 151

        psf_file = 'PSF_' + str(size_xy) + '_' + str(z) + '.tif'

        # Initial guess for PSF
        g = io.imread(os.path.join(self.psf_dir, psf_file), plugin='pil')
        offset = int((z-size_z)/2)
        return g[offset:g.shape[0]-offset, :,:]

    def _constraints(self, f, psf):
        # Non-negativity
        f[(f < 0)] = 0
        psf[psf < 0] = 0

        # Avoid overflow
        f[(f < 1e-100)] = 0
        psf[(psf < 1e-100)] = 0

        s = np.append(f, psf)
        m = np.min(s)
        p = np.ptp(s)
        f = (f - m) / p

        # Unit summation of PSF
        psf /= np.sum(psf)

        return f, psf

    def _pad(self, img, pixels = 20, planes=20):
        (z, x, y) = img.shape
        padded = np.pad(img, ((planes, planes), (pixels, pixels), (pixels, pixels)), 'reflect')
        # padded = np.zeros((z+2*planes, x+2*pixels, y+2*pixels))
        # padded[planes:planes+z, pixels:x+pixels, pixels:y+pixels] = img
        return padded

    def _unpad(self, img, pixels=20 , planes=20):
        '''
        Crop the image by the number of pixels specified in x and y direction, by the amount of planes in z direction.
        :param img: Input image
        :param pixels: Image is reduced by the number of pixels specified in x and y direction on both sides
        :param planes: Image is reduced by the number of planes specified in z direction on both sides
        :return: Cropped image
        '''

        pixels = int(pixels)
        plane = int(planes)
        (z,x,y) = img.shape
        return img[planes:z-planes, pixels:x-pixels, pixels:y-pixels]

    #
    # def _generate_psf(self):
    #     ij = imagej.init('C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation/Fiji.app', headless = False)
    #     ij.ui().showUI()
    #     n = 10
    #     plugin = 'PSFGenerator'
    #     args = {
    #         'block_radius_x': 100,
    #         'block_radius_y': 100
    #     }
    #     ij.py.run_plugin(plugin, args)
    #
    #     pass

# class CAREDeconv(Deconvolver):
#     def __init__(self, args):
#         super().__init__(args)
#         self.train_flag = args['train']
#
#     def preprocess(self):
#         pass
#
#     def train(self, data_dir, validation_split =0.1, epochs =10, batch_size=8, train_steps=50):
#         (X, Y), (X_val, Y_val), axes = load_training_data(data_dir,
#                                                           validation_split=validation_split, verbose=True)
#         c = axes_dict(axes)['C']
#         n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
#
#         config = Config(axes, n_channel_in, n_channel_out, train_batch_size=batch_size,
#                         train_steps_per_epoch=train_steps, train_epochs=epochs)
#         model_dir = os.path.join(self.res_path,'models')
#         model = CARE(config, 'my_model', basedir=model_dir)
#         print(model.keras_model.summary())
#
#         history = model.train(X, Y, validation_data=(X_val, Y_val))
#
#         # Save training history
#         history.history['lr'] = [float(f) for f in history.history['lr']]
#         res_dict = dict(history.history, **history.params)
#         r = os.path.join(self.res_path,'result.json')
#         json.dump(res_dict, open(r, 'w'))
#
#         #Save model
#         mdl_path = os.path.join(self.res_path, 'TF_SavedModel.zip')
#         model.export_TF(mdl_path)
#
#         plt.figure(figsize=(16, 5))
#         plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
#
#         plt.figure(figsize=(20, 12))
#         _P = model.keras_model.predict(X_val[-5:])
#         if config.probabilistic:
#             _P = _P[..., :(_P.shape[-1] // 2)]
#         plot_some(X_val[-5:], Y_val[-5:], _P, pmax=99.5)
#         plt.suptitle('5 example validation patches\n'
#                      'top row: input (source),  '
#                      'middle row: target (ground truth),  '
#                      'bottom row: predicted from source')
#         plt.show()
#         return model_dir, mdl_path
#
#     def predict(self, X, model_dir, Y= None, plot =False):
#
#         axes = 'YX'
#
#         model = CARE(config=None, name='my_model', basedir='models')
#         restored = model.predict(X, axes)
#
#         if not(Y is None) and plot:
#             plt.figure(figsize=(15, 10))
#             plot_some(np.stack([X, restored, Y]),
#                       title_list=[['low', 'CARE', 'GT']],
#                       pmin=2, pmax=99.8)
#
#             plt.figure(figsize=(10, 5))
#             for _x, _name in zip((X, restored, Y), ('low', 'CARE', 'GT')):
#                 plt.plot(normalize(_x, 1, 99.7)[180], label=_name, lw=2)
#             plt.legend()
#             plt.show()
#         return restored

class Mu_Net(Deconvolver):
    def __init__(self, args):
        super().__init__(args)
        self.train_flag = args['train']
        self.denoiser = den.Denoiser(args)

    def preprocess(self):
        return NotImplementedError

    def train(self, data_dir, validation_split =0.1, epochs =10, batch_size=8, train_steps=50):
        self.denoiser.train()

    def predict(self, X, model_dir, Y= None, plot =False):
        batch_sz = 1
        self.denoiser.load_model(batch_sz)

        # read image
        # dir_path = './example_data/small/'
        # noise_level = 1 # from 1 to 4
        # img_name = '%s/n%d_000001.tif' % (dir_path,noise_level)

        # denoising process
        L0_pred, L1_pred, L2_pred, denoised_img = self.denoiser.denoising_patch(X)
        tif.imsave('L0_pred.tif', L0_pred.astype('uint16'))
        tif.imsave('L1_pred.tif', L1_pred.astype('uint16'))
        tif.imsave('L2_pred.tif', L2_pred.astype('uint16'))
        tif.imsave('denoised_img.tif', denoised_img.astype('uint16'))


# class FBP_ConvNet(Deconvolver):
#     def __init__(self, args):
#         super().__init__(args)
#
#     def preprocess(self):
#         return self.data_path
#
#     def train(self, data_dir, validation_split=0.1, epochs =10, batch_size=8, train_steps=50):
#         fbp.train.main(config)
#         return 0
#
#     def predict(self, X, model_dir, Y= None, plot =False):
#         return 0


REGISTRY = {}
# REGISTRY['csbdeep'] = CAREDeconv
# REGISTRY['mu-net'] = Mu_Net
# REGISTRY['fbpconvnet'] =FBP_ConvNet