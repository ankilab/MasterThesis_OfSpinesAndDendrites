from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os
from functools import partial
import multiprocessing
from scipy.signal import convolve
from skimage.filters import gaussian
from skimage import io
import tifffile as tif
import deconv.utils as du


class Deconvolver:
    def __init__(self, args):
        self.data_path = os.path.join(os.getcwd(), args['data_path'])
        dir = os.path.join(os.getcwd(), self.data_path, args['result_path'])
        self.res_path = dir

        if not os.path.exists(dir):
            os.makedirs(dir)

    def preprocess(self, **kwargs):
        return NotImplementedError

    def train(self, **kwargs):
        return NotImplementedError

    def predict(self, **kwargs):
        return NotImplementedError

    def predict_img(self, **kwargs):
        return NotImplementedError


class BlindRL(Deconvolver):
    """
    Blind deconvolution based on:
    Fish, D. A., et al. "Blind deconvolution by means of the Richardson–Lucy algorithm." JOSA A 12.1 (1995): 58-65.
    """
    def __init__(self, args):
        super().__init__(args)
        self.psf_dir = args['psf']
        self.last_img = None

        # Required to reduce boundary artifacts
        self.pixels_padding = 10 if not 'pixels_padding' in args.keys() else args['pixels_padding']
        self.planes_padding = 5 if not 'planes_padding' in args.keys() else args['planes_padding']
        self._init_res_dict()

    def preprocess(self, img, sigma=1):
        smoothed = gaussian(img, sigma=sigma)
        return smoothed

    def train(self, **kwargs):
        pass

    def predict(self, data_dir, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=100,
                eval_img_steps=False, save_intermediate_res=False, parallel=True):
        """
        Iterate through folder and deconvolve all tif-images found
        :param data_dir: Directory with tif-files
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param eval_img_steps: Calculate image quality metrics after each iteration (not relevant here)
        :param save_intermediate_res: True, if results should be stored after each iteration (not relevant here)
        :param parallel: True, if as many processes as available cores should be launched
        :return:
        """

        self.data_path = data_dir
        self._init_res_dict()

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        l_file = len(files)

        # Determine amount of processes required
        n_cores = multiprocessing.cpu_count() if parallel else 1
        n_processes = 1
        if parallel:
            if l_file <= n_cores:
                n_processes = l_file
            else:
                n_processes = n_cores

        f_x = partial(self._process_img, n_iter_outer=n_iter_outer, n_iter_image=n_iter_image, n_iter_psf=n_iter_psf,
                      sigma=sigma, eval_img_steps=eval_img_steps, save_intermediate_res=save_intermediate_res,
                      plot_frequency=plot_frequency)

        # Launch processes
        with multiprocessing.Pool(processes=n_processes) as p:
            p.map(f_x, files)

    def _init_res_dict(self):
        self.res_dict = {}

    def _load_img(self, file_name):
        """
        Load image by filename and generate corresponding PSF
        :param file_name: Name of tif-file to be loaded
        :return: Image as np-array, corresponding PSF
        """
        # Get image
        X = np.float32(io.imread(os.path.join(self.data_path, file_name)))

        # Get PSF
        g = np.float32(self._get_psf(X.shape[1] + self.pixels_padding * 2, X.shape[0] + self.planes_padding * 2))
        return X, g

    def _process_img(self, file_name, n_iter_outer, n_iter_image, n_iter_psf, sigma,
                     eval_img_steps, save_intermediate_res, plot_frequency=100):
        """
        Process image: Load and deconvolve
        :param file_name:
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param save_intermediate_res: True, if results should be stored after each iteration (not relevant here)
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration (not relevant here)
        :return:
        """
        X, g = self._load_img(file_name)
        self.predict_img(X, g, n_iter_outer, n_iter_image, n_iter_psf, sigma, plot_frequency=plot_frequency,
                         eval_img_steps=eval_img_steps, save_intermediate_res=save_intermediate_res,
                         file_name=file_name)

    def predict_img(self, X, psf, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=0,
                    eval_img_steps=False, save_intermediate_res=False, file_name=''):
        """
        Deconvolve image
        :param X: Image (np.array)
        :param psf: PSF (np.array)
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration (not relevant here)
        :param save_intermediate_res: True, if results should be stored after each iteration (not relevant here)
        :param file_name:
        :return:
        """

        # Preprocessing
        X, g = self._constraints(X, psf)
        X_padded = self._pad(X, self.pixels_padding, self.planes_padding)
        X_smoothed = self.preprocess(X_padded, sigma=sigma)

        # Initial guess for object distribution
        # f = np.full(X_smoothed.shape, 0.5)
        f = X_smoothed.copy()
        psf = np.array(psf)
        epsilon = 1e-9  # Avoid division by 0

        # Blind RL iterations
        for k in range(n_iter_outer):

            # Save intermediate result
            if save_intermediate_res:
                self._save_res(f, psf, str(k)+str(n_iter_psf)+ str(n_iter_image), file_name)

            for i in range(n_iter_psf):  # m RL iterations, refining PSF
                psf = convolve((X_smoothed / (convolve(psf, f, mode='same') + epsilon)), f[::-1, ::-1, ::-1],
                               mode='same') * psf
            for i in range(n_iter_image):  # m RL iterations, refining reconstruction
                f = convolve((X_smoothed / (convolve(f, psf, mode='same') + epsilon)), psf[::-1, ::-1, ::-1],
                             mode='same') * f

            f, psf = self._constraints(f, psf)

            print(f'Image {file_name}, Iteration {k} completed.')
        f_unpad, psf_unpad = self._save_res(f, psf, str(n_iter_outer)+str(n_iter_psf)+ str(n_iter_image), file_name)
        return f_unpad, psf_unpad, None

    def _get_psf(self, size_xy, size_z):
        """
        Load PSF-file and extract relevant z-planes
        :param size_xy: Size in X and Y- direction (after padding)
        :param size_z: Size in Z direction (after padding)
        :return: PSF
        """
        xy = 552
        z = 150
        g = du.read_psf_file(xy, z, self.psf_dir)

        # Initial guess for PSF
        offset = int((z - size_z) / 2)
        offset_xy = int((xy - size_xy) / 2)
        psf = g[offset:g.shape[0] - offset, offset_xy:g.shape[1] - offset_xy, offset_xy:g.shape[2] - offset_xy] \
            if size_z % 2 == 0 else \
            g[offset+1:g.shape[0] - offset, offset_xy:g.shape[1] - offset_xy, offset_xy:g.shape[2] - offset_xy]
        # psf= psf**2
        return psf


    def _save_res(self, f, psf, iteration, f_name='x.tif'):
        f_unpad = self._unpad(f, self.pixels_padding, self.planes_padding)
        psf_unpad = self._unpad(psf, self.pixels_padding, self.planes_padding)
        name_img = os.path.join(self.res_path, iteration + f_name)
        tif.imsave(name_img, f_unpad)
        name_psf = os.path.join(self.res_path, iteration + f_name[:-4] + '_psf.tif')
        tif.imsave(name_psf, psf_unpad)
        return f_unpad, psf_unpad

    def _constraints(self, f, psf):
        """
        Constraints to enable and imporve deconvolution
        :param f: Image (np.array)
        :param psf: PSF (np.array)
        :return: Adjusted image and PSF
        """

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

    def _pad(self, img, pixels=20, planes=10):
        """
        Pad image
        :param img: Original image (np.array)
        :param pixels: amount to pad in x and y direction
        :param planes: amount to pad in z direction
        :return: padded image
        """
        return np.pad(img, ((planes, planes), (pixels, pixels), (pixels, pixels)), 'reflect')

    def _unpad(self, img, pixels=20, planes=10):
        '''
        Crop the image by the number of pixels specified in x and y direction, by the amount of planes in z direction.
        :param img: Input image
        :param pixels: Image is reduced by the number of pixels specified in x and y direction on both sides
        :param planes: Image is reduced by the number of planes specified in z direction on both sides
        :return: Cropped image
        '''

        pixels = int(pixels)
        planes = int(planes)
        (z, x, y) = img.shape
        return img[planes:z - planes, pixels:x - pixels, pixels:y - pixels]



