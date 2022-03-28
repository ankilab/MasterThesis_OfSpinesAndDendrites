from __future__ import print_function, unicode_literals, absolute_import, division

import timeit

import numpy as np
import os
from functools import partial
import multiprocessing
from scipy.signal import convolve
# from skimage.filters import gaussian
from denoising import GaussianFilter
from skimage import io
import tifffile as tif
import deconv.utils as du
from .deconvolver import Deconvolver


class BlindRL(Deconvolver):
    """
    Blind deconvolution based on:
    Fish, D. A., et al. "Blind deconvolution by means of the Richardson–Lucy algorithm." JOSA A 12.1 (1995): 58-65.
    """

    def __init__(self, args):
        """
        Initialize an object providing functionality to deconvolve images using blind Richardson–Lucy.

        :param args: args['psf']: PSF file location, args['result_path']: file location where results are to be stored
                    optional - args['pixels_padding'], args['planes_padding']: specify how many pixels ans planes to pad the input
        :type args: dict
        """
        super().__init__(args)
        self.psf_dir = args['psf']
        self.last_img = None

        # Required to reduce boundary artifacts
        self.pixels_padding = 10 if not 'pixels_padding' in args.keys() else args['pixels_padding']
        self.planes_padding = 5 if not 'planes_padding' in args.keys() else args['planes_padding']

    def preprocess(self, img, sigma=1):
        """
        Preprocess image using Gaussian filter

        :param img: Input image
        :type img: nd.array
        :param sigma: Standard deviation of Gaussian filter, defaults to 1
        :type sigma: float, optional
        :return: Preprocessed image
        :rtype: nd.array
        """
        den = GaussianFilter()
        return den.denoise(img, sigma)

    def train(self,**kwargs):
        """
        Blind RL training: None required.
        """
        pass

    def predict(self, data_dir, n_iter_outer=5, n_iter_image=1, n_iter_psf=3, sigma=1,
                save_intermediate_res=False, parallel=True, n_processes=1):
        """
        Iterate through folder and deconvolve all tif-images found. The results are stored in the folder specified at
        object initialization. It can be adjusted by setting "obj.res_path = './'".

        :param data_dir: Directory with tif-files
        :type data_dir: string
        :param n_iter_outer: RL-iterations, defaults to 5
        :type n_iter_outer: int, optional
        :param n_iter_image: Convolution iterations on image, defaults to 1
        :type n_iter_image: int, optional
        :param n_iter_psf: Convolution iterations on psf, defaults to 3
        :type n_iter_psf: int, optional
        :param sigma: Gaussian-smoothing parameter, defaults to 1
        :type sigma: float, optional
        :param save_intermediate_res: True, if results should be stored after each iteration, defaults to False
        :type save_intermediate_res: bool, optional
        :param parallel: True, if as many processes as available cores should be launched
        :param n_processes: Number of processes to run in parallel
        """

        self.data_path = data_dir

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        l_file = len(files)

        # Determine amount of processes required
        n_cores = multiprocessing.cpu_count() if parallel else 1
        if parallel:
            if l_file <= n_cores:
                n_processes = l_file
            else:
                n_processes = n_cores

        # Parallel processing using specified number of processes
        if n_processes>1:
            f_x = partial(self.process_img, n_iter_outer=n_iter_outer, n_iter_image=n_iter_image, n_iter_psf=n_iter_psf,
                          sigma=sigma, save_intermediate_res=save_intermediate_res)

            # Launch processes
            with multiprocessing.Pool(processes=n_processes) as p:
                p.map(f_x, files)

        # Sequential Processing
        else:
            print('Executes sequential implementation as only one process is launched to reduce overhead.')
            self.predict_seq(data_dir, n_iter_outer, n_iter_image, n_iter_psf, sigma, save_intermediate_res)

    def predict_seq(self, data_dir, n_iter_outer=5, n_iter_image=1, n_iter_psf=3, sigma=1, save_intermediate_res=False):
        """
        Iterate through folder and deconvolve all tif-images found sequentially. The results are stored in the folder specified at
        object initialization. It can be adjusted by setting "obj.res_path = './'".

        :param data_dir: Directory with tif-files
        :type data_dir: string
        :param n_iter_outer: RL-iterations, defaults to 5
        :type n_iter_outer: int, optional
        :param n_iter_image: Convolution iterations on image, defaults to 1
        :type n_iter_image: int, optional
        :param n_iter_psf: Convolution iterations on psf, defaults to 3
        :type n_iter_psf: int, optional
        :param sigma: Gaussian-smoothing parameter, defaults to 1
        :type sigma: float, optional
        :param save_intermediate_res: True, if results should be stored after each iteration, defaults to False
        :type save_intermediate_res: bool, optional
        :return: Time required for deconvolution for each image
        :rtype: list[float]
        """

        self.data_path = data_dir

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

        timing = []
        for f in files:
            start = timeit.default_timer()
            self.process_img(f, n_iter_outer=n_iter_outer, n_iter_image=n_iter_image, n_iter_psf=n_iter_psf,
                             sigma=sigma, save_intermediate_res=save_intermediate_res)
            timing.append(timeit.default_timer()-start)
        return timing

    def _load_img(self, file_name):
        """
        Load image by filename and generate corresponding PSF.

        :param file_name: Name of tif-file to be loaded
        :type file_name: string
        :return: Image, corresponding PSF
        :rtype: nd.array, nd.array
        """
        # Get image
        X = np.float32(io.imread(os.path.join(self.data_path, file_name)))

        # Get PSF
        g = np.float32(self._get_psf(X.shape[1] + self.pixels_padding * 2, X.shape[0] + self.planes_padding * 2))
        return X, g

    def process_img(self, file_name, n_iter_outer=5, n_iter_image=1, n_iter_psf=3, sigma=1, save_intermediate_res=False,
                    output_name=None):
        """
        Process image: Load and deconvolve the image specified in file name

        :param file_name: Location of input file
        :type file_name: string
        :param n_iter_outer: RL-iterations, defaults to 5
        :type n_iter_outer: int, optional
        :param n_iter_image: Convolution iterations on image, defaults to 1
        :type n_iter_image: int, optional
        :param n_iter_psf: Convolution iterations on psf, defaults to 3
        :type n_iter_psf: int, optional
        :param sigma: Gaussian-smoothing parameter, defaults to 1
        :type sigma: float, optional
        :param save_intermediate_res: True, if results should be stored after each iteration, defaults to False
        :type save_intermediate_res: bool, optional
        :param output_name: Name of file the processed image is to be stored, defaults to None (stored as input file name)
        :type sigma: string, optional
        """
        X, g = self._load_img(file_name)
        if output_name is None:
            output_name=file_name
        self.predict_img(X, g, n_iter_outer, n_iter_image, n_iter_psf, sigma,
                         save_intermediate_res=save_intermediate_res,
                         file_name=output_name)

    def predict_img(self, X, psf, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, save_intermediate_res=False,
                    file_name=''):
        """
        Deconvolve image.

        :param X: Input image
        :type X: nd.array
        :param psf: PSF
        :type psf: nd.array
        :param n_iter_outer: RL-iterations, defaults to 5
        :type n_iter_outer: int, optional
        :param n_iter_image: Convolution iterations on image, defaults to 1
        :type n_iter_image: int, optional
        :param n_iter_psf: Convolution iterations on psf, defaults to 3
        :type n_iter_psf: int, optional
        :param sigma: Gaussian-smoothing parameter, defaults to 1
        :type sigma: float, optional
        :param save_intermediate_res: True, if results should be stored after each iteration, defaults to False
        :type save_intermediate_res: bool, optional
        :param file_name: File name used to store the deconvolution result
        :type file_name: string, optional
        :return: Deconvolved image, estimated PSF, None
        :rtype: nd.array, nd.array, NoneType
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
                self._save_res(f, psf, str(k) + str(n_iter_psf) + str(n_iter_image), file_name)

            for i in range(n_iter_psf):  # m RL iterations, refining PSF
                psf = convolve((X_smoothed / (convolve(psf, f, mode='same') + epsilon)), f[::-1, ::-1, ::-1],
                               mode='same') * psf
            for i in range(n_iter_image):  # m RL iterations, refining reconstruction
                f = convolve((X_smoothed / (convolve(f, psf, mode='same') + epsilon)), psf[::-1, ::-1, ::-1],
                             mode='same') * f

            f, psf = self._constraints(f, psf)

            print(f'Image {file_name}, Iteration {k+1} completed.')
        f_unpad, psf_unpad = self._save_res(f, psf, str(n_iter_outer) + str(n_iter_psf) + str(n_iter_image), file_name)
        return f_unpad, psf_unpad, None

    def _get_psf(self, size_xy, size_z):
        """
        Load PSF-file and extract relevant z-planes

        :param size_xy: Size in X and Y- direction (after padding)
        :type size_xy: int
        :param size_z: Size in Z direction (after padding)
        :type size_z: int
        :return: PSF
        :rtype: nd.array
        """
        xy = 552
        z = 150
        g = du.read_psf_file(xy, z, self.psf_dir)

        # Initial guess for PSF
        # Get relevant part of PSF according to image size
        offset = int((z - size_z) / 2)
        offset_xy = int((xy - size_xy) / 2)
        psf = g[offset:g.shape[0] - offset, offset_xy:g.shape[1] - offset_xy, offset_xy:g.shape[2] - offset_xy] \
            if size_z % 2 == 0 else \
            g[offset + 1:g.shape[0] - offset, offset_xy:g.shape[1] - offset_xy, offset_xy:g.shape[2] - offset_xy]
        # psf= psf**2
        return psf

    def _save_res(self, f, psf, iteration, f_name='x.tif'):
        """
        Save deconvolution result.

        :param f: Computed image
        :type f: nd.array
        :param psf: Computed PSF
        :type psf: nd.array
        :param iteration: Specifies after which number of iterations the result was obtained
        :type iteration: string
        :param f_name: File name for results to store as, defaults to 'x.tif'
        :type f_name: string, optional
        :return: input image and PSF shrinked to original size (without padding)
        :rtype: nd.array, nd.array
        """
        # Cut image and PSF to get size before padding
        f_unpad = self._unpad(f, self.pixels_padding, self.planes_padding)
        psf_unpad = self._unpad(psf, self.pixels_padding, self.planes_padding)

        # Save results to file
        name_img = os.path.join(self.res_path, iteration + f_name)
        tif.imwrite(name_img, f_unpad)
        name_psf = os.path.join(self.res_path, iteration + f_name[:-4] + '_psf.tif')
        tif.imwrite(name_psf, psf_unpad)
        return f_unpad, psf_unpad

    def _constraints(self, f, psf):
        """
        Constraints to enable and imporve deconvolution

        :param f: Image
        :type f: nd.array
        :param psf: PSF
        :type psf: nd.array
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
        Pad image.

        :param img: Original image
        :type img: nd.array
        :param pixels: amount to pad in x and y direction, defaults to 20
        :param planes: amount to pad in z direction, defaults to 10
        :return: padded image
        """
        return np.pad(img, ((planes, planes), (pixels, pixels), (pixels, pixels)), 'reflect')

    def _unpad(self, img, pixels=20, planes=10):
        '''
        Crop the image by the number of pixels specified in x and y direction, by the amount of planes in z direction.

        :param img: Input image
        :type img: nd.array
        :param pixels: Image is reduced by the number of pixels specified in x and y direction on both sides,
        defaults to 20
        :type pixels: int, optional
        :param planes: Image is reduced by the number of planes specified in z direction on both sides, defaults to 10
        :type planes: int, optional
        :return: Cropped image
        '''

        pixels = int(pixels)
        planes = int(planes)
        (z, x, y) = img.shape
        return img[planes:z - planes, pixels:x - pixels, pixels:y - pixels]
