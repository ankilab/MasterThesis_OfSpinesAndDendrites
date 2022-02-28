from .deconvolver import Deconvolver
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import partial
import multiprocessing
from scipy.signal import fftconvolve, convolve
from skimage.filters import gaussian
from skimage import io
import timeit
import imagequalitymetrics
import tifffile as tif
import pickle
#from deconv.wb_backprojector import get_BackProjector
# from wb_backprojector import _get_psf_x
# from .utils import c_convolve

MAX_VAL = 12870
MIN_VAL = -2327


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
        self.pixels_padding = 20 if not 'pixels_padding' in args.keys() else args['pixels_padding']
        self.planes_padding = 10 if not 'planes_padding' in args.keys() else args['planes_padding']
        self._init_res_dict()

    def _rescale(self, img, min_val = None, max_val = None):
        """
        Rescale image to range of [0,1]

        :param img: 
        :return:
        """
        min_v=MIN_VAL if min_val is None else min_val
        max_v = MAX_VAL if max_val is None else max_val
        img = (img - min_v)
        return img / max_v

    def preprocess(self, img, sigma=1):

        img = gaussian(img, sigma=sigma)
        return img

    def train(self, **kwargs):
        pass

    def predict(self, data_dir, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=100,
                eval_img_steps=False, save_intermediate_res=False, parallel=True, preprocess=False):
        """
        Iterate through folder and deconvolve all tif-images found.

        :param data_dir: Directory with tif-files
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermdeiate results be plotted? If 1, after every iteration
        :param eval_img_steps: Calculate image quality metrics after each iteration
        :param save_intermediate_res: True, if results should be stored after each iteration
        :param parallel: True, if as many processes as available cores should be launched
        :return:
        """

        self.data_path = data_dir
        self._init_res_dict()

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        l_file = len(files)
        self.res_dict['n_iter_outer'] = n_iter_outer
        self.res_dict['n_iter_image'] = n_iter_image
        self.res_dict['n_iter_psf'] = n_iter_psf
        self.res_dict['sigma'] = sigma

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
                      plot_frequency=plot_frequency, preprocess=preprocess)

        with multiprocessing.Pool(processes=n_processes) as p:
            p.map(f_x, files)

        # Save results
        with open(
                os.path.join(self.res_path, f'results_blind_rl_{n_iter_outer}_{n_iter_image}_{n_iter_psf}_{sigma}.pkl'),
                'wb') \
                as outfile:
            pickle.dump(self.res_dict, outfile, pickle.HIGHEST_PROTOCOL)

    def predict_seq(self, data_dir, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=100,
                eval_img_steps=False, save_intermediate_res=False, parallel=True, preprocess=False):
        """
        Iterate through folder and deconvolve all tif-images found.

        :param data_dir: Directory with tif-files
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermdeiate results be plotted? If 1, after every iteration
        :param eval_img_steps: Calculate image quality metrics after each iteration
        :param save_intermediate_res: True, if results should be stored after each iteration
        :param parallel: True, if as many processes as available cores should be launched
        :return:
        """

        self.data_path = data_dir
        self._init_res_dict()

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        self.res_dict['n_iter_outer'] = n_iter_outer
        self.res_dict['n_iter_image'] = n_iter_image
        self.res_dict['n_iter_psf'] = n_iter_psf
        self.res_dict['sigma'] = sigma

        for f in files:
            self._process_img(f, n_iter_outer=n_iter_outer, n_iter_image=n_iter_image, n_iter_psf=n_iter_psf,
                      sigma=sigma, eval_img_steps=eval_img_steps, save_intermediate_res=save_intermediate_res,
                      plot_frequency=plot_frequency, preprocess=preprocess)

        # Save results
        with open(
                os.path.join(self.res_path, f'results_blind_rl_{n_iter_outer}_{n_iter_image}_{n_iter_psf}_{sigma}.pkl'),
                'wb') \
                as outfile:
            pickle.dump(self.res_dict, outfile, pickle.HIGHEST_PROTOCOL)

    # def predict_gpu(self, data_dir, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=100,
    #                 eval_img_steps=False, save_intermediate_res=False, parallel=False):
    #
    #     self.data_path = data_dir
    #     self._init_res_dict()
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
    #     self.res_dict['n_iter_outer'] = n_iter_outer
    #     self.res_dict['n_iter_image'] = n_iter_image
    #     self.res_dict['n_iter_psf'] = n_iter_psf
    #     self.res_dict['sigma'] = sigma
    #     self.res_dict['Runtime_per_image'] = []
    #
    #     for file in files:
    #         X, g = self._load_img(file)
    #         X_padded = self._pad(X, self.pixels_padding, self.planes_padding)
    #         X = self.preprocess(X_padded, sigma=sigma)
    #
    #         Xc = torch.tensor(X[None,None,...], device=device)
    #         psf = torch.tensor(g[None,None,...], device=device)
    #         Xc, psf = self._constraints(Xc, psf)
    #
    #         start = timeit.default_timer()
    #
    #         # Initial guess for object distribution
    #         f = torch.full_like(Xc, 0.5, dtype=torch.float32, device=device)
    #         epsilon = 1e-9  # Avoid division by 0
    #         met = imagequalitymetrics.ImageQualityMetrics()
    #         res = {}
    #         res['brisque'] = []
    #         res['snr'] = []
    #         res['brisque_img_steps'] = []
    #         res['snr_img_steps'] = []
    #
    #         # Blind RL iterations
    #         for k in range(n_iter_outer):
    #             # Save intermediate result
    #             if save_intermediate_res:
    #                 self._save_res(f.detach().cpu().numpy(), psf.detach().cpu().numpy(), k, file)
    #
    #             for i in range(n_iter_psf):  # m RL iterations, refining PSF
    #                 psf = F.conv3d((Xc / (F.conv3d(psf, f, padding='same') + epsilon)), torch.flip(f,[2,3,4]),
    #                                        padding='same') * psf
    #                 print('Here 1')
    #             for i in range(n_iter_image):  # m RL iterations, refining reconstruction
    #                 f = F.conv3d((Xc / (F.conv3d(f, psf, padding='same') + epsilon)), torch.flip(psf,[2,3,4]),
    #                                      padding='same') * f
    #
    #                 if eval_img_steps:
    #                     f_1, psf_1 = self._constraints(f, psf)
    #                     res['brisque_img_steps'].append(met.brisque(f_1))
    #                     res['snr_img_steps'].append(met.snr(f_1))
    #                 print('Here 2')
    #
    #             f, psf = self._constraints(f, psf)
    #             # Evaluate intermediate result
    #             res['brisque'].append(met.brisque(f))
    #             res['snr'].append(met.snr(f))
    #
    #             print(f'{file}, Iteration {k} completed.')
    #         stop = timeit.default_timer()
    #         res['Runtime'].append(stop - start)
    #         self.res_dict[file[:-4]] = res
    #         self._save_res(f.detach().cpu().numpy(), psf.detach().cpu().numpy(), n_iter_outer, file, sigma)
    #
    #     # Save results
    #     with open(os.path.join(self.res_path, f'results_{n_iter_outer}_{n_iter_image}_{n_iter_psf}_{sigma}.pkl'), 'wb') \
    #             as outfile:
    #         pickle.dump(self.res_dict, outfile, pickle.HIGHEST_PROTOCOL)

    def _init_res_dict(self):
        self.res_dict = {}

    def _load_img(self, file_name):
        """
        Load image by filename and generate corresponding PSF

        :param file_name: Name of tif-file to be loaded
        :return: Image as np-array, corresponding PSF
        """
        X = np.float32(io.imread(os.path.join(self.data_path, file_name)))
        g = np.float32(self._get_psf(X.shape[1] + self.pixels_padding * 2, X.shape[0] + self.planes_padding * 2))
        # g = np.float32(self._get_psf(X.shape[1], X.shape[0]))
        return X, g

    def _process_img(self, file_name, n_iter_outer, n_iter_image, n_iter_psf, sigma,
                     eval_img_steps, save_intermediate_res, plot_frequency=100, preprocess=False):
        """
        Process image: Load and deconvolve.

        :param file_name:
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param eval_img_steps: Calculate image quality metrics after each iteration (not relevant here)
        :param save_intermediate_res: True, if results should be stored after each iteration (not relevant here)
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :return:
        """
        X, g = self._load_img(file_name)
        self.predict_img(X, g, n_iter_outer, n_iter_image, n_iter_psf, sigma, plot_frequency=plot_frequency,
                         eval_img_steps=eval_img_steps, save_intermediate_res=save_intermediate_res,
                         file_name=file_name, preprocess=preprocess)

    def predict_img(self, X, psf, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=0,
                    eval_img_steps=False, save_intermediate_res=False, file_name='', preprocess=False):
        """
        Deconvolve image
        :param X: Image (np.array)
        :param psf: PSF (np.array)
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param save_intermediate_res: True, if results should be stored after each iteration (not relevant here)
        :param file_name:
        :return: REsulting image, PSF and dict of metrics measured during processing
        """
        # Start time measurement
        start = timeit.default_timer()

        # Preprocessing
        X, g = self._constraints(X, psf)
        X_padded = self._pad(X, self.pixels_padding, self.planes_padding)
        X_smoothed = self.preprocess(X_padded, sigma=sigma) if preprocess else X_padded

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
        res['Runtime'] = []

        # Blind RL iterations
        for k in range(n_iter_outer):

            # Save intermediate result
            if save_intermediate_res:
                self._save_res(f, psf, k, str(k)+str(n_iter_psf) + str(n_iter_image) +file_name, sigma)

            for i in range(n_iter_psf):  # m RL iterations, refining PSF
                psf = fftconvolve((X_smoothed / (fftconvolve(psf, f, mode='same') + epsilon)), f[::-1, ::-1, ::-1],
                                  mode='same') * psf
                # psf = self._convolution_step(X_smoothed, psf, f)
            for i in range(n_iter_image):  # m RL iterations, refining reconstruction
                f = fftconvolve((X_smoothed / (fftconvolve(f, psf, mode='same') + epsilon)), psf[::-1, ::-1, ::-1],
                                mode='same') * f
                # f = self._convolution_step(X_smoothed, f, psf)

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
            print(f'Image {file_name}, Iteration {k} completed.')
        stop = timeit.default_timer()
        res['Runtime'].append(stop - start)
        self.res_dict[file_name[:-4]] = res
        f_unpad, psf_unpad = self._save_res(f, psf, n_iter_outer, str(n_iter_outer)+str(n_iter_psf) + str(n_iter_image) +file_name, sigma)
        return f_unpad, psf_unpad, res

    def predict_img_fft(self, X, psf, n_iter_outer=10, n_iter_image=5, n_iter_psf=5, sigma=1, plot_frequency=0,
                    eval_img_steps=False, save_intermediate_res=False, file_name='', preprocess=False):
        """
        Deconvolve image
        :param X: Image (np.array)
        :param psf: PSF (np.array)
        :param n_iter_outer: RL-iterations
        :param n_iter_image: Convolution iterations on image
        :param n_iter_psf: Convolution iterations on psf
        :param sigma: Gaussian-smoothing parameter
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param plot_frequency: How often should intermediate results be plotted? If 1, after every iteration
        :param save_intermediate_res: True, if results should be stored after each iteration (not relevant here)
        :param file_name:
        :return: REsulting image, PSF and dict of metrics measured during processing
        """

        def conv3d_s(vol, otf):
            return np.real(np.fft.ifftn(np.fft.fftn(vol) * otf))

        # Start time measurement
        start = timeit.default_timer()

        # Preprocessing
        X, g = self._constraints(X, psf)
        X_padded = self._pad(X, self.pixels_padding, self.planes_padding)
        X_smoothed = self.preprocess(X_padded, sigma=sigma) if preprocess else X_padded

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
        res['Runtime'] = []
        otf=np.fft.fftn(np.fft.ifftshift(psf))

        # Blind RL iterations
        for k in range(n_iter_outer):

            # Save intermediate result
            if save_intermediate_res:
                self._save_res(f, psf, k, str(k)+str(n_iter_psf) + str(n_iter_image) +file_name, sigma)

            for i in range(n_iter_psf):  # m RL iterations, refining PSF
                psf = fftconvolve((X_smoothed / (fftconvolve(psf, f, mode='same') + epsilon)), f[::-1, ::-1, ::-1],
                                  mode='same') * psf
                # psf = self._convolution_step(X_smoothed, psf, f)
            # Back Projector
            alpha = 0.05
            beta = 1
            n = 10
            resFlag = 1
            iRes = [2.44, 2.44, 10]
            verboseFlag = 1

            bp = get_BackProjector(psf, alpha, beta, n, resFlag, iRes, verboseFlag)
            OTF_fp = np.fft.fftn(np.fft.ifftshift(psf))
            OTF_bp = np.fft.fftn(np.fft.ifftshift(bp))
            for i in range(n_iter_image):  # m RL iterations, refining reconstruction
                f = f * conv3d_s(X_smoothed / conv3d_s(f, OTF_fp), OTF_bp)
                # f = self._convolution_step(X_smoothed, f, psf)

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
            print(f'Image {file_name}, Iteration {k} completed.')
        stop = timeit.default_timer()
        res['Runtime'].append(stop - start)
        self.res_dict[file_name[:-4]] = res
        f_unpad, psf_unpad = self._save_res(f, psf, n_iter_outer, str(n_iter_outer)+str(n_iter_psf) + str(n_iter_image) +file_name, sigma)
        return f_unpad, psf_unpad, res

    def _get_psf(self, size_xy, size_z):
        """
        Load PSF-file and extract relevant z-planes
        :param size_xy: Size in X and Y- direction (after padding)
        :param size_z: Size in Z direction (after padding)
        :return: PSF
        """
        # test_flag = self.args['test_flag'] if 'test_flag' in self.args.keys() else False
        # if size_z % 2 == 1 or test_flag:
        #     z = 151
        #     print('Test')
        #     g = self._read_psf_file(size_xy, z)
        #
        #     # Initial guess for PSF
        #     offset = int((z - size_z) / 2)
        #     psf = g[offset+1:g.shape[0] - offset, :, :]
        #
        # else:
        z = 150
        g = self._read_psf_file(size_xy, z)

        # Initial guess for PSF
        offset = int((z - size_z) / 2)
        psf = g[offset:g.shape[0] - offset, :, :] if size_z % 2 == 0 else g[offset + 1:g.shape[0] - offset, :, :]
        psf /= np.sum(psf)
        # psf = psf**2
        psf /= np.sum(psf)
        # tif.imsave('psf_power4.tif', psf)

        return psf

    def _read_psf_file(self, size_xy, z):
        psf_file = 'PSF_' + str(size_xy) + '_' + str(z) + '.tif'
        g = np.float32(io.imread(os.path.join(self.psf_dir, psf_file), plugin='pil'))
        return g

    def _save_res(self, f, psf, iteration, f_name='x.tif', sigma=2):
        f_unpad = self._unpad(f, self.pixels_padding, self.planes_padding)
        psf_unpad = self._unpad(psf, self.pixels_padding, self.planes_padding)
        name_img = os.path.join(self.res_path, f_name)
        tif.imsave(name_img, f_unpad)
        name_psf = os.path.join(self.res_path, f_name[:-4] + '_psf.tif')
        tif.imsave(name_psf, psf_unpad)
        return f_unpad, psf_unpad

    def _constraints(self, f, psf):
        """
        Constraints to enable and imporve deconvolution
        :param f: Image (np.array)
        :param psf: PSF (np.array)
        :return: Adjusted image and PSF
        """

        # Avoid overflow /Non-negativity
        f[(f < 1e-100)] = 0
        psf[(psf < 1e-100)] = 0


        # psf = np.clip(psf, 0, 1)
        # f = np.clip(f, 0, 1)
        s = np.append(f, psf)
        m = np.min(s)
        p = np.ptp(s)
        f = (f - m) / p
        print(np.max(f))

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
