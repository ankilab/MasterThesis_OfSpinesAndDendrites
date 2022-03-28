from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
# from skimage.restoration import denoise_nl_means, estimate_sigma
# import numpy as np
# import bm3d


class Denoiser:
    """
    Super-class for all implemented denoising techniques. All derived classes implement a prepare and denoise function.
    """

    def __init__(self):
        pass

    def prepare(self, **kwargs):
        pass

    def denoise(self, **kwargs):
        return NotImplementedError


class GaussianFilter(Denoiser):
    """
    Gaussian filter.
    """

    def __init__(self):
        super().__init__()

    def denoise(self, img, sigma):
        """
        Denoise image using Gaussian filter

        :param img: Input image
        :type img: nd.array
        :param sigma: Standard deviation of Gaussian filter
        :type sigma: float
        :return: Denoised image
        :rtype: nd.array
        """
        return gaussian(img, sigma)


class BilateralFilter(Denoiser):
    """
    Bilateral filter. Based on:
    "Tomasi, Carlo, and Roberto Manduchi. "Bilateral filtering for gray and color images." Sixth international
    conference on computer vision (IEEE Cat. No. 98CH36271). IEEE, 1998."
    """

    def __init__(self):
        super().__init__()

    def denoise(self, img):
        """
        Denoise 2D-image.

        :param img: Input image
        :type img: nd.array
        :return: Denoised image
        :rtype: nd.array
        """
        # for 2d images
        return denoise_bilateral(img)


class Neighbor2Neighbor(Denoiser):
    """
    Denoiser using Neighbor2Neighbor approach. Based on:
    "Huang, Tao, et al. "Neighbor2neighbor: Self-supervised denoising from single noisy images." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021."
    """

    def __init__(self):
        from Neighbor2Neighbor import neighbor2neighbor as n2n

        super().__init__()
        self.model_path= None
        self.n2n = n2n.Neighbor2Neighbor()

    def prepare(self, train_data, val_data, res_path, lr =3e-4, n_epochs=100, batch_size = 4):
        """
        Train denoising model.

        :param train_data: Training data
        :param val_data: Validation data
        :param res_path: Path to store models and interim results
        :type res_path: string
        :param lr: Leanring rate, defaults to 3e-4
        :type lr: float, optional
        :param n_epochs: Nuber of epochs to train, defaults to 100
        :type n_epochs: int, optional
        :param batch_size: Training batch size
        :type batch_size: int, optional
        :return: Model path
        :rtype: string
        """

        self.model_path=self.n2n.train(train_data, val_data, res_path, lr, n_epochs, batch_size)
        return self.model_path

    def denoise(self, img, model_path=''):
        """
        Denoise image.

        :param img: Input image
        :type img: nd.array
        :param model_path: Denoising model path. If nothing specified, defaults to object object-intern path
        :type model_path: string, optional
        :return:
        """
        if model_path != '':
            self.model_path = model_path
        res=self.n2n.predict_image(self.model_path, img)
        return res


# class NLMeans(Denoiser):
#     #not tested
#     def __init__(self):
#         super().__init__()
#
#     def denoise(self, img):
#         patch_kw = dict(patch_size=50,  # 5x5 patches
#                         patch_distance=60)  # 13x13 search area
#         sigma_est = np.mean(estimate_sigma(img))
#         res = denoise_nl_means(img, h=0.8 * sigma_est, fast_mode=True,
#                                         **patch_kw)
#         return res
#
#
# class BM3D(Denoiser):
#     # not tested
#     def __init__(self):
#         super().__init__()
#
#     def denoise(self, img):
#         sigma_est = np.mean(estimate_sigma(img))
#         print(sigma_est)
#         res = bm3d.bm3d(img, 1) #, stage_arg=bm3d.BM3DStages.ALL_STAGES)
#         return res










