from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
# import bm3d


class Denoiser:
    def __init__(self):
        pass

    def prepare(self, **kwargs):
        pass

    def denoise(self, **kwargs):
        return NotImplementedError


class GaussianFilter(Denoiser):
    def __init__(self):
        super().__init__()

    def denoise(self, img, sigma):
        return gaussian(img, sigma)


class BilateralFilter(Denoiser):
    def __init__(self):
        super().__init__()

    def denoise(self, img):
        # for 2d images
        return denoise_bilateral(img)


class Neighbor2Neighbor(Denoiser):

    def __init__(self):
        from Neighbor2Neighbor import neighbor2neighbor as n2n

        super().__init__()
        self.model_path= None
        self.n2n = n2n.Neighbor2Neighbor()

    def prepare(self, train_data, val_data, res_path, lr =3e-4, n_epochs=100, batch_size = 4):
        # train_data = 'D:/jo77pihe/Registered/Raw_32/2D/subset'
        # val_data = 'D:/jo77pihe/Registered/Raw_32/2D/validation'
        # res_path = 'D:/jo77pihe/N2N_result'
        self.model_path=self.n2n.train(train_data, val_data, res_path, lr, n_epochs, batch_size)

    def denoise(self, img, model_path=''):
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










