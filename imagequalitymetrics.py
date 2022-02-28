from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
# import imquality.brisque as brisque
# Doku: https://sewar.readthedocs.io/en/latest/
# import sewar
import numpy as np
# import matlab.engine


class ImageQualityMetrics:
    def __init__(self):
        pass

    def compute_all(self, img, gt_img=None):
        res = {}
        if gt_img is not None:
            res['mse'] = self.mse(img,gt_img)
            res['ssim'] = self.ssim(img,gt_img)
            res['msssim'] = self.ssim(img, gt_img)
            res['psnr'] = self.psnr(img, gt_img)
            # res['vif'] = self.vifp(img, gt_img)
            # res['uqi'] = self.uqi(img, gt_img)

        res['niqe'] = self.niqe(img)
        res['brisque'] = self.brisque(img)
        res['snr'] = self.snr(img)
        return res

    def mse(self, img, gt_img):
        # img = self._rescale(img)
        # gt_img = self._rescale(gt_img)
        val = np.sum((gt_img.astype("float") - img.astype("float")) ** 2)
        val /= float(gt_img.shape[0] * gt_img.shape[1])
        return val

    def ssim(self, img, gt_img, win_size=None):
        # img = self._rescale(img)
        # gt_img = self._rescale(gt_img)
        val = sk_ssim(img, gt_img, win_size=win_size)
        return val

    def mssim(self, img, gt_img):
        # img = self._rescale(img)
        # gt_img = self._rescale(gt_img)
        return 0#sewar.msssim(gt_img, img)

    def niqe(self, img):
        # eng = matlab.engine.start_matlab()
        # eng.niqe(matlab.double(img.tolist()))
        # eng.quit()
        import skvideo
	    #from skvideo.measure import niqe as sk_niqe
        return np.min(skvideo.measure.niqe(img))

    def brisque(self, img):
        return 0 #brisque.score(img)

    def psnr(self, img, gt_img, data_range = 15197):
        # img = self._rescale(img)
        # gt_img = self._rescale(gt_img)
        # print('x')
        # if img.min() <0:
        #     img -= MIN_VAL
        # if gt_img.min()<0:
        #     gt_img -=MIN_VAL
        return sk_psnr(gt_img, img,data_range =data_range)

    def uqi(self, img, gt_img):

        return 0 #sewar.uqi(gt_img, img)

    def vifp(self, img, gt_img):
        # Pixel Based Visual Information Fidelity

        return 0 #sewar.vifp(gt_img, img)

    def snr(self, img):
        m = img.mean()
        sd = img.std()
        return  m / sd if sd != 0 else np.nan