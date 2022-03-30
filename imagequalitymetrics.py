from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
# import imquality.brisque as brisque
# Doku: https://sewar.readthedocs.io/en/latest/
# import sewar
import numpy as np


class ImageQualityMetrics:
    """
    Class providing functionality for image quality metrics.
    """

    def __init__(self):
        pass

    def compute_all(self, img, gt_img=None):
        """
        Compute all metrics implemented.

        :param img: Moving/distorted image
        :type img: nd.array
        :param gt_img: Ground truth image
        :type gt_img: nd.array
        :return: Quality metrics values
        :rtype: dict[float]
        """

        res = {}
        if gt_img is not None:
            res['mse'] = self.mse(img,gt_img)
            res['ssim'] = self.ssim(img,gt_img)
            # res['msssim'] = self.ssim(img, gt_img)
            res['psnr'] = self.psnr(img, gt_img)
            # res['vif'] = self.vifp(img, gt_img)
            # res['uqi'] = self.uqi(img, gt_img)

        res['niqe'] = self.niqe(img)
        #res['brisque'] = self.brisque(img)
        res['snr'] = self.snr(img)
        return res

    def mse(self, img, gt_img):
        """
        Mean squared error.

        :param img: Moving/distorted image
        :type img: nd.array
        :param gt_img: Ground truth image
        :type gt_img: nd.array
        :return: Mean squared error
        :rtype: float
        """

        val = np.sum((gt_img.astype("float") - img.astype("float")) ** 2)
        val /= float(gt_img.shape[0] * gt_img.shape[1])
        return val

    def ssim(self, img, gt_img, win_size=None):
        """
        Structured similarity.

        :param img: Moving/distorted image
        :type img: nd.array
        :param gt_img: Ground truth image
        :type gt_img: nd.array
        :return: Structured similarity
        :rtype: float
        """
        val = sk_ssim(img, gt_img, win_size=win_size)
        return val

    # def mssim(self, img, gt_img):
    #     return 0#sewar.msssim(gt_img, img)

    def niqe(self, img):
        """
        NIQE score.

        :param img: Moving/distorted image
        :type img: nd.array
        :param gt_img: Ground truth image
        :type gt_img: nd.array
        :return: NIQE score
        :rtype: float
        """
        # Requires certain version of scipy
        import skvideo
        return np.min(skvideo.measure.niqe(img))

    # def brisque(self, img):
    #     return 0 #brisque.score(img)

    def psnr(self, img, gt_img, data_range = 15197):
        """
        Peak signal-to-noise ratio (PSNR).

        :param img: Moving/distorted image
        :type img: nd.array
        :param gt_img: Ground truth image
        :type gt_img: nd.array
        :return: PSNR
        :rtype: float
        """

        return sk_psnr(gt_img, img,data_range =data_range)

    # def uqi(self, img, gt_img):
    #     return 0 #sewar.uqi(gt_img, img)

    # def vifp(self, img, gt_img):
    #     # Pixel Based Visual Information Fidelity
    #     return 0 #sewar.vifp(gt_img, img)

    def snr(self, img):
        """
        Signal-to-noise ratio (SNR).

        :param img: Moving/distorted image
        :type img: nd.array
        :param gt_img: Ground truth image
        :type gt_img: nd.array
        :return: SNR
        :rtype: float
        """

        m = img.mean()
        sd = img.std()
        return  m / sd if sd != 0 else np.nan