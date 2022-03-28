# Based on: Guo, Min, et al. "Rapid image deconvolution and multiview fusion for optical microscopy."
# Nature biotechnology 38.11 (2020): 1337-1346
import numpy as np
import tifffile
import os
from denoising import GaussianFilter
from deconv.deconvolver import Deconvolver
import deconv.utils as du


class WBBackProjectorDeconv(Deconvolver):
    """
    Non-blind Richardson-Lucy with Wiener-Butterworth backprojector
    Based on: Guo, Min, et al. "Rapid image deconvolution and multiview fusion for optical microscopy."
    Nature biotechnology 38.11 (2020): 1337-1346
    """

    def __init__(self, args):
        super().__init__(args)
        self.psf_dir =args.get("psf", './PSF')
        self.counter =0

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

    def train(self, **kwargs):
        """
        RL training: None required.
        """
        pass

    def predict(self, data_dir, n_iter=1, sigma=1):
        """
        Deconvolve all tif-images within folder specified.

        :param data_dir: Directory with tif-files
        :type data_dir: string
        :param n_iter: Number of iterations, defaults to 1
        :type n_iter: int, optional
        :param sigma: Gaussian-smoothing parameter, defaults to 1
        :type sigma: float, optional
        """
        self.data_path = data_dir
        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        for f in files:
            self.predict_img(tifffile.imread(os.path.join(data_dir, f)), n_iter=n_iter,
                             name=os.path.join(self.res_path, f))

    def predict_img(self, img, n_iter=1, name= None, sigma=1.5):
        """
        Deconvolve image and save to file if wanted.

        :param img: Input image
        :type img: nd.array
        :param n_iter: Number of iterations, defaults to 1
        :type n_iter: int, optional
        :param name: Name to save deconvolved file as, if name is NoneType it is not saved. Defaults to None
        :type name: string, optional
        :param sigma: Gaussian-smoothing parameter, defaults to 1
        :type sigma: float, optional
        :return: Deconvolved image
        :rtype: nd.array
        """
        fp, bp = self._get_fb_bp(img.shape[0], img.shape[0], self.psf_dir)
        OTF_fp = np.fft.fftn(np.fft.ifftshift(fp))
        OTF_bp = np.fft.fftn(np.fft.ifftshift(bp))
        img = self.preprocess(img, sigma)

        estimate = img.copy()
        for i in range(n_iter):
            estimate = estimate * self.conv3d_s(img / self.conv3d_s(estimate, OTF_fp), OTF_bp)

        if name is None:
            name ='wb_deconv_' + str(self.counter) +'.tif'
            self.counter +=1
        tifffile.imwrite(name, estimate)
        return estimate

    def _get_fb_bp(self, size_xy, size_z, psf_dir):
        """
        Load PSF-file and extract relevant z-planes

        :param size_xy: Size in X and Y- direction (after padding)
        :param size_z: Size in Z direction (after padding)
        :return: PSF
        """

        z = size_z
        g = du.read_psf_file(size_xy, z, psf_dir)

        # Initial guess for PSF
        offset = int((z - size_z) / 2)

        # Forward Projector
        psf1 = g[offset:g.shape[0] - offset, :, :] if size_z % 2 == 0 else g[offset + 1:g.shape[0] - offset, :, :]
        psf1 /= np.sum(psf1)
        # psf1 = psf1**4
        # psf1 /= np.sum(psf1)

        # Back Projector
        alpha = 0.05
        n = 10
        psf2 = self.get_BackProjector(psf1, alpha, n)

        return psf1, psf2

    def get_BackProjector(self, psf1, alpha, n):
        """
        Compute back-projector from PSF (forward-projector)

        :param psf1: PSF
        :type psf: nd.array
        :param alpha:
        :type alpha: float
        :param n:
        :type n: int
        :return: Back-projector
        """
        (Sx, Sy, Sz) = psf1.shape
        Scx = (Sx + 1) / 2
        Scy = (Sy + 1) / 2
        Scz = (Sz + 1) / 2

        Soz = int(np.round((Sz + 1) / 2, decimals=0))

        FWHMx, FWHMy, FWHMz = self.fwhm_PSF(psf1)
        flipped = np.flip(psf1, axis=(0,1,2))
        OTF_flip=np.fft.fftn(np.fft.ifftshift(flipped))
        OTF_abs = np.fft.fftshift(np.abs(OTF_flip))
        M = np.max(OTF_abs)
        OTF_abs_norm = OTF_abs / M
        resx = FWHMx
        resy = FWHMy
        resz = FWHMz

        px = 1 / Sx
        py = 1 / Sy
        pz = 1 / Sz

        tx = 1 / resx / px
        ty = 1 / resy / py
        tz = 1 / resz / pz

        tplane = np.squeeze(np.max(OTF_abs_norm, axis = 2))
        tline=np.max(tplane, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx, decimals=0), 1))
        to2 = int(np.minimum(np.round(Scx + tx, decimals=0), Sx-1))
        beta_fpx = (tline[to1] + tline[to2]) / 2

        tplane = np.squeeze(np.max(OTF_abs_norm, axis = 2))
        tline=np.max(tplane, axis=0)
        to1 = int(np.maximum(np.round(Scy - ty, decimals=0), 1))
        to2 = int(np.minimum(np.round(Scy + ty, decimals=0), Sy-1))
        beta_fpy = (tline[to1] + tline[to2]) / 2

        tplane = np.squeeze(np.max(OTF_abs_norm, axis = 0))
        tline=np.max(tplane, axis=0)
        to1 = int(np.maximum(np.round(Scz - tz, decimals=0), 1))
        to2 = int(np.minimum(np.round(Scz + tz, decimals=0), Sz-1))
        beta_fpz = (tline[to1] + tline[to2]) / 2

        beta_fp = (beta_fpx + beta_fpy + beta_fpz) / 3
        beta = beta_fp

        OTF_flip_norm = OTF_flip / M
        OTF_Wiener = OTF_flip_norm/ (np.abs(OTF_flip_norm)**2 + alpha)

        OTF_Wiener_abs = np.fft.fftshift(np.abs(OTF_Wiener))
        tplane = np.abs(np.squeeze(OTF_Wiener_abs[:,:, Soz]))
        tline = np.max(tplane,axis=1)
        to1 = int(np.maximum(np.round(Scx - tx, decimals=0), 1))
        to2 = int(np.minimum(np.round(Scx + tx, decimals=0), Sx-1))
        beta_wienerx = (tline[to1] + tline[to2]) / 2


        kcx = tx
        kcy = ty
        kcz = tz
        ee = beta_wienerx / beta**2 - 1
        mask = np.zeros((Sx, Sy, Sz))
        for i in range(Sx):
            for j in range(Sy):
                for k in range(Sz):

                    w = ((i - Scx) / kcx)**2 + ((j - Scy) / kcy)**2 + ((k - Scz) / kcz)**2
                    mask[i, j, k] = 1 / np.sqrt(1 + ee * w**n)

        mask = np.fft.ifftshift(mask)
        OTF_bp = mask* OTF_Wiener

        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

        return PSF_bp

    def fwhm_PSF(self, PSF, pixelSize=1, cFlag=False):
        """
        Feed back the full width at half maximun of the input PSF

        :param PSF: PSF
        :type PSF: nd.array
        :param pixelSize:
        :param cFlag: False: use maximum's position as PSF center position,
                    True: use matrix's center position as PSF center position, defaults to False
        :type cFlag: bool, optional
        :return: FWHM in x-, y- and z-direction
        """

        (Sx,Sy,Sz) = PSF.shape

        if(cFlag):
            indx = np.floor((Sx+1)/2)
            indy = np.floor((Sy+1)/2)
            indz = np.floor((Sz+1)/2)
        else:
            (indx,indy,indz) = np.unravel_index(np.argmax(PSF, axis=None), PSF.shape) # find maximum value and position
        x = np.arange(0,Sx)
        x = np.transpose(x)
        y = PSF[:,indy,indz]
        FWHMx = self.fwhm(x, y)

        x = np.arange(0,Sy)
        x = np.transpose(x)
        y = PSF[indx,:,indz]
        FWHMy = self.fwhm(x, y)

        x = np.arange(0,Sz)
        x = np.transpose(x)
        y = PSF[indx,indy,:]
        # y = y(:);
        FWHMz = self.fwhm(x, y)

        FWHMx = FWHMx*pixelSize
        FWHMy = FWHMy*pixelSize
        FWHMz = FWHMz*pixelSize

        return FWHMx,FWHMy,FWHMz


    def fwhm(self,x,y):
        """
        Full-Width at Half-Maximum (FWHM) of the waveform y(x) and its polarity.

        :param x:
        :param y:
        :return:
        """
        y = y / y.max()
        N = len(y)
        lev50 = 0.01
        if y[0] < lev50:                  # find index of center (max or min) of pulse
            centerindex=np.argmax(y)
        else:
            centerindex=np.argmin(y)
        i = 1
        while np.sign(y[i] - lev50) == np.sign(y[i - 1] - lev50):
            i = i+1

        interp = (lev50-y[i-1]) / (y[i]-y[i-1])
        tlead = x[i-1] + interp*(x[i]-x[i-1])
        i = centerindex+1                      #start search for next crossing at center
        while (np.sign(y[i] - lev50) == np.sign(y[i - 1] - lev50)) & (i <= N - 1):
            i = i+1

        if i != N:
            interp = (lev50-y[i-1]) / (y[i]-y[i-1])
            ttrail = x[i-1] + interp*(x[i]-x[i-1])
            width = ttrail - tlead
        else:
            width = np.nan
        return width

    def conv3d_s(self, vol, otf):
        """

        :param vol:
        :param otf: OTF (PSF in frequency domain)
        :return:
        """
        return np.real(np.fft.ifftn(np.fft.fftn(vol)*otf))


if __name__ == "__main__":
    args={}
    args['data_path'] = 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Registered/Raw/'
    args['res_path'] = '../WB_Backprojector'
    args['psf']="./PSF"
    deconvolver = WBBackProjectorDeconv(args)




