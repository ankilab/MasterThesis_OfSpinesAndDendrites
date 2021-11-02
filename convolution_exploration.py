# Source: https://numbersmithy.com/2d-and-3d-convolutions-using-numpy/
import numpy as np
from scipy.signal import fftconvolve, convolve
from skimage import io
import os
import timeit
from scipy.signal import fftconvolve, convolve
import tifffile as tif
from cupyx.scipy import ndimage
from cupyx.scipy import signal
import cupy
import deconv
from numba import njit

def _get_psf(size_xy, size_z):
    file_path = '.\\PSF'
    z = size_z

    psf_file = 'PSF_' + str(size_xy) + '_' + str(z) + '.tif'

    # Initial guess for PSF
    g = io.imread(os.path.join(file_path,psf_file), plugin='pil')
    g = np.float32(g)
    offset = int((z - size_z) / 2)
    return g[offset:g.shape[0] - offset, :, :]

def fft(array):
  fft = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array)))
  return fft

def ifft(array):
  ifft = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(array)))
  return ifft

def conv_3D(array, kernel):
  conv = np.abs(ifft(fft(array)*fft(kernel)))
  return conv

# @njit(parallel=True)
def custom_conv(img, ker):
    img_d, img_h, img_w = img.shape
    ker_d, ker_h, ker_w = ker.shape
    b, pad, stride, img, ker = 0, 0, 1, img, ker

    pad_img = np.pad(img, ((0, 0), (1, 1), (1, 1)), mode='constant')  # pad the input images with zeros around

    i0 = np.int8(np.repeat(np.arange(ker_h), ker_h))
    i1 = np.int8(np.repeat(np.arange(img_h), img_h))
    j0 = np.int8(np.tile(np.arange(ker_w), ker_h))
    j1 = np.int8(np.tile(np.arange(img_h), img_w))
    i0 = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j0 = j0.reshape(-1, 1) + j1.reshape(1, -1)

    select_img = pad_img[ :, i0,
                      j0].squeeze()  # receptive feild pixels are selected based on the index***[1,9,100] reshaped to [9,100]
    weights = ker.reshape(ker_h * ker_w, -1)  # weights reshaped to [9,1]
    convolve = weights.transpose() @ select_img  # convolution operation [1,9]*[9,100] ----> [1,100]
    convolve = convolve.reshape(img_d, img_h, img_w)

    return convolve

if __name__ == "__main__":
    X = io.imread(os.path.join('.\\Registered', 'Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif'))
    X = np.float32(X)
    X -=np.min(X)
    X /= np.max(X)
    # z_pad_s = X.shape[0]//2
    # y_pad_s = X.shape[2]//2
    # x_pad_s = X.shape[1] // 2
    # X_pad= np.pad(X, ((z_pad_s, z_pad_s), (x_pad_s, x_pad_s), (y_pad_s, y_pad_s)), 'constant', constant_values =((0,0),(0,0),(0,0)))
    # psf_pad= np.pad(psf, ((z_pad_s, z_pad_s), (x_pad_s, x_pad_s), (y_pad_s, y_pad_s)), 'constant', constant_values =(0,0,0))
    psf = _get_psf(X.shape[1], X.shape[0])



    # Variante 4:
    start = timeit.default_timer()
    xt = cupy.array(X)
    res = ndimage.convolve(xt, cupy.array(psf), mode='constant')
    stop = timeit.default_timer()
    print( 'Option 4 (cupyx.scipy.ndimage.convolve) took ' + str(stop-start) +' s')

    # # # Variante 5:
    # start = timeit.default_timer()
    # conv_5 = signal.fftconvolve(cupy.array(X), cupy.array(psf), mode='same')
    # stop = timeit.default_timer()
    # print( 'Option 5 (cupyx.scipy.fftconvolve) took ' + str(stop-start) +' s')


    # Variante 2:
    start = timeit.default_timer()
    conv_2 = convolve(X, psf, 'same')
    stop = timeit.default_timer()
    print( 'Option 2 (scipy.signal.convolve) took ' + str(stop-start) +' s')

    # Variante 3:
    start = timeit.default_timer()
    conv_3 = fftconvolve(X, psf, 'same')
    stop = timeit.default_timer()
    print( 'Option 3 (scipy.signal.fftconvolve) took ' + str(stop-start) +' s')

    # xt = np.isclose(conv_2, cupy.asnumpy(res)).astype(int)
    # xt_3 = np.isclose(conv_2, conv_3).astype(int)
    # print(conv_2.size - np.sum(xt))
    # print(conv_2.size - xt_3)

    # Variante 5:
    start = timeit.default_timer()
    conv_5 = custom_conv(X, psf)
    stop = timeit.default_timer()
    print( 'Option 5 (Custom conv) took ' + str(stop-start) +' s')



    # print(cupy.min(xt))
    # start = timeit.default_timer()
    # n_a = cupy.asnumpy(res)
    # stop = timeit.default_timer()
    # print( 'Tranfer from GPU took ' + str(stop-start) +' s')


    # tif.imsave('res_cupy.tif', n_a)

    ##########################################
    args = {}
    args['data_path']= ''
    args['source_folder']= './Registered/Heatmap'
    args['target_folder']= ''
    args['result_path'] = 'Blind_RL_P'

    args['psf'] = "./PSF"
    blind_rl = deconv.BlindRL(args)



