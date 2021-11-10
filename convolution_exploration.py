import numpy as np
from skimage import io
import os
import timeit
from scipy.signal import fftconvolve, convolve
from cupyx.scipy import ndimage
import cupy
import torch
import torch.nn.functional as F
import time
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def _get_psf(size_xy, size_z):
    """
    Load PSF-file and extract relevant z-planes
    :param size_xy: Size in X and Y- direction (after padding)
    :param size_z: Size in Z direction (after padding)
    :return: PSF
    """
    z = 150 if size_z % 2 == 0 else 151

    psf_file = 'PSF_' + str(size_xy) + '_' + str(z) + '.tif'

    # Initial guess for PSF
    g = io.imread(os.path.join('.\\PSF', psf_file), plugin='pil')
    offset = int((z - size_z) / 2)
    return g[offset:g.shape[0] - offset, :, :]

def _pad(img, pixels=20, planes=10):
    """
    Pad image
    :param img: Original image (np.array)
    :param pixels: amount to pad in x and y direction
    :param planes: amount to pad in z direction
    :return: padded image
    """
    return np.pad(img, ((planes, planes), (pixels, pixels), (pixels, pixels)), 'reflect')


if __name__ == "__main__":
    files = [f for f in os.listdir('.\\Registered') if f.endswith('.tif')]

    for i in range(3):
        X = io.imread(os.path.join('.\\Registered', files[i]))
        X = np.float32(X)
        # X -=np.min(X)
        # X /= np.max(X)
        X = _pad(X)

        psf = np.float32(_get_psf(X.shape[1], X.shape[0]))

        # Variante 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.cuda.synchronize()
        start = time.time()
        Xc = torch.tensor(X[None, None, ...], device=device, requires_grad=False)
        psf_t = torch.tensor(psf[None, None, ...], device=device, requires_grad=False)

        torch.cuda.synchronize()
        stop_1 =time.time()
        print( 'Option 1 preprocessing took ' + str(stop_1-start) +' s')
        torch.cuda.synchronize()
        start_1 = time.time()
        res_torch = F.conv3d(Xc, psf_t, padding='same')
        torch.cuda.synchronize()
        stop = time.time()
        print( 'Option 1 (torch F.conv3d) took ' + str(stop-start_1) +' s')

        torch.cuda.synchronize()
        start_2 = time.time()
        res_torch = res_torch[0,0,:,:,:]
        (z, x, y) = res_torch.shape
        res_torch= res_torch[10:z - 10, 20:x - 20, 20:y - 20]
        num_torch = res_torch.to('cpu').detach.numpy()
        torch.cuda.synchronize()
        stop_2 = time.time()
        print( 'GPU to CPU (torch) took ' + str(stop_2-start_2) +' s')
        print( 'Torch total ' + str(stop_2-start) +' s')

        # Variante 4:
        start = timeit.default_timer()
        xt = cupy.array(X)
        # xt =cupy.pad(xt, ((10, 10), (20, 20), (20, 20)), 'reflect')
        stop_1 =timeit.default_timer()
        print( 'Option 4 preprocessing took ' + str(stop_1-start) +' s')
        start_1 = timeit.default_timer()
        res = ndimage.convolve(xt, cupy.array(psf), mode='constant')
        stop = timeit.default_timer()
        print( 'Option 4 (cupyx.scipy.ndimage.convolve) took ' + str(stop-start_1) +' s')

        start_2 = timeit.default_timer()
        (z, x, y) = res.shape
        res= res[10:z - 10, 20:x - 20, 20:y - 20]
        n_a = cupy.asnumpy(res)
        stop_2 = timeit.default_timer()
        print( 'GPU to CPU took ' + str(stop_2-start_2) +' s')
        print( 'Cupy total ' + str(stop_2-start) +' s')

        # Variante 2:
        start = timeit.default_timer()
        conv_2 = convolve(X, psf, 'same')
        stop = timeit.default_timer()
        print('Option 2 (scipy.signal.convolve) took ' + str(stop - start) + ' s')

        # Variante 3:
        start = timeit.default_timer()
        conv_3 = fftconvolve(X, psf, 'same')
        stop = timeit.default_timer()
        print('Option 3 (scipy.signal.fftconvolve) took ' + str(stop - start) + ' s')

    # # # Variante 5:
    # start = timeit.default_timer()
    # conv_5 = signal.fftconvolve(cupy.array(X), cupy.array(psf), mode='same')
    # stop = timeit.default_timer()
    # print( 'Option 5 (cupyx.scipy.fftconvolve) took ' + str(stop-start) +' s')




    # xt = np.isclose(conv_2, cupy.asnumpy(res)).astype(int)
    # xt_3 = np.isclose(conv_2, conv_3).astype(int)
    # print(conv_2.size - np.sum(xt))
    # print(conv_2.size - xt_3)

    # # Variante 5:
    # start = timeit.default_timer()
    # conv_5 = custom_conv(X, psf)
    # stop = timeit.default_timer()
    # print( 'Option 5 (Custom conv) took ' + str(stop-start) +' s')



    # print(cupy.min(xt))
    # start = timeit.default_timer()
    # n_a = cupy.asnumpy(res)
    # stop = timeit.default_timer()
    # print( 'Tranfer from GPU took ' + str(stop-start) +' s')


    # tif.imsave('res_cupy.tif', n_a)

    # ##########################################
    # args = {}
    # args['data_path']= ''
    # args['source_folder']= './Registered/Heatmap'
    # args['target_folder']= ''
    # args['result_path'] = 'Blind_RL_P'
    #
    # args['psf'] = "./PSF"
    # blind_rl = deconv.BlindRL(args)



